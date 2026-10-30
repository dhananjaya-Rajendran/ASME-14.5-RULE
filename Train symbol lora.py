"""
LoRA fine-tuning of Qwen2.5-VL-3B-Instruct on technical symbol crops (GD&T, CAD conventions).
Path 1 from the training summary: local CPU, i7-8665U, 16 GB RAM.

DATASET FORMAT (JSONL, one example per line):
  {"image": "crops/frame_001.png",
   "instruction": "Read this feature control frame and return JSON.",
   "answer": "{\"symbol\": \"position\", \"tolerance\": \"0.25\", \"modifier\": \"M\", \"datums\": [\"A\",\"B\"]}"}
Image paths are resolved relative to the JSONL file's folder.

USAGE:
  python train_symbol_lora.py --data data/train.jsonl --epochs 3
  python train_symbol_lora.py --data data/train.jsonl --resume outputs/lora_symbols/checkpoint-200

OUTPUT:
  outputs/lora_symbols/checkpoint-N/   (adapter only, a few tens of MB)
  outputs/lora_symbols/final/          (adapter + processor, ready for merge / OpenVINO export)
"""
import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Thread pinning must happen before torch import
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SCRIPT_DIR = Path(__file__).resolve().parent

# Regex (PEFT uses fullmatch):
#  - all attention + MLP linears in the language model (anything NOT under "visual")
#  - the vision->LLM merger MLP (visual.merger.mlp.0 / .2), which stays trainable
#    while the ViT blocks are frozen and carry no LoRA.
TARGET_REGEX = (
    r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    r"|.*visual\.merger\.mlp\.(0|2)$"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="Path to train JSONL (default: data/symbols_train.jsonl)")
    p.add_argument("--init-data", action="store_true",
                   help="Create a starter JSONL manifest and example instructions, then exit")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--out", default="outputs/lora_symbols")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad-accum", type=int, default=8, help="Effective batch size (batch is 1 for RAM)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    p.add_argument("--max-pixels", type=int, default=384 * 28 * 28)
    p.add_argument("--max-len", type=int, default=1024, help="Skip examples longer than this many tokens")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                   help="bf16 = low RAM (~12 GB). fp32 needs ~16+ GB and will OOM on this machine.")
    p.add_argument("--save-every", type=int, default=50, help="Save adapter every N optimizer steps")
    p.add_argument("--eval-every", type=int, default=50, help="Validate every N optimizer steps")
    p.add_argument("--resume", default=None, help="Path to an adapter checkpoint to continue from")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.data is None:
        args.data = str(SCRIPT_DIR / "data" / "symbols_train.jsonl")
    return args


# --------------------------------------------------------------------------- data
def load_jsonl(path):
    base = os.path.dirname(os.path.abspath(path))
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not isinstance(r, dict):
                raise ValueError(f"Line {ln}: each JSONL record must be a JSON object")
            for k in ("image", "instruction", "answer"):
                if k not in r:
                    raise ValueError(f"Line {ln}: missing key '{k}'")
            if not os.path.isabs(r["image"]):
                r["image"] = os.path.join(base, r["image"])
            if not os.path.exists(r["image"]):
                raise FileNotFoundError(f"Line {ln}: image not found: {r['image']}")
            if not str(r["instruction"]).strip() or not str(r["answer"]).strip():
                raise ValueError(f"Line {ln}: instruction and answer must not be empty")
            if not isinstance(r["answer"], str):
                r["answer"] = json.dumps(r["answer"], ensure_ascii=False)
            rows.append(r)
    return rows


def write_dataset_template(path):
    """Write an editable dataset example; never assign labels to crops automatically."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {target}")
    example = {
        "image": "crops/replace_with_labeled_crop.png",
        "instruction": "Read the visible feature control frame. Return only JSON. Use unknown for unreadable fields.",
        "answer": json.dumps({
            "characteristic_symbol": "position", "tolerance_value": 0.25,
            "diameter_zone": True, "material_condition": "MMC",
            "datums": ["A", "B", "C"],
        }),
    }
    target.write_text(json.dumps(example, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created dataset template: {target}")
    print("Replace the image path and answer with a human-verified label for each crop.")
    print("Keep image paths relative to this JSONL file, or use absolute paths.")


def build_example(processor, row, max_len):
    """Tokenise one example; mask everything except the assistant answer from the loss."""
    image = Image.open(row["image"]).convert("RGB")
    user = {"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": row["instruction"]}]}
    asst = {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]}

    prompt_text = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template([user, asst], tokenize=False)

    prompt_ids = processor(text=[prompt_text], images=[image], return_tensors="pt")["input_ids"]
    inputs = processor(text=[full_text], images=[image], return_tensors="pt")

    if inputs["input_ids"].shape[1] > max_len:
        return None

    labels = inputs["input_ids"].clone()
    labels[:, : prompt_ids.shape[1]] = -100                       # mask system/user/image tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs


# --------------------------------------------------------------------------- model
def load_model(args):
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="cpu")
    model.config.use_cache = False

    # Freeze everything; LoRA will re-enable only its own adapters
    for p in model.parameters():
        p.requires_grad = False

    if args.resume:
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
    else:
        cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                         bias="none", task_type="CAUSAL_LM", target_modules=TARGET_REGEX)
        model = get_peft_model(model, cfg)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Keep adapter weights in fp32 for stable optimiser updates (they are tiny)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
    if trainable == 0:
        raise RuntimeError("No trainable params: LoRA target regex matched nothing. "
                           "Print model.named_modules() and adjust TARGET_REGEX.")
    return model


def evaluate(model, processor, rows, max_len):
    with torch.no_grad():
        model.eval()
        losses = []
        for r in rows:
            ex = build_example(processor, r, max_len)
            if ex is None:
                continue
            losses.append(model(**ex).loss.item())
        model.train()
    return sum(losses) / max(len(losses), 1)


def save_adapter(model, processor, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    print(f"  saved adapter -> {path}")


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    if args.init_data:
        write_dataset_template(args.data)
        return
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        template = data_path.parent / "symbols_train.example.jsonl"
        raise SystemExit(
            f"Training data not found: {data_path}\n"
            "The training script needs human-labeled image crops; existing output crops "
            "do not contain trustworthy symbol labels.\n"
            f"Create a starter manifest with:\n  python \"{Path(__file__).name}\" "
            f"--init-data --data \"{template}\"\n"
            "Edit the manifest, then train with --data <your-jsonl>."
        )
    rows = load_jsonl(str(data_path))
    if len(rows) < 2:
        raise SystemExit(f"Need at least 2 labeled examples to train; found {len(rows)} in {data_path}.")
    try:
        import torch
        from PIL import Image
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "LoRA training dependencies are missing from this Python environment. "
            "Install compatible torch, transformers, and peft packages in the same "
            "environment, then rerun this script."
        ) from exc
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    random.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_split)) if len(rows) > 10 else 0
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"Examples: {len(train_rows)} train / {len(val_rows)} val")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model = load_model(args)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, foreach=False)

    steps_per_epoch = math.ceil(len(train_rows) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_at(step):  # linear warmup, cosine decay
        if step < warmup:
            return args.lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    print(f"Optimizer steps: {total_steps} (accum={args.grad_accum}, warmup={warmup})")

    opt_step, best_val = 0, float("inf")
    t0 = time.time()
    try:
        for epoch in range(args.epochs):
            random.shuffle(train_rows)
            running, micro = 0.0, 0
            for i, row in enumerate(train_rows):
                ex = build_example(processor, row, args.max_len)
                if ex is None:
                    print(f"  skipped (too long): {row['image']}")
                    continue

                loss = model(**ex).loss
                (loss / args.grad_accum).backward()
                running += loss.item()
                micro += 1
                del ex, loss

                last = i == len(train_rows) - 1
                if micro == args.grad_accum or (last and micro > 0):
                    for g in optimizer.param_groups:
                        g["lr"] = lr_at(opt_step)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1
                    gc.collect()

                    elapsed = time.time() - t0
                    eta = elapsed / opt_step * (total_steps - opt_step)
                    print(f"epoch {epoch + 1}/{args.epochs} step {opt_step}/{total_steps} "
                          f"loss {running / micro:.4f} lr {lr_at(opt_step):.2e} "
                          f"elapsed {elapsed / 60:.1f}m eta {eta / 60:.1f}m")
                    running, micro = 0.0, 0

                    if val_rows and opt_step % args.eval_every == 0:
                        v = evaluate(model, processor, val_rows, args.max_len)
                        print(f"  val loss {v:.4f}")
                        if v < best_val:
                            best_val = v
                            save_adapter(model, processor, os.path.join(args.out, "best"))
                    if opt_step % args.save_every == 0:
                        save_adapter(model, processor, os.path.join(args.out, f"checkpoint-{opt_step}"))
    except KeyboardInterrupt:
        print("\nInterrupted: saving current adapter.")

    save_adapter(model, processor, os.path.join(args.out, "final"))
    print("Done. Merge with: PeftModel.from_pretrained(base, 'final').merge_and_unload()")


if __name__ == "__main__":
    main()
