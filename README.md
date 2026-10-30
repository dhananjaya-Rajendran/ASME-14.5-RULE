# ASME Y14.5 GD&T Validator

A local Python desktop and command-line tool for extracting feature control frames (FCFs) from engineering drawing PDFs and checking supported GD&T rules. The current pipeline uses vector-based PDF geometry detection and a Qwen vision-language model through Ollama. **YOLO is not used or required.**

The validator performs a limited automated review. It does not establish full drawing compliance; review the drawing and applicable standard with a qualified reviewer.

## Requirements

- Windows, Python 3.10 or newer, and a working Tkinter installation for the desktop app.
- Ollama running locally, with a vision-capable model such as `qwen2.5vl:7b` installed.
- Python dependencies from `gdt_validator/requirements.txt`.
- An authorized copy of the applicable ASME standard if you want to build the local standards knowledge base. The current default knowledge-base edition is ASME Y14.5M-1994.

## Setup (PowerShell)

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r gdt_validator\requirements.txt
```

If PowerShell blocks activation, call the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r gdt_validator\requirements.txt
```

Install and start Ollama separately, then pull the model:

```powershell
ollama pull qwen2.5vl:7b
```

Check that the Python client can reach the Ollama service and see the model:

```powershell
python -m gdt_validator.main --check-ollama
```

Use the **same Python interpreter** to install dependencies and run the app. For example, with the repository's existing environment:

```powershell
.\gdt_env\Scripts\python.exe -m pip install -r gdt_validator\requirements.txt
```

## Start the desktop app

From the repository root:

```powershell
python -m gdt_validator.main
```

Choose a drawing PDF and output folder, select the PDF page (zero-based), and run validation. The app saves crops and generated files below the output folder, in a subfolder named after the PDF.

## Validate from the command line

```powershell
python -m gdt_validator.main --pdf "C:\drawings\part.pdf" --page 0 --output-dir output
```

Choose another installed Ollama model with `--model`:

```powershell
python -m gdt_validator.main --pdf "C:\drawings\part.pdf" --model qwen2.5vl:7b
```

The automatic path detects FCF regions and asks the VLM to read them. If required values are missing or uncertain, the run withholds the report and writes `manual_review.json` with source crop paths and extracted values. Check those fields against the drawing before proceeding.

### Manually reviewed FCF data

You can create a crop-linked blank JSON file for manual transcription:

```powershell
python -m gdt_validator.main --pdf "C:\drawings\part.pdf" --page 0 --create-review-template "reviewed_fcf.json"
```

Fill in the template's `characteristic_symbol`, `tolerance_value`, `material_condition`, datums, and declared datums after checking the source drawing. `material_condition` accepts `RFS`, `MMC`, or `LMC`. Then generate the report from the reviewed JSON:

```powershell
python -m gdt_validator.main --pdf "C:\drawings\part.pdf" --page 0 --reviewed-fcf-json "reviewed_fcf.json"
```

The JSON must contain a `feature_control_frames` array and a `declared_datums` array. Each frame needs an `id`, `bbox`, characteristic, tolerance, and material condition. Datum references are objects with a `datum_letter` field; a datum material condition defaults to RFS. Example:

```json
{
  "drawing_id": "part",
  "declared_datums": ["A", "B", "C"],
  "feature_control_frames": [
    {
      "id": "fcf_001",
      "bbox": [100, 200, 300, 250],
      "characteristic_symbol": "position",
      "tolerance_value": 0.25,
      "diameter_zone": true,
      "material_condition": "MMC",
      "datums": [
        {"datum_letter": "A"},
        {"datum_letter": "B"},
        {"datum_letter": "C"}
      ]
    }
  ]
}
```

## Generated files

For a drawing named `part.pdf`, the default output is under `output/part/`:

- `fcf_crops/` and `datum_crops/`: extracted regions for review.
- `manual_review.json`: unresolved automatic extraction fields, when needed.
- `annotated.pdf`: source drawing annotated with findings.
- `report.pdf`: validation summary and rule findings.
- `feedback.sqlite3`: local report feedback data.

## Standards knowledge base

Build or refresh the local OCR archive and vector index from an authorized standard PDF:

```powershell
python -m gdt_validator.main --build-kb "C:\standards\ASME_Y14_5.pdf"
```

The command reports OCR failures and index counts. The validator can also index an existing OCR archive with `--index-kb`. Use `--kb-path` and `--chroma-path` to choose storage locations.

## Manual text annotation and NER (optional)

The separate text NER tools label GD&T entities in pasted OCR/plain text. They are a standalone data-preparation workflow and are **not currently connected to the PDF validation pipeline**. Install the project requirements (including spaCy), then run:

```powershell
python -m gdt_validator.tools.annotate_symbols --out gdt_validator/data/gdt_ner.jsonl
python -m gdt_validator.tools.train_gdt_ner train --data gdt_validator/data/gdt_ner.jsonl --out gdt_validator/models/gdt_ner
python -m gdt_validator.tools.train_gdt_ner predict --model gdt_validator/models/gdt_ner --text "POSITION ⌀0.25 M A B"
```

Add multiple text samples in the annotation window, choose the entity label, and select the exact character span. Save with the **Save** button or Ctrl+S. A useful NER model requires a representative set of accurately labeled examples.

## Troubleshooting

- **`No module named gdt_validator`:** run module commands from the repository root. From inside `gdt_validator`, use the script form, for example `python tools/annotate_symbols.py`.
- **`spaCy is missing`:** install requirements using the same interpreter used for NER training: `python -m pip install -r gdt_validator\requirements.txt`.
- **Ollama connection/model error:** start the Ollama service, run `ollama pull qwen2.5vl:7b`, and check with `python -m gdt_validator.main --check-ollama`.
- **No FCFs found:** verify the selected zero-based page and that the PDF contains detectable vector lines. Scanned/raster-only drawings may need OCR/vector preprocessing or manual review.
- **Report withheld:** inspect `manual_review.json` and the linked crops, fill or correct the required values, then rerun with `--reviewed-fcf-json`.
