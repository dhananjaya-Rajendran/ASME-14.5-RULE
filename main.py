import argparse
import json
import sys
from pathlib import Path

# Support both `python -m gdt_validator.main` from the workspace root and
# `python main.py` from inside the gdt_validator directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gdt_validator.ingestion.datum_vector_detector import (
    VectorDatumDetector, resolve_declared_datums,
)
from gdt_validator.ingestion.fcf_vector_detector import VectorFCFDetector
from gdt_validator.ingestion.leader_line_tracer import LeaderLineTracer
from gdt_validator.reporting.pdf_annotator import annotate_drawing_pdf
from gdt_validator.reporting.report_builder import EngineeringReportBuilder
from gdt_validator.knowledge_base import (
    DEFAULT_CHROMA_PATH, DEFAULT_KB_PATH, build_chroma_index, build_knowledge_base,
)
from gdt_validator.schemas import DrawingExtraction, FeatureControlFrame
from gdt_validator.validation.asme_rule_engine import ASMEY145RuleEngine
from gdt_validator.validation.rag_explainer import ASMERagExplainer
from gdt_validator.vision.vlm_client import analyze_roi_with_qwen
from gdt_validator.feedback import FeedbackStore


def run_pipeline(pdf_path: str, page_num: int = 0, output_dir: str = "output",
                 model: str = "qwen2.5vl:7b",
                 reviewed_fcf_json: str | None = None) -> dict:
    pdf_file = Path(pdf_path).expanduser().resolve()
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")
    work_dir = Path(output_dir) / pdf_file.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    if reviewed_fcf_json:
        reviewed_path = Path(reviewed_fcf_json).expanduser().resolve()
        reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
        if not isinstance(reviewed, dict) or not isinstance(reviewed.get("feature_control_frames"), list):
            raise ValueError("Reviewed FCF JSON must contain a feature_control_frames array.")
        fcfs = [FeatureControlFrame.model_validate(item)
                for item in reviewed["feature_control_frames"]]
        declared = reviewed.get("declared_datums", [])
        if not isinstance(declared, list) or not all(
                isinstance(item, str) and len(item.strip()) == 1 for item in declared):
            raise ValueError("declared_datums must contain single-letter strings.")
        if not fcfs:
            raise ValueError("Reviewed FCF JSON contains no feature control frames.")
        if any(frame.tolerance_value is None or frame.material_condition is None
               or frame.characteristic_symbol == "unknown" for frame in fcfs):
            raise ValueError("Reviewed FCF JSON still contains unverified required FCF fields.")
    else:
        datum_rois = VectorDatumDetector().detect_datum_symbols(
            str(pdf_file), page_num=page_num, out_dir=str(work_dir / "datum_crops"))
        declared = resolve_declared_datums(datum_rois, out_dir=str(work_dir / "datum_crops"), model=model)
        fcf_rois = VectorFCFDetector().extract_fcf_rois(
            str(pdf_file), page_num=page_num, out_dir=str(work_dir / "fcf_crops"))
        if not fcf_rois:
            raise ValueError(
                f"No feature control frames were detected on page {page_num}. "
                "No validation report was generated. Check the selected page, scan quality, "
                "and FCF detector crops before interpreting any result as compliant.")
        tracer = LeaderLineTracer()
        fcfs = []
        unresolved = []
        for roi in fcf_rois:
            data = analyze_roi_with_qwen(
                roi["image_path"], roi["raw_pdf_text"], roi["roi_id"],
                roi["bbox"], model=model)
            data.update(tracer.trace_fcf_target(str(pdf_file), roi["bbox"], page_num=page_num))
            frame = FeatureControlFrame.model_validate(data)
            fcfs.append(frame)
            missing = []
            if frame.characteristic_symbol == "unknown":
                missing.append("characteristic_symbol")
            if frame.tolerance_value is None:
                missing.append("tolerance_value")
            if frame.material_condition is None:
                missing.append("material_condition")
            if missing:
                unresolved.append({
                    "fcf_id": frame.id, "missing_fields": missing,
                    "bbox": frame.bbox, "crop_image": roi["image_path"],
                    "current_extraction": frame.model_dump(mode="json"),
                })

    if not reviewed_fcf_json and unresolved:
        review_path = work_dir / "manual_review.json"
        review_path.write_text(json.dumps({
            "drawing_id": pdf_file.stem,
            "message": "Review the source crops and provide explicit values before generating a validation report.",
            "required_developer_inputs": [
                "Confirm characteristic symbol class for each FCF crop.",
                "Transcribe the numeric tolerance and whether the tolerance zone has a diameter symbol.",
                "Transcribe each datum reference in order and any material modifier in the FCF cells.",
                "Confirm the applicable ASME Y14.5 edition and provide the authorized standard source PDF if the local RAG archive is absent.",
                "Confirm each extracted FCF field against its source crop before validation.",
            ],
            "items": unresolved,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        raise ValueError(
            f"Report withheld: {len(unresolved)} FCF(s) have unverified fields. "
            f"Review crops and fill {review_path} before rerunning.")

    drawing = DrawingExtraction(
        drawing_id=pdf_file.stem,
        declared_datums=declared,
        feature_control_frames=fcfs,
    )
    violations = ASMERagExplainer().enrich_violations(ASMEY145RuleEngine().validate(drawing))
    annotated_path = work_dir / "annotated.pdf"
    report_path = work_dir / "report.pdf"
    annotate_drawing_pdf(str(pdf_file), str(annotated_path), violations, page_num=page_num)
    feedback_store = FeedbackStore(work_dir / "feedback.sqlite3")
    report_id = feedback_store.record_report(pdf_file.name, drawing, violations)
    EngineeringReportBuilder(str(report_path)).generate_report(
        drawing, violations, source_pdf_path=str(pdf_file))
    return {"drawing": drawing, "violations": violations,
            "annotated_pdf": str(annotated_path), "report_pdf": str(report_path),
            "fcf_count": len(fcfs), "datum_count": len(declared), "report_id": report_id,
            "feedback_db": str(feedback_store.path)}


def create_review_template(pdf_path: str, page_num: int, output_path: str) -> str:
    """Create crop-linked blank FCF data for manual transcription."""
    pdf_file = Path(pdf_path).expanduser().resolve()
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")
    target = Path(output_path).expanduser().resolve()
    rois = VectorFCFDetector().extract_fcf_rois(
        str(pdf_file), page_num=page_num, out_dir=str(target.parent / "fcf_crops"))
    if not rois:
        raise ValueError("No FCF candidates found; check the selected page and scan quality.")
    content = {
        "drawing_id": pdf_file.stem,
        "applicable_standard": "ASME Y14.5M-1994",
        "declared_datums": [],
        "feature_control_frames": [
            {"id": roi["roi_id"], "bbox": roi["bbox"],
             "characteristic_symbol": "", "tolerance_value": None,
             "diameter_zone": False, "material_condition": None, "datums": [],
             "associated_feature_type": "unknown", "is_feature_of_size": None,
             "has_size_tolerance_on_feature": None,
             "crop_image": roi["image_path"],
             "developer_note": "Transcribe verified values from this crop."}
            for roi in rois
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(target)


def main() -> None:
    # With no command-line arguments, launch the file-picker interface.
    if len(sys.argv) == 1:
        from gdt_validator.ui import main as launch_ui

        launch_ui()
        return
    parser = argparse.ArgumentParser(description="Extract and check GD&T feature control frames in a PDF.")
    parser.add_argument("--gui", action="store_true", help="Open the desktop interface.")
    parser.add_argument("--pdf", help="Input drawing PDF")
    parser.add_argument("--page", type=int, default=0, help="Zero-based page number")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument("--reviewed-fcf-json", help="JSON containing developer-verified FCF data and declared datums.")
    parser.add_argument("--create-review-template", metavar="JSON",
                        help="Create a crop-linked JSON template for manual FCF transcription.")
    parser.add_argument("--build-kb", metavar="PDF", help="OCR a standards PDF into the local RAG knowledge base.")
    parser.add_argument("--index-kb", action="store_true", help="Build the Chroma HNSW index from an existing OCR archive.")
    parser.add_argument("--kb-path", default=str(DEFAULT_KB_PATH), help="SQLite OCR archive path.")
    parser.add_argument("--chroma-path", default=str(DEFAULT_CHROMA_PATH), help="Persistent Chroma storage path.")
    parser.add_argument("--ocr-dpi", type=int, default=120, help="OCR render resolution when building the knowledge base.")
    parser.add_argument(
        "--check-ollama", action="store_true",
        help="Check that Ollama is reachable and the selected model is installed, then exit.",
    )
    parser.add_argument("--feedback-db", help="Feedback database created by a prior report.")
    parser.add_argument("--report-id", help="Report id returned by validation.")
    parser.add_argument("--feedback-label", type=int, choices=[0, 1],
                        help="Label report: 1 = useful/correct, 0 = incorrect.")
    parser.add_argument("--feedback-comment", default="", help="Optional feedback note.")
    args = parser.parse_args()
    if args.gui:
        from gdt_validator.ui import main as launch_ui

        launch_ui()
        return
    if args.create_review_template:
        if not args.pdf:
            parser.error("--create-review-template requires --pdf")
        print(f"Review template: {create_review_template(args.pdf, args.page, args.create_review_template)}")
        return
    if args.build_kb:
        result = build_knowledge_base(
            args.build_kb, database_path=args.kb_path, dpi=args.ocr_dpi,
            chroma_path=args.chroma_path,
            progress_callback=lambda stage, current, total, detail: print(
                f"{stage}: {current}/{total}; {detail} chunks", flush=True))
        print(f"Knowledge base created: {result['database_path']}")
        print(f"Indexed {result['indexed_pages']} pages into {result['chunk_count']} chunks.")
        print(f"Chroma collection: {result['collection_name']} ({result['vector_count']} vectors)")
        if result["failed_pages"]:
            print(f"OCR failed on {len(result['failed_pages'])} page(s).")
        return
    if args.index_kb:
        result = build_chroma_index(
            database_path=args.kb_path, chroma_path=args.chroma_path,
            progress_callback=lambda stage, current, total, detail: print(
                f"{stage}: {current}/{total}; {detail} vectors", flush=True))
        print(f"Chroma collection {result['collection_name']} indexed {result['vector_count']} chunks.")
        print(f"Storage: {result['vector_store_path']}")
        return
    if args.check_ollama:
        check_ollama(args.model)
        print(f"Ollama is ready with model {args.model!r}.")
        return
    if args.feedback_label is not None:
        if not args.feedback_db or not args.report_id:
            parser.error("--feedback-label requires --feedback-db and --report-id")
        store = FeedbackStore(args.feedback_db)
        count = store.add_feedback(args.report_id, args.feedback_label, args.feedback_comment)
        score = store.score(args.report_id)
        print(f"Feedback saved; trained on {count} labeled sample(s).")
        if score is not None:
            print(f"Review-priority score: {score:.3f}")
        return
    if not args.pdf:
        parser.error("--pdf is required unless --check-ollama or --build-kb is used")
    result = run_pipeline(args.pdf, page_num=args.page, output_dir=args.output_dir, model=args.model,
                          reviewed_fcf_json=args.reviewed_fcf_json)
    print(f"Pipeline completed: {result['fcf_count']} FCF(s), "
          f"{len(result['violations'])} violation(s).")
    print(f"Annotated PDF: {result['annotated_pdf']}")
    print(f"Report PDF: {result['report_pdf']}")


def check_ollama(model: str) -> None:
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "The Ollama Python client is missing. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    try:
        models = ollama.list()
    except Exception as exc:
        raise RuntimeError(
            "Cannot reach Ollama. Start the Ollama app/service and retry."
        ) from exc
    available = {
        item.get("name") or item.get("model")
        for item in models.get("models", [])
    }
    if model not in available and f"{model}:latest" not in available:
        raise RuntimeError(
            f"Ollama is running, but model {model!r} is not installed. "
            f"Run `ollama pull {model}` first."
        )


if __name__ == "__main__":
    main()
