"""Tkinter desktop interface for selecting and validating drawing PDFs."""

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from gdt_validator.main import check_ollama, run_pipeline
from gdt_validator.feedback import FeedbackStore
from gdt_validator.knowledge_base import (
    DEFAULT_CHROMA_PATH, DEFAULT_KB_PATH, build_chroma_index, build_knowledge_base,
)


class ValidatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ASME Y14.5 GD&T Validator")
        self.root.geometry("760x430")
        self.root.minsize(680, 390)

        self.pdf_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path.cwd() / "output"))
        self.model_var = tk.StringVar(value="qwen2.5vl:7b")
        self.page_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="Choose a drawing PDF to begin.")
        self.messages: queue.Queue = queue.Queue()
        self.last_result = None
        self._build_ui()
        self.root.after(150, self._poll_messages)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="ASME Y14.5 GD&T Validator", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 18))

        ttk.Label(frame, text="Drawing PDF").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.pdf_var).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Browse…", command=self._browse_pdf).grid(row=1, column=2)

        ttk.Label(frame, text="Output folder").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.output_var).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Browse…", command=self._browse_output).grid(row=2, column=2)

        ttk.Label(frame, text="Ollama model").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.model_var).grid(row=3, column=1, sticky="ew", padx=8)
        ttk.Label(frame, text="e.g. qwen2.5vl:7b").grid(row=3, column=2, sticky="w")

        ttk.Label(frame, text="PDF page (zero-based)").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Spinbox(frame, from_=0, to=9999, textvariable=self.page_var, width=8).grid(
            row=4, column=1, sticky="w", padx=8)

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(22, 8))
        ttk.Label(frame, textvariable=self.status_var, wraplength=640).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(0, 12))

        buttons = ttk.Frame(frame)
        buttons.grid(row=8, column=0, columnspan=3, sticky="ew")
        self.kb_button = ttk.Button(buttons, text="Build / refresh from scanned PDF",
                                    command=self._build_kb)
        self.kb_button.pack(side="left", padx=5)
        self.index_button = ttk.Button(buttons, text="Index existing OCR archive",
                                       command=self._index_kb)
        self.index_button.pack(side="left", padx=5)
        ttk.Button(buttons, text="Check Ollama", command=self._check_ollama).pack(side="left", padx=5)
        self.run_button = ttk.Button(buttons, text="Validate drawing", command=self._run_pipeline)
        self.run_button.pack(side="left", padx=5)
        self.feedback_button = ttk.Button(buttons, text="Rate last report", command=self._rate_report)
        self.feedback_button.pack(side="left", padx=5)
        ttk.Button(buttons, text="Retrain from feedback", command=self._train_feedback).pack(side="left", padx=5)
        ttk.Label(frame, text=f"OCR archive: {DEFAULT_KB_PATH}\nVector store: {DEFAULT_CHROMA_PATH}",
                  wraplength=650).grid(row=9, column=0, columnspan=3, sticky="w", padx=5, pady=(8, 0))

    def _browse_pdf(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select engineering drawing PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if filename:
            self.pdf_var.set(filename)

    def _browse_output(self) -> None:
        directory = filedialog.askdirectory(title="Select output folder")
        if directory:
            self.output_var.set(directory)

    def _set_busy(self, busy: bool, message: str) -> None:
        self.status_var.set(message)
        self.run_button.configure(state="disabled" if busy else "normal")
        if hasattr(self, "feedback_button"):
            self.feedback_button.configure(state="disabled" if busy else "normal")
        if hasattr(self, "kb_button"):
            self.kb_button.configure(state="disabled" if busy else "normal")
        if hasattr(self, "index_button"):
            self.index_button.configure(state="disabled" if busy else "normal")
        if busy:
            self.progress.start(10)
        else:
            self.progress.stop()

    def _check_ollama(self) -> None:
        model = self.model_var.get().strip()
        if not model:
            messagebox.showerror("Missing model", "Enter an Ollama model name.")
            return
        self._set_busy(True, "Checking Ollama service and model…")

        def worker():
            try:
                check_ollama(model)
                self.messages.put(("success", f"Ollama is ready with model {model!r}."))
            except Exception as exc:
                self.messages.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _build_kb(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select ASME Y14.5M-1994 scan to index",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not filename:
            return
        self._set_busy(True, "Preparing the OCR knowledge base…")

        def progress(stage, current, total, detail):
            label = "OCR pages" if stage == "ocr" else "HNSW vectors"
            unit = "chunks" if stage == "ocr" else "vectors"
            self.messages.put(("progress", f"Building {label}: {current}/{total} ({detail} {unit})…"))

        def worker():
            try:
                result = build_knowledge_base(filename, progress_callback=progress)
                self.messages.put(("kb_complete", result))
            except Exception as exc:
                self.messages.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _index_kb(self) -> None:
        self._set_busy(True, "Building the Chroma HNSW index from the OCR archive…")

        def progress(stage, current, total, detail):
            self.messages.put(("progress", f"Embedding chunks: batch {current}/{total} ({detail} vectors)…"))

        def worker():
            try:
                result = build_chroma_index(progress_callback=progress)
                self.messages.put(("kb_complete", result))
            except Exception as exc:
                self.messages.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _run_pipeline(self) -> None:
        pdf_path = self.pdf_var.get().strip()
        output_dir = self.output_var.get().strip()
        model = self.model_var.get().strip()
        try:
            page_num = int(self.page_var.get())
            if page_num < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid page", "Enter a zero-based page number (0 or higher).")
            return
        if not pdf_path or not Path(pdf_path).is_file():
            messagebox.showerror("Select a PDF", "Browse to an existing drawing PDF first.")
            return
        if not output_dir or not model:
            messagebox.showerror("Missing information", "Choose an output folder and enter a model name.")
            return

        self._set_busy(True, "Validating drawing. This can take several minutes…")

        def worker():
            try:
                result = run_pipeline(pdf_path, page_num=page_num,
                                      output_dir=output_dir, model=model)
                self.messages.put(("complete", result))
            except Exception as exc:
                self.messages.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _rate_report(self) -> None:
        if not self.last_result:
            messagebox.showinfo("No report yet", "Validate a drawing before rating a report.")
            return
        positive = messagebox.askyesno(
            "Report feedback", "Was this report useful and factually correct?\n\n"
            "Choose No if the extraction or findings need correction.")
        comment = simpledialog.askstring("Feedback details", "Optional: what should be corrected?") or ""
        try:
            store = FeedbackStore(self.last_result["feedback_db"])
            count = store.add_feedback(self.last_result["report_id"], int(positive), comment)
            probability = store.score(self.last_result["report_id"])
            messagebox.showinfo(
                "Feedback saved",
                f"Saved label for report {self.last_result['report_id']}. "
                f"The local model was updated from {count} feedback sample(s). "
                + (f"Current review-priority score: {probability:.2f}." if probability is not None else ""),
            )
        except Exception as exc:
            messagebox.showerror("Feedback failed", str(exc))

    def _train_feedback(self) -> None:
        if not self.last_result:
            messagebox.showinfo("No feedback data", "Validate a drawing first to create a feedback database.")
            return
        self._set_busy(True, "Retraining the local feedback model...")
        store = FeedbackStore(self.last_result["feedback_db"])
        try:
            message = store.retrain()
            self._set_busy(False, "Ready.")
            messagebox.showinfo("Feedback model updated", message)
        except Exception as exc:
            self._set_busy(False, "Ready.")
            messagebox.showerror("Retraining failed", str(exc))

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "progress":
                    self.status_var.set(payload)
                    continue
                self._set_busy(False, "Ready.")
                if kind == "error":
                    self.status_var.set("Operation failed.")
                    messagebox.showerror("GD&T Validator", payload)
                elif kind == "success":
                    self.status_var.set(payload)
                    messagebox.showinfo("Ollama ready", payload)
                elif kind == "kb_complete":
                    self.status_var.set(
                        f"Chroma HNSW index ready: {payload['vector_count']} vectors.")
                    message = (f"Indexed {payload['vector_count']} chunks in Chroma.\n\n"
                               f"Collection: {payload['collection_name']}\n"
                               f"Vector store:\n{payload['vector_store_path']}\n\n"
                               f"HNSW settings: {payload['hnsw_metadata']}")
                    if payload.get("indexed_pages") is not None:
                        message += (f"\n\nOCR pages: {payload['indexed_pages']}\n"
                                    f"OCR chunks: {payload['chunk_count']}\n"
                                    f"Pages with OCR errors: {len(payload['failed_pages'])}")
                    messagebox.showinfo("Knowledge base ready", message)
                else:
                    self.last_result = payload
                    self.status_var.set(
                        f"Complete: {payload['fcf_count']} FCF(s), "
                        f"{len(payload['violations'])} violation(s).")
                    messagebox.showinfo(
                        "Validation complete",
                        f"Annotated drawing:\n{payload['annotated_pdf']}\n\n"
                        f"Report:\n{payload['report_pdf']}\n\n"
                        "You can rate this report using the Feedback button.",
                    )
        except queue.Empty:
            pass
        self.root.after(150, self._poll_messages)


def main() -> None:
    root = tk.Tk()
    ValidatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
