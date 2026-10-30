"""Persist report feedback and train a small review-priority model."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gdt_validator.schemas import DrawingExtraction, ValidationViolation


def _feature_vector(payload: dict) -> list[float]:
    """Stable numeric features describing an extraction/report, not ASME validity."""
    frames = payload.get("feature_control_frames", [])
    findings = payload.get("findings", [])
    return [
        1.0,
        min(len(frames), 100) / 100,
        min(len(findings), 100) / 100,
        sum(f.get("severity") == "ERROR" for f in findings) / max(1, len(findings)),
        sum(f.get("severity") == "WARNING" for f in findings) / max(1, len(findings)),
        sum(f.get("severity") == "REQUIRES_CAD_DATA" for f in findings) / max(1, len(findings)),
        sum(bool(f.get("knowledge_context")) for f in findings) / max(1, len(findings)),
        sum(not f.get("datums") for f in frames) / max(1, len(frames)),
        sum(f.get("characteristic_symbol") == "unknown" for f in frames) / max(1, len(frames)),
    ]


class FeedbackStore:
    """SQLite report/label archive with logistic-regression SGD backprop updates."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY, source_name TEXT NOT NULL,
                created_at TEXT NOT NULL, payload TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY, report_id TEXT NOT NULL,
                label INTEGER NOT NULL CHECK(label IN (0,1)), comment TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(report_id) REFERENCES reports(report_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS model_weights (
                feature_index INTEGER PRIMARY KEY, weight REAL NOT NULL)""")

    def record_report(self, source_name: str, drawing: DrawingExtraction,
                      violations: list[ValidationViolation]) -> str:
        payload = {
            "drawing_id": drawing.drawing_id,
            "applicable_standard": drawing.applicable_standard,
            "feature_control_frames": [f.model_dump(mode="json") for f in drawing.feature_control_frames],
            "findings": [v.model_dump(mode="json") for v in violations],
        }
        report_id = hashlib.sha256(
            (source_name + datetime.now(timezone.utc).isoformat() + json.dumps(payload, sort_keys=True)).encode()
        ).hexdigest()[:16]
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO reports VALUES (?, ?, ?, ?)",
                       (report_id, source_name, datetime.now(timezone.utc).isoformat(),
                        json.dumps(payload, ensure_ascii=False)))
        return report_id

    def add_feedback(self, report_id: str, label: int, comment: str = "") -> int:
        if label not in (0, 1):
            raise ValueError("Feedback label must be 1 (useful/correct) or 0 (incorrect).")
        if len(comment) > 4000:
            raise ValueError("Feedback comments must be 4000 characters or fewer.")
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT payload FROM reports WHERE report_id=?", (report_id,)).fetchone()
            if not row:
                raise KeyError(f"Unknown report id: {report_id}")
            db.execute("INSERT INTO feedback(report_id,label,comment,created_at) VALUES (?,?,?,?)",
                       (report_id, label, comment.strip(), datetime.now(timezone.utc).isoformat()))
            self._train(db)
            count = db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        return count

    def retrain(self, epochs: int = 30, learning_rate: float = 0.15) -> str:
        """Recompute weights from all labeled samples using gradient descent."""
        with sqlite3.connect(self.path) as db:
            count = db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            if count == 0:
                return "No labeled samples are available."
            self._train(db, epochs=epochs, learning_rate=learning_rate)
            labels = {row[0] for row in db.execute("SELECT DISTINCT label FROM feedback")}
        classes = "both positive and negative" if labels == {0, 1} else "one feedback class"
        return f"Updated model from {count} labeled sample(s) ({classes}); {epochs} SGD epochs completed."

    @staticmethod
    def _train(db: sqlite3.Connection, epochs: int = 30, learning_rate: float = 0.15) -> None:
        examples = [(json.loads(payload), label) for payload, label in db.execute(
            "SELECT reports.payload, feedback.label FROM feedback JOIN reports USING(report_id)")]
        if not examples:
            return
        weights = [row[0] for row in db.execute("SELECT weight FROM model_weights ORDER BY feature_index")]
        size = len(_feature_vector(examples[0][0]))
        weights = (weights + [0.0] * size)[:size]
        # SGD computes the logistic loss gradient (backprop through sigmoid).
        for _ in range(epochs):
            for payload, target in examples:
                features = _feature_vector(payload)
                score = sum(w * x for w, x in zip(weights, features))
                prediction = 1 / (1 + math.exp(-max(-30, min(30, score))))
                error = prediction - target
                weights = [w - learning_rate * error * x
                           for w, x in zip(weights, features)]
        db.execute("DELETE FROM model_weights")
        db.executemany("INSERT INTO model_weights VALUES (?,?)", enumerate(weights))

    def score(self, report_id: str) -> Optional[float]:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT payload FROM reports WHERE report_id=?", (report_id,)).fetchone()
            if not row:
                return None
            weights = [item[0] for item in db.execute(
                "SELECT weight FROM model_weights ORDER BY feature_index")]
        if not weights:
            return None
        score = sum(w * x for w, x in zip(weights, _feature_vector(json.loads(row[0]))))
        return 1 / (1 + math.exp(-max(-30, min(30, score))))

