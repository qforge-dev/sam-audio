"""Local browser app for manually reviewing an acquired audio dataset."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

Decision = Literal["good", "perfect", "not_ok"]
Reason = Literal[
    "lacking_music",
    "lacking_background_audio",
    "vocal_music",
    "speech_not_dialogue",
    "too_low_quality",
    "too_quiet",
    "distorted_or_clipped",
    "wrong_balance",
    "other",
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ReviewUpdate(BaseModel):
    decision: Decision
    reasons: list[Reason] = Field(default_factory=list)
    note: str = Field(default="", max_length=1000)


class ReviewStore:
    def __init__(
        self,
        dataset_dir: Path,
        *,
        audio_directory: str,
        annotations_path: Path | None = None,
    ):
        self.dataset_dir = dataset_dir.resolve()
        self.audio_directory = audio_directory
        self.audio_dir = (self.dataset_dir / audio_directory).resolve()
        self.manifest_path = self.dataset_dir / "manifest.json"
        self.annotations_path = (
            annotations_path.resolve()
            if annotations_path
            else self.dataset_dir / "manual-review.json"
        )
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")
        self.manifest = json.loads(self.manifest_path.read_text())
        self.records_by_name = {
            Path(record["local_path"]).name: record
            for record in self.manifest.get("records", [])
        }
        self.filenames = self._selected_filenames()
        self.filename_set = set(self.filenames)
        self.lock = threading.Lock()
        self.reviews = self._load_reviews()

    def _selected_filenames(self) -> list[str]:
        subset = self.manifest.get("balanced_listening_subset", {})
        if self.audio_directory == subset.get("local_directory"):
            candidates = [str(name) for name in subset.get("filenames", [])]
        else:
            candidates = [path.name for path in sorted(self.audio_dir.glob("*.wav"))]
        filenames = [
            name
            for name in candidates
            if Path(name).name == name and (self.audio_dir / name).is_file()
        ]
        if not filenames:
            raise ValueError(f"No WAV files found in {self.audio_dir}")
        return filenames

    def _load_reviews(self) -> dict[str, dict[str, Any]]:
        if not self.annotations_path.exists():
            return {}
        payload = json.loads(self.annotations_path.read_text())
        reviews = payload.get("reviews", {})
        if not isinstance(reviews, dict):
            raise ValueError("manual-review.json has an invalid reviews object")
        return {
            filename: review
            for filename, review in reviews.items()
            if filename in self.filename_set and isinstance(review, dict)
        }

    def _save(self) -> None:
        payload = {
            "schema_version": 1,
            "dataset_dir": str(self.dataset_dir),
            "audio_directory": self.audio_directory,
            "updated_at": _now(),
            "reviews": self.reviews,
        }
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.annotations_path.with_suffix(
            self.annotations_path.suffix + ".tmp"
        )
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, self.annotations_path)

    def _record_summary(self, filename: str) -> dict[str, Any]:
        record = self.records_by_name.get(filename, {})
        validation = record.get("m2d_validation", {})
        labels = Counter(
            label["name"]
            for window in validation.get("windows", [])
            for label in window.get("top_labels", [])[:3]
        )
        return {
            "filename": filename,
            "title": record.get("title"),
            "uploader": record.get("uploader"),
            "source_url": record.get("source_url"),
            "background_bucket": validation.get("background_bucket"),
            "speech_coverage": validation.get("speech_coverage"),
            "background_coverage": validation.get("background_coverage"),
            "overlap_coverage": validation.get("overlap_coverage"),
            "vocal_music_coverage": validation.get("vocal_music_coverage"),
            "top_labels": [name for name, _ in labels.most_common(6)],
            "review": self.reviews.get(filename),
        }

    def state(self) -> dict[str, Any]:
        clips = [self._record_summary(filename) for filename in self.filenames]
        decisions = Counter(
            review.get("decision") for review in self.reviews.values()
        )
        reviewed = sum(decisions.values())
        return {
            "dataset": {
                "name": self.manifest.get("name"),
                "dataset_dir": str(self.dataset_dir),
                "audio_directory": self.audio_directory,
                "annotations_path": str(self.annotations_path),
            },
            "summary": {
                "total": len(self.filenames),
                "reviewed": reviewed,
                "unreviewed": len(self.filenames) - reviewed,
                "good": decisions["good"],
                "perfect": decisions["perfect"],
                "not_ok": decisions["not_ok"],
            },
            "reason_labels": {
                "lacking_music": "Lacking music",
                "lacking_background_audio": "Lacking background audio / SFX",
                "vocal_music": "Singing or vocal music",
                "speech_not_dialogue": "Speech is not dialogue",
                "too_low_quality": "Too low quality",
                "too_quiet": "Too quiet",
                "distorted_or_clipped": "Distorted or clipped",
                "wrong_balance": "Wrong voice/background balance",
                "other": "Other",
            },
            "clips": clips,
        }

    def audio_path(self, filename: str) -> Path:
        if Path(filename).name != filename or filename not in self.filename_set:
            raise KeyError(filename)
        path = self.audio_dir / filename
        if not path.is_file():
            raise KeyError(filename)
        return path

    def update(self, filename: str, update: ReviewUpdate) -> dict[str, Any]:
        if filename not in self.filename_set:
            raise KeyError(filename)
        if update.decision == "not_ok" and not (update.reasons or update.note.strip()):
            raise ValueError("Not OK requires at least one reason or a note")
        if "other" in update.reasons and not update.note.strip():
            raise ValueError("The Other reason requires a note")
        reasons = (
            list(dict.fromkeys(update.reasons))
            if update.decision == "not_ok"
            else []
        )
        note = update.note.strip() if update.decision == "not_ok" else ""
        review = {
            "decision": update.decision,
            "reasons": reasons,
            "note": note,
            "updated_at": _now(),
        }
        with self.lock:
            self.reviews[filename] = review
            self._save()
        return review

    def clear(self, filename: str) -> None:
        if filename not in self.filename_set:
            raise KeyError(filename)
        with self.lock:
            self.reviews.pop(filename, None)
            self._save()

    def export_csv(self) -> str:
        destination = io.StringIO()
        fields = [
            "filename",
            "decision",
            "reasons",
            "note",
            "updated_at",
            "background_bucket",
            "title",
            "source_url",
        ]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for filename in self.filenames:
            summary = self._record_summary(filename)
            review = self.reviews.get(filename, {})
            writer.writerow(
                {
                    "filename": filename,
                    "decision": review.get("decision", ""),
                    "reasons": "|".join(review.get("reasons", [])),
                    "note": review.get("note", ""),
                    "updated_at": review.get("updated_at", ""),
                    "background_bucket": summary["background_bucket"],
                    "title": summary["title"],
                    "source_url": summary["source_url"],
                }
            )
        return destination.getvalue()


def create_review_app(store: ReviewStore) -> FastAPI:
    app = FastAPI(title="SAM Audio Manual Review", version="1.0.0")
    html_path = Path(__file__).parent / "web" / "manual_review.html"

    def page() -> HTMLResponse:
        return HTMLResponse(html_path.read_text())

    @app.get("/")
    def index() -> HTMLResponse:
        return page()

    @app.get("/clip/{filename}")
    def clip_page(filename: str) -> HTMLResponse:
        try:
            store.audio_path(filename)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Clip not found") from error
        return page()

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return store.state()

    @app.get("/api/audio/{filename}")
    def audio(filename: str) -> FileResponse:
        try:
            path = store.audio_path(filename)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Clip not found") from error
        return FileResponse(path, media_type="audio/wav")

    @app.put("/api/reviews/{filename}")
    def update_review(filename: str, update: ReviewUpdate) -> dict[str, Any]:
        try:
            review = store.update(filename, update)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Clip not found") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "filename": filename,
            "review": review,
            "summary": store.state()["summary"],
        }

    @app.delete("/api/reviews/{filename}")
    def clear_review(filename: str) -> dict[str, Any]:
        try:
            store.clear(filename)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Clip not found") from error
        return {
            "filename": filename,
            "review": None,
            "summary": store.state()["summary"],
        }

    @app.get("/api/export.csv")
    def export_csv() -> StreamingResponse:
        return StreamingResponse(
            iter([store.export_csv()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=manual-review.csv"},
        )

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--audio-directory", default="balanced-audio")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    return parser


def main() -> None:
    args = _parser().parse_args()
    store = ReviewStore(
        args.dataset_dir,
        audio_directory=args.audio_directory,
        annotations_path=args.annotations,
    )
    uvicorn.run(create_review_app(store), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
