"""Client for the local SAM Audio model service."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class SeparationResult:
    directory: Path
    metadata: dict[str, Any]
    stems: dict[str, Path]
    response_headers: dict[str, str]


class SAMAudioClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def separate(
        self,
        audio_path: Path,
        output_dir: Path,
        *,
        order: str = "music_first",
        targets: tuple[str, ...] = ("music", "voice"),
    ) -> SeparationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_path = output_dir / "separation.zip"
        with audio_path.open("rb") as audio, httpx.Client(timeout=None) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/v1/separate",
                files={"audio": (audio_path.name, audio, "audio/wav")},
                data={"order": order, "targets": ",".join(targets)},
            ) as response:
                response.raise_for_status()
                headers = dict(response.headers)
                with archive_path.open("wb") as destination:
                    for block in response.iter_bytes():
                        destination.write(block)
        extracted = output_dir / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            archive.extractall(extracted)
        metadata = json.loads((extracted / "metadata.json").read_text())
        artifact_metadata = metadata.get("artifacts", {})
        canonical = artifact_metadata.get("canonical_stems", {})
        required = set(targets) | {"sfx"}
        if not required.issubset(canonical):
            raise ValueError(
                f"SAM API metadata must map canonical stems: {sorted(required)}"
            )
        stems = {stem: extracted / canonical[stem] for stem in sorted(required)}
        missing = [stem for stem, path in stems.items() if not path.is_file()]
        if missing:
            raise ValueError(f"SAM API response omitted stem files: {missing}")
        return SeparationResult(
            directory=extracted,
            metadata=metadata,
            stems=stems,
            response_headers=headers,
        )
