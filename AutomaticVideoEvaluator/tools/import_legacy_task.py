#!/usr/bin/env python3
"""Convert one legacy AVE task into the self-contained release schema."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-splits", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source_splits = args.source_splits.resolve()
    repository_root = args.repository_root.resolve()
    destination = args.destination.resolve()
    videos = destination / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for split in ("train", "val", "test"):
        rows = json.loads((source_splits / f"{split}.json").read_text(encoding="utf-8"))
        output = []
        for row in rows:
            source_video = Path(row["Video"])
            if not source_video.is_absolute():
                source_video = repository_root / source_video
            source_video = source_video.resolve()
            key = str(source_video)
            if key not in copied:
                filename = hashlib.sha256(key.encode()).hexdigest()[:20] + source_video.suffix.lower()
                shutil.copy2(source_video, videos / filename)
                copied[key] = f"videos/{filename}"
            output.append({
                "id": str(row["ID"]),
                "prompt": row.get("Prompt", ""),
                "reference_images": row.get("ReferenceImages", []),
                "video": copied[key],
                "label": row.get("Label"),
                "feedback": row["Feedback"],
            })
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{split}.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
