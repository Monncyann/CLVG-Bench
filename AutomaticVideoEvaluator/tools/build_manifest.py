#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build checksums for one prepared AVE task.")
    parser.add_argument("--data-dir", default="data/abnormality")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / args.data_dir
    splits = {}
    generated_videos = set()
    reference_images = set()
    reference_videos = set()
    for name in ("train", "val", "test"):
        path = data / f"{name}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        generated_videos.update(row["video"] for row in rows)
        for row in rows:
            reference_images.update(row.get("reference_images", []))
            reference_videos.update(row.get("reference_videos", []))
        splits[name] = {
            "samples": len(rows),
            "positive_abnormality": sum(bool(row["feedback"]["abnormality"].strip()) for row in rows),
            "sha256": sha256(path),
        }
    files = []
    referenced = generated_videos | reference_images | reference_videos
    for relative in sorted(referenced):
        path = data / relative
        if relative in generated_videos:
            kind = "generated_video"
        elif relative in reference_images:
            kind = "reference_image"
        else:
            kind = "reference_video"
        files.append({
            "path": relative,
            "kind": kind,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    payload = {
        "format_version": 1,
        "splits": splits,
        "unique_videos": len(generated_videos),
        "unique_reference_images": len(reference_images),
        "unique_reference_videos": len(reference_videos),
        "video_bytes": sum(item["bytes"] for item in files if item["kind"] == "generated_video"),
        "media_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    (data / "MANIFEST.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
