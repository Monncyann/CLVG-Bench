#!/usr/bin/env python3
"""Recover prompt-following media and metadata from the legacy source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv"}


def resolve_legacy_path(repository_root: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    if candidate.is_file():
        return candidate.resolve()

    # The legacy JSON contains one directory named "101 " although the folder
    # on disk is "101". Only trim leading/trailing whitespace from components.
    normalized = Path(*[part.strip() for part in Path(value).parts])
    if not normalized.is_absolute():
        normalized = repository_root / normalized
    if not normalized.is_file():
        raise FileNotFoundError(f"Referenced source file does not exist: {value}")
    return normalized.resolve()


def unique_existing(paths: Iterable[Path], suffixes: set[str]) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved.suffix.lower() not in suffixes:
            continue
        if not resolved.is_file():
            raise FileNotFoundError(f"Referenced conditioning file does not exist: {resolved}")
        if resolved not in seen:
            seen.add(resolved)
            output.append(resolved)
    return output


def resolve_conditioning_path(folder: Path, value: str) -> Path:
    """Resolve metadata paths, including legacy paths whose directories were flattened."""
    path = Path(value)
    candidates = [path if path.is_absolute() else folder / path, folder / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def conditioning_media(row: dict, video: Path, repository_root: Path) -> tuple[list[Path], list[Path]]:
    folder = video.parent
    metadata_path = folder / "metadata.json"
    images: list[Path] = []
    videos: list[Path] = []
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        images.extend(resolve_conditioning_path(folder, value) for value in metadata.get("images", []))
        videos.extend(resolve_conditioning_path(folder, value) for value in metadata.get("videos", []))
    else:
        for value in row.get("ReferenceImages", []):
            path = Path(value)
            images.append(path if path.is_absolute() else repository_root / path)
        if not images:
            images.extend(sorted(folder.glob("reference_image_*")))
    return unique_existing(images, IMAGE_SUFFIXES), unique_existing(videos, VIDEO_SUFFIXES)


def asset_name(source: Path, repository_root: Path) -> str:
    try:
        identity = source.relative_to(repository_root).as_posix()
    except ValueError:
        identity = str(source)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20] + source.suffix.lower()


def copy_asset(source: Path, destination: Path, subdirectory: str, repository_root: Path) -> str:
    relative = Path(subdirectory) / asset_name(source, repository_root)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return relative.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Legacy dataset.json")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Replace destination/all.json")
    args = parser.parse_args()

    source = args.source.resolve()
    repository_root = args.repository_root.resolve()
    destination = args.destination.resolve()
    all_json = destination / "all.json"
    if all_json.exists() and not args.force:
        raise FileExistsError(f"Refusing to replace {all_json}; pass --force to rebuild")

    rows = json.loads(source.read_text(encoding="utf-8"))
    output: list[dict] = []
    for row in rows:
        source_video = resolve_legacy_path(repository_root, row["Video"])
        reference_images, reference_videos = conditioning_media(row, source_video, repository_root)
        feedback = row.get("Feedback", {})
        normalized = {
            "id": str(row["ID"]),
            "prompt": row.get("Prompt", ""),
            "reference_images": [
                copy_asset(path, destination, "reference_images", repository_root)
                for path in reference_images
            ],
            "reference_videos": [
                copy_asset(path, destination, "reference_videos", repository_root)
                for path in reference_videos
            ],
            "video": copy_asset(source_video, destination, "videos", repository_root),
            "label": row.get("Label"),
            "feedback": {
                "abnormality": feedback.get("abnormality", ""),
                "prompt_following": feedback.get("prompt_following", ""),
                "consistency": feedback.get("consistency", ""),
            },
        }
        output.append(normalized)

    if len({row["id"] for row in output}) != len(output):
        raise ValueError("The source dataset contains duplicate IDs")
    destination.mkdir(parents=True, exist_ok=True)
    all_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Imported {len(output)} rows, "
        f"{len({row['video'] for row in output})} generated videos, "
        f"{len({path for row in output for path in row['reference_images']})} reference images, "
        f"{len({path for row in output for path in row['reference_videos']})} reference videos"
    )


if __name__ == "__main__":
    main()
