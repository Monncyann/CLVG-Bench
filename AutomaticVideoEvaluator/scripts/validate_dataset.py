#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ave.io import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AVE metadata and media files.")
    parser.add_argument("--data-dir", default="data/abnormality")
    parser.add_argument("--checksums", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = root / args.data_dir
    seen_ids: set[str] = set()
    seen_video_paths: set[str] = set()
    seen_reference_images: set[str] = set()
    seen_reference_videos: set[str] = set()
    split_rows: dict[str, list[dict]] = {}
    split_video_paths: dict[str, set[str]] = {}
    total = 0
    missing: list[str] = []
    for split in ("train", "val", "test"):
        rows = load_json(data_dir / f"{split}.json")
        split_rows[split] = rows
        split_video_paths[split] = set()
        positive = 0
        for row in rows:
            required = {"id", "prompt", "reference_images", "video", "label", "feedback"}
            allowed = required | {"reference_videos"}
            if not required.issubset(row) or not set(row).issubset(allowed):
                raise ValueError(f"{split}/{row.get('id')}: invalid keys")
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate ID across splits: {row['id']}")
            seen_ids.add(row["id"])
            split_video_paths[split].add(row["video"])
            media_paths = [(row["video"], seen_video_paths)]
            media_paths.extend((path, seen_reference_images) for path in row["reference_images"])
            media_paths.extend((path, seen_reference_videos) for path in row.get("reference_videos", []))
            for relative, collection in media_paths:
                media = (data_dir / relative).resolve()
                if data_dir.resolve() not in media.parents:
                    raise ValueError(f"{split}/{row['id']}: media path escapes task directory: {relative}")
                if not media.is_file():
                    missing.append(str(media))
                collection.add(relative)
            positive += bool(row["feedback"].get("abnormality", "").strip())
        total += len(rows)
        print(f"{split}: {len(rows)} rows ({positive} positive, {len(rows)-positive} negative)")
    for index, first in enumerate(("train", "val", "test")):
        for second in ("train", "val", "test")[index + 1 :]:
            overlap = split_video_paths[first] & split_video_paths[second]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise ValueError(
                    f"Generated-video leakage between {first} and {second}: {examples}"
                )
    if missing:
        raise FileNotFoundError("Missing media:\n" + "\n".join(missing))
    all_paths = seen_video_paths | seen_reference_images | seen_reference_videos
    print(
        f"OK: {total} rows, {len(seen_video_paths)} unique generated videos, "
        f"{len(seen_reference_images)} reference images, "
        f"{len(seen_reference_videos)} reference videos, "
        "no cross-split ID or generated-video overlap"
    )
    if args.checksums:
        manifest = load_json(data_dir / "MANIFEST.json")
        expected = {item["path"]: item["sha256"] for item in manifest["files"]}
        if set(expected) != all_paths:
            raise ValueError("MANIFEST.json file list does not match split references")
        for relative in sorted(all_paths):
            digest = hashlib.sha256()
            with (data_dir / relative).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected[relative]:
                raise ValueError(f"Checksum mismatch: {relative}")
        print(f"OK: verified SHA-256 for {len(all_paths)} media files")

    all_json = data_dir / "all.json"
    split_plan = data_dir / "SPLIT_PLAN.json"
    if all_json.is_file():
        all_rows = load_json(all_json)
        all_ids = [str(row["id"]) for row in all_rows]
        all_videos = [row["video"] for row in all_rows]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("all.json contains duplicate IDs")
        if len(set(all_videos)) != len(all_videos):
            raise ValueError("all.json contains duplicate generated-video paths")
        if set(all_videos) != seen_video_paths:
            raise ValueError("all.json generated videos do not match the split union")
        print(f"OK: all.json contains {len(all_rows)} unique records and videos")

        if split_plan.is_file():
            plan = load_json(split_plan)
            digest = hashlib.sha256(all_json.read_bytes()).hexdigest()
            if digest != plan["source_sha256"]:
                raise ValueError("SPLIT_PLAN.json source checksum does not match all.json")
            by_id = {str(row["id"]): row for row in all_rows}
            for split in ("train", "val", "test"):
                rebuilt = []
                for entry in plan["splits"][split]:
                    source_id = entry["source_id"]
                    if source_id not in by_id:
                        raise ValueError(f"{split}: unknown split-plan source ID {source_id}")
                    row = copy.deepcopy(by_id[source_id])
                    row["id"] = entry["id"]
                    rebuilt.append(row)
                if rebuilt != split_rows[split]:
                    raise ValueError(f"{split}: SPLIT_PLAN.json does not reconstruct checked-in rows")
                split_digest = hashlib.sha256((data_dir / f"{split}.json").read_bytes()).hexdigest()
                if split_digest != plan["expected_sha256"][split]:
                    raise ValueError(f"{split}: checksum does not match SPLIT_PLAN.json")
            print("OK: SPLIT_PLAN.json exactly reconstructs train/val/test")


if __name__ == "__main__":
    main()
