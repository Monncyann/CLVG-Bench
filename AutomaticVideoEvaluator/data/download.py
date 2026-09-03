#!/usr/bin/env python3
"""Download and verify the AVE dataset from Hugging Face Hub."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASKS = ("abnormality", "perception", "prompt_following")
DEFAULT_REPO_ID = "JianhuiWei/AVE_data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_task(data_dir: Path, task: str) -> None:
    task_dir = data_dir / task
    manifest_path = task_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for split, metadata in manifest["splits"].items():
        path = task_dir / f"{split}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split: {path}")
        if sha256(path) != metadata["sha256"]:
            raise ValueError(f"Split checksum mismatch: {path}")

    for item in manifest["files"]:
        path = (task_dir / item["path"]).resolve()
        if task_dir.resolve() not in path.parents:
            raise ValueError(f"Manifest path escapes task directory: {item['path']}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing media: {path}")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Media checksum mismatch: {path}")

    all_path = task_dir / "all.json"
    plan_path = task_dir / "SPLIT_PLAN.json"
    if not all_path.is_file() or not plan_path.is_file():
        raise FileNotFoundError(f"Missing all.json or SPLIT_PLAN.json in {task_dir}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if sha256(all_path) != plan["source_sha256"]:
        raise ValueError(f"all.json checksum mismatch: {all_path}")
    for split, expected in plan["expected_sha256"].items():
        if sha256(task_dir / f"{split}.json") != expected:
            raise ValueError(f"Split-plan checksum mismatch: {task_dir / f'{split}.json'}")

    print(
        f"Verified {task}: {manifest['unique_videos']} generated videos, "
        f"{len(manifest['files'])} media files"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument("--revision", default="main", help="Branch, tag, or commit hash")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Download destination; defaults to the directory containing this script",
    )
    parser.add_argument("--task", choices=TASKS, help="Download only one task")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    destination = args.local_dir.resolve()
    selected_tasks = (args.task,) if args.task else TASKS
    if not args.verify_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise SystemExit(
                "huggingface_hub is required; run `python -m pip install -r requirements.txt`"
            ) from error

        allow_patterns = None
        if args.task:
            allow_patterns = [f"{args.task}/**", "README.md"]
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=destination,
            allow_patterns=allow_patterns,
            max_workers=args.max_workers,
        )
        print(f"Downloaded {args.repo_id}@{args.revision} to {destination}")

    if not args.skip_verification:
        for task in selected_tasks:
            verify_task(destination, task)


if __name__ == "__main__":
    main()
