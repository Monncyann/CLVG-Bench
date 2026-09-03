#!/usr/bin/env python3
"""Create, balance, or exactly reconstruct AVE dataset splits."""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ave.io import load_json, save_json


SPLIT_NAMES = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_positive(row: dict[str, Any]) -> bool:
    return str(row["label"]) != "0"


def allocate(size: int, ratios: tuple[float, float, float]) -> list[int]:
    """Allocate a total by normalized ratios using the largest-remainder rule."""
    total = sum(ratios)
    exact = [size * ratio / total for ratio in ratios]
    counts = [int(value) for value in exact]
    order = sorted(range(3), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[: size - sum(counts)]:
        counts[index] += 1
    return counts


def validate_unique(rows: list[dict[str, Any]]) -> None:
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("all.json contains duplicate IDs")
    if len({row["video"] for row in rows}) != len(rows):
        raise ValueError("all.json contains duplicate generated-video paths")


def stratified_split(
    rows: list[dict[str, Any]], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Create a deterministic split while preserving the binary label ratio."""
    groups: dict[bool, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: str(item["id"])):
        groups[is_positive(row)].append(row)

    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for label, group in sorted(groups.items()):
        random.Random(f"{seed}:label:{int(label)}").shuffle(group)
        counts = allocate(len(group), ratios)
        start = 0
        for name, count in zip(SPLIT_NAMES, counts):
            splits[name].extend(copy.deepcopy(group[start : start + count]))
            start += count
    for name in SPLIT_NAMES:
        random.Random(f"{seed}:split:{name}").shuffle(splits[name])
    return splits


def resample(
    rows: list[dict[str, Any]], target_count: int, seed: str
) -> list[dict[str, Any]]:
    """Resize one class and add unique IDs to oversampled metadata aliases."""
    if target_count and not rows:
        raise ValueError("Cannot resample an empty class to a non-zero size")
    rng = random.Random(seed)
    if len(rows) >= target_count:
        return copy.deepcopy(rows if len(rows) == target_count else rng.sample(rows, target_count))

    result = copy.deepcopy(rows)
    for index, item in enumerate(rng.choices(rows, k=target_count - len(rows))):
        alias = copy.deepcopy(item)
        alias["id"] = f"{alias['id']}_aug_{index}"
        result.append(alias)
    return result


def balanced_split(
    rows: list[dict[str, Any]],
    ratios: tuple[float, float, float],
    seed: int,
    total_size: int,
    positive_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    """Split unique records first, then balance each split independently."""
    pools = stratified_split(rows, ratios, seed)
    target_sizes = dict(zip(SPLIT_NAMES, allocate(total_size, ratios)))
    output: dict[str, list[dict[str, Any]]] = {}
    for name in SPLIT_NAMES:
        positives = [row for row in pools[name] if is_positive(row)]
        negatives = [row for row in pools[name] if not is_positive(row)]
        target_positive = int(target_sizes[name] * positive_ratio)
        target_negative = target_sizes[name] - target_positive
        combined = (
            resample(negatives, target_negative, f"{seed}:{name}:negative")
            + resample(positives, target_positive, f"{seed}:{name}:positive")
        )
        random.Random(f"{seed}:{name}:balanced").shuffle(combined)
        output[name] = combined
    return output


def released_split(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct the checked-in split exactly from all.json and its plan."""
    all_path = data_dir / "all.json"
    plan = load_json(data_dir / "SPLIT_PLAN.json")
    if sha256(all_path) != plan["source_sha256"]:
        raise ValueError("all.json SHA-256 does not match SPLIT_PLAN.json")
    rows = load_json(all_path)
    validate_unique(rows)
    by_id = {str(row["id"]): row for row in rows}
    splits: dict[str, list[dict[str, Any]]] = {}
    for name in SPLIT_NAMES:
        rebuilt = []
        for entry in plan["splits"][name]:
            if entry["source_id"] not in by_id:
                raise KeyError(f"Unknown source_id in split plan: {entry['source_id']}")
            row = copy.deepcopy(by_id[entry["source_id"]])
            row["id"] = entry["id"]
            rebuilt.append(row)
        splits[name] = rebuilt
    return splits


def materialize_media(
    data_dir: Path,
    output_dir: Path,
    splits: dict[str, list[dict[str, Any]]],
    mode: str,
) -> int:
    """Make a generated split directly runnable without duplicating by default."""
    if mode == "none":
        return 0
    referenced: set[str] = set()
    for rows in splits.values():
        for row in rows:
            referenced.add(row["video"])
            referenced.update(row.get("reference_images", []))
            referenced.update(row.get("reference_videos", []))

    source_root = data_dir.resolve()
    output_root = output_dir.resolve()
    for relative in sorted(referenced):
        if Path(relative).is_absolute():
            raise ValueError(f"Media path must be relative: {relative}")
        source = (source_root / relative).resolve()
        target = output_root / relative
        if source_root not in source.parents:
            raise ValueError(f"Media path escapes source task directory: {relative}")
        if not source.is_file():
            raise FileNotFoundError(f"Missing source media: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if os.path.samefile(source, target) or sha256(source) == sha256(target):
                continue
            raise FileExistsError(f"Refusing to replace different media file: {target}")
        if mode == "hardlink":
            try:
                os.link(source, target)
            except OSError as error:
                raise OSError(
                    f"Cannot hard-link {source} to {target}; use --media-mode copy "
                    "when source and output are on different filesystems"
                ) from error
        elif mode == "copy":
            shutil.copy2(source, target)
        else:
            raise ValueError(f"Unknown media mode: {mode}")
    return len(referenced)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Task directory relative to project root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=("released", "stratified", "balanced"),
        default="released",
    )
    parser.add_argument("--input", default="all.json", help="Input filename inside data-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument("--total-size", type=int)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    parser.add_argument(
        "--media-mode",
        choices=("hardlink", "copy", "none"),
        default="hardlink",
        help="Materialize referenced media in output-dir (default: hardlink, no extra file data)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if any(ratio <= 0 for ratio in args.ratios):
        raise ValueError("All split ratios must be positive")
    if not 0.0 <= args.positive_ratio <= 1.0:
        raise ValueError("positive-ratio must be between 0 and 1")
    if args.strategy == "balanced" and (args.total_size is None or args.total_size <= 0):
        raise ValueError("balanced strategy requires a positive --total-size")

    project_root = Path(__file__).resolve().parents[1]
    data_dir = (project_root / args.data_dir).resolve()
    output_dir = args.output_dir.resolve()
    targets = [output_dir / f"{name}.json" for name in SPLIT_NAMES]
    if any(path.exists() for path in targets) and not args.force:
        raise FileExistsError("Refusing to replace split files; choose another output-dir or pass --force")

    if args.strategy == "released":
        splits = released_split(data_dir)
    else:
        source = data_dir / args.input
        rows = load_json(source)
        validate_unique(rows)
        ratios = tuple(args.ratios)
        if args.strategy == "stratified":
            splits = stratified_split(rows, ratios, args.seed)
        else:
            splits = balanced_split(
                rows, ratios, args.seed, args.total_size, args.positive_ratio
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}
    for name in SPLIT_NAMES:
        target = output_dir / f"{name}.json"
        save_json(target, splits[name])
        output_hashes[name] = sha256(target)
        positives = sum(is_positive(row) for row in splits[name])
        print(
            f"{name}: {len(splits[name])} rows "
            f"({positives} positive, {len(splits[name]) - positives} negative)"
        )

    media_count = materialize_media(data_dir, output_dir, splits, args.media_mode)
    if media_count:
        print(f"{args.media_mode}: materialized {media_count} referenced media files")

    if args.strategy == "released":
        expected = load_json(data_dir / "SPLIT_PLAN.json")["expected_sha256"]
        if output_hashes != expected:
            raise ValueError("Released split reconstruction checksum mismatch")
        print("SHA-256 verified against the released split")
    else:
        source = data_dir / args.input
        save_json(output_dir / "SPLIT_INFO.json", {
            "format_version": 1,
            "official_split": False,
            "source": args.input,
            "source_sha256": sha256(source),
            "strategy": args.strategy,
            "seed": args.seed,
            "ratios": dict(zip(SPLIT_NAMES, args.ratios)),
            "total_size": args.total_size,
            "positive_ratio": args.positive_ratio if args.strategy == "balanced" else None,
            "output_sha256": output_hashes,
        })


if __name__ == "__main__":
    main()
