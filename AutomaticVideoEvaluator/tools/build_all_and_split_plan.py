#!/usr/bin/env python3
"""Build a unique all.json and an exact reconstruction plan from fixed splits."""
from __future__ import annotations

import argparse
import hashlib
import sys
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


def equivalent_except_id(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return {key: value for key, value in first.items() if key != "id"} == {
        key: value for key, value in second.items() if key != "id"
    }


def preferred_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    originals = [row for row in rows if "_aug_" not in str(row["id"])]
    candidate = originals[0] if originals else rows[0]
    if any(not equivalent_except_id(candidate, row) for row in rows):
        raise ValueError(f"Rows sharing {candidate['video']} differ in fields other than id")
    return candidate


def legacy_order(paths: list[Path]) -> list[str]:
    order: list[str] = []
    for path in paths:
        for row in load_json(path):
            identifier = row.get("id", row.get("ID"))
            if identifier is None:
                raise ValueError(f"No id/ID field in {path}")
            order.append(str(identifier))
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Task directory relative to project root")
    parser.add_argument(
        "--source-order",
        type=Path,
        nargs="*",
        default=[],
        help="Optional legacy JSON file(s) used only to restore canonical source order",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_dir = (project_root / args.data_dir).resolve()
    all_path = data_dir / "all.json"
    plan_path = data_dir / "SPLIT_PLAN.json"
    if not args.force and (all_path.exists() or plan_path.exists()):
        raise FileExistsError("Refusing to replace all.json/SPLIT_PLAN.json; pass --force")

    splits = {name: load_json(data_dir / f"{name}.json") for name in SPLIT_NAMES}
    by_video: dict[str, list[dict[str, Any]]] = {}
    for name in SPLIT_NAMES:
        for row in splits[name]:
            by_video.setdefault(row["video"], []).append(row)

    canonical_rows = [preferred_row(rows) for rows in by_video.values()]
    by_id = {str(row["id"]): row for row in canonical_rows}
    if len(by_id) != len(canonical_rows):
        raise ValueError("Canonical rows contain duplicate IDs")

    requested_order = legacy_order([path.resolve() for path in args.source_order])
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for identifier in requested_order:
        if identifier in by_id and identifier not in seen:
            ordered_ids.append(identifier)
            seen.add(identifier)
    for row in canonical_rows:
        identifier = str(row["id"])
        if identifier not in seen:
            ordered_ids.append(identifier)
            seen.add(identifier)
    all_rows = [by_id[identifier] for identifier in ordered_ids]
    save_json(all_path, all_rows)

    canonical_by_video = {row["video"]: str(row["id"]) for row in all_rows}
    plan_splits = {
        name: [
            {"id": str(row["id"]), "source_id": canonical_by_video[row["video"]]}
            for row in splits[name]
        ]
        for name in SPLIT_NAMES
    }
    save_json(plan_path, {
        "format_version": 1,
        "source": "all.json",
        "source_sha256": sha256(all_path),
        "purpose": "Exact reconstruction of the checked-in split, including within-split oversampled aliases",
        "splits": plan_splits,
        "expected_sha256": {
            name: sha256(data_dir / f"{name}.json") for name in SPLIT_NAMES
        },
    })
    aliases = sum(
        entry["id"] != entry["source_id"]
        for entries in plan_splits.values()
        for entry in entries
    )
    print(
        f"{data_dir.name}: {len(all_rows)} unique records; "
        f"{sum(len(rows) for rows in splits.values())} split rows; {aliases} aliases"
    )


if __name__ == "__main__":
    main()
