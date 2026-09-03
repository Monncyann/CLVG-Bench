#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ave.config import ALGORITHMS, load_config
from ave.evaluator import Evaluator
from ave.io import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen AVE prompt on one split."
    )
    parser.add_argument("--config", default="configs/abnormality.json")
    parser.add_argument(
        "--prompt",
        help="Prompt JSON to evaluate (default: initial_prompt from the config)",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS))
    args = parser.parse_args()
    config, root = load_config(args.config)
    if args.algorithm:
        config.algorithm = args.algorithm
        config.validate()
    prompt_path = args.prompt or config.initial_prompt
    prompt = load_json(root / prompt_path)["System Prompt"]
    data = load_json(root / config.data_dir / f"{args.split}.json")
    evaluator = Evaluator(config, root)
    ordered: list[dict | None] = [None] * len(data)
    failures = []
    workers = min(config.max_workers, len(data))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ave-eval") as pool:
        future_to_index = {
            pool.submit(evaluator.evaluate_item, item, prompt): index
            for index, item in enumerate(data)
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(data),
            desc=f"evaluate-{args.split}",
            unit="item",
            dynamic_ncols=True,
        ):
            index = future_to_index[future]
            try:
                ordered[index] = future.result()
            except Exception as error:
                failures.append(
                    {
                        "id": data[index].get("id"),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    successful = [item for item in ordered if item is not None]
    if not successful:
        raise RuntimeError("All evaluation items failed")
    result = evaluator.summarize(successful)
    result["requested_items"] = len(data)
    result["successful_items"] = len(successful)
    result["failures"] = failures
    result["usage"] = evaluator.usage_summary()
    save_json(root / args.output, result)
    print(result["metrics"])


if __name__ == "__main__":
    main()
