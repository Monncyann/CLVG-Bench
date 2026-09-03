#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ave.config import ALGORITHMS, load_config
from ave.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an Adaptive Video Evaluator prompt."
    )
    parser.add_argument("--config", default="configs/abnormality.json")
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS))
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config, root = load_config(args.config)
    if args.algorithm:
        config.algorithm = args.algorithm
        if not args.output_dir:
            task_name = Path(config.data_dir).name
            config.output_dir = f"outputs/{task_name}_{args.algorithm}"
    if args.output_dir:
        config.output_dir = args.output_dir
    config.validate()
    Trainer(config, root).run()


if __name__ == "__main__":
    main()
