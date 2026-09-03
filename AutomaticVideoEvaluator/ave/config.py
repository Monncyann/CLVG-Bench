from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


ALGORITHMS = {
    "textgrad",
    "textgrad_semantic_loss",
    "gepa",
    "gepa_semantic_loss",
}
METRICS = {"mcc", "f1", "recall_minus_fpr", "accuracy"}
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


@dataclass
class Config:
    data_dir: str = "data"
    output_dir: str = "outputs/run"
    initial_prompt: str = "prompts/abnormality.json"
    dimension: str = "abnormality"
    judge_model: str = "doubao-seed-2-0-lite-260215"
    optimizer_model: str = "doubao-seed-2-0-pro-260215"
    matcher_model: str = "doubao-seed-2-0-pro-260215"
    algorithm: str = "gepa_semantic_loss"
    metric: str = "recall_minus_fpr"
    temperature: float = 0.0001
    max_tokens: int = 32768
    votes: int = 5
    video_sampling_fps: float = 2.0
    max_frames: int = 32
    batch_size: int = 3
    max_steps: int = 70
    max_rollouts: int = 4500
    budget_usd: float = 30.0
    budget_input_cost_per_million: float = 0.084
    budget_output_cost_per_million: float = 0.50
    judge_input_cost_per_million: float = 0.084
    judge_output_cost_per_million: float = 0.50
    optimizer_input_cost_per_million: float = 0.448
    optimizer_output_cost_per_million: float = 2.24
    matcher_input_cost_per_million: float = 0.448
    matcher_output_cost_per_million: float = 2.24
    dominance_margin: float = 0.0
    pareto_top_ratio: float = 1.0
    max_retries: int = 10
    seed: int = 47
    max_workers: int = 64
    request_timeout_seconds: float = 240.0
    optimizer_max_payload_mb: float = 45.0
    skip_perfect_score: bool = True
    equal_as_greater: bool = True
    max_iterations_without_improvement: int = 10
    validation_accumulation_steps: int = 1
    test_every_validations: int = 3

    @property
    def semantic_matching(self) -> bool:
        return self.algorithm.endswith("semantic_loss")

    @property
    def gepa(self) -> bool:
        return self.algorithm.startswith("gepa")

    @property
    def prompt_selection(self) -> str:
        return "pareto_sampling" if self.gepa else "get_best"

    def validate(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {', '.join(sorted(ALGORITHMS))}: {self.algorithm}"
            )
        if self.metric not in METRICS:
            raise ValueError(
                f"metric must be one of {', '.join(sorted(METRICS))}: {self.metric}"
            )
        if self.votes < 1 or self.votes % 2 == 0:
            raise ValueError("votes must be a positive odd integer")
        if self.video_sampling_fps <= 0:
            raise ValueError("video_sampling_fps must be positive")
        if self.max_frames < 1 or self.batch_size < 1:
            raise ValueError("max_frames and batch_size must be positive")
        if self.max_steps < 0 or self.max_rollouts < 1 or self.budget_usd <= 0:
            raise ValueError(
                "max_steps must be non-negative; max_rollouts and budget_usd must be positive"
            )
        if self.max_retries < 1:
            raise ValueError("max_retries must be positive")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        if self.request_timeout_seconds <= 0 or self.optimizer_max_payload_mb <= 0:
            raise ValueError(
                "request timeout and optimizer payload limit must be positive"
            )
        if self.max_iterations_without_improvement < 1:
            raise ValueError("max_iterations_without_improvement must be positive")
        if self.validation_accumulation_steps < 1 or self.test_every_validations < 1:
            raise ValueError("validation/test intervals must be positive")
        if self.dominance_margin < 0:
            raise ValueError("dominance_margin must be non-negative")
        if not 0 < self.pareto_top_ratio <= 1:
            raise ValueError("pareto_top_ratio must be in (0, 1]")
        costs = (
            self.budget_input_cost_per_million,
            self.budget_output_cost_per_million,
            self.judge_input_cost_per_million,
            self.judge_output_cost_per_million,
            self.optimizer_input_cost_per_million,
            self.optimizer_output_cost_per_million,
            self.matcher_input_cost_per_million,
            self.matcher_output_cost_per_million,
        )
        if any(cost < 0 for cost in costs):
            raise ValueError("token prices must be non-negative")


def load_config(path: str | Path) -> tuple[Config, Path]:
    config_path = Path(path).resolve()
    payload: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {f.name for f in fields(Config)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    config = Config(**payload)
    config.validate()
    root = config_path.parent.parent
    return config, root


def api_settings() -> tuple[str, str]:
    key = os.environ.get("AVE_API_KEY")
    if not key:
        raise RuntimeError(
            "AVE_API_KEY is not set. Copy .env.example or export it in your shell."
        )
    return key, os.environ.get("AVE_BASE_URL") or DEFAULT_ARK_BASE_URL
