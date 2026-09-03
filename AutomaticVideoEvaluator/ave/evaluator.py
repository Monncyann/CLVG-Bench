from __future__ import annotations

import math
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from .client import ModelClient
from .config import Config
from .io import resolve_inside
from .metrics import aggregate, calculate
from .prompts import MATCHER_PROMPT
from .video import image_data_url, sample_video


def _text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


class ItemEvaluationError(RuntimeError):
    """Preserve model usage when an item exhausts its evaluation retries."""

    def __init__(
        self,
        error: Exception,
        usage: dict[str, dict[str, int]],
    ) -> None:
        super().__init__(f"{type(error).__name__}: {error}")
        self.usage = usage


class Evaluator:
    def __init__(
        self, config: Config, project_root: Path, client: ModelClient | None = None
    ):
        self.config = config
        self.root = project_root
        self.client = client or ModelClient(config)
        self.usage = {
            role: {"input_tokens": 0, "output_tokens": 0}
            for role in ("judge", "matcher", "optimizer")
        }
        self._usage_lock = threading.Lock()
        self._local = threading.local()

    @staticmethod
    def empty_usage() -> dict[str, dict[str, int]]:
        return {
            role: {"input_tokens": 0, "output_tokens": 0}
            for role in ("judge", "matcher", "optimizer")
        }

    @property
    def input_tokens(self) -> int:
        return sum(values["input_tokens"] for values in self.usage_snapshot().values())

    @property
    def output_tokens(self) -> int:
        return sum(values["output_tokens"] for values in self.usage_snapshot().values())

    def record(self, result, role: str) -> dict[str, Any]:
        if role not in self.usage:
            raise ValueError(f"Unknown model role: {role}")
        with self._usage_lock:
            self.usage[role]["input_tokens"] += result.input_tokens
            self.usage[role]["output_tokens"] += result.output_tokens
        local_usage = getattr(self._local, "usage", None)
        if local_usage is not None:
            local_usage[role]["input_tokens"] += result.input_tokens
            local_usage[role]["output_tokens"] += result.output_tokens
        return result.json

    def add_usage(self, usage: dict[str, Any]) -> None:
        """Restore token usage associated with a cached model result."""
        with self._usage_lock:
            for role in self.usage:
                values = usage.get(role, {})
                self.usage[role]["input_tokens"] += int(values.get("input_tokens", 0))
                self.usage[role]["output_tokens"] += int(values.get("output_tokens", 0))

    def usage_snapshot(self) -> dict[str, dict[str, int]]:
        with self._usage_lock:
            return {role: values.copy() for role, values in self.usage.items()}

    def usage_summary(self) -> dict[str, Any]:
        roles = self.usage_snapshot()
        return {
            "by_role": roles,
            "total": {
                "input_tokens": sum(value["input_tokens"] for value in roles.values()),
                "output_tokens": sum(
                    value["output_tokens"] for value in roles.values()
                ),
            },
            "estimated_cost_usd": self.estimated_cost(),
        }

    def estimated_cost(self) -> float:
        total = 0.0
        for role, values in self.usage_snapshot().items():
            input_rate = getattr(self.config, f"{role}_input_cost_per_million")
            output_rate = getattr(self.config, f"{role}_output_cost_per_million")
            total += values["input_tokens"] * input_rate / 1_000_000
            total += values["output_tokens"] * output_rate / 1_000_000
        return total

    def content_for_item(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Build the legacy-ordered visual input for the judge."""
        data_root = self.root / self.config.data_dir
        video_path = resolve_inside(data_root, item["video"])
        content: list[dict[str, Any]] = []

        reference_images = item.get("reference_images", [])
        if not reference_images:
            content.append(_text("The user's input reference image is empty."))
        for index, relative in enumerate(reference_images):
            content.append(_text(f"This is the user's input reference image {index}: "))
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url(resolve_inside(data_root, relative))
                    },
                }
            )

        reference_videos = item.get("reference_videos", [])
        for video_index, relative in enumerate(reference_videos, start=1):
            reference_path = resolve_inside(data_root, relative)
            content.append(
                _text(f"This is the user's input reference video {video_index}: ")
            )
            for frame in sample_video(
                reference_path, self.config.video_sampling_fps, self.config.max_frames
            ):
                content.append(_text(f"[{frame['timestamp']:.1f} second]"))
                content.append(
                    {"type": "image_url", "image_url": {"url": frame["data_url"]}}
                )

        content.append(_text("This is the user's input video: "))
        for frame in sample_video(
            video_path, self.config.video_sampling_fps, self.config.max_frames
        ):
            content.append(_text(f"[{frame['timestamp']:.1f} second]"))
            content.append(
                {"type": "image_url", "image_url": {"url": frame["data_url"]}}
            )
        content.append(
            _text(f'This is the user\'s input text: " {item.get("prompt", "")}"')
        )
        return content

    def optimizer_content_for_item(
        self,
        item: dict[str, Any],
        max_encoded_bytes: int,
    ) -> list[dict[str, Any]]:
        """Build legacy-style optimizer evidence within a per-item media budget."""
        if max_encoded_bytes < 1:
            raise ValueError("optimizer media budget must be positive")
        data_root = self.root / self.config.data_dir
        reference_images = item.get("reference_images", [])
        reference_videos = item.get("reference_videos", [])
        reserve = min(64 * 1024, max_encoded_bytes // 20)
        available = max(1, max_encoded_bytes - reserve)

        image_share = int(available * 0.20) if reference_images else 0
        reference_video_share = int(available * 0.20) if reference_videos else 0
        generated_share = max(1, available - image_share - reference_video_share)
        content: list[dict[str, Any]] = [
            _text(f"This is the video for the datapoint {item['id']}:")
        ]
        for frame in sample_video(
            resolve_inside(data_root, item["video"]),
            self.config.video_sampling_fps,
            self.config.max_frames,
            max_encoded_bytes=generated_share,
        ):
            content.append(_text(f"[{frame['timestamp']:.1f} second]"))
            content.append(
                {"type": "image_url", "image_url": {"url": frame["data_url"]}}
            )

        if reference_images:
            content.append(
                _text(f"These are the reference images for the datapoint {item['id']}:")
            )
            per_image = max(1, image_share // len(reference_images))
            for relative in reference_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url(
                                resolve_inside(data_root, relative),
                                max_encoded_bytes=per_image,
                            )
                        },
                    }
                )

        if reference_videos:
            content.append(
                _text(f"These are the reference videos for the datapoint {item['id']}:")
            )
            per_video = max(1, reference_video_share // len(reference_videos))
            for relative in reference_videos:
                for frame in sample_video(
                    resolve_inside(data_root, relative),
                    self.config.video_sampling_fps,
                    self.config.max_frames,
                    max_encoded_bytes=per_video,
                ):
                    content.append(_text(f"[{frame['timestamp']:.1f} second]"))
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": frame["data_url"]},
                        }
                    )
        return content

    def judge(self, item: dict[str, Any], system_prompt: str) -> dict[str, Any]:
        content = self.content_for_item(item)
        outputs = []
        votes_needed = self.config.votes // 2 + 1
        for _ in range(self.config.votes):
            last_error: Exception | None = None
            for _attempt in range(self.config.max_retries):
                try:
                    result = self.record(
                        self.client.json_completion(
                            self.config.judge_model, system_prompt, content
                        ),
                        "judge",
                    )
                    judgement = str(result.get("judgement", "")).strip().lower()
                    weakness = result.get("weakness")
                    if "reasoning" not in result or judgement not in {"yes", "no"}:
                        raise ValueError(f"Invalid judge response: {result}")
                    if not isinstance(weakness, list) or not all(
                        isinstance(value, str) and value.strip() for value in weakness
                    ):
                        raise ValueError(f"Invalid weakness list: {result}")
                    if (judgement == "yes") != bool(weakness):
                        raise ValueError(
                            f"Inconsistent judgement and weakness list: {result}"
                        )
                    normalized = dict(result)
                    normalized["judgement"] = judgement
                    normalized["weakness"] = [value.strip() for value in weakness]
                    outputs.append(normalized)
                    break
                except Exception as error:
                    last_error = error
            else:
                raise RuntimeError("Judge failed after all retries") from last_error

            counter = Counter(output["judgement"] for output in outputs)
            if counter.most_common(1)[0][1] >= votes_needed:
                break

        winner = Counter(x["judgement"].strip().lower() for x in outputs).most_common(
            1
        )[0][0]
        selected = dict(next(x for x in outputs if x["judgement"] == winner))
        selected["voting_history"] = [x["judgement"] for x in outputs]
        return selected

    def match(
        self, predicted: list[str], reference: str
    ) -> tuple[dict[str, float], dict[str, Any]]:
        has_prediction = bool(predicted)
        has_reference = bool(reference.strip())
        if not self.config.semantic_matching or not (has_prediction and has_reference):
            matrix = {
                "TP": float(has_prediction and has_reference),
                "TN": float(not has_prediction and not has_reference),
                "FP": float(has_prediction and not has_reference),
                "FN": float(not has_prediction and has_reference),
            }
            return matrix, {"mode": "rough_or_edge_case"}
        content = [
            _text(f"This is the Model Evaluations: {predicted}"),
            _text(f"This is the Reference: {reference}"),
        ]
        last_error: Exception | None = None
        for _attempt in range(self.config.max_retries):
            try:
                result = self.record(
                    self.client.json_completion(
                        self.config.matcher_model, MATCHER_PROMPT, content
                    ),
                    "matcher",
                )
                counts = result.get("matrix_counts")
                if not isinstance(counts, dict):
                    raise ValueError("matrix_counts must be an object")
                matrix = {key: float(counts[key]) for key in ("TP", "TN", "FP", "FN")}
                if any(
                    not math.isfinite(value) or value < 0 for value in matrix.values()
                ):
                    raise ValueError("confusion counts must be finite and non-negative")
                if matrix["TN"] != 0:
                    raise ValueError(
                        "TN must be zero when both weakness sets are non-empty"
                    )
                total = sum(matrix.values())
                if total <= 0:
                    raise ValueError("semantic matcher returned an empty matrix")
                reasoning = result.get("reasoning", {})
                if not isinstance(reasoning, dict):
                    raise ValueError("matcher reasoning must be an object")
                if not isinstance(reasoning.get("overall_reasoning"), str):
                    raise ValueError(
                        "matcher reasoning.overall_reasoning must be a string"
                    )
                for key in (
                    "matched_reference_points",
                    "extra_valid_points",
                    "missed_points",
                    "hallucinated_points",
                ):
                    values = reasoning.get(key)
                    if not isinstance(values, list) or not all(
                        isinstance(value, str) for value in values
                    ):
                        raise ValueError(
                            f"matcher reasoning.{key} must be a list of strings"
                        )
                return {key: value / total for key, value in matrix.items()}, reasoning
            except Exception as error:
                last_error = error
        raise RuntimeError("Semantic matcher failed after all retries") from last_error

    def evaluate_item(self, item: dict[str, Any], system_prompt: str) -> dict[str, Any]:
        judged = self.judge(item, system_prompt)
        reference = item["feedback"][self.config.dimension]
        matrix, match_reasoning = self.match(judged["weakness"], reference)
        return {
            "id": item["id"],
            "prediction": judged,
            "reference": reference,
            "match_reasoning": match_reasoning,
            "confusion_matrix": matrix,
            "metrics": calculate(matrix),
        }

    def evaluate_item_with_usage(
        self,
        item: dict[str, Any],
        system_prompt: str,
    ) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
        """Evaluate one item and return the per-item usage needed for resume."""
        previous = getattr(self._local, "usage", None)
        local_usage = self.empty_usage()
        self._local.usage = local_usage
        try:
            return self.evaluate_item(item, system_prompt), local_usage
        except Exception as error:
            raise ItemEvaluationError(error, local_usage) from error
        finally:
            if previous is None:
                del self._local.usage
            else:
                self._local.usage = previous

    def summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        matrix = aggregate(x["confusion_matrix"] for x in results)
        return {
            "confusion_matrix": matrix,
            "metrics": calculate(matrix),
            "items": results,
        }
