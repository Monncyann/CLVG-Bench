from __future__ import annotations

import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .candidates import PromptCandidate, PromptTable
from .client import ModelClient
from .config import Config
from .evaluator import Evaluator, ItemEvaluationError, _text
from .io import load_json, resolve_inside, save_json
from .prompts import OPTIMIZER_PROMPT


PROTOCOL_VERSION = 3


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_stage_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not normalized:
        raise ValueError("Evaluation stage name is empty")
    return normalized


def _merge_usage(
    target: dict[str, dict[str, int]],
    added: dict[str, Any],
) -> None:
    for role in target:
        values = added.get(role, {})
        target[role]["input_tokens"] += int(values.get("input_tokens", 0))
        target[role]["output_tokens"] += int(values.get("output_tokens", 0))


class Trainer:
    """Legacy-compatible AVE system-prompt optimization."""

    def __init__(self, config: Config, root: Path, client: ModelClient | None = None):
        self.config = config
        self.root = root
        self.client = client or ModelClient(config)
        self.evaluator = Evaluator(config, root, self.client)
        self.output = root / config.output_dir
        self.output.mkdir(parents=True, exist_ok=True)
        self.rollouts = 0
        self.table = PromptTable(
            seed=config.seed,
            dominance_margin=config.dominance_margin,
            pareto_top_ratio=config.pareto_top_ratio,
        )
        self.run_fingerprint: str | None = None

    def load_split(self, name: str) -> list[dict[str, Any]]:
        rows = load_json(self.root / self.config.data_dir / f"{name}.json")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{name}.json must contain a non-empty list")
        return rows

    def cost(self) -> float:
        # The historical runner applies the task/judge rate to every model call
        # when deciding whether the experiment has exhausted its budget.
        return (
            self.evaluator.input_tokens
            * self.config.budget_input_cost_per_million
            / 1_000_000
            + self.evaluator.output_tokens
            * self.config.budget_output_cost_per_million
            / 1_000_000
        )

    def _validate_split(self, name: str, rows: list[dict[str, Any]]) -> set[str]:
        data_root = self.root / self.config.data_dir
        ids: set[str] = set()
        videos: set[str] = set()
        for index, item in enumerate(rows):
            if not isinstance(item, dict):
                raise ValueError(f"{name}[{index}] must be an object")
            item_id = str(item.get("id", "")).strip()
            if not item_id:
                raise ValueError(f"{name}[{index}] has no id")
            if item_id in ids:
                raise ValueError(f"{name}.json contains duplicate id: {item_id}")
            ids.add(item_id)
            video = item.get("video")
            if not isinstance(video, str) or not video:
                raise ValueError(f"{name}[{index}] has no video path")
            video_path = resolve_inside(data_root, video)
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing generated video: {video_path}")
            videos.add(str(video_path))
            feedback = item.get("feedback")
            if not isinstance(feedback, dict) or not isinstance(
                feedback.get(self.config.dimension), str
            ):
                raise ValueError(
                    f"{name}[{index}].feedback.{self.config.dimension} must be a string"
                )
            for field in ("reference_images", "reference_videos"):
                references = item.get(field, [])
                if not isinstance(references, list) or not all(
                    isinstance(value, str) and value for value in references
                ):
                    raise ValueError(f"{name}[{index}].{field} must be a list of paths")
                for relative in references:
                    reference_path = resolve_inside(data_root, relative)
                    if not reference_path.is_file():
                        raise FileNotFoundError(
                            f"Missing reference media: {reference_path}"
                        )
        return videos

    def validate_data(
        self,
        train: list[dict[str, Any]],
        validation: list[dict[str, Any]],
        test: list[dict[str, Any]],
    ) -> None:
        split_videos = {
            "train": self._validate_split("train", train),
            "val": self._validate_split("val", validation),
            "test": self._validate_split("test", test),
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = split_videos[left] & split_videos[right]
            if overlap:
                raise ValueError(
                    f"Generated-video leakage between {left} and {right}: "
                    f"{len(overlap)} shared path(s)"
                )
        positive = sum(
            bool(item["feedback"][self.config.dimension].strip())
            for item in validation
        )
        if positive == 0 or positive == len(validation):
            raise ValueError(
                "val.json must contain both positive and negative examples for "
                f"the {self.config.dimension} metric (positive={positive}, "
                f"negative={len(validation) - positive})"
            )

    def _ensure_run_identity(self, initial_prompt: str) -> None:
        identity = {
            "protocol_version": PROTOCOL_VERSION,
            "config": asdict(self.config),
            "initial_prompt_sha256": hashlib.sha256(
                initial_prompt.encode("utf-8")
            ).hexdigest(),
        }
        fingerprint = _fingerprint(identity)
        metadata_path = self.output / "run_metadata.json"
        if metadata_path.exists():
            previous = load_json(metadata_path)
            if previous.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    f"{self.output} belongs to a different configuration. "
                    "Use a new output_dir to avoid mixing baseline results."
                )
        else:
            known_outputs = (
                "candidate_table.json",
                "history.json",
                "run_summary.json",
                "best_prompt.json",
            )
            if any((self.output / name).exists() for name in known_outputs):
                raise RuntimeError(
                    f"{self.output} contains an older run without run_metadata.json. "
                    "Use a new output_dir so the old results are not overwritten."
                )
            save_json(metadata_path, {"fingerprint": fingerprint, **identity})
        self.run_fingerprint = fingerprint

    def _evaluation_cache_path(self, stage_name: str) -> Path:
        return self.output / "checkpoints" / f"{_safe_stage_name(stage_name)}.json"

    def evaluate(
        self,
        data: list[dict[str, Any]],
        prompt: str,
        stage_name: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate rows concurrently, preserving row order and incremental cache state."""
        cache_path = self._evaluation_cache_path(stage_name) if stage_name else None
        fingerprint = _fingerprint(
            {
                "run": self.run_fingerprint or asdict(self.config),
                "prompt": prompt,
                "data": data,
            }
        )
        entries: dict[str, dict[str, Any]] = {}
        failures: dict[str, str] = {}
        stage_usage = self.evaluator.empty_usage()
        if cache_path and cache_path.exists():
            cached = load_json(cache_path)
            if cached.get("fingerprint") == fingerprint:
                entries = dict(cached.get("entries", {}))
                failures = dict(cached.get("failures", {}))
                cached_usage = cached.get("usage")
                if isinstance(cached_usage, dict):
                    _merge_usage(stage_usage, cached_usage)
                else:
                    # Backward compatibility with checkpoints written before
                    # stage-level usage was introduced.
                    for entry in entries.values():
                        _merge_usage(stage_usage, entry.get("usage", {}))
                self.evaluator.add_usage(stage_usage)

        keys = [f"{index}:{item['id']}" for index, item in enumerate(data)]
        pending = [
            (index, key, item)
            for index, (key, item) in enumerate(zip(keys, data))
            if key not in entries
        ]

        def save_cache() -> None:
            if cache_path:
                save_json(
                    cache_path,
                    {
                        "fingerprint": fingerprint,
                        "stage": stage_name,
                        "entries": entries,
                        "failures": failures,
                        "usage": stage_usage,
                    },
                )

        progress = tqdm(
            total=len(data),
            initial=len(data) - len(pending),
            desc=stage_name or "evaluate",
            unit="item",
            dynamic_ncols=True,
        )
        try:
            if pending:
                workers = min(self.config.max_workers, len(pending))
                with ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="ave-eval"
                ) as pool:
                    future_to_entry = {
                        pool.submit(
                            self.evaluator.evaluate_item_with_usage, item, prompt
                        ): (key, item)
                        for _index, key, item in pending
                    }
                    for future in as_completed(future_to_entry):
                        key, item = future_to_entry[future]
                        try:
                            result, usage = future.result()
                            entries[key] = {"result": result, "usage": usage}
                            _merge_usage(stage_usage, usage)
                            failures.pop(key, None)
                        except Exception as error:
                            if isinstance(error, ItemEvaluationError):
                                _merge_usage(stage_usage, error.usage)
                            failures[key] = f"{type(error).__name__}: {error}"
                            tqdm.write(
                                f"AVE evaluation failed for {item['id']}: {failures[key]}"
                            )
                        save_cache()
                        progress.update(1)
        finally:
            progress.close()

        self.rollouts += len(data)
        ordered_results = [entries[key]["result"] for key in keys if key in entries]
        if not ordered_results:
            raise RuntimeError(
                f"All items failed during evaluation stage {stage_name or 'evaluate'}"
            )
        result = self.evaluator.summarize(ordered_results)
        result["requested_items"] = len(data)
        result["successful_items"] = len(ordered_results)
        result["failures"] = [
            {"id": data[index]["id"], "error": failures[key]}
            for index, key in enumerate(keys)
            if key in failures and key not in entries
        ]
        return result

    @staticmethod
    def instance_scores(evaluation: dict[str, Any]) -> dict[str, float]:
        return {
            str(item["id"]): float(item["metrics"]["accuracy"])
            for item in evaluation["items"]
        }

    def sample_batch(
        self, train: list[dict[str, Any]], iteration: int
    ) -> list[dict[str, Any]]:
        size = min(self.config.batch_size, len(train))
        return random.Random(self.config.seed + iteration).sample(train, size)

    @staticmethod
    def _matching_weakness(result: dict[str, Any]) -> str:
        reasoning = result.get("match_reasoning", {})
        missed = (
            reasoning.get("missed_points", []) if isinstance(reasoning, dict) else []
        )
        hallucinated = (
            reasoning.get("hallucinated_points", [])
            if isinstance(reasoning, dict)
            else []
        )
        missed_text = (
            "it failed to capture the following reference points: " + "; ".join(missed)
            if missed
            else ""
        )
        hallucinated_text = (
            "it generated redundant/incorrect claims " + "; ".join(hallucinated)
            if hallucinated
            else ""
        )
        if missed_text and hallucinated_text:
            return f"{missed_text}, and {hallucinated_text}"
        if missed_text or hallucinated_text:
            return missed_text or hallucinated_text
        matrix = result.get("confusion_matrix", {})
        if float(matrix.get("FP", 0)) > 0:
            return "it generated redundant/incorrect claims " + str(
                result.get("prediction", {}).get("weakness", [])
            )
        if float(matrix.get("FN", 0)) > 0:
            return "it failed to capture the following reference points: " + str(
                result.get("reference", "")
            )
        return "no obvious weaknesses found"

    def revise(
        self,
        prompt: str,
        batch_result: dict[str, Any],
        batch: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any], dict[str, dict[str, int]]]:
        result_by_id = {str(result["id"]): result for result in batch_result["items"]}
        successful_batch = [item for item in batch if str(item["id"]) in result_by_id]
        if not successful_batch:
            raise RuntimeError(
                "Cannot optimize a prompt because every mini-batch item failed"
            )

        trace: dict[str, Any] = {}
        for item in successful_batch:
            result = result_by_id[str(item["id"])]
            trace[str(item["id"])] = {
                "inference_trace": {
                    "input_prompt": item.get("prompt", ""),
                    "model_evaluation_reasoning": result["prediction"].get(
                        "reasoning", ""
                    ),
                    "model_evaluation": result["prediction"].get("weakness", []),
                    "reference_weakness_of_the_video": result["reference"],
                    "model_eval_weakness": self._matching_weakness(result),
                }
            }

        final_text = _text(
            f"This is the System Prompt: {prompt}; This is the inference trace: {trace}"
        )
        payload_limit = round(self.config.optimizer_max_payload_mb * 1024 * 1024)
        fixed_size = len(json.dumps([final_text], ensure_ascii=False).encode("utf-8"))
        remaining = payload_limit - fixed_size - 4096
        if remaining < len(successful_batch):
            raise ValueError(
                "Optimizer text trace alone exceeds optimizer_max_payload_mb"
            )
        per_item_budget = remaining // len(successful_batch)
        content: list[dict[str, Any]] = []
        for item in successful_batch:
            content.extend(
                self.evaluator.optimizer_content_for_item(item, per_item_budget)
            )
        content.append(final_text)
        payload_size = len(json.dumps(content, ensure_ascii=False).encode("utf-8"))
        if payload_size > payload_limit:
            raise ValueError(
                f"Optimizer payload is {payload_size / 1024 / 1024:.2f} MiB; "
                f"limit is {self.config.optimizer_max_payload_mb:.2f} MiB"
            )

        usage = self.evaluator.empty_usage()
        last_error: Exception | None = None
        reminder: dict[str, Any] | None = None
        for _attempt in range(self.config.max_retries):
            request_content = content + ([reminder] if reminder else [])
            try:
                model_result = self.client.json_completion(
                    self.config.optimizer_model,
                    OPTIMIZER_PROMPT,
                    request_content,
                )
                response = self.evaluator.record(model_result, "optimizer")
                usage["optimizer"]["input_tokens"] += model_result.input_tokens
                usage["optimizer"]["output_tokens"] += model_result.output_tokens
                revised = response.get("revised_prompt")
                if not isinstance(revised, str) or not revised.strip():
                    reminder = _text("Remember to return a non-empty revised_prompt.")
                    raise ValueError("optimizer did not return revised_prompt")
                required = (
                    "Input Format:",
                    "Output Format:",
                    "reasoning",
                    "judgement",
                    "weakness",
                )
                missing = [value for value in required if value not in revised]
                if missing:
                    reminder = _text(
                        "Remember to preserve these required strings in revised_prompt: "
                        + ", ".join(missing)
                    )
                    raise ValueError(
                        "revised_prompt is missing required format content: "
                        + ", ".join(missing)
                    )
                return revised.strip(), response.get("reasoning", {}), usage
            except Exception as error:
                last_error = error
        raise RuntimeError("Prompt optimizer failed after all retries") from last_error

    def _proposal(
        self,
        iteration: int,
        parent: PromptCandidate,
        before: dict[str, Any],
        batch: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        path = self.output / f"prompt_{iteration:03d}_proposal.json"
        fingerprint = _fingerprint(
            {
                "run": self.run_fingerprint,
                "parent_prompt": parent.prompt,
                "batch": batch,
                "before": before,
                "optimizer_prompt": OPTIMIZER_PROMPT,
            }
        )
        if path.exists():
            cached = load_json(path)
            if cached.get("fingerprint") == fingerprint:
                self.evaluator.add_usage(cached.get("optimizer_usage", {}))
                return cached["System Prompt"], cached.get("optimizer_reasoning", {})

        revised, reasoning, usage = self.revise(parent.prompt, before, batch)
        save_json(
            path,
            {
                "System Prompt": revised,
                "parent_step": parent.step,
                "fingerprint": fingerprint,
                "optimizer_reasoning": reasoning,
                "optimizer_usage": usage,
            },
        )
        return revised, reasoning

    def add_candidate(
        self,
        *,
        step: int,
        prompt: str,
        validation: dict[str, Any],
        parent_step: int | None,
    ) -> PromptCandidate:
        prompt_file = f"prompt_{step:03d}.json"
        save_json(self.output / prompt_file, {"System Prompt": prompt})
        candidate = PromptCandidate(
            step=step,
            prompt=prompt,
            prompt_file=prompt_file,
            validation_score=float(validation["metrics"][self.config.metric]),
            instance_scores=self.instance_scores(validation),
            parent_step=parent_step,
        )
        self.table.add(candidate)
        save_json(self.output / "candidate_table.json", self.table.metadata())
        best = self.table.top(1)[0]
        save_json(
            self.output / "best_prompt.json",
            {
                "System Prompt": best.prompt,
                "step": best.step,
                "score": best.validation_score,
            },
        )
        return candidate

    def _save_state(
        self,
        *,
        iteration: int,
        optimizer_steps: int,
        validation_count: int,
        no_improvement_iterations: int,
        parent: PromptCandidate,
    ) -> None:
        save_json(
            self.output / "training_state.json",
            {
                "iteration": iteration,
                "optimizer_steps": optimizer_steps,
                "validation_count": validation_count,
                "no_improvement_iterations": no_improvement_iterations,
                "parent_step": parent.step,
                "rollouts": self.rollouts,
                "budget_cost_usd": self.cost(),
                "resume": "Rerun the same command; completed stage caches are reused.",
            },
        )

    def run(self) -> None:
        train = self.load_split("train")
        validation = self.load_split("val")
        test = self.load_split("test")
        self.validate_data(train, validation, test)
        prompt_payload = load_json(self.root / self.config.initial_prompt)
        initial_prompt = prompt_payload["System Prompt"]
        if not isinstance(initial_prompt, str) or not initial_prompt.strip():
            raise ValueError("Initial prompt must contain a non-empty System Prompt")
        self._ensure_run_identity(initial_prompt)

        # Historical protocol: test before optimization, then initial validation.
        initial_test = self.evaluate(test, initial_prompt, "000_initial_test")
        save_json(self.output / "initial_test_results.json", initial_test)
        initial_validation = self.evaluate(validation, initial_prompt, "000_validation")
        initial = self.add_candidate(
            step=0,
            prompt=initial_prompt,
            validation=initial_validation,
            parent_step=None,
        )
        history: list[dict[str, Any]] = [
            {
                "step": 0,
                "iteration": 0,
                "optimizer_step": 0,
                "candidate_step": initial.step,
                "initial_test": initial_test,
                "validation": initial_validation,
                "budget_cost_usd": self.cost(),
            }
        ]
        save_json(self.output / "history.json", history)

        parent = initial
        iteration = 0
        optimizer_steps = 0
        validation_count = 0
        validation_accumulation = 0
        no_improvement_iterations = 0
        periodic_test_count = 0
        stopped_reason = "max_steps"
        self._save_state(
            iteration=iteration,
            optimizer_steps=optimizer_steps,
            validation_count=validation_count,
            no_improvement_iterations=no_improvement_iterations,
            parent=parent,
        )

        while True:
            if self.cost() >= self.config.budget_usd:
                stopped_reason = "optimization_budget"
                break
            if self.rollouts >= self.config.max_rollouts:
                stopped_reason = "optimization_rollouts"
                break
            if optimizer_steps >= self.config.max_steps:
                stopped_reason = "max_steps"
                break

            iteration += 1
            batch = self.sample_batch(train, iteration)
            before = self.evaluate(
                batch, parent.prompt, f"{iteration:03d}_before_update"
            )
            before_score = float(before["metrics"][self.config.metric])
            if self.config.skip_perfect_score and before_score >= 1.0:
                history.append(
                    {
                        "step": iteration,
                        "iteration": iteration,
                        "optimizer_step": optimizer_steps,
                        "parent_step": parent.step,
                        "before": before,
                        "skipped_perfect_score": True,
                        "budget_cost_usd": self.cost(),
                    }
                )
                save_json(self.output / "history.json", history)
                self._save_state(
                    iteration=iteration,
                    optimizer_steps=optimizer_steps,
                    validation_count=validation_count,
                    no_improvement_iterations=no_improvement_iterations,
                    parent=parent,
                )
                continue

            proposed_prompt, optimizer_reasoning = self._proposal(
                iteration, parent, before, batch
            )
            optimizer_steps += 1
            after = (
                self.evaluate(batch, proposed_prompt, f"{iteration:03d}_after_update")
                if self.config.gepa
                else None
            )
            after_score = (
                float(after["metrics"][self.config.metric])
                if after is not None
                else None
            )
            mini_batch_accepted = (
                after is None
                or after_score > before_score
                or (self.config.equal_as_greater and after_score == before_score)
            )
            forced_validation = False
            should_accumulate = mini_batch_accepted
            if not mini_batch_accepted:
                no_improvement_iterations += 1
                if (
                    no_improvement_iterations
                    >= self.config.max_iterations_without_improvement
                ):
                    should_accumulate = True
                    forced_validation = True

            record: dict[str, Any] = {
                "step": iteration,
                "iteration": iteration,
                "optimizer_step": optimizer_steps,
                "parent_step": parent.step,
                "parent_validation_score": parent.validation_score,
                "before": before,
                "after": after,
                "mini_batch_accepted": mini_batch_accepted,
                "forced_validation": forced_validation,
                "optimizer_reasoning": optimizer_reasoning,
            }

            candidate: PromptCandidate | None = None
            if should_accumulate:
                no_improvement_iterations = 0
                validation_accumulation += 1
                if validation_accumulation >= self.config.validation_accumulation_steps:
                    validation_accumulation = 0
                    candidate_validation = self.evaluate(
                        validation,
                        proposed_prompt,
                        f"{iteration:03d}_validation",
                    )
                    candidate = self.add_candidate(
                        step=iteration,
                        prompt=proposed_prompt,
                        validation=candidate_validation,
                        parent_step=parent.step,
                    )
                    record["candidate_step"] = candidate.step
                    record["validation"] = candidate_validation
                    parent = self.table.select(
                        self.config.prompt_selection,
                        local_seed=iteration,
                    )
                    validation_count += 1
                    if validation_count % self.config.test_every_validations == 0:
                        periodic_test_count += 1
                        periodic = self.evaluate(
                            test,
                            proposed_prompt,
                            f"{iteration:03d}_periodic_test",
                        )
                        periodic_path = self.output / f"test_step_{iteration:03d}.json"
                        save_json(periodic_path, periodic)
                        record["periodic_test"] = {
                            "file": periodic_path.name,
                            "metrics": periodic["metrics"],
                        }
                else:
                    parent = self.table.latest()
            else:
                parent = self.table.latest()

            record["accepted"] = candidate is not None
            record["next_parent_step"] = parent.step
            record["budget_cost_usd"] = self.cost()
            history.append(record)
            save_json(self.output / "history.json", history)
            self._save_state(
                iteration=iteration,
                optimizer_steps=optimizer_steps,
                validation_count=validation_count,
                no_improvement_iterations=no_improvement_iterations,
                parent=parent,
            )

        optimization_cost = self.cost()
        optimization_rollouts = self.rollouts
        best = self.table.top(1)[0]
        final = self.evaluate(test, best.prompt, "final_test_best_1")
        save_json(self.output / "test_results.json", final)
        save_json(
            self.output / "run_summary.json",
            {
                "algorithm": self.config.algorithm,
                "prompt_selection": self.config.prompt_selection,
                "metric": self.config.metric,
                "seed": self.config.seed,
                "best_step": best.step,
                "best_validation_score": best.validation_score,
                "initial_test_metrics": initial_test["metrics"],
                "test_metrics": final["metrics"],
                "candidate_count": len(self.table.candidates),
                "iterations": iteration,
                "optimizer_steps": optimizer_steps,
                "validation_count": validation_count,
                "periodic_test_count": periodic_test_count,
                "stopped_reason": stopped_reason,
                "optimization_budget_cost_usd": optimization_cost,
                "total_budget_cost_usd": self.cost(),
                "total_role_estimated_cost_usd": self.evaluator.estimated_cost(),
                "optimization_rollouts": optimization_rollouts,
                "total_rollouts": self.rollouts,
                "usage": self.evaluator.usage_summary(),
            },
        )
