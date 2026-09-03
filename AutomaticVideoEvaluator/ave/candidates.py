from __future__ import annotations

import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PromptCandidate:
    step: int
    prompt: str
    prompt_file: str
    validation_score: float
    instance_scores: dict[str, float]
    parent_step: int | None

    def metadata(self) -> dict:
        payload = asdict(self)
        payload.pop("prompt")
        return payload


class PromptTable:
    """Validation-backed prompt table used by TextGrad and GEPA selection."""

    def __init__(
        self,
        *,
        seed: int,
        dominance_margin: float = 0.0,
        pareto_top_ratio: float = 1.0,
    ) -> None:
        self.seed = seed
        self.dominance_margin = dominance_margin
        self.pareto_top_ratio = pareto_top_ratio
        self.candidates: list[PromptCandidate] = []

    def add(self, candidate: PromptCandidate) -> None:
        if any(existing.step == candidate.step for existing in self.candidates):
            raise ValueError(f"Duplicate candidate step: {candidate.step}")
        # The legacy runner keeps a candidate when only part of a validation
        # batch succeeds. Pareto comparisons therefore use the common IDs.
        self.candidates.append(candidate)

    def best(self) -> PromptCandidate:
        if not self.candidates:
            raise ValueError("The prompt table is empty")
        # max() preserves the first candidate on a tie, matching the legacy manager.
        return max(self.candidates, key=lambda candidate: candidate.validation_score)

    def latest(self) -> PromptCandidate:
        if not self.candidates:
            raise ValueError("The prompt table is empty")
        return max(self.candidates, key=lambda candidate: candidate.step)

    @staticmethod
    def _legacy_filename(candidate: PromptCandidate) -> str:
        """Return the filename used by the historical ParamsManager."""
        return f"Params_{candidate.step}.json"

    def _dominates(self, better: PromptCandidate, worse: PromptCandidate) -> bool:
        common = sorted(set(better.instance_scores) & set(worse.instance_scores))
        if not common:
            return False
        differences = [
            better.instance_scores[item_id] - worse.instance_scores[item_id]
            for item_id in common
        ]
        return all(
            difference >= -self.dominance_margin for difference in differences
        ) and any(difference > self.dominance_margin for difference in differences)

    def pareto_front(self) -> list[PromptCandidate]:
        front = [
            candidate
            for candidate in self.candidates
            if not any(
                other is not candidate and self._dominates(other, candidate)
                for other in self.candidates
            )
        ]
        # The historical implementation sorted unpadded Params_N.json paths
        # lexicographically before seeded Pareto sampling. Preserve that order
        # even though release prompt files use zero-padded names.
        return sorted(front, key=self._legacy_filename)

    def top(self, k: int) -> list[PromptCandidate]:
        """Return validation candidates using the historical final-test tie break."""
        if k < 1:
            raise ValueError("k must be positive")
        return sorted(
            self.candidates,
            key=lambda candidate: (
                -candidate.validation_score,
                self._legacy_filename(candidate),
            ),
        )[:k]

    @staticmethod
    def _weights(scores: list[float]) -> list[float]:
        minimum = min(scores)
        if minimum < 0:
            shifted = [score - minimum for score in scores]
            floor = 0.1 * sum(shifted)
            weights = [score + floor for score in shifted]
        else:
            weights = scores[:]
        return weights if sum(weights) > 0 else [1.0] * len(scores)

    def select(self, strategy: str, *, local_seed: int) -> PromptCandidate:
        if strategy == "get_best":
            return self.best()
        if strategy != "pareto_sampling":
            raise ValueError(f"Unknown prompt-selection strategy: {strategy}")

        front = self.pareto_front()
        if not front:
            return self.best()
        keep = int(len(front) * self.pareto_top_ratio)
        if keep > 0:
            # Python's stable sort retains the legacy filename order on score
            # ties, matching the old ParamsManager implementation.
            front = sorted(
                front,
                key=lambda candidate: -candidate.validation_score,
            )[:keep]
            front.sort(key=self._legacy_filename)
        weights = self._weights([candidate.validation_score for candidate in front])
        return random.Random(self.seed + local_seed).choices(
            front, weights=weights, k=1
        )[0]

    def metadata(self) -> dict:
        return {
            "candidates": [candidate.metadata() for candidate in self.candidates],
            "best_step": self.top(1)[0].step,
            "get_best_parent_step": self.best().step,
            "pareto_steps": [candidate.step for candidate in self.pareto_front()],
        }
