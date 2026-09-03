from __future__ import annotations

import math
from collections.abc import Iterable


KEYS = ("TP", "TN", "FP", "FN")


def aggregate(matrices: Iterable[dict[str, float]]) -> dict[str, float]:
    total = {key: 0.0 for key in KEYS}
    for matrix in matrices:
        for key in KEYS:
            total[key] += float(matrix.get(key, 0.0))
    return total


def calculate(matrix: dict[str, float]) -> dict[str, float]:
    tp, tn, fp, fn = (matrix[k] for k in KEYS)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    accuracy = (tp + tn) / sum(matrix.values()) if sum(matrix.values()) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "recall_minus_fpr": recall - fpr,
        "f1": f1,
        "mcc": mcc,
        "accuracy": accuracy,
    }

