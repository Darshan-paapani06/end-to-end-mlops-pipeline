from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    threshold: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_binary_classifier(y_true, y_probability, threshold: float = 0.5) -> ClassificationMetrics:
    """Evaluate a binary classifier using stakeholder-friendly metrics."""
    y_probability = np.asarray(y_probability)
    y_pred = (y_probability >= threshold).astype(int)
    return ClassificationMetrics(
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        roc_auc=round(float(roc_auc_score(y_true, y_probability)), 4),
        threshold=threshold,
    )


def find_recall_focused_threshold(y_true, y_probability, minimum_precision: float = 0.35) -> float:
    """Select a threshold that prioritizes recall while keeping precision above a business floor."""
    best_threshold = 0.5
    best_recall = -1.0
    for threshold in np.arange(0.20, 0.81, 0.01):
        metrics = evaluate_binary_classifier(y_true, y_probability, float(threshold))
        if metrics.precision >= minimum_precision and metrics.recall > best_recall:
            best_recall = metrics.recall
            best_threshold = float(threshold)
    return round(best_threshold, 2)
