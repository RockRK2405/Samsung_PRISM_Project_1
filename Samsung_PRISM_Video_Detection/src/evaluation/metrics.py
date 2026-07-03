"""Video-level evaluation metrics.

Positive class is ``1`` (synthetic). Everything is reported from the
detector's point of view — "how well do we catch fakes."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClassificationReport:
    """Aggregate metrics for a run."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    auc: float
    n_real: int
    n_synthetic: int
    per_manipulation_recall: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a plain-dict snapshot suitable for JSON serialisation."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
            "auc": self.auc,
            "n_real": self.n_real,
            "n_synthetic": self.n_synthetic,
            "per_manipulation_recall": self.per_manipulation_recall,
        }


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, tn, fn)`` for binary classification with positive class ``1``."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC via the Mann-Whitney U formulation — no sklearn dependency."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.0
    # Rank all scores; ties get average rank.
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    # Break ties by averaging.
    _, inv, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    tie_avg = np.zeros(len(counts), dtype=np.float64)
    for i in range(len(counts)):
        idxs = np.where(inv == i)[0]
        tie_avg[i] = ranks[idxs].mean()
    ranks = tie_avg[inv]
    rank_sum_pos = ranks[: len(pos)].sum()
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def compute_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    manipulations: list[str] | None = None,
    threshold: float = 0.5,
) -> ClassificationReport:
    """Compute the full metrics report from video-level scores.

    Args:
        y_true: Shape ``(N,)`` — ground-truth binary labels (0 real, 1 synthetic).
        y_score: Shape ``(N,)`` — model's synthetic-probability per video.
        manipulations: Optional length-N list of manipulation tags. Used
            only to compute per-manipulation recall on the fake subset.
        threshold: Decision threshold applied to ``y_score``.

    Returns:
        A populated :class:`ClassificationReport`.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tp, fp, tn, fn = _binary_counts(y_true, y_pred)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    auc = _roc_auc(y_true, y_score)

    per_manip: dict[str, float] = {}
    if manipulations is not None:
        manips = np.asarray(manipulations)
        for name in np.unique(manips):
            mask = manips == name
            if y_true[mask].size == 0:
                continue
            # For fakes (label==1) recall = TP / (TP + FN) within the subset.
            # For reals (label==0) we report specificity instead, under the
            # same key, since recall on all-negatives is undefined.
            subset_true = y_true[mask]
            subset_pred = y_pred[mask]
            if (subset_true == 1).all():
                per_manip[str(name)] = _safe_div(
                    int(((subset_pred == 1) & (subset_true == 1)).sum()),
                    int((subset_true == 1).sum()),
                )
            elif (subset_true == 0).all():
                per_manip[str(name)] = _safe_div(
                    int(((subset_pred == 0) & (subset_true == 0)).sum()),
                    int((subset_true == 0).sum()),
                )

    return ClassificationReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        auc=auc,
        n_real=int((y_true == 0).sum()),
        n_synthetic=int((y_true == 1).sum()),
        per_manipulation_recall=per_manip,
    )
