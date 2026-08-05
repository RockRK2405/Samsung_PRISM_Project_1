"""Evaluation metrics + threshold re-testing. Pure functions over stored rows.

All metrics are computed from stored ground-truth labels and stored
model probabilities — no model reruns. Threshold re-testing (requirement
14) works entirely on stored `synthetic_probability`, so sweeping
thresholds is instant.
"""

from __future__ import annotations

from typing import Any

# Ground-truth / prediction label constants.
_REAL = "Real"
_SYNTH = "Synthetic"


def confusion_counts(rows: list[dict], threshold: float | None = None) -> dict[str, int]:
    """Count TP/TN/FP/FN over rows that have a ground-truth label.

    Positive class = Synthetic. If ``threshold`` is given, predictions are
    recomputed from each row's stored ``synthetic_probability``; otherwise
    the row's stored ``prediction`` is used as-is.
    """
    tp = tn = fp = fn = 0
    for r in rows:
        gt = r.get("ground_truth")
        if gt not in (_REAL, _SYNTH):
            continue  # skip Unknown / unlabeled
        pred = _predict_at(r, threshold)
        if pred is None:
            continue
        if gt == _SYNTH and pred == _SYNTH:
            tp += 1
        elif gt == _REAL and pred == _REAL:
            tn += 1
        elif gt == _REAL and pred == _SYNTH:
            fp += 1
        elif gt == _SYNTH and pred == _REAL:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics(rows: list[dict], threshold: float | None = None) -> dict[str, Any]:
    """Accuracy / precision / recall / F1 / FPR / FNR + confusion matrix."""
    c = confusion_counts(rows, threshold)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else None
    )
    fpr = fp / (fp + tn) if (fp + tn) else None
    fnr = fn / (fn + tp) if (fn + tp) else None

    return {
        "labeled_count": total,
        "accuracy": _pct(accuracy),
        "precision": _pct(precision),
        "recall": _pct(recall),
        "f1": _pct(f1),
        "false_positive_rate": _pct(fpr),
        "false_negative_rate": _pct(fnr),
        "confusion_matrix": c,
    }


def threshold_sweep(rows: list[dict], thresholds: list[float]) -> list[dict]:
    """Recompute metrics across candidate thresholds without model reruns."""
    sweep = []
    for t in thresholds:
        m = metrics(rows, threshold=t)
        sweep.append({
            "threshold": round(t, 4),
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "false_positive_rate": m["false_positive_rate"],
            "false_negative_rate": m["false_negative_rate"],
        })
    return sweep


def per_generator(rows: list[dict], threshold: float | None = None) -> list[dict]:
    """Detection rate per synthetic generator (Gemini/Veo/Sora/...)."""
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("ground_truth") != _SYNTH:
            continue
        gen = r.get("generator") or "Unknown"
        buckets.setdefault(gen, []).append(r)

    out = []
    for gen, grp in sorted(buckets.items()):
        detected = sum(1 for r in grp if _predict_at(r, threshold) == _SYNTH)
        out.append({
            "generator": gen,
            "videos_tested": len(grp),
            "correctly_detected": detected,
            "detection_rate": _pct(detected / len(grp) if grp else None),
        })
    return out


def score_distributions(rows: list[dict]) -> dict[str, list[float]]:
    """Synthetic-probability values split by ground truth, for histograms."""
    real_scores = [r["synthetic_probability"] for r in rows
                   if r.get("ground_truth") == _REAL and r.get("synthetic_probability") is not None]
    synth_scores = [r["synthetic_probability"] for r in rows
                    if r.get("ground_truth") == _SYNTH and r.get("synthetic_probability") is not None]
    return {"real": real_scores, "synthetic": synth_scores}


# ---------------------------------------------------------------------
def _predict_at(row: dict, threshold: float | None) -> str | None:
    """Prediction for a row, optionally recomputed at a new threshold."""
    if threshold is None:
        return row.get("prediction")
    prob = row.get("synthetic_probability")
    if prob is None:
        return None
    return _SYNTH if prob >= threshold else _REAL


def _pct(x: float | None) -> float | None:
    return round(100.0 * x, 2) if x is not None else None
