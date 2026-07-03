"""Unit tests for the metrics module."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import compute_report


def test_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.9, 0.8])
    r = compute_report(y_true, y_score)
    assert r.accuracy == 1.0
    assert r.precision == 1.0
    assert r.recall == 1.0
    assert r.f1 == 1.0
    assert r.fpr == 0.0
    assert r.auc == 1.0


def test_all_wrong_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.9, 0.8, 0.1, 0.2])
    r = compute_report(y_true, y_score)
    assert r.accuracy == 0.0
    assert r.recall == 0.0
    assert r.fpr == 1.0
    assert r.auc == 0.0


def test_per_manipulation_breakdown() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_score = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    manips = ["Deepfakes", "Deepfakes", "FaceSwap", "FaceSwap", "original", "original"]
    r = compute_report(y_true, y_score, manipulations=manips)
    assert r.per_manipulation_recall["Deepfakes"] == 1.0
    assert r.per_manipulation_recall["FaceSwap"] == 0.0
    assert r.per_manipulation_recall["original"] == 1.0     # specificity for reals


def test_counts_reflect_class_balance() -> None:
    y_true = np.array([0, 0, 0, 1, 1])
    r = compute_report(y_true, np.array([0.0, 0.0, 0.0, 1.0, 1.0]))
    assert r.n_real == 3
    assert r.n_synthetic == 2
