"""Model loader and cache for the PRISM web application.

Loads the trained stylometric model on startup and provides access to
the :class:`PrismEnsemble` for the API routes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ensemble_instance = None
_metrics_cache: dict[str, Any] | None = None


def get_ensemble():
    """Return the cached :class:`PrismEnsemble` singleton."""
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = _load_ensemble()
    return _ensemble_instance


def get_metrics() -> dict[str, Any]:
    """Return cached model metrics."""
    global _metrics_cache
    if _metrics_cache is None:
        _metrics_cache = _load_metrics()
    return _metrics_cache


def _find_model_dir() -> Path:
    """Search common artifact directories for a trained model."""
    project_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        project_root / "artifacts" / "enhanced",
        project_root / "artifacts" / "hc3_text_baseline",
        project_root / "artifacts" / "hc3_text_baseline_smoke",
    ]
    for candidate in candidates:
        if (candidate / "model.pkl").exists():
            return candidate

    # Fallback: no trained model — ensemble will still work with perplexity + patterns
    return candidates[0]


def _load_ensemble():
    """Load the PrismEnsemble from the best available model directory."""
    from prism_text_detector.ensemble import PrismEnsemble

    model_dir = _find_model_dir()
    if (model_dir / "model.pkl").exists():
        print(f"[PRISM] Loading model from: {model_dir}")
        return PrismEnsemble.load(model_dir)

    print("[PRISM] No trained model found — using perplexity + linguistic patterns only")
    return PrismEnsemble()


def _load_metrics() -> dict[str, Any]:
    """Load metrics.json from the model directory."""
    model_dir = _find_model_dir()
    metrics_path = model_dir / "metrics.json"

    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    # Return placeholder metrics when no training has been done
    return {
        "accuracy": 0.0,
        "f1_synthetic": 0.0,
        "roc_auc": 0.0,
        "false_positive_rate_real_flagged_synthetic": 0.0,
        "precision_synthetic": 0.0,
        "recall_synthetic": 0.0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 0,
        "mode": "no model trained yet",
        "model_info": {
            "version": "2.0.0",
            "status": "No trained model — train with: python -m prism_text_detector.train_baseline --data <path> --out artifacts/enhanced --mode enhanced",
        },
    }
