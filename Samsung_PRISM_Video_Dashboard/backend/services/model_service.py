"""Model integration layer — the ONE place the real detector plugs in.

`predict_video(video_path)` is the single, clean entry point the rest of
the dashboard calls. It wraps the existing Video Detection module's
`VideoDetector` (imported from the sibling repo via the path in
config.yaml — the same sys.path pattern the fusion engine uses).

If the checkpoint can't be loaded, it falls back to a clearly-marked
MOCK MODEL so the UI still runs — but mock results are always flagged
`"mock": True` and NEVER presented as real model output (requirement 5).

================================================================
TO SWAP IN A DIFFERENT MODEL:
    Replace the body of `_RealModel.predict()` (or point
    config.yaml:model.video_module_path at a different module).
    The dashboard only depends on the dict returned by
    `predict_video()` — nothing else.
================================================================
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

from backend.config_loader import checkpoint_path, get_config, video_module_root

logger = logging.getLogger(__name__)

# Serialises access to the single shared VideoDetector. The model is one
# in-memory instance and its GradCAM step temporarily relocates it across
# devices (MPS <-> CPU); two concurrent predict() calls (e.g. the Compare
# page firing both videos in parallel) would then race and raise
# "Input type (MPSFloatType) and weight type (torch.FloatTensor) should be
# the same". Inference is the bottleneck anyway, so queuing is free.
_predict_lock = threading.Lock()

# Module-level singletons so we load the checkpoint once, not per request.
_real_model: "_RealModel | None" = None
_mock_active: bool = False
_load_error: str | None = None


# =====================================================================
# Real model wrapper
# =====================================================================
class _RealModel:
    """Thin wrapper around the existing VideoDetector."""

    def __init__(self) -> None:
        module_root = video_module_root()
        if str(module_root) not in sys.path:
            sys.path.insert(0, str(module_root))

        # Imported lazily so a missing video module degrades to mock mode
        # rather than crashing the whole backend at import time.
        from src.inference import VideoDetector  # type: ignore

        cfg = get_config()["model"]
        ckpt = checkpoint_path()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        self._detector = VideoDetector(
            checkpoint_path=ckpt,
            threshold=float(cfg["decision_threshold"]),
            frames_per_video=int(cfg["frames_sampled"]),
            image_size=int(cfg["input_size"]),
            device=cfg.get("device"),
        )
        self._produce_explanation = bool(cfg.get("produce_explanation", True))
        logger.info("Real model loaded from %s", ckpt)

    def predict(self, video_path: str | Path) -> dict[str, Any]:
        # Hold the lock for the whole inference so a concurrent request
        # can't relocate the shared model's device mid-forward.
        with _predict_lock:
            t0 = time.perf_counter()
            result = self._detector.predict(
                video_path, produce_explanation=self._produce_explanation
            )
            processing_time = time.perf_counter() - t0
        return _detection_result_to_payload(result, processing_time, mock=False)


# =====================================================================
# Mock model (clearly marked, never dressed up as real)
# =====================================================================
class _MockModel:
    """Deterministic placeholder so the UI runs without a checkpoint."""

    def predict(self, video_path: str | Path) -> dict[str, Any]:
        import hashlib

        # Deterministic pseudo-score from the filename so the same file
        # always yields the same mock result (useful for UI testing).
        h = int(hashlib.sha256(str(video_path).encode()).hexdigest(), 16)
        prob = (h % 1000) / 1000.0
        threshold = float(get_config()["model"]["decision_threshold"])
        n = int(get_config()["model"]["frames_sampled"])
        per_frame = [round(min(1.0, max(0.0, prob + ((i % 7) - 3) * 0.03)), 4) for i in range(n)]
        return _scores_to_payload(
            per_frame_scores=per_frame,
            prob_synthetic=prob,
            threshold=threshold,
            processing_time=0.01,
            explainability_score=None,
            heatmap_dir=None,
            timeline_path=None,
            meta={"note": "MOCK MODEL — inference model not connected"},
            mock=True,
        )


# =====================================================================
# DetectionResult -> dashboard payload
# =====================================================================
def _detection_result_to_payload(result: Any, processing_time: float, mock: bool) -> dict[str, Any]:
    """Map a real DetectionResult dataclass to the dashboard's dict schema."""
    return _scores_to_payload(
        per_frame_scores=list(result.per_frame_scores),
        prob_synthetic=float(result.prob_synthetic),
        threshold=float(result.threshold),
        processing_time=processing_time,
        explainability_score=result.explainability_score,
        heatmap_dir=result.heatmap_dir,
        timeline_path=result.timeline_path,
        meta=dict(result.meta or {}),
        mock=mock,
        num_frames_used=int(result.num_frames_used),
    )


def _scores_to_payload(
    per_frame_scores: list[float],
    prob_synthetic: float,
    threshold: float,
    processing_time: float,
    explainability_score: float | None,
    heatmap_dir: str | None,
    timeline_path: str | None,
    meta: dict[str, Any],
    mock: bool,
    num_frames_used: int | None = None,
) -> dict[str, Any]:
    """Build the canonical dashboard result dict from raw scores.

    Everything here is derived from real model outputs (per-frame scores,
    video-level prob, threshold) — nothing is invented.
    """
    prediction = "Synthetic" if prob_synthetic >= threshold else "Real"
    max_margin = max(threshold, 1.0 - threshold)
    confidence = min(abs(prob_synthetic - threshold) / max_margin, 1.0) if max_margin else 0.0
    n = num_frames_used if num_frames_used is not None else len(per_frame_scores)

    # Per-frame breakdown (timestamps are approximate — the model samples
    # uniformly and does not store true timestamps; the frontend labels
    # these with "≈"). Frame timestamps get filled in by video_service
    # once it knows the clip duration.
    frame_predictions = [
        {
            "frame_index": i,
            "synthetic_probability": round(float(s), 4),
            "real_probability": round(1.0 - float(s), 4),
            "predicted_class": "Synthetic" if s >= threshold else "Real",
            "confidence": round(min(abs(float(s) - threshold) / max_margin, 1.0) if max_margin else 0.0, 4),
            "suspicious": bool(s >= threshold),
        }
        for i, s in enumerate(per_frame_scores)
    ]

    temporal = _temporal_stats(per_frame_scores, threshold)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "real_probability": round(1.0 - prob_synthetic, 4),
        "synthetic_probability": round(prob_synthetic, 4),
        "threshold": threshold,
        "num_frames_used": n,
        "frame_predictions": frame_predictions,
        "temporal": temporal,
        "processing_time": round(processing_time, 4),
        "explainability_score": explainability_score,
        "heatmap_dir": heatmap_dir,
        "timeline_path": timeline_path,
        "meta": meta,
        "mock": mock,
    }


def _temporal_stats(scores: list[float], threshold: float) -> dict[str, Any]:
    """Temporal consistency stats — computed from real per-frame scores."""
    if not scores:
        return {
            "avg_synthetic": None, "max_synthetic": None, "min_synthetic": None,
            "variance": None, "fluctuations": 0, "suspicious_frames": 0,
            "suspicious_pct": 0.0, "total_frames": 0,
        }
    n = len(scores)
    avg = sum(scores) / n
    var = sum((s - avg) ** 2 for s in scores) / n
    suspicious = sum(1 for s in scores if s >= threshold)
    # Fluctuations = number of times the per-frame call flips real<->synthetic.
    flips = sum(
        1 for a, b in zip(scores, scores[1:])
        if (a >= threshold) != (b >= threshold)
    )
    return {
        "avg_synthetic": round(avg, 4),
        "max_synthetic": round(max(scores), 4),
        "min_synthetic": round(min(scores), 4),
        "variance": round(var, 6),
        "fluctuations": flips,
        "suspicious_frames": suspicious,
        "suspicious_pct": round(100.0 * suspicious / n, 2),
        "total_frames": n,
    }


# =====================================================================
# Public API
# =====================================================================
def init_model() -> None:
    """Load the real model once at startup; fall back to mock on failure."""
    global _real_model, _mock_active, _load_error
    try:
        _real_model = _RealModel()
        _mock_active = False
        _load_error = None
    except Exception as exc:  # noqa: BLE001 - degrade to mock, never crash boot
        _real_model = None
        _mock_active = True
        _load_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Real model unavailable, using MOCK model. Reason: %s", _load_error)


def model_status() -> dict[str, Any]:
    """What the UI needs to show the MOCK/REAL banner + model info panel."""
    cfg = get_config()["model"]
    if _real_model is None and not _mock_active:
        init_model()
    return {
        "mock": _mock_active,
        "connected": not _mock_active,
        "load_error": _load_error,
        "name": cfg["name"],
        "architecture": cfg["architecture"],
        "checkpoint": str(checkpoint_path()),
        "version": cfg["version"],
        "device": cfg.get("device") or "auto (mps > cuda > cpu)",
        "input_size": cfg["input_size"],
        "frame_sampling_strategy": cfg["frame_sampling_strategy"],
        "frames_sampled": cfg["frames_sampled"],
        "decision_threshold": cfg["decision_threshold"],
    }


def predict_video(video_path: str | Path) -> dict[str, Any]:
    """Run detection on one video. Returns the canonical dashboard dict.

    This is the single integration seam. Everything downstream (routes,
    evaluation, DB, UI) depends only on the shape of this return value.
    """
    if _real_model is None and not _mock_active:
        init_model()
    model = _real_model if _real_model is not None else _MockModel()
    return model.predict(video_path)
