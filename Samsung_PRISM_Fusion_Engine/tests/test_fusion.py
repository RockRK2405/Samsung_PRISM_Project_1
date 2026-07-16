"""Unit tests for the weighted-aggregation + cross-modal-flag logic.

Pure logic tests -- constructs ModalityResult objects by hand, no
adapters or model weights involved.
"""

from __future__ import annotations

from fusion_engine.fusion import fuse
from fusion_engine.schema import ModalityResult

_WEIGHTS = {"audio": 0.32, "video": 0.30, "image": 0.20, "text": 0.18}


def _unavailable(modality: str) -> ModalityResult:
    return ModalityResult(modality=modality, prob_synthetic=0.0, confidence=0.0, available=False)


def test_all_four_confident_synthetic() -> None:
    results = {
        "image": ModalityResult(modality="image", prob_synthetic=0.9, confidence=0.9),
        "audio": ModalityResult(modality="audio", prob_synthetic=0.95, confidence=0.95),
        "text": ModalityResult(modality="text", prob_synthetic=0.8, confidence=0.8),
        "video": ModalityResult(modality="video", prob_synthetic=0.97, confidence=0.9),
    }
    out = fuse(results, weights=_WEIGHTS)
    assert out.verdict == "synthetic"
    assert out.fused_score > 0.8
    assert out.fused_confidence > 0.5


def test_all_four_confident_real() -> None:
    results = {
        "image": ModalityResult(modality="image", prob_synthetic=0.05, confidence=0.9),
        "audio": ModalityResult(modality="audio", prob_synthetic=0.02, confidence=0.95),
        "text": ModalityResult(modality="text", prob_synthetic=0.1, confidence=0.8),
        "video": ModalityResult(modality="video", prob_synthetic=0.03, confidence=0.9),
    }
    out = fuse(results, weights=_WEIGHTS)
    assert out.verdict == "real"
    assert out.fused_score < 0.2


def test_missing_modalities_are_excluded_not_penalised() -> None:
    """A text-only submission should fuse cleanly on text alone."""
    results = {
        "image": _unavailable("image"),
        "audio": _unavailable("audio"),
        "text": ModalityResult(modality="text", prob_synthetic=0.9, confidence=0.9),
        "video": _unavailable("video"),
    }
    out = fuse(results, weights=_WEIGHTS)
    assert out.weights_used["image"] == 0.0
    assert out.weights_used["audio"] == 0.0
    assert out.weights_used["video"] == 0.0
    assert out.weights_used["text"] == _WEIGHTS["text"]
    assert out.verdict == "synthetic"


def test_no_modalities_available_returns_uncertain() -> None:
    results = {m: _unavailable(m) for m in _WEIGHTS}
    out = fuse(results, weights=_WEIGHTS)
    assert out.verdict == "uncertain"
    assert out.fused_confidence == 0.0


def test_low_confidence_across_the_board_forces_uncertain() -> None:
    results = {
        "image": ModalityResult(modality="image", prob_synthetic=0.9, confidence=0.1),
        "audio": ModalityResult(modality="audio", prob_synthetic=0.9, confidence=0.1),
        "text": _unavailable("text"),
        "video": _unavailable("video"),
    }
    out = fuse(results, weights=_WEIGHTS, uncertain_confidence_floor=0.35)
    assert out.verdict == "uncertain"


def test_disagreement_flag_raised_and_confidence_dampened() -> None:
    results = {
        "image": _unavailable("image"),
        "audio": ModalityResult(modality="audio", prob_synthetic=0.95, confidence=0.9),
        "text": _unavailable("text"),
        "video": ModalityResult(modality="video", prob_synthetic=0.05, confidence=0.9),
    }
    out = fuse(results, weights=_WEIGHTS, disagreement_threshold=0.5)
    assert len(out.cross_modal_flags) == 1
    assert "audio" in out.cross_modal_flags[0] and "video" in out.cross_modal_flags[0]


def test_no_disagreement_flag_when_scores_agree() -> None:
    results = {
        "image": _unavailable("image"),
        "audio": ModalityResult(modality="audio", prob_synthetic=0.6, confidence=0.9),
        "text": _unavailable("text"),
        "video": ModalityResult(modality="video", prob_synthetic=0.65, confidence=0.9),
    }
    out = fuse(results, weights=_WEIGHTS, disagreement_threshold=0.5)
    assert out.cross_modal_flags == []


def test_weights_used_reflects_only_available_modalities() -> None:
    results = {
        "image": ModalityResult(modality="image", prob_synthetic=0.5, confidence=0.5),
        "audio": _unavailable("audio"),
        "text": _unavailable("text"),
        "video": _unavailable("video"),
    }
    out = fuse(results, weights=_WEIGHTS)
    assert out.weights_used == {"image": 0.20, "audio": 0.0, "text": 0.0, "video": 0.0}


def test_explanation_mentions_verdict_and_modalities() -> None:
    results = {
        "image": _unavailable("image"),
        "audio": ModalityResult(modality="audio", prob_synthetic=0.9, confidence=0.9),
        "text": _unavailable("text"),
        "video": _unavailable("video"),
    }
    out = fuse(results, weights=_WEIGHTS)
    assert "audio" in out.explanation
    assert out.verdict in out.explanation
