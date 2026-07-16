"""Unit tests for the shared wire-format dataclasses."""

from __future__ import annotations

import pytest

from fusion_engine.schema import MODALITIES, ModalityResult


def test_modality_result_accepts_valid_values() -> None:
    r = ModalityResult(modality="video", prob_synthetic=0.8, confidence=0.9)
    assert r.modality == "video"
    assert r.available is True


def test_modality_result_rejects_unknown_modality() -> None:
    with pytest.raises(ValueError):
        ModalityResult(modality="smell", prob_synthetic=0.5, confidence=0.5)


def test_modality_result_rejects_out_of_range_prob() -> None:
    with pytest.raises(ValueError):
        ModalityResult(modality="text", prob_synthetic=1.5, confidence=0.5)


def test_modality_result_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError):
        ModalityResult(modality="text", prob_synthetic=0.5, confidence=-0.1)


def test_unavailable_result_skips_range_validation() -> None:
    # An unavailable result may carry placeholder 0.0 values -- must not raise.
    r = ModalityResult(modality="image", prob_synthetic=0.0, confidence=0.0, available=False)
    assert r.available is False


def test_to_dict_round_trips_basic_fields() -> None:
    r = ModalityResult(modality="audio", prob_synthetic=0.42, confidence=0.7, raw={"x": 1})
    d = r.to_dict()
    assert d["modality"] == "audio"
    assert d["prob_synthetic"] == 0.42
    assert d["raw"] == {"x": 1}


def test_all_four_modalities_defined() -> None:
    assert set(MODALITIES) == {"image", "audio", "text", "video"}
