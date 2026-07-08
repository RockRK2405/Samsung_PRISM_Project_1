"""Tests for the fusion-engine integration contract.

Covers the wire format (schema constants + DetectionResult round-trip)
without touching video decoding or the actual model — that's exercised
by the smoke test in scripts/predict.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.inference import (
    MODALITY,
    SCHEMA_VERSION,
    DetectionResult,
)


def _sample_payload() -> dict:
    return {
        "video_path": "/tmp/clip.mp4",
        "prediction": "synthetic",
        "prob_synthetic": 0.87,
        "confidence": 0.71,
        "threshold": 0.5872,
        "per_frame_scores": [0.9, 0.85, 0.88, 0.82],
        "num_frames_used": 4,
        "explainability_score": 0.92,
        "heatmap_dir": "/tmp/heatmaps",
        "timeline_path": "/tmp/timeline.png",
        "meta": {"backbone": "tf_efficientnet_b0.ns_jft_in1k"},
        "schema_version": 1,
        "modality": "video",
    }


def test_constants_are_stable() -> None:
    """MODALITY and SCHEMA_VERSION are part of the wire contract — do not rename."""
    assert MODALITY == "video"
    assert SCHEMA_VERSION == 1


def test_default_fields_populate_schema_and_modality() -> None:
    r = DetectionResult(
        video_path="a.mp4",
        prediction="real",
        prob_synthetic=0.1,
        confidence=0.9,
        threshold=0.5,
        per_frame_scores=[0.1, 0.1],
        num_frames_used=2,
    )
    assert r.modality == "video"
    assert r.schema_version == 1


def test_to_dict_from_dict_round_trip() -> None:
    original = DetectionResult(**{
        k: v for k, v in _sample_payload().items()
    })
    round_tripped = DetectionResult.from_dict(original.to_dict())
    assert round_tripped == original


def test_from_dict_ignores_unknown_fields() -> None:
    payload = _sample_payload()
    payload["future_field_from_v2"] = {"foo": "bar"}
    r = DetectionResult.from_dict(payload)
    assert r.prob_synthetic == payload["prob_synthetic"]


def test_from_dict_missing_required_field_raises() -> None:
    payload = _sample_payload()
    del payload["prediction"]
    with pytest.raises(TypeError):
        DetectionResult.from_dict(payload)


def test_json_round_trip(tmp_path: Path) -> None:
    r = DetectionResult.from_dict(_sample_payload())
    out = tmp_path / "clip.json"
    r.to_json(out)
    loaded = DetectionResult.from_json(out)
    assert loaded == r


def test_schema_file_exists_and_lists_required_fields() -> None:
    """The JSON schema file should live at schemas/… and match our required set."""
    import json

    project_root = Path(__file__).resolve().parents[1]
    schema_path = project_root / "schemas" / "video_detection_result.schema.json"
    assert schema_path.is_file(), f"missing schema: {schema_path}"
    schema = json.loads(schema_path.read_text())
    required = set(schema["required"])
    # The schema and the dataclass agree on what's mandatory:
    dataclass_required = {
        "schema_version", "modality", "video_path", "prediction",
        "prob_synthetic", "confidence", "threshold",
        "per_frame_scores", "num_frames_used",
    }
    assert required == dataclass_required
