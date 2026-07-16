"""Integration test for the Image adapter against the real checkpoint.

Uses a synthetic in-memory image (no dependency on any dataset being
present) purely to verify the adapter's plumbing: checkpoint loads,
forward pass runs, output is a well-formed ModalityResult. This is NOT
an accuracy test -- see Samsung_PRISM_Image_Detection's own notebooks
for that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fusion_engine.adapters.image_adapter import ImageAdapter, _DEFAULT_CHECKPOINT


@pytest.mark.skipif(
    not _DEFAULT_CHECKPOINT.is_file(),
    reason="Image checkpoint not present in this checkout",
)
def test_image_adapter_loads_checkpoint_and_predicts(tmp_path: Path) -> None:
    adapter = ImageAdapter()

    # A simple synthetic RGB image -- just needs to be a valid image file.
    arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    img_path = tmp_path / "test.png"
    Image.fromarray(arr).save(img_path)

    result = adapter.predict(img_path)

    assert result.modality == "image"
    assert result.available is True
    assert 0.0 <= result.prob_synthetic <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.raw["prediction"] in {"FAKE", "REAL"}
    assert result.latency_ms is not None and result.latency_ms > 0


@pytest.mark.skipif(
    not _DEFAULT_CHECKPOINT.is_file(),
    reason="Image checkpoint not present in this checkout",
)
def test_image_adapter_handles_bad_file_gracefully(tmp_path: Path) -> None:
    adapter = ImageAdapter()
    bad_path = tmp_path / "not_an_image.png"
    bad_path.write_bytes(b"this is not a valid image file")

    result = adapter.predict(bad_path)

    assert result.available is False
    assert result.error is not None
    assert result.prob_synthetic == 0.0


def test_image_adapter_raises_on_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ImageAdapter(checkpoint_path=tmp_path / "does_not_exist.pth")
