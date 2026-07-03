"""Unit tests for the frame extractor's index-selection logic.

We test the pure helper ``_uniform_indices`` directly; end-to-end
extraction requires a real video file and is deferred to a manual
smoke test.
"""

from __future__ import annotations

import pytest

from src.preprocessing.frame_extractor import FrameExtractor


def test_uniform_indices_long_video() -> None:
    """Long enough video → evenly-spaced, monotonically increasing indices."""
    indices = FrameExtractor._uniform_indices(total=300, n=32)
    assert len(indices) == 32
    assert indices[0] == 0
    assert indices[-1] == 299
    assert all(a <= b for a, b in zip(indices, indices[1:]))


def test_uniform_indices_short_video_pads_by_clamp() -> None:
    """When the clip is shorter than n, indices clamp to the last frame."""
    indices = FrameExtractor._uniform_indices(total=5, n=32)
    assert len(indices) == 32
    assert indices[:5] == [0, 1, 2, 3, 4]
    assert all(i == 4 for i in indices[5:])


def test_uniform_indices_exactly_n() -> None:
    """total == n should yield the identity mapping."""
    indices = FrameExtractor._uniform_indices(total=32, n=32)
    assert indices == list(range(32))


def test_constructor_rejects_bad_num_frames() -> None:
    with pytest.raises(ValueError):
        FrameExtractor(num_frames=0)


def test_constructor_rejects_unknown_strategy() -> None:
    with pytest.raises(NotImplementedError):
        FrameExtractor(num_frames=16, strategy="dense")
