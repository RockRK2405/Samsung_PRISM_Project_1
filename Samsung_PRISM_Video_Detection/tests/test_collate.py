"""Tests for the padded video collate function."""

from __future__ import annotations

import torch

from src.datasets import pad_video_collate


def _mk(t: int, label: int, video_id: str) -> dict:
    return {
        "images": torch.zeros(t, 3, 224, 224),
        "label": label,
        "video_id": video_id,
        "manipulation": "original" if label == 0 else "Deepfakes",
    }


def test_collate_pads_short_clips() -> None:
    batch = [_mk(30, 1, "a"), _mk(32, 0, "b")]
    out = pad_video_collate(batch, pad_to=32)
    assert out["images"].shape == (2, 32, 3, 224, 224)
    assert out["padding_mask"].shape == (2, 32)
    # Sample 0 (30 frames) has 2 padded positions at the tail.
    assert out["padding_mask"][0, -1].item() is True
    assert out["padding_mask"][0, -2].item() is True
    assert out["padding_mask"][0, 29].item() is False
    # Sample 1 (32 frames) has no padding.
    assert not out["padding_mask"][1].any().item()


def test_collate_truncates_overlong_clips() -> None:
    batch = [_mk(40, 1, "x")]
    out = pad_video_collate(batch, pad_to=32)
    assert out["images"].shape == (1, 32, 3, 224, 224)
    assert not out["padding_mask"].any().item()


def test_collate_labels_are_long_tensor() -> None:
    batch = [_mk(32, 0, "a"), _mk(32, 1, "b")]
    out = pad_video_collate(batch, pad_to=32)
    assert out["label"].dtype == torch.long
    assert out["label"].tolist() == [0, 1]
