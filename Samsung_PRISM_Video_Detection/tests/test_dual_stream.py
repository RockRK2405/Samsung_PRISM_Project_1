"""Shape + factory tests for the dual-stream detector.

Uses ``pretrained=False`` so tests don't try to hit the HuggingFace hub.
"""

from __future__ import annotations

import torch

from src.models import BaselineDetector, DualStreamDetector, build_model_from_config


def _dummy_batch(b: int = 2, size: int = 224) -> torch.Tensor:
    return torch.randn(b, 3, size, size)


def test_frame_forward_shape() -> None:
    model = DualStreamDetector(pretrained=False, freeze_rgb=False)
    model.eval()
    with torch.no_grad():
        out = model(_dummy_batch(b=2))
    assert out.shape == (2, 2)


def test_video_forward_shape_meanpool_over_frames() -> None:
    model = DualStreamDetector(pretrained=False, freeze_rgb=False)
    model.eval()
    video = torch.randn(2, 32, 3, 224, 224)
    with torch.no_grad():
        out = model.forward_video(video)
    assert out.shape == (2, 2)


def test_freeze_rgb_disables_grads_on_rgb_only() -> None:
    model = DualStreamDetector(pretrained=False, freeze_rgb=True)
    rgb_grads = [p.requires_grad for p in model.rgb_backbone.parameters()]
    freq_grads = [p.requires_grad for p in model.freq_backbone.parameters()]
    cls_grads = [p.requires_grad for p in model.classifier.parameters()]
    assert not any(rgb_grads)
    assert all(freq_grads)
    assert all(cls_grads)


def test_fft_preprocessing_preserves_shape() -> None:
    x = torch.randn(2, 3, 224, 224)
    out = DualStreamDetector._log_fft_magnitude(x)
    assert out.shape == x.shape
    # Per-image, per-channel z-score → mean ~ 0.
    assert out.mean(dim=(-1, -2)).abs().max().item() < 1e-4


def test_factory_picks_dual_when_frequency_stream_enabled() -> None:
    cfg = {
        "backbone": "tf_efficientnet_b0.ns_jft_in1k",
        "pretrained": False,
        "input": {"frames_per_clip": 32},
        "spatial_stream": {"freeze": False},
        "frequency_stream": {"enabled": True},
        "temporal_head": {"type": "mean_pool"},
        "classifier": {"dropout": 0.1},
    }
    m = build_model_from_config(cfg)
    assert isinstance(m, DualStreamDetector)


def test_factory_falls_back_to_baseline_when_freq_disabled() -> None:
    cfg = {
        "backbone": "tf_efficientnet_b0.ns_jft_in1k",
        "pretrained": False,
        "input": {"frames_per_clip": 32},
        "spatial_stream": {"freeze": False},
        "frequency_stream": {"enabled": False},
        "temporal_head": {"type": "mean_pool"},
        "classifier": {"dropout": 0.1},
    }
    m = build_model_from_config(cfg)
    assert isinstance(m, BaselineDetector)
