"""Shape + config-dispatch tests for the temporal detector.

We instantiate the model with ``pretrained=False`` so tests don't try
to hit the HuggingFace hub during CI.
"""

from __future__ import annotations

import torch

from src.models import BaselineDetector, TemporalDetector, build_model_from_config


def _dummy_video(b: int = 2, t: int = 32, size: int = 224) -> torch.Tensor:
    return torch.randn(b, t, 3, size, size)


def test_temporal_forward_shape() -> None:
    model = TemporalDetector(pretrained=False, num_frames=32, hidden_dim=64,
                             num_layers=1, num_heads=2)
    model.eval()
    with torch.no_grad():
        out = model.forward_video(_dummy_video(b=2, t=32))
    assert out.shape == (2, 2)


def test_temporal_forward_with_padding_mask() -> None:
    """Padding mask must be honoured — no crash and same output shape."""
    model = TemporalDetector(pretrained=False, num_frames=32, hidden_dim=64,
                             num_layers=1, num_heads=2)
    model.eval()
    b, t = 2, 32
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[0, -4:] = True   # last 4 frames of sample 0 are padding
    with torch.no_grad():
        out = model.forward_video(_dummy_video(b=b, t=t), padding_mask=mask)
    assert out.shape == (b, 2)


def test_factory_dispatches_on_head_type() -> None:
    mean_cfg = {
        "backbone": "tf_efficientnet_b0.ns_jft_in1k",
        "pretrained": False,
        "input": {"frames_per_clip": 32},
        "temporal_head": {"type": "mean_pool"},
        "classifier": {"dropout": 0.1},
    }
    transformer_cfg = {
        **mean_cfg,
        "temporal_head": {"type": "transformer", "hidden_dim": 64,
                          "num_layers": 1, "num_heads": 2},
    }
    assert isinstance(build_model_from_config(mean_cfg), BaselineDetector)
    assert isinstance(build_model_from_config(transformer_cfg), TemporalDetector)


def test_factory_rejects_unknown_head_type() -> None:
    bad_cfg = {
        "backbone": "tf_efficientnet_b0.ns_jft_in1k",
        "pretrained": False,
        "input": {"frames_per_clip": 32},
        "temporal_head": {"type": "definitely_not_a_real_head"},
        "classifier": {"dropout": 0.1},
    }
    try:
        build_model_from_config(bad_cfg)
    except ValueError as exc:
        assert "definitely_not_a_real_head" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown head type")
