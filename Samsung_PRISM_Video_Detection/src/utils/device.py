"""Compute-device selection.

Picks the best available PyTorch device: MPS on Apple Silicon, then
CUDA, then CPU. Callers should always go through :func:`select_device`
rather than referring to ``"mps"`` / ``"cuda"`` strings directly.
"""

from __future__ import annotations

import torch


def select_device(preferred: str | None = None) -> torch.device:
    """Return the best available compute device.

    Args:
        preferred: Optional explicit override. One of ``"mps"``, ``"cuda"``,
            ``"cpu"``, or ``None`` (auto-select). Unavailable choices fall
            back silently to the next-best option.

    Returns:
        A :class:`torch.device` — never ``None``.
    """
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
