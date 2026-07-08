"""Per-frame synthetic-probability timeline plot.

A single line plot showing how the detector's confidence evolves across
the 32 sampled frames of a video. Useful for a human reviewer to spot
frame-local ambiguity ("was uncertain around frames 12–18") rather than
seeing only a single video-level number.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for CLI scripts
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402


def plot_timeline(
    scores: list[float],
    threshold: float,
    out_path: Path,
    title: str = "",
) -> None:
    """Save a per-frame probability timeline as a PNG.

    Args:
        scores: One synthetic-probability per sampled frame.
        threshold: Decision threshold (drawn as a red dashed line).
        out_path: Where to save the PNG. Parent dirs are created.
        title: Optional plot title.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    frames = np.arange(len(scores))
    ax.plot(frames, scores, marker="o", linewidth=1.4, color="#2264d1", zorder=3)
    ax.fill_between(frames, scores, threshold,
                    where=[s >= threshold for s in scores],
                    color="#d13a3a", alpha=0.25, label="above threshold")
    ax.axhline(threshold, color="#d13a3a", linestyle="--", linewidth=1.2,
               label=f"threshold = {threshold:.3f}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("P(synthetic)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
