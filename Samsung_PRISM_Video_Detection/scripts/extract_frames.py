"""Standalone frame-extraction CLI.

Most users should run ``scripts/prepare_dataset.py faces`` instead — that
command fuses frame extraction and face cropping in a single pass so no
intermediate frame files are written to disk.

This script exists for debugging: given a single video path, decode the
configured number of uniformly-sampled frames and write them to disk so
you can eyeball what the frame extractor is producing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import typer
from PIL import Image

from src.preprocessing import FrameExtractor
from src.utils import get_logger, load_yaml, project_root

app = typer.Typer(add_completion=False, help="Debug helper — dump sampled frames of one video.")
logger = get_logger(__name__)


@app.command()
def main(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to a .mp4 file."),
    out_dir: Path = typer.Option(
        Path("outputs/visualizations/frames"),
        help="Where to write the JPEGs (relative to project root).",
    ),
    num_frames: int | None = typer.Option(
        None, help="Override configs/dataset.yaml frames_per_video."
    ),
) -> None:
    """Decode ``num_frames`` uniform frames from ``video`` and save them as JPEGs."""
    ds_cfg = load_yaml(project_root() / "configs" / "dataset.yaml")["dataset"]
    n = num_frames or int(ds_cfg["sampling"]["frames_per_video"])
    extractor = FrameExtractor(num_frames=n, strategy=str(ds_cfg["sampling"]["strategy"]))

    frames = extractor.extract(video)
    resolved_out = out_dir if out_dir.is_absolute() else (project_root() / out_dir)
    resolved_out.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(resolved_out / f"{video.stem}_{i:02d}.jpg", quality=95)

    logger.info("Wrote %d frames of %s to %s", len(frames), video.name, resolved_out)


if __name__ == "__main__":
    app()
