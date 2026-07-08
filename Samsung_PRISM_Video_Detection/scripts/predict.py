"""Run the detector on a single video and print / save its DetectionResult.

Wraps :class:`src.inference.VideoDetector`. Writes JSON to
``outputs/predictions/<video-stem>.json`` and (if explanation is on)
GradCAM overlays + a per-frame timeline PNG under
``outputs/explainability/<video-stem>/``.

Examples
--------
::

    python scripts/predict.py --video path/to/clip.mp4
    python scripts/predict.py --video path/to/clip.mp4 --no-explanation
    python scripts/predict.py --video path/to/clip.mp4 \
        --checkpoint checkpoints/best_dual.pt --threshold 0.5872
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import typer

from src.inference import VideoDetector
from src.utils import get_logger, load_paths, project_root

app = typer.Typer(add_completion=False, help="Predict on a single video.")
logger = get_logger(__name__)


@app.command()
def main(
    video: Path = typer.Option(..., help="Path to a video file."),
    checkpoint: Path = typer.Option(
        Path("checkpoints/best.pt"), help="Path to the .pt checkpoint."
    ),
    threshold: float = typer.Option(
        0.5872, help="Decision threshold (default: FF++-val-tuned for FPR ≤ 5%%)."
    ),
    device: str | None = typer.Option(None, help="Force device: mps | cuda | cpu."),
    produce_explanation: bool = typer.Option(
        True, "--explanation/--no-explanation",
        help="Also produce GradCAM heatmaps + per-frame timeline.",
    ),
) -> None:
    """Predict on ``video`` and write the JSON payload to outputs/predictions/."""
    root = project_root()
    paths = load_paths()

    ckpt_path = checkpoint if checkpoint.is_absolute() else (root / checkpoint)
    if not ckpt_path.exists():
        raise typer.BadParameter(f"Checkpoint not found: {ckpt_path}")
    if not video.is_file():
        raise typer.BadParameter(f"Video not found: {video}")

    detector = VideoDetector(ckpt_path, threshold=threshold, device=device)
    result = detector.predict(video, produce_explanation=produce_explanation)

    preds_dir = Path(paths["outputs"]["predictions"])
    out_json = preds_dir / f"{video.stem}.json"
    result.to_json(out_json)

    logger.info("--- Prediction summary ---")
    logger.info("Video           : %s", result.video_path)
    logger.info("Prediction      : %s", result.prediction)
    logger.info("P(synthetic)    : %.4f", result.prob_synthetic)
    logger.info("Confidence      : %.4f", result.confidence)
    logger.info("Threshold       : %.4f", result.threshold)
    logger.info("Frames used     : %d", result.num_frames_used)
    if result.explainability_score is not None:
        logger.info("Explainability  : %.4f", result.explainability_score)
        logger.info("Heatmaps        : %s", result.heatmap_dir)
        logger.info("Timeline        : %s", result.timeline_path)
    logger.info("JSON payload    : %s", out_json)


if __name__ == "__main__":
    app()
