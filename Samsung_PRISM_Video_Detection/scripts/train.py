"""Train the baseline video-deepfake detector.

Reads ``configs/train.yaml`` and ``configs/model.yaml``, runs the loop
in :mod:`src.training.trainer`, and saves the best-val-F1 checkpoint
to ``checkpoints/best.pt``.

Example
-------
::

    python scripts/train.py                       # use configs as-is
    python scripts/train.py --epochs 10           # override one field
    python scripts/train.py --device cpu          # force CPU
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import typer

from src.training import run_training
from src.utils import get_logger, load_paths, load_yaml, project_root

app = typer.Typer(add_completion=False, help="Train the baseline detector.")
logger = get_logger(__name__)


@app.command()
def main(
    epochs: int | None = typer.Option(None, help="Override train.yaml epochs."),
    batch_size: int | None = typer.Option(None, help="Override train.yaml batch_size."),
    lr: float | None = typer.Option(None, help="Override optimizer.lr."),
    device: str | None = typer.Option(None, help="Force device: mps | cuda | cpu."),
    extra_training_root: list[Path] = typer.Option(
        [],
        "--extra-training-root",
        help=(
            "Extra face-crop dataset root to concatenate into the training "
            "set. Can be passed multiple times. Val/test remain on FF++ so "
            "numbers stay comparable. Example: "
            "`--extra-training-root data/processed/faces/celeb_df_v2`."
        ),
    ),
) -> None:
    """Kick off training with the current configs (with optional CLI overrides)."""
    root = project_root()
    train_yaml = load_yaml(root / "configs" / "train.yaml")
    model_yaml = load_yaml(root / "configs" / "model.yaml")
    paths = load_paths()

    if epochs is not None:
        train_yaml["train"]["epochs"] = int(epochs)
    if batch_size is not None:
        train_yaml["train"]["batch_size"] = int(batch_size)
    if lr is not None:
        train_yaml["train"]["optimizer"]["lr"] = float(lr)
    if device is not None:
        train_yaml["train"]["device"] = device

    faces_root = paths["data"]["processed"] / "faces" / "ff_c23"
    if not faces_root.is_dir():
        raise typer.BadParameter(
            f"Face crops not found under {faces_root}. Run "
            "`python scripts/prepare_dataset.py faces` first."
        )

    resolved_extras: list[Path] = []
    for p in extra_training_root:
        pp = p if p.is_absolute() else (root / p)
        if not (pp / "train").is_dir():
            raise typer.BadParameter(
                f"--extra-training-root {p} has no 'train/' subfolder — "
                "did the prep script finish?"
            )
        resolved_extras.append(pp)

    ckpt = run_training(
        train_yaml, model_yaml, paths, faces_root,
        extra_training_roots=resolved_extras or None,
    )
    logger.info("Done. Best checkpoint: %s", ckpt)


if __name__ == "__main__":
    app()
