"""Evaluate a saved checkpoint on a val or test split.

Supports both **in-distribution** evaluation (default, on FF++ c23) and
**cross-dataset generalisation** (``--dataset celeb_df_v2``). For cross-
dataset runs threshold tuning is performed on FF++ val — the source
distribution — since the target dataset has no train/val we've fitted.

Writes a JSON report to ``outputs/metrics/`` (filename includes the
dataset tag when it's not the default FF++) and prints a compact
summary to the terminal.

Examples
--------
::

    # In-distribution (FF++ test) — Milestone 4/5A
    python scripts/evaluate.py --split test --tune-threshold-fpr 0.05

    # Cross-dataset (Celeb-DF v2 test) — Milestone 5B
    python scripts/evaluate.py --split test --dataset celeb_df_v2 \\
        --checkpoint checkpoints/best.pt --tune-threshold-fpr 0.05
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import typer

from src.training import run_evaluation
from src.utils import get_logger, load_paths, load_yaml, project_root

app = typer.Typer(add_completion=False, help="Evaluate a checkpoint.")
logger = get_logger(__name__)

_SUPPORTED_DATASETS = ("ff_c23", "celeb_df_v2")


@app.command()
def main(
    checkpoint: Path = typer.Option(
        Path("checkpoints/best.pt"), help="Path to the .pt checkpoint."
    ),
    split: str = typer.Option("test", help='Split to evaluate: "val" or "test".'),
    dataset: str = typer.Option(
        "ff_c23",
        help=f"Which face-crop cache to evaluate. One of: {', '.join(_SUPPORTED_DATASETS)}.",
    ),
    device: str | None = typer.Option(None, help="Force device: mps | cuda | cpu."),
    tune_threshold_fpr: float | None = typer.Option(
        None,
        help=(
            "If set, tune the decision threshold on FF++ val to keep FPR ≤ "
            "this value, then apply it to the requested (dataset, split). "
            "Use 0.05 for the ≤5%% project target."
        ),
    ),
) -> None:
    """Load ``checkpoint`` and report metrics on (``dataset``, ``split``)."""
    if dataset not in _SUPPORTED_DATASETS:
        raise typer.BadParameter(
            f"Unknown --dataset '{dataset}'. Choose one of: {_SUPPORTED_DATASETS}"
        )

    root = project_root()
    model_yaml = load_yaml(root / "configs" / "model.yaml")
    paths = load_paths()

    ckpt_path = checkpoint if checkpoint.is_absolute() else (root / checkpoint)
    if not ckpt_path.exists():
        raise typer.BadParameter(f"Checkpoint not found: {ckpt_path}")

    faces_root = paths["data"]["processed"] / "faces" / dataset
    if not faces_root.is_dir():
        hint = (
            "python scripts/prepare_dataset.py faces"
            if dataset == "ff_c23"
            else "python scripts/prepare_celeb_df.py --dataset-root <path>"
        )
        raise typer.BadParameter(
            f"Face crops for '{dataset}' not found at {faces_root}. Run: {hint}"
        )

    # Always tune on FF++ val (the distribution we trained on) so
    # cross-dataset runs use a legitimately-derived threshold.
    tune_faces_root = paths["data"]["processed"] / "faces" / "ff_c23"

    result = run_evaluation(
        ckpt_path, model_yaml, paths, faces_root, split, device,
        tune_threshold_fpr=tune_threshold_fpr,
        tune_faces_root=tune_faces_root if tune_threshold_fpr is not None else None,
        dataset_name=dataset,
    )

    r = result["report"]
    logger.info("--- Evaluation summary [dataset=%s, split=%s] ---", dataset, split)
    logger.info("Threshold: %.4f", result["threshold"])
    logger.info("Accuracy : %.4f", r.accuracy)
    logger.info("Precision: %.4f", r.precision)
    logger.info("Recall   : %.4f", r.recall)
    logger.info("F1       : %.4f", r.f1)
    logger.info("FPR      : %.4f", r.fpr)
    logger.info("AUC      : %.4f", r.auc)
    logger.info("n_real=%d n_synthetic=%d", r.n_real, r.n_synthetic)
    for name, val in r.per_manipulation_recall.items():
        logger.info("  %-20s %.4f", name, val)


if __name__ == "__main__":
    app()
