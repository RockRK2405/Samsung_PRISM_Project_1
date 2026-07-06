"""Evaluate a saved checkpoint on the val or test split.

Writes a JSON report to ``outputs/metrics/baseline_<split>.json`` and
prints a compact summary to the terminal.

Example
-------
::

    python scripts/evaluate.py                              # test split, best.pt
    python scripts/evaluate.py --split val                  # val split
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
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

app = typer.Typer(add_completion=False, help="Evaluate a baseline checkpoint.")
logger = get_logger(__name__)


@app.command()
def main(
    checkpoint: Path = typer.Option(
        Path("checkpoints/best.pt"), help="Path to the .pt checkpoint."
    ),
    split: str = typer.Option("test", help='Split to evaluate: "val" or "test".'),
    device: str | None = typer.Option(None, help="Force device: mps | cuda | cpu."),
    tune_threshold_fpr: float | None = typer.Option(
        None,
        help=(
            "If set, tune the decision threshold on val to keep FPR ≤ this "
            "value, then apply it to `split`. Use 0.05 for the ≤5%% project "
            "target."
        ),
    ),
) -> None:
    """Load ``checkpoint`` and report metrics on ``split``."""
    root = project_root()
    model_yaml = load_yaml(root / "configs" / "model.yaml")
    paths = load_paths()

    ckpt_path = checkpoint if checkpoint.is_absolute() else (root / checkpoint)
    if not ckpt_path.exists():
        raise typer.BadParameter(f"Checkpoint not found: {ckpt_path}")

    faces_root = paths["data"]["processed"] / "faces" / "ff_c23"
    result = run_evaluation(
        ckpt_path, model_yaml, paths, faces_root, split, device,
        tune_threshold_fpr=tune_threshold_fpr,
    )

    r = result["report"]
    logger.info("--- Evaluation summary [%s] ---", split)
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
