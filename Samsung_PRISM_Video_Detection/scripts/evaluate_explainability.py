"""Compute the mean explainability score over the FF++ test set.

Runs GradCAM on every cached test-set face crop, computes the
face-localisation score per frame, averages per video, then reports
the aggregate mean. The worklet target is ≥ 0.85 (i.e. 85%).

Only the cached crops from ``prepare_dataset.py`` are used — no video
decoding — so this script is fast (~5–10 min on M5 CPU GradCAM).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import FaceVideoDataset
from src.explainability import FrameGradCAM, video_explainability_score
from src.models import build_model_from_config
from src.utils import get_logger, load_paths, project_root

app = typer.Typer(add_completion=False, help="Aggregate explainability score on cached crops.")
logger = get_logger(__name__)


@app.command()
def main(
    checkpoint: Path = typer.Option(
        Path("checkpoints/best.pt"), help="Path to the .pt checkpoint."
    ),
    split: str = typer.Option("test", help='Split to evaluate: "val" or "test".'),
    limit: int | None = typer.Option(None, help="Process at most N videos (smoke test)."),
    dataset: str = typer.Option("ff_c23", help="Which face-crop cache to use."),
) -> None:
    """Compute the mean face-localisation score across ``split`` videos."""
    root = project_root()
    paths = load_paths()

    ckpt_path = checkpoint if checkpoint.is_absolute() else (root / checkpoint)
    if not ckpt_path.exists():
        raise typer.BadParameter(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = (ckpt.get("config") or {}).get("model")
    if not model_cfg:
        raise typer.BadParameter(f"Checkpoint {ckpt_path} has no embedded model config.")

    model = build_model_from_config(model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cammer = FrameGradCAM(model, device="cpu")

    faces_root = paths["data"]["processed"] / "faces" / dataset
    ds = FaceVideoDataset(faces_root, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    per_video: list[float] = []
    per_manip: dict[str, list[float]] = {}
    total = min(limit, len(ds)) if limit else len(ds)
    for i, batch in enumerate(tqdm(loader, total=total, unit="vid")):
        if limit is not None and i >= limit:
            break
        images = batch["images"][0]                          # (T, 3, H, W)
        manip = str(batch["manipulation"][0])
        cams = cammer(images)                                # (T, H, W)
        score = video_explainability_score(cams)
        per_video.append(score)
        per_manip.setdefault(manip, []).append(score)

    mean_score = float(np.mean(per_video)) if per_video else 0.0

    metrics_dir = Path(paths["outputs"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if dataset == "ff_c23" else f"_{dataset}"
    out_path = metrics_dir / f"explainability{tag}_{split}.json"
    payload = {
        "mean_explainability_score": mean_score,
        "n_videos": len(per_video),
        "per_manipulation_mean": {k: float(np.mean(v)) for k, v in per_manip.items()},
        "threshold_target": 0.85,
        "meets_target": mean_score >= 0.85,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    logger.info("--- Explainability report [dataset=%s, split=%s] ---", dataset, split)
    logger.info("Mean explainability score : %.4f  (target ≥ 0.85 → %s)",
                mean_score, "PASS ✅" if mean_score >= 0.85 else "MISS")
    logger.info("Videos scored             : %d", len(per_video))
    for name, vals in sorted(per_manip.items()):
        logger.info("  %-20s %.4f  (n=%d)", name, float(np.mean(vals)), len(vals))
    logger.info("Wrote report to %s", out_path)


if __name__ == "__main__":
    app()
