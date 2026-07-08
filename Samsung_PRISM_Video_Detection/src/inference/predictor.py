"""Single-video prediction API — the fusion engine's entrypoint.

Given a raw video path this class runs the full pipeline
frame extraction → face detection → per-frame model inference →
mean-pool aggregation → optional GradCAM + timeline, and returns a
serialisable :class:`DetectionResult` matching the schema the fusion
engine expects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.datasets.ff_dataset import _to_tensor
from src.explainability import (
    FrameGradCAM,
    plot_timeline,
    save_overlay,
    video_explainability_score,
)
from src.models import BaselineDetector, DualStreamDetector, build_model_from_config
from src.preprocessing import FrameExtractor, MTCNNFaceDetector
from src.utils import get_logger, select_device
from src.utils.paths import project_root

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------

#: Public modality identifier — the fusion engine keys per-modality
#: modules by this string. Do not rename.
MODALITY: str = "video"

#: Bumped when the DetectionResult wire format changes in an incompatible
#: way (renamed field, removed field, changed semantics). Adding new
#: optional fields does NOT bump the version.
SCHEMA_VERSION: int = 1


@dataclass
class DetectionResult:
    """Per-video prediction payload consumed by the fusion engine.

    Attributes:
        schema_version: Wire-format version. Match against
            :data:`SCHEMA_VERSION` on the fusion side.
        modality: Always ``"video"``. Lets the fusion engine confirm which
            per-modality module produced the payload.
        video_path: The input video's path.
        prediction: ``"real"`` or ``"synthetic"``.
        prob_synthetic: Video-level P(synthetic) in ``[0, 1]``.
        confidence: Margin from the decision threshold, mapped to
            ``[0, 1]`` — 0 means "right on the boundary", 1 means
            "as certain as we can be about this decision".
        threshold: The decision threshold that produced ``prediction``.
        per_frame_scores: One P(synthetic) per usable extracted frame.
        num_frames_used: Number of frames the model actually scored
            (may be < ``frames_per_video`` if MTCNN missed some).
        explainability_score: Face-localisation fraction in ``[0, 1]``.
        heatmap_dir: Directory holding per-frame overlays (if produced).
        timeline_path: Path to the per-frame timeline PNG (if produced).
        meta: Freeform extras (model name, checkpoint path, git commit, ...).
    """

    video_path: str
    prediction: str
    prob_synthetic: float
    confidence: float
    threshold: float
    per_frame_scores: list[float]
    num_frames_used: int
    explainability_score: float | None = None
    heatmap_dir: str | None = None
    timeline_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    modality: str = MODALITY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DetectionResult":
        """Reconstruct a :class:`DetectionResult` from a plain dict.

        Used by the fusion engine (and by our tests) to load a stored
        payload back into a typed object. Unknown fields are silently
        dropped; missing required fields raise ``KeyError``.
        """
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        clean = {k: v for k, v in payload.items() if k in allowed}
        return cls(**clean)

    @classmethod
    def from_json(cls, path: Path) -> "DetectionResult":
        """Load a :class:`DetectionResult` from a JSON file on disk."""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------

class VideoDetector:
    """Load a checkpoint once, then run :meth:`predict` on any video."""

    def __init__(
        self,
        checkpoint_path: Path,
        threshold: float = 0.5,
        frames_per_video: int = 32,
        image_size: int = 224,
        device: str | torch.device | None = None,
        gradcam_device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = float(threshold)
        self.device = select_device(device) if isinstance(device, str) or device is None else device
        self.gradcam_device = torch.device(gradcam_device)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        model_cfg = (ckpt.get("config") or {}).get("model")
        if not model_cfg:
            raise ValueError(
                f"Checkpoint {self.checkpoint_path} has no embedded model config. "
                "Retrain with the current trainer or pass configs/model.yaml explicitly."
            )
        self.model_cfg = model_cfg
        self.model: torch.nn.Module = build_model_from_config(model_cfg).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self._frames_per_video = int(frames_per_video)
        self._image_size = int(image_size)

        self._extractor = FrameExtractor(num_frames=self._frames_per_video, strategy="uniform")
        self._detector = MTCNNFaceDetector(image_size=self._image_size, margin=20, device="cpu")

        logger.info(
            "VideoDetector ready: checkpoint=%s model=%s device=%s threshold=%.3f",
            self.checkpoint_path.name, type(self.model).__name__, self.device, self.threshold,
        )

    @classmethod
    def load_default(cls, device: str | torch.device | None = None) -> "VideoDetector":
        """Load the project's canonical production detector.

        Reads ``checkpoints/best.pt`` (the Milestone-4 baseline that
        remained the best model on FF++) and applies the FF++-val-tuned
        FPR ≤ 5% threshold (``0.5872``). This is the entry point the
        fusion engine should call when it doesn't want to hand-pick a
        checkpoint.
        """
        root = project_root()
        return cls(
            checkpoint_path=root / "checkpoints" / "best.pt",
            threshold=0.5872,
            device=device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_frames(self, tensors: torch.Tensor) -> np.ndarray:
        """Return one P(synthetic) per frame in ``tensors`` (shape ``(N, 3, H, W)``)."""
        tensors = tensors.to(self.device)
        # Use the per-frame forward — DualStreamDetector.forward is per-frame,
        # BaselineDetector.forward is per-frame. Both return (N, 2) logits.
        logits = self.model(tensors)
        return torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

    def predict(
        self,
        video_path: str | Path,
        produce_explanation: bool = True,
        output_dir: str | Path | None = None,
    ) -> DetectionResult:
        """Full pipeline video → :class:`DetectionResult`.

        Args:
            video_path: Path to any video OpenCV can decode.
            produce_explanation: When ``True`` (default), also compute
                GradCAM heatmaps + per-frame timeline + explainability score.
            output_dir: Where to write explanation artefacts. When
                ``None`` and ``produce_explanation`` is on, defaults to
                ``outputs/explainability/<video-stem>/``.

        Returns:
            A populated :class:`DetectionResult`.
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 1. Extract frames
        frames_rgb = self._extractor.extract(video_path)

        # 2. Detect + crop faces
        crops, missed = self._detector.crop_video_frames(frames_rgb)
        if not crops:
            raise RuntimeError(
                f"No face detected in any of the {len(frames_rgb)} sampled frames of {video_path}"
            )
        if missed:
            logger.info("Face missed in %d / %d frames of %s", len(missed), len(frames_rgb), video_path.name)

        # 3. Normalise + stack
        from PIL import Image
        tensors = torch.stack(
            [_to_tensor(Image.fromarray(c)) for c in crops], dim=0
        )   # (N, 3, H, W)

        # 4. Score per frame
        per_frame = self._score_frames(tensors)                    # (N,)
        prob_synth = float(np.mean(per_frame))
        prediction = "synthetic" if prob_synth >= self.threshold else "real"

        # 5. Confidence — margin from threshold, mapped to [0, 1]
        max_margin = max(self.threshold, 1.0 - self.threshold)
        confidence = float(min(abs(prob_synth - self.threshold) / max_margin, 1.0))

        result = DetectionResult(
            video_path=str(video_path),
            prediction=prediction,
            prob_synthetic=prob_synth,
            confidence=confidence,
            threshold=self.threshold,
            per_frame_scores=[float(x) for x in per_frame],
            num_frames_used=int(len(per_frame)),
            meta={
                "checkpoint": str(self.checkpoint_path),
                "model_class": type(self.model).__name__,
                "backbone": str(self.model_cfg.get("backbone")),
                "frames_per_video_requested": self._frames_per_video,
                "frames_with_no_face": int(len(missed)),
            },
        )

        # 6. Explanation artefacts
        if produce_explanation:
            self._attach_explanation(result, crops, tensors, output_dir)
        return result

    # ------------------------------------------------------------------

    def _attach_explanation(
        self,
        result: DetectionResult,
        crops: list[np.ndarray],
        tensors: torch.Tensor,
        output_dir: str | Path | None,
    ) -> None:
        """Compute GradCAM + timeline + explainability score for ``result``."""
        if not isinstance(self.model, (BaselineDetector, DualStreamDetector)):
            logger.info(
                "Skipping GradCAM: %s not yet supported for explanation.",
                type(self.model).__name__,
            )
            return

        # Choose output directory
        stem = Path(result.video_path).stem
        out_dir = Path(output_dir) if output_dir is not None else (
            project_root() / "outputs" / "explainability" / stem
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # GradCAM on CPU (grad-required; MPS can be flaky with hooks)
        cammer = FrameGradCAM(self.model, device=self.gradcam_device, threshold=self.threshold)
        cams = cammer(tensors)                                       # (N, H, W)

        # Overlays
        for i, (rgb, cam) in enumerate(zip(crops, cams)):
            save_overlay(rgb, cam, out_dir / f"frame_{i:02d}_gradcam.png")

        # Timeline
        timeline_path = out_dir / "timeline.png"
        plot_timeline(
            result.per_frame_scores,
            threshold=self.threshold,
            out_path=timeline_path,
            title=f"{stem} — P(synthetic) per frame",
        )

        # Explainability score
        result.explainability_score = video_explainability_score(cams)
        result.heatmap_dir = str(out_dir)
        result.timeline_path = str(timeline_path)
        logger.info(
            "Explanation written to %s (explainability_score=%.4f)",
            out_dir, result.explainability_score,
        )
