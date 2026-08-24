"""Multi-target detector — routes between the face path and the general
(non-face objects/scenes) path. See ADR-006.

This is ADDITIVE: the existing :class:`~src.inference.predictor.VideoDetector`
and its ``predict()``/``api.predict()`` entrypoint (what the fusion engine
imports today) are UNCHANGED. This class is an opt-in upgrade for when
``checkpoints/general.pt`` exists (trained per ADR-006 / train_general.py).
Switching the fusion engine's default over to this class is a deliberate
follow-up decision once the general-path checkpoint is trained and
validated — not bundled into this change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.datasets.ff_dataset import _to_tensor
from src.inference.predictor import DetectionResult, VideoDetector
from src.models.baseline import BaselineDetector
from src.preprocessing import FrameExtractor, MTCNNFaceDetector
from src.utils import get_logger, select_device
from src.utils.paths import project_root

logger = get_logger(__name__)

#: Fraction of sampled frames that must contain a detected face for this
#: submission to be treated as face-centric content. Below this, the
#: face-path's signal would be noise (a handful of accidental detections),
#: so only the general path runs. Provisional -- not tuned on validation
#: data yet (ADR-006 "What this ADR does NOT decide yet").
_FACE_COVERAGE_THRESHOLD = 0.5


class MultiTargetVideoDetector:
    """Routes a video to the face-path model, the general-path model, or both.

    Attributes:
        face_detector: The existing, unchanged face-path ``VideoDetector``.
        general_model: The new general-content ``BaselineDetector``,
            loaded from ``checkpoints/general.pt``.
    """

    def __init__(
        self,
        face_checkpoint: Path | None = None,
        general_checkpoint: Path | None = None,
        face_coverage_threshold: float = _FACE_COVERAGE_THRESHOLD,
        device: str | torch.device | None = None,
        image_size: int = 224,
        frames_per_video: int = 32,
    ) -> None:
        root = project_root()
        face_checkpoint = face_checkpoint or (root / "checkpoints" / "best.pt")
        general_checkpoint = general_checkpoint or (root / "checkpoints" / "general.pt")

        self.face_detector = VideoDetector(checkpoint_path=face_checkpoint, threshold=0.5872, device=device)
        self.device = self.face_detector.device
        self.face_coverage_threshold = float(face_coverage_threshold)
        self._image_size = int(image_size)

        if not general_checkpoint.is_file():
            raise FileNotFoundError(
                f"General-path checkpoint not found: {general_checkpoint}. "
                "Train it first with scripts/train_general.py (see ADR-006)."
            )
        ckpt = torch.load(general_checkpoint, map_location=self.device, weights_only=False)
        model_cfg = ckpt.get("config", {}).get("model", {})
        self.general_model = BaselineDetector(
            backbone_name=model_cfg.get("backbone", "tf_efficientnet_b0.ns_jft_in1k"),
            pretrained=False,  # we're loading trained weights, not ImageNet init
            dropout=model_cfg.get("classifier", {}).get("dropout", 0.2),
        ).to(self.device)
        self.general_model.load_state_dict(ckpt["state_dict"])
        self.general_model.eval()

        self._extractor = FrameExtractor(num_frames=frames_per_video, strategy="uniform")
        # Reuse the SAME MTCNN detector instance the face-path already
        # built (avoids double-initialising MTCNN).
        self._face_locator: MTCNNFaceDetector = self.face_detector._detector

        logger.info(
            "MultiTargetVideoDetector ready: face_ckpt=%s general_ckpt=%s coverage_threshold=%.2f",
            face_checkpoint.name, general_checkpoint.name, self.face_coverage_threshold,
        )

    @torch.no_grad()
    def predict(self, video_path: str | Path) -> DetectionResult:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        frames_rgb = self._extractor.extract(video_path)
        crops, missed = self._face_locator.crop_video_frames(frames_rgb)
        face_coverage = 1.0 - (len(missed) / len(frames_rgb))

        general_score = self._score_general(frames_rgb)

        if face_coverage >= self.face_coverage_threshold and crops:
            face_result = self.face_detector.predict(video_path, produce_explanation=False)
            face_score = face_result.prob_synthetic
            # Either path flagging synthetic is enough -- see ADR-006
            # "Routing logic" for why max() rather than averaging.
            prob_synth = max(face_score, general_score)
            path_used = "face+general"
        else:
            face_score = None
            prob_synth = general_score
            path_used = "general_only"

        prediction = "synthetic" if prob_synth >= 0.5 else "real"
        confidence = float(min(abs(prob_synth - 0.5) / 0.5, 1.0))

        return DetectionResult(
            video_path=str(video_path),
            prediction=prediction,
            prob_synthetic=float(prob_synth),
            confidence=confidence,
            threshold=0.5,
            per_frame_scores=[],  # per-frame breakdown not unified across two models in v1
            num_frames_used=len(frames_rgb),
            meta={
                "router": "MultiTargetVideoDetector (ADR-006)",
                "path_used": path_used,
                "face_coverage": round(face_coverage, 4),
                "face_score": round(face_score, 4) if face_score is not None else None,
                "general_score": round(general_score, 4),
            },
        )

    def _score_general(self, frames_rgb: list[np.ndarray]) -> float:
        tensors = torch.stack(
            [_to_tensor(Image.fromarray(f).resize((self._image_size, self._image_size))) for f in frames_rgb],
            dim=0,
        ).to(self.device)
        logits = self.general_model(tensors)
        per_frame = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return float(np.mean(per_frame))
