"""Unified video detector — one entry point for every kind of synthetic video.

Any ingested video goes through two branches that cover different threats:

* **general** — ``VideoSpatialNet`` (EXP-004, ``best_model.pt``) on FULL
  frames, scored as one clip so the trained attention-pool temporal head is
  used. Always runs. Targets fully AI-generated video (text-to-video), and is
  the only branch that can assess faceless content (landscapes, objects).
* **face** — the face-forensics ``BaselineDetector`` (EXP-006,
  ``face_celebdf_identity.pt``) on MTCNN face crops. Runs only when enough
  sampled frames contain a face; otherwise it reports ``not_applicable``
  instead of raising (the old face-only path crashed on faceless video).

Fusion is an OR over the branches that ran: the video is ``synthetic`` if
any applicable branch crosses its own threshold. This favours recall —
a face swap and a text-to-video clip are different threats and either is
enough. The cost is that false-positive rates add up across branches; the
per-branch scores are always returned so that can be measured, not guessed.

Output is the same :class:`DetectionResult` (``modality="video"``) the rest
of the project consumes, with per-branch detail under ``meta["branches"]``.
That shape is the contract an audio module's result should mirror, so the
multimodal fusion stage can treat every modality the same way.

Preprocessing for each branch reproduces its training-time eval transform
exactly; changing it silently invalidates the measured numbers.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.inference.predictor import DetectionResult, VideoDetector
from src.preprocessing import FrameExtractor
from src.utils import get_logger, select_device

logger = get_logger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class _GeneralBranch:
    """EXP-004 ``VideoSpatialNet``: full frames -> one clip-level score."""

    def __init__(self, checkpoint: Path, device: torch.device, threshold: float) -> None:
        from src.models.video_spatial import build_video_model_from_config

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = copy.deepcopy(ckpt.get("config") or {})
        if "model" not in cfg:
            raise ValueError(f"{checkpoint} has no embedded config.model; cannot rebuild it.")
        # We are loading trained weights: never fetch ImageNet/HF weights.
        cfg["model"]["pretrained"] = False
        state = ckpt.get("model_state") or ckpt.get("state_dict")
        if state is None:
            raise ValueError(f"{checkpoint} has neither 'model_state' nor 'state_dict'.")

        self.model = build_video_model_from_config(cfg)
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.device = device
        self.threshold = float(threshold)
        self.checkpoint = Path(checkpoint)
        video = cfg.get("video", {}) or {}
        self.num_frames = int(video.get("num_frames", 16))
        self.image_size = int(video.get("image_size", 224))
        self.family = str(cfg["model"].get("family", "frame_baseline"))
        self.backbone = str(cfg["model"].get("backbone", "?"))
        # Exactly VideoManifestDataset's eval transform.
        self._tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    @torch.no_grad()
    def score(self, frames_rgb: list[np.ndarray]) -> tuple[float, list[float]]:
        """Return (clip score, per-frame diagnostic scores)."""
        clip = torch.stack([self._tf(Image.fromarray(f)) for f in frames_rgb], dim=0)  # (T,3,H,W)
        clip = clip.to(self.device)
        clip_prob = float(torch.softmax(self.model(clip.unsqueeze(0)), dim=-1)[0, 1])

        # Per-frame view: each frame as a 1-frame clip. The model was trained
        # on 16-frame clips, so these are a diagnostic timeline only -- the
        # verdict always comes from the clip score above.
        per_frame: list[float] = []
        timeline_source = "none"
        if self.family in {"frame_baseline", "frame_temporal", "spatial"}:
            probs = torch.softmax(self.model(clip.unsqueeze(1)), dim=-1)[:, 1]
            per_frame = [float(x) for x in probs.cpu()]
        return clip_prob, per_frame


class UnifiedVideoDetector:
    """General AI-video + face-swap detection, fused into one verdict."""

    def __init__(
        self,
        general_checkpoint: Path | None,
        face_checkpoint: Path | None,
        general_threshold: float = 0.5,
        face_threshold: float = 0.5,
        face_coverage_threshold: float = 0.5,
        device: str | torch.device | None = None,
        image_size: int = 224,
    ) -> None:
        self.device = select_device(device) if isinstance(device, str) or device is None else device
        self.face_coverage_threshold = float(face_coverage_threshold)

        self.general: _GeneralBranch | None = None
        if general_checkpoint is not None and Path(general_checkpoint).is_file():
            self.general = _GeneralBranch(Path(general_checkpoint), self.device, general_threshold)

        self.face: VideoDetector | None = None
        if face_checkpoint is not None and Path(face_checkpoint).is_file():
            self.face = VideoDetector(
                checkpoint_path=Path(face_checkpoint),
                threshold=face_threshold,
                image_size=image_size,
                device=self.device,
            )

        if self.general is None and self.face is None:
            raise FileNotFoundError(
                f"No usable checkpoint: general={general_checkpoint} face={face_checkpoint}"
            )

        n = self.general.num_frames if self.general else 16
        self._extractor = FrameExtractor(num_frames=n, strategy="uniform")
        logger.info(
            "UnifiedVideoDetector ready: general=%s face=%s device=%s",
            self.general.checkpoint.name if self.general else "OFF",
            self.face.checkpoint_path.name if self.face else "OFF",
            self.device,
        )

    def predict(
        self,
        video_path: str | Path,
        produce_explanation: bool = False,
        output_dir: str | Path | None = None,
    ) -> DetectionResult:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        frames = self._extractor.extract(video_path)  # one decode, shared by both branches
        branches: dict[str, dict[str, Any]] = {}
        per_frame: list[float] = []

        # --- general branch: always runs, works with or without faces
        if self.general is not None:
            g_prob, g_frames = self.general.score(frames)
            branches["general"] = {
                "status": "ran",
                "prob_synthetic": round(g_prob, 4),
                "threshold": self.general.threshold,
                "fired": g_prob >= self.general.threshold,
                "model": f"VideoSpatialNet ({self.general.backbone})",
                "checkpoint": self.general.checkpoint.name,
                "detects": "fully AI-generated video (text-to-video), any content",
            }
            per_frame = g_frames
            timeline_source = "general (1-frame diagnostic; verdict uses the 16-frame clip score)"
        else:
            branches["general"] = {"status": "disabled"}

        # --- face branch: only when faces are actually present
        crops: list[np.ndarray] = []
        tensors = None
        if self.face is not None:
            crops, missed = self.face._detector.crop_video_frames(frames)
            coverage = 1.0 - len(missed) / max(len(frames), 1)
            if crops and coverage >= self.face_coverage_threshold:
                from src.datasets.ff_dataset import _to_tensor

                tensors = torch.stack([_to_tensor(Image.fromarray(c)) for c in crops], dim=0)
                f_scores = self.face._score_frames(tensors)
                f_prob = float(np.mean(f_scores))
                branches["face"] = {
                    "status": "ran",
                    "prob_synthetic": round(f_prob, 4),
                    "threshold": self.face.threshold,
                    "fired": f_prob >= self.face.threshold,
                    "face_coverage": round(coverage, 4),
                    "model": f"BaselineDetector ({self.face.model_cfg.get('backbone')})",
                    "checkpoint": self.face.checkpoint_path.name,
                    "detects": "face-swap deepfakes",
                }
                per_frame = [float(x) for x in f_scores]  # face timeline is the trained one
                timeline_source = "face branch (per face crop)"
            else:
                branches["face"] = {
                    "status": "not_applicable",
                    "face_coverage": round(coverage, 4),
                    "reason": f"faces in {coverage:.0%} of frames (< {self.face_coverage_threshold:.0%})",
                }
        else:
            branches["face"] = {"status": "disabled"}

        ran = {k: v for k, v in branches.items() if v.get("status") == "ran"}
        if not ran:
            # Only possible with the general branch disabled and no faces found.
            raise RuntimeError(
                "No branch could assess this video: the general AI-video model is not "
                "loaded and the face branch found no faces. Configure general_checkpoint."
            )
        fired = [k for k, v in ran.items() if v["fired"]]
        prob = max(v["prob_synthetic"] for v in ran.values())
        prediction = "synthetic" if fired else "real"
        threshold = (ran[fired[0]] if fired else max(ran.values(), key=lambda v: v["prob_synthetic"]))["threshold"]
        margin = max(threshold, 1.0 - threshold)
        confidence = float(min(abs(prob - threshold) / margin, 1.0))

        if fired:
            reason = " + ".join(f"{k} branch ({ran[k]['prob_synthetic']:.2f})" for k in fired)
            reason = f"flagged by {reason}"
        else:
            reason = "no branch crossed its threshold (" + ", ".join(
                f"{k} {v['prob_synthetic']:.2f}" for k, v in ran.items()) + ")"

        result = DetectionResult(
            video_path=str(video_path),
            prediction=prediction,
            prob_synthetic=prob,
            confidence=confidence,
            threshold=threshold,
            per_frame_scores=per_frame,
            num_frames_used=len(frames),
            meta={
                "pipeline": "UnifiedVideoDetector",
                "branches": branches,
                "fired": fired,
                "verdict_reason": reason,
                "timeline_source": timeline_source,
                "fusion": "OR over applicable branches (recall-first)",
            },
        )

        # GradCAM is only defined for the face branch's CNN.
        if produce_explanation and tensors is not None and self.face is not None:
            try:
                self.face._attach_explanation(result, crops, tensors, output_dir)
            except Exception as exc:  # noqa: BLE001 - an explanation must never cost the verdict
                logger.warning("Explanation failed (%s: %s); verdict kept.", type(exc).__name__, exc)
        return result
