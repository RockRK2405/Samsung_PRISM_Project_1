"""Adapter for the Image module — EfficientNet-B0 fine-tuned on CIFAKE.

Wraps the checkpoint at
``Samsung_PRISM_Image_Detection/checkpoints/image_detector_v1.pth``
(verified: strict `load_state_dict` succeeds against a fresh
`torchvision.models.efficientnet_b0` with a 2-class head — see the
fusion-engine integration notes for the verification steps).

Source notebook: ``Samsung_PRISM_Image_Detection/Image_Detection_Model(V1).ipynb``
(training) and ``(V2).ipynb`` (inference reference — this adapter
reimplements V2's prediction cell as a real function).

Class convention (from V1's `ImageFolder` + `target_names=["FAKE","REAL"]`):
    index 0 = FAKE, index 1 = REAL — the OPPOSITE of the audio module's
    convention (index 0 = REAL, index 1 = FAKE). This adapter normalises
    both to the fusion engine's universal convention:
    ``prob_synthetic`` = P(the content is synthetic/fake), always.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

from fusion_engine.schema import ModalityResult

# Matches V1/V2's test-time transform exactly — no augmentation at inference.
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# V1's ImageFolder ordering: alphabetical -> FAKE=0, REAL=1.
_FAKE_INDEX = 0
_REAL_INDEX = 1

_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[4]
    / "Samsung_PRISM_Image_Detection"
    / "checkpoints"
    / "image_detector_v1.pth"
)


class ImageAdapter:
    """Loads the Image module's checkpoint once; call :meth:`predict` per image."""

    def __init__(self, checkpoint_path: Path | str = _DEFAULT_CHECKPOINT, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Image checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device)
        self.model = efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()

    def predict(self, image_path: str | Path) -> ModalityResult:
        """Run the image detector on a single image file.

        Args:
            image_path: Path to a JPEG/PNG image.

        Returns:
            A :class:`ModalityResult` with ``modality="image"``. On any
            failure (bad file, decode error), returns
            ``available=False`` with the exception message in ``error``
            rather than raising — the fusion engine must be able to
            proceed with the other three modalities.
        """
        t0 = time.perf_counter()
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = _TRANSFORM(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)[0]

            prob_fake = float(probs[_FAKE_INDEX].item())
            prob_real = float(probs[_REAL_INDEX].item())
            confidence = max(prob_fake, prob_real)  # distance from the 50/50 boundary

            latency_ms = (time.perf_counter() - t0) * 1000
            return ModalityResult(
                modality="image",
                prob_synthetic=prob_fake,
                confidence=confidence,
                available=True,
                latency_ms=latency_ms,
                raw={
                    "image_path": str(image_path),
                    "prediction": "FAKE" if prob_fake >= prob_real else "REAL",
                    "prob_fake": prob_fake,
                    "prob_real": prob_real,
                    "checkpoint": str(self.checkpoint_path.name),
                    "backbone": "efficientnet_b0",
                    "dataset": "CIFAKE",
                },
            )
        except Exception as exc:  # noqa: BLE001 - must not crash the fusion pipeline
            return ModalityResult(
                modality="image",
                prob_synthetic=0.0,
                confidence=0.0,
                available=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
