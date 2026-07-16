"""Top-level fusion engine — the one class the QC dashboard talks to.

Lazily loads whichever of the four modality adapters it can (a missing
checkpoint or dependency for one modality does not prevent the others
from working), then combines whatever inputs are supplied for a given
submission into one :class:`~fusion_engine.schema.FusionResult`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fusion_engine.fusion import fuse
from fusion_engine.schema import FusionResult, ModalityResult

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fusion_weights.yaml"


class FusionEngine:
    """Loads available modality adapters and fuses their outputs.

    Attributes:
        adapters: mapping of modality name -> loaded adapter instance.
            Only contains modalities that loaded successfully; check
            :attr:`load_errors` for what didn't.
        load_errors: mapping of modality name -> error string, for any
            adapter that failed to initialise (missing checkpoint,
            missing dependency, etc.). The engine still works with the
            remaining modalities.
    """

    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG) -> None:
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.weights: dict[str, float] = cfg["weights"]
        self.decision_threshold: float = cfg["decision_threshold"]
        self.uncertain_confidence_floor: float = cfg["uncertain_confidence_floor"]
        self.disagreement_threshold: float = cfg["cross_modal_disagreement_threshold"]

        self.adapters: dict[str, Any] = {}
        self.load_errors: dict[str, str] = {}
        self._load_adapters()

    def _load_adapters(self) -> None:
        loaders = {
            "image": self._load_image,
            "audio": self._load_audio,
            "text": self._load_text,
            "video": self._load_video,
        }
        for name, loader in loaders.items():
            try:
                self.adapters[name] = loader()
            except Exception as exc:  # noqa: BLE001 - one bad adapter must not sink the rest
                self.load_errors[name] = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _load_image():
        from fusion_engine.adapters.image_adapter import ImageAdapter

        return ImageAdapter()

    @staticmethod
    def _load_audio():
        from fusion_engine.adapters.audio_adapter import AudioAdapter

        return AudioAdapter()

    @staticmethod
    def _load_text():
        from fusion_engine.adapters.text_adapter import TextAdapter

        return TextAdapter()

    @staticmethod
    def _load_video():
        from fusion_engine.adapters.video_adapter import VideoAdapter

        return VideoAdapter()

    def analyze(
        self,
        image_path: str | Path | None = None,
        audio_path: str | Path | None = None,
        text: str | None = None,
        video_path: str | Path | None = None,
    ) -> FusionResult:
        """Run whichever modalities have an input, then fuse the results.

        Pass only the inputs you have for this submission — a
        text-only submission just passes ``text=...`` and the other
        three modalities are recorded as unavailable (not run), which
        the fusion math already accounts for.

        Args:
            image_path: Path to an image file, if this submission has one.
            audio_path: Path to an audio file, if this submission has one.
            text: Text content, if this submission has one.
            video_path: Path to a video file, if this submission has one.

        Returns:
            A :class:`FusionResult` describing the combined verdict.
        """
        inputs = {
            "image": image_path,
            "audio": audio_path,
            "text": text,
            "video": video_path,
        }

        results: dict[str, ModalityResult] = {}
        for modality, value in inputs.items():
            if value is None:
                results[modality] = ModalityResult(
                    modality=modality,
                    prob_synthetic=0.0,
                    confidence=0.0,
                    available=False,
                    error="no input provided for this modality",
                )
                continue

            adapter = self.adapters.get(modality)
            if adapter is None:
                results[modality] = ModalityResult(
                    modality=modality,
                    prob_synthetic=0.0,
                    confidence=0.0,
                    available=False,
                    error=self.load_errors.get(modality, "adapter failed to load"),
                )
                continue

            results[modality] = adapter.predict(value)

        return fuse(
            results,
            weights=self.weights,
            decision_threshold=self.decision_threshold,
            uncertain_confidence_floor=self.uncertain_confidence_floor,
            disagreement_threshold=self.disagreement_threshold,
        )
