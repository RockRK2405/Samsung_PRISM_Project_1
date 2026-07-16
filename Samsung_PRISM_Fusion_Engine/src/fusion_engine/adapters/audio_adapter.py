"""Adapter for the Audio module — lightweight CNN on log-Mel spectrograms.

Wraps the checkpoint at
``Samsung_PRISM_Audio_Detection/audio_authenticity_detector/outputs/best_audio_model.pt``.

Preprocessing constants and the `AudioCNN` architecture are reimplemented
here verbatim from
``Samsung_PRISM_Audio_Detection/audio_authenticity_detector/src/{preprocess,model}.py``
rather than imported across the module boundary — that source uses bare
imports (``from preprocess import ...``) that only resolve when its own
``src/`` directory is the working directory, which would require fragile
``sys.path`` surgery here. Reimplementing ~40 lines of pure-numpy/torch
logic is more robust than depending on another team's script-style layout.

Class convention: index 0 = REAL, index 1 = FAKE (confirmed in
``infer.py``: ``fake_prob = probs[1]``, ``real_prob = probs[0]``).
"""

from __future__ import annotations

import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

from fusion_engine.schema import ModalityResult

# --- Preprocessing constants, copied from audio_authenticity_detector/src/preprocess.py ---
_SAMPLE_RATE = 16000
_DURATION_SEC = 4.0
_N_SAMPLES = int(_SAMPLE_RATE * _DURATION_SEC)
_N_MELS = 128
_N_FFT = 1024
_HOP_LENGTH = 256
_FIXED_FRAMES = 251  # matches AudioSpectrogramDataset's default

_REAL_INDEX = 0
_FAKE_INDEX = 1

_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[4]
    / "Samsung_PRISM_Audio_Detection"
    / "audio_authenticity_detector"
    / "outputs"
    / "best_audio_model.pt"
)


class AudioCNN(nn.Module):
    """Verbatim copy of audio_authenticity_detector/src/model.py::AudioCNN.

    Must stay structurally identical to the source for `load_state_dict`
    to succeed — do not refactor without re-verifying against the
    checkpoint's key names.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def _load_fixed_length_audio(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=_SAMPLE_RATE, mono=True)
    if len(y) >= _N_SAMPLES:
        start = max(0, (len(y) - _N_SAMPLES) // 2)
        y = y[start : start + _N_SAMPLES]
    else:
        y = np.pad(y, (0, _N_SAMPLES - len(y)), mode="constant")
    return y


def _wav_to_logmel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=_SAMPLE_RATE, n_fft=_N_FFT, hop_length=_HOP_LENGTH, n_mels=_N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.astype(np.float32)


def _fix_time_dim(mel: np.ndarray, fixed_frames: int = _FIXED_FRAMES) -> np.ndarray:
    _n_mels, t = mel.shape
    if t >= fixed_frames:
        return mel[:, :fixed_frames]
    return np.pad(mel, ((0, 0), (0, fixed_frames - t)), mode="constant")


class AudioAdapter:
    """Loads the Audio module's checkpoint once; call :meth:`predict` per clip."""

    def __init__(self, checkpoint_path: Path | str = _DEFAULT_CHECKPOINT, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Audio checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device)
        self.model = AudioCNN().to(self.device)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, wav_path: str | Path) -> ModalityResult:
        """Run the audio detector on a single audio file.

        Args:
            wav_path: Path to a .wav/.flac/.mp3 audio file.

        Returns:
            A :class:`ModalityResult` with ``modality="audio"``. On any
            failure, returns ``available=False`` rather than raising.
        """
        t0 = time.perf_counter()
        try:
            y = _load_fixed_length_audio(str(wav_path))
            mel = _wav_to_logmel(y)
            mel = _fix_time_dim(mel)
            x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0]

            prob_fake = float(probs[_FAKE_INDEX].item())
            prob_real = float(probs[_REAL_INDEX].item())
            confidence = max(prob_fake, prob_real)

            latency_ms = (time.perf_counter() - t0) * 1000
            return ModalityResult(
                modality="audio",
                prob_synthetic=prob_fake,
                confidence=confidence,
                available=True,
                latency_ms=latency_ms,
                raw={
                    "file": str(wav_path),
                    "label": "FAKE" if prob_fake >= 0.5 else "REAL",
                    "audio_authenticity_score": round(prob_real, 4),
                    "synthetic_speech_probability": round(prob_fake, 4),
                    "checkpoint": self.checkpoint_path.name,
                },
            )
        except Exception as exc:  # noqa: BLE001 - must not crash the fusion pipeline
            return ModalityResult(
                modality="audio",
                prob_synthetic=0.0,
                confidence=0.0,
                available=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
