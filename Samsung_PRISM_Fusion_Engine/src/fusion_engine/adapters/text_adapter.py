"""Adapter for the Text module — PrismEnsemble (stylometric + perplexity + rules).

Imports ``prism_text_detector`` directly from
``Samsung_PRISM_Text_Detection/PRISM/src/`` (added to ``sys.path`` at
import time — that package has a proper ``__init__.py`` and is meant to
be installed via its ``pyproject.toml``, so this is safe unlike the
Audio module's bare-import scripts).

As of this writing no trained ``model.pkl`` exists in the repo (see the
fusion-engine integration audit), so :class:`PrismEnsemble` runs in its
graceful-degradation mode: perplexity + rule-based linguistic patterns
only, no stylometric ML signal. This adapter mirrors
``website/backend/model_loader.py``'s `_find_model_dir` search so that
the moment a real `model.pkl` is trained and dropped into
``PRISM/artifacts/enhanced/`` (or the other two candidate dirs), this
adapter picks it up automatically with zero code changes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fusion_engine.schema import ModalityResult

_TEXT_MODULE_ROOT = (
    Path(__file__).resolve().parents[4] / "Samsung_PRISM_Text_Detection" / "PRISM"
)
_TEXT_SRC = _TEXT_MODULE_ROOT / "src"

if str(_TEXT_SRC) not in sys.path:
    sys.path.insert(0, str(_TEXT_SRC))


def _find_model_dir() -> Path:
    """Mirrors website/backend/model_loader.py::_find_model_dir."""
    candidates = [
        _TEXT_MODULE_ROOT / "artifacts" / "enhanced",
        _TEXT_MODULE_ROOT / "artifacts" / "hc3_text_baseline",
        _TEXT_MODULE_ROOT / "artifacts" / "hc3_text_baseline_smoke",
    ]
    for candidate in candidates:
        if (candidate / "model.pkl").exists():
            return candidate
    return candidates[0]  # no trained model — caller falls back to bare PrismEnsemble()


class TextAdapter:
    """Loads the Text module's ensemble once; call :meth:`predict` per text."""

    def __init__(self) -> None:
        from prism_text_detector.ensemble import PrismEnsemble

        model_dir = _find_model_dir()
        if (model_dir / "model.pkl").exists():
            self.ensemble = PrismEnsemble.load(model_dir)
            self.has_trained_model = True
        else:
            self.ensemble = PrismEnsemble()
            self.has_trained_model = False

    def predict(self, text: str) -> ModalityResult:
        """Run the text detector on a string of text.

        Args:
            text: The transcript / document text to analyse.

        Returns:
            A :class:`ModalityResult` with ``modality="text"``. On any
            failure, returns ``available=False`` rather than raising.
            Also returns ``available=False`` for empty/whitespace-only
            input, since ``PrismEnsemble`` special-cases that as
            "Text Too Short" rather than a real verdict.
        """
        t0 = time.perf_counter()
        try:
            if not text or not text.strip():
                return ModalityResult(
                    modality="text",
                    prob_synthetic=0.0,
                    confidence=0.0,
                    available=False,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    error="empty or whitespace-only text",
                )

            result = self.ensemble.analyze(text)
            prob_synthetic = float(result["synthetic_probability"])

            # PrismEnsemble's "Text Too Short" verdict means it couldn't
            # form a real opinion -- treat as unavailable, not confidently real.
            verdict = result.get("verdict", "")
            if verdict == "Text Too Short":
                return ModalityResult(
                    modality="text",
                    prob_synthetic=prob_synthetic,
                    confidence=0.0,
                    available=False,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    raw=result,
                    error="text too short for a reliable verdict",
                )

            # Map PrismEnsemble's qualitative confidence label to [0, 1].
            confidence_map = {
                "very high": 0.95,
                "high": 0.80,
                "medium": 0.60,
                "low": 0.35,
                "very low (insufficient data)": 0.10,
            }
            confidence = confidence_map.get(result.get("confidence", "low"), 0.5)

            latency_ms = (time.perf_counter() - t0) * 1000
            return ModalityResult(
                modality="text",
                prob_synthetic=prob_synthetic,
                confidence=confidence,
                available=True,
                latency_ms=latency_ms,
                raw={**result, "has_trained_stylometric_model": self.has_trained_model},
            )
        except Exception as exc:  # noqa: BLE001 - must not crash the fusion pipeline
            return ModalityResult(
                modality="text",
                prob_synthetic=0.0,
                confidence=0.0,
                available=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
