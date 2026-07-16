"""Multi-signal ensemble combiner for AI text detection.

Combines three independent detection signals:
    Signal 1: TF-IDF + Stylometric features + Classifier
    Signal 2: DistilGPT-2 Perplexity + Surprisal diversity
    Signal 3: Linguistic pattern detection (rule-based)

The ensemble produces a calibrated probability and an explainable verdict.
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from prism_text_detector.advanced_features import (
    AdvancedFeaturesTransformer,
    extract_advanced_features,
)
from prism_text_detector.explainability import (
    build_explanation,
    confidence_level,
    highlight_sentences,
    verdict_from_probability,
)
from prism_text_detector.linguistic_patterns import analyze_patterns


# ---------------------------------------------------------------------------
# Perplexity import (optional — works without torch)
# ---------------------------------------------------------------------------
try:
    from prism_text_detector.perplexity import (
        PerplexityAnalyzer,
        analyze_text_perplexity,
        is_available as perplexity_available,
    )
except ImportError:
    perplexity_available = lambda: False  # noqa: E731
    analyze_text_perplexity = None
    PerplexityAnalyzer = None


# ---------------------------------------------------------------------------
# Default ensemble weights (used when no trained combiner exists)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "stylometric": 0.45,
    "perplexity": 0.30,
    "linguistic": 0.25,
}

# Weights when perplexity is unavailable
FALLBACK_WEIGHTS = {
    "stylometric": 0.65,
    "perplexity": 0.0,
    "linguistic": 0.35,
}


class PrismEnsemble:
    """Multi-signal ensemble for AI text detection with explainability.

    Usage::

        ensemble = PrismEnsemble.load("artifacts/enhanced")
        result = ensemble.analyze("Some text to check...")
        print(result["verdict"], result["confidence"])
    """

    def __init__(
        self,
        stylometric_model: Any | None = None,
        weights: dict[str, float] | None = None,
        threshold: float = 0.5,
        model_info: dict[str, Any] | None = None,
    ) -> None:
        self.stylometric_model = stylometric_model
        self.threshold = threshold
        self.model_info = model_info or {}
        self._perplexity_analyzer: PerplexityAnalyzer | None = None

        # Decide weights based on perplexity availability
        if weights is not None:
            self.weights = weights
        elif perplexity_available():
            self.weights = DEFAULT_WEIGHTS.copy()
        else:
            self.weights = FALLBACK_WEIGHTS.copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        """Save ensemble artifacts to *directory*."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        if self.stylometric_model is not None:
            with (out / "model.pkl").open("wb") as fh:
                pickle.dump(self.stylometric_model, fh)

        meta = {
            "weights": self.weights,
            "threshold": self.threshold,
            "model_info": self.model_info,
            "perplexity_available": perplexity_available(),
        }
        (out / "ensemble_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "PrismEnsemble":
        """Load a saved ensemble from *directory*."""
        d = Path(directory)

        model = None
        model_path = d / "model.pkl"
        if model_path.exists():
            with model_path.open("rb") as fh:
                model = pickle.load(fh)

        meta_path = d / "ensemble_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            meta = {}

        return cls(
            stylometric_model=model,
            weights=meta.get("weights"),
            threshold=meta.get("threshold", 0.5),
            model_info=meta.get("model_info", {}),
        )

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------
    def analyze(self, text: str) -> dict[str, Any]:
        """Analyse a single text and return an explainable result dict.

        Keys in the returned dict:
            verdict, confidence, authenticity_score, synthetic_probability,
            signals, linguistic_flags, explanation, feature_highlights, sentences
        """
        if not text or not text.strip():
            return _empty_result()

        # --- Signal 1: Stylometric model ---
        sig1_score = 0.5
        if self.stylometric_model is not None:
            try:
                prob = self.stylometric_model.predict_proba([text])[:, 1]
                sig1_score = float(prob[0])
            except Exception:
                sig1_score = 0.5

        # --- Signal 2: Perplexity ---
        sig2_score = 0.5
        sig2_available = False
        ppl_data: dict[str, Any] = {}
        if perplexity_available() and analyze_text_perplexity is not None:
            try:
                ppl_data = analyze_text_perplexity(text)
                sig2_available = ppl_data.get("available", False)
                if sig2_available:
                    sig2_score = ppl_data.get("ai_score", 0.5)
            except Exception:
                sig2_available = False

        # --- Signal 3: Linguistic patterns ---
        pattern_result = analyze_patterns(text)
        sig3_score = pattern_result.get("overall_score", 0.0)

        # --- Combine ---
        weights = self.weights.copy()
        if not sig2_available:
            # Re-distribute perplexity weight
            ppl_w = weights.get("perplexity", 0.0)
            weights["perplexity"] = 0.0
            remaining = weights.get("stylometric", 0.45) + weights.get("linguistic", 0.25)
            if remaining > 0:
                scale = (remaining + ppl_w) / remaining
                weights["stylometric"] = weights.get("stylometric", 0.45) * scale
                weights["linguistic"] = weights.get("linguistic", 0.25) * scale

        synthetic_probability = (
            weights["stylometric"] * sig1_score
            + weights["perplexity"] * sig2_score
            + weights["linguistic"] * sig3_score
        )
        synthetic_probability = float(np.clip(synthetic_probability, 0.0, 1.0))

        authenticity_score = round(100.0 * (1.0 - synthetic_probability), 1)
        import re
        word_count = len(re.findall(r"\w+", text))
        
        verdict = verdict_from_probability(synthetic_probability, word_count)
        conf = confidence_level(synthetic_probability, word_count)

        # --- Feature highlights for radar chart ---
        adv_features = _safe_advanced_features(text)
        feature_highlights = _build_feature_highlights(adv_features, ppl_data)

        # --- Sentence-level analysis ---
        sentences = highlight_sentences(text, self.stylometric_model)

        # --- Explanation ---
        signals_info = {
            "stylometric": {"score": round(sig1_score, 4), "label": _score_label(sig1_score)},
            "perplexity": {
                "score": round(sig2_score, 4),
                "label": _ppl_label(sig2_score, ppl_data),
                "available": sig2_available,
            },
            "linguistic": {"score": round(sig3_score, 4), "label": _pattern_label(sig3_score)},
        }

        explanation = build_explanation(
            sig1_score=sig1_score,
            sig2_score=sig2_score,
            sig2_available=sig2_available,
            sig3_score=sig3_score,
            perplexity_value=ppl_data.get("perplexity"),
            flags=pattern_result.get("flags", []),
            synthetic_probability=synthetic_probability,
            word_count=word_count,
        )

        return {
            "verdict": verdict,
            "confidence": conf,
            "authenticity_score": authenticity_score,
            "synthetic_probability": round(synthetic_probability, 4),
            "signals": signals_info,
            "linguistic_flags": pattern_result.get("flags", []),
            "explanation": explanation,
            "feature_highlights": feature_highlights,
            "sentences": sentences,
        }

    def analyze_batch(self, texts: list[str]) -> dict[str, Any]:
        """Analyse multiple texts. Returns {results: [...], summary: {...}}."""
        results = []
        counters = {"human": 0, "ai": 0, "uncertain": 0}
        for text in texts:
            r = self.analyze(text)
            results.append({
                "text": text[:200] + ("..." if len(text) > 200 else ""),
                "verdict": r["verdict"],
                "confidence": r["confidence"],
                "authenticity_score": r["authenticity_score"],
                "synthetic_probability": r["synthetic_probability"],
            })
            v = r["verdict"].lower()
            if "human" in v:
                counters["human"] += 1
            elif "ai" in v:
                counters["ai"] += 1
            else:
                counters["uncertain"] += 1

        return {
            "results": results,
            "summary": {
                "total": len(results),
                **counters,
            },
        }


# ======================================================================
# Helpers
# ======================================================================

def _empty_result() -> dict[str, Any]:
    return {
        "verdict": "Invalid Input",
        "confidence": "low",
        "authenticity_score": 0.0,
        "synthetic_probability": 0.5,
        "signals": {
            "stylometric": {"score": 0.5, "label": "N/A"},
            "perplexity": {"score": 0.5, "label": "N/A", "available": False},
            "linguistic": {"score": 0.0, "label": "N/A"},
        },
        "linguistic_flags": [],
        "explanation": "The provided text is empty or too short to analyse.",
        "feature_highlights": {},
        "sentences": [],
    }


def _score_label(score: float) -> str:
    if score >= 0.75:
        return "Strong AI signal"
    if score >= 0.55:
        return "Moderate AI signal"
    if score <= 0.25:
        return "Strong human signal"
    if score <= 0.45:
        return "Moderate human signal"
    return "Inconclusive"


def _ppl_label(score: float, ppl_data: dict) -> str:
    ppl = ppl_data.get("perplexity")
    if ppl is None:
        return "Unavailable"
    if score >= 0.65:
        return f"Low perplexity ({ppl:.0f}) — AI-like"
    if score <= 0.35:
        return f"High perplexity ({ppl:.0f}) — human-like"
    return f"Medium perplexity ({ppl:.0f})"


def _pattern_label(score: float) -> str:
    if score >= 0.5:
        return "Many AI patterns detected"
    if score >= 0.2:
        return "Some AI patterns"
    return "Few AI patterns"


def _safe_advanced_features(text: str) -> dict[str, float]:
    """Extract advanced features, returning empty dict on failure."""
    try:
        from prism_text_detector.advanced_features import FEATURE_NAMES

        values = extract_advanced_features(text)
        return dict(zip(FEATURE_NAMES, values))
    except Exception:
        return {}


# Human / AI reference averages (approximate, for radar chart context)
_REFERENCE = {
    "burstiness":        {"human_avg": 0.55, "ai_avg": 0.22},
    "lexical_diversity":  {"human_avg": 0.68, "ai_avg": 0.52},
    "hapax_legomena_ratio": {"human_avg": 0.42, "ai_avg": 0.28},
    "perplexity":         {"human_avg": 130.0, "ai_avg": 42.0},
    "sentence_complexity": {"human_avg": 0.58, "ai_avg": 0.32},
    "compression_ratio":   {"human_avg": 0.48, "ai_avg": 0.36},
}


def _build_feature_highlights(
    adv: dict[str, float], ppl_data: dict
) -> dict[str, dict]:
    highlights: dict[str, dict] = {}

    mapping = {
        "burstiness": "burstiness_coefficient",
        "lexical_diversity": "lexical_diversity",
        "hapax_ratio": "hapax_legomena_ratio",
        "compression_ratio": "compression_ratio",
    }

    for display_name, feat_name in mapping.items():
        val = adv.get(feat_name)
        if val is not None:
            ref = _REFERENCE.get(display_name, _REFERENCE.get(feat_name, {}))
            highlights[display_name] = {
                "value": round(val, 3),
                "human_avg": ref.get("human_avg", 0.5),
                "ai_avg": ref.get("ai_avg", 0.3),
            }

    # Sentence complexity from std_sentence_length + burstiness
    std_sl = adv.get("std_sentence_length")
    if std_sl is not None:
        highlights["sentence_complexity"] = {
            "value": round(min(std_sl / 15.0, 1.0), 3),
            "human_avg": 0.58,
            "ai_avg": 0.32,
        }

    # Perplexity
    ppl_val = ppl_data.get("perplexity")
    if ppl_val is not None:
        highlights["perplexity"] = {
            "value": round(ppl_val, 1),
            "human_avg": 130.0,
            "ai_avg": 42.0,
        }

    return highlights
