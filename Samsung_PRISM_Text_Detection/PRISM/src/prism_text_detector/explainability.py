"""Explainability module for PRISM AI text detection.

Generates human-readable verdicts, confidence levels, natural language
explanations, and sentence-level highlighting for detection results.
"""
from __future__ import annotations

import re
from typing import Any


# ======================================================================
# Verdict & Confidence
# ======================================================================

def verdict_from_probability(probability: float, word_count: int = 100) -> str:
    """Return a human-readable verdict string."""
    if word_count < 50:
        return "Text Too Short"
    
    if probability >= 0.75:
        return "Likely AI-Generated"
    if probability >= 0.55:
        return "Possibly AI-Generated"
    if probability <= 0.25:
        return "Likely Human"
    if probability <= 0.45:
        return "Likely Human"
    return "Uncertain — Needs Review"


def confidence_level(probability: float, word_count: int = 100) -> str:
    """Return confidence level based on distance from 0.5."""
    if word_count < 50:
        return "very low (insufficient data)"
        
    distance = abs(probability - 0.5)
    if distance >= 0.40:
        return "very high"
    if distance >= 0.25:
        return "high"
    if distance >= 0.12:
        return "medium"
    return "low"


# ======================================================================
# Explanation Builder
# ======================================================================

def build_explanation(
    sig1_score: float,
    sig2_score: float,
    sig2_available: bool,
    sig3_score: float,
    perplexity_value: float | None,
    flags: list[str],
    synthetic_probability: float,
    word_count: int = 100,
) -> str:
    """Build a natural-language explanation from the three signal scores.

    Returns a coherent paragraph explaining *why* the verdict was given.
    """
    if word_count < 50:
        return (
            f"The provided text ({word_count} words) is too short for reliable statistical analysis. "
            f"Stylometric features and perplexity require at least 50-100 words of context to generate "
            f"meaningful results. Please provide a longer sample (e.g. a full paragraph) for an accurate AI detection verdict."
        )

    parts: list[str] = []

    # --- Stylometric signal ---
    if sig1_score >= 0.80:
        parts.append(
            f"The stylometric analysis strongly indicates AI generation "
            f"(score: {sig1_score:.2f}). The text's vocabulary, sentence "
            f"structure, and statistical patterns closely match known AI output."
        )
    elif sig1_score >= 0.60:
        parts.append(
            f"The stylometric analysis moderately suggests AI generation "
            f"(score: {sig1_score:.2f}). Some features like sentence rhythm "
            f"and vocabulary choices lean toward AI patterns."
        )
    elif sig1_score <= 0.20:
        parts.append(
            f"The stylometric analysis strongly supports human authorship "
            f"(score: {sig1_score:.2f}). The text shows natural variation in "
            f"sentence structure and vocabulary typical of human writing."
        )
    elif sig1_score <= 0.40:
        parts.append(
            f"The stylometric analysis moderately supports human authorship "
            f"(score: {sig1_score:.2f}). Writing patterns are mostly "
            f"consistent with human composition."
        )
    else:
        parts.append(
            f"The stylometric analysis is inconclusive (score: {sig1_score:.2f}). "
            f"The text shows a mix of human and AI-like patterns."
        )

    # --- Perplexity signal ---
    if sig2_available and perplexity_value is not None:
        if sig2_score >= 0.65:
            parts.append(
                f"Perplexity analysis shows the text is highly predictable "
                f"(perplexity: {perplexity_value:.0f}), which is characteristic "
                f"of AI-generated content. Language models find this text "
                f"unsurprising."
            )
        elif sig2_score <= 0.35:
            parts.append(
                f"Perplexity analysis shows the text has high unpredictability "
                f"(perplexity: {perplexity_value:.0f}), which supports human "
                f"authorship. The text contains surprising word choices and "
                f"varied expression."
            )
        else:
            parts.append(
                f"Perplexity analysis gives a mid-range reading "
                f"(perplexity: {perplexity_value:.0f}), which is not "
                f"conclusive on its own."
            )
    else:
        parts.append(
            "Perplexity analysis was not available for this evaluation."
        )

    # --- Linguistic patterns ---
    if sig3_score >= 0.4:
        parts.append(
            f"Multiple AI-typical writing patterns were detected "
            f"(pattern score: {sig3_score:.2f})."
        )
    elif sig3_score >= 0.15:
        parts.append(
            f"A few AI-typical patterns were found "
            f"(pattern score: {sig3_score:.2f})."
        )
    else:
        parts.append("No significant AI-typical writing patterns were found.")

    # --- Flags ---
    if flags:
        flag_str = ", ".join(flags[:5])
        parts.append(f"Specific flags: {flag_str}.")

    # --- Final confidence ---
    conf = confidence_level(synthetic_probability)
    if synthetic_probability >= 0.55:
        parts.append(
            f"Overall, the combined evidence suggests this text is AI-generated "
            f"with {conf} confidence (combined score: {synthetic_probability:.2f})."
        )
    elif synthetic_probability <= 0.45:
        parts.append(
            f"Overall, the combined evidence suggests this text is human-written "
            f"with {conf} confidence (combined score: {synthetic_probability:.2f})."
        )
    else:
        parts.append(
            f"Overall, the evidence is mixed and the text falls in the "
            f"uncertain zone (combined score: {synthetic_probability:.2f}). "
            f"Manual review is recommended."
        )

    return " ".join(parts)


# ======================================================================
# Sentence-Level Highlighting
# ======================================================================

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def highlight_sentences(
    text: str, model: Any | None = None, max_sentences: int = 30
) -> list[dict[str, Any]]:
    """Split text into sentences and score each for AI likelihood.

    If *model* is provided (an sklearn pipeline with ``predict_proba``),
    each sentence is scored individually. Otherwise a simple heuristic
    is used based on sentence length uniformity.

    Returns a list of ``{"text": str, "ai_score": float}`` dicts.
    """
    if not text or not text.strip():
        return []

    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return [{"text": text.strip(), "ai_score": 0.5}]

    sentences = sentences[:max_sentences]

    if model is not None:
        return _model_sentence_scores(sentences, model)

    return _heuristic_sentence_scores(sentences)


def _model_sentence_scores(
    sentences: list[str], model: Any
) -> list[dict[str, Any]]:
    """Score each sentence using the stylometric model."""
    results: list[dict[str, Any]] = []

    # Score the full text and each leave-one-out variant for relative importance
    try:
        full_text = " ".join(sentences)
        full_prob = float(model.predict_proba([full_text])[:, 1][0])
    except Exception:
        return [{"text": s, "ai_score": 0.5} for s in sentences]

    for i, sentence in enumerate(sentences):
        if len(sentences) >= 3:
            # Leave-one-out: how does removing this sentence change the score?
            reduced = " ".join(s for j, s in enumerate(sentences) if j != i)
            try:
                reduced_prob = float(model.predict_proba([reduced])[:, 1][0])
                # If removing the sentence lowers the AI score, it was contributing to AI-ness
                contribution = full_prob - reduced_prob
                # Normalise to 0-1 range
                ai_score = float(max(0.0, min(1.0, 0.5 + contribution * 3)))
            except Exception:
                ai_score = full_prob
        else:
            # Too few sentences for leave-one-out
            try:
                ai_score = float(model.predict_proba([sentence])[:, 1][0])
            except Exception:
                ai_score = 0.5

        results.append({"text": sentence, "ai_score": round(ai_score, 3)})

    return results


def _heuristic_sentence_scores(
    sentences: list[str],
) -> list[dict[str, Any]]:
    """Score sentences using simple heuristics (no model needed)."""
    import numpy as np

    _WORD_RE = re.compile(r"[A-Za-z0-9']+")
    _AI_PHRASES = {
        "in conclusion", "moreover", "furthermore", "additionally",
        "on the other hand", "plays a vital role", "it is important",
        "it should be noted", "in today's world", "in the modern era",
    }

    results: list[dict[str, Any]] = []
    lengths = [len(_WORD_RE.findall(s)) for s in sentences]
    mean_len = float(np.mean(lengths)) if lengths else 10.0

    for sentence, wc in zip(sentences, lengths):
        score = 0.3  # baseline

        lower = sentence.lower()
        # Check for AI transition phrases
        for phrase in _AI_PHRASES:
            if phrase in lower:
                score += 0.2
                break

        # Very uniform sentence length is AI-like
        if mean_len > 0 and abs(wc - mean_len) / max(mean_len, 1) < 0.15:
            score += 0.1

        # Formal structure
        if lower.startswith(("this ", "these ", "those ", "such ", "the ")):
            score += 0.05

        results.append({
            "text": sentence,
            "ai_score": round(min(score, 1.0), 3),
        })

    return results
