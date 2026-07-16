"""Weighted score aggregation + cross-modal consistency checks.

Pure functions over :class:`ModalityResult` dicts — no model loading,
no I/O. Kept separate from ``engine.py`` (which owns the adapters) so
this logic is trivially unit-testable without any of the four teammates'
model dependencies installed.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from fusion_engine.schema import FusionResult, ModalityResult

_DEFAULT_WEIGHTS = {"audio": 0.32, "video": 0.30, "image": 0.20, "text": 0.18}
_DEFAULT_DECISION_THRESHOLD = 0.5
_DEFAULT_UNCERTAIN_FLOOR = 0.35
_DEFAULT_DISAGREEMENT_THRESHOLD = 0.5


def fuse(
    modality_results: dict[str, ModalityResult],
    weights: dict[str, float] | None = None,
    decision_threshold: float = _DEFAULT_DECISION_THRESHOLD,
    uncertain_confidence_floor: float = _DEFAULT_UNCERTAIN_FLOOR,
    disagreement_threshold: float = _DEFAULT_DISAGREEMENT_THRESHOLD,
) -> FusionResult:
    """Combine per-modality results into one verdict.

    Args:
        modality_results: One :class:`ModalityResult` per modality that
            was *attempted* (including unavailable ones — their weight
            drops to zero automatically, they are not silently omitted).
        weights: Base per-modality weights, renormalised over whichever
            modalities are ``available``. Defaults to
            :data:`_DEFAULT_WEIGHTS` (see ``configs/fusion_weights.yaml``
            for the documented rationale).
        decision_threshold: ``fused_score >= this`` -> ``"synthetic"``.
        uncertain_confidence_floor: below this, verdict is forced to
            ``"uncertain"`` regardless of the score — too little
            reliable evidence to make a call either way.
        disagreement_threshold: minimum |prob difference| between two
            available modalities to raise a cross-modal disagreement flag.

    Returns:
        A populated :class:`FusionResult`.
    """
    weights = dict(weights or _DEFAULT_WEIGHTS)
    available = {m: r for m, r in modality_results.items() if r.available}

    weights_used: dict[str, float] = {m: 0.0 for m in modality_results}

    if not available:
        return FusionResult(
            verdict="uncertain",
            fused_score=0.5,
            fused_confidence=0.0,
            modality_results=modality_results,
            weights_used=weights_used,
            explanation="No modality could produce a result for this submission.",
        )

    # Effective weight per modality = base weight * that modality's own
    # confidence. A modality that ran but says "I'm not sure" should
    # count for less than one that ran and is confident.
    effective = {m: weights.get(m, 0.0) * r.confidence for m, r in available.items()}
    total_effective = sum(effective.values())

    if total_effective <= 1e-9:
        # Every available modality reported zero confidence — fall back
        # to a plain unweighted average so we still produce a number
        # rather than dividing by zero.
        fused_score = sum(r.prob_synthetic for r in available.values()) / len(available)
        fused_confidence = 0.0
    else:
        fused_score = sum(
            effective[m] * r.prob_synthetic for m, r in available.items()
        ) / total_effective
        # Coverage+confidence signal: how much of the *total possible*
        # weight did we actually get high-confidence answers for.
        total_possible = sum(weights.get(m, 0.0) for m in available) or 1.0
        fused_confidence = min(total_effective / total_possible, 1.0)

    for m in available:
        weights_used[m] = round(weights.get(m, 0.0), 4)

    # Disagreement penalty: if two available modalities strongly
    # disagree, that's real signal the fused number is less trustworthy
    # than the raw math suggests.
    flags = _cross_modal_flags(available, disagreement_threshold)
    if flags:
        fused_confidence *= 0.85  # modest, not punitive — flags are informational

    if fused_confidence < uncertain_confidence_floor:
        verdict = "uncertain"
    elif fused_score >= decision_threshold:
        verdict = "synthetic"
    else:
        verdict = "real"

    explanation = _build_explanation(available, modality_results, fused_score, fused_confidence, verdict, flags)

    return FusionResult(
        verdict=verdict,
        fused_score=round(fused_score, 4),
        fused_confidence=round(fused_confidence, 4),
        modality_results=modality_results,
        weights_used=weights_used,
        cross_modal_flags=flags,
        explanation=explanation,
    )


def _cross_modal_flags(
    available: dict[str, ModalityResult], disagreement_threshold: float
) -> list[str]:
    """Flag pairs of available modalities whose scores strongly disagree.

    This is a v1 heuristic stand-in for the worklet's "Cross-Modal
    Validation" step (architecture slide 6: "Audio vs. lip sync · Text
    transcript vs. speech"). A real lip-sync-mismatch or
    transcript-vs-speech check requires frame-level audio/video
    temporal alignment — out of scope for this pass. What this DOES
    catch: e.g. video module says 90% synthetic while audio module says
    5% synthetic on the same submission — worth a human's attention
    even without deeper cross-modal modelling.
    """
    flags: list[str] = []
    for (mod_a, res_a), (mod_b, res_b) in combinations(available.items(), 2):
        gap = abs(res_a.prob_synthetic - res_b.prob_synthetic)
        if gap >= disagreement_threshold:
            flags.append(
                f"{mod_a} ({res_a.prob_synthetic:.2f}) and {mod_b} ({res_b.prob_synthetic:.2f}) "
                f"disagree by {gap:.2f} — recommend manual review"
            )
    return flags


def _build_explanation(
    available: dict[str, ModalityResult],
    all_results: dict[str, ModalityResult],
    fused_score: float,
    fused_confidence: float,
    verdict: str,
    flags: list[str],
) -> str:
    parts: list[str] = []
    ran = ", ".join(sorted(available))
    missing = sorted(m for m in all_results if m not in available)
    parts.append(f"Modalities analysed: {ran or 'none'}.")
    if missing:
        parts.append(f"Not available: {', '.join(missing)}.")

    per_mod = "; ".join(
        f"{m}={r.prob_synthetic:.2f} (conf {r.confidence:.2f})" for m, r in sorted(available.items())
    )
    parts.append(f"Per-modality synthetic probability: {per_mod}.")
    parts.append(
        f"Fused score {fused_score:.2f} at confidence {fused_confidence:.2f} -> verdict: {verdict}."
    )
    if flags:
        parts.append("Cross-modal disagreement detected: " + " | ".join(flags))
    return " ".join(parts)
