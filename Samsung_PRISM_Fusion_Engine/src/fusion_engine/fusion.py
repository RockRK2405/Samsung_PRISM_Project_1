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
# A modality only counts as "voting" (for conflict detection) if it is at
# least this confident. Below it, the modality is treated as abstaining --
# its disagreement with another modality is not a genuine cross-modal
# conflict, just one detector declining to commit. Introduced after the
# first benchmark showed a 100% flag rate: flags fired on every clip
# because low-confidence modalities (e.g. text at conf 0.50) were being
# counted as disagreeing. Gating on confidence makes a raised flag mean
# something a reviewer should actually act on.
_DEFAULT_CONFLICT_CONFIDENCE_GATE = 0.6


def fuse(
    modality_results: dict[str, ModalityResult],
    weights: dict[str, float] | None = None,
    decision_threshold: float = _DEFAULT_DECISION_THRESHOLD,
    uncertain_confidence_floor: float = _DEFAULT_UNCERTAIN_FLOOR,
    disagreement_threshold: float = _DEFAULT_DISAGREEMENT_THRESHOLD,
    conflict_confidence_gate: float = _DEFAULT_CONFLICT_CONFIDENCE_GATE,
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
        conflict_confidence_gate: both modalities in a disagreeing pair
            must report at least this confidence for the pair to count as
            a genuine conflict (see :data:`_DEFAULT_CONFLICT_CONFIDENCE_GATE`).

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

    # Disagreement penalty: if two *confident* available modalities
    # strongly disagree, that's real signal the fused number is less
    # trustworthy than the raw math suggests.
    flags = _cross_modal_flags(available, disagreement_threshold, conflict_confidence_gate)
    if flags:
        fused_confidence *= 0.85  # modest, not punitive — flags are informational

    # A confident cross-modal conflict that straddles the decision line
    # (one confident modality says synthetic, another confident one says
    # real) is exactly the case a single averaged scalar should not decide
    # on its own. Route it to human review rather than emitting a
    # low-confidence "real"/"synthetic" -- this is the cost-aware QC
    # behaviour the worklet's dashboard calls for, and it stops the engine
    # silently passing a fake as real just because two confident detectors
    # cancelled out. (Weighted averaging alone would call morgan_freeman
    # "real" at conf 0.73 despite image confidently flagging it synthetic.)
    straddling_conflict = _confident_conflict_straddles_threshold(
        available, conflict_confidence_gate, disagreement_threshold, decision_threshold
    )

    if fused_confidence < uncertain_confidence_floor or straddling_conflict:
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
    available: dict[str, ModalityResult],
    disagreement_threshold: float,
    conflict_confidence_gate: float,
) -> list[str]:
    """Flag pairs of *confident* available modalities whose scores disagree.

    This is a v1 heuristic stand-in for the worklet's "Cross-Modal
    Validation" step (architecture slide 6: "Audio vs. lip sync · Text
    transcript vs. speech"). A real lip-sync-mismatch or
    transcript-vs-speech check requires frame-level audio/video
    temporal alignment — out of scope for this pass. What this DOES
    catch: e.g. video module confidently says 90% synthetic while audio
    module confidently says 5% synthetic on the same submission — worth a
    human's attention even without deeper cross-modal modelling.

    Both modalities in a pair must clear ``conflict_confidence_gate`` —
    a disagreement where one side is merely abstaining (low confidence)
    is not a genuine conflict and is not flagged. This is what keeps the
    flag meaningful rather than firing on essentially every submission.
    """
    flags: list[str] = []
    for (mod_a, res_a), (mod_b, res_b) in combinations(available.items(), 2):
        if res_a.confidence < conflict_confidence_gate or res_b.confidence < conflict_confidence_gate:
            continue
        gap = abs(res_a.prob_synthetic - res_b.prob_synthetic)
        if gap >= disagreement_threshold:
            flags.append(
                f"{mod_a} ({res_a.prob_synthetic:.2f}) and {mod_b} ({res_b.prob_synthetic:.2f}) "
                f"disagree by {gap:.2f} — recommend manual review"
            )
    return flags


def _confident_conflict_straddles_threshold(
    available: dict[str, ModalityResult],
    conflict_confidence_gate: float,
    disagreement_threshold: float,
    decision_threshold: float,
) -> bool:
    """True if two confident modalities land on opposite sides of the line.

    "Opposite sides" means one confident modality's ``prob_synthetic`` is
    at or above ``decision_threshold`` (it says synthetic) while another
    confident modality is below it (it says real), and the two differ by
    at least ``disagreement_threshold``. In that situation a single
    averaged scalar is not a trustworthy verdict — the submission should
    go to human review (verdict ``"uncertain"``) instead.
    """
    for (mod_a, res_a), (mod_b, res_b) in combinations(available.items(), 2):
        if res_a.confidence < conflict_confidence_gate or res_b.confidence < conflict_confidence_gate:
            continue
        if abs(res_a.prob_synthetic - res_b.prob_synthetic) < disagreement_threshold:
            continue
        a_synth = res_a.prob_synthetic >= decision_threshold
        b_synth = res_b.prob_synthetic >= decision_threshold
        if a_synth != b_synth:
            return True
    return False


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
