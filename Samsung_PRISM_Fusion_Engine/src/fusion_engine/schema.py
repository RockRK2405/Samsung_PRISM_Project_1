"""Shared wire format every modality adapter normalises into.

Mirrors the pattern already established by the video module
(`Samsung_PRISM_Video_Detection/src/inference/predictor.py`): every
modality emits a small, versioned, JSON-serialisable result so the
fusion layer never has to know about torch tensors, sklearn pipelines,
or each teammate's private return-value conventions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Bumped only on incompatible wire-format changes.
SCHEMA_VERSION: int = 1

#: The four modalities this worklet covers.
MODALITIES = ("image", "audio", "text", "video")


@dataclass
class ModalityResult:
    """One modality's normalised prediction.

    Attributes:
        modality: One of :data:`MODALITIES`.
        prob_synthetic: P(synthetic) in ``[0, 1]``. The single number every
            modality MUST provide, regardless of how its own model natively
            represents its output.
        confidence: How much the fusion engine should trust this number,
            in ``[0, 1]``. ``0`` = this modality has no opinion / failed to
            run; ``1`` = maximally confident. Distinct from prob_synthetic —
            a model can be very confident the answer is 0.5 (rare) or
            not confident at all that it's 0.99.
        available: ``False`` when this modality couldn't run at all (e.g.
            no audio track in a video-only submission, or model failed to
            load). Fusion must not silently treat a missing modality as
            "confidently real".
        latency_ms: Wall-clock inference time, for the cost-aware routing
            layer and the worklet's <5s end-to-end latency KPI.
        raw: The modality's own native output dict, preserved verbatim for
            debugging / the QC dashboard's "show me why" drill-down.
        error: Populated when ``available=False`` due to an exception.
    """

    modality: str
    prob_synthetic: float
    confidence: float
    available: bool = True
    latency_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {self.modality!r}")
        if self.available:
            if not (0.0 <= self.prob_synthetic <= 1.0):
                raise ValueError(f"prob_synthetic out of range: {self.prob_synthetic}")
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(f"confidence out of range: {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FusionResult:
    """The fusion engine's final verdict over however many modalities ran.

    Attributes:
        verdict: ``"real"`` | ``"synthetic"`` | ``"uncertain"``.
        fused_score: Weighted P(synthetic) across all available modalities,
            in ``[0, 1]``.
        fused_confidence: How much to trust ``fused_score``, in ``[0, 1]``.
        modality_results: Every modality's :class:`ModalityResult`, keyed
            by modality name — including ones that didn't run.
        weights_used: The per-modality weight actually applied (zero for
            unavailable modalities), so the QC dashboard can show its work.
        cross_modal_flags: Any cross-modal inconsistency findings (e.g.
            lip-sync mismatch). Empty list when none were checked or none
            were found.
        explanation: Natural-language summary for the QC dashboard.
    """

    verdict: str
    fused_score: float
    fused_confidence: float
    modality_results: dict[str, ModalityResult]
    weights_used: dict[str, float]
    cross_modal_flags: list[str] = field(default_factory=list)
    explanation: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict() recurses through dataclasses automatically, but
        # modality_results' values are dataclasses inside a plain dict --
        # asdict handles that fine too, so this is just for clarity.
        return d
