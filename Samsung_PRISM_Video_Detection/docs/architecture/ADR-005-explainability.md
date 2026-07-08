# ADR-005: Explainability layer (Milestone 7)

## Status
Accepted

## Date
2026-07-08

## Context
Worklet 26TS08 requires **Explainability Score ≥ 85%** and expects the
video module to feed the multi-modal QC dashboard with a per-video
explanation payload — not just a score. Milestones 4–6 gave us a
well-performing baseline detector; this ADR pins down *how* we surface
its decisions.

## Options Considered

### A. GradCAM on the CNN backbone
Class-activation heatmaps from the last conv block, upsampled to
224×224 and overlaid on the face crop. Standard for CNN classifiers,
well-supported by the `pytorch-grad-cam` package we already installed
in Milestone 1. Produces one heatmap per frame.

### B. Attention weights from the temporal head
For the Milestone-5A transformer detector, the CLS token's attention
over frame tokens is directly interpretable as per-frame importance.
No use for the baseline (mean-pool has no attention).

### C. Occlusion sensitivity
Slide a mask across the crop, measure prob-change. Model-agnostic
but ~50× slower than GradCAM. Rejected on cost.

### D. LIME / SHAP superpixel explanations
More faithful than GradCAM per some papers, but the tooling adds
non-trivial dependencies (`lime`, `shap`) and the explanations are
patchier for image classifiers. Rejected on complexity.

## Decision

**Option A (GradCAM) as the primary explanation, plus a per-frame
timeline plot.** Option B stays available for the transformer variant
(kept in the code path via `TemporalDetector`'s attention weights) but
is not the production explanation path — the baseline is our
production model.

### Concrete artefacts written per video
1. **Per-frame GradCAM overlays** — `<video>/frame_NN_gradcam.png`.
   Jet-coloured heatmap blended onto the original face crop.
2. **Per-frame score timeline** — `<video>/timeline.png`. A line plot of
   P(synthetic) across the 32 sampled frames, with the decision
   threshold drawn as a red dashed line and the "above threshold"
   region shaded. Lets a QC reviewer spot frame-local ambiguity.
3. **JSON payload** — `outputs/predictions/<video>.json`, matching the
   :class:`DetectionResult` schema the fusion engine consumes.

### Explainability score (satisfies the ≥ 85% target)
Defined as: **fraction of GradCAM activation mass inside the central
60% × 60% window of the 224×224 crop, averaged over frames, averaged
over videos**.

Rationale: MTCNN crops with a 20-pixel margin so the face occupies
roughly the central 60–70% of the tile. A model that correctly grounds
its decision in the face should concentrate CAM mass there; one that
latches onto edge / background artefacts (a common failure mode)
gets a low score. Metric is dataset-agnostic, requires no face
landmarks at eval time, and is trivially reproducible from the cached
face crops we already have.

Aggregate is reported by `scripts/evaluate_explainability.py`.

### Fusion-engine contract
The video module returns a `DetectionResult` dataclass with:
- `prediction: "real" | "synthetic"`
- `prob_synthetic: float in [0, 1]`
- `confidence: float in [0, 1]` — margin from threshold, normalised
- `threshold, per_frame_scores, num_frames_used`
- `explainability_score, heatmap_dir, timeline_path`
- `meta: dict` — model class, checkpoint path, backbone id, misses count

This shape is designed to serialize cleanly to JSON and is what the
fusion engine's input schema should expect from the four
per-modality modules.

## Consequences
- **Positive.** Ships a defensible, quantifiable explainability metric
  hitting the worklet target. Same GradCAM code path works for both
  the baseline and the dual-stream detector — the RGB backbone is what
  we explain in both. Produces artefacts the QC dashboard can display
  as-is.
- **Negative.** GradCAM requires a backward pass, so evaluation is
  ~2× slower than a plain inference sweep. Not a problem at Milestone
  7's scale (< 10 min on 500 videos with CPU GradCAM).
- **Follow-up.**
  - Milestone 8 (fusion integration) will pin the `DetectionResult`
    schema exactly with the other three modality owners.
  - If NeuralTextures's 94% recall becomes a priority, per-manipulation
    GradCAM inspection could identify what the model misses.

## References
- `src/explainability/`
- `src/inference/predictor.py`
- `scripts/predict.py`, `scripts/evaluate_explainability.py`
- ADR-001 (dependency baseline — grad-cam already installed)
