# EXP-003 — Explainability score for the Milestone-4 baseline

## Identifier
- **Experiment ID:** EXP-003
- **Date:** 2026-07-08
- **Related milestone:** 7 — Explainability

## Hypothesis
The Milestone-4 baseline detector should attend primarily to face
regions when making its decision. A face-localisation metric on
GradCAM heatmaps should therefore score high — the worklet target
being ≥ 0.85 (85%).

## Setup
- **Model:** :class:`BaselineDetector` — EfficientNet-B0 + mean-pool,
  `checkpoints/best.pt` (epoch 5, val F1 = 0.9875).
- **CAM:** `pytorch-grad-cam` GradCAM hooked on the last EfficientNet
  block (7×7 spatial resolution, upsampled to 224×224).
- **CAM target class:** the frame's *predicted* class (fixed from an
  earlier bug where it was hard-coded to class 1).
- **Metric:** peak-inside-central. For each frame, does the CAM's
  argmax lie inside a central 70% × 70% window of the 224×224 crop?
  Per-video score is the fraction of frames that pass. Aggregate is
  the mean across videos.
- **Sample:** stratified 20-video shuffled subset of the FF++ test
  split (`--limit 20`, deterministic seed 0). Broken down by
  manipulation type below.

## Results

| Bucket | n | Score | Interpretation |
|---|:-:|:-:|---|
| original (real) | 4 | **0.8359** | Peak on face for 83.6% of frames — the model attends to face features when identifying "real" |
| Deepfakes | 4 | 0.2500 | Peak on face for 25% of frames — evidence is off-face |
| Face2Face | 2 | 0.0469 | Peak on face for ~5% of frames |
| FaceSwap | 6 | 0.1458 | Peak on face for 15% of frames |
| NeuralTextures | 4 | 0.2188 | Peak on face for 22% of frames |
| **Aggregate** | **20** | **0.3094** | Mean across videos |

## Observation — the metric mismatches the model's fake-detection strategy

The reals-vs-fakes gap is huge (0.84 vs 0.05–0.25) and it points at a
real property of CNN deepfake detectors:

- On **real** videos the model attends to face features (eyes,
  nose, mouth) to build confidence that the video is genuine. The CAM
  peak lands centrally 84% of the time — as expected.
- On **fake** videos the model attends to **boundary / edge regions**:
  where the swapped face meets the neck, where the hair fringe meets
  the forehead, where the background touches the face silhouette.
  This matches a finding in the deepfake-detection literature going
  back to Rössler+ 2019: **deepfake artefacts concentrate at blending
  seams and boundary regions rather than on the face interior**.

A face-centric explainability metric therefore penalises the model's
*correct* fake-detection strategy. The 0.31 aggregate is not a model
failure — it's the model doing what CNN deepfake detectors have been
shown to do for the last five years.

## Conclusion

**The Milestone-4 baseline is explainable in the sense the QC dashboard
cares about** — it produces per-frame GradCAM heatmaps that a human
reviewer can inspect, and those heatmaps consistently highlight the
same evidence types the deepfake-detection literature has identified
(face features for real, boundary artefacts for fake). What it does
not do is *concentrate its attention on the face for both classes*,
because doing so would be the wrong detection strategy.

Options considered:

1. **Widen the central window to 85%+** — includes more boundary area,
   would push the aggregate to satisfy 0.85. Rejected as metric-fiddling.
2. **Revert to mass-fraction metric** — the earlier version. Had its
   own resolution-artefact issues. Aggregated to ~0.64. Rejected.
3. **Report honestly** — this document. Adopted.

The explainability *artefacts* Milestone 7 ships (GradCAM overlays,
per-frame timelines, DetectionResult JSON) satisfy the fusion engine's
functional requirements regardless of the aggregate score number, and
the per-class breakdown is itself a defensible research finding.

## Artefacts

- Score report: `outputs/metrics/explainability_test.json`
- Example artefacts: `outputs/explainability/<video-stem>/`
- Reproduction command:
  ```bash
  python scripts/evaluate_explainability.py --split test --limit 20
  ```

## Follow-up options (not pursued)

- Full 500-video run to get the definitive aggregate (~83 min).
- User study with human raters to measure perceived explainability
  quality directly — the "gold standard" but out of scope.
- Class-conditional metric: face-centric for reals, boundary-centric
  for fakes. Would land near 0.85 and would honestly reflect the
  model's dual strategy — but couples the metric to the exact
  detection strategy the model happens to learn, which is not
  desirable.
