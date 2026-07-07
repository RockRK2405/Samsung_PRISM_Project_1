# EXP-001 — Cross-dataset generalisation on Celeb-DF v2

## Identifier
- **Experiment ID:** EXP-001
- **Date:** 2026-07-07
- **Related milestone:** 5B — Cross-dataset generalisation

## Hypothesis
The FF++-trained baseline detector should transfer with **some**
performance drop to Celeb-DF v2. Literature suggests published FF++-only
models land at 60–75% AUC on Celeb-DF; if we're inside that band we've
built a normal (not pathologically over-fit) FF++ detector. Below that,
we've over-fit; above it, we've somehow generalised better than the
average paper (unlikely).

## Setup
- **Model:** Milestone 4 baseline — EfficientNet-B0 + mean-pool head,
  checkpoint `checkpoints/best.pt` (val F1 = 0.9875).
- **Threshold:** tuned on FF++ val for FPR ≤ 5% → 0.5872.
- **Source dataset (train):** FaceForensics++ c23, 4 canonical
  manipulations. (Nothing from Celeb-DF was seen during training.)
- **Target dataset (eval):** Celeb-DF v2 official test split — 518 videos
  (108 Celeb-real + 70 YouTube-real + 340 Celeb-synthesis).
- **Git commit:** `c34e830`.
- **Hardware:** MacBook M5 Pro, MPS.

## Results

### Overall

| Metric | Value |
|---|:-:|
| Accuracy | 0.7162 |
| Precision | 0.7140 |
| Recall | 0.9471 |
| **F1** | **0.8142** |
| **FPR** | **0.7247** |
| **AUC** | **0.7076** |

### Per-source

| Bucket | n | Correct-rate | Reading |
|---|:-:|:-:|---|
| Celeb-synthesis (fake) | 340 | 0.9471 | Model catches 94.7% of Celeb-DF fakes |
| Celeb-real (studio real) | 108 | 0.2963 | 70% of studio-quality reals mis-flagged as fake |
| YouTube-real (in-wild real) | 70 | 0.2429 | 76% of in-the-wild reals mis-flagged as fake |

### FF++ vs Celeb-DF at a glance

| Metric | FF++ (in-dist) | Celeb-DF (cross) | Δ |
|---|:-:|:-:|:-:|
| F1 | 0.9862 | 0.8142 | −17 pp |
| AUC | 0.9946 | 0.7076 | **−29 pp** |
| FPR | 0.03 | 0.72 | **+69 pp** |
| Fake recall | 0.98 | 0.947 | −3 pp |

## Observations

1. **AUC lands squarely inside the 60–75% range published for FF++-only
   models tested on Celeb-DF.** The finding is normal, not
   pathological — this is *the* well-documented cross-dataset gap in
   deepfake detection.
2. **Failure is one-sided.** The model is biased toward "fake" — it
   catches 94.7% of Celeb-DF fakes, but flags 70–76% of reals as fake.
3. **YouTube-real fails harder than Celeb-real** (75.7% vs 70.4% FPR).
   In-the-wild videos are further from FF++'s (largely
   YouTube-interview) training distribution than professionally-produced
   celebrity content, so the "unfamiliar → probably fake" default
   fires more often.
4. **Threshold tuning cannot fix this.** The 0.5872 threshold tuned on
   FF++ val is the *most permissive* value that keeps FPR ≤ 5% on
   FF++. Raising it further would help Celeb-DF FPR but tank fake
   recall — the underlying ranking (AUC=0.7076) is the ceiling.
5. **F1 = 0.81 is misleadingly optimistic.** Class imbalance (340 fakes
   vs 178 reals) + the model's fake-happy bias inflates F1. AUC is the
   honest headline.

## Conclusion

The FF++-trained baseline **generalises to unseen deepfake generators
in the "typical" band predicted by the literature** — strong recall on
new fakes, but severe over-firing on unfamiliar reals. This is exactly
the empirical failure mode the worklet's fusion engine and adaptive
controller are designed to mitigate:

- Per-frame confidence + explainability (Milestone 7) tell the QC
  dashboard *when* to distrust the video-module signal.
- Cross-modal fusion with image/audio/text detectors dilutes the
  video module's overconfident single-modality mistakes.
- Adaptive routing sends ambiguous cases to a heavier model instead of
  trusting the light detector everywhere.

**This experiment supports the worklet's architectural premise.**

## Artefacts

- Checkpoint: `checkpoints/best.pt`
- Face-crop cache: `data/processed/faces/celeb_df_v2/`
- Metrics JSON: `outputs/metrics/baseline_celeb_df_v2_test_tuned.json`
- Reproduction command:
  ```bash
  python scripts/evaluate.py --dataset celeb_df_v2 --split test \
      --checkpoint checkpoints/best.pt --tune-threshold-fpr 0.05
  ```
