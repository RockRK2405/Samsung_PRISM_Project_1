# EXP-002 — Dual-stream (RGB + FFT) detector

## Identifier
- **Experiment ID:** EXP-002
- **Date:** 2026-07-07
- **Related milestone:** 6 — Dual-stream RGB + frequency-domain upgrade

## Hypothesis
Adding a frequency-domain stream (FFT log-magnitude of face crops) to
the Milestone-4 RGB baseline should improve cross-dataset generalisation
(Celeb-DF AUC), because GAN fingerprints in the frequency spectrum
tend to persist across generators even when pixel-space output has
been polished (F3-Net, SPSL, and 2023–25 follow-ups).

Target: Celeb-DF AUC ≥ 0.75 (baseline was 0.7076).

## Setup
- **Model:** :class:`DualStreamDetector` — two EfficientNet-B0 backbones,
  concatenated 2560-d features, linear classifier.
  * RGB backbone: **warm-started from `checkpoints/best.pt` and frozen.**
  * FFT backbone: ImageNet init, trained end-to-end.
- **FFT preprocessing:** ``fftshift(log(1 + |FFT2(x)|))`` per image
  per channel, z-scored (computed on CPU because MPS lacks complex-dtype
  support).
- **Training:** 5 epochs, frame-level, batch 24, AdamW 3e-4 + cosine.
  Effective wall time ~85 min on M5 Pro (a mid-run laptop-sleep
  suspended epoch 2 by ~33 min without corrupting training).
- **Checkpoint:** `checkpoints/best_dual.pt` (epoch 2, val F1 = 0.9887).
- **Git commit:** `a2e131c`.

## Results

### FF++ test (in-distribution, threshold 0.4947 tuned on FF++ val)

| Metric | Baseline (M4) | Dual (M6) | Δ |
|---|:-:|:-:|:-:|
| Accuracy | 0.9740 | 0.9780 | +0.4 pp |
| F1 | 0.9862 | 0.9862 | 0.00 |
| Precision | 0.9924 | 0.9924 | 0.00 |
| Recall | 0.9800 | 0.9800 | 0.00 |
| FPR | 0.0300 | 0.0300 | 0.00 |
| AUC | 0.9946 | 0.9948 | +0.02 pp |
| Deepfakes recall | 1.0000 | 1.0000 | 0.00 |
| Face2Face recall | 0.9900 | 0.9900 | 0.00 |
| FaceSwap recall | 0.9900 | 0.9900 | 0.00 |
| NeuralTextures recall | 0.9400 | 0.9400 | 0.00 |

### Celeb-DF test (cross-dataset, same threshold)

| Metric | Baseline (M4) | Dual (M6) | Δ |
|---|:-:|:-:|:-:|
| F1 | 0.8142 | 0.8120 | −0.22 pp |
| **AUC** | **0.7076** | **0.7124** | **+0.48 pp** (noise) |
| FPR | 0.7247 | 0.7697 | −4.5 pp (worse) |
| Recall (fake) | 0.9471 | 0.9588 | +1.2 pp |
| Celeb-synthesis recall | 0.9471 | 0.9588 | +1.2 pp |
| Celeb-real specificity | 0.2963 | 0.2407 | −5.6 pp |
| YouTube-real specificity | 0.2429 | 0.2143 | −2.9 pp |

## Observations

1. **FF++ test numbers are bit-for-bit tied.** F1, precision, recall,
   FPR, and per-manipulation recall are identical to four decimal
   places. The frozen RGB backbone is dominating; the FFT stream is
   contributing exactly zero on in-distribution data.
2. **Celeb-DF AUC lift is within noise.** +0.48 pp on a bootstrap std
   of ~2 pp for a 518-sample AUC. The frequency stream did not close
   the cross-dataset gap.
3. **The FPR got slightly worse.** Dual-stream flags more reals as
   fake on Celeb-DF (77% vs 72%). Combined with the recall gain, the
   model became a hair more fake-happy overall — not the direction we
   wanted.
4. **Training loss was flat from epoch 1.** Loss went 0.127 → 0.121
   over five epochs — the FFT branch converges to something
   near-degenerate almost immediately. Suggests the classifier learned
   to rely on RGB features and treat the FFT branch as low-signal.

## Root-cause analysis

Two plausible reasons the FFT stream did not earn its keep:

1. **Frozen-RGB bias.** With the RGB backbone frozen at a
   near-saturated Milestone-4 state, the linear classifier had
   already-strong features on hand and little gradient pressure to
   adapt to the noisy new FFT features. A follow-up with the RGB
   backbone unfrozen (jointly fine-tuned) might buy something, but
   doubles training cost.
2. **Dataset era.** The classical FFT-fingerprint literature evaluates
   against 2018–2020 GAN generators. FF++ and Celeb-DF's synthesis
   pipelines pre-date those papers and are already a documented
   distribution shift; the fingerprints they leave, if any, are not
   the ones this simple vanilla FFT preprocessing highlights.

## Conclusion

**Negative result.** On FF++ / Celeb-DF, a frozen-RGB + FFT dual-stream
detector does not measurably outperform the Milestone-4 mean-pool
baseline — neither in-distribution nor cross-dataset. Baseline
`best.pt` remains the production model for downstream milestones.

This is the second time the project has landed on the finding that
**mean-pool over a strong ImageNet-pretrained CNN is a hard-to-beat
baseline on this specific data**:
- Milestone 5A: adding a transformer temporal head did not help.
- Milestone 6: adding a frequency-domain stream did not help.

Both are defensible findings for the mentor report. Both experiments
also produced usable side artefacts — Milestone 5A's transformer
attention weights and Milestone 6's dual-stream architecture — that
Milestone 7 (explainability) will still surface for the QC dashboard
even though they did not improve accuracy.

## Follow-up options (not pursued)

- Unfreeze the RGB backbone and jointly fine-tune (higher compute).
- Try alternative frequency transforms (DCT, DWT, phase spectrum).
- Train from ImageNet init instead of warm-starting from `best.pt` —
  might let the FFT stream matter more relatively.
- Use a Celeb-DF-inclusive training set (would change the whole story).

## Artefacts

- Checkpoint: `checkpoints/best_dual.pt`
- Metrics: `outputs/metrics/dual_test_tuned.json`,
  `outputs/metrics/dual_celeb_df_v2_test_tuned.json`
- Reproduction commands:
  ```bash
  # Training (~85 min on M5 Pro / MPS)
  python scripts/train.py

  # In-distribution evaluation
  python scripts/evaluate.py --split test --tune-threshold-fpr 0.05 \
      --checkpoint checkpoints/best_dual.pt

  # Cross-dataset evaluation
  python scripts/evaluate.py --dataset celeb_df_v2 --split test \
      --tune-threshold-fpr 0.05 --checkpoint checkpoints/best_dual.pt
  ```
