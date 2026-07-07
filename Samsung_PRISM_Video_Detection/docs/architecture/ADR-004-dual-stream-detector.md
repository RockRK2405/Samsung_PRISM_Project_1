# ADR-004: Dual-stream (RGB + frequency) detector — Milestone 6

## Status
Accepted

## Date
2026-07-07

## Context
After Milestones 4 (baseline mean-pool) and 5 (temporal head, negative
result on FF++; cross-dataset AUC=0.71 on Celeb-DF), the remaining
gap in the video module is *generalisation to unseen deepfake
generators*. The most-established architectural remedy for that gap
is a **frequency-domain stream** alongside the RGB stream: GANs (and
increasingly diffusion generators) leave characteristic fingerprints
in the FFT spectrum that persist even when the pixel-space output has
been polished (F3-Net, SPSL, and the 2023-25 dual-stream follow-ups).

## Options Considered

### A. FFT-only replacement of the RGB stream
Cheap. Loses the strong RGB signal we already validated in
Milestone 4. Rejected.

### B. Feed the FFT log-magnitude as extra channels of the RGB backbone
Simplest possible dual-signal design (6-channel input instead of 3).
Requires modifying the backbone's first conv layer — throws away the
ImageNet-pretrained first-layer weights. Rejected.

### C. Two independent backbones, concatenated features
Two EfficientNet-B0 backbones — one on the RGB crop, one on the log-FFT
magnitude — concatenate the pooled features, single linear classifier
on the 2560-d combined vector. Doubles backbone params (~8M vs 4M) but
preserves the ImageNet init for both streams and matches the
architecture used by every recent dual-stream deepfake paper.

## Decision

**Option C**, with two training refinements to keep the cost small
and the experiment clean:

1. **Warm-start the RGB stream from Milestone 4's ``checkpoints/best.pt``
   and freeze it.** No need to re-learn what the baseline already knows;
   this also isolates the experiment (any improvement is attributable
   to the FFT stream alone).
2. **Frequency stream is trained end-to-end from ImageNet init.** Log-
   magnitude spectra don't look like ImageNet images so the ImageNet
   prior is weak, but it's still a better start than random weights and
   costs nothing to keep.

Video-level aggregation is mean-pool (the transformer head under-
performed in Milestone 5A on FF++; there is no reason to reintroduce it here).

### Concrete configuration (locked into ``configs/model.yaml``)
| Field | Value |
|---|---|
| ``spatial_stream.enabled`` | ``true`` |
| ``spatial_stream.freeze`` | ``true`` |
| ``spatial_stream.warm_start`` | ``checkpoints/best.pt`` |
| ``frequency_stream.enabled`` | ``true`` |
| ``frequency_stream.transform`` | ``log_fft_magnitude`` |
| ``temporal_head.type`` | ``mean_pool`` |

Training: frame-level, batch 24, 5 epochs, AdamW 3e-4 (same knobs as
Milestone 4).

## Consequences

- **Positive.** Isolates the frequency-stream contribution as a clean
  experiment: does adding an FFT branch improve cross-dataset AUC?
  Backbone parameter budget still small (~8M total, ~4M trainable).
  Fully composable via config — setting
  ``frequency_stream.enabled: false`` reverts to Milestone 4 behaviour
  in one flag flip.
- **Negative.** FFT preprocessing on every frame adds a small
  per-batch cost (~5–10% wall-time overhead in our profiling).
  Frozen-RGB training means we don't jointly re-tune the RGB
  representations to the new classifier — a follow-up unfrozen run
  might squeeze more if the frozen result is promising.
- **Follow-up.**
  - If dual-stream improves Celeb-DF AUC by a meaningful margin, try
    a second training pass with the RGB backbone unfrozen.
  - Milestone 7 (explainability) will surface both streams' GradCAM
    maps side-by-side.

## References
- `src/models/dual_stream.py`
- `configs/model.yaml`
- ADR-001 (dependency baseline — `timm`)
- Milestone 4 and 5 result docs in `docs/experiments/`.
