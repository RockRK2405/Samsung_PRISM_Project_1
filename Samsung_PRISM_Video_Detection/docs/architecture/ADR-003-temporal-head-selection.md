# ADR-003: Temporal aggregation head — transformer over mean-pool

## Status
Accepted

## Date
2026-07-06

## Context
Milestone 4's mean-pool baseline hit F1 = 0.9862 / FPR = 0.03 / AUC =
0.9946 on the FF++ c23 test set, but per-manipulation recall showed a
clear weak spot: **NeuralTextures at 94%**, ~5-6 pp behind the other
three methods. This matches the literature — NeuralTextures is the
only FF++ method that uses learned GAN texture generation, so its
artefacts live in the temporal dimension (subtle flicker, motion
incoherence) rather than in any single frame.

Mean-pool treats every frame as independent evidence, then averages
the votes. It cannot represent "frames A and B look plausible on their
own but the transition A→B looks wrong." Some form of sequence model
is required to break through the NeuralTextures ceiling and to give
the fusion engine explainability signals per time-segment.

## Options Considered

### Option A — BiLSTM over per-frame features
Small, fast, well understood for sequence modelling. Downside: no
attention weights, so we get less useful signal for Milestone 7's
explainability requirement. Requires `pack_padded_sequence` gymnastics
for variable-length clips.

### Option B — Transformer encoder with CLS token
Standard Vision-Transformer-style aggregation: a learnable CLS token
plus positional embeddings for T frame tokens, run through 1-2
encoder layers, classify from the CLS output. Padding is handled
natively via ``src_key_padding_mask``. Attention weights over the T
frame slots are recoverable and directly usable as per-frame
importance scores — a free deliverable for Milestone 7.

### Option C — Convolutional temporal head (Temporal Convolutional Network)
Dilated 1-D convs over the feature time series. Fast and simple, but
also opaque — no natural per-frame attention signal.

## Decision
Adopt **Option B** — a 2-layer transformer encoder with a CLS token
and learned positional embeddings over the 32 frame tokens. Backbone
features are projected from 1280 → 256 (``hidden_dim``) to keep the
transformer cheap; 4 attention heads (head dim 64) is the standard
configuration.

### Concrete configuration (locked into ``configs/model.yaml``)
| Field | Value |
|---|---|
| ``temporal_head.type`` | ``transformer`` |
| ``temporal_head.hidden_dim`` | 256 |
| ``temporal_head.num_layers`` | 2 |
| ``temporal_head.num_heads`` | 4 |

Video-level training is enabled automatically when the trainer sees a
non-``mean_pool`` head — it swaps ``FaceFrameDataset`` for
``FaceVideoDataset`` and applies the padded collate function, so
short-clip videos (a few % of FF++) are handled correctly. Baseline
training is untouched — setting ``temporal_head.type: mean_pool`` in
the config rolls the pipeline back to Milestone 4 with no code change.

## Consequences
- **Positive.** Explainability signal (attention weights) comes for
  free; padding is handled natively; the whole model is a drop-in
  replacement selectable from config.
- **Negative.** Training is video-level now, so effective batch size
  in image units is `batch_size × 32` — we drop the CLI batch size
  from 32 (baseline) to 8 (video-level). Wall time per epoch is
  comparable because the total number of frame forwards is unchanged.
- **Follow-up.**
  - Milestone 7 will surface the attention weights via
    ``src/explainability/`` as per-frame importance heatmaps.
  - Consider warm-starting the backbone from Milestone 4's ``best.pt``
    if the first temporal run underperforms.

## References
- ``src/models/temporal.py``
- ``configs/model.yaml``
- ADR-001 (dependency baseline — ``timm``)
- Milestone 4 test results in README §7.
