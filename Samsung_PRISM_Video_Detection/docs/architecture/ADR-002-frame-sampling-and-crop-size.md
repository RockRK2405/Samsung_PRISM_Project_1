# ADR-002: Frame sampling strategy and face-crop size

## Status
Accepted

## Date
2026-07-02

## Context
Before writing the frame-extraction pipeline (Milestone 3), three
knobs in `configs/dataset.yaml` needed concrete values:

1. `frames_per_video` — how many frames to sample per clip.
2. `sampling.strategy` — how to pick those frames.
3. `face_crop_size` — target resolution of the cached face crops.

These decisions constrain (a) disk usage of the face-crop cache,
(b) training-batch shape, and (c) which pretrained backbones we can
load without resizing.

## Decision

| Setting          | Value       |
|------------------|-------------|
| frames_per_video | **32**      |
| sampling         | **uniform** |
| face_crop_size   | **224**     |

### Rationale — frames_per_video = 32
16 is the literature default (VideoMAE, X-CLIP, TimeSFormer) and would
also work. 32 was chosen because temporal artefacts (flicker, unnatural
blinking, motion incoherence) are the primary failure mode we care
about for video — halving temporal resolution to save disk is the
wrong trade-off when our target hardware (M5 Pro, 24 GB unified) can
comfortably handle the larger cache (~160 GB projected). If Milestone 8
cross-dataset evaluation shows overfitting to temporal patterns, we can
downsample at dataloader time without re-extracting.

### Rationale — uniform sampling
Uniform (N evenly-spaced frames across the full clip) matches the
majority of published FF++ baselines, so our numbers stay directly
comparable. Dense (consecutive) sampling biases the model toward local
motion at the cost of whole-video context; keyframe sampling gives
negligible benefit on FF++ because clips are short (a few seconds).

### Rationale — 224×224 crop
Every backbone we plan to trial in Milestones 4–6 (EfficientNet-B0 to
B4, ViT-Base/16, XceptionNet variants via `timm`) accepts 224×224 as a
first-class input. Choosing anything else forces resizing at load time
and loses information. XceptionNet's original 299×299 was considered
but rejected — we are not committing to XceptionNet as the sole
backbone, so the extra pixels would be wasted for every other model.

## Consequences
- **Positive:** All three values baked into `configs/dataset.yaml`;
  Milestone 3 extraction code can be written with no further ambiguity.
- **Negative:** Face-crop cache will be sizeable (~150 GB order of
  magnitude across all 5000 videos × 32 crops × 224²). Well within
  disk headroom, but worth monitoring.
- **Follow-up work:**
  - ADR-003 — face-detection backend selection (MediaPipe vs
    facenet-pytorch), decided after Milestone 3 A/B trial.
  - Revisit `frames_per_video` if evaluation reveals overfitting.

## References
- `configs/dataset.yaml`
- ADR-001 (dependency baseline — `timm` for backbones)
- README Milestones 3–6
