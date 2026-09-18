# ADR-007 — General AI-Generated Video Detection (GenVidBench)

**Status:** In progress — Phase 0-1 (config + manifests), Phase 2 (spatial
baseline + trainer + eval) and Phase 3 (VideoMAE / TimeSformer wrappers)
code landed; awaiting first training run on GenVidBench.
**Date:** 2026-09
**Supersedes scope of:** none (additive; face milestones 1-6 and ADR-006 remain)

## Context

The 26TS08 requirements were refreshed: for the VIDEO module, the
priority is now **maximum detection quality and generalization**, not
inference cost. The module must move beyond face deepfakes to detect:

- fully AI-generated video (Sora/Kling/SVD-class generators)
- AI-edited video
- face deepfakes (existing capability)
- synthetic objects/regions
- temporal inconsistencies

The existing detector (milestones 1-6) is a **face-deepfake** classifier
(EfficientNet-B0 + mean-pool / transformer / dual-stream), trained on
FF++ face crops. EXP-001 documented its core weakness: it overfits FF++
(in-dist AUC 0.995) and collapses cross-dataset (Celeb-DF AUC 0.708,
FPR 0.72). It has **never seen AI-generated video** — the "general path"
in ADR-006 is GenImage *still images*, not video.

## Decision

Add a **general AI-video detection track** built on **GenVidBench**
(6.8M videos, 8 generators + 2 real sources), kept architecturally
separate from the face track so neither breaks the other. The eventual
system is multi-branch (general-video + face + object) with learned
fusion (Parts 16, 20).

### Cross-source-cross-generator protocol (the key design choice)

GenVidBench pairs real and fake videos on identical prompts/content, so a
detector cannot cheat on content. We train and test on **disjoint
generators and disjoint real sources**:

| Split | Fake generators | Real source |
|-------|-----------------|-------------|
| train / val | Pika, VideoCrafter2, ModelScope, Text2Video-Zero | Vript |
| test (unseen) | SVD, MuseV, Mora, CogVideo | HD-VG-130M |

This directly measures *generalization to unseen generators* (Part 25) —
the property the FF++ model lacked. Val is carved from the train
generators (disjoint videos, video-level split — Part 12) so it tracks
fit; test is the true unseen-generator measurement.

Generator metadata is preserved in every manifest row so we can later
rotate held-out generators and report seen-vs-unseen performance.

### What Phase 0-1 delivers (this ADR)

- `configs/video_training.yaml` — one centralized config (Part 28).
- `scripts/prepare_genvidbench.py` — generator-balanced, deterministic,
  layout-tolerant manifest builder → `data/manifests/{train,val,test}.csv`
  with columns `video_id, video_path, label, generator, source, dataset,
  split`. Does **not** download the dataset (Part 32).

### Phase 2-3 delivered (code; not yet trained)

- Phase 2: `VideoManifestDataset` + `VideoSpatialNet` (timm backbone +
  mean/attention/transformer pooling) + heavy-model trainer (AMP, grad
  accum, staged unfreeze, early stop, full metrics) + `train_video.py` /
  `evaluate_video.py`.
- Phase 3: `VideoTransformerNet` wrapping HF VideoMAE / TimeSformer behind
  the same interface; unified `build_video_model_from_config` dispatches on
  `model.family` so spatial-vs-foundation is a one-line config ablation.

### Deferred to later phases (planned, not yet built)

- Phase 4: face-branch backbone upgrade + temporal face analysis.
- Phase 5: learned multi-branch fusion.
- Phase 6: object/region branch + partial-region localization.
- Phase 7: full cross-dataset / cross-generator / robustness eval + ablations.

## Consequences

- **Honest scope:** until Phase 2-3 train on these manifests, we cannot
  claim general-video detection works — only that the data pipeline is
  ready. No metric will be reported without a real evaluation (Part 35).
- Heavy models (VideoMAE etc.) need a GPU; the pipeline auto-selects
  CUDA > MPS > CPU and documents MPS op-compatibility fallbacks (Part 27).
- The face track (configs/model.yaml, train.yaml) is untouched and still
  runnable; this track uses its own config namespace.
