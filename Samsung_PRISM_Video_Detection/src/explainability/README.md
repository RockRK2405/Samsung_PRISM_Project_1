# `src/explainability`

**Purpose.** Turn a model's decision on a video into an artefact the QC
dashboard and fusion engine can consume.

Planned responsibilities (not yet implemented):

- GradCAM / GradCAM++ on per-frame CNN backbones.
- Attention-weight extraction from transformer / temporal heads.
- Per-frame synthetic-probability timelines.
- Heatmap overlays saved to `outputs/explainability/`.
- Explainability-score computation for the ≥85% project target.
