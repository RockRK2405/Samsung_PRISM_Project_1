# `notebooks/`

Exploratory Jupyter notebooks live here — dataset EDA, sanity-checking
face-detection output, visualising per-frame model scores, and so on.

Convention:
- Prefix each notebook with a two-digit ordinal and a short slug, e.g.
  `01_dataset_eda.ipynb`, `02_face_detection_sanity_check.ipynb`.
- Keep notebooks re-runnable: no hard-coded absolute paths — read from
  `configs/paths.yaml` instead.
- Move any code that stabilises into `src/` — notebooks are for
  exploration, not for production logic.

No notebooks exist yet.
