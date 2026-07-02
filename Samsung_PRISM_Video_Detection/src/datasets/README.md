# `src/datasets`

**Purpose.** PyTorch `Dataset` and `DataLoader` definitions that expose
the preprocessed face-crop streams to training and evaluation code.

Planned responsibilities (not yet implemented):

- `FaceForensicsPPDataset`, `CelebDFv2Dataset`, `DFDCDataset`, `WildDeepfakeDataset`.
- Manifest-driven splits (train / val / test) reproducible from
  `configs/dataset.yaml`.
- Sampler utilities (balanced sampling of real vs synthetic).
- Optional augmentation pipelines.
