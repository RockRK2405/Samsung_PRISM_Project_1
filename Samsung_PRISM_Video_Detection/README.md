# Samsung PRISM — Video Detection Module

**Worklet 26TS08 — A Cost-Aware Framework for Synthetic Data Detection in Data Acquisition Pipelines**

This repository contains the **Video Detection Module** of the Samsung PRISM
Worklet 26TS08 project. The larger project builds a multi-modal system that
verifies whether incoming data (image, text, audio, video) is genuine or
synthetically generated, and fuses per-modality signals into a single trust
score. This module is responsible for the **video modality** only.

---

## 1. Project Objective

Given an input video, determine whether it is:

- **Real** — captured by a physical camera of a real scene, or
- **Synthetic / Deepfake** — produced or manipulated by a generative model
  (face-swap, face-reenactment, or talking-face / lip-sync generation).

The module must output:

1. A synthetic-probability score in `[0, 1]`.
2. A confidence value.
3. An explanation artifact (per-frame scores / attention heatmaps) suitable
   for the project's downstream QC dashboard and fusion engine.

Success targets, inherited from the worklet specification:

| Metric                          | Target        |
|---------------------------------|---------------|
| F1 Score                        | ≥ 0.92        |
| False Positive Rate             | ≤ 5%          |
| Explainability Score            | ≥ 85%         |
| Cross-dataset generalization    | Validated on a held-out dataset |

---

## 2. Current Development Stage

> **Stage 0 — Repository Scaffolding.**
>
> Only the project foundation exists at this point: directory layout,
> configuration templates, documentation templates, and package skeletons.
>
> **No ML models, no training code, no preprocessing logic, and no
> datasets have been implemented or downloaded yet.** All source
> directories contain only package initializers and placeholder README
> files that describe their intended purpose.

The current commit is the reference starting point from which all
subsequent implementation work will grow.

---

## 3. Repository Structure

```
Samsung_PRISM_Video_Detection/
│
├── .venv/                       # Local Python virtual environment (git-ignored)
├── configs/                     # YAML configuration templates
├── data/
│   ├── raw/                     # Original video files, untouched
│   ├── processed/               # Extracted frames / cropped faces
│   ├── metadata/                # Per-video metadata (fps, codec, duration, labels)
│   └── cache/                   # Intermediate artefacts safe to delete
│
├── datasets/                    # Dataset registration / download manifests
│
├── docs/
│   ├── papers/                  # Literature notes and paper summaries
│   ├── architecture/            # Architecture decision records (ADRs)
│   ├── meeting_notes/           # Mentor / team meeting notes
│   └── experiments/             # Experiment logs
│
├── notebooks/                   # Exploratory Jupyter notebooks
│
├── outputs/
│   ├── metrics/                 # Evaluation metrics (JSON / CSV)
│   ├── visualizations/          # Plots, confusion matrices
│   ├── predictions/             # Per-sample predictions
│   └── explainability/          # GradCAM / attention heatmaps
│
├── checkpoints/                 # Saved model weights (git-ignored)
├── logs/                        # Training / evaluation logs (git-ignored)
├── scripts/                     # CLI entrypoints (train, evaluate, predict, ...)
│
├── src/                         # Library code — importable Python package
│   ├── preprocessing/           # Frame extraction, face detection, cropping
│   ├── datasets/                # PyTorch Dataset / DataLoader definitions
│   ├── models/                  # Model architectures
│   ├── training/                # Training loops, optimizers, schedulers
│   ├── evaluation/              # Metrics, evaluation pipelines
│   ├── explainability/          # GradCAM, attention visualization
│   ├── inference/               # Single-video inference API
│   ├── utils/                   # Logging, IO, config loading
│   └── configs/                 # Config dataclasses / schema
│
├── tests/                       # Unit & integration tests
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 4. Planned Architecture

The eventual detection pipeline is designed as a linear sequence of
independently-testable stages:

```
Video
  ↓  Metadata Extraction
  ↓  Frame Extraction
  ↓  Frame Sampling
  ↓  Face Detection
  ↓  Face Tracking
  ↓  Face Cropping
  ↓  Preprocessing (normalization, augmentation)
  ↓  Feature Extraction (spatial stream + frequency stream)
  ↓  Temporal Modeling (LSTM / cross-frame attention)
  ↓  Binary Classification (Real vs Synthetic)
  ↓  Explainability (per-frame scores, attention heatmaps)
  ↓  Output (probability, confidence, explanation)
```

The target reference architecture is a **dual-stream (RGB + frequency-domain)
ViT-based per-frame feature extractor** with a **temporal aggregation head**
(LSTM or cross-frame attention). This will be preceded by a lighter CNN
baseline (XceptionNet / EfficientNet family) so that the pipeline can be
validated end-to-end before the more expensive model is trained.

None of this is implemented yet — it is documented here as the intended
target so future modules can be built into their correct slots.

---

## 5. Future Milestones

| # | Milestone                                                       | Status  |
|---|-----------------------------------------------------------------|---------|
| 0 | Repository scaffolding                                          | Current |
| 1 | Environment setup & tooling                                     | Planned |
| 2 | Dataset survey & subset selection (FaceForensics++, Celeb-DF)   | Planned |
| 3 | Frame extraction & face-detection pipeline                      | Planned |
| 4 | Per-frame CNN baseline (XceptionNet / EfficientNet)             | Planned |
| 5 | Temporal aggregation head                                       | Planned |
| 6 | Dual-stream (RGB + frequency) upgrade                           | Planned |
| 7 | Explainability layer (GradCAM / attention maps)                 | Planned |
| 8 | Cross-dataset evaluation & robustness testing                   | Planned |
| 9 | Integration hooks for the multi-modal fusion engine             | Planned |

---

## 6. Environment Setup (macOS, Apple Silicon)

The module is developed and tested on Apple Silicon (M-series) laptops
using Python 3.11 and the built-in `venv` module. Conda is intentionally
not used.

### 6.1 Create the virtual environment

```bash
cd Samsung_PRISM_Video_Detection
python3.11 -m venv .venv
```

### 6.2 Activate the environment

```bash
source .venv/bin/activate
```

To deactivate later:

```bash
deactivate
```

### 6.3 Upgrade base tooling

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 6.4 Install dependencies

> Dependencies are intentionally not pinned yet. `requirements.txt` will
> be populated in Milestone 1 once concrete libraries are selected.

```bash
pip install -r requirements.txt
```

---

## 7. Current Status

- Repository skeleton created.
- Directory layout finalised.
- Documentation and configuration templates in place.
- **No implementation code, no ML models, no datasets present.**

Next step: Milestone 1 — environment tooling and dependency selection.
