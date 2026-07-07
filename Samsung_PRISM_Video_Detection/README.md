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
| 1 | Environment setup & tooling                                     | **Done** — dependency baseline recorded in `docs/architecture/ADR-001-dependency-baseline.md` |
| 2 | Dataset survey & subset selection (FaceForensics++, Celeb-DF)   | **Done** — FF++ c23 primary set + 32 frames uniform / 224×224 crops locked in (`ADR-002`) |
| 3 | Frame extraction & face-detection pipeline                      | **Done** — `src/preprocessing/`, `scripts/prepare_dataset.py` |
| 4 | Per-frame CNN baseline (XceptionNet / EfficientNet)             | **Done** — EfficientNet-B0 + mean-pool (`src/models/baseline.py`) |
| 5 | Temporal aggregation head                                       | **Done — negative result kept.** Transformer head shipped and config-selectable (`ADR-003`), but on FF++ c23 it under-performed mean-pool at the 5-epoch training budget (F1=0.9748 vs 0.9862, NeuralTextures −5 pp). Baseline `best.pt` remains the production model. |
| 5B | Cross-dataset generalisation (Celeb-DF v2)                      | **Done** — FF++ baseline evaluated on Celeb-DF 518-video test split. AUC=0.7076, F1=0.8142, FPR=0.72. Landed inside the 60–75% AUC band published for FF++-only-trained models on Celeb-DF (see `docs/experiments/EXP-001-celeb-df-cross-dataset.md`). Confirms the domain-shift failure mode the fusion engine is designed to mitigate. |
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

Runtime dependencies are in `requirements.txt`; developer tooling
(pytest, ruff, mypy, pre-commit) is in `requirements-dev.txt`.

```bash
# Runtime only:
pip install -r requirements.txt

# Runtime + development tools:
pip install -r requirements.txt -r requirements-dev.txt
```

The rationale behind the current dependency choices lives in
[`docs/architecture/ADR-001-dependency-baseline.md`](docs/architecture/ADR-001-dependency-baseline.md).

### 6.5 Convenience shortcuts (`make`)

A minimal `Makefile` wraps the most common workflows. From the project
root, once the venv exists:

```bash
make help          # list available targets
make install-dev   # create venv + install runtime + dev deps
make lint          # ruff check
make format        # ruff fix + format
make typecheck     # mypy
make test          # pytest
```

---

## 7. Current Status

- Repository skeleton created.
- Environment + dependency baseline installed and validated on Apple Silicon.
- Dataset locked in (FaceForensics++ c23 via Kaggle mirror) — 4 canonical
  manipulations plus `original/` reals.
- Preprocessing pipeline implemented: video → 32 uniform frames →
  224×224 face crops via MTCNN, with a leak-free 80/10/10 split by
  source-video group id.
- **Baseline model trained** — EfficientNet-B0 + mean-pool, 5 epochs
  in ~60 min on MPS. Final held-out test-set metrics (val-tuned
  threshold = 0.5872 for FPR ≤ 5%):

  | Metric | Test (@0.5) | **Test (tuned)** | Target | Status |
  |---|:-:|:-:|:-:|:-:|
  | F1 | 0.9837 | **0.9862** | ≥ 0.92 | ✅ +6.6 pp |
  | FPR | 0.0600 | **0.0300** | ≤ 0.05 | ✅ 2 pp under |
  | Precision | 0.9850 | 0.9924 | — | Improved |
  | Recall | 0.9825 | 0.9800 | — | −0.25 pp trade |
  | AUC | 0.9946 | 0.9946 | — | Excellent |

  Per-manipulation recall (tuned): Deepfakes 100%, Face2Face 99%,
  FaceSwap 99%, NeuralTextures **94%** (hardest — GAN-based, matches
  published literature). Real-video specificity: 97%.

### Running the preprocessing pipeline

From the module root, with the venv active:

```bash
# Step 1 — build the train/val/test manifest (fast, seconds)
python scripts/prepare_dataset.py manifest

# Step 2 — smoke test on 2 videos per split before the long run
python scripts/prepare_dataset.py faces --limit 2

# Step 3 — full run (this is the slow one — hours on a laptop)
python scripts/prepare_dataset.py faces
```

Output lands under `data/processed/faces/ff_c23/<split>/<label>/<video>/`.
Re-running the same command skips videos whose output dir is already
populated; pass `--overwrite` to force re-extraction.

### Training the baseline model

Once the face crops exist, train the Milestone-4 baseline
(EfficientNet-B0 + mean-pool video head):

```bash
# First run downloads the timm ImageNet weights (~20 MB)
python scripts/train.py                         # uses configs as-is
python scripts/train.py --epochs 10             # override any field
python scripts/train.py --device cpu            # force a device

# Evaluate the best checkpoint on the test split
python scripts/evaluate.py --split test

# Tune the decision threshold on val so FPR ≤ 5% at test
python scripts/evaluate.py --split test --tune-threshold-fpr 0.05
```

### Cross-dataset evaluation on Celeb-DF v2 (Milestone 5B)

Get Celeb-DF v2 onto disk (the official request form is at
https://github.com/yuezunli/celeb-deepfakeforensics; several Kaggle
mirrors also exist). You need the folders `Celeb-real/`,
`YouTube-real/`, `Celeb-synthesis/`, plus the file
`List_of_testing_videos.txt`.

Extract face crops for the official 518-video test split:

```bash
# Smoke test on 5 videos first (~1 min)
python scripts/prepare_celeb_df.py --dataset-root ~/Downloads/Celeb-DF-v2 --limit 5

# Full extraction (~30–60 min on CPU MTCNN)
python scripts/prepare_celeb_df.py --dataset-root ~/Downloads/Celeb-DF-v2
```

Evaluate the FF++-trained baseline on Celeb-DF (threshold tuned on
FF++ val — the distribution the model actually knows):

```bash
python scripts/evaluate.py --dataset celeb_df_v2 --split test \\
    --checkpoint checkpoints/best.pt --tune-threshold-fpr 0.05
```

Artefacts:

* `checkpoints/best.pt` — best-val-F1 model weights
* `outputs/metrics/baseline_test.json` — final metrics report
* `logs/mlruns/` — MLflow run history (browse with `mlflow ui`)
