# Samsung PRISM — Video Detection Module

**Worklet 26TS08 — A Cost-Aware Framework for Synthetic Data Detection in Data Acquisition Pipelines**

The **Video Detection Module** of Samsung PRISM Worklet 26TS08. The larger
project verifies whether incoming data (image, text, audio, video) is genuine
or synthetically generated; this module owns the **video modality**.

> **Scope note.** The worklet is titled "cost-aware", but for the *video
> model itself* the priority is **detection quality and generalization** —
> heavy models are allowed. Cost-aware routing lives elsewhere in PRISM and
> is not a constraint on this module's architecture.

---

## 1. What this module is

Not "just a face deepfake detector." The target is a **general synthetic-video
detector**: given an arbitrary video, decide whether it is genuine,
AI-generated, AI-edited, or contains synthetic content — **even when there is
no face or human in frame.**

Detection scope (target capabilities):

| Category | Example | Branch |
|----------|---------|--------|
| **A. Fully AI-generated video** | text/image-to-video (Sora/Kling/SVD-class), any subject | **General-video (primary)** |
| **B. AI-generated human** | synthetic full body / profile / distant / occluded | Human branch (planned) |
| **C. Face deepfake** | face-swap, reenactment, expression manipulation | Face branch (built) |
| **D. AI-edited real video** | real clip with AI-modified person/object/background | General + fusion |
| **E. Synthetic object/region** | real video containing an AI-generated car/building | Object branch (planned) |
| **F. Temporal artifacts** | flicker, unnatural motion, identity drift | Temporal reasoning |

The **face branch is one component**, not the whole detector.

---

## 2. Current status (2026-09)

Two tracks live in the repo. Status legend: ✅ built+verified · 🟢 code (untrained) · 🟠 partial · ⏳ planned.

### Track 1 — Face deepfake (complete, milestones 1-9)
Trained EfficientNet-B0 detector on FaceForensics++; full preprocessing,
training, evaluation, explainability, and fusion-integration API. **Verified
results below.**

### Track 2 — General AI-video (ADR-007, in progress)
| Phase | Status | Component |
|-------|--------|-----------|
| 0 config | ✅ | `configs/video_training.yaml` |
| 1 data pipeline | ✅ verified | `scripts/prepare_genvidbench.py` — GenVidBench manifests, cross-generator, zero leakage |
| 2 spatial baseline | 🟢 | `src/models/video_spatial.py` (ConvNeXt + attention/transformer pool) + trainer + eval |
| 3 video transformer | 🟢 | `src/models/video_transformer.py` (VideoMAE / TimeSformer) |
| 4-10 face upgrade, human, object, temporal, fusion, rich explainability | ⏳ | see `docs/VIDEO_MODULE_MASTERPLAN.md` |

> **Honest scope:** the general-video track is **code, not yet trained** — no
> general-video metric is claimed until it runs on GenVidBench. Full roadmap
> and resume runbook: [`docs/VIDEO_MODULE_MASTERPLAN.md`](docs/VIDEO_MODULE_MASTERPLAN.md).

---

## 3. Architecture

**Target multi-branch system:**

```
ARBITRARY VIDEO → preprocessing → frame/clip sampling
        ├── GENERAL VIDEO branch   (VideoMAE / TimeSformer / ConvNeXt+temporal)   [PRIMARY]
        ├── FACE branch            (MTCNN → forensic CNN → temporal aggregation)
        ├── HUMAN branch           (person detector → region → AIGC classifier)   [planned]
        └── OBJECT/REGION branch   (YOLO/GDINO → crop → AIGC classifier)          [planned]
                         ↓ learned FUSION (MLP / transformer) ↓
              FINAL SCORE + rich explanation (frames / faces / humans / objects / temporal)
```

**Built today:**
- **General-video branch** — per-frame timm backbone + swappable temporal head
  (mean-pool / attention-pool / transformer), OR a pretrained video
  transformer (VideoMAE 16-frame, TimeSformer 8-frame). One config key
  (`model.family`) switches between them for head-to-head ablation.
- **Face branch** — video → 32 uniform frames → MTCNN 224×224 crops →
  EfficientNet-B0 → mean-pool → binary head, with GradCAM + per-frame timeline
  explainability and a versioned `DetectionResult` JSON API.
- **Multi-target router (ADR-006)** — combines the face path with a
  general-content path (max-score when a face is present, general-only
  otherwise).

---

## 4. Models used

| Branch | Model(s) | Role | Status |
|--------|----------|------|--------|
| General video | ConvNeXt-Tiny / EfficientNet / Swin + temporal pool | spatial baseline | 🟢 |
| General video | **VideoMAE-base**, **TimeSformer** (HF) | spatio-temporal transformer | 🟢 |
| Face detect | **MTCNN** (facenet-pytorch) | *locate* faces (not classify) | ✅ |
| Face forensics | **EfficientNet-B0** + mean-pool | real/fake face | ✅ trained |
| Face forensics | Transformer temporal head, RGB+FFT dual-stream | experiments | ✅ (negative results kept) |
| Human / object | ViT / ConvNeXt on detected crops | region AIGC | ⏳ |
| Fusion | MLP / transformer over branch features | learned combine | ⏳ |

Model selection is **empirical** — validation + cross-generator performance
decides, not model size (Part §5 of the spec).

---

## 5. Research work & findings

Documented experiments (all reproducible, negatives kept — no cherry-picking):

| Experiment | Finding |
|-----------|---------|
| **Face baseline (M4)** | EfficientNet-B0 + mean-pool: FF++ test F1 **0.9862**, FPR **0.03**, AUC **0.9946** (tuned threshold 0.5872). Per-manipulation recall: DeepFakes 100%, Face2Face/FaceSwap 99%, NeuralTextures 94%. |
| **EXP-001 cross-dataset** | FF++→Celeb-DF v2: AUC **0.7076**, FPR **0.72** — a 29 pp AUC drop. Documents the domain-shift failure of FF++-only training (in the 60-75% band published for such models). |
| **EXP-002 dual-stream FFT** | Frequency stream did **not** improve FF++ accuracy nor close the Celeb-DF gap (+0.48 pp AUC = noise). Negative result kept. |
| **EXP-003 explainability** | GradCAM face-localisation 0.31 aggregate; model attends to face for reals (0.84) and boundary artifacts for fakes (0.05-0.25) — a known CNN-deepfake strategy. |
| **GenVidBench pipeline** | Cross-source-cross-generator protocol implemented and verified: train on seen generators, test on **unseen** ones (measures true generalization, not generator memorization). |

**Key research thesis:** the FF++ model overfits dataset-specific artifacts
(EXP-001). The general-video track on GenVidBench, with held-out-generator
evaluation, directly targets that generalization gap — the central research
contribution of this module.

---

## 6. Gaps (honest)

1. **General-video track untrained** — Phases 2-3 are code; no metric yet.
2. **Cross-dataset generalization weak** on the face track (Celeb-DF AUC 0.71) — the problem Phases 5+ must fix.
3. **Missing branches** — AI-human (B), object/region (E), learned fusion, and rich multi-branch explainability are planned, not built.
4. **Only Option-B data** (GenVidBench Pair1, cross-generator) is in progress; full cross-*source* (Pair2, ~194GB) is a later scale-up.
5. **Explainability is face-only** (GradCAM); the §14 rich JSON schema (per-branch scores, faces/humans/objects with bboxes) is not implemented.
6. **No robustness eval yet** (compression / resolution / FPS degradation).

---

## 7. Datasets

| Purpose | Dataset | Status |
|---------|---------|--------|
| General AI video | **GenVidBench** (Vript/HD-VG real; pika, vc2, ms, t2vz, cogvideo, mora, musev, svd fakes) | downloading (Pair1) |
| Face forensics | **FaceForensics++ c23** (DeepFakes, Face2Face, FaceSwap, NeuralTextures) | trained |
| Cross-dataset test | **Celeb-DF v2**, **DFDC** | present |
| Object/region pretrain | GenImage / COCO | planned |

GenVidBench facts are read from its **official label files** (authoritative),
not guessed. Splits are always video-level, deterministic, seeded, with
generator metadata preserved (no data leakage).

---

## 8. Repository layout

```
configs/            # video_training.yaml (general track) + model/train/dataset (face track)
data/               # raw videos, manifests, processed crops (git-ignored)
docs/
  architecture/     # ADR-001..007 (design decisions)
  experiments/      # EXP-001..003 (results)
  VIDEO_MODULE_MASTERPLAN.md   # full roadmap + resume runbook
scripts/            # prepare_genvidbench, train_video, evaluate_video (general);
                    # prepare_dataset, train, evaluate, predict (face)
src/
  preprocessing/    # frame extraction, MTCNN face detection
  datasets/         # VideoManifestDataset (general), FF++ face datasets
  models/           # video_spatial, video_transformer (general); baseline, temporal, dual_stream (face)
  training/         # video_trainer (general), trainer (face)
  evaluation/       # metrics
  explainability/   # GradCAM, timelines
  inference/        # single-video API, multi-target router
checkpoints/        # weights (git-ignored)
```

---

## 9. Environment setup (macOS Apple Silicon / Linux CUDA)

Python 3.11, `venv` (not conda). PyTorch uses default wheels (MPS on Apple
Silicon; CUDA build on Linux GPU).

```bash
cd Samsung_PRISM_Video_Detection
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt          # + requirements-dev.txt for tooling
```

Device auto-selects CUDA > MPS > CPU. VideoMAE/TimeSformer have known MPS op
gaps — run those on CUDA; the spatial baseline runs fine on MPS.

---

## 10. Quickstart

### General AI-video track (ADR-007) — the current focus
```bash
# 1. Build manifests from a downloaded GenVidBench subset (see MASTERPLAN)
python scripts/prepare_genvidbench.py --config configs/video_training.yaml

# 2. Train the spatial baseline (ConvNeXt + attention pool)
python scripts/train_video.py --config configs/video_training.yaml --name exp01

# 3. Evaluate with per-generator (seen vs unseen) breakdown
python scripts/evaluate_video.py --checkpoint checkpoints/best_model.pt

# 4. Swap to a video transformer: set model.family: videomae (or timesformer)
#    in configs/video_training.yaml, then re-run steps 2-3 and compare.
```
Full step-by-step (download → extract → manifests → train → compare) is in
[`docs/VIDEO_MODULE_MASTERPLAN.md`](docs/VIDEO_MODULE_MASTERPLAN.md).

### Face deepfake track (complete)
```bash
python scripts/prepare_dataset.py manifest          # build FF++ manifest
python scripts/prepare_dataset.py faces             # extract face crops
python scripts/train.py                             # train baseline
python scripts/evaluate.py --split test --tune-threshold-fpr 0.05
python scripts/evaluate.py --dataset celeb_df_v2 --split test   # cross-dataset
python scripts/predict.py --video path/to/clip.mp4  # single-video + GradCAM
```

Artefacts: `checkpoints/*.pt`, `reports/*.json`, `experiments/<name>/`,
`outputs/explainability/<clip>/`.

---

## 11. Success targets (worklet spec)

| Metric | Target | Face track | General track |
|--------|--------|-----------|---------------|
| F1 | ≥ 0.92 | ✅ 0.986 (FF++) | ⏳ pending |
| FPR | ≤ 5% | ✅ 3% (FF++) | ⏳ pending |
| Cross-dataset generalization | validated | 🟠 0.71 AUC (weak) | ⏳ unseen-generator eval ready |
| Explainability | ≥ 85% | 🟠 face-localisation 0.31 | ⏳ |

Numbers are only reported once measured. Every architecture change is
compared against the previous baseline; regressions are documented, not
hidden (project philosophy §22).
