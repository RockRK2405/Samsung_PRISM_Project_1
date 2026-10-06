# Samsung PRISM 26TS08 — Video Module: Complete Work Log

**Purpose of this file.** A self-contained record of *everything built, tested
and measured* in the video module, written so an AI assistant (or a new
teammate) can be handed this single file and reason about the project with no
repo access and no prior chat history.

**Last verified against the code:** 2026-10-06
**Verification method:** every claim below was read out of the repository at
commit `3788617` on branch `claude/quirky-goodall-3ny0im`. Numbers come from
committed experiment docs and metrics JSON, not from memory.

---

## How to read this file

Three status vocabularies are used, and they mean different things:

| Marker | Meaning |
|---|---|
| ✅ **Measured** | A number exists in a committed report/doc. Trustworthy. |
| 🟢 **Built, unrun** | Code exists and imports, but was never executed on real data. No number. |
| ⏳ **Not started** | Planned only. |
| ⚠️ **Broken** | Code exists but cannot work as written. Defect documented. |

**The single most important fact:** the research pipeline is validated and has
real results, but the **deployed system detects nothing** — not because the
models are weak, but because no trained weights exist in the repo and the
serving code cannot load the best model's architecture. Details in §7.

---

## 1. Identity and scope

- **Worklet:** Samsung PRISM **26TS08** — "A Cost-Aware Framework for Synthetic
  Data Detection in Data Acquisition Pipelines."
- **Owner of this module:** Rudra Khale (VIT, PRISM intern). Owns the **VIDEO**
  module. Teammates own audio / text / image / fusion.
- **Personal repo:** `github.com/RockRK2405/Samsung_PRISM_Project_1`
  - Current working branch: `claude/quirky-goodall-3ny0im`
  - Earlier branch: `claude/samsung-prism-video-detection-kctonh`
- **Samsung enterprise delivery target:**
  `github.ecodesamsung.com/SRIB-PRISM/26TS08VITV_A_cost-Aware_Framework_for_synthetic_Data_Detection_in_Data_Acquisition_Pipelines`
  - Fork-and-PR workflow only (fork access, not direct push).
  - **PR #20 (video module) is MERGED into `SRIB-PRISM:main`**, delivered under
    a folder named `Video_Dashboard/`.
  - Passed Samsung's **AVAS** dependency scan after pinning upper bounds.

### Repo layout

```
Samsung_PRISM_Project_1/
├── VIDEO_MODULE_WORK_LOG.md          ← this file
├── PROJECT_STATUS.md                 ← ⚠️ STALE (2026-07-23, pre-pivot). Ignore.
├── Samsung_PRISM_Video_Detection/    ← models, training, experiments (6,651 LOC Python)
└── Samsung_PRISM_Video_Dashboard/    ← FastAPI + React + Docker (1,082 backend / 1,155 frontend LOC)
```

---

## 2. The scope pivot (most important context for any decision)

The module began as a **face-deepfake detector only**. Mid-project it was
**redefined, with approval, to a GENERAL synthetic-video detector**, because
face-only detection structurally misses:

- (A) fully AI-generated video
- (B) AI-generated humans with no clear face
- (C) face deepfakes ← *the original and now only one* of six cases
- (D) AI-edited real video
- (E) synthetic objects/regions inside real video
- (F) temporal artifacts

**General AI-video detection is PRIMARY. Face deepfake is ONE branch.**

This pivot was later **vindicated empirically** — see §5, EXP-004 vs EXP-005.

### Standing rules set by the owner

1. For the video model, **do NOT optimize inference cost / latency / GPU
   memory.** Priority is **detection quality + generalization.** Heavy models
   are fine. (Cost-awareness is a *fusion-engine/router* concern, not this
   model's — despite the worklet title.)
2. **Do not invent dataset details.** Inspect real metadata; say "not verified"
   when unknown.
3. **Do not claim improvement without experimental evidence.** Always report
   baseline vs new. **Keep negative results** and state them plainly.
4. Don't conflate the tasks: face *detection* ≠ face *forensics* ≠ general
   video AIGC ≠ object forensics.

---

## 3. Architecture — the multi-branch design

Shared recipe per branch: sample frames → backbone → per-frame score →
aggregate to a video-level score. Branches stay separate; a learned fusion head
(Phase 9) will combine them later.

| Branch | Status | Method |
|---|---|---|
| **General AI-video** | ✅ Measured | `VideoSpatialNet`: ConvNeXt-Tiny (`convnext_tiny.fb_in22k`) + **attention-pool** temporal head + linear classifier |
| **Face forensics** | ✅ Measured | MTCNN face crop → per-frame CNN (`BaselineDetector`, timm backbone, 2-class head + mean-pool) |
| Video transformers | 🟢 Built, unrun | `VideoTransformerNet` — HF VideoMAE / TimeSformer, lazy-imported |
| Dual-stream RGB+FFT | ✅ Measured (negative) | `DualStreamDetector` — two EfficientNet-B0, concat features |
| Temporal head | ✅ Measured (negative) | `TemporalDetector` — transformer over frame features |
| Synthetic-human | ⏳ Not started | Planned: person detector → human crop → AIGC classifier |
| Object/region forensics | ⏳ Not started | Phase 7 |
| Temporal forensics | ⏳ Not started | Phase 8 |
| Learned fusion | ⏳ Not started | Phase 9 |
| Rich explainability | 🟠 Partial | GradCAM works; rich per-branch JSON schema is Phase 10 |

### Key source files (`Samsung_PRISM_Video_Detection/src/`)

| File | What it is |
|---|---|
| `models/baseline.py` | `BaselineDetector` — timm backbone-agnostic, `num_classes=2`, per-frame `forward`, `forward_video` mean-pools. Used by **both** the face path and `train_general.py`. |
| `models/video_spatial.py` | `VideoSpatialNet` + `build_video_model_from_config()` factory (dispatches on `model.family`). **The best model lives here.** |
| `models/video_transformer.py` | `VideoTransformerNet` (HF VideoMAE / TimeSformer) |
| `models/dual_stream.py`, `models/temporal.py` | EXP-002 / Milestone-5A heads (both negative results) |
| `models/__init__.py` | `build_model_from_config()` — the **serving** factory. ⚠️ Cannot build `VideoSpatialNet` (see §7.2). |
| `datasets/general_image_dataset.py` | `GeneralImageDataset` + `ForensicDegrade` aug (JPEG q40-90, resolution 0.4-0.9, mild blur) |
| `datasets/video_dataset.py`, `datasets/ff_dataset.py` | Clip loader; FF++ face-crop loader |
| `training/video_trainer.py` | AMP (CUDA-only), grad-accum, staged unfreeze, early stop on val AUC |
| `inference/predictor.py` | `VideoDetector` — what the dashboard loads. ⚠️ face-only, see §7 |
| `inference/multi_target_predictor.py` | `MultiTargetVideoDetector` — face/general router (ADR-006). ⚠️ see §7 |
| `explainability/` | GradCAM, overlays, per-frame timeline, explainability scorer |

### Scripts (`scripts/`)

`prepare_genvidbench.py` (authoritative-label GenVidBench prep, verified zero
leakage) · `prepare_face_dataset_kaggle.py` (MTCNN crops for FF++/Celeb-DF/DFDC,
`--holdout <tag>` routes a dataset to test) · `train_general.py` ·
`train_video.py` · `evaluate_video.py` · `eval_face_crops.py` (crop-level
cross-dataset eval + val-tuned threshold for target FPR) · `export_subset.py` ·
plus the original face-track `train.py` / `evaluate.py` /
`evaluate_explainability.py` / `prepare_celeb_df*.py` / `extract_frames.py`.

### Architecture Decision Records (`docs/architecture/`)

| ADR | Subject |
|---|---|
| ADR-001 | Baseline dependency set |
| ADR-002 | Frame sampling strategy + face-crop size |
| ADR-003 | Temporal aggregation head — transformer over mean-pool |
| ADR-004 | Dual-stream (RGB + frequency) detector |
| ADR-005 | Explainability layer |
| ADR-006 | Multi-target detection — faces, humans AND general objects/scenes |
| ADR-007 | General AI-generated video detection (GenVidBench) |

**48 test functions** across 11 test files (`tests/`).

---

## 4. Datasets (all verified against real metadata — not guessed)

### General branch — GenVidBench
HuggingFace `jian-0/GenVidBench`. Ships as monolithic `.rar`/`.7z`.
Official label files `Pair1_labels.txt` / `Pair2_labels.txt`, format
`<rel_path> <label>` (1=fake, 0=real); generator = 2nd path component.

| Group | Fakes | Reals |
|---|---|---|
| **Pair1** (VidProM prompts) | pika, videocrafter2 (vc2), modelscope (ms), text2video-zero (t2vz) — ~13.5k each | vript (20,131) |
| **Pair2** (HD-VG) | svd, musev, mora, cogvideo — ~13.5k each | hd_vg_130m (13,416) |

Archive sizes: Pair1 ≈ 74 GB, Pair2 ≈ 120 GB (194 GB total).
Folder abbreviation map: `ms`→modelscope, `vc2`→videocrafter2,
`t2vz`→text2video-zero, `hd_vg_130m`→hd-vg-130m.

**Active protocol "Option B":** Pair1 only, **cross-generator within Pair1** —
train on pika + vc2 + ms + vript; test on held-out **t2vz** + disjoint vript.
`prepare_genvidbench.py` has `_partition_reals()` for a disjoint train/test real
split; **zero leakage verified**.
*Option A* (full 194 GB, cross-**source**: train Pair1 → test Pair2) is a
documented later scale-up; the config carries both.

> ⚠️ **Generation gap.** All Pair1 generators are **2023-era** text-to-video.
> Nothing in training resembles Sora / Veo / Kling / Runway Gen-3. Performance
> on 2025-26 content is **unmeasured**.

### Face branch — three standard deepfake datasets
Kaggle mount paths verified 2026-09-20:

| Dataset | Path | Contents |
|---|---|---|
| **FF++ c23** | `/kaggle/input/datasets/ahmedelbanby/faceforensicsplusplus-c23-deepfakebench-structure/videos/FaceForensics++` | `original_sequences/` 1000 real + `manipulated_sequences/` 5 methods, 5000 fake |
| **Celeb-DF v2** | `/kaggle/input/datasets/reubensuju/celeb-df-v2` | `Celeb-real/` 590 + `YouTube-real/` 300 real; `Celeb-synthesis/` 5639 fake |
| **DFDC** | `/kaggle/input/datasets/pranay22077/dfdc-10` | 10 parts, per-part `metadata.json` → REAL/FAKE; ~2512 real / 17397 fake |

### Evaluation methodology (applied throughout)
Always test on data **disjoint from training** — unseen *generator* or unseen
*dataset*, never a random-frame split. Video-level splits only; deterministic
seeded manifests. Full suite reported every time: accuracy, precision, recall,
F1, ROC-AUC, PR-AUC, FPR, FNR, confusion matrix.
**Robustness sweeps (JPEG/resize/blur/noise/FPS) are planned but never run.**

---

## 5. Experiments — all results, positive and negative

### 5.1 Face-deepfake track (chronological)

The face track ran as "Milestones". Its production checkpoint was
`checkpoints/best.pt` — **EfficientNet-B0 + mean-pool**, val F1 0.9875.

| Milestone | What was tried | Outcome |
|---|---|---|
| M4 | EfficientNet-B0 + mean-pool baseline | FF++ test: F1 **0.9862**, AUC **0.9946**, FPR 0.03 |
| M5A | Transformer temporal head (ADR-003) | **Negative** — did not beat mean-pool |
| M6 | Dual-stream RGB + FFT (ADR-004) | **Negative** — EXP-002 below |
| M7 | GradCAM explainability (ADR-005) | Built; EXP-003 below |
| M8 | Fusion-engine integration hooks | Built — `DetectionResult` wire format |

**EXP-001 — FF++ → Celeb-DF v2 cross-dataset** (2026-07-07)
Threshold 0.5872 (tuned on FF++ val for FPR ≤ 5%). 518 Celeb-DF test videos.

| Metric | FF++ (in-dist) | Celeb-DF (cross) | Δ |
|---|:-:|:-:|:-:|
| F1 | 0.9862 | 0.8142 | −17 pp |
| **ROC-AUC** | **0.9946** | **0.7076** | **−29 pp** |
| **FPR** | **0.03** | **0.7247** | **+69 pp** |
| Fake recall | 0.98 | 0.9471 | −3 pp |

Per-source correct-rate: Celeb-synthesis (fake) 0.9471 · Celeb-real 0.2963 ·
YouTube-real 0.2429.

**Reading:** failure is *one-sided* — the model catches 94.7% of fakes but
mis-flags **70–76% of real videos as fake**. 0.71 AUC sits squarely inside the
60–75% band the literature reports for FF++-only models on Celeb-DF, so this is
the *normal* documented cross-dataset gap, not a bug. Threshold tuning cannot
fix it; AUC is the ceiling. **F1 = 0.81 is misleadingly optimistic** (class
imbalance + fake-happy bias) — quote AUC, not F1.

**EXP-002 — Dual-stream RGB + FFT** (2026-07-07). Hypothesis: GAN frequency
fingerprints persist across generators → better cross-dataset AUC. Target ≥ 0.75.

| Metric | Baseline (M4) | Dual (M6) | Δ |
|---|:-:|:-:|:-:|
| FF++ F1 | 0.9862 | 0.9862 | **0.00 (bit-for-bit tied)** |
| Celeb-DF AUC | 0.7076 | 0.7124 | +0.48 pp (**within noise**, bootstrap σ ≈ 2 pp) |
| Celeb-DF FPR | 0.7247 | 0.7697 | −4.5 pp (**worse**) |

**NEGATIVE result.** In-distribution metrics identical to four decimals — the
frozen RGB backbone dominated and the FFT stream contributed ~zero. Training
loss was flat from epoch 1 (0.127 → 0.121 over 5 epochs). Root causes diagnosed:
(a) frozen-RGB bias left no gradient pressure to use noisy FFT features;
(b) FFT-fingerprint literature targets 2018-20 GANs, not FF++/Celeb-DF
pipelines. Baseline `best.pt` remained production.

**EXP-003 — Explainability metric** (2026-07-08). GradCAM peak-inside-central-70%
window, 20-video stratified FF++ subset.

| Bucket | n | Score |
|---|:-:|:-:|
| original (real) | 4 | **0.8359** |
| Deepfakes | 4 | 0.2500 |
| Face2Face | 2 | 0.0469 |
| FaceSwap | 6 | 0.1458 |
| NeuralTextures | 4 | 0.2188 |
| **Aggregate** | **20** | **0.3094** |

**The metric mismatches the model's strategy, and that is the finding.** On
*real* videos the model attends to face interior (0.84 — as expected). On
*fakes* it attends to **blending seams and boundary regions** — which matches
Rössler+ 2019 onward: deepfake artifacts concentrate at boundaries, not the
face interior. A face-centric metric therefore *penalises correct behaviour*.
Three options were weighed; widening the window to hit 0.85 was **rejected as
metric-fiddling**, and honest reporting was adopted.

**EXP-005 — Face cross-dataset with more data + forensic aug** (2026-09-20)
Train FF++ c23 **+** Celeb-DF v2 with `ForensicDegrade`; hold out **DFDC**
entirely. Extraction: 600 videos/class/dataset, 6 frames/video →
12,233 train / 2,159 val / 7,158 test crops.

| Backbone | Val AUC (in-dist) | **DFDC AUC** | Recall | FPR | FNR |
|---|:-:|:-:|:-:|:-:|:-:|
| Xception (`legacy_xception`) | 0.9913 | 0.6407 | 0.2904 | 0.1258 | 0.7096 |
| **ConvNeXt-Tiny** ⭐ | 0.9890 | **0.6564** | 0.2960 | 0.1228 | 0.7040 |

**NEGATIVE result.** ~0.99 val AUC collapses to ~0.65 on unseen DFDC; recall
~0.29 means **the model misses ~71% of DFDC fakes.** Forensic augmentation kept
FPR low (~0.12) but **did not transfer recall**.

> **Honest caveat (important — do not overstate this):** EXP-005 is **not**
> apples-to-apples with EXP-001. EXP-001 held out Celeb-DF; EXP-005 held out
> DFDC *and had Celeb-DF in training*. So EXP-005 is **not a claimed
> regression**. The defensible claim is only: *"cross-dataset generalization for
> the face branch is weak and persists (~0.65) even with more data plus
> forensic augmentation."* DFDC is the hardest of the three benchmarks.

### 5.2 General AI-video track

**EXP-004 ⭐ — GenVidBench cross-generator baseline** (2026-09-19)
`VideoSpatialNet` = ConvNeXt-Tiny (28.1M params) + attention-pool + linear head.
Option B split: train pika + vc2 + ms + vript (10,000 videos); val = held-out
videos from the *same* generators (2,000); **test = text2video-zero (UNSEEN
generator) + disjoint vript (2,000)**. 16 frames/video, 224×224, video-level
splits, zero leakage verified.

Training: **1 epoch, backbone FROZEN** (head-only), batch 8, AdamW lr 1e-4, MPS.

| Metric | Test (UNSEEN generator t2vz) |
|---|:-:|
| Accuracy | 0.8675 |
| Precision | 0.9016 |
| Recall | 0.8250 |
| **F1** | **0.8616** |
| **ROC-AUC** | **0.9426** |
| PR-AUC | 0.9422 |
| FPR | 0.0900 |
| FNR | 0.1750 |

Per-group @ 0.5: t2vz (unseen fake) recall 0.8250, mean score 0.688 ·
vript (real) specificity 0.9100, mean score 0.222.
Val (seen generators): acc 0.8975, F1 0.8982, AUC 0.9632.

> **This number is a FLOOR, not a converged result.** Only epoch 1 finished.
> Epoch 2 (backbone unfrozen) thrashed MPS memory (~6× slower) and was stopped;
> the epoch-1 checkpoint was evaluated. A proper CUDA fine-tune is untried and
> is the cheapest available gain.

### 5.3 The headline finding — state this to mentors

> The **general branch generalizes to an unseen *generator* at ROC-AUC 0.9426**,
> while the **face branch generalizes to an unseen *dataset* at only 0.65–0.71
> ROC-AUC** — even after adding a second training dataset and forensic
> augmentation.

That is the **empirical justification for the §2 pivot**: general synthetic-video
detection generalizes where face-artifact-specific detection did not. Note the
comparison is *unseen generator* vs *unseen dataset* — related but not identical
axes of generalization; say so rather than implying a controlled comparison.

### 5.4 Experiment scoreboard

| Exp | Track | Result | Verdict |
|---|---|---|---|
| EXP-001 | Face, FF++→Celeb-DF | AUC 0.7076, FPR 0.7247 | Expected gap |
| EXP-002 | Face, dual-stream FFT | AUC 0.7124 (noise), FPR worse | **Negative** |
| EXP-003 | Face, explainability | Aggregate 0.3094; real 0.84 vs fake 0.05-0.25 | Metric mismatch, reported honestly |
| **EXP-004** ⭐ | **General, unseen generator** | **AUC 0.9426** | **Best result in project** |
| EXP-005 | Face, +data +aug, →DFDC | AUC 0.6564 best, recall 0.296 | **Negative** |

**Three of five experiments are negative results, kept and documented.** That is
deliberate per the §2 rules and is itself a deliverable.

Docs: `docs/experiments/EXP-00{1..5}-*.md`.
Metrics JSON: `reports/EXP-005-face-crossdataset.json` (the only metrics JSON
committed to git).

---

## 6. The dashboard (`Samsung_PRISM_Video_Dashboard/`)

FastAPI backend + React (Vite + Tailwind) frontend, Docker-ready
(`docker compose up --build` → `http://localhost:8000`).

**Pages:** Dashboard · TestVideo · BatchTesting · CompareVideos · Evaluation ·
History · ModelInfo. Charts via a shared `Charts.jsx`. SQLite experiment history
under `outputs/` (bind-mounted, survives restarts).

**Integration design (good):** `backend/services/model_service.py` is the **one**
integration seam. It adds the sibling detection module to `sys.path` and imports
`src.inference` — the same pattern the fusion engine uses. `config/config.yaml`
holds everything model-related; nothing is hardcoded in the app.

Checkpoint paths it expects (relative to `video_module_path`):
- face: `checkpoints/best.pt`
- general: `checkpoints/general.pt`
- `use_multi_target: true` → uses `MultiTargetVideoDetector` when both exist

**Mock fallback (honest, and currently always active):** if a checkpoint can't
load, it serves a **clearly-marked mock** (`"mock": true`) so the UI still runs.
It never dresses mock output up as real. See §7.1 for why this is always on.

**Engineering fixes already made:** serialized model access behind a lock to fix
an MPS/CPU device race on the Compare page; lean CPU-only Docker deps to fix a
build timeout/bloat.

---

## 7. ⚠️ CURRENT STATE: why the deployed system detects nothing

Diagnosed 2026-09-26, verified against code. **Full write-up with file:line
citations: `Samsung_PRISM_Video_Detection/docs/DETECTION_ENABLEMENT_PLAN.md`.**
Seven defects; the cause is **integration, not model quality.**

**7.1 — No checkpoints exist → every result is a hash of the filename.**
`checkpoints/` holds only `.gitkeep` (and `*.pt` is git-ignored by design, so
weights never arrive with a clone). `_RealModel.__init__` raises
`FileNotFoundError` → `init_model()` catches it → `_MockModel` returns
`(sha256(filename) % 1000) / 1000`. **No video has ever been classified by the
deployed system.** Tell-tale: rename the file, get a different verdict.

**7.2 — The 0.94-AUC model is structurally unservable.** There are **two
non-communicating model factories**:

| Path | Factory | Can build |
|---|---|---|
| Research (`train_video.py`, `evaluate_video.py`) | `build_video_model_from_config` | `VideoSpatialNet`, `VideoTransformerNet` |
| **Serving** (`inference/predictor.py`) | `build_model_from_config` | `BaselineDetector`, `TemporalDetector`, `DualStreamDetector` |

The serving factory dispatches only on `frequency_stream.enabled` and
`temporal_head.type` — it has **no `VideoSpatialNet` branch** (the class is
imported and re-exported in `models/__init__.py` but never in the dispatch
body). **EXP-004's winner has no inference code path at all.**

**7.3 — General path hardcoded to the wrong class.** `multi_target_predictor.py`
builds the general model as `BaselineDetector(...)` unconditionally, ignoring
the checkpoint's embedded `model.family`. A `VideoSpatialNet` `general.pt` fails
`load_state_dict` (keys `pool.score.*`, `classifier.0.*` don't exist there).

**7.4 — The strongest branch is gated behind the weakest.** The router builds
the **face** detector first in `__init__`, and `model_service` raises when
`best.pt` is missing. So general-video detection **cannot run without a face
checkpoint** — the 0.94-AUC branch requires the 0.65-AUC branch to exist.

**7.5 — Temporal modelling discarded at inference.** `_score_general` scores
frames independently and returns `np.mean(...)` — throwing away the **trained
attention-pool head**. Even once 7.2/7.3 are fixed, the model would be served
under an aggregation it was never trained with (train/inference skew).

**7.6 — Face path crashes on faceless video.** `predictor.py` raises
`RuntimeError("No face detected...")`. A fully AI-generated clip has no face by
construction — the exact target case. The router guards this via a coverage
threshold, but bare `VideoDetector` (the fallback when `general.pt` is absent)
does not.

**7.7 — Thresholds inconsistent and uncalibrated.** `config.yaml` sets
`decision_threshold: 0.5872` (tuned on **FF++ face crops**); the router
**ignores it** and hardcodes `0.5`; and 0.5872 is meaningless for the general
model, which was never calibrated on GenVidBench. Fusion is `max(face, general)`
— chosen in ADR-006 for recall but **never recalibrated**, so it compounds FPR
(≈0.09 and ≈0.12 per branch → up to ~0.20 if independent). **Deployed FPR is
unknown.**

**7.8 — No image support.** `allowed_extensions` is video-only and
`validate_upload` rejects everything else; there is no image entry point.
*But* `BaselineDetector` and `GeneralImageDataset` are already **per-image**
classifiers — an image is just `T=1`. The model side is ready; only the
API/validation layer blocks it.

**Secondary:** resize skew (training `transforms.Resize` bilinear vs inference
PIL `Image.resize()` bicubic) · `predictor.py` omits `weights_only` (fine under
the pinned `torch<2.6`, breaks when that lifts) · `PROJECT_STATUS.md` is stale
and describes a removed 4-modality era · no checkpoint→experiment provenance
record, which is how results get orphaned when weights live off-git.

---

## 8. Checkpoint inventory — where the weights actually are

**Nothing is in git.** `*.pt` and `checkpoints/*` are git-ignored (deliberate).

| Checkpoint | Produced by | Metrics | Location |
|---|---|---|---|
| `best.pt` | M4 face baseline | val F1 0.9875; FF++ AUC 0.9946 | Local/Kaggle only — **not in repo** |
| `best_dual.pt` | EXP-002 dual-stream | val F1 0.9887 | Not in repo |
| `best_model.pt` | **EXP-004 general** ⭐ | val AUC 0.9632, test AUC 0.9426 | Kaggle committed-run output |
| `face_xception.pt` | EXP-005 | DFDC AUC 0.6407 | Kaggle committed-run output |
| `face_convnext.pt` | EXP-005 ⭐ face | DFDC AUC 0.6564 | Kaggle committed-run output |

Also from the EXP-005 Kaggle run: `reports/xception_dfdc.json`,
`reports/convnext_dfdc.json`, and **`face_crops.zip`** (extracted MTCNN crops +
manifests — saved specifically so the 3-hour extraction need not repeat).
Suggested git-ignored local home for the crops:
`Samsung_PRISM_Video_Detection/data/raw/face_crops_exp005/`.

> **Highest-leverage housekeeping:** there is **no record mapping checkpoint →
> experiment → dataset → metrics → run URL → sha256.** Create
> `checkpoints/MANIFEST.md` (git-tracked; weights stay off-git).

---

## 9. What to do next

Full phased plan: `Samsung_PRISM_Video_Detection/docs/DETECTION_ENABLEMENT_PLAN.md`.
Ordering principle: **make one real model run end-to-end before improving any
model.**

| Phase | Work | Effort | Needs GPU? |
|---|---|---|---|
| **0** | **Unblock serving** — unify the two factories; clip-aware scoring (preserve attention-pool); decouple branches; honour embedded config; return `unassessable` instead of raising; align resize | ~1 day | No |
| **1** | **Recover or retrain a checkpoint** — pull Kaggle outputs; verify EXP-004 reproduces **0.9426 ± 0.01**; add `checkpoints/MANIFEST.md` | 1 day (or ~4 GPU-h) | Only if retraining |
| **2** | **First true end-to-end detection + image support** — smoke-test ~20 clips, add `.jpg/.png/.webp` (`T=1`), surface which branch ran. **Write up as EXP-006.** | ~1 day | No |
| **3** | **Calibrate** — per-branch thresholds on each branch's own val split for FPR ≤ 5%; measure the `max()` fusion vs noisy-OR / weighted mean / coverage-gated routing; reliability curves. **EXP-007** | ~1 day | No |
| **4** | **Close the generation gap** — build a held-out 2025-26 eval set (Sora/Veo/Kling/Runway/modern faceswaps); measure the drop; **then** finish EXP-004's unfinished fine-tune on CUDA (unfreeze, 16→32 frames); head-to-head vs VideoMAE/TimeSformer. **EXP-008** | 1-2 wk | Yes |
| **5** | **Robustness** — JPEG/resize/blur/noise/frame-drop/FPS sweeps. **EXP-009** | ~3 d | Some |
| **6** | **Face branch: fix or formally de-scope** — run the prepared Celeb-DF holdout for the apples-to-apples vs EXP-001's 0.71 that EXP-005 couldn't give; try video-level temporal aggregation over face crops (EXP-005 was crop-level, discarding temporal identity drift). If still ~0.65, **declare it a low-weight auxiliary signal** — a documented negative is a valid deliverable. **EXP-010** | ~1 wk | Yes |
| 7-11 | Master-plan branches: synthetic-human → object/region → temporal forensics → learned fusion → rich explainability. **If time is short: temporal forensics → human → fusion.** | — | Yes |

### Two open risks worth stating up front
1. **Generation gap (§4).** Training generators are 2023-era. 2025-26 content
   performance is unmeasured and is the most likely reason detection would still
   feel weak after Phase 0-3.
2. **Faceswap coverage.** The general branch learned *fully generated* video. A
   faceswap is **real video with a locally edited region** — a different signal.
   Only the face branch targets it, and that is the weak branch. May need
   local/patch-level supervision rather than one video-level label.

---

## 10. Gotchas already paid for — do not re-hit these

- **MPS (Apple Silicon):** `torch.autocast(device_type='mps')` raises → use
  `contextlib.nullcontext()` when not CUDA. **MTCNN returns `None` for every
  frame on MPS → pin it to CPU.** Full-backbone video training thrashes MPS
  memory (~6× slower) — this is what truncated EXP-004 at one epoch.
- **Kaggle interactive sessions WIPE `/kaggle/working`** on idle-timeout / the
  "still using this session?" prompt. **A 3-hour extraction was lost this way
  twice.** Fix: run long jobs as a **committed notebook** ("Save & Run All /
  Commit") — headless, survives disconnects, output downloadable as a version.
- **Decoupled imports:** `src/training/__init__.py` and
  `src/preprocessing/__init__.py` guard `mlflow` / `facenet_pytorch` in
  try/except so the general track imports without face-track deps.
- **macOS auto-unzips downloads** → `.pt`/`.zip` arrive as *folders*. Move the
  folder into place; don't unzip again.
- **timm backbone names that work:** `legacy_xception` (20.8M),
  `convnext_tiny.fb_in22k_ft_in1k` (27.8M), `tf_efficientnet_b0.ns_jft_in1k`.
- **AVAS dependency scan** requires upper bounds: `torch>=2.2,<2.6`,
  `torchvision>=0.17,<0.21`, `transformers>=4.40,<5`, `mlflow>=2.13,<3`.
- **Enterprise GitHub auth:** Samsung username + a PAT generated **on the
  enterprise instance**, SSO-authorized for the SRIB-PRISM org. The github.com
  account does not exist on the enterprise server.

---

## 11. Project history in one pass

The repo briefly hosted **all four modalities** — audio, image, text and a v1
fusion engine (`Samsung_PRISM_Fusion_Engine/`) were added for integration, then
removed in commit `a85c52c` *"chore: remove non-video modules — keep only Video
Detection + Dashboard"*. **`PROJECT_STATUS.md` still describes that era** and is
the one actively misleading file in the repo.

Arc: face baseline (M4) → two negative upgrade attempts (M5A temporal, M6
dual-stream) → explainability (M7) → fusion hooks (M8) → cross-modal fusion era
→ **scope pivot to general AI-video** → GenVidBench pipeline (ADR-007) →
EXP-004 ⭐ → face cross-dataset retry (EXP-005, negative) → Samsung PR #20
merged → serving-path audit (2026-09-26).

---

## 12. Conventions

- Develop and push on branch `claude/quirky-goodall-3ny0im`. Do **not** open PRs
  on the personal repo unless asked.
- Every result gets an **EXP-NNN doc + a metrics JSON**. **Negative results are
  kept.**
- No metric is claimed until actually measured. No fabricated dataset paths,
  labels or checkpoints.
- Never put model identifiers in commits / PRs / code — chat only.

---

## 13. Quick answers to likely questions

**"Does it detect deepfakes right now?"** No. It returns a hash of the filename
(§7.1). The cause is integration, not model quality.

**"What's the best result?"** EXP-004: ROC-AUC **0.9426** on a completely unseen
generator — general AI-video branch, frozen backbone, one epoch. A floor.

**"Is the face branch usable?"** Weakly. ~0.99 in-distribution, but 0.65–0.71
cross-dataset and it misses ~71% of DFDC fakes. Phase 6 decides fix vs de-scope.

**"How long to a working demo?"** Phases 0-2 ≈ 3 days, no GPU and no dataset
download, assuming the Kaggle checkpoints are recoverable.

**"What should I never claim?"** That EXP-005 shows a regression versus EXP-001
(different held-out sets — §5.1). That the deployed system has any accuracy
number (none has been measured). That the general branch handles faceswaps
(untested — §9).
