# PROJECT MEMORY — Samsung PRISM Video Detection Module (26TS08)

> Portable handoff file. Everything a fresh Claude session (or a new account)
> needs to continue this project without the prior chat history. Written
> 2026-09-25. Read this top-to-bottom before doing any work.

---

## 0. Who / what / where

- **Owner:** Rudra Khale (VIT student, PRISM intern).
  - Personal GitHub: `RockRK2405` (github.com), email `khalerishabhproff@gmail.com`.
  - Samsung Enterprise GitHub: `rudra-khale2023` on `github.ecodesamsung.com`.
- **Worklet:** Samsung PRISM **26TS08** — "A Cost-Aware Framework for Synthetic
  Data Detection in Data Acquisition Pipelines." A multimodal platform (image /
  audio / text / video) that flags AI-generated content submitted as genuine
  data. Rudra owns the **VIDEO** module (a teammate owns audio; others own
  text/image/metadata/fusion).
- **Personal repo:** `github.com/RockRK2405/Samsung_PRISM_Project_1`
  - Working branch: **`claude/samsung-prism-video-detection-kctonh`**
  - Layout: two sibling top-level folders —
    - `Samsung_PRISM_Video_Detection/` (the model + training + experiments)
    - `Samsung_PRISM_Video_Dashboard/` (FastAPI backend + React frontend, Docker)
    - plus `PROJECT_STATUS.md`
- **Samsung Enterprise repo (delivery target):**
  `github.ecodesamsung.com/SRIB-PRISM/26TS08VITV_A_cost-Aware_Framework_for_synthetic_Data_Detection_in_Data_Acquisition_Pipelines`
  - Rudra has only fork access → fork-and-PR workflow.
  - **PR #20 (video module) is MERGED into SRIB-PRISM:main.** Delivered under a
    folder named `Video_Dashboard/` containing both sibling folders.

---

## 1. The scope decision (most important context)

Originally scoped as a **face-deepfake detector only**. Mid-project this was
**redefined (with approval) to a GENERAL synthetic-video detector**, because
face-only detection misses: (A) fully AI-generated video, (B) AI-generated
humans without a clear face, (C) face deepfakes, (D) AI-edited real video,
(E) synthetic objects/regions, (F) temporal artifacts.

**General AI-video detection is PRIMARY. Face deepfake is ONE branch.**

### Hard rules the owner set (follow these)
- For the VIDEO model, **DO NOT optimize for inference cost / latency / GPU
  memory.** Priority is **DETECTION QUALITY + GENERALIZATION.** Heavy models OK.
  (Cost-awareness is a *fusion-engine/router* concern, not the video model's.)
- **DO NOT HALLUCINATE DATASET DETAILS.** Inspect real repos/metadata; say
  "Not verified yet" if unknown.
- **DO NOT CLAIM IMPROVEMENT WITHOUT EXPERIMENTAL EVIDENCE.** Compare baseline
  vs new; **keep negative results** and report them honestly.
- Don't confuse tasks: face detection ≠ face forensics ≠ general video AIGC ≠
  object forensics.

---

## 2. Architecture (multi-branch)

Same core recipe per branch: sample frames → CNN backbone → per-frame score →
aggregate to a video-level score. Branches stay separate; a learned fusion
head (future Phase 9) will combine them.

| Branch | Status | Method |
|---|---|---|
| **General AI-video** | ✅ built + validated | `VideoSpatialNet`: ConvNeXt-Tiny (`convnext_tiny.fb_in22k`) + **attention-pool** temporal head + linear classifier. Also supports VideoMAE/TimeSformer via `VideoTransformerNet` (HF, lazy import). |
| **Face forensics** | ✅ built + validated | MTCNN face crop → per-frame CNN (`BaselineDetector`, timm backbone, 2-class head + mean-pool). Backbones tried: EfficientNet-B0, Xception, ConvNeXt-Tiny. |
| **Synthetic-human** | 🔶 next / starting | Planned: person detector (YOLO/RTMDet) → human-region crop → AIGC classifier (ViT/ConvNeXt). Works when face branch can't (full body, back, distant, occluded). |
| Object/region forensics | ⏳ not started | Object detector → crop → AIGC forensic classifier (Phase 7). |
| Temporal forensics | ⏳ not started | Frame-to-frame anomaly signals (Phase 8). |
| Learned fusion | ⏳ not started | MLP/transformer over {general, temporal, face, human, object} scores (Phase 9). |
| Rich explainability | ⏳ partial | GradCAM exists; rich per-branch JSON schema is Phase 10. |

Key model files (`Samsung_PRISM_Video_Detection/src/models/`):
- `baseline.py` — `BaselineDetector` (timm backbone-agnostic; `num_classes=2`;
  `forward` per-frame, `forward_video` mean-pools). This is what BOTH the face
  path and `train_general.py` use.
- `video_spatial.py` — `VideoSpatialNet` + `build_video_model_from_config()`
  factory (dispatch on `model.family`: frame_baseline / videomae / timesformer).
- `video_transformer.py` — `VideoTransformerNet` (HF VideoMAE/TimeSformer).
- `dual_stream.py`, `temporal.py` — dual-stream FFT + temporal heads (EXP-002).

---

## 3. Datasets (verified, real)

### General branch — GenVidBench
- HuggingFace `jian-0/GenVidBench`. Large AI-video benchmark.
- **Pair1** (VidProM prompts): fakes = pika, videocrafter2 (vc2), modelscope
  (ms), text2video-zero (t2vz); real = vript.
- **Pair2** (HD-VG): fakes = svd, musev, mora, cogvideo; real = hd_vg_130m.
- Ships as monolithic `.rar`/`.7z`. Official label files
  `Pair1_labels.txt`/`Pair2_labels.txt`, format `<rel_path> <label>`
  (1=fake, 0=real); generator = 2nd path component.
- Folder abbrev map: ms→modelscope, vc2→videocrafter2, t2vz→text2video-zero,
  hd_vg_130m→hd-vg-130m.
- **Active protocol "Option B":** Pair1 only (~74GB), **cross-generator within
  Pair1** — train pika/vc2/ms + vript, test held-out **t2vz** + disjoint vript.
- Prep script `scripts/prepare_genvidbench.py` (reads official labels; has
  `_partition_reals()` for disjoint train/test real split; verified zero leakage).

### Face branch — three standard deepfake datasets (used on Kaggle)
Kaggle mount paths (verified 2026-09-20):
- **FF++ c23:** `/kaggle/input/datasets/ahmedelbanby/faceforensicsplusplus-c23-deepfakebench-structure/videos/FaceForensics++`
  (has `original_sequences/` real + `manipulated_sequences/` 5 methods fake; 1000 real / 5000 fake).
- **Celeb-DF v2:** `/kaggle/input/datasets/reubensuju/celeb-df-v2`
  (`Celeb-real/` 590 + `YouTube-real/` 300 real; `Celeb-synthesis/` 5639 fake).
- **DFDC:** `/kaggle/input/datasets/pranay22077/dfdc-10`
  (10 parts, per-part `metadata.json` gives REAL/FAKE; ~2512 real / 17397 fake).

---

## 4. Evaluation methodology

Always test on data **disjoint from training** (unseen generator OR unseen
dataset), never a random-frame split. Report the FULL suite every time:
accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR, confusion.
Cross protocols: FF++→Celeb-DF→DFDC (face); Pair1→Pair2 & cross-generator
(video). Robustness (JPEG/resize/blur/noise/FPS) is planned, not done.

---

## 5. Experiments & RESULTS (the actual findings — positive AND negative)

| Exp | Setup | Result |
|---|---|---|
| **EXP-001** | Face baseline, train FF++ → test **cross-dataset** Celeb-DF | AUC **0.71**, FPR 0.72 (poor generalization) |
| **EXP-002** | Dual-stream FFT head | see doc |
| **EXP-003** | Explainability metric | see doc |
| **EXP-004** ⭐ | **General branch**, train pika/vc2/ms+vript, test **UNSEEN generator t2vz** (frozen backbone, 1 epoch, MPS) | **AUC 0.9426, F1 0.8616, recall 0.8250, FPR 0.09**; vript specificity 0.91. Val (seen gens) AUC 0.9632. |
| **EXP-005** | **Face branch** retrained FF+++Celeb-DF **+ forensic aug**, backbones Xception vs ConvNeXt, test **held-out DFDC** | In-dist val ~0.99 AUC. Cross-dataset DFDC: Xception AUC **0.6407**, ConvNeXt AUC **0.6564** (winner), recall ~0.29, FPR ~0.12. **NEGATIVE result.** |

### The headline finding (say this to mentors)
The **general branch generalizes to an unseen *generator* at 0.94 AUC**, while
the **face branch generalizes to an unseen *dataset* at only 0.65–0.71 AUC**,
even after adding datasets + forensic augmentation. → Empirical justification
for making general detection primary, not face-only.

### EXP-005 honest caveat
Not apples-to-apples with EXP-001 (EXP-001 tested Celeb-DF; EXP-005 held out
DFDC, and Celeb-DF was in training). So EXP-005 is NOT a claimed regression —
the defensible claim is: "cross-dataset generalization for the face branch is
weak and persists (~0.65) even with more data + augmentation." DFDC is the
hardest benchmark. Forensic aug helped *calibration* (low FPR) not *recall*.

Docs: `docs/experiments/EXP-001..005-*.md`, metrics
`reports/EXP-005-face-crossdataset.json`. Full roadmap:
`docs/VIDEO_MODULE_MASTERPLAN.md`.

---

## 6. Key code files (video detection module)

- `configs/video_training.yaml` — central config. Option B active.
- `scripts/prepare_genvidbench.py` — GenVidBench prep (authoritative labels).
- `scripts/train_general.py` — trains `BaselineDetector` on image/frame data;
  `--backbone`, `--forensic-aug` flags; saves `checkpoints/*.pt` in the
  `{state_dict, config:{model:{...}}}` format `VideoDetector` expects.
- `scripts/prepare_face_dataset_kaggle.py` — MTCNN face-crop extractor for
  FF+++Celeb+DFDC; `--holdout <tag>` routes a dataset into `face_test.csv`
  for cross-dataset eval; MTCNN pinned to CPU on Apple Silicon (MPS returns
  None), uses CUDA on Kaggle.
- `scripts/eval_face_crops.py` — crop-level cross-dataset eval, full metrics +
  val-tuned threshold for a target FPR.
- `src/datasets/general_image_dataset.py` — `GeneralImageDataset` + the
  `ForensicDegrade` aug class (JPEG recompress q40-90, resolution 0.4-0.9,
  mild blur) enabled via `forensic_aug=True`.
- `src/training/video_trainer.py` — AMP (CUDA-only), grad-accum, staged
  unfreeze, early stop on val_auc.
- `src/inference/predictor.py` (`VideoDetector`), `multi_target_predictor.py`
  (`MultiTargetVideoDetector`) — what the dashboard loads.

### Gotchas learned (don't re-hit these)
- **MPS:** `torch.autocast(device_type='mps')` raises; use
  `contextlib.nullcontext()` when not CUDA. MTCNN returns None on MPS → pin CPU.
  Full-backbone video training thrashes MPS memory (~6× slower).
- **Imports decoupled:** `src/training/__init__.py` & `src/preprocessing/__init__.py`
  guard mlflow / facenet_pytorch with try/except so the general track imports
  without them.
- **`.pt` files are git-ignored** (`*.pt`, `checkpoints/*`, `!.gitkeep`).
  Checkpoints live on Kaggle/local, never in git.
- **Kaggle interactive sessions WIPE `/kaggle/working` on idle-timeout / the
  "still using this session?" prompt.** Lost a 3h extraction twice this way.
  **FIX: run long jobs as a committed notebook ("Save & Run All / Commit") —
  headless, survives disconnects, saves output as a downloadable version.**
- macOS auto-unzips downloads → `.pt`/`.zip` arrive as FOLDERS. Just move the
  folder into place; no unzip needed.
- timm backbone names that work: `legacy_xception` (20.8M),
  `convnext_tiny.fb_in22k_ft_in1k` (27.8M), `tf_efficientnet_b0.ns_jft_in1k`.

---

## 7. Dashboard (`Samsung_PRISM_Video_Dashboard/`)

- FastAPI backend + React (Vite) frontend, Docker-ready
  (`docker compose up --build` → http://localhost:8000).
- **No code changes needed to run** — it auto-loads checkpoints from the
  sibling module. `config/config.yaml` points at:
  - face checkpoint: `../Samsung_PRISM_Video_Detection/checkpoints/best.pt`
  - general checkpoint: `.../checkpoints/general.pt`
  - `use_multi_target: true` → uses `MultiTargetVideoDetector` when both exist.
- If a checkpoint is missing → falls back to a **clearly-marked MOCK model**
  (`"mock": true`), never fakes real output. Status panel shows connected/mock.
- **To make it run for real:** drop checkpoint files into
  `Samsung_PRISM_Video_Detection/checkpoints/` named `best.pt` (face) and
  `general.pt` (general). `model_service.py` is the ONE integration seam.

---

## 8. Where the trained weights / data are

- Not in git. From the Kaggle EXP-005 committed run, outputs were:
  `checkpoints/face_xception.pt`, `checkpoints/face_convnext.pt`,
  `reports/xception_dfdc.json`, `reports/convnext_dfdc.json`,
  and `face_crops.zip` (the extracted MTCNN crops + manifests — saved so the
  3h extraction need not repeat).
- Recommended local home for the crops (git-ignored):
  `Samsung_PRISM_Video_Detection/data/raw/face_crops_exp005/`.

---

## 9. Delivery status to Samsung

- **PR #20 MERGED** into SRIB-PRISM:main. Contains both folders under
  `Video_Dashboard/`, all docs/experiments, dashboard. `*.pt` excluded.
- Passed Samsung's **AVAS** dependency scan after pinning upper bounds in
  `requirements.txt`: `torch>=2.2,<2.6`, `torchvision>=0.17,<0.21`,
  `transformers>=4.40,<5`, `mlflow>=2.13,<3` (excludes flagged CVE versions).
- Fork-and-PR workflow: fork on `github.ecodesamsung.com/rudra-khale2023/...`,
  branch `video-module-rudra`. Auth = Samsung username + a PAT generated ON the
  enterprise instance (SSO-authorized for SRIB-PRISM org). github.com account
  is separate and does NOT exist on the enterprise server.

---

## 10. PENDING / NEXT WORK (in priority order)

1. **Phase-5 Celeb-DF run (apples-to-apples vs EXP-001's 0.71).** A committed
   Kaggle cell is ready: train FF+++DFDC (forensic aug), **hold out Celeb-DF**,
   both backbones, eval on held-out Celeb-DF. If AUC clears 0.71 with FPR ≪
   0.72 → positive Phase-5 result → write up as EXP-006. If not → honest
   negative, still valuable. (Consolidated cell was provided in chat; re-derive
   from `prepare_face_dataset_kaggle.py --holdout celeb`.)
2. **Synthetic-HUMAN branch (Phase 6)** — NOT started. Person detector
   (YOLO/RTMDet) → human-region crop → AIGC classifier. New files planned:
   `src/branches/human_branch.py`, `scripts/train_human_aigc.py`.
3. Phase 7 object/region forensics; Phase 8 temporal forensics;
   Phase 9 learned fusion; Phase 10 rich explainability JSON.

---

## 11. Conventions

- Develop on branch `claude/samsung-prism-video-detection-kctonh`; commit +
  push there. Do NOT open PRs on the personal repo unless asked. If that
  branch's PR was already merged, restart it from latest default branch.
- Never put model identifiers in commits/PRs/code — chat only.
- Every result gets an EXP-NNN doc + a metrics JSON; negative results kept.
