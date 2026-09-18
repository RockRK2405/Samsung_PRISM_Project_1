# Video Module — Master Plan & Runbook (26TS08)

**Purpose:** single source of truth for the Video Detection module's full
build-out. Written so a fresh session can read this back and resume without
re-deriving context. Priority = **detection quality + generalization**;
cost-awareness is NOT a constraint for the video model.

**Core framing:** GENERAL AI-VIDEO detection is the PRIMARY problem. Face
deepfake is ONE branch. AI-human, object/region, temporal, and fusion are
additional required capabilities. Do not build "just another face detector".

---

## 0. Status legend
- ✅ built + verified   🟢 built (code, untrained)   ⏳ planned   🟠 partial

## 1. Where we are (2026-09)

| Component | Status | Notes |
|-----------|--------|-------|
| Repo audit | ✅ | done |
| `configs/video_training.yaml` | ✅ | centralized config |
| GenVidBench manifest pipeline | ✅ | `scripts/prepare_genvidbench.py`, verified: zero leakage, cross-generator |
| Video dataset | 🟢 | `src/datasets/video_dataset.py` |
| Spatial baseline (ConvNeXt+pool) | 🟢 | `src/models/video_spatial.py` |
| VideoMAE / TimeSformer | 🟢 | `src/models/video_transformer.py` |
| Trainer + eval | 🟢 | `src/training/video_trainer.py`, `scripts/{train,evaluate}_video.py` |
| Face track (FF++) | ✅ | pre-existing; Celeb-DF AUC 0.71 (weak cross-dataset) |
| Human branch | ⏳ | not built |
| Object/region branch | ⏳ | not built |
| Learned fusion | ⏳ | not built |
| Rich explainability | 🟠 | face GradCAM only |
| Robustness eval | ⏳ | not built |

**Verified GenVidBench facts** (from official label files, not guessed):
- Real: Vript (20,131), HD-VG-130M (13,416)
- Fake Pair1: pika, vc2(videocrafter2), ms(modelscope), t2vz(text2video-zero) ~13.5k each
- Fake Pair2: cogvideo, mora, musev, svd ~13.5k each
- Archives: `Pair1/*.rar` (~74GB), `Pair2/*.{rar,7z}` (~120GB); labels `<rel_path> <label>` (1=fake,0=real), generator = 2nd path component.

**Active data plan:** Option B (Pair1 only, ~74GB) — cross-generator WITHIN
Pair1: train pika+vc2+ms, test held-out t2vz; real=vript split disjointly.
Option A (full 194GB, cross-source) is a later scale-up (config has both).

---

## 2. Hardware reality
- Mac M5 Pro (MPS): trains spatial baseline fine; VideoMAE/TimeSformer may
  hit MPS op gaps → run those on CUDA, or CPU smoke test. Documented, not
  silently degraded.
- Seed-lab box: 24-core CPU, NO GPU (Intel UHD only) → good for data
  prep/extraction, NOT heavy training.
- Kaggle T4 / cloud GPU: fallback for heavy transformer training.

---

## 3. RESUME RUNBOOK — do these in order when you come back

### Step 1 — finish Pair1 download (~74GB)
```bash
cd <repo>/Samsung_PRISM_Video_Detection
export HF_HUB_DISABLE_XET=1   # if Xet hangs; else omit
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('jian-0/GenVidBench', repo_type='dataset',
  allow_patterns=['GenVidBench/Pair1/*','GenVidBench/Pair1_labels.txt'],
  local_dir='data/raw/genvidbench')
print('done')"
```
Verify: `ls -lh data/raw/genvidbench/GenVidBench/Pair1/` → 5 archives + label.

### Step 2 — extract the SMALLEST first, verify layout
```bash
cd data/raw/genvidbench/GenVidBench/Pair1
unar -q ms.rar        # brew install unar p7zip  (if missing)
ls ms | head           # EXPECT: ms/ms-*.mp4  (folder named 'ms' with mp4s)
```
The label paths are `Pair1/ms/ms-xxx.mp4`, so extraction must yield
`Pair1/ms/*.mp4`. If `unar` produced a differently-named folder, tell me the
actual layout BEFORE extracting the rest (prevents the FF++ path mess).

### Step 3 — extract the rest
```bash
for f in pika t2vz vc2 vript; do unar -q "$f.rar"; done
# sanity: every generator folder should exist under Pair1/
ls -d Pair1/*/ 2>/dev/null || ls -d */
```

### Step 4 — build manifests (Option B active in config)
```bash
cd <repo>/Samsung_PRISM_Video_Detection
python3 scripts/prepare_genvidbench.py --config configs/video_training.yaml \
  --data-root data/raw/genvidbench/GenVidBench \
  --labels data/raw/genvidbench/GenVidBench/Pair1_labels.txt
```
Confirms on-disk counts, writes `data/manifests/{train,val,test}.csv`.
Expect: train ~pika+vc2+ms fakes + vript; test = t2vz + disjoint vript.

### Step 5 — install deps (once)
```bash
pip install -r requirements.txt   # includes timm, transformers>=4.40, sklearn
```

### Step 6 — PHASE 1 baseline number (record it — Part 35 anchor)
Train the spatial baseline (frame_baseline = ConvNeXt + attention pool):
```bash
python3 scripts/train_video.py --config configs/video_training.yaml \
  --name exp01_convnext_attn --epochs 15
python3 scripts/evaluate_video.py --checkpoint checkpoints/best_model.pt \
  --out reports/exp01_convnext.json
```
This is the BASELINE all later models are compared against.
On MPS start with batch_size 8, num_frames 16; if OOM drop to 8 frames / bs 4.

### Step 7 — PHASE 3 head-to-head: VideoMAE, then TimeSformer
```bash
# VideoMAE (16 frames, matches num_frames=16)
python3 scripts/train_video.py --config configs/video_training.yaml \
  --backbone none --temporal none --name exp02_videomae \
  --num-frames 16   # AND set model.family: videomae in the config
python3 scripts/evaluate_video.py --checkpoint checkpoints/best_model.pt \
  --out reports/exp02_videomae.json

# TimeSformer expects 8 frames — set model.family: timesformer, num_frames: 8
```
(Set `model.family` in the config for these — the CLI backbone/temporal
flags apply to the spatial family only.) If MPS errors on the transformer,
run on a CUDA box / Kaggle. Compare exp01 vs exp02 vs exp03 on val + test
AUC — pick the winner empirically (no assumed best).

### Step 8 — record the comparison table
Fill `reports/baseline_results.json` with baseline vs each model:
Accuracy, F1, ROC-AUC, PR-AUC, FPR + per-generator (seen vs unseen t2vz).

---

## 4. FULL PHASE PLAN (spec §13 ↔ ADR-007)

### Phase 1 — Baseline metrics ✅pipeline / ⏳run
Run the spatial baseline on GenVidBench Option B; record the metric suite.
Files: (built) train/evaluate_video. Output: `reports/exp01_*.json`.

### Phase 2 — GenVidBench general video 🟢
Spatial backbone + temporal aggregation. Ablate temporal head
(mean_pool vs attention_pool vs transformer) via `model.temporal.type`.
Models to test: ConvNeXt-Tiny, EfficientNet-B3, Xception, Swin-Tiny.

### Phase 3 — Strong video transformer 🟢code
VideoMAE-base (16f) and TimeSformer (8f) via `model.family`. Later, if
feasible on GPU: ViViT, Video-Swin, X-CLIP. Select by validation, not size.

### Phase 4 — Face detection pipeline ✅(exists)
MTCNN present. Optionally add RetinaFace for recall on profile/occluded
faces. Face detector LOCATES faces only — never classifies real/fake.
Planned file: `src/preprocessing/retinaface_detector.py` (optional).

### Phase 5 — Face forensics + cross-dataset ⏳
Upgrade forensic backbone (Xception/ConvNeXt/Swin) on FF++ face crops;
temporal face aggregation; evaluate FF++→Celeb-DF→DFDC cross-dataset.
Reuse existing `train.py`/`evaluate.py`. New: `scripts/eval_cross_dataset.py`.
Goal: beat the current Celeb-DF AUC 0.71 / FPR 0.72.

### Phase 6 — Synthetic-HUMAN branch ⏳ (spec §8, Category B)
Person detection (YOLO/RTMDet) → human-region crop → AIGC classifier
(ViT/ConvNeXt). Must work when the FACE branch can't (full body, back,
distant, occluded). Complements — does NOT replace — the general-video model.
New files: `src/branches/human_branch.py`, `scripts/train_human_aigc.py`.
Pretrain region classifier on GenImage/COCO-derived synthetic-human crops.

### Phase 7 — Object/region forensics ⏳ (spec §9, Category E)
Object detector (YOLO / Grounding-DINO) → object crop → AIGC forensic
classifier. Detects AI object inside a REAL video. Detector says "what
object"; forensic model says "is it synthetic". Keep separate.
New: `src/branches/object_branch.py`, `scripts/train_object_aigc.py`.

### Phase 8 — Temporal forensics ⏳ (spec §10/§F, Category F)
Explicit frame-to-frame anomaly signals (motion, identity, geometry,
texture, lighting, landmark stability) on top of the temporal transformer.
New: `src/branches/temporal_forensics.py`.

### Phase 9 — Learned multi-branch fusion ⏳ (spec §9)
Concatenate {general_video, temporal, face, human, object} embeddings+scores
→ MLP/transformer fusion → final score. No fixed hand-picked weights.
New: `src/models/fusion.py`, `scripts/train_fusion.py`.

### Phase 10 — Rich explainability ⏳ (spec §14)
Aggregate per-branch evidence into the JSON schema: classification,
synthetic_probability, per-branch scores, suspicious_frames, faces[],
humans[], objects[] with bboxes + probabilities. Extend existing GradCAM.
New: `src/inference/rich_explainer.py`, `schemas/rich_result.schema.json`.

---

## 5. Evaluation strategy (spec §15-16) — per task, never one number
For EACH of general-video / face / object / human / temporal / fusion:
Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR, confusion.
Plus:
- Seen vs UNSEEN generator (GenVidBench held-out) — pipeline already emits per-generator recall.
- Cross-dataset: FF++→Celeb-DF→DFDC (face); Pair1→Pair2 (video, after Option A).
- Robustness (spec §18): JPEG compression, resize, blur, noise,
  brightness/contrast, resolution degradation, frame dropping, FPS change.
  New: `src/evaluation/robustness.py`, `scripts/eval_robustness.py`.
Reporting plots: confusion, ROC, PR, training curve, score distribution.
New: `scripts/plot_results.py`.

## 6. Data-leakage rules (spec §17) — ALREADY enforced in Phase 1
Video-level splits only; deterministic seeded manifests; generator metadata
preserved; train/val/test video-disjoint (verified). Never random-frame split.

## 7. Files to create (by phase) — nothing created until its phase runs
- P5: `scripts/eval_cross_dataset.py`
- P6: `src/branches/human_branch.py`, `scripts/train_human_aigc.py`
- P7: `src/branches/object_branch.py`, `scripts/train_object_aigc.py`
- P8: `src/branches/temporal_forensics.py`
- P9: `src/models/fusion.py`, `scripts/train_fusion.py`
- P10: `src/inference/rich_explainer.py`, `schemas/rich_result.schema.json`
- Eval: `src/evaluation/robustness.py`, `scripts/eval_robustness.py`, `scripts/plot_results.py`

## 8. Honest-scope guardrails (spec §21-22)
- No metric claimed until it is actually measured.
- Every phase reports baseline-vs-new; regressions diagnosed, not hidden.
- No fabricated dataset paths/labels/checkpoints. GenVidBench facts above
  are from the real label files; anything unverified is marked "not verified".

---

## 9. When you resume — tell me
1. Download state (`ls -lh .../Pair1/`).
2. Result of extracting `ms.rar` (`ls ms | head`) so I confirm layout.
Then I deliver the exact next code/commands for the phase we're on, one at a
time. Current next action: **finish download → extract ms.rar → verify → build manifests → train Phase-1 baseline.**
