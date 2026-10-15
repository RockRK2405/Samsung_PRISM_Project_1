# Checkpoint manifest

`*.pt` is git-ignored by design, so **weights never live in this repo**. This
file is the git-tracked record of which checkpoint produced which result.

All rows below were verified 2026-10-15 by loading each file and reading its
embedded `config.model` — not inferred from filenames.

> **Add a row whenever you train something you report, with size AND sha256.**
> Names collide across runs; the hash is the only reliable identity. The
> `best.pt` incident below is what happens without this discipline.

---

## Verified checkpoints (local: `Samsung_PRISM_Video_Detection/checkpoints/`)

| File | Size | Architecture (from embedded config) | What it is | sha256 |
|---|:-:|---|---|---|
| **`best_model.pt`** ⭐ | 114.9 MB | `frame_baseline` · `convnext_tiny.fb_in22k` · **attention_pool** (512/2/8) | **EXP-004 — GenVidBench cross-generator, test AUC 0.9426** (val 0.9632). `VideoSpatialNet`. The project's best result. | `c20bbaed…dc15` |
| **`face_celebdf_identity.pt`** ⭐ | 111.4 MB | `convnext_tiny.fb_in22k_ft_in1k` · mean_pool | **EXP-006 — Celeb-DF identity-disjoint, test AUC 0.9644** (val 0.9829) | `0d59bbdf…b99d` |
| `last_model.pt` | 114.9 MB | identical config to `best_model.pt` | EXP-004 **last**-epoch state (aborted epoch 2). Different hash from `best_model.pt`, so **not** a duplicate. `best_model.pt` is the evaluated one. | `58c89c2d…134c` |
| `best_dual.pt` | 32.7 MB | `tf_efficientnet_b0.ns_jft_in1k` · mean_pool (2 backbones' worth of size) | **EXP-002 — dual-stream RGB+FFT**, Celeb-DF AUC 0.7124 (**negative result**) | `662f99d0…8f06` |
| `best_temporal.pt` | 24.0 MB | `tf_efficientnet_b0.ns_jft_in1k` · **transformer** head (256/2/4) | **Milestone 5A — temporal transformer head** (**negative result**) | `8f93ae35…a055` |
| `general.pt` | 15.1 MB | `tf_efficientnet_b0.ns_jft_in1k` · mean_pool | ADR-006 general path, GenImage-family **still images**. No published metrics. | `97ce303f…633e` |
| ⚠️ `best.pt` | 16.3 MB | `tf_efficientnet_b0.ns_jft_in1k` · mean_pool | **NOT the face baseline.** Written by `train_general.py` (carries its meta block) on 2026-09-06. A GenImage still-image model. See below. | `34d6f17f…4cfe` |

## ⚠️ The M4 face baseline appears to be lost

`best.pt` is the filename `train.py` (the **face** trainer) writes to, and it is
what `docs/experiments/EXP-001` and `EXP-002` evaluated: EfficientNet-B0 +
mean-pool, val F1 0.9875, FF++ test AUC 0.9946, Celeb-DF v2 AUC 0.7076.

The file on disk is **not** that model:

- it carries `meta: {task: general_ai_content_detection, dataset: GenImage-family,
  note: "Trained on still images, not video"}` — a block only
  `train_general.py` writes; `train.py` writes none;
- it is dated **2026-09-06**, while EXP-001/EXP-002 are dated **2026-07-07/08**.

So someone ran `train_general.py --out checkpoints/best.pt`, overriding that
script's `general.pt` default, and **overwrote the face baseline**.
`best.metrics.json` (same timestamp, 106 bytes) describes that general run, not
the face model.

**Consequence:** EXP-001's and EXP-002's checkpoint no longer exists. Those
results stand on their written records only and cannot be re-evaluated or
served. Both are reproducible in principle (FF++ c23 + `train.py`), but not
recoverable as files.

**Consequence for the dashboard:** `config/config.yaml` points
`checkpoint: checkpoints/best.pt`, labels it *"BaselineDetector (EfficientNet-B0
+ mean-pool)"* face model, and applies `decision_threshold: 0.5872` — an
FF++-val-tuned value. All three are now wrong for the file present. Running it
as-is loads a still-image GenImage model, feeds it MTCNN face crops, and judges
it at a threshold tuned for a different model on a different dataset. Point the
config at `face_celebdf_identity.pt` with `decision_threshold: 0.5` instead.

## ⚠️ `train_general.py` writes a hardcoded, often-false `meta`

`_save_checkpoint()` hardcodes `task: general_ai_content_detection`,
`dataset: GenImage-family`, `note: "Trained on still images, not video"`
regardless of what was actually trained. So `face_celebdf_identity.pt` — trained
on **Celeb-DF v2 face crops** — claims to be a GenImage still-image model. Do
not trust that block on any checkpoint from this script; trust this manifest.

---

## EXP-004 — `best_model.pt` (recovered 2026-10-15)

Previously recorded as *not located*. Found on the owner's MacBook and
confirmed by config: `family: frame_baseline`, `backbone: convnext_tiny.fb_in22k`,
`temporal: {type: attention_pool, hidden_dim: 512, num_layers: 2, num_heads: 8}`
— matching EXP-004's documented `VideoSpatialNet` and `configs/video_training.yaml`.

- **Result:** test ROC-AUC **0.9426** on the unseen generator `text2video-zero`
  (F1 0.8616, recall 0.8250, FPR 0.0900); val AUC 0.9632
- **Training:** 1 epoch, backbone frozen, batch 8, AdamW lr 1e-4, MPS — stopped
  when epoch 2 thrashed memory. **The number is a floor, not a converged result.**
- **Not yet servable:** `VideoSpatialNet` has no branch in
  `build_model_from_config`, so `VideoDetector` cannot load it
  (`DETECTION_ENABLEMENT_PLAN.md` §7.2 — Phase 0, ~1 day, no GPU).

## EXP-006 — `face_celebdf_identity.pt`

- **Result:** test ROC-AUC **0.9644** ±0.05 on 56 held-out identity keys
  (14 celebrities + 42 solo clips); val AUC 0.9829
- **Training:** 8 epochs, batch 64, AdamW lr 3e-4, 224×224, `--forensic-aug`,
  CUDA T4, seed 42; Celeb-DF v2, MTCNN crops, 6 frames/video, cap 700/class
- **Reports:** `EXP-006-celebdf-identity.json`, `EXP-006-eval-raw.json`,
  `EXP-006-splits-summary.json`, `EXP-006-test-manifest.csv`,
  `EXP-006-train-val-metrics.json`

> **Caution:** Celeb-DF v2 was the *training* pool for this checkpoint,
> overriding the spreadsheet's "External Test" role. Do not evaluate it on
> Celeb-DF v2 and present that as unseen.

---

## Backup status

As of 2026-10-15 these seven files (~430 MB) exist in **one place**: the owner's
MacBook. They are the only surviving copies of EXP-002, EXP-004 and
Milestone 5A. `*.pt` is git-ignored, so git will never hold them.

**Mirror this folder** — Drive, external disk, anywhere. One disk failure erases
three published results permanently, and `best.pt` already demonstrates how
cheaply a checkpoint is lost.

## To add a row

Load the file, read `config.model`, and record: filename, size, architecture,
EXP id, headline metric, and **sha256**. Never identify a checkpoint by
filename alone.
