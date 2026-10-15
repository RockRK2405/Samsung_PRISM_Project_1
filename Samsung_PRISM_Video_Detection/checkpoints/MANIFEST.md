# Checkpoint manifest

`*.pt` is git-ignored by design, so **weights never live in this repo**. This
file is the git-tracked record of which checkpoint produced which result, so a
number in `reports/` can always be traced to the model behind it.

**Add a row whenever you train something you report.** A checkpoint with no row
here is an orphan: in a month nobody will know what produced it.

| File | EXP | Trained on | Protocol | Headline | Location |
|---|---|---|---|---|---|
| `face_celebdf_identity.pt` | **EXP-006** | Celeb-DF v2 (700+700 videos, 6 frames) | identity-disjoint train/val/test | **ROC-AUC 0.9644** (±0.05) on 56 held-out identity keys (14 celebrities + 42 solo clips) | Google Drive `MyDrive/prism_video/checkpoints/` — **111.4 MB**, sha256 `0d59bbdf…b99d` |
| `best.pt` | M4 / EXP-001 | FF++ c23 | FF++ → Celeb-DF v2 cross-dataset | val F1 0.9875; FF++ AUC 0.9946; Celeb-DF AUC 0.7076 | **not located** — local/Kaggle only |
| `best_dual.pt` | EXP-002 | FF++ c23, RGB+FFT dual-stream | FF++ → Celeb-DF v2 | val F1 0.9887; Celeb-DF AUC 0.7124 (**negative**) | **not located** |
| `best_model.pt` | **EXP-004** ⭐ | GenVidBench Pair1 (pika/vc2/ms + vript) | cross-generator, unseen t2vz | val AUC 0.9632; **test AUC 0.9426** | Kaggle committed-run output |
| `face_xception.pt` | EXP-005 | FF++ + Celeb-DF v2, forensic aug | → held-out DFDC | DFDC AUC 0.6407 (**negative**) | Kaggle committed-run output |
| `face_convnext.pt` | EXP-005 | FF++ + Celeb-DF v2, forensic aug | → held-out DFDC | DFDC AUC 0.6564 (**negative**) | Kaggle committed-run output |
| `face_dfdc_convnext.pt` | — | DFDC, **5 fake videos** | — | AUC 0.4173 — **BROKEN, do not cite** | Drive; delete it |

## EXP-006 — full provenance

- **Checkpoint:** `face_celebdf_identity.pt`
- **Architecture:** `BaselineDetector` + `convnext_tiny.fb_in22k_ft_in1k` (27.82M)
- **Training:** 8 epochs, batch 64, AdamW lr 3e-4, 224×224, `--forensic-aug`, CUDA T4, seed 42
- **Data:** Celeb-DF v2 (`kaggle:reubensuju/celeb-df-v2`, 6529 mp4 verified), MTCNN crops, 6 frames/video, cap 700/class
- **Split:** identity-disjoint (`--split-key identity`), 185/42/56 identity keys, 325 spanning videos dropped
- **Result:** ROC-AUC 0.9644, recall 0.9103, FPR 0.0727 @ 0.5 — on **26 fake / 156 real** test videos
- **Reports:** `EXP-006-celebdf-identity.json` (curated), `EXP-006-eval-raw.json` (raw tool output), `EXP-006-splits-summary.json`, `EXP-006-test-manifest.csv`
- **Doc:** `docs/experiments/EXP-006-celebdf-identity-disjoint.md`
- **File identity:** 111,363,931 bytes (111.4 MB)
  `sha256: 0d59bbdf73114965941a3a80f9ca299cfa6f65d0bfbcd865cd4c60b9e10eb99d`
  Verify before trusting any copy: `sha256sum face_celebdf_identity.pt`
- **Backup status:** ⚠️ **single copy, on Google Drive only.** Not in git
  (`*.pt` is ignored by design) and not mirrored anywhere. If that Drive folder
  is cleared the result is unreproducible without re-running prep + training
  (~90 min on a T4; the split is seeded at 42, so it would reproduce).

> **Caution:** Celeb-DF v2 was used as the TRAINING pool for this checkpoint,
> overriding the spreadsheet's "External Test" designation. Do **not** evaluate
> `face_celebdf_identity.pt` on Celeb-DF v2 and present it as unseen.

## To add a row

Record filename, EXP id, training data, protocol, headline metric, and where
the file actually is. **Always include size + sha256** (`sha256sum <file>.pt`):
names collide across runs, and the hash is the only way to tell two
checkpoints of the same name apart — or to confirm a copy is intact.

Checkpoints with no row here, or rows with no hash, are the reason EXP-001,
EXP-002 and EXP-004's weights are now marked *not located*.
