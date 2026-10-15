# Pipeline file index — current set

**Generated 2026-10-14.** Supersedes every earlier copy handed over in chat.
If a file is not listed here, it is stale — discard it.

---

## The two workflows

These are independent. Pick by what you are doing.

| Goal | Use |
|---|---|
| **Train a new model** on the spreadsheet's datasets | `notebooks/Colab_Pipeline_v2.ipynb` |
| **Test a model that already exists** | `scripts/selftest_detector.py` → `scripts/score_manifest.py` |

---

## A · Train & test pipeline (Colab)

| File | Role |
|---|---|
| `notebooks/Colab_Pipeline_v2.ipynb` | The orchestrator. Upload to Colab, run top to bottom. |
| `configs/dataset_registry.yaml` | The spreadsheet (20 datasets) as machine-readable config: split role, priority, acquisition method, layout globs, verified flag. |
| `scripts/prepare_from_registry.py` | Registry + on-disk video → `train.csv` / `val.csv` / `test_<dataset>.csv`, video-disjoint, leakage-asserted. |

Uses the project's **existing** `scripts/train_general.py` and
`scripts/eval_face_crops.py` unchanged — those are the scripts that produced
EXP-005 on Kaggle CUDA, so they are proven.

**Protocol:** train DFDC (2 parts) → validate on a video-disjoint split →
test on fully held-out Celeb-DF v2.

## B · Testing an existing model

| File | Role |
|---|---|
| `scripts/selftest_detector.py` | **Run first.** Is the model real or is the dashboard serving mock output? Rename-invariance test. Exit 0 = real, 1 = mock. |
| `scripts/score_manifest.py` | Labeled manifest → full metric suite + per-group + threshold sweep. Pure-Python AUC/PR-AUC, no sklearn. Refuses to score mock output. |
| `scripts/build_manifest.py` | Labeled directories → `path,label,group` CSV, seeded. |
| `scripts/make_smoke_clips.py` | 9 pipeline edge-case clips. **No ground truth** — for crash-hunting only, never a report. |
| `docs/TESTING_GUIDE.md` | Four test tiers with acquisition commands, incl. GenVidBench for the general branch. |

---

## Verified Colab environment (2026-10-14)

| | |
|---|---|
| Python / torch / torchvision | 3.13 · 2.11.0+cu130 · 0.26.0+cu130 |
| numpy | **2.1.3** — must not be downgraded |
| timm / cv2 | 1.0.30 · 4.14.0 |
| GPU / disk | Tesla T4 · ~107 GB total |

**`facenet-pytorch` must install with `--no-deps`.** Its metadata pins
`numpy<2.0.0` and `torch<2.3.0`; honouring those makes pip compile numpy 1.26
from source on Python 3.13 (no cp313 wheel exists) and then downgrade torch.
Its source has **zero** numpy-2 incompatibilities, and its bundled MTCNN
weights contain only `OrderedDict` + `FloatStorage` + `_rebuild_tensor_v2`,
all on the `weights_only=True` allowlist — so the pins are stale, not real.

**Upgrade the kaggle CLI first.** 2.0.2 returns `404` on `DownloadDataset`;
2.2.4+ works.

---

## Dataset status

| Dataset | Role | Status |
|---|---|---|
| **Celeb-DF v2** | external_test | ✅ Kaggle `reubensuju/celeb-df-v2` — 9.29 GB, 6529 mp4, layout verified |
| **DFDC** | train_pool | ⚠️ Kaggle `pranay22077/dfdc-10` — **96.5 GB total**, download per part with `-f`; layout verified |
| **FF++** | train_pool | ❌ The `ahmedelbanby/...deepfakebench-structure` mirror ships PNG frames **and contains Celeb-DF**. Using it while testing on Celeb-DF v2 is silent leakage. Get FF++ via the official EULA instead. |
| the other 17 | mixed | `verified: false`, skipped by default. `--inspect <name>`, fix globs, set `verified: true`. |

### The leakage limit you must know
`prepare_from_registry.py` enforces leakage rules on registry **keys**. It
cannot see that one archive physically contains another dataset's videos.
**A passing leakage check does not mean the archive is clean.** Look inside any
new mirror and delete anything duplicating an `external_test` set.

---

## Reference numbers

| Experiment | Protocol | ROC-AUC | Recall | FPR |
|---|---|:-:|:-:|:-:|
| EXP-001 | FF++ → Celeb-DF v2 | 0.7076 | 0.9471 | 0.7247 |
| EXP-005 | FF+++Celeb-DF → DFDC | 0.6564 | 0.2960 | 0.1228 |
| **EXP-004** | GenVidBench, unseen generator | **0.9426** | 0.8250 | 0.0900 |

In-distribution validation runs ~0.99 AUC and says nothing about
generalization. Only held-out numbers count. Quote **ROC-AUC, not F1** — EXP-001
scored F1 0.81 while mis-flagging 72% of real videos.

---

## Scope

Every dataset in the spreadsheet is a **face-deepfake** set, so workflow A
trains the **face branch** — the weaker one (0.64-0.71 cross-dataset vs the
general branch's 0.9426 on an unseen generator). This is the right tool for
Phase 6, *fix or formally de-scope the face branch*.

It cannot improve general AI-video detection. No dataset here contains fully
AI-generated video; that needs GenVidBench or a modern Sora/Veo/Kling set.
See `docs/TESTING_GUIDE.md` Tier 2 and Tier 4.

---

## Known caveats

* Workflow B's scripts are compile-checked and their metrics are unit-tested
  against hand-computed references, but have **never run against a real
  model** — no torch in the authoring environment.
* `prepare_from_registry.py` has run its selection, glob, `metadata_json`,
  overlap-guard and inspect paths against fixture trees; its MTCNN extraction
  path has not.
* 17 of 20 registry layouts are unverified by design and marked as such.
