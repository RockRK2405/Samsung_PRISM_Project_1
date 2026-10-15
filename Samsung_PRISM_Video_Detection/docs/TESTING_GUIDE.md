# Testing Guide — how to check whether the model actually works

**Written:** 2026-10-14. Tooling in `scripts/` was written for this guide but
**has not been executed against a real model** — this container has no torch,
no opencv and no ffmpeg. The metric functions are unit-verified against
hand-computed references; the model-loading paths are not. Expect to fix a
small thing or two on first run.

---

## The one rule

**Run `selftest_detector.py` before you look at any accuracy number.**

As of the last pushed commit there is no checkpoint in `checkpoints/`, so the
dashboard serves a mock whose score is `sha256(file path)`. It is flagged
`"mock": true` in the JSON, but on the UI it looks exactly like a real verdict.
If you test accuracy against mock output you will produce a number that is
pure noise and looks plausible — roughly 50% accuracy, which reads as "the model
is weak" rather than "the model never ran."

```bash
cd Samsung_PRISM_Video_Detection
python scripts/selftest_detector.py --video /any/video.mp4
```

The decisive check inside it is **rename invariance**: it copies the same bytes
to three different filenames and scores each. A real model gives the same score
three times; a path-hash mock gives three unrelated scores. Exit code `0` = real
model, `1` = mock, `2` = inconclusive.

---

## The four tiers of test set

Work down the list. Each tier answers a different question, and skipping to
Tier 2 without Tier 0 passing wastes a GPU day.

| Tier | Question | Data | Labels? |
|---|---|---|---|
| **0** | Does the plumbing survive awkward input? | `make_smoke_clips.py` (generated) | **No** |
| **1** | Are scores sane on a handful of real clips? | ~20 clips you gather | Yes |
| **2** | **Does it reproduce EXP-004's 0.9426?** | GenVidBench t2vz + vript | Yes |
| **3** | How bad is the face branch cross-dataset? | Celeb-DF v2 / DFDC | Yes |
| **4** | Does it survive 2025-26 generators? | Sora/Veo/Kling/Runway | Yes |

---

## Tier 0 — Pipeline smoke clips (minutes, no download)

```bash
python scripts/make_smoke_clips.py --out data/smoke
```

Generates nine procedurally-drawn files, each targeting a failure mode that has
actually bitten this project: faceless clips, a 4-frame clip, a 1-frame clip,
64×64, 9:16 portrait, an all-identical-frames clip, and a still PNG.

> ⚠️ **These have NO ground truth.** They are drawn gradients and noise — not
> AI-generated, not real. A score on them means nothing. **Never put them in a
> manifest or a report.** You are watching for *crashes*, not scores.

Two failures are **expected** today and are known defects, not surprises:

- `faceless_*.mp4` through the face-only path → `RuntimeError: No face detected`
  (work log §7.6 — it should degrade to "unassessable", not raise).
- `still_image.png` through the dashboard → rejected at upload (§7.8).

Anything *else* that crashes is a new plumbing bug worth fixing.

## Tier 1 — Quick sanity set (~1 hour, ~20 clips)

Gather by hand, no big download:

- **10 real:** your own phone footage, or any Pexels/Pixabay stock clip.
- **10 synthetic:** anything you can generate or grab from a public showcase —
  Sora/Veo/Kling/Runway demo reels, Pika's gallery, HuggingFace Spaces output.

```bash
python scripts/build_manifest.py --out data/quick.csv \
    --real ~/testdata/real:phone_and_stock \
    --fake ~/testdata/ai:modern_t2v

python scripts/score_manifest.py --manifest data/quick.csv \
    --out reports/quick_test.json
```

**What this can and cannot tell you.** With n=20 the confidence interval on AUC
is roughly ±0.18 — so 0.6 and 0.9 are not distinguishable. Use it to catch
gross breakage (all scores identical, every clip "fake", scores inverted), not
to measure quality. Do not quote this number to anyone.

## Tier 2 — The canonical test: reproduce EXP-004 ⭐

**This is the test that matters.** It is the exact held-out split behind the
project's headline result, so it is the only test that tells you whether the
model you have is the model the docs describe.

**Protocol:** train on pika + videocrafter2 + modelscope; test on **t2vz
(completely unseen generator)** + disjoint vript reals.

You only need **two** of Pair1's five archives — `t2vz` (fakes) and `vript`
(reals). Per-archive sizes are not individually verified; Pair1 totals ~74 GB
across five, so budget roughly 15-25 GB each.

```bash
export HF_HUB_DISABLE_XET=1   # only if Xet transfer hangs

python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "jian-0/GenVidBench", repo_type="dataset",
    allow_patterns=[
        "GenVidBench/Pair1/t2vz.rar",
        "GenVidBench/Pair1/vript.rar",
        "GenVidBench/Pair1_labels.txt",
    ],
    local_dir="data/raw/genvidbench",
)
PY

cd data/raw/genvidbench/GenVidBench/Pair1
unar -q t2vz.rar && unar -q vript.rar     # brew install unar  /  apt install unar
ls t2vz | head -3                          # EXPECT: t2vz/t2vz-*.mp4
```

> **Verify the layout before extracting more.** The official labels use paths
> like `Pair1/t2vz/t2vz-xxx.mp4`, so extraction must yield a folder named
> `t2vz` containing mp4s. If `unar` produced a differently-named folder, fix the
> paths before building a manifest — a past FF++ path mismatch cost real time.

Then either use the project's own prep script (preferred — it reads the
**official** label file and guarantees the disjoint real split):

```bash
cd ../../../..   # back to Samsung_PRISM_Video_Detection
python scripts/prepare_genvidbench.py --config configs/video_training.yaml \
    --data-root data/raw/genvidbench/GenVidBench \
    --labels data/raw/genvidbench/GenVidBench/Pair1_labels.txt

python scripts/evaluate_video.py --checkpoint checkpoints/general.pt \
    --split test --out reports/tier2_genvidbench.json
```

…or go through the generic scorer, which gives the per-group breakdown:

```bash
python scripts/build_manifest.py --out data/tier2.csv \
    --real data/raw/genvidbench/GenVidBench/Pair1/vript:vript \
    --fake data/raw/genvidbench/GenVidBench/Pair1/t2vz:text2video-zero \
    --per-group 1000

python scripts/score_manifest.py --manifest data/tier2.csv \
    --out reports/tier2_genvidbench.json
```

### Pass/fail — compare against EXP-004

| Metric | EXP-004 (n=2000) | Read your result as |
|---|:-:|---|
| **ROC-AUC** | **0.9426** | 0.90-0.97 → reproduced ✅ · 0.75-0.89 → something differs ⚠️ · <0.70 → wrong checkpoint or broken preprocessing ❌ |
| F1 | 0.8616 | |
| Recall (t2vz) | 0.8250 | |
| FPR (vript) | 0.0900 | |
| Accuracy | 0.8675 | |

A result **far above** 0.95 is as suspicious as one far below — check you did
not accidentally include a *training* generator (pika/vc2/ms) in the test set,
or reuse vript videos that were in training. The prep script's
`_partition_reals()` exists precisely to prevent the latter.

## Tier 3 — Face branch cross-dataset

Both reference numbers are **negative results**; reproducing them means your
setup is correct, not that the model is good.

```bash
# Celeb-DF v2 (EXP-001's held-out set)
python scripts/build_manifest.py --out data/celeb.csv \
    --real /data/celeb-df-v2/Celeb-real:celeb_real \
    --real /data/celeb-df-v2/YouTube-real:youtube_real \
    --fake /data/celeb-df-v2/Celeb-synthesis:celeb_synthesis \
    --per-group 300
python scripts/score_manifest.py --manifest data/celeb.csv --threshold 0.5872
```

| Reference | AUC | Recall | FPR |
|---|:-:|:-:|:-:|
| EXP-001 FF++→Celeb-DF | 0.7076 | 0.9471 | **0.7247** |
| EXP-005 →DFDC (ConvNeXt) | 0.6564 | 0.2960 | 0.1228 |

Note the threshold: `0.5872` is the FF++-val-tuned value, and it is the right
one for the **face** checkpoint only. Do **not** use it for the general model —
it was never calibrated on GenVidBench (work log §7.7).

Expect the EXP-001 FPR of 0.72 to reproduce: the model mis-flags ~70-76% of real
videos. That is the documented cross-dataset gap, not a bug in your run.

## Tier 4 — Modern generators (the real open question)

Every training generator is **2023-era**. Nothing in the model has seen Sora,
Veo, Kling or Runway Gen-3, and performance on them is **completely unmeasured**.
This is the most likely reason detection feels weak even once Tier 2 passes.

Build 100-200 clips of 2025-26 output plus matched modern real footage, **hold it
out entirely** (never train on it), and record provenance for every clip. Even
at n=150 (CI roughly ±0.07) the headline — does it hold up or collapse — will be
unambiguous. **A large drop is a publishable finding, not a failure.**

---

## Reading the output honestly

**Sample size sets your resolution.** Rough 95% CI on AUC:

| n (balanced) | ±AUC |
|---|---|
| 20 | ±0.18 |
| 100 | ±0.08 |
| 500 | ±0.03 |
| 2000 | ±0.02 |

(Hanley-McNeil, assuming a true AUC near 0.85; widen these if the real AUC is
closer to 0.5.)

So a 20-clip test cannot tell 0.65 from 0.90. Quote AUC with its n, always.

**Quote AUC, not F1.** EXP-001 scored F1 **0.81** while mis-flagging 72% of real
videos — class imbalance plus a fake-happy bias inflates F1 badly. AUC is
threshold-free and the honest headline. `score_manifest.py` prints both and
marks the two to lead with.

**Check the per-group table, not just the overall row.** One pooled number hides
the thing you care about. EXP-004's 0.9426 decomposes into t2vz recall 0.825 and
vript specificity 0.910 — a model could post a decent overall AUC while failing
one generator completely.

**Use the threshold sweep.** The scorer sweeps 0.05-0.95. If accuracy peaks far
from 0.5, the model is mis-calibrated rather than weak — a calibration fix
(Phase 3), not a retraining job.

---

## Common traps

| Trap | Why it bites |
|---|---|
| Testing on mock output | ~50% accuracy that reads as "weak model" instead of "no model" |
| Testing on a training generator | Measures memorisation; inflates AUC toward 1.0 |
| Random **frame** split | Same video's frames in train and test → near-perfect, meaningless |
| Face threshold on the general model | 0.5872 was tuned on FF++ face crops; meaningless elsewhere |
| Smoke clips in a report | They have no ground truth |
| Quoting F1 from an imbalanced set | See EXP-001: F1 0.81 with FPR 0.72 |
| Faceless clips "proving" robustness | They crash the face path today (§7.6) |

---

## Quick reference

```bash
# 0. is the model even running?      -> exit 0 = real, 1 = mock
python scripts/selftest_detector.py --video any.mp4

# 1. plumbing edge cases (no labels, no report)
python scripts/make_smoke_clips.py --out data/smoke

# 2. build a labeled manifest
python scripts/build_manifest.py --out data/t.csv \
    --real DIR:group --fake DIR:group --per-group 500

# 3. full metric suite + per-group + threshold sweep
python scripts/score_manifest.py --manifest data/t.csv --out reports/t.json

# bypass the dashboard and hit the detector classes directly
python scripts/selftest_detector.py --video any.mp4 --mode direct
```

**If `selftest_detector.py` returns 1 (mock) and a checkpoint IS present**, you
are hitting the serving defects in `DETECTION_ENABLEMENT_PLAN.md` — most likely
§7.2 (the serving factory cannot build `VideoSpatialNet`, the EXP-004
architecture). That is Phase 0 work, roughly a day, no GPU required.
