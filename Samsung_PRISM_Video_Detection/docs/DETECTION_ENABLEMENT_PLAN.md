# Detection Enablement Plan — why nothing is detected, and the phased fix

**Written:** 2026-09-26
**Scope:** Video Detection module + Dashboard (26TS08 video branch)
**Status of this doc:** diagnosis is verified against the code in this repo
(file:line cited). The plan is proposed, not yet executed.

> **Read this before `VIDEO_MODULE_MASTERPLAN.md`.** The master plan describes
> *research* phases (6-10) and is correct. This doc covers a different problem
> the master plan does not: the trained research models **cannot be served at
> all**, so the system detects nothing regardless of how good the models get.
> Phase 0 here blocks everything in the master plan.

---

## Part 1 — Diagnosis: why the project detects no deepfakes today

The cause is **not** model quality. It is that **no model is ever executed.**
There are seven distinct defects, in two classes.

### 1.0 The headline: every "detection" you have seen is a hash of the filename

`checkpoints/` contains only `.gitkeep` — verified: no `.pt` files exist
anywhere in the repo (and `*.pt` is git-ignored by design, so they never
arrive with a clone).

Consequence chain:

1. `model_service.py:70-71` — `_RealModel.__init__` raises `FileNotFoundError`
   when `checkpoints/best.pt` is missing.
2. `model_service.py:265` — `init_model()` catches it and sets
   `_mock_active = True`.
3. `_MockModel.predict()` returns
   `prob = (sha256(filename) % 1000) / 1000`.

**So the dashboard's output is a deterministic hash of the upload's filename.**
It is correctly flagged `"mock": true` (the code is honest — it never dresses
mock output as real), but functionally the system has never classified a video.
Rename the same file and the verdict changes; that is the signature to look for.

This alone explains "not detecting deepfakes, faceswaps or AI-generated
videos/images". The remaining six defects mean that **simply dropping
checkpoints in would still not work.**

### 1.1 The 0.94-AUC model is structurally unservable — two model factories

The repo has **two parallel, non-communicating model factories**:

| Path | Factory | Can build |
|---|---|---|
| **Research** (`scripts/train_video.py`, `evaluate_video.py:32,66`) | `build_video_model_from_config` (`video_spatial.py:134`) | `VideoSpatialNet`, `VideoTransformerNet` |
| **Serving** (`src/inference/predictor.py:27,144`) | `build_model_from_config` (`models/__init__.py:35`) | `BaselineDetector`, `TemporalDetector`, `DualStreamDetector` |

`build_model_from_config` dispatches on `frequency_stream.enabled` and
`temporal_head.type` only — **it has no branch that returns `VideoSpatialNet`.**
`VideoSpatialNet` is imported and re-exported in `models/__init__.py:15,24` but
never referenced in the dispatch body.

**EXP-004's winner — `VideoSpatialNet` (ConvNeXt-Tiny + attention-pool), ROC-AUC
0.9426 on the unseen `t2vz` generator — is the single best result in this
project and there is no code path that can load it for inference.** Handing
`predictor.py` that checkpoint raises on `build_model_from_config`, or (worse)
silently builds a `BaselineDetector` and fails `load_state_dict`.

This is the most important finding in this document.

### 1.2 The general path is hardcoded to the wrong architecture

`multi_target_predictor.py:73-79` constructs the general model as
`BaselineDetector(...)` unconditionally, ignoring `model.family` in the
checkpoint's embedded config. A `VideoSpatialNet` `general.pt` cannot load here
either — the state-dict keys (`pool.score.*`, `classifier.0.*`) do not exist on
`BaselineDetector`.

### 1.3 The strongest branch is gated behind the weakest one

`multi_target_predictor.py:61` builds the **face** `VideoDetector` *first*, in
`__init__`, before the general model. Combined with `model_service.py:70-71`
raising on a missing `best.pt`:

**You cannot run general-video detection without a face checkpoint.** The
branch that generalizes (0.94 AUC) is unreachable unless the branch that does
not (0.65 AUC) is present. That is backwards, and it is why a general-only
deployment is impossible today.

### 1.4 Temporal modelling is discarded at inference

`multi_target_predictor.py:135-142` — `_score_general` scores frames
independently and returns `np.mean(per_frame)`.

The whole point of `VideoSpatialNet` is the **learned attention-pool** over
frames. Mean-pooling at inference throws that away: even with 1.1-1.3 fixed,
you would serve a model under an aggregation it was never trained with.
Train/inference skew, straight into the score.

### 1.5 The face path crashes on faceless AI video

`predictor.py:218-219` raises `RuntimeError("No face detected in any of the N
sampled frames")`.

A fully AI-generated clip (Sora/Veo/Kling landscape, object, animation) has no
face by construction. The face path therefore **throws rather than returning
"cannot assess"**. `MultiTargetVideoDetector` guards this with
`_FACE_COVERAGE_THRESHOLD`, but bare `VideoDetector` — which is what
`model_service` falls back to whenever `general.pt` is absent — does not.

### 1.6 Thresholds are uncalibrated and inconsistent

- `config.yaml` sets `decision_threshold: 0.5872` — tuned on **FF++ face-crop
  validation** for FPR ≤ 5%.
- `multi_target_predictor.py:115` **ignores it** and hardcodes `0.5`.
- That 0.5872 is applied to the general model too, where it has no meaning —
  it was never calibrated on GenVidBench.

Additionally, `multi_target_predictor.py:108` fuses by `max(face, general)`.
With two independently-calibrated detectors at FPR ≈ 0.09 and ≈ 0.12,
`max()` **compounds false positives** (up to ~0.20 if errors are independent).
ADR-006 chose `max()` deliberately for recall, but it was never re-calibrated
after fusion — so the deployed FPR is unknown and likely ~2× the per-branch
number.

### 1.7 Images are not supported at all

You asked about **images** as well as video. `config.yaml` sets
`allowed_extensions: [".mp4",".mov",".avi",".mkv",".webm"]` and
`video_service.py:29-31` rejects anything else. There is no image entry point
in `detection.py`, and `_run_one` calls `video_service.probe_metadata()` which
assumes a decodable video stream.

Note the asymmetry worth exploiting: `BaselineDetector` and `GeneralImageDataset`
are already **per-image** classifiers. A single image is just `T=1`. The model
side is ready; only the API/validation layer blocks it.

### 1.8 Secondary / latent issues

- **Resize skew:** training uses `transforms.Resize` (bilinear,
  `general_image_dataset.py:118`); `multi_target_predictor.py:137` uses PIL
  `Image.resize()` (bicubic default). Small but free to fix.
- **`torch.load` without `weights_only`:** `predictor.py:136` omits it while
  `multi_target_predictor.py:70` passes `weights_only=False`. Harmless under
  the pinned `torch<2.6`, but `predictor.py` breaks the moment that pin lifts.
- **`PROJECT_STATUS.md` is stale** — dated 2026-07-23 and still describes the
  4-modality fusion era (audio/text/image owners, `Samsung_PRISM_Fusion_Engine/`
  which is not in this repo). It contradicts the current video-only scope and
  will mislead a mentor reading the repo root.
- **No checkpoint provenance:** nothing records which EXP produced which `.pt`.
  Since weights live off-git (Kaggle), this is how results get orphaned.

### 1.9 Research-quality gaps (real, but *not* why it detects nothing)

These are genuine and belong in the master plan — they are downstream of Phase 0:

- **Face branch generalizes poorly.** EXP-005: val AUC ≈ 0.99 → unseen DFDC AUC
  0.6564, recall 0.2960 (misses ~70% of fakes). Honest negative result.
- **General branch is under-trained.** EXP-004 is **1 epoch, backbone frozen**,
  stopped when MPS thrashed. 0.94 AUC is a *floor*, not a converged number.
- **Generation gap.** GenVidBench Pair1 generators (pika, VideoCrafter2,
  ModelScope, Text2Video-Zero) are **2023-era** T2V. Nothing in the training
  set resembles Sora / Veo / Kling / Runway Gen-3. Expect a large drop on 2025-26
  content — currently unmeasured.
- **Faceswap coverage.** The general branch was trained on *fully generated*
  video. A faceswap is a **real video with a locally edited region** — a
  different signal. Only the face branch targets it, and that is the weak one.
- **No robustness eval** (JPEG/resize/blur/FPS) — planned, never run.
- **Phases 6-10** (human, object, temporal, fusion, rich explainability) unstarted.

### Summary table

| # | Defect | Class | Blocks |
|---|---|---|---|
| 1.0 | No checkpoints → mock always | Serving | Everything |
| 1.1 | `VideoSpatialNet` unservable (two factories) | Serving | Best model |
| 1.2 | General path hardcoded to `BaselineDetector` | Serving | General branch |
| 1.3 | General gated behind face checkpoint | Serving | General-only deploy |
| 1.4 | Attention-pool discarded (mean-pool at inference) | Serving | Score validity |
| 1.5 | Face path raises on faceless video | Serving | AI-generated video |
| 1.6 | Thresholds uncalibrated + `max()` fusion | Serving | Trustworthy FPR |
| 1.7 | No image support | Serving | Image detection |
| 1.9 | Under-trained / generation gap / no robustness | Research | Accuracy |

---

## Part 2 — Phase-by-phase implementation

Ordering principle: **make one real model run end-to-end before improving any
model.** Phases 0-3 are the unblock; 4+ is quality. Phases 0-2 need no GPU and
no dataset download.

Each phase lists **Goal → Changes → Exit criteria**. Do not start a phase until
the previous one's exit criteria are met and recorded.

---

### Phase 0 — Make the serving path able to load the models we already trained
**Effort:** ~1 day, no GPU, no data. **This is the unblock — do it first.**

**Goal:** `VideoSpatialNet` loads and runs for inference, and the general branch
runs without a face checkpoint.

**Changes**

1. **Unify the two factories.** In `src/models/__init__.py`, make
   `build_model_from_config` dispatch on `model.family` *first* and delegate to
   `build_video_model_from_config` for `frame_baseline` / `frame_temporal` /
   `videomae` / `timesformer`; fall through to the existing
   frequency/temporal/baseline logic when `family` is absent (keeps every old
   face checkpoint loading unchanged). One factory, both paths.
2. **Add a clip-aware scoring path.** `VideoSpatialNet.forward` takes
   `(B,T,3,H,W)` and returns one logit pair per *video*; `predictor.py:_score_frames`
   assumes per-frame `(N,3,H,W)`. Branch on the built model: video-level models
   get the full clip in one call (preserving attention-pool), per-frame models
   keep today's behaviour. Fixes **1.4**.
3. **Decouple the branches.** Give `MultiTargetVideoDetector` independent
   optional checkpoints — face-only, general-only, or both — and make
   `model_service._RealModel` load whichever exist instead of raising when
   `best.pt` is absent (`model_service.py:70`). Reserve mock for *zero*
   checkpoints. Fixes **1.3**.
4. **Honour the checkpoint's own architecture** in
   `multi_target_predictor.py:73` — build the general model through the unified
   factory using the embedded config, not a hardcoded `BaselineDetector`.
   Fixes **1.2**.
5. **Return, don't raise, when no face is found.** `predictor.py:218` should
   yield a result with `prediction="unassessable"` / `prob_synthetic=None` and
   a reason in `meta`, so a faceless clip degrades instead of 500-ing.
   Fixes **1.5**.
6. **Align resize + `weights_only`** with training (bilinear;
   `weights_only=False` explicit in `predictor.py:136`). Fixes **1.8**.

**Exit criteria**
- A unit test builds `VideoSpatialNet` from a saved checkpoint config through
  `build_model_from_config` and runs a forward pass on a synthetic
  `(1,16,3,224,224)` tensor.
- A test asserts general-only construction succeeds with **no** `best.pt`.
- A test asserts a synthetic faceless clip returns `unassessable`, not an
  exception.
- `model_status()["mock"] is False` with only `general.pt` present.

---

### Phase 1 — Recover or retrain a servable checkpoint
**Effort:** 1 day recovery, or ~4 GPU-hours retrain.

**Goal:** a real `.pt` on disk that Phase 0's path can load.

The EXP-004 checkpoint (`best_model.pt`, val AUC 0.9632) and the EXP-005
checkpoints (`face_convnext.pt`, `face_xception.pt`) were produced on Kaggle and
are **not in this repo**. Per `PROJECT_MEMORY.md §8` they exist as committed-run
outputs.

**Changes**
1. **Recover first, retrain only if lost.** Download the Kaggle run outputs to
   `Samsung_PRISM_Video_Detection/checkpoints/`. Confirm each loads via Phase 0's
   factory and that its embedded `config.model.family` is present — EXP-004
   predates the unified factory, so it may need a one-time config-stamping
   migration (write a tiny `scripts/migrate_checkpoint.py` rather than editing
   `.pt` files by hand).
2. **If unrecoverable**, retrain EXP-004's config on a T4 as a *committed*
   Kaggle notebook (never interactive — `PROJECT_MEMORY.md §6` records losing
   3h of work twice to idle-timeout).
3. **Add checkpoint provenance.** A git-tracked `checkpoints/MANIFEST.md`
   mapping filename → EXP-ID → dataset/split → metrics → Kaggle run URL →
   sha256. Weights stay off-git; the *record* must not.

**Exit criteria**
- `general.pt` loads; a forward pass reproduces EXP-004 test AUC **0.9426 ±
  0.01** on the held-out `t2vz` manifest. *If it does not reproduce, stop — the
  checkpoint or the manifest is not what the doc claims.*
- `MANIFEST.md` committed.

---

### Phase 2 — First true end-to-end detection + image support
**Effort:** ~1 day.

**Goal:** upload a real AI-generated video **and a real AI-generated image** to
the dashboard and get a genuine, non-mock verdict.

**Changes**
1. **Smoke-test the real path.** Run ~20 clips (10 real, 10 `t2vz`) through
   `predict_video()` and assert `mock is False` and that scores separate.
2. **Add the image path** (fixes **1.7**): extend `allowed_extensions` with
   `.jpg/.jpeg/.png/.webp`; branch in `video_service.validate_upload` and
   `probe_metadata` so an image skips video probing; treat an image as `T=1`.
   The model side already works per-image — this is an API-layer change only.
   Suppress the temporal panel in the UI for `T=1` (variance/fluctuations are
   meaningless on one frame — do not display zeros as if measured).
3. **Surface the truth in the UI.** Show which branch(es) ran, per-branch
   scores, and `unassessable` explicitly. A user must never see "Real" when the
   honest answer is "no face found, general branch only".

**Exit criteria**
- Documented run: a real AI video and a real AI image both scored, `mock: false`,
  screenshots in `reports/`.
- **Write this up as EXP-006** — the first genuine end-to-end detection. Keep
  the numbers whatever they are.

---

### Phase 3 — Calibrate thresholds and fusion honestly
**Effort:** ~1 day. Fixes **1.6**.

**Goal:** one defensible operating point per branch, and a fusion rule whose
FPR is measured rather than assumed.

**Changes**
1. **Per-branch thresholds** chosen on each branch's *own* validation split for
   a stated target FPR (≤ 5% per the worklet metric). Store them in the
   checkpoint or config — never a literal in code. Delete the hardcoded `0.5`
   at `multi_target_predictor.py:115`; stop applying the FF++ face threshold
   (`0.5872`) to the general model.
2. **Measure the `max()` fusion.** Compute combined FPR/recall on a mixed set.
   If `max()` compounds FPR beyond target, compare against: noisy-OR, weighted
   mean, and coverage-gated routing. **Pick empirically and record the losers.**
3. **Reliability curve** per branch, so `confidence` means something. Today
   `confidence` is a threshold margin, not a probability.

**Exit criteria**
- `reports/EXP-007-calibration.json` with per-branch and fused
  accuracy/precision/recall/F1/ROC-AUC/PR-AUC/FPR/FNR + confusion, plus the
  fusion-rule comparison table including rejected options.

---

### Phase 4 — Close the generation gap (the real accuracy work)
**Effort:** 1-2 weeks, GPU. **This is the highest-value research phase.**

The most likely reason detection would still feel weak after Phases 0-3:
**the model has never seen a modern generator.**

**Changes**
1. **Build a 2025-26 eval set** — Sora, Veo, Kling, Runway Gen-3, Pika 2.x,
   plus modern faceswaps. Do **not** train on it; hold it out entirely. Even
   100-200 clips gives a real generalization number.
   *Record provenance for every clip; do not guess licences.*
2. **Measure the drop** — EXP-004's model on 2023 generators vs 2025-26. Publish
   the gap honestly; a large drop is the finding, not a failure.
3. **Then** finish the training EXP-004 never did: unfreeze the backbone, train
   to convergence on CUDA (not MPS), 16→32 frames. This is the cheapest real
   gain available, and it needs no new data.
4. **Head-to-head** vs VideoMAE / TimeSformer (master-plan Phase 3 — code
   exists, never run) on the *same* held-out split.
5. **Add faceswap-shaped training signal** — the general branch sees fully
   generated video; faceswaps are locally edited real video. Consider
   local/patch-level supervision rather than one video-level label.

**Exit criteria**
- `reports/EXP-008-modern-generators.json`: converged general model vs frozen
  baseline vs VideoMAE/TimeSformer, on 2023 **and** 2025-26 held-out sets.
- Winner selected on validation, never on parameter count.

---

### Phase 5 — Robustness
**Effort:** ~3 days. Master plan §5, never run.

JPEG/resize/blur/noise/brightness/frame-drop/FPS sweeps
(`src/evaluation/robustness.py`, `scripts/eval_robustness.py`). Report AUC vs
degradation curves. A detector that collapses under WhatsApp recompression is
not deployable, and right now that is unmeasured.

**Exit:** `reports/EXP-009-robustness.json` + curves.

---

### Phase 6 — Face branch: fix or formally de-scope
**Effort:** ~1 week.

EXP-005 left this at 0.65 AUC / 0.30 recall cross-dataset. Two honest options —
pick one, on evidence:

- **Fix:** run the prepared Celeb-DF holdout (`PROJECT_MEMORY.md §10.1`,
  `--holdout celeb`) for the apples-to-apples comparison against EXP-001's 0.71
  that EXP-005 could not provide. Then try video-level temporal aggregation over
  face crops (EXP-005 was crop-level, which discards temporal identity drift —
  a strong faceswap cue).
- **De-scope:** if it stays ≈0.65, state plainly that the face branch is a
  low-weight auxiliary signal and let the general branch carry faceswap
  detection. **A documented negative is a valid deliverable** — say it rather
  than shipping a branch that misses 70% of fakes.

**Exit:** EXP-010 with the Celeb-DF apples-to-apples number and an explicit
keep/de-scope decision.

---

### Phases 7-11 — The master plan's remaining research branches

These are unchanged from `VIDEO_MODULE_MASTERPLAN.md §4`; they now sit on a
working serving path, and each one plugs into Phase 0's unified factory:

| Phase | Branch | Notes |
|---|---|---|
| 7 | Synthetic-human (master P6) | Person detector → human crop → AIGC classifier. Covers full-body/back/distant where the face branch cannot. |
| 8 | Object/region forensics (P7) | AI object inside real video. Detector locates; forensic model judges. |
| 9 | Temporal forensics (P8) | Explicit frame-to-frame anomaly signals — most likely to catch *edited-real* video, the case both current branches are weakest on. |
| 10 | Learned fusion (P9) | MLP over branch embeddings+scores. **Replaces Phase 3's hand-picked rule** — only meaningful once ≥3 branches are calibrated. |
| 11 | Rich explainability (P10) | Per-branch evidence JSON + `schemas/rich_result.schema.json`. The worklet's ≥85% explainability target is unquantified today. |

**Recommended order if time is short: 9 → 7 → 10.** Temporal forensics targets
the gap the current branches genuinely cannot cover; fusion is worth little
before there are branches to fuse.

---

## Part 3 — Housekeeping to do alongside

- **Rewrite or archive `PROJECT_STATUS.md`** (repo root). It is two months stale
  and describes modules absent from this repo (§1.8). A mentor opening the repo
  reads it first and is actively misled.
- **Reconcile `PROJECT_MEMORY.md`** — its branch (`claude/samsung-prism-video-detection-kctonh`)
  is not the current working branch, and `docs/PROJECT_MEMORY.md` duplicates the
  uploaded copy. Keep one.
- **`checkpoints/MANIFEST.md`** (Phase 1) — the single highest-leverage
  housekeeping item, because weights live off-git.

---

## Part 4 — What to tell the mentors

Two sentences, both defensible from this repo:

> The research pipeline is validated — the general AI-video branch reaches
> **ROC-AUC 0.9426 on a completely unseen generator** (EXP-004), while the face
> branch reaches only 0.65-0.71 cross-dataset (EXP-001/005) — which is the
> empirical justification for making general synthetic-video detection primary
> rather than face-only.
>
> The end-to-end system does not yet detect, for an **integration** reason, not
> a modelling one: the inference path and the training path use two different
> model factories, so the 0.94-AUC architecture has no servable code path, and
> with no checkpoint on disk the dashboard runs a clearly-labelled mock. Phase 0
> of the enablement plan closes that gap in about a day.

Do **not** claim any accuracy number for the deployed system until Phase 2 has
produced one. Everything above is either measured (EXP-001/004/005) or explicitly
marked as unmeasured.
