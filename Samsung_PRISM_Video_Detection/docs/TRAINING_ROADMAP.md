# AI-Generated Video Detection — Training Roadmap (Colab + Google Drive)

**Goal:** one integrated detector that flags any synthetic video — fully
AI-generated (text/image-to-video), faceless content (landscapes, objects,
animals), modern commercial generators — plus the existing face-swap branch,
later fused with audio.

**Compute/storage:** Google Colab (T4 GPU) + Google Drive (350 GB).
**Baseline to beat:** EXP-004 `best_model.pt` — ConvNeXt-Tiny + attention-pool,
1 epoch, frozen backbone, unseen-generator ROC-AUC 0.9426. Every new model is
compared against it on the same held-out split.

> Rule carried over from the project: a dataset fact that has not been checked
> against the real files is marked **(verify)**. Do not treat it as settled.

---

## 0. The one decision that shapes everything: how to split

A detector that memorises *which generator* made a video looks perfect in
testing and fails on the next generator. So every phase uses
**leave-generator-out**: some generators are never seen in training and are
used only for testing. That unseen-generator AUC is the only number that
matters. A random train/test split of the same generators is meaningless here.

Same for reals: real test videos must come from videos (ideally a source)
never used in training.

---

## 1. Model choice — train two candidates, keep the winner

| | **A. CNN baseline (current)** | **B. CLIP features + temporal head (recommended)** |
|---|---|---|
| Model | ConvNeXt-Tiny per frame → attention-pool → classifier | Frozen CLIP ViT-L/14 image encoder per frame → small temporal attention head |
| Trained params | ~28M (full fine-tune) | ~1–3M (head only) |
| Why | Already built (`VideoSpatialNet`), learns pixel-level artifacts | Frozen CLIP features are known to generalise across unseen generators for AI-image detection (UnivFD, Ojha et al., CVPR 2023); extending to video by pooling frames is the natural first step |
| Colab cost | ~30–60 min/epoch on T4 for 10k videos **(estimate)** | Extract features **once** (~1–2 h per 50k videos **(estimate)**), then each training run takes minutes |
| Risk | Overfits to seen generators (EXP-004 showed it can still do well) | Misses very low-level artifacts; weaker on heavy compression |

**Plan:** train both in Phase 1, pick by unseen-generator AUC. B is likely
the better generaliser *and* far cheaper on Colab, because you extract
features once and then iterate on the head in minutes. If both are close,
keep both and fuse them (Phase 4).

**Classification** (which generator made it) is a second head on the same
features: a multi-class softmax over training generators plus "real". It can
only name generators it has seen; for anything else it should say "unknown
AI" (low max-probability). The **binary real/AI head is the deliverable**;
attribution is a bonus.

---

## 2. Drive layout and the storage trick

Raw video does not fit and is too slow to read from Drive. **Never train from
raw video on Drive.** Download each archive to Colab's local disk, extract
frames/features, write compact shards to Drive, delete the raw archive.

```
MyDrive/prism_video/
├── raw/            # temporary only — delete after extraction
├── frames/         # 16 frames/video, short side 256px JPEG, .tar shards + .jsonl index (~0.3 MB/video)
│   └── <generator or real-source>/shard-00000.{tar,jsonl}
├── features/       # CLIP ViT-L/14 per-frame features, fp16 .npy shards   (~32 KB/video)
├── manifests/      # train/val/test CSVs (video_id, label, generator, source, content)
├── checkpoints/    # every run, never overwritten; name + sha256 in MANIFEST.md
└── reports/        # one JSON per evaluation
```

Budget at 50k videos: frames ≈ 12–15 GB, features ≈ 2 GB. Even 200k videos
fits comfortably in 350 GB. Small-file I/O on Drive is the real bottleneck,
which is why frames go into `.tar` shards rather than millions of JPEGs.

**Colab survival rules** (the project has lost runs to these):
- Checkpoint to Drive **every epoch**; every training script must resume.
- Run long jobs with the browser tab active; free Colab disconnects on idle.
- Pin dependencies once (see `notebooks/Colab_Pipeline_v2.ipynb`): install
  `facenet-pytorch` with `--no-deps`, upgrade `kaggle`, never let numpy downgrade.

---

## Phase 1 — Fully AI-generated video: detection + classification

**Data (verified, already used by this project):** GenVidBench —
`huggingface.co/datasets/jian-0/GenVidBench`
- Pair1 (~74 GB): fakes `pika`, `vc2` (VideoCrafter2), `ms` (ModelScope),
  `t2vz` (Text2Video-Zero); reals `vript`.
- Pair2 (~120 GB): fakes `svd`, `musev`, `mora`, `cogvideo`; reals `hd_vg_130m`.
- Labels: `Pair1_labels.txt` / `Pair2_labels.txt`, `<rel_path> <label>`.

**Split:**
| | Fakes | Reals |
|---|---|---|
| train | pika, vc2, ms | vript (train portion) |
| val | held-out videos of pika, vc2, ms | vript (val portion) |
| **test-A** (unseen generator) | t2vz | vript (held-out) |
| **test-B** (unseen generators + unseen real source) | svd, musev, mora, cogvideo | hd_vg_130m |

Your existing `data/manifests/{train,val,test}.csv` already implement
train/val/test-A; reuse them so results stay comparable to EXP-004.

**Steps:**
1. (`notebooks/Phase1_Data_Prep.ipynb`) Download Pair1 archive by archive → extract 16 uniform frames/video →
   frame shards on Drive → delete archive. Then Pair2 the same way (test-B
   only, never trained on).
2. Extract CLIP ViT-L/14 features from the frame shards → feature shards.
3. Train **B** (CLIP + temporal head): binary head + generator-attribution head.
4. Train **A** (ConvNeXt + attention-pool) with a full fine-tune, not EXP-004's
   frozen 1 epoch: 5–10 epochs, unfreeze after epoch 1, early-stop on val AUC.
5. Evaluate both on test-A and test-B. Report AUC, recall, FPR, per-generator recall.

**Exit criterion:** test-A AUC ≥ EXP-004's 0.9426 and a recorded test-B number.
test-B is expected to be lower; how much lower is the finding.

---

## Phase 2 — Faceless content coverage (landscapes, objects, animals)

Not a new model: make sure the Phase 1 winner works **across content types**,
and fix it where it does not.

1. **Tag content.** Add a `content` column to every manifest row (people,
   face-closeup, nature/landscape, object, animal, urban, abstract). A CLIP
   zero-shot prompt over the middle frame does this automatically; spot-check 50.
2. **Evaluate per slice.** Recall and FPR per content type on test-A/test-B.
   Weak slices are where the work is.
3. **Fill gaps in reals.** If real landscapes or objects are under-represented,
   the model learns "nature ⇒ AI". Add real footage of those types, e.g. stock
   sites with permissive licences such as Pexels or Pixabay **(verify licence
   per clip)**, or Kinetics-style action clips **(verify availability)**.
   Match resolution and compression to the fakes, or the model learns the
   codec instead of the generator.
4. **Robustness.** Re-test with JPEG/H.264 recompression, downscaling and
   frame-rate change (WhatsApp/YouTube-style). Add these as training
   augmentations if recall collapses.
5. Fine-tune the winner on the rebalanced data; re-evaluate all slices.

**Exit criterion:** no content slice more than ~0.05 AUC below the overall
number, and FPR on real nature/object footage measured (not assumed).

---

## Phase 3 — Modern generators (Sora, Veo, Kling, Runway, Luma, Pika 2.x …)

The largest gap: nothing so far has seen a 2024–26 generator.

1. **Build the test set FIRST and freeze it.** Collect clips per generator
   from official showcase/gallery pages and public prompt galleries, or generate
   your own on free tiers **(check each tool's terms)**. Record source URL,
   generator, date and prompt for every clip. Pair with real footage of
   matching content and quality. Aim for ≥50 clips per generator.
2. **Measure the drop** of the Phase 1/2 model on it, before any training on
   modern data. This number is a result in itself.
3. **Candidate training data** (all **verify** before relying on them):
   GenVideo, the benchmark released with the DeMamba paper (2024), is reported
   to include commercial generators. Check licence, access and the exact
   generator list. Plus your own collected clips beyond the frozen test set.
4. **Fine-tune with leave-one-generator-out:** train on all modern generators
   but one, test on the held-out one; rotate. This estimates performance on
   the *next* generator you have not seen, which is the realistic deployment
   question.
5. Keep test-A / test-B in every evaluation, so improving on modern
   generators does not silently break older ones.

**Exit criterion:** a per-generator table (old + modern) with leave-one-out
AUC, and the gap stated honestly.

---

## Phase 4 — Merge into one integrated detector

"Merging checkpoints" here does **not** mean averaging weights (different
architectures and tasks cannot be averaged). It means one detector that runs
the right models and fuses their scores. Most of this already exists:
`src/inference/unified_detector.py`.

1. **Final branches:** general = Phase 3 winner; face = `face_celebdf_identity.pt`
   (EXP-006); optionally the CNN as a second general branch if Phase 1 showed
   A and B make different mistakes.
2. **Calibrate** each branch's threshold on a validation set (target FPR ≤ 5%),
   not on test, and not copied from config.
3. **Fusion:** start with the current OR rule; then train a tiny logistic
   regression on [branch scores + face coverage] on validation data and keep
   it only if it beats OR on test.
4. **Single deployable bundle:** one file holding every branch's state_dict,
   config, threshold and the fusion weights, plus sha256 in `MANIFEST.md`. The
   dashboard loads that one file.
5. Re-run every frozen test set through the dashboard path (as done for
   EXP-006) before calling it done.

## Phase 5 — Audio

Your teammate's module maps its output to the shared `DetectionResult`
contract (`docs/UNIFIED_PIPELINE.md`); a cross-modal fusion layer combines
video + audio the same way branches are fused. Needs a dataset with both real
and fake audio-visual pairs to evaluate. FakeAVCeleb is in your dataset
registry **(request form)**.

---

## Order of work and rough effort

| Step | Effort (estimate) |
|---|---|
| Phase 1 data: Pair1 + Pair2 download → frame/feature shards | 2–4 days, mostly unattended |
| Phase 1 training both candidates + eval | 3–5 days |
| Phase 2 tagging, slice eval, real-footage top-up, robustness | 1 week |
| Phase 3 collect + freeze modern test set | 1 week (the slow part is collection) |
| Phase 3 fine-tune + leave-one-out eval | 3–5 days |
| Phase 4 integration + calibration + dashboard verification | 2–3 days |

Each phase ends with an `EXP-NNN` doc + metrics JSON, negatives included.
