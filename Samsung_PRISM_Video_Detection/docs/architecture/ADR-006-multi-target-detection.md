# ADR-006 — Multi-target detection: faces, humans, AND general objects/scenes

## Context

Every checkpoint and dataset in this module to date (FaceForensics++,
Celeb-DF, EXP-001) targets one narrow task: **face-swap / reenactment
deepfake detection on a video that already contains a face.** Samsung's
team has now asked for a broader scope — detect AI-generated content
across **humans, faces, AND objects/scenes** (e.g. a fully Sora/Veo/
Midjourney-generated video of a car, a room, an animal — content with
no manipulated face at all, sometimes no face whatsoever).

The current pipeline **cannot see this class of input**. Its first
preprocessing step is MTCNN face detection; a video with no face never
reaches the classifier. This is not a fine-tune — it is a second
detection path.

## Decision

Add a **second, independent detection path** ("general path") that
classifies full video frames for AI-generation artifacts, without
requiring a face. Keep the existing face-path (MTCNN → crop →
`BaselineDetector`) completely unchanged — it is validated (EXP-001) and
still the right tool when a face IS present. Route between the two paths
(or combine both) at inference time.

**Why additive, not a from-scratch redesign:**
1. The face-path's F1=0.9875 (FF++) / F1=0.8142 (Celeb-DF cross-dataset,
   EXP-001) result is real, measured, and would be thrown away by a
   ground-up rebuild.
2. `BaselineDetector` (`src/models/baseline.py`) is architecturally
   already a generic per-image classifier — `forward()` takes any batch
   of `(B, 3, 224, 224)` images and has zero face-specific logic. The
   *only* thing that made the existing checkpoint face-specific was the
   **data it was trained on** (MTCNN-cropped faces). Training a second
   instance of the exact same class on general (non-face) AI-generated
   images gives us a second checkpoint for free, architecturally — no
   new model code needed for v1 of the general path.

## Architecture

```
                         ┌─── FrameExtractor (existing, unchanged) ───┐
                         │   32 uniformly-sampled RGB frames          │
                         └──────────────────┬──────────────────────────┘
                                             │
                     ┌───────────────────────┴────────────────────────┐
                     ▼                                                 ▼
         MTCNN face detection                               Resize full frame
         (existing, unchanged)                               to 224×224
                     │                                                 │
        face found in ≥N frames?                                       │
           │                  │                                        │
          yes                 no                                       │
           │                  └────────────────┬───────────────────────┘
           ▼                                    ▼
  Face-path model                      General-path model
  (checkpoints/best.pt,                (checkpoints/general.pt,
   existing, UNCHANGED)                 NEW — GenImage-trained)
           │                                    │
           └─────────────────┬──────────────────┘
                              ▼
                     Router / combiner
              (see "Routing logic" below)
                              │
                              ▼
                   Final prob_synthetic
```

## Routing logic (v1, deliberately simple)

- If MTCNN finds a face in **≥50% of the 32 sampled frames** → this is
  face-centric content. Run the face-path model on the face crops (as
  today). **Also** run the general-path model on the full frames.
  Final score = `max(face_score, general_score)` — either path flagging
  synthetic is enough to flag the submission. (Rationale: a face-swapped
  person standing in a fully-AI-generated room should be caught by
  *either* signal; taking the max avoids one weak modality dragging down
  a correct catch from the other, mirroring the lesson from the fusion
  engine's confidence-gating work.)
- If face coverage is **<50%** → no reliable face signal. Run
  **general-path only**. (Running face-path on a handful of accidental
  face detections would be noise, not signal — same principle as the
  fusion engine's "abstain when a modality lacks real evidence.")
- **Cost-aware framing (matches the worklet's stated goal):** the
  general-path model is cheap (single EfficientNet-B0 forward pass,
  no MTCNN). Always running it first as a low-cost filter, and only
  paying for MTCNN + face-path when the general-path result is
  ambiguous, is a natural v2 optimization — noted here, NOT implemented
  in v1 (v1 always runs both when a face is present, for correctness
  first).

## Dataset for the general path

**Chosen: GenImage (Stable Diffusion v1.4 subset), via Kaggle:**
`vtphatt2/genimage-stable-diffusion-v1-4` — no request form, ungated,
part of the peer-reviewed GenImage benchmark (1M-scale, real/fake pairs
across ImageNet-style object/scene classes). Confirmed reachable from
the training machine (Kaggle CLI already works there per prior fusion-
engine dataset work).

Supplementary/fallback options if the SDv1.4 subset alone underperforms,
same benchmark family, also Kaggle-hosted and ungated:
- `renhuang8/genimage-subset-detection`
- `aloktantrik/a-dataset-of-34500-labeled-images`

**Honest scope note:** GenImage is *images*, not video — there is no
large, free, ungated dataset of general (non-face) AI-*generated video*
at the time of writing (Sora/Veo-specific video datasets are either
unreleased or heavily gated). Training the general-path model on
GenImage's still images and applying it per-frame is a legitimate first
step (the existing architecture is already per-frame + mean-pool), but
it will not have learned any *temporal* artifact specific to generated
video — only generated-image artifacts. This is the same kind of
honest generalization gap already documented in EXP-001, and should be
reported the same way, not glossed over.

## What this ADR does NOT decide yet

- Whether the general-path model should eventually be a **different**
  backbone than EfficientNet-B0 (GenImage benchmark leaderboards often
  favor ResNet-50 or CLIP-based detectors for cross-generator
  generalization) — v1 keeps EfficientNet-B0 for consistency and to
  isolate "does a second path help at all" from "which architecture is
  best," per this module's existing habit of one isolated experiment at
  a time.
- Final routing thresholds/coverage percentage (50% face-coverage cutoff
  above is a starting guess, not tuned — to be revisited once we have
  validation data, same caveat as the fusion engine's provisional
  weights).

## Execution plan (division of labor)

Per team agreement: all code (dataset prep, training script, router) is
written and debugged here. Downloading the dataset and running training
on a GPU (Colab) is executed by the module owner. See
`scripts/prepare_general_dataset.py` and `scripts/train_general.py`.
