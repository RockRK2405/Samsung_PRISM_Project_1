# Dataset Notes — FaceForensics++ (c23)

## Dataset
- **Name:** FaceForensics++ (FF++)
- **Version / release:** c23 (H.264 constant-rate-factor 23, "medium compression")
- **Original source:** Rössler et al., *"FaceForensics++: Learning to Detect
  Manipulated Facial Images"*, ICCV 2019.
- **Access route used for this project:** Kaggle mirror —
  <https://www.kaggle.com/datasets/xdxd003/ff-c23>
- **License / attribution note:** The Kaggle upload is a re-hosted mirror
  of the original FF++ dataset. For the final report and any external
  publication we must:
  1. Cite the original Rössler et al. paper as the true source.
  2. Note that the download route was the Kaggle mirror.
  3. Confirm the mirror complies with the original FF++ terms of use
     before releasing any derivative artefacts.

## Scale
- **Real videos:** 1,000 (YouTube-sourced pristine footage).
- **Synthetic videos:** 4,000 — 1,000 each across four manipulation methods:
  - **Deepfakes** — identity swap via autoencoder.
  - **Face2Face** — facial reenactment (expression transfer).
  - **FaceSwap** — graphics-based face swap.
  - **NeuralTextures** — GAN-based reenactment.
- **Total:** 5,000 videos.
- **Approximate size on disk:** ~16 GB at c23 compression.

## Characteristics
- **Resolution / FPS:** varies per source video; commonly ~480p–720p at ~25–30 fps.
- **Compression level:** c23 only (raw and c40 versions are out of scope for now).
- **Subject diversity:** moderate — YouTube-sourced faces, English-language
  interview / talking-head content dominates. Known to be less diverse
  than Celeb-DF v2 or DFDC.
- **Known biases / limitations:**
  - Fake generators are all from 2018–2019 vintage; models trained *only*
    on FF++ historically generalise poorly to newer generators — this is
    the primary motivation for adding a second (out-of-distribution)
    evaluation dataset in Milestone 8.
  - Backgrounds and lighting are limited compared to in-the-wild data.

## Intended Use in this Project
- **Milestone:** 2 (dataset survey) → 3 (frame + face extraction)
  → 4–6 (baseline & upgraded models) → 7 (explainability).
- **Split role:** primary **training + validation** dataset.
- **Held-out cross-dataset test:** *not* FF++ — will use Celeb-DF v2 or a
  DFDC subset (decision deferred to Milestone 8).
- **Manipulation subset for v1:** likely all four methods, so we can
  report per-method accuracy — but a two-method sanity run may come
  first to validate the pipeline quickly.

## Storage & Cache Plan
- **Raw video location** (see `configs/paths.yaml`): `data/raw/ff_c23/`.
- **Extracted frames:** `data/processed/frames/ff_c23/<split>/<label>/<video_id>/`.
- **Face crops (cached, primary training input):**
  `data/processed/faces/ff_c23/<split>/<label>/<video_id>/`.
- **Per-video metadata / labels manifest:** `data/metadata/ff_c23.json`.

## Baseline Comparability
Choosing c23 (and not raw) is a deliberate decision: the vast majority of
published deepfake-detection benchmarks — including the original FF++
paper's XceptionNet baseline — report numbers on c23. This means our
results will be *directly* comparable to the literature without extra
qualification.

## Open Issues
- [ ] Confirm the Kaggle mirror is complete (all 5000 videos, all 4 methods).
- [ ] Verify checksum / file counts after download.
- [ ] Decide train / val split ratio (candidate: 80 / 10 / 10, video-level, no subject leakage).
- [ ] Confirm the Kaggle mirror's terms of use are compatible with our worklet's redistribution needs.
