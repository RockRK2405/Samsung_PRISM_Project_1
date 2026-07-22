# Integration notes — what was found in each module, and how it was wired

This records the actual state of each teammate's module at the time
the fusion engine was built, and the specific adaptation each adapter
had to make. Written so a future contributor (including a teammate
retraining their own model) knows exactly what to update and where.

## Audio — `Samsung_PRISM_Audio_Detection/audio_authenticity_detector/`

- **Checkpoint**: `outputs/best_audio_model.pt` — a plain `state_dict()`
  for `AudioCNN` (4-block CNN, ~1.2M params, input `(1,1,128,251)`
  log-Mel spectrogram).
- **Adapter approach**: `fusion_engine/adapters/audio_adapter.py`
  **reimplements** the preprocessing (`preprocess.py`'s
  `load_fixed_length_audio` / `wav_to_logmel`) and the `AudioCNN`
  architecture verbatim, rather than importing across the module
  boundary. Reason: the source module's `src/` files use bare imports
  (`from preprocess import ...`) that only resolve when that specific
  `src/` directory is the working directory — reusing them from here
  would require fragile `sys.path` manipulation. Reimplementing ~40
  lines of pure numpy/torch logic was judged more robust than that.
- **If the audio model gets retrained**: as long as the new checkpoint
  is still a `state_dict()` for the same `AudioCNN` architecture (same
  layer shapes), just replace the `.pt` file — no adapter code changes
  needed. If the architecture changes (e.g. switching to the
  `Wav2Vec2Head` Track-B option already stubbed in `model.py`), update
  `AudioCNN` in `audio_adapter.py` to match, or better, have this
  adapter genuinely import the module's `model.py` once it gets a
  proper `__init__.py` / installable package layout.
- **Class convention**: index 0 = REAL, index 1 = FAKE.

## Text — `Samsung_PRISM_Text_Detection/PRISM/`

- **No trained model exists yet.** `src/prism_text_detector/ensemble.py`'s
  `PrismEnsemble` gracefully degrades to perplexity + rule-based
  linguistic-pattern scoring when no `model.pkl` is found — confirmed
  via `website/backend/model_loader.py::_find_model_dir()`'s search
  logic, which checks (in order):
  `artifacts/enhanced/`, `artifacts/hc3_text_baseline/`,
  `artifacts/hc3_text_baseline_smoke/`.
- **Adapter approach**: `text_adapter.py` imports
  `prism_text_detector` directly (adds
  `Samsung_PRISM_Text_Detection/PRISM/src` to `sys.path`) since that
  package has a proper `__init__.py` and is meant to be pip-installed
  per its `pyproject.toml` — no reimplementation needed, unlike Audio.
  Mirrors `_find_model_dir()`'s search order so it picks up a real
  trained model automatically.
- **To upgrade from degraded mode**: train a real model with (per
  `PROJECT_BRIEF.md`):
  ```bash
  python -m prism_text_detector.train_baseline \
      --data <HC3_dataset_path> \
      --out artifacts/enhanced \
      --target-fpr 0.05
  ```
  Drop the resulting `artifacts/enhanced/model.pkl` into the repo and
  the adapter will pick it up with zero code changes.

  HC3 itself isn't in the repo — pull it locally first:
  ```bash
  pip install datasets
  python -c "
  from datasets import load_dataset
  ds = load_dataset('Hello-SimpleAI/HC3', 'all')['train']
  ds.to_json('hc3_all.jsonl')
  "
  python -m prism_text_detector.train_baseline \
      --data hc3_all.jsonl --out artifacts/enhanced --target-fpr 0.05
  ```
  Also blocked from this remote session for the same reason as the image
  retraining above (`huggingface.co` egress denied) — run locally.

- **Perplexity signal disabled (separate issue) — fixed by capping
  `transformers`, NOT by bumping torch**: an unbounded
  `transformers>=4.30` install will happily resolve to a release that
  requires `torch>=2.4` at runtime, and silently drops its PyTorch
  backend below that (tokenizer/config-only mode — no crash, just a
  quiet capability loss, which is what you saw).

  **First attempt at this fix was wrong and briefly broke the Video
  adapter** — bumping torch to `>=2.4` satisfies transformers, but
  `facenet-pytorch==2.6.0` (the video module's MTCNN face detector)
  hard-pins `torch<2.3`. Both adapters share one venv in the fusion
  engine, so torch can't be bumped for one without breaking the other.
  Reverted.

  **Correct fix**: keep torch pinned to `>=2.2,<2.3` (satisfies
  facenet-pytorch) and cap `transformers` to `>=4.30,<4.45` instead —
  well inside the range that still supports DistilGPT-2 perplexity
  scoring on older torch. Both `pyproject.toml`'s `perplexity` extra and
  the fusion engine's `requirements.txt` now read this way. If a future
  perplexity feature genuinely needs a newer `transformers`, the torch
  ceiling is the thing to renegotiate then (e.g. dropping
  `facenet-pytorch` for `mediapipe`, which the video module's
  `requirements.txt` already lists as an untried alternative face
  detector — see ADR-002 there).

  To apply in an existing venv:
  ```bash
  pip install "torch>=2.2,<2.3" "torchvision>=0.17,<0.18"
  pip install "transformers>=4.30,<4.45"
  ```
- **Extra dependency not in any requirements.txt**: `app.py`'s file
  upload endpoints import `docx` and `PyPDF2` — irrelevant to the
  fusion engine (we call `PrismEnsemble.analyze()` directly, not
  through the Flask app), but worth knowing if you ever reuse
  `website/backend/app.py` as-is.

## Image — `Samsung_PRISM_Image_Detection/`

- **No `src/` package existed** — only two Colab notebooks (`V1`
  training, `V2` inference-plus-forensics). No importable function
  anywhere.
- **Checkpoint**: `checkpoints/image_detector_v1.pth` — added to the
  repo after the fact (it only existed on the teammate's local
  machine, downloaded from Colab via `files.download()`, until it was
  committed). **Verified before use** (see below) — this matters
  because a checkpoint that "exists" doesn't guarantee it's actually
  the fine-tuned weights rather than an accidental save of the
  untrained default.
- **Verification performed**:
  1. `classifier.1.weight` shape is `[2, 1280]`, not `[1000, 1280]` —
     confirms the 2-class CIFAKE head was actually trained and saved,
     not left at ImageNet's default 1000-class head.
  2. Strict `load_state_dict()` against a freshly-constructed
     `torchvision.models.efficientnet_b0(weights=None)` with the same
     2-class head succeeded with **all 360 keys matched** — confirms
     architecture-checkpoint compatibility exactly.
  3. Forward-pass sanity check on two different synthetic test images
     produced different (non-degenerate) probability outputs —
     confirms the model responds to input rather than being a
     constant/broken function.
- **Adapter approach**: `image_adapter.py` reimplements V2's
  prediction-cell logic as a real `predict()` function, including V1's
  exact test-time transform (`Resize(224,224)` → `ToTensor()` →
  ImageNet normalisation — no augmentation).
- **Class convention**: index 0 = FAKE, index 1 = REAL (from V1's
  `ImageFolder` alphabetical ordering) — **the opposite of Audio's
  convention**. The adapter normalises both to the fusion engine's
  universal `prob_synthetic` meaning; this is exactly the kind of
  per-module convention mismatch the shared schema exists to hide.
- **Not carried over from V2**: the notebook's ad-hoc FFT-entropy /
  PRNU-noise "forensic ensemble" vote (majority vote against the CNN,
  ±0.05/-0.15 confidence nudges). Its thresholds (`fft_entropy > 5.0`,
  `prnu_std < 20`) had no supporting validation in the notebook.
  Treat as an experimental v2 addition if the image team wants to
  formalise and validate it later — don't wire it into fusion
  un-validated.
- **Training caveat**: V1 trained for only 1 epoch on a 20k-image
  CIFAKE subset — a proof of concept, not a finished training run.
  Expect this to be the weakest-validated of the four checkpoints
  despite the healthy 95.18% reported accuracy.
- **Retraining path (added)**: `Samsung_PRISM_Image_Detection/scripts/train.py`
  is a non-notebook port of V1's training loop — same transforms, same
  2-class head swap, but with `--epochs` and `--subset-size` as real CLI
  args instead of hardcoded `1` / `20000`. To produce a stronger
  checkpoint:
  ```bash
  # CIFAKE (Kaggle, public dataset — no request form):
  kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images --unzip -p CIFAKE
  python scripts/train.py --data-root CIFAKE --epochs 8 --subset-size 0 \
      --out checkpoints/image_detector_v2.pth
  ```
  Blocked from running it in this remote session — Kaggle needs your own
  credentials and this environment's egress policy also does not allow
  `huggingface.co`/dataset-host traffic (confirmed via
  `/root/.ccr/__agentproxy/status`, `connect_rejected` / 403 to
  `huggingface.co:443`). Run it locally, then commit the resulting
  `.pth` + its sibling `.metrics.json` — `image_adapter.py` needs no code
  change as long as the new checkpoint is still a 2-class EfficientNet-B0
  `state_dict()`.

## Video — `Samsung_PRISM_Video_Detection/`

- Already had a clean, versioned public API
  (`src.inference.VideoDetector` / `predict()`) from that module's own
  Milestone 8 (fusion-integration) work — no adaptation needed beyond
  wrapping its `DetectionResult` into a `ModalityResult`.
- `video_adapter.py` uses `VideoDetector.load_default()`, which loads
  `checkpoints/best.pt` — the Milestone-4 mean-pool baseline (the
  cross-dataset-tested one), not the dual-stream or mixed-training
  checkpoints from later experiments. See that module's
  `docs/experiments/` for why the baseline remains the production
  choice (temporal-head and dual-stream experiments were both
  documented negative results).
- `produce_explanation` defaults to `False` in the adapter (skips
  GradCAM + timeline generation) to keep the fusion engine's hot path
  fast; the QC dashboard should request an explanation on-demand only
  for a specific flagged submission, not on every single one.
