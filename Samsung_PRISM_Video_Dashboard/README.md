# Video Synthetic Media Detection — Research Testing Dashboard

A local FastAPI + React dashboard for testing the Video Deepfake Detection
module: ingest real and AI-generated (Gemini/Veo/Sora/…) videos, run them
through the detector, inspect frame-level + temporal behaviour, and measure
F1 / FPR against ground-truth labels.

The dashboard is **model-agnostic**. It plugs into the existing Video
Detection module through a single function, `model_service.predict_video()`,
which wraps `src.inference.VideoDetector` from the sibling repo. Nothing
about the model architecture is duplicated here.

## Architecture

```
Upload → validate (video_service) → OpenCV metadata probe
       → predict_video() [model_service wraps VideoDetector]
            → FrameExtractor(32, uniform) → MTCNN face crop
              → EfficientNet-B0 per-frame → mean-pool → DetectionResult
       → derive temporal stats + approx frame timestamps
       → SQLite (experiments.db)
       → FastAPI JSON → React dashboard
```

If the model checkpoint can't be loaded, the backend falls back to a
clearly-marked **MOCK MODEL** (every mock result carries `"mock": true`
and the UI shows a banner). Mock output is never presented as real.

## Setup

Use the **same venv** that already has the video module's torch/model deps,
so the dashboard can import the real model:

```bash
cd Samsung_PRISM_Video_Dashboard
source ../Samsung_PRISM_Fusion_Engine/.venv/bin/activate
pip install -r requirements.txt
```

## Run

**Backend** (from the dashboard root):
```bash
uvicorn backend.main:app --reload --port 8000
```
- API docs: http://127.0.0.1:8000/docs
- On startup it logs whether the REAL model or the MOCK model is active.

**Frontend** (dev, separate terminal):
```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

For a single-command production run, build the frontend once
(`npm run build`) — FastAPI then serves it at http://127.0.0.1:8000.

## Configuration

Everything model- and path-related lives in `config/config.yaml` — the
checkpoint path, decision threshold, frames sampled, device, etc. Change
it there, not in code.

## What the connected model provides vs. what's hidden

The real `DetectionResult` supplies: prediction, P(synthetic), per-frame
scores, threshold, GradCAM heatmaps, and a face-localisation
explainability score. The dashboard **derives** temporal statistics and
approximate frame timestamps from those. It **gracefully hides** sections
the model does not produce — face bounding boxes and attention/artifact
maps — rather than fabricating them.
