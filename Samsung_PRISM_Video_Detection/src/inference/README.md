# `src/inference`

**Purpose.** Single-video prediction API — the entry point the fusion
engine will eventually call.

Planned responsibilities (not yet implemented):

- `predict(video_path) -> DetectionResult` API.
- Batched inference for offline evaluation.
- Confidence calibration.
- Output schema matching the fusion-engine contract (score, confidence,
  explanation payload).
