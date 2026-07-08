# Fusion-Engine Integration — Video Module

This document describes how the multi-modal fusion engine consumes the
video-detection module. Written for the Worklet 26TS08 fusion-engine
owner and the other three per-modality owners (image, audio, text).

## 1. Module identifier

```python
from src.inference import MODALITY, SCHEMA_VERSION

assert MODALITY == "video"
assert SCHEMA_VERSION == 1
```

Route incoming payloads by `MODALITY`. Bump-check `SCHEMA_VERSION` on
the fusion side and refuse to consume payloads from a version you
don't understand — we bump only on **incompatible** wire-format
changes (renamed field, removed field, changed semantics). Adding new
optional fields does NOT bump the version.

## 2. Public API surface

Three things you'll want:

```python
from src.inference import predict, VideoDetector, DetectionResult
```

### 2.1 One-shot — cheapest to use

```python
result = predict("path/to/clip.mp4")
# result is a DetectionResult instance
```

Uses the project's canonical production detector (Milestone-4 baseline,
FF++-val-tuned threshold ≈ 0.5872). Detector is loaded on first call
and cached for subsequent calls in the same process.

### 2.2 Repeated / batched — for the fusion pipeline

```python
detector = VideoDetector.load_default()
for path in queue:
    r = detector.predict(path)
    ...
```

Or with an explicit checkpoint / threshold / device:

```python
detector = VideoDetector(
    checkpoint_path="checkpoints/best.pt",
    threshold=0.5872,
    device="mps",     # optional; auto-selects if omitted
)
```

### 2.3 Skip explanation for latency-critical calls

`predict(..., produce_explanation=False)` skips the GradCAM +
timeline pass and returns a `DetectionResult` with `explainability_score`,
`heatmap_dir`, `timeline_path` all set to `None`. Score + confidence
are unchanged.

## 3. Output contract — `DetectionResult`

Every prediction returns a `DetectionResult` dataclass whose JSON
serialisation matches [`schemas/video_detection_result.schema.json`](../schemas/video_detection_result.schema.json).

Required fields (present on every payload):

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int (const 1) | Wire-format version |
| `modality` | str (const "video") | Fixed identifier |
| `video_path` | str | Input path |
| `prediction` | `"real"` \| `"synthetic"` | Discrete decision |
| `prob_synthetic` | float in [0, 1] | Video-level P(synthetic), mean-pooled |
| `confidence` | float in [0, 1] | Threshold margin normalised |
| `threshold` | float in [0, 1] | Threshold that produced the prediction |
| `per_frame_scores` | list[float] | One P(synthetic) per usable frame |
| `num_frames_used` | int | How many frames actually scored |
| `meta` | dict | Model class, checkpoint, backbone id |

Optional fields (present when `produce_explanation=True`):

| Field | Type | Meaning |
|---|---|---|
| `explainability_score` | float in [0, 1] \| null | Peak-inside-central over frames |
| `heatmap_dir` | str \| null | Directory of per-frame GradCAM PNGs |
| `timeline_path` | str \| null | Per-frame P(synthetic) plot PNG |

## 4. Deserialising a stored payload

```python
result = DetectionResult.from_json("outputs/predictions/clip.json")
# or
result = DetectionResult.from_dict(json_dict)
```

`from_dict` drops unknown fields silently — safe against a future
minor version that adds fields you don't know about yet.

## 5. Confidence semantics

`confidence` is the **threshold margin normalised to [0, 1]**:

```
confidence = min(|prob_synthetic - threshold| / max(threshold, 1 - threshold), 1)
```

That means:
- `0.0` — the model was right on the decision boundary (least trust)
- `1.0` — the model was maximally far from the boundary

Use this for adaptive routing: low confidence → escalate to a heavier
model or a human reviewer; high confidence → trust the light detector
alone. It is **not** an entropy / softmax margin — those are useless
for calibration decisions when the threshold isn't 0.5.

## 6. Known limitations

- Only the `prob_synthetic` is calibrated (via the FF++-val threshold
  tuning). `confidence` is not calibrated against real user
  disagreement — it's a bounded margin, not a probability.
- Cross-dataset performance drops sharply (see EXP-001 in
  `docs/experiments/`) — on Celeb-DF the model over-fires on unfamiliar
  reals. The fusion engine should not trust the video module's
  standalone decision on out-of-distribution inputs.
- Explainability heatmaps concentrate on the face for reals and on
  boundary artefacts for fakes — see EXP-003. This is not a bug; it
  matches how CNN deepfake detectors are known to work. Just be aware
  that a face-centric heatmap doesn't necessarily mean "correctly
  detected fake."

## 7. Health-check smoke test

Fastest way for the fusion engine to validate a checkout of this
module works:

```python
from src.inference import predict

r = predict("path/to/any/video.mp4")
assert r.modality == "video"
assert r.schema_version == 1
assert 0 <= r.prob_synthetic <= 1
assert r.prediction in {"real", "synthetic"}
```

If those four assertions pass, the video module is wired up correctly.
