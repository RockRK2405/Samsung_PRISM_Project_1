# `src/preprocessing`

**Purpose.** Turn a raw video file into a tensor-ready stream of face
crops that downstream models can consume.

Planned responsibilities (not yet implemented):

- Metadata extraction (fps, duration, codec, resolution).
- Frame extraction and sampling (uniform / keyframe / dense strategies).
- Face detection (MediaPipe / RetinaFace).
- Face tracking across frames (to keep identity consistent).
- Face-crop generation and disk caching.
- Normalisation / resizing utilities shared by every model.

No implementation lives here yet.
