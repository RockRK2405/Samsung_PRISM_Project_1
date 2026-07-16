# PRISM Worklet 26TS08 — Multi-Modal Fusion Engine

Combines the four per-modality synthetic-data detectors (Image, Audio,
Text, Video) into one verdict, per the worklet's architecture (slides
2-3, 6): parallel modality detectors → fusion & scoring → explainable
QC decision.

## Status

**v1 — weighted-aggregation fusion, shipped and tested.**

| Modality | Adapter status | Model status |
|---|---|---|
| Audio | ✅ Wired | ✅ Trained checkpoint (`best_audio_model.pt`, F1 99.53%) |
| Video | ✅ Wired | ✅ Trained checkpoint (`best.pt`, F1 98.62% on FF++) |
| Image | ✅ Wired | ✅ Trained checkpoint (`image_detector_v1.pth`, verified — see below) |
| Text | ✅ Wired | ⚠️ No trained `model.pkl` yet — runs in degraded mode (perplexity + rule-based only). Picks up a real model automatically the moment one is trained; see `docs/INTEGRATION_NOTES.md`. |

Not yet implemented (see `docs/CROSS_MODAL.md` for what v1 covers instead):
- True cross-attention fusion layer (worklet's stated end-state architecture)
- Frame-level lip-sync mismatch detection
- Cost-aware adaptive routing (run cheap modalities first, escalate to expensive ones only when uncertain)

## Architecture

```
image_path ─┐
audio_path ─┼─→ FusionEngine.analyze() ─┬─→ ImageAdapter.predict()  ─┐
text       ─┤                          ├─→ AudioAdapter.predict()  ─┤
video_path ─┘                          ├─→ TextAdapter.predict()   ─┼─→ fuse() ─→ FusionResult
                                        └─→ VideoAdapter.predict()  ─┘
```

Each adapter wraps one teammate's model behind a uniform
`predict(...) -> ModalityResult` call (see `src/fusion_engine/schema.py`).
`fuse()` combines whichever `ModalityResult`s are available into one
`FusionResult` — modalities with no input, or whose adapter failed to
load, contribute zero weight rather than being silently treated as
"confidently real."

## Install

```bash
cd Samsung_PRISM_Fusion_Engine
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### CLI

```bash
# Any subset of the four inputs
python scripts/analyze.py --video path/to/clip.mp4 --text "the transcript"
python scripts/analyze.py --image path/to/photo.jpg --audio path/to/clip.wav
python scripts/analyze.py --video path/to/clip.mp4 --json-out result.json
```

### Python API

```python
from fusion_engine import FusionEngine

engine = FusionEngine()
result = engine.analyze(
    video_path="clip.mp4",
    text="the accompanying transcript",
)

print(result.verdict)            # "real" | "synthetic" | "uncertain"
print(result.fused_score)        # 0-1
print(result.fused_confidence)   # 0-1
print(result.cross_modal_flags)  # e.g. disagreement warnings
print(result.explanation)        # natural-language summary
```

## Output schema

See `src/fusion_engine/schema.py` for the authoritative dataclasses.
Every `ModalityResult` and the top-level `FusionResult` carry a
`schema_version` field (currently `1`) — bump only on incompatible
wire-format changes.

```python
FusionResult(
    verdict="synthetic",
    fused_score=0.87,
    fused_confidence=0.79,
    modality_results={...},          # one ModalityResult per modality, incl. unavailable ones
    weights_used={"audio": 0.32, "video": 0.30, "image": 0.0, "text": 0.18},
    cross_modal_flags=["audio (0.95) and video (0.05) disagree by 0.90 — recommend manual review"],
    explanation="Modalities analysed: audio, text, video. ...",
)
```

## Fusion weights

`configs/fusion_weights.yaml` holds the per-modality weights and
decision thresholds, with the rationale documented inline. These are a
**provisional v1 starting point** — the worklet architecture calls for
weights "tuned on validation set performance," which requires a joint
cross-modality validation set that doesn't exist yet (each module was
validated independently on its own dataset). Retune once that exists.

## Testing

```bash
python -m pytest tests/ -v
```

19 tests, split by dependency footprint:
- `test_schema.py`, `test_fusion.py` — pure logic, no ML dependencies, always runnable.
- `test_image_adapter.py` — real integration test against the actual checkpoint (skips gracefully if the checkpoint file isn't present in a given checkout).

Audio/Text adapter tests are not yet included in this pass — they
depend on `librosa`/`scikit-learn` respectively, which weren't
installed in the sandbox this was built in. Add `test_audio_adapter.py`
/ `test_text_adapter.py` following `test_image_adapter.py`'s pattern
once those dependencies are confirmed available in your environment.

## Directory layout

```
Samsung_PRISM_Fusion_Engine/
├── configs/
│   └── fusion_weights.yaml       # per-modality weights + thresholds
├── docs/
│   ├── INTEGRATION_NOTES.md      # what was found in each teammate's module + how it was wired
│   └── CROSS_MODAL.md            # v1 heuristic vs the worklet's full cross-attention vision
├── scripts/
│   └── analyze.py                # CLI entrypoint
├── src/fusion_engine/
│   ├── schema.py                 # ModalityResult, FusionResult dataclasses
│   ├── fusion.py                 # weighted aggregation + cross-modal flags (pure logic)
│   ├── engine.py                 # FusionEngine — loads adapters, orchestrates analyze()
│   └── adapters/
│       ├── image_adapter.py
│       ├── audio_adapter.py
│       ├── text_adapter.py
│       └── video_adapter.py
├── tests/
├── requirements.txt
└── README.md
```

## Known limitations (read before demoing)

- **Text module has no trained ML model yet** — its contribution to
  fusion is currently perplexity + rule-based only. Train
  `prism_text_detector.train_baseline` and drop the output into
  `Samsung_PRISM_Text_Detection/PRISM/artifacts/enhanced/` to upgrade
  it with zero code changes.
- **Image checkpoint was trained for 1 epoch** (proof-of-concept per
  the training notebook) — treat its contribution as a weaker signal
  until retrained further.
- **Cross-modal flags are a disagreement heuristic**, not true
  lip-sync or transcript-vs-speech verification — see
  `docs/CROSS_MODAL.md` for the honest scope of what v1 checks.
- **Fusion weights are provisional**, not tuned against a real joint
  validation set (none exists yet across all four modalities).
