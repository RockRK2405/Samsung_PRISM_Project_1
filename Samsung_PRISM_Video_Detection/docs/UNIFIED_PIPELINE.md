# Unified Video Detection Pipeline

One entry point — `src.inference.UnifiedVideoDetector` — for every kind of
synthetic video. The dashboard uses it when `config.yaml` has
`pipeline: "unified"` (the default).

```
                         ┌─────────────────────────────────────────────┐
 video ──► 16 uniform ──►│ GENERAL branch  (always runs)               │──► p_general
           frames, RGB   │ VideoSpatialNet, ConvNeXt-Tiny + attn-pool  │
           (one decode)  │ full frames, scored as ONE clip             │
                         │ best_model.pt · EXP-004                     │
                         ├─────────────────────────────────────────────┤
                         │ FACE branch  (only if faces in ≥50% frames) │──► p_face
                         │ MTCNN crops → ConvNeXt-Tiny → mean-pool     │    or not_applicable
                         │ face_celebdf_identity.pt · EXP-006          │
                         └─────────────────────────────────────────────┘
                                             │
                          synthetic if ANY applicable branch ≥ its threshold
                                             ▼
                 DetectionResult(modality="video", prediction, prob_synthetic,
                                 meta.branches = per-branch scores + status)
```

## What each kind of video hits

| Input | Branch that decides | Measured evidence |
|---|---|---|
| Face-swap deepfake | face (general also scores) | **0.9896** video-level AUC, unseen identities, Celeb-DF v2 — *verified through this path* |
| Fully AI-generated video (text-to-video) | general | **0.9426** AUC, unseen generator t2vz — *offline only; not yet re-verified through this path* |
| Faceless AI video (landscape, object) | general (face → `not_applicable`) | covered by the general branch; no faceless-specific test yet |
| Modern generators (Sora / Veo / Kling / Runway) | general | **never measured** — the general model saw only 2023-era generators |

Fusion is recall-first: a video flagged by either branch is synthetic. False
positive rates therefore add across branches. Per-branch scores are always
returned, so measure this on real footage rather than assuming it.

## Verify each row (on the Mac, from `Samsung_PRISM_Video_Detection/`)

**1. Face-swap (re-check through the unified path)**
```bash
python3 scripts/score_manifest.py --manifest data/exp006_replay.csv \
  --out reports/unified_celebdf.json --mode direct
```
Expect AUC ≈ 0.99, as before. The JSON's `per_branch` shows both scores per video.

**2. Text-to-video (the GenVidBench test split EXP-004 used)**
```bash
python3 scripts/score_manifest.py --manifest data/manifests/test.csv \
  --out reports/unified_genvidbench.json --mode direct
```
Needs the GenVidBench videos still on disk at the paths in `test.csv`. Expect
AUC ≈ 0.94; per-group lines show `text2video-zero` recall and `vript` specificity.

**3. Modern generators — the open question**
Collect 30+ AI clips (Sora/Veo/Kling/Runway showcase pages) and 30+ real
clips of similar content, then:
```bash
python3 scripts/build_manifest.py --out data/modern.csv \
  --real ~/modern/real:real --fake ~/modern/ai:modern_t2v
python3 scripts/score_manifest.py --manifest data/modern.csv \
  --out reports/unified_modern.json --mode direct
```
A large drop from 0.94 here is a finding, not a failure: it means the general
branch needs fine-tuning on newer generators (full fine-tune on GPU; EXP-004
was 1 frozen epoch).

## Integrating audio (and the multimodal fusion stage)

Every modality should return the same shape the video pipeline returns:

```python
DetectionResult(
    modality="audio",            # "video" | "audio" | "image" | "text"
    prediction="synthetic"|"real",
    prob_synthetic=float,        # P(synthetic) in [0, 1], same meaning everywhere
    threshold=float,
    confidence=float,
    meta={"branches": {...}, "verdict_reason": str},
)
```

The audio module only needs a thin adapter that maps its output to this shape
(same `prob_synthetic` convention, 1 = synthetic). A multimodal fusion stage
then receives a list of these and combines them exactly as the video pipeline
combines its branches — so cross-modal disagreement (e.g. real video + cloned
voice) is visible per modality instead of averaged away.
