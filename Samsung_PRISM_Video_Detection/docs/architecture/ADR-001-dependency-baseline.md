# ADR-001: Baseline dependency set for the Video Detection Module

## Status
Accepted

## Date
2026-07-02

## Context
Milestone 1 requires committing to a concrete Python dependency set so
that development can move past scaffolding. The module targets Python
3.11 on macOS Apple Silicon, with PyTorch on the MPS backend as the
training runtime.

Two questions had to be answered:
1. Which libraries do we depend on today?
2. How strictly are they pinned?

## Options Considered

### Q1 — Library choices
- **Deep learning framework:** PyTorch vs TensorFlow/JAX.
  PyTorch chosen: dominant in the deepfake-detection literature we
  will build on (XceptionNet, ViT via `timm`, GradCAM tooling), and
  first-class MPS support on Apple Silicon without a custom index URL.
- **Video decoding:** OpenCV alone vs OpenCV + PyAV.
  Both included. OpenCV is convenient; PyAV is more robust for edge
  cases (variable frame rates, unusual codecs). Cheap to keep both.
- **Face detection:** MediaPipe vs facenet-pytorch (RetinaFace / MTCNN).
  Both listed for Milestone 3 A/B testing. ADR-002 will pick the winner.
- **Backbones:** `timm` chosen for one-line access to XceptionNet,
  EfficientNet, and ViT variants — matches the model progression in
  README milestones 4–6.
- **Experiment tracking:** MLflow chosen over Weights & Biases. Runs
  locally without an external account, which suits a laptop dev setup;
  W&B remains an option later if collaboration warrants it.
- **CLI framework:** Typer, for typed CLIs in `scripts/`.
- **Explainability:** grad-cam + captum, to cover both CNN and
  attention-based interpretability needs.

### Q2 — Pinning strategy
- **Exact pins (`==`)** — maximum reproducibility, but constant
  version-bump friction on a 4-month project.
- **Lower bounds (`>=`)** — reproducible enough for a research module,
  no friction on minor upgrades.
- **Full lockfile (pip-tools / uv)** — overkill until the pipeline
  actually runs end-to-end.

## Decision
Adopt PyTorch + `timm` as the modelling stack, MediaPipe + facenet-pytorch
side-by-side for face detection until ADR-002, PyAV + OpenCV for video
IO, MLflow for tracking, Typer for CLIs. Pin with lower bounds (`>=`)
for now; revisit once Milestone 6 (dual-stream model) is complete and
lock the environment then.

Runtime deps live in `requirements.txt`; development-only tools
(pytest, ruff, mypy, pre-commit) live in `requirements-dev.txt` so the
production install stays lean.

## Consequences
- **Positive:** Clear, small, well-motivated dependency list. No CUDA
  dependencies, so laptop-first development is friction-free.
- **Negative:** Carrying two face-detection libraries until ADR-002
  costs ~200 MB of install size. Acceptable.
- **Follow-up work:**
  - ADR-002 — pick a single face-detection backend after Milestone 3.
  - ADR-003 — device-selection policy (MPS / CUDA / CPU).
  - Produce a full lockfile at Milestone 6.

## References
- `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`
- README §4 Planned Architecture, §6 Environment Setup
