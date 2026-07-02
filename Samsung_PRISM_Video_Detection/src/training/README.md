# `src/training`

**Purpose.** Training loops, optimisers, schedulers, checkpointing, and
experiment-tracking integration.

Planned responsibilities (not yet implemented):

- Reusable `Trainer` class driven by `configs/train.yaml`.
- Optimiser / scheduler factories.
- Loss functions (cross-entropy, focal, contrastive variants).
- Mixed-precision + MPS support for Apple Silicon.
- MLflow / Weights & Biases hooks.
- Deterministic seed control.
