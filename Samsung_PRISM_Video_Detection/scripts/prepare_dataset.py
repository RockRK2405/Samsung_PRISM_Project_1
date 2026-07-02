"""Run face detection + cropping over extracted frames.

Milestone: 3 — Frame extraction & face-detection pipeline.

Planned behaviour:
    - Read extracted frames from ``data/processed/frames/``.
    - Detect faces (MediaPipe / RetinaFace) and track them across frames.
    - Crop, align, and cache faces to ``data/processed/faces/``.
    - Emit train / val / test manifests to ``data/metadata/``.

Nothing is implemented yet.
"""

# TODO(milestone-3): plug in a face-detection backend.
# TODO(milestone-3): implement identity-preserving cross-frame tracking.
# TODO(milestone-3): implement deterministic train/val/test splitting.


def main() -> None:
    """Entry point — not yet implemented."""
    raise NotImplementedError("scripts/prepare_dataset.py is a placeholder.")


if __name__ == "__main__":
    main()
