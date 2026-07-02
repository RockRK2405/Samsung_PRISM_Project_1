"""Train a video-deepfake detector from the current configuration.

Milestones: 4 (baseline CNN) → 5 (temporal head) → 6 (dual-stream).

Planned behaviour:
    - Load ``configs/train.yaml`` + ``configs/model.yaml`` + ``configs/dataset.yaml``.
    - Instantiate the model, dataloaders, optimiser, scheduler, and tracker.
    - Run the training loop, checkpoint the best model into ``checkpoints/``.
    - Write metrics into ``outputs/metrics/`` and logs into ``logs/``.

Nothing is implemented yet.
"""

# TODO(milestone-4+): implement training entry point wired to src/training.


def main() -> None:
    """Entry point — not yet implemented."""
    raise NotImplementedError("scripts/train.py is a placeholder.")


if __name__ == "__main__":
    main()
