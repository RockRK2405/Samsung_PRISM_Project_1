"""Run the detector on a single video from the command line.

Milestone: 7+ — Explainability & integration.

Planned behaviour:
    - Accept a path to a video file.
    - Load the current production checkpoint.
    - Run the full pipeline: preprocess -> features -> temporal head -> classifier.
    - Emit a JSON payload matching the fusion-engine contract, plus an
      explainability artefact (heatmap / per-frame timeline).

Nothing is implemented yet.
"""

# TODO(milestone-7+): implement single-video inference CLI backed by src/inference.


def main() -> None:
    """Entry point — not yet implemented."""
    raise NotImplementedError("scripts/predict.py is a placeholder.")


if __name__ == "__main__":
    main()
