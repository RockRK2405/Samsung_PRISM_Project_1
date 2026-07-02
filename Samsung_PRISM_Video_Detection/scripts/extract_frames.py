"""Decode videos into sampled frames and write them under ``data/processed/``.

Milestone: 3 — Frame extraction & face-detection pipeline.

Planned behaviour:
    - Read the list of source videos from ``data/metadata/<dataset>.json``.
    - Apply the sampling strategy configured in ``configs/dataset.yaml``
      (uniform / dense / keyframe).
    - Write extracted frames as PNG/JPEG under ``data/processed/frames/``.

Nothing is implemented yet.
"""

# TODO(milestone-3): implement video decoding via PyAV or OpenCV.
# TODO(milestone-3): implement configurable sampling strategies.
# TODO(milestone-3): write per-video frame manifests to data/metadata/.


def main() -> None:
    """Entry point — not yet implemented."""
    raise NotImplementedError("scripts/extract_frames.py is a placeholder.")


if __name__ == "__main__":
    main()
