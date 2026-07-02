"""Download a deepfake-detection dataset into ``data/raw/``.

Milestone: 2 — Dataset survey & subset selection.

This script will resolve the target dataset (FaceForensics++, Celeb-DF v2,
DFDC sample, WildDeepfake, ...) from ``configs/dataset.yaml``, verify the
user's access credentials / EULA acceptance, download the requested
subset, and record a manifest under ``data/metadata/``.

Nothing is implemented yet.
"""

# TODO(milestone-2): parse configs/dataset.yaml and select target dataset.
# TODO(milestone-2): verify EULA / access token where required.
# TODO(milestone-2): stream download with resume support and checksum verification.
# TODO(milestone-2): write manifest JSON to data/metadata/<dataset>.json.


def main() -> None:
    """Entry point — not yet implemented."""
    raise NotImplementedError("scripts/download_dataset.py is a placeholder.")


if __name__ == "__main__":
    main()
