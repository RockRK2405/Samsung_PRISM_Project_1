"""Unit tests for the Celeb-DF v2 test-list parser.

Only the parser is exercised — full extraction touches OpenCV and
MTCNN and is covered by the manual smoke test in
scripts/prepare_celeb_df.py.
"""

from __future__ import annotations

from pathlib import Path

from src.preprocessing.celeb_df import parse_test_list


def _write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Build a minimal Celeb-DF-shaped folder tree + test list."""
    root = tmp_path / "Celeb-DF-v2"
    (root / "Celeb-real").mkdir(parents=True)
    (root / "YouTube-real").mkdir(parents=True)
    (root / "Celeb-synthesis").mkdir(parents=True)

    # Create empty .mp4 stubs so the parser's existence check passes.
    for rel in [
        "Celeb-real/id0_0000.mp4",
        "YouTube-real/00170.mp4",
        "Celeb-synthesis/id0_id16_0000.mp4",
        "Celeb-synthesis/id2_id5_0001.mp4",
    ]:
        (root / rel).write_bytes(b"")

    list_path = root / "List_of_testing_videos.txt"
    list_path.write_text(
        # Celeb-DF convention: 1=real, 0=fake
        "1 Celeb-real/id0_0000.mp4\n"
        "1 YouTube-real/00170.mp4\n"
        "0 Celeb-synthesis/id0_id16_0000.mp4\n"
        "0 Celeb-synthesis/id2_id5_0001.mp4\n"
    )
    return root, list_path


def test_parser_flips_label_convention(tmp_path: Path) -> None:
    root, list_path = _write_dataset(tmp_path)
    records = parse_test_list(list_path, root)
    assert len(records) == 4
    reals = [r for r in records if r.label == 0]
    fakes = [r for r in records if r.label == 1]
    # Celeb-DF's 2 lines with `1` (real) become our label 0.
    assert len(reals) == 2
    assert len(fakes) == 2


def test_parser_recovers_manipulation_tag(tmp_path: Path) -> None:
    root, list_path = _write_dataset(tmp_path)
    records = parse_test_list(list_path, root)
    manips = {r.manipulation for r in records}
    assert manips == {"Celeb-real", "YouTube-real", "Celeb-synthesis"}


def test_parser_skips_missing_files(tmp_path: Path) -> None:
    root, list_path = _write_dataset(tmp_path)
    # Append a line pointing at a file that doesn't exist.
    with list_path.open("a") as f:
        f.write("0 Celeb-synthesis/DOES_NOT_EXIST.mp4\n")
    records = parse_test_list(list_path, root)
    assert len(records) == 4    # missing file dropped, others kept
