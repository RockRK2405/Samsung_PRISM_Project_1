"""Unit tests for the Celeb-DF v2 test-list parser.

Only the parser is exercised — full extraction touches OpenCV and
MTCNN and is covered by the manual smoke test in
scripts/prepare_celeb_df.py.
"""

from __future__ import annotations

from pathlib import Path

from src.preprocessing.celeb_df import parse_test_list, scan_celeb_df_training, _identity_of


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


def test_identity_of_maps_families_to_shared_key() -> None:
    """A Celeb-real and Celeb-synthesis video sharing source id share identity."""
    assert _identity_of("Celeb-real/id0_0000.mp4") == "celeb_id0"
    assert _identity_of("Celeb-synthesis/id0_id16_0000.mp4") == "celeb_id0"
    assert _identity_of("Celeb-synthesis/id0_id5_9999.mp4") == "celeb_id0"
    # YouTube-real videos are their own subjects.
    assert _identity_of("YouTube-real/00170.mp4") == "yt_00170"
    assert _identity_of("YouTube-real/00171.mp4") == "yt_00171"


def test_scan_training_excludes_test_identities(tmp_path: Path) -> None:
    """No identity in the test list can appear in train or val."""
    root, list_path = _write_dataset(tmp_path)
    # Add extra videos with the SAME identities as the test list — those must be excluded.
    (root / "Celeb-real" / "id0_9999.mp4").write_bytes(b"")            # shares id0 with test
    (root / "Celeb-synthesis" / "id0_id99_9999.mp4").write_bytes(b"")   # shares id0 too
    (root / "Celeb-real" / "id100_0000.mp4").write_bytes(b"")           # new identity, safe
    (root / "Celeb-synthesis" / "id100_id5_0000.mp4").write_bytes(b"")  # new identity, safe

    splits = scan_celeb_df_training(root, list_path, val_ratio=0.5, seed=0)
    all_train_val_paths = {r.video_path for r in splits["train"] + splits["val"]}
    for p in all_train_val_paths:
        assert "id0_" not in p.name.split("_id")[0] + "_"   # no id0-prefixed videos leaked
    # And no id0_ sourced synthesis either — split by _
    for p in all_train_val_paths:
        assert not p.name.startswith("id0_")


def test_scan_training_identity_disjoint_between_splits(tmp_path: Path) -> None:
    root = tmp_path / "Celeb-DF-v2"
    (root / "Celeb-real").mkdir(parents=True)
    (root / "Celeb-synthesis").mkdir(parents=True)
    (root / "YouTube-real").mkdir(parents=True)
    for i in range(20):
        (root / "Celeb-real" / f"id{i}_0000.mp4").write_bytes(b"")
        (root / "Celeb-synthesis" / f"id{i}_id{(i + 3) % 20}_0000.mp4").write_bytes(b"")
    # Empty test list — nothing excluded.
    list_path = root / "List_of_testing_videos.txt"
    list_path.write_text("")

    splits = scan_celeb_df_training(root, list_path, val_ratio=0.3, seed=0)

    def ids(records: list) -> set:
        return {_identity_of(f"{r.manipulation}/{r.video_path.name}") for r in records}

    assert ids(splits["train"]).isdisjoint(ids(splits["val"]))


def test_parser_skips_missing_files(tmp_path: Path) -> None:
    root, list_path = _write_dataset(tmp_path)
    # Append a line pointing at a file that doesn't exist.
    with list_path.open("a") as f:
        f.write("0 Celeb-synthesis/DOES_NOT_EXIST.mp4\n")
    records = parse_test_list(list_path, root)
    assert len(records) == 4    # missing file dropped, others kept
