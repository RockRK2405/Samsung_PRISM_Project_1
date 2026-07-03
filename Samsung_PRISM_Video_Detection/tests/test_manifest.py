"""Unit tests for the FF++ manifest scanner and group split.

The scanner is exercised against a tiny synthetic filesystem so we
don't depend on the real 17 GB dataset being present.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.preprocessing.manifest import (
    FF_MANIPULATIONS,
    FF_REAL_FOLDER,
    group_split,
    scan_ff_c23,
    write_manifest,
)


def _make_fake_ff(tmp_path: Path, num_ids: int = 10) -> Path:
    """Create a stub FF++ tree with ``num_ids`` real videos and matching fakes."""
    root = tmp_path / "FaceForensics++_C23"
    (root / FF_REAL_FOLDER).mkdir(parents=True)
    for i in range(num_ids):
        (root / FF_REAL_FOLDER / f"{i:03d}.mp4").write_bytes(b"")
    for manip in FF_MANIPULATIONS:
        (root / manip).mkdir(parents=True)
        for i in range(num_ids):
            (root / manip / f"{i:03d}_{(i + 7) % 1000:03d}.mp4").write_bytes(b"")
    return root


def test_scan_returns_expected_record_count(tmp_path: Path) -> None:
    root = _make_fake_ff(tmp_path, num_ids=10)
    records = scan_ff_c23(root)
    # 10 real + 4 * 10 fakes = 50
    assert len(records) == 50
    assert sum(1 for r in records if r.label == "real") == 10
    assert sum(1 for r in records if r.label == "synthetic") == 40


def test_group_id_maps_fakes_to_source_real(tmp_path: Path) -> None:
    root = _make_fake_ff(tmp_path, num_ids=3)
    records = scan_ff_c23(root)
    # Every fake named 001_XXX must have group_id == "001"
    fake_001 = [r for r in records if r.label == "synthetic" and r.video_id.startswith("001_")]
    assert fake_001, "expected fakes with source id 001"
    assert all(r.group_id == "001" for r in fake_001)


def test_group_split_has_no_group_leakage(tmp_path: Path) -> None:
    root = _make_fake_ff(tmp_path, num_ids=20)
    records = scan_ff_c23(root)
    splits = group_split(records, ratios=(0.8, 0.1, 0.1), seed=42)

    train_groups = {r.group_id for r in splits["train"]}
    val_groups = {r.group_id for r in splits["val"]}
    test_groups = {r.group_id for r in splits["test"]}
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_write_manifest_roundtrip(tmp_path: Path) -> None:
    root = _make_fake_ff(tmp_path, num_ids=5)
    records = scan_ff_c23(root)
    splits = group_split(records, ratios=(0.6, 0.2, 0.2), seed=0)
    out = tmp_path / "manifest.json"
    write_manifest(splits, out)

    with out.open() as f:
        payload = json.load(f)
    assert set(payload["splits"]) == {"train", "val", "test"}
    assert payload["meta"]["num_train"] == len(splits["train"])
