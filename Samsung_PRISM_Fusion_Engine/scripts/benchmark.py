"""Batch benchmark for the fusion engine.

Runs the full 4-modality pipeline over a directory of labeled video clips
and prints aggregate metrics (accuracy, FPR, per-modality accuracy,
cross-modal flag rate). Auto-derives audio + a frame + a Whisper
transcript from each video so one command tests every modality at once.

Manifest format (JSONL, one object per line):

    {"video": "test_videos/real.mp4",              "label": "real"}
    {"video": "test_videos/morgan_freeman_fake.mp4", "label": "synthetic"}

Example
-------
::

    python scripts/benchmark.py --manifest bench.jsonl --out benchmark.json

Whisper is optional -- pass ``--skip-text`` to skip transcription and
run 3-of-4 modalities per clip (much faster).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from fusion_engine import FusionEngine
from fusion_engine.schema import FusionResult


def _extract_audio(video: Path, out_dir: Path) -> Path:
    dst = out_dir / f"{video.stem}.wav"
    if dst.exists():
        return dst
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", str(video),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(dst),
        ],
        check=True,
    )
    return dst


def _extract_frame(video: Path, out_dir: Path, frame_index: int = 50) -> Path:
    dst = out_dir / f"{video.stem}.jpg"
    if dst.exists():
        return dst
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", str(video),
            "-vf", f"select=eq(n\\,{frame_index})",
            "-vframes", "1", "-update", "1",
            str(dst),
        ],
        check=True,
    )
    return dst


def _transcribe(audio: Path, out_dir: Path, whisper_model: str) -> str:
    txt = out_dir / f"{audio.stem}.txt"
    if not txt.exists():
        subprocess.run(
            [
                "whisper", str(audio),
                "--model", whisper_model,
                "--output_format", "txt",
                "--output_dir", str(out_dir),
                "--fp16", "False",
            ],
            check=True,
            capture_output=True,
        )
    return txt.read_text().strip()


def _label_to_prob(label: str) -> float:
    normalised = label.strip().lower()
    if normalised in {"real", "authentic", "0"}:
        return 0.0
    if normalised in {"synthetic", "fake", "deepfake", "1"}:
        return 1.0
    raise ValueError(f"unknown label: {label!r}")


def _row(result: FusionResult, truth: float) -> dict:
    row = {
        "verdict": result.verdict,
        "fused_score": result.fused_score,
        "fused_confidence": result.fused_confidence,
        "truth_synthetic": truth,
        "cross_modal_flags_count": len(result.cross_modal_flags),
        "per_modality": {},
    }
    for name, mr in result.modality_results.items():
        row["per_modality"][name] = {
            "available": mr.available,
            "prob_synthetic": mr.prob_synthetic if mr.available else None,
            "confidence": mr.confidence if mr.available else None,
            "latency_ms": mr.latency_ms,
        }
    return row


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.derived_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(entries)} entries from {manifest_path}")

    engine = FusionEngine()
    if engine.load_errors:
        print("Adapter load warnings:")
        for modality, err in engine.load_errors.items():
            print(f"  {modality}: {err}")

    per_clip: list[dict] = []
    start_total = time.perf_counter()
    for i, entry in enumerate(entries, 1):
        video = (manifest_path.parent / entry["video"]).resolve()
        truth = _label_to_prob(entry["label"])
        if not video.exists():
            print(f"[{i}/{len(entries)}] SKIP {video} — file not found")
            continue

        print(f"[{i}/{len(entries)}] {video.name} (truth={entry['label']})")
        audio = _extract_audio(video, out_dir)
        image = _extract_frame(video, out_dir)
        text = None
        if not args.skip_text:
            text = _transcribe(audio, out_dir, args.whisper_model)

        result = engine.analyze(
            image_path=image,
            audio_path=audio,
            text=text,
            video_path=video,
        )
        row = _row(result, truth)
        row["clip"] = video.name
        per_clip.append(row)
        print(
            f"    -> verdict={result.verdict:9s} "
            f"score={result.fused_score:.3f} conf={result.fused_confidence:.3f} "
            f"flags={len(result.cross_modal_flags)}"
        )

    total_seconds = time.perf_counter() - start_total

    aggregate = _aggregate(per_clip, engine.decision_threshold)
    report = {
        "manifest": str(manifest_path),
        "num_clips_seen": len(per_clip),
        "total_seconds": total_seconds,
        "aggregate": aggregate,
        "per_clip": per_clip,
    }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print()
    print("--- Aggregate ---")
    print(json.dumps(aggregate, indent=2))
    print(f"\nWrote full report to {args.out}")


def _aggregate(rows: list[dict], threshold: float) -> dict:
    if not rows:
        return {"num_clips": 0}

    n = len(rows)
    fused_correct = sum(
        1 for r in rows if (r["fused_score"] >= threshold) == (r["truth_synthetic"] >= 0.5)
    )
    real_rows = [r for r in rows if r["truth_synthetic"] < 0.5]
    fake_rows = [r for r in rows if r["truth_synthetic"] >= 0.5]
    fp = sum(1 for r in real_rows if r["fused_score"] >= threshold)
    fn = sum(1 for r in fake_rows if r["fused_score"] < threshold)

    per_modality: dict[str, dict] = {}
    for modality in ("image", "audio", "text", "video"):
        avail = [r for r in rows if r["per_modality"].get(modality, {}).get("available")]
        if not avail:
            per_modality[modality] = {"available_on": 0}
            continue
        correct = sum(
            1
            for r in avail
            if (r["per_modality"][modality]["prob_synthetic"] >= threshold)
            == (r["truth_synthetic"] >= 0.5)
        )
        per_modality[modality] = {
            "available_on": len(avail),
            "accuracy": correct / len(avail),
            "mean_confidence": mean(r["per_modality"][modality]["confidence"] for r in avail),
            "mean_latency_ms": mean(r["per_modality"][modality]["latency_ms"] for r in avail),
        }

    return {
        "num_clips": n,
        "num_real": len(real_rows),
        "num_synthetic": len(fake_rows),
        "fused_accuracy": fused_correct / n,
        "false_positive_rate": (fp / len(real_rows)) if real_rows else None,
        "false_negative_rate": (fn / len(fake_rows)) if fake_rows else None,
        "flag_rate": mean(r["cross_modal_flags_count"] > 0 for r in rows),
        "per_modality": per_modality,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch fusion-engine benchmark.")
    parser.add_argument("--manifest", required=True, help="JSONL file with video paths + labels.")
    parser.add_argument("--out", default="benchmark.json", help="Output JSON report path.")
    parser.add_argument("--derived-dir", default="benchmark_derived", help="Directory for cached audio/frame/transcript per clip.")
    parser.add_argument("--skip-text", action="store_true", help="Skip Whisper transcription (runs 3 of 4 modalities).")
    parser.add_argument("--whisper-model", default="tiny", help="Whisper model size (tiny is fastest).")
    return parser.parse_args()


if __name__ == "__main__":
    main()
