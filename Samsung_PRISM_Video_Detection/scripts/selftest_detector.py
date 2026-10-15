#!/usr/bin/env python3
"""Is the model ACTUALLY running, or is the dashboard serving mock output?

Run this FIRST, before any accuracy test. It takes ~1 minute and answers the
only question that matters up front: are the numbers you see coming from a
neural network, or from `sha256(filename)`?

Background: when no checkpoint can be loaded, the dashboard falls back to a
deliberately-marked mock whose score is a hash of the file's PATH. The mock is
honest (`"mock": true`), but if you are reading scores off the UI it is easy to
miss. This script detects it three independent ways, so it still catches the
case where the mock flag is lost somewhere in the stack.

    python scripts/selftest_detector.py --video path/to/any.mp4

The decisive check is RENAME INVARIANCE: the same bytes, under three different
filenames. A real model returns the same score three times. A path-hash mock
returns three unrelated scores.

Exit codes:  0 = real model confirmed   1 = mock detected   2 = inconclusive
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODULE_ROOT = _HERE.parent                      # Samsung_PRISM_Video_Detection/
_DASHBOARD = _MODULE_ROOT.parent / "Samsung_PRISM_Video_Dashboard"

# Scores closer than this are "the same number" (float/GPU jitter, not signal).
_SAME = 1e-4


def _hr(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


# ---------------------------------------------------------------- checkpoints
def check_checkpoints() -> dict:
    """Report which checkpoint files exist on disk, and how big they are."""
    _hr("1. CHECKPOINTS ON DISK")
    ckpt_dir = _MODULE_ROOT / "checkpoints"
    found = {}
    if not ckpt_dir.is_dir():
        print(f"  MISSING directory: {ckpt_dir}")
        return found

    pts = sorted(p for p in ckpt_dir.glob("*.pt"))
    if not pts:
        print(f"  No .pt files in {ckpt_dir}")
        print("  -> The dashboard CANNOT load a real model. It will serve MOCK output.")
        print("     Fix: put a trained checkpoint here (see docs/TESTING_GUIDE.md).")
        return found

    for p in pts:
        mb = p.stat().st_size / 1e6
        found[p.name] = mb
        print(f"  {p.name:28s} {mb:8.1f} MB")

    # The two names the dashboard config actually looks for.
    for want, role in (("best.pt", "face branch"), ("general.pt", "general branch")):
        state = "present" if (ckpt_dir / want).is_file() else "ABSENT"
        print(f"  config expects {want:14s} ({role:15s}) -> {state}")
    return found


# ------------------------------------------------------------------ predictor
def load_predictor(mode: str):
    """Return a `(name, fn)` where fn(path) -> dict with a score and mock flag.

    `mode="dashboard"` tests the full deployed stack (what the UI shows).
    `mode="direct"` bypasses the dashboard and calls the detector classes.
    """
    if mode == "dashboard":
        if str(_DASHBOARD) not in sys.path:
            sys.path.insert(0, str(_DASHBOARD))
        from backend.services import model_service  # type: ignore

        status = model_service.model_status()
        print(json.dumps(status, indent=2, default=str))

        def fn(path: Path) -> dict:
            r = model_service.predict_video(path)
            return {
                "score": r.get("synthetic_probability"),
                "mock": r.get("mock"),
                "prediction": r.get("prediction"),
                "meta": r.get("meta", {}),
                "frames": r.get("num_frames_used"),
            }

        return "dashboard/model_service.predict_video", fn, status

    # direct
    if str(_MODULE_ROOT) not in sys.path:
        sys.path.insert(0, str(_MODULE_ROOT))
    ckpt_dir = _MODULE_ROOT / "checkpoints"
    general, face = ckpt_dir / "general.pt", ckpt_dir / "best.pt"

    if general.is_file() and face.is_file():
        from src.inference import MultiTargetVideoDetector  # type: ignore

        det = MultiTargetVideoDetector(face_checkpoint=face, general_checkpoint=general)
        name = "MultiTargetVideoDetector"
    elif face.is_file():
        from src.inference import VideoDetector  # type: ignore

        det = VideoDetector(checkpoint_path=face)
        name = "VideoDetector (face only)"
    else:
        raise FileNotFoundError(
            "Need checkpoints/best.pt (face) or both best.pt+general.pt. "
            "General-only is blocked by a known defect -- see §7.4 of the work log."
        )

    def fn(path: Path) -> dict:
        r = det.predict(path)
        return {
            "score": float(r.prob_synthetic),
            "mock": False,
            "prediction": r.prediction,
            "meta": dict(r.meta or {}),
            "frames": r.num_frames_used,
        }

    return name, fn, {"mock": False, "note": "direct import, no dashboard"}


# ----------------------------------------------------------------- the checks
def check_rename_invariance(fn, video: Path, n: int = 3) -> tuple[bool | None, list]:
    """THE decisive check. Same bytes, different filenames, compare scores."""
    _hr("3. RENAME-INVARIANCE  (the decisive mock test)")
    print("  Copying the SAME video to differently-named files and scoring each.")
    print("  Real model -> identical scores.  Path-hash mock -> unrelated scores.\n")

    results = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(n):
            # Deliberately very different names -> very different hashes.
            dest = Path(td) / f"{'z' * (i + 1)}_rename_probe_{i}{video.suffix}"
            shutil.copy2(video, dest)
            try:
                out = fn(dest)
            except Exception as exc:  # noqa: BLE001 - report, don't crash the suite
                print(f"  {dest.name:34s} ERROR {type(exc).__name__}: {exc}")
                results.append(None)
                continue
            results.append(out["score"])
            print(f"  {dest.name:34s} score={out['score']}  mock={out['mock']}")

    got = [r for r in results if r is not None]
    if len(got) < 2:
        print("\n  INCONCLUSIVE - fewer than two scores produced.")
        return None, results

    spread = max(got) - min(got)
    print(f"\n  spread = {spread:.6f}")
    if spread <= _SAME:
        print("  VERDICT: score is independent of filename -> consistent with a REAL model.")
        return True, results
    print("  VERDICT: score CHANGES with the filename -> this is the MOCK.")
    print("           Nothing is being classified. Every number you have seen is a hash.")
    return False, results


def check_determinism(fn, video: Path) -> bool | None:
    """Same file, same name, twice. Catches nondeterminism (dropout left on)."""
    _hr("4. DETERMINISM  (same file scored twice)")
    try:
        a, b = fn(video)["score"], fn(video)["score"]
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR {type(exc).__name__}: {exc}")
        return None
    print(f"  run1={a}  run2={b}  delta={abs(a - b):.6f}")
    if abs(a - b) <= _SAME:
        print("  OK - deterministic (model is in eval mode).")
        return True
    print("  WARNING - nondeterministic. Check for dropout/augmentation at inference.")
    return False


def check_branches(fn, video: Path) -> None:
    """Show which branch actually ran -- the thing the UI hides."""
    _hr("5. WHICH BRANCH RAN")
    try:
        out = fn(video)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR {type(exc).__name__}: {exc}")
        return
    meta = out["meta"]
    print(f"  prediction      : {out['prediction']}")
    print(f"  score           : {out['score']}")
    print(f"  frames used     : {out['frames']}")
    for k in ("router", "path_used", "face_coverage", "face_score", "general_score",
              "model_class", "backbone", "checkpoint", "frames_with_no_face", "note"):
        if k in meta:
            print(f"  {k:16s}: {meta[k]}")
    if not meta:
        print("  (no meta returned)")


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Any video the detector can read.")
    ap.add_argument("--mode", choices=["dashboard", "direct"], default="dashboard",
                    help="dashboard = full deployed stack (default); direct = detector classes.")
    ap.add_argument("--copies", type=int, default=3, help="Rename probes (default 3).")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        print(f"ERROR: no such file: {video}")
        return 2

    print(f"video : {video}  ({video.stat().st_size / 1e6:.1f} MB)")
    print(f"mode  : {args.mode}")

    ckpts = check_checkpoints()

    _hr("2. MODEL STATUS AS THE APP REPORTS IT")
    try:
        name, fn, status = load_predictor(args.mode)
        print(f"\n  entry point: {name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED to load a predictor: {type(exc).__name__}: {exc}")
        print("\n  -> Cannot test the model. Resolve the load error first.")
        return 2

    if status.get("mock") is True:
        print("\n  *** model_status() reports mock=True ***")
        print(f"      load_error: {status.get('load_error')}")

    real, _ = check_rename_invariance(fn, video, args.copies)
    check_determinism(fn, video)
    check_branches(fn, video)

    _hr("VERDICT")
    if real is False or status.get("mock") is True:
        print("  MOCK OUTPUT. The model is not running.")
        print("  Next: get a checkpoint in place, then re-run this script.")
        print("  If a checkpoint IS present and this still fails, you are hitting the")
        print("  serving defects in docs/DETECTION_ENABLEMENT_PLAN.md (Phase 0).")
        return 1
    if real is True:
        print("  REAL MODEL CONFIRMED. Scores do not depend on the filename.")
        print(f"  Checkpoints loaded: {', '.join(ckpts) or 'n/a'}")
        print("\n  You may now run an accuracy test:")
        print("    python scripts/score_manifest.py --manifest <labeled.csv>")
        return 0
    print("  INCONCLUSIVE. See errors above.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
