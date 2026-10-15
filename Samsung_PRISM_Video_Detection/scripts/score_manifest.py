#!/usr/bin/env python3
"""Score a labeled test set and print the full metric suite.

Takes a CSV and emits exactly the metrics the project reports everywhere else
(accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR, confusion), so your
numbers are directly comparable to EXP-004 (AUC 0.9426) and EXP-005 (0.6564).

Manifest format -- header required, `group` optional:

    path,label,group
    /data/t2vz/t2vz-001.mp4,1,text2video-zero
    /data/vript/vript-001.mp4,0,vript

    label: 1 = synthetic/fake, 0 = real.
    group: free-text bucket (generator, dataset, source). Reported separately,
           which is how you see "unseen generator recall" rather than one
           number that hides it.

    python scripts/score_manifest.py --manifest test.csv --out reports/my_test.json

ROC-AUC and PR-AUC are computed in pure Python (no sklearn needed) so this runs
in any environment that can load the model at all.

NOTE: run scripts/selftest_detector.py first. If the detector is serving mock
output, every metric below is meaningless -- this script refuses to proceed
when it detects that, unless you pass --allow-mock.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MODULE_ROOT = _HERE.parent
_DASHBOARD = _MODULE_ROOT.parent / "Samsung_PRISM_Video_Dashboard"


# ------------------------------------------------------------------- metrics
def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    """Rank-based ROC-AUC (Mann-Whitney U) with correct tie handling."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None  # undefined with a single class present

    # Average ranks within tied score groups.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0           # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1

    rank_sum = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def pr_auc(labels: list[int], scores: list[float]) -> float | None:
    """Average precision — the step-wise PR-AUC, same convention as sklearn."""
    pos = sum(labels)
    if pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def suite(labels: list[int], scores: list[float], threshold: float) -> dict:
    """Threshold-dependent metrics + the two threshold-free AUCs."""
    tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= threshold)
    fn = sum(1 for l, s in zip(labels, scores) if l == 1 and s < threshold)
    fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= threshold)
    tn = sum(1 for l, s in zip(labels, scores) if l == 0 and s < threshold)

    def div(a: int, b: int) -> float | None:
        return a / b if b else None

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else None)
    return {
        "n": len(labels), "n_real": labels.count(0), "n_fake": labels.count(1),
        "threshold": threshold,
        "accuracy": div(tp + tn, len(labels)),
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc(labels, scores), "pr_auc": pr_auc(labels, scores),
        "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def threshold_sweep(labels: list[int], scores: list[float]) -> list[dict]:
    """Accuracy/FPR/recall across thresholds — shows if a better cut exists."""
    rows = []
    for t in [i / 20 for i in range(1, 20)]:
        m = suite(labels, scores, t)
        rows.append({"threshold": round(t, 2), "accuracy": m["accuracy"],
                     "recall": m["recall"], "fpr": m["fpr"], "f1": m["f1"]})
    return rows


def _fmt(v: object, nd: int = 4) -> str:
    return f"{v:.{nd}f}" if isinstance(v, float) else ("n/a" if v is None else str(v))


def _print_suite(title: str, m: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"  n = {m['n']}  ({m['n_real']} real / {m['n_fake']} fake)   "
          f"threshold = {m['threshold']}")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr"):
        star = " *" if k in ("roc_auc", "f1") else ""
        print(f"  {k:10s} {_fmt(m[k])}{star}")
    c = m["confusion"]
    print(f"  confusion   TP={c['tp']}  FP={c['fp']}  TN={c['tn']}  FN={c['fn']}")


# ----------------------------------------------------------------- predictor
def load_predictor(mode: str):
    if mode == "dashboard":
        if str(_DASHBOARD) not in sys.path:
            sys.path.insert(0, str(_DASHBOARD))
        from backend.services import model_service  # type: ignore
        status = model_service.model_status()

        def fn(p: Path) -> tuple[float, bool]:
            r = model_service.predict_video(p)
            return float(r["synthetic_probability"]), bool(r.get("mock"))

        return fn, status

    if str(_MODULE_ROOT) not in sys.path:
        sys.path.insert(0, str(_MODULE_ROOT))
    ckpt = _MODULE_ROOT / "checkpoints"
    if (ckpt / "general.pt").is_file() and (ckpt / "best.pt").is_file():
        from src.inference import MultiTargetVideoDetector  # type: ignore
        det = MultiTargetVideoDetector(face_checkpoint=ckpt / "best.pt",
                                       general_checkpoint=ckpt / "general.pt")
    else:
        from src.inference import VideoDetector  # type: ignore
        det = VideoDetector(checkpoint_path=ckpt / "best.pt")

    def fn(p: Path) -> tuple[float, bool]:
        return float(det.predict(p).prob_synthetic), False

    return fn, {"mock": False}


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="CSV: path,label[,group]")
    ap.add_argument("--out", default="reports/test_results.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mode", choices=["dashboard", "direct"], default="dashboard")
    ap.add_argument("--limit", type=int, default=0, help="Score only the first N rows.")
    ap.add_argument("--allow-mock", action="store_true",
                    help="Proceed even if the detector is serving mock output.")
    args = ap.parse_args()

    rows = []
    with Path(args.manifest).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("path"):
                continue
            rows.append({"path": r["path"], "label": int(r["label"]),
                         "group": (r.get("group") or "all").strip()})
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("ERROR: manifest produced zero usable rows.")
        return 2
    print(f"manifest : {args.manifest}  ({len(rows)} rows)")

    fn, status = load_predictor(args.mode)
    if status.get("mock") and not args.allow_mock:
        print("\nREFUSING TO SCORE: the detector is serving MOCK output.")
        print(f"  load_error: {status.get('load_error')}")
        print("  Every metric would be a hash of your filenames, not model behaviour.")
        print("  Run scripts/selftest_detector.py, fix the load, then retry.")
        print("  (Override with --allow-mock only to test this script's own plumbing.)")
        return 1

    labels: list[int] = []
    scores: list[float] = []
    per_group: dict[str, list[tuple[int, float]]] = defaultdict(list)
    failures: list[dict] = []
    saw_mock = False
    t0 = time.perf_counter()

    for i, r in enumerate(rows, 1):
        p = Path(r["path"]).expanduser()
        if not p.is_file():
            failures.append({"path": str(p), "error": "FileNotFound"})
            print(f"  [{i}/{len(rows)}] MISSING {p}")
            continue
        try:
            score, mock = fn(p)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            failures.append({"path": str(p), "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [{i}/{len(rows)}] FAILED  {p.name}: {type(exc).__name__}: {exc}")
            continue
        saw_mock = saw_mock or mock
        labels.append(r["label"])
        scores.append(score)
        per_group[r["group"]].append((r["label"], score))
        print(f"  [{i}/{len(rows)}] {p.name[:44]:44s} label={r['label']} score={score:.4f}")

    elapsed = time.perf_counter() - t0
    if not labels:
        print("\nERROR: nothing scored successfully.")
        return 2

    print(f"\nscored {len(labels)}/{len(rows)} in {elapsed:.1f}s "
          f"({elapsed / max(len(labels), 1):.2f}s per item)")

    overall = suite(labels, scores, args.threshold)
    _print_suite(f"OVERALL @ threshold {args.threshold}", overall)

    groups = {}
    if len(per_group) > 1:
        print("\n--- PER GROUP ---")
        for g, pairs in sorted(per_group.items()):
            gl = [l for l, _ in pairs]
            gs = [s for _, s in pairs]
            gm = suite(gl, gs, args.threshold)
            groups[g] = gm
            # A single-class group has no AUC; report the rate that is defined.
            if gm["n_fake"] and not gm["n_real"]:
                rate, lbl = gm["recall"], "recall"
            elif gm["n_real"] and not gm["n_fake"]:
                rate = 1.0 - gm["fpr"] if gm["fpr"] is not None else None
                lbl = "specificity"
            else:
                rate, lbl = gm["roc_auc"], "roc_auc"
            mean_s = sum(gs) / len(gs)
            print(f"  {g:24s} n={gm['n']:5d}  {lbl:11s}={_fmt(rate)}  "
                  f"mean_score={mean_s:.4f}")

    print("\n--- THRESHOLD SWEEP (overall) ---")
    print(f"  {'thr':>5s} {'acc':>7s} {'recall':>7s} {'fpr':>7s} {'f1':>7s}")
    for row in threshold_sweep(labels, scores):
        print(f"  {row['threshold']:5.2f} {_fmt(row['accuracy']):>7s} "
              f"{_fmt(row['recall']):>7s} {_fmt(row['fpr']):>7s} {_fmt(row['f1']):>7s}")

    payload = {
        "manifest": str(args.manifest),
        "mode": args.mode,
        "mock_output": saw_mock,
        "n_requested": len(rows),
        "n_scored": len(labels),
        "elapsed_sec": round(elapsed, 2),
        "overall": overall,
        "per_group": groups,
        "threshold_sweep": threshold_sweep(labels, scores),
        "failures": failures,
        "scores": [{"path": r["path"], "label": r["label"], "group": r["group"]}
                   for r in rows if Path(r["path"]).expanduser().is_file()][: len(labels)],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if saw_mock:
        print("\n*** WARNING: mock output was served. These metrics are NOT real. ***")
        return 1
    if failures:
        print(f"\nNote: {len(failures)} item(s) failed — see 'failures' in the JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
