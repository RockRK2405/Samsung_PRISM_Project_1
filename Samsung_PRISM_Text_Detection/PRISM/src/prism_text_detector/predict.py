"""Updated predict module with ensemble support.

Supports both the original baseline model and the new multi-signal ensemble.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def main() -> None:
    args = parse_args()

    if args.ensemble:
        _run_ensemble(args)
    else:
        _run_baseline(args)


def _run_baseline(args: argparse.Namespace) -> None:
    """Original baseline prediction with a pickled sklearn pipeline."""
    with Path(args.model).open("rb") as handle:
        model = pickle.load(handle)

    text = _get_text(args)

    probability = float(model.predict_proba([text])[:, 1][0])
    payload = {
        "label_convention": "0=human/real, 1=synthetic/AI-generated",
        "synthetic_probability": probability,
        "predicted_label": int(probability >= args.threshold),
        "verdict": "synthetic" if probability >= args.threshold else "human",
        "threshold": args.threshold,
    }
    print(json.dumps(payload, indent=2))


def _run_ensemble(args: argparse.Namespace) -> None:
    """Multi-signal ensemble prediction with explainability."""
    from prism_text_detector.ensemble import PrismEnsemble

    ensemble = PrismEnsemble.load(args.model_dir)
    text = _get_text(args)
    result = ensemble.analyze(text)

    if args.brief:
        brief = {
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "authenticity_score": result["authenticity_score"],
            "synthetic_probability": result["synthetic_probability"],
        }
        print(json.dumps(brief, indent=2))
    else:
        print(json.dumps(result, indent=2, default=str))


def _get_text(args: argparse.Namespace) -> str:
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        raise ValueError("Provide --text or --file.")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained PRISM text detector."
    )
    parser.add_argument("--text", default=None, help="Text string to analyse.")
    parser.add_argument("--file", default=None, help="File containing text to analyse.")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Synthetic probability threshold (baseline mode).",
    )

    # Baseline mode
    parser.add_argument(
        "--model", default="artifacts/hc3_text_baseline/model.pkl",
        help="Path to a pickled sklearn model (baseline mode).",
    )

    # Ensemble mode
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Use the multi-signal ensemble instead of the baseline.",
    )
    parser.add_argument(
        "--model-dir", default="artifacts/enhanced",
        help="Directory containing ensemble artifacts (ensemble mode).",
    )
    parser.add_argument(
        "--brief", action="store_true",
        help="Print only verdict, confidence, and scores (ensemble mode).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
