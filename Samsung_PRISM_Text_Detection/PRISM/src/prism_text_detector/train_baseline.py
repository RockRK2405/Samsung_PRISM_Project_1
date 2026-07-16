"""Training pipeline for the PRISM text detector.

Supports two modes:
    1. ``baseline`` — TF-IDF + Stylometry + LogisticRegression (original)
    2. ``enhanced`` — Advanced features + GradientBoosting + Ensemble metadata

Both modes produce a model.pkl and metrics.json.  The enhanced mode also
saves ensemble_meta.json for use with :class:`PrismEnsemble`.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from prism_text_detector.data_hc3 import balanced_sample, load_hc3
from prism_text_detector.features import build_text_baseline


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load & sample data
    # ------------------------------------------------------------------
    print(f"Loading data from: {args.data}")
    data = load_hc3(args.data, min_chars=args.min_chars)
    data = balanced_sample(data, args.sample_per_label, random_state=args.random_state)
    print(f"Total rows after sampling: {len(data)}")
    print(data["label"].value_counts().sort_index().to_string())

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    holdout_size = args.validation_size + args.test_size
    if not 0 < holdout_size < 1:
        raise ValueError("--validation-size + --test-size must be between 0 and 1.")

    train, holdout = train_test_split(
        data,
        test_size=holdout_size,
        random_state=args.random_state,
        stratify=data["label"],
    )
    relative_test_size = args.test_size / holdout_size
    validation, test = train_test_split(
        holdout,
        test_size=relative_test_size,
        random_state=args.random_state,
        stratify=holdout["label"],
    )
    print(f"Split: train={len(train)}, val={len(validation)}, test={len(test)}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    if args.mode == "enhanced":
        model = _build_enhanced_model(args)
        print("Mode: enhanced (advanced features + gradient boosting)")
    else:
        model = build_text_baseline(
            max_features=args.max_features,
            min_df=args.min_df,
            c_value=args.c_value,
        )
        print("Mode: baseline (TF-IDF + logistic regression)")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    start = time.perf_counter()
    model.fit(train["text"], train["label"])
    train_seconds = time.perf_counter() - start
    print(f"Training took {train_seconds:.1f}s")

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------
    threshold = args.threshold
    if args.target_fpr is not None:
        validation_probabilities = model.predict_proba(validation["text"])[:, 1]
        threshold = choose_threshold(
            validation["label"], validation_probabilities, args.target_fpr
        )
        print(f"Tuned threshold: {threshold:.4f} (target FPR: {args.target_fpr})")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    start = time.perf_counter()
    probabilities = model.predict_proba(test["text"])[:, 1]
    latency_seconds = time.perf_counter() - start

    predictions = (probabilities >= threshold).astype(int)
    metrics = evaluate(
        y_true=test["label"],
        y_pred=predictions,
        y_prob=probabilities,
        threshold=threshold,
        target_fpr=args.target_fpr,
        train_seconds=train_seconds,
        latency_seconds=latency_seconds,
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        total_rows=len(data),
        mode=args.mode,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    with (out_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Save ensemble metadata for the enhanced mode
    if args.mode == "enhanced":
        ensemble_meta = {
            "weights": {"stylometric": 0.45, "perplexity": 0.30, "linguistic": 0.25},
            "threshold": threshold,
            "model_info": {
                "version": "2.0.0",
                "mode": "enhanced",
                "features_count": 30,
                "classifier": "HistGradientBoosting + FeatureUnion",
                "train_rows": len(train),
            },
        }
        (out_dir / "ensemble_meta.json").write_text(
            json.dumps(ensemble_meta, indent=2), encoding="utf-8"
        )

    predictions_frame = pd.DataFrame(
        {
            "text": test["text"].values,
            "true_label": test["label"].values,
            "synthetic_probability": probabilities,
            "predicted_label": predictions,
            "source": test["source"].values,
        }
    )
    predictions_frame.to_csv(out_dir / "test_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model and metrics to: {out_dir.resolve()}")


def _build_enhanced_model(args: argparse.Namespace) -> Pipeline:
    """Build the Turnitin-grade pipeline with advanced features + LSA semantic features + gradient boosting."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.compose import ColumnTransformer

    from prism_text_detector.advanced_features import AdvancedFeaturesTransformer

    # LSA Pipeline for word-level semantics
    word_lsa = Pipeline([
        ("tfidf_word", TfidfVectorizer(ngram_range=(1, 3), max_features=10000, analyzer='word', sublinear_tf=True)),
        ("svd_word", TruncatedSVD(n_components=128, random_state=args.random_state))
    ])

    # LSA Pipeline for character-level morphology (catches AI structural quirks)
    char_lsa = Pipeline([
        ("tfidf_char", TfidfVectorizer(ngram_range=(3, 5), max_features=10000, analyzer='char_wb', sublinear_tf=True)),
        ("svd_char", TruncatedSVD(n_components=128, random_state=args.random_state))
    ])

    # Advanced statistical features
    stats_pipeline = Pipeline([
        ("stats_extract", AdvancedFeaturesTransformer()),
        ("scale", StandardScaler())
    ])

    # Combine all feature sets
    combined_features = FeatureUnion([
        ("stats", stats_pipeline),
        ("semantic_words", word_lsa),
        ("morphology_chars", char_lsa)
    ])

    return Pipeline(
        steps=[
            ("features", combined_features),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    max_depth=6,
                    learning_rate=0.1,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=args.random_state,
                ),
            ),
        ]
    )


def choose_threshold(y_true, probabilities, target_fpr: float) -> float:
    """Pick the highest-F1 threshold that keeps validation FPR under target_fpr."""
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()
        fpr = fp / max(1, fp + tn)
        score = f1_score(y_true, predictions, zero_division=0)
        if fpr <= target_fpr and score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold


def evaluate(
    y_true,
    y_pred,
    y_prob,
    threshold: float,
    target_fpr: float | None,
    train_seconds: float,
    latency_seconds: float,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    total_rows: int,
    mode: str = "baseline",
) -> dict[str, float | int | str]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "mode": mode,
        "label_convention": "0=human/real, 1=synthetic/AI-generated",
        "threshold": threshold,
        "target_false_positive_rate": target_fpr,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_synthetic": precision_score(y_true, y_pred, zero_division=0),
        "recall_synthetic": recall_score(y_true, y_pred, zero_division=0),
        "f1_synthetic": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "false_positive_rate_real_flagged_synthetic": fp / max(1, fp + tn),
        "false_negative_rate_synthetic_missed": fn / max(1, fn + tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "train_seconds": train_seconds,
        "test_latency_seconds_total": latency_seconds,
        "test_latency_ms_per_sample": latency_seconds * 1000 / max(1, test_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a cost-aware HC3 text synthetic-data detector."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to HC3 file or directory in JSON, JSONL, or CSV format.",
    )
    parser.add_argument(
        "--out", default="artifacts/hc3_text_baseline",
        help="Output directory for model and metrics.",
    )
    parser.add_argument(
        "--mode", choices=["baseline", "enhanced"], default="baseline",
        help="Training mode: 'baseline' or 'enhanced' (advanced features + GBT).",
    )
    parser.add_argument(
        "--sample-per-label", type=int, default=None,
        help="Optional balanced sample size per class.",
    )
    parser.add_argument(
        "--min-chars", type=int, default=30,
        help="Drop texts shorter than this many characters.",
    )
    parser.add_argument(
        "--validation-size", type=float, default=0.15,
        help="Validation fraction for threshold selection.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15,
        help="Held-out test fraction.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Synthetic probability threshold when --target-fpr is omitted.",
    )
    parser.add_argument(
        "--target-fpr", type=float, default=None,
        help="Tune threshold on validation data to keep false positives below this rate.",
    )
    parser.add_argument(
        "--max-features", type=int, default=50000,
        help="Max TF-IDF features per vectorizer.",
    )
    parser.add_argument(
        "--min-df", type=int, default=2,
        help="Minimum document frequency for TF-IDF terms.",
    )
    parser.add_argument(
        "--c-value", type=float, default=4.0,
        help="Logistic regression C value (baseline mode).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
