from __future__ import annotations

import re
import string

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler


WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_RE = re.compile(r"[.!?]+")


class TextStatsTransformer(BaseEstimator, TransformerMixin):
    """Small stylometry feature block for cheap, explainable text signals."""

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        return np.asarray([_stats(str(text)) for text in texts], dtype=float)


def build_text_baseline(max_features: int = 50000, min_df: int = 2, c_value: float = 4.0) -> Pipeline:
    features = FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    lowercase=True,
                    min_df=min_df,
                    max_features=max_features,
                    strip_accents="unicode",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    lowercase=True,
                    min_df=min_df,
                    max_features=max_features,
                ),
            ),
            (
                "stylometry",
                Pipeline(
                    steps=[
                        ("stats", TextStatsTransformer()),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def _stats(text: str) -> list[float]:
    chars = len(text)
    words = WORD_RE.findall(text)
    word_count = len(words)
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    sentence_count = max(1, len(sentences))
    word_lengths = [len(word) for word in words] or [0]
    sentence_lengths = [len(WORD_RE.findall(sentence)) for sentence in sentences] or [word_count]
    unique_words = len({word.lower() for word in words})
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    uppercase_count = sum(1 for char in text if char.isupper())
    digit_count = sum(1 for char in text if char.isdigit())
    newline_count = text.count("\n")

    repeated_pairs = 0
    lowered = [word.lower() for word in words]
    for first, second in zip(lowered, lowered[1:]):
        if first == second:
            repeated_pairs += 1

    return [
        chars,
        word_count,
        sentence_count,
        chars / max(1, word_count),
        float(np.mean(word_lengths)),
        float(np.std(word_lengths)),
        float(np.mean(sentence_lengths)),
        float(np.std(sentence_lengths)),
        unique_words / max(1, word_count),
        punctuation_count / max(1, chars),
        uppercase_count / max(1, chars),
        digit_count / max(1, chars),
        newline_count,
        repeated_pairs / max(1, word_count),
    ]

