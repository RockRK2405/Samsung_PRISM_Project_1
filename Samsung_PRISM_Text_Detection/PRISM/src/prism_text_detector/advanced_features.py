from __future__ import annotations

import math
import re
import string
import zlib
from collections import Counter
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES: list[str] = [
    "burstiness_coefficient",
    "hapax_legomena_ratio",
    "mattr_50",
    "yules_k",
    "flesch_kincaid_grade",
    "coleman_liau_index",
    "automated_readability_index",
    "function_word_frequency",
    "discourse_marker_density",
    "contraction_ratio",
    "question_frequency",
    "first_person_pronoun_ratio",
    "passive_voice_estimate",
    "sentence_start_diversity",
    "comma_to_period_ratio",
    "avg_word_frequency_rank",
    "paragraph_length_variance",
    "conjunction_starting_ratio",
    "compression_ratio",
    "mean_word_length",
    "std_word_length",
    "mean_sentence_length",
    "std_sentence_length",
    "lexical_diversity",
    "punctuation_ratio",
    "uppercase_ratio",
    "digit_ratio",
    "exclamation_ratio",
    "semicolon_colon_ratio",
    "avg_words_per_paragraph",
]

# Simple frequency list of top words (approximate ranks for demonstration)
# In a real scenario, this would be a loaded lexicon of 5000+ words.
_TOP_WORDS = {
    "the": 1, "be": 2, "to": 3, "of": 4, "and": 5, "a": 6, "in": 7, "that": 8, "have": 9, "i": 10,
    "it": 11, "for": 12, "not": 13, "on": 14, "with": 15, "he": 16, "as": 17, "you": 18, "do": 19, "at": 20,
    "this": 21, "but": 22, "his": 23, "by": 24, "from": 25, "they": 26, "we": 27, "say": 28, "her": 29, "she": 30,
    "or": 31, "an": 32, "will": 33, "my": 34, "one": 35, "all": 36, "would": 37, "there": 38, "their": 39, "what": 40,
    "so": 41, "up": 42, "out": 43, "if": 44, "about": 45, "who": 46, "get": 47, "which": 48, "go": 49, "me": 50,
}

_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall",
}

_DISCOURSE_MARKERS = {
    "however", "moreover", "furthermore", "additionally", "consequently",
    "nevertheless", "therefore", "thus", "hence", "accordingly",
}

_FIRST_PERSON_PRONOUNS = {"i", "me", "my", "mine", "myself"}

_CONJUNCTIONS = {"and", "but", "or", "so", "yet"}

_WORD_RE = re.compile(r"[A-Za-z0-9\']+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SYLLABLE_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)
_PASSIVE_RE = re.compile(r"\b(was|were|been|being)\s+[a-z]+ed\b", re.IGNORECASE)


def _count_syllables(word: str) -> int:
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
    word = re.sub(r'^y', '', word)
    return max(1, len(_SYLLABLE_RE.findall(word)))


def extract_advanced_features(text: str) -> list[float]:
    """Extract all advanced features from a single text string."""
    text_len = len(text)
    if text_len == 0:
        return [0.0] * len(FEATURE_NAMES)

    words = _WORD_RE.findall(text)
    word_count = len(words)
    if word_count == 0:
        return [0.0] * len(FEATURE_NAMES)

    lower_words = [w.lower() for w in words]
    word_freqs = Counter(lower_words)
    
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    sentence_count = max(1, len(sentences))
    sentence_lengths = [len(_WORD_RE.findall(s)) for s in sentences]
    
    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]
    paragraph_count = max(1, len(paragraphs))
    paragraph_lengths = [len(_WORD_RE.findall(p)) for p in paragraphs]
    
    char_count = sum(len(w) for w in words)
    syllable_count = sum(_count_syllables(w) for w in words)
    
    # 1. Burstiness coefficient
    mean_sl = np.mean(sentence_lengths)
    std_sl = np.std(sentence_lengths)
    burstiness = float(std_sl / mean_sl) if mean_sl > 0 else 0.0
    
    # 2. Hapax legomena ratio
    hapax_count = sum(1 for count in word_freqs.values() if count == 1)
    hapax_ratio = hapax_count / word_count
    
    # 3. MATTR (window=50)
    window_size = 50
    if word_count < window_size:
        mattr_50 = len(set(lower_words)) / word_count
    else:
        ttrs = []
        for i in range(word_count - window_size + 1):
            window = lower_words[i:i+window_size]
            ttrs.append(len(set(window)) / window_size)
        mattr_50 = float(np.mean(ttrs))
        
    # 4. Yule's K
    m1 = word_count
    m2 = sum(count ** 2 for count in word_freqs.values())
    yules_k = 10000.0 * (m2 - m1) / (m1 * m1) if m1 > 0 else 0.0
    
    # 5. Flesch-Kincaid Grade Level
    # 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    fk_grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
    
    # 6. Coleman-Liau Index
    # 0.0588 * L - 0.296 * S - 15.8 (L=chars per 100 words, S=sentences per 100 words)
    l = (char_count / word_count) * 100
    s = (sentence_count / word_count) * 100
    cl_index = 0.0588 * l - 0.296 * s - 15.8
    
    # 7. Automated Readability Index
    # 4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43
    ari = 4.71 * (char_count / word_count) + 0.5 * (word_count / sentence_count) - 21.43
    
    # 8. Function word frequency
    func_count = sum(word_freqs.get(fw, 0) for fw in _FUNCTION_WORDS)
    func_ratio = func_count / word_count
    
    # 9. Discourse marker density
    disc_count = sum(word_freqs.get(dm, 0) for dm in _DISCOURSE_MARKERS)
    disc_ratio = disc_count / word_count
    
    # 10. Contraction ratio
    contract_count = sum(1 for w in words if "'" in w)
    contract_ratio = contract_count / word_count
    
    # 11. Question frequency
    q_count = sum(1 for s in sentences if s.endswith('?'))
    q_ratio = q_count / sentence_count
    
    # 12. First-person pronoun ratio
    fp_count = sum(word_freqs.get(p, 0) for p in _FIRST_PERSON_PRONOUNS)
    fp_ratio = fp_count / word_count
    
    # 13. Passive voice estimate
    passive_count = sum(len(_PASSIVE_RE.findall(s)) for s in sentences)
    passive_ratio = passive_count / sentence_count
    
    # 14. Sentence start diversity
    starts = []
    for s in sentences:
        s_words = _WORD_RE.findall(s)
        if s_words:
            starts.append(s_words[0].lower())
    start_div = len(set(starts)) / sentence_count if starts else 0.0
    
    # 15. Comma-to-period ratio
    comma_count = text.count(",")
    period_count = text.count(".")
    comma_period_ratio = comma_count / max(1, period_count)
    
    # 16. Average word frequency rank
    ranks = [_TOP_WORDS.get(w, 5000) for w in lower_words]
    avg_rank = float(np.mean(ranks))
    
    # 17. Paragraph length variance
    par_len_var = float(np.std(paragraph_lengths))
    
    # 18. Conjunction-starting ratio
    conj_start_count = sum(1 for start in starts if start in _CONJUNCTIONS)
    conj_start_ratio = conj_start_count / sentence_count
    
    # 19. Compression ratio
    encoded = text.encode('utf-8')
    compressed = zlib.compress(encoded)
    comp_ratio = len(compressed) / max(1, len(encoded))
    
    # 20. Mean word length
    word_lengths = [len(w) for w in words]
    mean_wl = float(np.mean(word_lengths))
    
    # 21. Std of word lengths
    std_wl = float(np.std(word_lengths))
    
    # 22. Mean sentence length
    mean_sl_words = float(np.mean(sentence_lengths))
    
    # 23. Std of sentence length
    std_sl_words = float(np.std(sentence_lengths))
    
    # 24. Lexical diversity
    lex_div = len(set(lower_words)) / word_count
    
    # 25. Punctuation ratio
    punct_count = sum(1 for c in text if c in string.punctuation)
    punct_ratio = punct_count / max(1, text_len)
    
    # 26. Uppercase ratio
    upper_count = sum(1 for c in text if c.isupper())
    upper_ratio = upper_count / max(1, text_len)
    
    # 27. Digit ratio
    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / max(1, text_len)
    
    # 28. Exclamation ratio
    excl_count = text.count("!")
    excl_ratio = excl_count / max(1, punct_count)
    
    # 29. Semicolon+colon ratio
    semi_colon_count = text.count(";") + text.count(":")
    semi_colon_ratio = semi_colon_count / max(1, punct_count)
    
    # 30. Average words per paragraph
    avg_words_par = float(np.mean(paragraph_lengths))

    features = [
        burstiness,
        hapax_ratio,
        mattr_50,
        yules_k,
        fk_grade,
        cl_index,
        ari,
        func_ratio,
        disc_ratio,
        contract_ratio,
        q_ratio,
        fp_ratio,
        passive_ratio,
        start_div,
        comma_period_ratio,
        avg_rank,
        par_len_var,
        conj_start_ratio,
        comp_ratio,
        mean_wl,
        std_wl,
        mean_sl_words,
        std_sl_words,
        lex_div,
        punct_ratio,
        upper_ratio,
        digit_ratio,
        excl_ratio,
        semi_colon_ratio,
        avg_words_par,
    ]
    
    # Ensure no NaN or Inf
    return [0.0 if math.isnan(f) or math.isinf(f) else float(f) for f in features]


class AdvancedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for use in pipelines."""
    
    def fit(self, texts: Any, y: Any = None) -> 'AdvancedFeaturesTransformer':
        return self
        
    def transform(self, texts: Any) -> np.ndarray:
        return np.asarray([extract_advanced_features(str(t)) for t in texts], dtype=float)
