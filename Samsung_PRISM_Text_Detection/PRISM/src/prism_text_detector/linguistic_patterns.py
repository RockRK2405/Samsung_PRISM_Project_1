from __future__ import annotations

import re

# 1. AI-style transition phrases
AI_TRANSITIONS = {
    "in conclusion", "moreover", "furthermore", "additionally", 
    "on the other hand", "in today's world", "plays a vital role", 
    "bridge the gap", "valuable stepping stone", "it is worth noting", 
    "it is important to mention", "in the modern era", "has become increasingly", 
    "plays a crucial role", "in summary", "to summarize", "overall", "in essence"
}

# 2. Overconfident/absolute language
ABSOLUTE_WORDS = {
    "always", "never", "everyone", "nobody", "guaranteed", "definitely", 
    "undeniable", "proves", "100%", "completely", "totally", "must", 
    "certainly", "undoubtedly"
}

# 3. Vague source claims
VAGUE_SOURCES = {
    "experts say", "studies show", "research says", "many people believe", 
    "it is said", "sources claim", "reports suggest", "generally", 
    "according to experts", "scientists believe"
}

# 4. Sensational language
SENSATIONAL_WORDS = {
    "shocking", "secret", "exposed", "dangerous", "viral", "massive", 
    "unbelievable", "hidden truth", "must know", "alarming", "groundbreaking"
}

# 5. Hedging phrases
HEDGING_PHRASES = {
    "it should be noted", "it is important to recognize", "one might argue", 
    "it could be said", "there are various", "it is essential to"
}

def analyze_patterns(text: str) -> dict:
    """Analyze text for AI-typical patterns.
    
    Returns dict with counts, flags, and an overall score.
    """
    if not text or not text.strip():
        return {
            'ai_phrase_count': 0,
            'absolute_language_count': 0,
            'vague_source_count': 0,
            'sensational_count': 0,
            'hedging_count': 0,
            'structural_pattern_count': 0,
            'formulaic_opening_count': 0,
            'total_flag_count': 0,
            'flags': [],
            'overall_score': 0.0,
            'details': {}
        }

    lower_text = text.lower()
    
    # Count occurrences
    ai_phrase_count = sum(lower_text.count(p) for p in AI_TRANSITIONS)
    
    # We want whole word matches for single words to avoid partial matching (e.g., "must" in "mustard")
    absolute_count = 0
    for word in ABSOLUTE_WORDS:
        if " " in word or "%" in word:
            absolute_count += lower_text.count(word)
        else:
            absolute_count += len(re.findall(r'\b' + word + r'\b', lower_text))
            
    vague_count = sum(lower_text.count(p) for p in VAGUE_SOURCES)
    
    sensational_count = 0
    for word in SENSATIONAL_WORDS:
        if " " in word:
            sensational_count += lower_text.count(word)
        else:
            sensational_count += len(re.findall(r'\b' + word + r'\b', lower_text))
            
    hedging_count = sum(lower_text.count(p) for p in HEDGING_PHRASES)
    
    # 6. Structural patterns
    structural_count = 0
    # Numbered lists (e.g., 1., 2., 3.)
    structural_count += len(re.findall(r'^[ \t]*\d+\.[ \t]+', text, re.MULTILINE))
    # Bullet lists (e.g., -, *, •)
    structural_count += len(re.findall(r'^[ \t]*[-*•][ \t]+', text, re.MULTILINE))
    
    # 7. Formulaic openings
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    formulaic_count = 0
    for s in sentences:
        s_lower = s.lower()
        if s_lower.startswith(("this ", "these ", "those ", "such ")):
            formulaic_count += 1

    # Compile flags
    flags = []
    if ai_phrase_count > 0:
        flags.append("AI-style transition phrases detected")
    if absolute_count > 2:
        flags.append("Excessive overconfident/absolute language")
    if vague_count > 0:
        flags.append("Vague source claims ('experts say', etc.)")
    if sensational_count > 1:
        flags.append("Sensational language used")
    if hedging_count > 1:
        flags.append("Frequent hedging phrases")
    if structural_count > 3:
        flags.append("Highly structured formatting (lists/bullets)")
    if formulaic_count > 3:
        flags.append("Formulaic sentence openings")

    total_flags = ai_phrase_count + absolute_count + vague_count + sensational_count + hedging_count + structural_count + formulaic_count
    
    # Simple scoring heuristic (0 to 1)
    # Based on expected counts in typical text length
    score = min(1.0, (
        (ai_phrase_count * 0.15) +
        (absolute_count * 0.05) +
        (vague_count * 0.15) +
        (sensational_count * 0.05) +
        (hedging_count * 0.1) +
        (structural_count * 0.05) +
        (formulaic_count * 0.05)
    ))

    return {
        'ai_phrase_count': ai_phrase_count,
        'absolute_language_count': absolute_count,
        'vague_source_count': vague_count,
        'sensational_count': sensational_count,
        'hedging_count': hedging_count,
        'structural_pattern_count': structural_count,
        'formulaic_opening_count': formulaic_count,
        'total_flag_count': total_flags,
        'flags': flags,
        'overall_score': score,
        'details': {
            'ai_phrases': ai_phrase_count,
            'absolute': absolute_count,
            'vague': vague_count,
            'sensational': sensational_count,
            'hedging': hedging_count,
            'structural': structural_count,
            'formulaic': formulaic_count
        }
    }
