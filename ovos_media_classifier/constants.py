"""Shared confidence thresholds used across all OCP media classifiers.

Import these instead of defining inline magic numbers in individual modules.
"""

# Minimum softmax/predict_proba confidence to trust the domain head
DEFAULT_DOMAIN_THRESHOLD: float = 0.5

# Minimum softmax/predict_proba confidence to trust the play-intent head
DEFAULT_PLAY_THRESHOLD: float = 0.3

# Confidence returned by keyword-based classifiers on a keyword match
DEFAULT_KEYWORD_CONFIDENCE: float = 0.6

# Higher confidence for more specific keyword matches (sub-type refinement)
DEFAULT_KEYWORD_HIGH_CONFIDENCE: float = 0.7

# Lower confidence for ambiguous keyword matches (e.g. game, ASMR, comic)
DEFAULT_KEYWORD_LOW_CONFIDENCE: float = 0.4

# Confidence returned by Aho-Corasick NER on an entity hit
DEFAULT_NER_HIT_CONFIDENCE: float = 0.6
