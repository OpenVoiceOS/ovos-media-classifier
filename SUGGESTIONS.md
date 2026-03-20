
# Code Improvement Suggestions

This document outlines recommended improvements for the `ovos-media-classifier` codebase. These are suggestions for human review—implement based on priorities and project needs.

---

## 1. Type Hints Improvements

### Missing type annotations for class attributes

**Files:** `m2v.py`, `sklearn.py`, `padatious.py`, `ahocorasick.py`

**Issue:** Several classifier classes use untyped attributes:

```python
# m2v.py:52 - _model lacks type hint
def __init__(self, model, domain_threshold: float = 0.5,
             intent_threshold: float = 0.3) -> None:
    self._model = model  # Should be typed

# sklearn.py:64-70 - pipelines lack type hints
def __init__(
    self,
    play_pipeline,
    domain_pipeline=None,
    ...
):
    self._play = play_pipeline  # Should be typed
```

**Suggestion:** Add `Protocol` classes or type aliases for the model/pipeline interfaces:

```python
# Add to a new file or existing types module
from typing import Protocol, Any

class ClassifierModel(Protocol):
    """Protocol for hierarchical classification models."""
    def predict(self, texts: list[str]) -> tuple[list[str], list[str]]: ...
    def predict_proba(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]: ...
    domain_classes_: np.ndarray

class SklearnPipeline(Protocol):
    """Protocol for sklearn pipelines."""
    def predict(self, texts: list[str]) -> Any: ...
    def predict_proba(self, texts: list[Any]) -> np.ndarray: ...
```

### Incomplete return type annotations

**File:** `entities.py:85`

```python
# Current
def _http_get(session, url: str, **kwargs) -> Optional[dict | list]:
    # Should be more specific
```

---

## 2. Missing Docstrings

### Add class-level docstrings

**File:** `sklearn.py` - `SklearnMediaClassifier` class lacks a proper docstring (only has a comment).

**Suggestion:**
```python
class SklearnMediaClassifier(AbstractMediaClassifier):
    """Media classifier backed by scikit-learn TF-IDF + classifier pipelines.
    
    This classifier uses two optional sklearn pipelines:
      - play_pipeline: Multi-class classifier for OCPPlayIntent labels
      - domain_pipeline: Optional binary/ternary classifier for OCPDomain
    
    Args:
        play_pipeline: sklearn Pipeline that predicts OCPPlayIntent labels.
        domain_pipeline: Optional sklearn Pipeline for domain classification.
        play_threshold: Minimum probability for play predictions.
        domain_threshold: Minimum probability for domain predictions.
    
    Example:
        >>> clf = SklearnMediaClassifier.from_path("/path/to/model.joblib")
        >>> clf.classify("play jazz", "en-us")
        (MediaType.MUSIC, 0.85)
    """
```

### Add docstrings to internal methods

**Files:** Multiple internal helper methods lack docstrings.

**Recommendation:** Add docstrings to:
- `_VocMatcher._load()` in `keyword.py`
- `_VocMatcher.match()` in `keyword.py`  
- `_sync_from_ner()` in `entities.py`
- `_entities_to_intent()` in `ahocorasick.py`
- `_calc_intent()` in `padatious.py`
- `_predict_with_conf()` in `sklearn.py`

---

## 3. Magic Numbers and Constants

### Confidence values scattered throughout

**Issue:** Confidence thresholds are hardcoded in multiple places:

- `keyword.py`: 0.6, 0.7, 0.4 (various media types)
- `ahocorasick.py:113`: `_HIT_CONFIDENCE = 0.6`
- `sklearn.py`: Default thresholds 0.3 and 0.5
- `m2v.py`: Default thresholds 0.5 and 0.3
- `padatious.py`: Default threshold 0.5

**Suggestion:** Create a centralized constants module:

```python
# ovos_media_classifier/constants.py
"""Confidence thresholds and classification constants."""

# Default confidence thresholds
DEFAULT_KEYWORD_CONFIDENCE = 0.6
DEFAULT_KEYWORD_SUBCONFIDENCE = 0.7  # More specific matches
DEFAULT_KEYWORD_GENERIC = 0.4

DEFAULT_NER_CONFIDENCE = 0.6
DEFAULT_SKLEARN_PLAY_THRESHOLD = 0.3
DEFAULT_SKLEARN_DOMAIN_THRESHOLD = 0.5
DEFAULT_M2V_DOMAIN_THRESHOLD = 0.5
DEFAULT_M2V_INTENT_THRESHOLD = 0.3
DEFAULT_PADATIOUS_THRESHOLD = 0.5
```

### Priority ordering as constants

**File:** `ahocorasick.py:69-106`

The `_INTENT_PRIORITY` list is implementation detail that could be documented better or moved to `intents.py`.

---

## 4. Error Handling Improvements

### Inconsistent exception handling

**File:** `entities.py:129-133`

```python
def _sync_from_ner(self, ner) -> None:
    """Populate ``_by_label`` from an existing NER's word store."""
    try:
        for label, words in ner.get_all_words().items():
            self._by_label[label].update(words)
    except Exception:
        pass  # NER may not expose this; non-fatal
```

**Suggestion:** 
1. Log the exception for debugging
2. Consider catching specific exceptions rather than broad `Exception`

```python
def _sync_from_ner(self, ner) -> None:
    """Populate ``_by_label`` from an existing NER's word store."""
    try:
        for label, words in ner.get_all_words().items():
            self._by_label[label].update(words)
    except (AttributeError, TypeError) as e:
        LOG.debug("NER does not expose word store: %s", e)
```

### Import error messages could suggest installation

**File:** Multiple files have ImportError but with inconsistent messages.

**Standardize:**
```python
# Consistent format
REQUIRED_PACKAGES = {
    "torch": "torch>=2.0.0 (pip install ovos-media-classifier[m2v])",
    "sklearn": "scikit-learn (pip install ovos-media-classifier[sklearn])",
    # ...
}

def _require(package: str) -> None:
    """Raise ImportError with helpful message."""
    raise ImportError(
        f"{package} is required. Install with: pip install {REQUIRED_PACKAGES.get(package, package)}"
    )
```

---

## 5. Code Comments

### Add section comments for code organization

**Files:** All classifier files could benefit from clearer section comments:

```python
# ---------------------------------------------------------------------------
# Factory Methods
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Classification Implementation  
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Domain Classification
# ---------------------------------------------------------------------------
```

### Document tricky logic

**File:** `keyword.py:146-148`

```python
if _ok(MediaType.RADIO_THEATRE) and m(q, "AudioDramaKeyword", lang):
    # NOTE: must come before plain RADIO so "radio theatre" wins
    return MediaType.RADIO_THEATRE, 0.6
```

These comments are present but could be more comprehensive. Consider documenting:
- Why certain priority orderings exist
- Edge cases in the classification logic
- Assumptions about input data

---

## 6. Refactoring Opportunities

### Extract common classification logic

**Observation:** All classifiers implement similar logic:

```python
# Pattern repeated in keyword.py, ahocorasick.py, sklearn.py, m2v.py, padatious.py
if valid_labels is not None and media_type not in valid_labels:
    return MediaType.GENERIC, 0.0
```

**Suggestion:** Add a utility function in `base.py`:

```python
def _filter_by_valid_labels(
    media_type: MediaType,
    confidence: float,
    valid_labels: Optional[List[MediaType]],
) -> Tuple[MediaType, float]:
    """Return GENERIC if media_type not in valid_labels."""
    if valid_labels is not None and media_type not in valid_labels:
        return MediaType.GENERIC, 0.0
    return media_type, confidence
```

### Consider a base mixin for threshold handling

Multiple classifiers have similar threshold-checking logic. A mixin could reduce duplication.

---

## 7. Testing Improvements

### Add type hints to test helpers

**File:** `test/test_classifier.py:35-72`

The helper functions `_kw_clf()` and `_mock_m2v_model()` lack return type annotations.

---

## 8. Documentation Improvements

### README.md gaps

- No quick reference for confidence value meanings
- Missing API stability guarantees
- Could benefit from a comparison table of backends

### Add architecture diagram

A visual showing how the factory selects backends and how data flows would help new contributors.

---

## Priority Recommendations

| Priority | Item | Effort |
|----------|------|--------|
| High | Add type hints to classifier attributes | Medium |
| High | Create constants module for confidence values | Low |
| Medium | Add docstrings to SklearnMediaClassifier | Low |
| Medium | Standardize ImportError messages | Low |
| Low | Extract common classification utilities | Medium |
| Low | Add architecture diagram to docs | Medium |

---

## Files Requiring Most Attention

1. **`sklearn.py`** - Needs docstrings and type hints
2. **`m2v.py`** - Needs type hints for `_model` attribute
3. **`entities.py`** - Could benefit from standardized error handling
4. **All classifier files** - Would benefit from constants extraction
