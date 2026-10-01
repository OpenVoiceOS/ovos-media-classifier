"""Deterministic, dependency-light TEXT featurizer (numpy only).

The categorical features in :mod:`ovos_media_classifier.features` are binary
*presence* flags — they record that *a* genre/title/artist was named, never
*which* one (see ``docs/model.md`` §1). This module adds a second feature family
that *can* see the surface text: **hashed character n-grams** (and, optionally,
hashed word n-grams) of the utterance, projected to a fixed dimension with the
hashing trick and L2-normalized.

It is the same code at **train time** (``training/train_torch.py`` builds the
training matrix) and at **inference** (``OnnxMediaClassifier`` reproduces the
exact vector in numpy), so a bundle that declares a :class:`TextHashSpec` in its
``meta.json`` is fully self-describing — no torch, no transformers, no vocab file
at runtime, just numpy and the spec.

Why hashing (not a learned vocabulary)?
    A fixed hashing function needs **no stored vocabulary** — the runtime
    reproduces the projection from the spec (dim + ngram range + analyzer)
    alone. The trade-off is hash collisions, which at a few thousand dims over
    short utterances are negligible and act as mild regularization.

The hash is a stable 64-bit FNV-1a over the byte-encoded n-gram, with a sign bit
drawn from a second hash (signed hashing trick → unbiased collisions). Being
pure-python + numpy and seedless, it is byte-for-byte reproducible across
machines and Python versions.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

# FNV-1a 64-bit constants — a fast, well-distributed, dependency-free hash.
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF

# collapse runs of whitespace; keep word chars + spaces for the word analyzer
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


def _fnv1a(data: bytes) -> int:
    h = _FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & _MASK64
    return h


@dataclass
class TextHashSpec:
    """Self-describing config for the hashed-text featurizer.

    Recorded verbatim in a bundle's ``meta.json`` so the runtime reproduces the
    exact projection. All fields are JSON-scalars / small lists.

    Args:
        dim: output dimension (number of hash buckets).
        analyzer: ``"char"`` (character n-grams within word boundaries, the
            default) or ``"word"`` (whitespace-tokenized word n-grams).
        ngram_min / ngram_max: inclusive n-gram size range.
        lowercase: lowercase the utterance before extracting n-grams.
        char_pad: pad each word with a boundary marker so char n-grams capture
            word-initial / word-final subwords (only used for the char analyzer).
        prefix: feature-name prefix for the produced columns (``txt_*``).
    """

    dim: int = 4096
    analyzer: str = "char"
    ngram_min: int = 3
    ngram_max: int = 5
    lowercase: bool = True
    char_pad: bool = True
    prefix: str = "txt"

    def to_meta(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_meta(cls, meta: Optional[Dict[str, object]]) -> Optional["TextHashSpec"]:
        if not meta:
            return None
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in meta.items() if k in known})

    # ------------------------------------------------------------------
    # column names (so the bundle's feature_names can list them explicitly)
    # ------------------------------------------------------------------
    def feature_names(self) -> List[str]:
        return [f"{self.prefix}_{i}" for i in range(self.dim)]


# ---------------------------------------------------------------------------
# n-gram extraction
# ---------------------------------------------------------------------------

def _iter_char_ngrams(text: str, lo: int, hi: int, pad: bool) -> Iterable[str]:
    # one token per "word"; char n-grams stay within a word so "jazz" subwords
    # never bleed across the space into the next token.
    for word in text.split():
        w = f"\x02{word}\x03" if pad else word
        n = len(w)
        for k in range(lo, hi + 1):
            if n < k:
                continue
            for i in range(n - k + 1):
                yield w[i:i + k]


def _iter_word_ngrams(text: str, lo: int, hi: int) -> Iterable[str]:
    words = [w for w in _WORD_RE.split(text) if w]
    for k in range(lo, hi + 1):
        for i in range(len(words) - k + 1):
            yield " ".join(words[i:i + k])


def _normalize(text: str, spec: TextHashSpec) -> str:
    text = text or ""
    if spec.lowercase:
        text = text.lower()
    return _WS_RE.sub(" ", text).strip()


def hash_vector(text: str, spec: TextHashSpec, out=None):
    """Return the L2-normalized hashed-n-gram vector for *text* (numpy float32).

    Signed hashing trick: each n-gram lands in ``hash % dim`` with a ±1 sign from
    a second hash bit, so colliding n-grams cancel in expectation rather than
    pile up. The vector is L2-normalized (zero-vector stays zero).

    Args:
        text: raw utterance.
        spec: the :class:`TextHashSpec` (dim / analyzer / ngram range).
        out: optional pre-allocated ``(dim,)`` float32 array to fill in place
            (zeroed first); allocated fresh when ``None``.
    """
    import numpy as np

    if out is None:
        vec = np.zeros(spec.dim, dtype="float32")
    else:
        vec = out
        vec.fill(0.0)

    norm = _normalize(text, spec)
    if not norm:
        return vec

    if spec.analyzer == "word":
        grams = _iter_word_ngrams(norm, spec.ngram_min, spec.ngram_max)
    else:
        grams = _iter_char_ngrams(norm, spec.ngram_min, spec.ngram_max, spec.char_pad)

    dim = spec.dim
    for g in grams:
        h = _fnv1a(g.encode("utf-8"))
        bucket = h % dim
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[bucket] += sign

    n = float(np.sqrt(np.dot(vec, vec)))
    if n > 0.0:
        vec /= n
    return vec


def hash_matrix(texts: List[str], spec: TextHashSpec):
    """Stack :func:`hash_vector` over *texts* → ``(len(texts), dim)`` float32."""
    import numpy as np

    mat = np.zeros((len(texts), spec.dim), dtype="float32")
    buf = np.zeros(spec.dim, dtype="float32")
    for i, t in enumerate(texts):
        hash_vector(t, spec, out=buf)
        mat[i] = buf
    return mat
