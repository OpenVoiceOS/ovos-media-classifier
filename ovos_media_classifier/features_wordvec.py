"""Trained domain word-vector pooling featurizer (numpy only).

The categorical flags say *that* a genre/title was named; the char-hash block
(:mod:`ovos_media_classifier.features_text`) can see *which* subwords appeared.
This third family captures **media-domain semantics**: word vectors trained on the
full domain corpus (every entity pool — artist/track/album/movie/tv/anime/book/
podcast/game names — plus the relational co-occurrence records and the utterance
text). Tokens that co-occur in that corpus end up close in vector space, so
``jazz`` ≈ ``blues``, ``horror`` ≈ ``thriller``, and title-token associations are
expressible in a way binary flags never are.

Runtime is **numpy only** — no gensim, no torch. The trained embedding matrix is
saved as a ``.npy`` in the bundle alongside a token→row vocabulary; at inference
an utterance is tokenized, each known token looks up its row, and the rows are
mean-pooled (optionally L2-normalized) into one dense block. The vocab + pooling
config live in ``meta.json`` (:class:`WordVecSpec`) so :class:`OnnxMediaClassifier`
reproduces the exact vector.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Whitespace/punct tokenizer shared by corpus build, train and runtime."""
    text = text or ""
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


@dataclass
class WordVecSpec:
    """Self-describing config for the pooled word-vector featurizer.

    The embedding matrix itself is a separate ``.npy`` (``vectors_file``) and the
    vocabulary (token → row index) is recorded out-of-band in the bundle
    (``vocab_file``) to keep ``meta.json`` small. This spec carries only the
    scalar reproduction knobs.

    Args:
        dim: embedding dimension (== n columns of the ``.npy``).
        pooling: ``"mean"`` (default) or ``"sum"`` over the in-vocab token rows.
        lowercase: lowercase before tokenizing.
        normalize: L2-normalize the pooled block (mean of unit-ish vectors).
        vectors_file: ``.npy`` filename inside the bundle (``(vocab+1, dim)``;
            row 0 is the all-zero OOV / pad row).
        vocab_file: JSON filename mapping token → row index inside the bundle.
        prefix: feature-name prefix for the produced columns (``wv_*``).
    """

    dim: int = 100
    pooling: str = "mean"
    lowercase: bool = True
    normalize: bool = True
    vectors_file: str = "wordvec.npy"
    vocab_file: str = "wordvec_vocab.json"
    prefix: str = "wv"

    def to_meta(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_meta(cls, meta: Optional[Dict[str, object]]) -> Optional["WordVecSpec"]:
        if not meta:
            return None
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in meta.items() if k in known})

    def feature_names(self) -> List[str]:
        return [f"{self.prefix}_{i}" for i in range(self.dim)]


class WordVecPooler:
    """Numpy lookup-and-pool over a saved embedding matrix + vocab.

    Args:
        vectors: ``(n_rows, dim)`` float32 matrix; row 0 is the zero OOV row.
        vocab: token → row index (≥ 1; OOV tokens are skipped).
        spec: the :class:`WordVecSpec` (pooling / normalize / lowercase).
    """

    def __init__(self, vectors, vocab: Dict[str, int], spec: WordVecSpec) -> None:
        self._vectors = vectors
        self._vocab = vocab
        self._spec = spec

    def pool(self, text: str, out=None):
        """Pool the in-vocab token vectors of *text* → ``(dim,)`` float32."""
        import numpy as np

        spec = self._spec
        if out is None:
            vec = np.zeros(spec.dim, dtype="float32")
        else:
            vec = out
            vec.fill(0.0)

        rows = [self._vocab[t] for t in tokenize(text, spec.lowercase)
                if t in self._vocab]
        if not rows:
            return vec
        block = self._vectors[rows]               # (k, dim)
        if spec.pooling == "sum":
            np.add.reduce(block, axis=0, out=vec)
        else:  # mean
            vec[:] = block.mean(axis=0)
        if spec.normalize:
            n = float(np.sqrt(np.dot(vec, vec)))
            if n > 0.0:
                vec /= n
        return vec

    def matrix(self, texts: List[str]):
        import numpy as np

        mat = np.zeros((len(texts), self._spec.dim), dtype="float32")
        buf = np.zeros(self._spec.dim, dtype="float32")
        for i, t in enumerate(texts):
            self.pool(t, out=buf)
            mat[i] = buf
        return mat

    # ------------------------------------------------------------------
    @classmethod
    def from_bundle(cls, bundle_dir: str, spec: WordVecSpec) -> Optional["WordVecPooler"]:
        """Load the ``.npy`` matrix + vocab json from a bundle dir, or ``None``."""
        import json
        import os

        import numpy as np

        vpath = os.path.join(bundle_dir, spec.vectors_file)
        kpath = os.path.join(bundle_dir, spec.vocab_file)
        if not (os.path.isfile(vpath) and os.path.isfile(kpath)):
            return None
        vectors = np.load(vpath).astype("float32")
        with open(kpath, encoding="utf-8") as fh:
            vocab = {k: int(v) for k, v in json.load(fh).items()}
        return cls(vectors, vocab, spec)
