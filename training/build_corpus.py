#!/usr/bin/env python3
"""Train domain word vectors on the FULL media corpus → a bundle-ready matrix.

The binary categorical flags say *that* a genre/title was named; this trains
embeddings that capture *media-domain semantics* — ``jazz`` ≈ ``blues``,
``horror`` ≈ ``thriller``, title-token co-occurrence — which the flags can never
express.  The corpus is assembled from **everything at hand**:

* every entity pool under ``data/entities/*.csv`` (artist / track / album / movie
  / tv / anime / book / podcast / game / … names — millions of strings); each
  value is one "sentence" so its tokens co-occur;
* the relational records under ``data/relational/*.jsonl`` — each record's fields
  joined into one sentence so e.g. an artist, its album and its genre share a
  context window (the signal that pulls related domain tokens together);
* the 347k training utterances (``data/release/train.*``), raw ``sentence`` text.

A ``gensim`` Word2Vec (skip-gram) is trained, then exported **runtime-lean**: the
embedding matrix is saved as ``wordvec.npy`` (row 0 = zero OOV row) + a
``wordvec_vocab.json`` token→row map.  At inference
:class:`ovos_media_classifier.features_wordvec.WordVecPooler` mean-pools an
utterance's in-vocab rows in **numpy only** — no gensim, no torch.

Run (needs ``pip install ovos-media-classifier[train]``)::

    python -m training.build_corpus                 # → data/wordvec/
    python -m training.build_corpus --dim 100 --min-count 5
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Iterator, List

import numpy as np

from ovos_media_classifier.features_wordvec import WordVecSpec, tokenize

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)

# raise the field limit — some entity CSVs have very long values
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _iter_entity_sentences(entities_dir: str) -> Iterator[List[str]]:
    n_files = n_rows = 0
    for path in sorted(glob.glob(os.path.join(entities_dir, "*.csv"))):
        n_files += 1
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)  # skip the "value" header
            for row in reader:
                if not row:
                    continue
                toks = tokenize(" ".join(row))
                if toks:
                    n_rows += 1
                    yield toks
    print(f"  entities: {n_files} pools, {n_rows:,} sentences")


def _iter_relational_sentences(relational_dir: str) -> Iterator[List[str]]:
    n_files = n_rows = 0
    for path in sorted(glob.glob(os.path.join(relational_dir, "*.jsonl"))):
        n_files += 1
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                # join every field value so related tokens (artist/album/genre)
                # share one context window
                vals = [str(v) for v in rec.values() if v not in (None, "")]
                toks = tokenize(" ".join(vals))
                if toks:
                    n_rows += 1
                    yield toks
    print(f"  relational: {n_files} files, {n_rows:,} sentences")


def _iter_utterance_sentences(data_dir: str) -> Iterator[List[str]]:
    import pandas as pd
    n_rows = 0
    for name in ("train", "validation", "test"):
        pq = os.path.join(data_dir, f"{name}.parquet")
        csvp = os.path.join(data_dir, f"{name}.csv")
        if os.path.isfile(pq):
            df = pd.read_parquet(pq, columns=["sentence"])
        elif os.path.isfile(csvp):
            df = pd.read_csv(csvp, usecols=["sentence"])
        else:
            continue
        for s in df["sentence"]:
            toks = tokenize(str(s))
            if toks:
                n_rows += 1
                yield toks
    print(f"  utterances: {n_rows:,} sentences")


class _Corpus:
    """Re-iterable streaming corpus (gensim makes multiple passes)."""

    def __init__(self, entities_dir, relational_dir, data_dir):
        self._entities = entities_dir
        self._relational = relational_dir
        self._data = data_dir

    def __iter__(self):
        if os.path.isdir(self._entities):
            yield from _iter_entity_sentences(self._entities)
        if os.path.isdir(self._relational):
            yield from _iter_relational_sentences(self._relational)
        if os.path.isdir(self._data):
            yield from _iter_utterance_sentences(self._data)


def train_wordvec(entities_dir, relational_dir, data_dir, out_dir,
                  dim=100, window=5, min_count=5, epochs=5, sg=1, workers=4,
                  seed=42, max_vocab=200_000):
    """Train Word2Vec on the domain corpus and export ``.npy`` + vocab json."""
    from gensim.models import Word2Vec

    os.makedirs(out_dir, exist_ok=True)
    spec = WordVecSpec(dim=dim)

    print("Streaming corpus …")
    corpus = _Corpus(entities_dir, relational_dir, data_dir)

    print(f"Training Word2Vec(dim={dim}, window={window}, min_count={min_count}, "
          f"sg={sg}, epochs={epochs}) …")
    model = Word2Vec(
        sentences=corpus, vector_size=dim, window=window, min_count=min_count,
        workers=workers, sg=sg, epochs=epochs, seed=seed,
        max_final_vocab=max_vocab,
    )
    kv = model.wv
    vocab = list(kv.index_to_key)
    print(f"  vocabulary: {len(vocab):,} tokens")

    # row 0 is the zero OOV / pad row; tokens start at row 1
    matrix = np.zeros((len(vocab) + 1, dim), dtype="float32")
    token_to_row = {}
    for i, tok in enumerate(vocab):
        matrix[i + 1] = kv[tok]
        token_to_row[tok] = i + 1

    np.save(os.path.join(out_dir, spec.vectors_file), matrix)
    with open(os.path.join(out_dir, spec.vocab_file), "w", encoding="utf-8") as fh:
        json.dump(token_to_row, fh)
    spec_path = os.path.join(out_dir, "wordvec_spec.json")
    with open(spec_path, "w", encoding="utf-8") as fh:
        json.dump(spec.to_meta(), fh, indent=2)

    nbytes = (os.path.getsize(os.path.join(out_dir, spec.vectors_file))
              + os.path.getsize(os.path.join(out_dir, spec.vocab_file)))
    print(f"  wrote {out_dir}/{spec.vectors_file} + vocab ({nbytes/1024/1024:.1f} MiB)")

    # quick sanity probes — do related domain tokens cluster?
    for probe in ("jazz", "horror", "rock"):
        if probe in kv:
            sims = ", ".join(f"{w}" for w, _ in kv.most_similar(probe, topn=5))
            print(f"  {probe!r} ≈ {sims}")
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Train domain word vectors on the full media corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--entities-dir", default=os.path.join("data", "entities"))
    ap.add_argument("--relational-dir", default=os.path.join("data", "relational"))
    ap.add_argument("--data-dir", default=os.path.join("data", "release"))
    ap.add_argument("--out-dir", default=os.path.join("data", "wordvec"))
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--sg", type=int, default=1, help="1=skip-gram, 0=CBOW")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max-vocab", type=int, default=200_000)
    args = ap.parse_args(argv)

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)

    train_wordvec(
        _abs(args.entities_dir), _abs(args.relational_dir), _abs(args.data_dir),
        _abs(args.out_dir), dim=args.dim, window=args.window,
        min_count=args.min_count, epochs=args.epochs, sg=args.sg,
        workers=args.workers, max_vocab=args.max_vocab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
