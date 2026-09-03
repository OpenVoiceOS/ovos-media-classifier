#!/usr/bin/env python3
"""Train the guided-categorical-embeddings (GCE) **embedding-router** bundle.

Trains a :class:`PerAxisRouter` (independent per-axis heads) on the classifier
release dataset (``data/release/train.parquet``) with a **routing-aware**
objective and exports a per-axis ONNX bundle that
:class:`ovos_media_classifier.embedding.EmbeddingMediaClassifier` loads with
numpy + onnxruntime only.

Two-stream features (matches the runtime backend)
-------------------------------------------------
The model input is ``[static | entity]``:

* **static** — the categorical ``kw_* / verb_* / mod_* / fmt_* / kw_genre_*``
  columns (the exact menu the runtime
  :class:`~ovos_media_classifier.features.CategoricalFeatureExtractor` produces
  from ``.voc`` matching), one-hot via GCE's ``CategoricalVectorizer``.
* **entity** — one slot per NER label (``artist_name`` / ``movie_title`` /
  ``anime_title`` / …), taken from the dataset's pre-computed ``ner_*`` columns.
  GCE's ``EntityFeaturizer`` is seeded with representative entity strings from
  the dataset (``slot_values``) so the exported bundle records the entity layout
  and ships real seeds; at runtime the user's OWN library is injected into the
  same slots WITHOUT retraining.

Routing-aware objective (per axis)
----------------------------------
A ``GENERIC`` abstain class is appended to each axis (a fraction of rows are
relabelled GENERIC to teach the head the abstain column) with:

* ``cost_matrix`` — a confident WRONG route costs ``MISROUTE_COST``; routing to
  the cheap ``GENERIC`` column costs ``ABSTAIN_COST`` (``mis-route >> abstain``).
* ``abstain_label="GENERIC"`` + ``focal_gamma`` for calibration +
  ``temperature_scaling`` so the exported probabilities are calibrated and the
  runtime reject threshold is meaningful.

Usage::

    python -m training.train_embedding_router \
        --data data/release --out data/models/embedding_router \
        --max-rows 60000 --max-iter 60

Requires ``pip install ovos-media-classifier[train]`` and an importable
``guided-categorical-embeddings``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Tuple

import pandas as pd

_LOG = logging.getLogger("train_embedding_router")

# --- categorical (static) feature columns: the menu the runtime extractor can
# produce (keyword .voc columns + linguistic verb/mod/fmt cues + per-value genre).
from ovos_media_classifier.features import _KEYWORD_VOCABS, VALUE_FEATURE_COLS

STATIC_COLS: List[str] = [col for _, col in _KEYWORD_VOCABS] + list(VALUE_FEATURE_COLS)

# routing-aware cost knobs: a confident wrong route prunes the right provider
# (the real harm); routing to GENERIC is harmless (every provider still searches).
MISROUTE_COST = 10.0
ABSTAIN_COST = 1.0
# fraction of training rows relabelled to GENERIC so the head LEARNS the abstain
# column (otherwise GENERIC never appears as a target and is unreachable).
ABSTAIN_FRACTION = 0.12
GENERIC = "GENERIC"


def _static_dicts(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Sparse ``{col: "1"}`` dicts from the binary static feature columns."""
    cols = [c for c in STATIC_COLS if c in df.columns]
    arr = df[cols].to_numpy()
    out: List[Dict[str, str]] = []
    for row in arr:
        out.append({cols[i]: "1" for i in range(len(cols)) if row[i] == 1})
    return out


def _entity_columns(df: pd.DataFrame) -> List[str]:
    """Active ``ner_*`` columns (those that fire at least once), label-sorted."""
    ner = [c for c in df.columns if c.startswith("ner_")]
    active = sorted(c for c in ner if int(df[c].sum()) > 0)
    return active


def _entity_block(df: pd.DataFrame, ner_cols: List[str]):
    """The entity one-hot block (n, len(ner_cols)) from the dataset columns."""
    import numpy as np

    return df[ner_cols].to_numpy().astype("float32") if ner_cols \
        else np.zeros((len(df), 0), dtype="float32")


def _seed_featurizer(df: pd.DataFrame, entity_labels: List[str], cap: int):
    """A GCE ``EntityFeaturizer`` seeded with representative entity strings.

    Reads the dataset's ``slot_values`` JSON to collect a few real example
    phrases per NER label, so the exported bundle carries the entity layout +
    representative seeds (the user's library is injected on top at runtime).
    """
    from guided_categorical_embeddings.text_features import EntityFeaturizer

    samples: Dict[str, set] = {lbl: set() for lbl in entity_labels}
    label_set = set(entity_labels)
    for raw in df["slot_values"].dropna():
        try:
            d = json.loads(raw)
        except Exception:
            continue
        for label, val in d.items():
            if label in label_set and isinstance(val, str) and val.strip():
                if len(samples[label]) < cap:
                    samples[label].add(val.strip())
    feat = EntityFeaturizer()
    with feat.batch_register():
        for label in entity_labels:
            vals = sorted(samples.get(label) or [])
            if vals:
                feat.register_entity(label, vals)
    return feat


def _inject_abstain(labels: List[str], rng) -> List[str]:
    """Relabel a random ABSTAIN_FRACTION of rows to GENERIC (teach the column)."""
    import numpy as np

    y = list(labels)
    n = len(y)
    k = int(ABSTAIN_FRACTION * n)
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        y[i] = GENERIC
    return y


def _cost_matrix(labels: List[str]) -> Dict[str, Dict[str, float]]:
    """``{true: {pred: cost}}`` — wrong route expensive, GENERIC column cheap."""
    classes = sorted(set(labels))
    cm: Dict[str, Dict[str, float]] = {}
    for t in classes:
        row: Dict[str, float] = {}
        for p in classes:
            if p == t:
                row[p] = 0.0
            elif p == GENERIC:
                row[p] = ABSTAIN_COST          # abstaining is cheap (safe)
            else:
                row[p] = MISROUTE_COST         # confident wrong route is costly
        cm[t] = row
    return cm


def _cap_per_class(df: pd.DataFrame, col: str, cap: int, seed: int) -> pd.DataFrame:
    """Down-sample each class of *col* to at most *cap* rows (balance)."""
    parts = []
    for _, g in df.groupby(col):
        parts.append(g.sample(n=min(len(g), cap), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def train(data_dir: str, out_dir: str, max_rows: int, per_class_cap: int,
          max_iter: int, hidden: Tuple[int, ...], focal_gamma: float,
          seed: int) -> None:
    import numpy as np
    from guided_categorical_embeddings.features import FeatureCombiner
    from guided_categorical_embeddings.train.embeddings import PerAxisRouter
    from guided_categorical_embeddings.train.export import export_combiner_model
    from guided_categorical_embeddings.vectorizer import CategoricalVectorizer

    rng = np.random.default_rng(seed)
    train_path = os.path.join(data_dir, "train.parquet")
    _LOG.info(f"loading {train_path}")
    df = pd.read_parquet(train_path)
    df = df[df["domain"] == "ocp_play"].reset_index(drop=True)

    # balance media_type then cap total rows (keeps minority leaves represented)
    df = _cap_per_class(df, "media_type", per_class_cap, seed)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    _LOG.info(f"training rows: {len(df)}  media_types={df['media_type'].nunique()}")

    ner_cols = _entity_columns(df)
    entity_labels = [c[len("ner_"):] for c in ner_cols]
    _LOG.info(f"static cols={len([c for c in STATIC_COLS if c in df.columns])} "
              f"entity slots={len(entity_labels)}")

    static_dicts = _static_dicts(df)
    entity_block = _entity_block(df, ner_cols)

    # fit the static vectorizer; assemble the combined [static | entity] matrix.
    vectorizer = CategoricalVectorizer()
    static = vectorizer.fit_transform(static_dicts)
    X = np.hstack([static, entity_block]).astype("float32")
    _LOG.info(f"feature matrix: {X.shape} (static={static.shape[1]} "
              f"entity={entity_block.shape[1]})")

    # per-axis labels, each with a learned GENERIC abstain column.
    y_media = _inject_abstain(list(df["media_type"]), rng)
    y_play = _inject_abstain(list(df["playback_type"]), rng)

    axes = ["media_type", "playback_type"]
    router = PerAxisRouter(
        axes=axes,
        per_axis_kwargs={
            "media_type": {
                "abstain_label": GENERIC,
                "cost_matrix": _cost_matrix(y_media),
            },
            "playback_type": {
                "abstain_label": GENERIC,
                "cost_matrix": _cost_matrix(y_play),
            },
        },
        hidden_layer_sizes=hidden,
        max_iter=max_iter,
        focal_gamma=focal_gamma,
        temperature_scaling=True,
        early_stopping=True,
        random_state=seed,
        batch_size=2048,
        verbose=False,
    )
    _LOG.info(f"fitting PerAxisRouter axes={axes} hidden={hidden} "
              f"max_iter={max_iter} focal_gamma={focal_gamma}")
    router.fit(X, {"media_type": y_media, "playback_type": y_play})

    # seed an EntityFeaturizer for the bundle so the exported metadata records
    # the entity layout + representative seeds (runtime library is added on top).
    featurizer = _seed_featurizer(df, entity_labels, cap=64)
    # ensure the featurizer's label set matches the entity block layout exactly
    # (every column must exist as a label slot, even with no seeds).
    for label in entity_labels:
        if label not in featurizer.ner_entities:
            featurizer.register_entity(label, [f"__seed_{label}__"])

    os.makedirs(out_dir, exist_ok=True)
    for axis in axes:
        axis_dir = os.path.join(out_dir, axis)
        combiner = FeatureCombiner(vectorizer=vectorizer, featurizer=featurizer)
        export_combiner_model(router.trainers[axis], combiner, axis_dir)
        _LOG.info(f"exported {axis} -> {axis_dir} "
                  f"(temp={router.trainers[axis].temperature:.3f})")

    router_meta = {
        "axes": axes,
        "thresholds": {"media_type": 0.5, "playback_type": 0.5},
        "abstain_label": GENERIC,
        "static_cols": [c for c in STATIC_COLS if c in df.columns],
        "entity_labels": entity_labels,
        "misroute_cost": MISROUTE_COST,
        "abstain_cost": ABSTAIN_COST,
        "focal_gamma": focal_gamma,
        "train_rows": int(len(df)),
    }
    with open(os.path.join(out_dir, "router_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(router_meta, fh, indent=2)
    _LOG.info(f"wrote router bundle to {out_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/release")
    ap.add_argument("--out", default="data/models/embedding_router")
    ap.add_argument("--max-rows", type=int, default=60_000)
    ap.add_argument("--per-class-cap", type=int, default=6_000)
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--focal-gamma", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args.data, args.out, args.max_rows, args.per_class_cap, args.max_iter,
          tuple(args.hidden), args.focal_gamma, args.seed)


if __name__ == "__main__":
    main()
