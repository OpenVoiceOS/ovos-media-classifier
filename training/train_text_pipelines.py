#!/usr/bin/env python3
"""Train **self-contained text-pipeline** ONNX bundles (TF-IDF → linear/NB).

The categorical / char-hash / word-vector trainers all build a numeric feature
row that the runtime must reproduce.  This trainer takes the opposite, fully
self-describing route for the classic **bag-of-n-grams** representations: a
``sklearn`` ``Pipeline(TfidfVectorizer → classifier)`` is exported **whole** with
``skl2onnx``, so the vectorizer is **baked into the ONNX graph**.  The bundle's
input is the **raw utterance string** ``(1, 1)`` — there is *no* runtime
featurization code at all; onnxruntime tokenizes + tf-idf-weights in-graph.

Representations (each a variant directory under ``--out-dir``):

* ``tfidf_word12`` — TF-IDF **word** 1–2 grams
* ``tfidf_word13`` — TF-IDF **word** 1–3 grams
* ``tfidf_char35`` — TF-IDF **char** 3–5 grams (``analyzer='char'``)

``skl2onnx`` converts ``analyzer in {'word', 'char'}`` (char-ngrams tokenize on
``[^\n]`` in-graph and round-trip exactly); only ``char_wb`` (the word-boundary
char variant) is *not* implemented by the converter, so we use plain ``char`` —
the runtime behaviour is identical to scikit-learn (verified parity ≈ 0).

Classifiers tried per (variant, head), best by validation macro-F1, ONNX-clean:
``LogisticRegression`` · ``ComplementNB`` · ``LinearSVC`` (calibrated for proba).

The bundle is the **same self-describing format** the runtime already loads —
``<axis>.onnx`` + ``meta.json`` — with ``"input_kind": "text"`` so
:class:`~ovos_media_classifier.onnx.OnnxMediaClassifier` feeds the raw string.

Run (needs ``pip install ovos-media-classifier[train]``)::

    python -m training.train_text_pipelines              # → data/models_text/<variant>/
    python -m training.train_text_pipelines --variants tfidf_word12
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import Dict, List, Optional

import numpy as np

from ovos_media_classifier.constants import (
    DEFAULT_DOMAIN_THRESHOLD,
    DEFAULT_PLAY_THRESHOLD,
)
from ovos_media_classifier.intents import OCPDomain
from training.train_sklearn import (
    CONTENT_GENRE_TOP_K,
    DEFAULT_MULTILABEL_THRESHOLD,
    HEAD_SPECS,
    TAGS_TOP_K,
    _json_list,
    _read_split,
    ensure_dataset,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)

VARIANTS: Dict[str, dict] = {
    "tfidf_word12": {"analyzer": "word", "ngram_range": (1, 2)},
    "tfidf_word13": {"analyzer": "word", "ngram_range": (1, 3)},
    # char 3–5 grams — usually the strongest text feature for short noisy
    # utterances; ``char`` (not ``char_wb``) so skl2onnx can bake it in-graph.
    "tfidf_char35": {"analyzer": "char", "ngram_range": (3, 5)},
}


def _vectorizer(cfg):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # char n-grams need a bigger cap (a 3–5 char space is larger than word 1–2);
    # the cap keeps the baked-in vocabulary — and the ONNX graph — bounded.
    max_features = 60000 if cfg["analyzer"] == "char" else 40000
    return TfidfVectorizer(analyzer=cfg["analyzer"],
                           ngram_range=tuple(cfg["ngram_range"]),
                           min_df=2, max_features=max_features, lowercase=True)


def _classifier_zoo():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB
    from sklearn.svm import LinearSVC
    return {
        "logreg": lambda: LogisticRegression(max_iter=400, C=4.0),
        "complement_nb": lambda: ComplementNB(),
        "linsvc": lambda: CalibratedClassifierCV(LinearSVC(C=1.0), cv=3),
    }


def _pipeline(cfg, clf_factory):
    from sklearn.pipeline import Pipeline
    return Pipeline([("tfidf", _vectorizer(cfg)), ("clf", clf_factory())])


def _macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Heads — each trained on the raw ``sentence`` text
# ---------------------------------------------------------------------------

def train_domain_head(cfg, train_df, val_df, seed=42):
    """ocp_play vs a synthetic empty-string negative (no media text ⇒ not_ocp)."""
    rng = np.random.RandomState(seed)

    def _xy(df):
        pos = list(df["sentence"].astype(str))
        ypos = [OCPDomain.OCP_PLAY.value] * len(pos)
        neg = [""] * len(df)
        yneg = [OCPDomain.NOT_OCP.value] * len(neg)
        X = np.array(pos + neg, dtype=object)
        y = np.array(ypos + yneg, dtype=object)
        idx = rng.permutation(len(y))
        return X[idx], y[idx]

    Xtr, ytr = _xy(train_df)
    Xva, yva = _xy(val_df)
    return _select_single("domain", cfg, Xtr, ytr, Xva, yva)


def _select_single(name, cfg, Xtr, ytr, Xva, yva):
    best = None
    scores = {}
    for cname, factory in _classifier_zoo().items():
        try:
            pipe = _pipeline(cfg, factory)
            pipe.fit(list(Xtr), ytr)
            pred = pipe.predict(list(Xva))
        except Exception as exc:  # noqa: BLE001
            scores[cname] = {"error": str(exc)[:120]}
            continue
        f1 = _macro_f1(yva, pred)
        scores[cname] = {"macro_f1": round(f1, 4)}
        if best is None or f1 > best[2]:
            best = (cname, pipe, f1)
    if best is None:
        return {"head": name, "kind": "single", "status": "skipped",
                "reason": "all classifiers failed"}
    cname, pipe, f1 = best
    return {"head": name, "kind": "single", "best_model": cname,
            "val_macro_f1": round(f1, 4), "model": pipe,
            "classes": list(pipe.classes_), "candidates": scores}


def train_single_head(name, column, cfg, train_df, val_df):
    ytr_all = train_df[column].astype(str)
    classes = sorted({c for c in ytr_all.unique() if c != ""})
    if len(classes) < 2:
        return {"head": name, "kind": "single", "status": "skipped",
                "reason": f"degenerate ({len(classes)} class)"}
    mtr = ytr_all != ""
    mva = val_df[column].astype(str) != ""
    Xtr = train_df.loc[mtr, "sentence"].astype(str).to_numpy()
    ytr = train_df.loc[mtr, column].astype(str).to_numpy()
    Xva = val_df.loc[mva, "sentence"].astype(str).to_numpy()
    yva = val_df.loc[mva, column].astype(str).to_numpy()
    if len(set(yva)) < 2 or len(Xva) == 0:
        return {"head": name, "kind": "single", "status": "skipped",
                "reason": "no validation signal"}
    return _select_single(name, cfg, Xtr, ytr, Xva, yva)


def train_multi_head(name, column, cfg, train_df, val_df, top_k=None):
    from collections import Counter
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MultiLabelBinarizer

    lists = [_json_list(v) for v in train_df[column]]
    counts = Counter(g for lst in lists for g in lst)
    if not counts:
        return {"head": name, "kind": "multi", "status": "skipped",
                "reason": "no labels"}
    labels = sorted([g for g, _ in counts.most_common(top_k)] if top_k
                    else sorted(counts))
    lset = set(labels)
    mlb = MultiLabelBinarizer(classes=labels)

    def _Y(df):
        return mlb.fit_transform([[g for g in _json_list(v) if g in lset]
                                  for v in df[column]])

    Ytr, Yva = _Y(train_df), _Y(val_df)
    Xtr = train_df["sentence"].astype(str).tolist()
    Xva = val_df["sentence"].astype(str).tolist()
    pipe = Pipeline([("tfidf", _vectorizer(cfg)),
                     ("clf", OneVsRestClassifier(
                         LogisticRegression(max_iter=400, C=4.0), n_jobs=-1))])
    t0 = time.perf_counter()
    pipe.fit(Xtr, Ytr)
    proba = pipe.predict_proba(Xva)
    pred = (proba >= DEFAULT_MULTILABEL_THRESHOLD).astype(int)
    macro = float(f1_score(Yva, pred, average="macro", zero_division=0))
    return {"head": name, "kind": "multi", "best_model": "ovr_logreg",
            "val_macro_f1": round(macro, 4), "labels": labels, "model": pipe,
            "classes": labels, "threshold": DEFAULT_MULTILABEL_THRESHOLD,
            "fit_s": round(time.perf_counter() - t0, 1)}


# ---------------------------------------------------------------------------
# ONNX export — the whole pipeline (vectorizer baked in) takes string input
# ---------------------------------------------------------------------------

def _to_onnx(pipeline) -> bytes:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
    onx = convert_sklearn(
        pipeline, initial_types=[("input", StringTensorType([None, 1]))],
        options={id(pipeline): {"zipmap": False}}, target_opset=15)
    return onx.SerializeToString()


def _index_map(classes):
    return {str(i): str(c) for i, c in enumerate(classes)}


def export_bundle(out_dir, heads, extra_meta):
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(out_dir):
        if fn.endswith(".onnx"):
            os.remove(os.path.join(out_dir, fn))

    head_meta: Dict[str, dict] = {}
    for axis, info in heads.items():
        if info.get("status") == "skipped" or "model" not in info:
            continue
        with open(os.path.join(out_dir, f"{axis}.onnx"), "wb") as fh:
            fh.write(_to_onnx(info["model"]))
        entry = {"onnx": f"{axis}.onnx", "kind": info["kind"],
                 "labels": _index_map(info["classes"]),
                 "best_model": info.get("best_model")}
        if info["kind"] == "multi":
            entry["threshold"] = info.get("threshold", DEFAULT_MULTILABEL_THRESHOLD)
        head_meta[axis] = entry

    domain_labels = head_meta.get("domain", {}).get("labels", {})
    play_labels = head_meta.get("media_type", {}).get("labels", {})
    if "media_type" in head_meta:
        shutil.copyfile(os.path.join(out_dir, "media_type.onnx"),
                        os.path.join(out_dir, "play.onnx"))

    meta = {
        "input_kind": "text",          # the graph takes the raw utterance string
        "feature_names": [],           # none — the vectorizer is in-graph
        "heads": head_meta,
        "domain_labels": domain_labels, "play_labels": play_labels,
        "input_name": "input",
        "domain_threshold": DEFAULT_DOMAIN_THRESHOLD,
        "play_threshold": DEFAULT_PLAY_THRESHOLD,
        "trained_by": "training/train_text_pipelines.py",
        **(extra_meta or {}),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return out_dir


def _strip_models(heads):
    return {k: {kk: vv for kk, vv in v.items() if kk != "model"}
            for k, v in heads.items()}


def run_variant(name, cfg, train_df, val_df, out_root, seed):
    print(f"\n=== variant: {name}  {cfg} ===")
    heads: Dict[str, dict] = {}
    heads["domain"] = train_domain_head(cfg, train_df, val_df, seed=seed)
    print(f"  domain macroF1={heads['domain'].get('val_macro_f1')} "
          f"({heads['domain'].get('best_model')})")
    for axis, column, kind in HEAD_SPECS:
        if kind == "single":
            info = train_single_head(axis, column, cfg, train_df, val_df)
        else:
            top_k = {"tags": TAGS_TOP_K, "content_genre": CONTENT_GENRE_TOP_K}.get(axis)
            info = train_multi_head(axis, column, cfg, train_df, val_df, top_k=top_k)
        heads[axis] = info
        if info.get("status") == "skipped":
            print(f"  {axis}: skipped — {info['reason']}")
        else:
            print(f"  {axis} macroF1={info['val_macro_f1']} "
                  f"({info.get('best_model', 'ovr_logreg')})")

    out_dir = os.path.join(out_root, name)
    export_bundle(out_dir, heads, extra_meta={"variant": name, "tfidf": cfg})
    size = sum(os.path.getsize(os.path.join(out_dir, f))
               for f in os.listdir(out_dir))
    print(f"  → bundle {out_dir}  ({size/1024/1024:.1f} MiB)")
    return {"variant": name, "tfidf": cfg, "bundle": out_dir,
            "bundle_bytes": size, "heads": _strip_models(heads)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Train self-contained TF-IDF→clf ONNX text-pipeline bundles",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--data-dir", default=os.path.join("data", "release"))
    ap.add_argument("--out-dir", default=os.path.join("data", "models_text"))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)

    data_dir, out_dir = _abs(args.data_dir), _abs(args.out_dir)
    ensure_dataset(data_dir)
    train_df = _read_split(data_dir, "train")
    val_df = _read_split(data_dir, "validation")
    print(f"train={len(train_df):,}  validation={len(val_df):,}")

    summary = {"data_dir": data_dir, "variants": {}}
    for name in args.variants:
        if name not in VARIANTS:
            print(f"unknown variant {name!r}; skipping")
            continue
        t0 = time.perf_counter()
        rep = run_variant(name, VARIANTS[name], train_df, val_df, out_dir, args.seed)
        rep["train_s"] = round(time.perf_counter() - t0, 1)
        summary["variants"][name] = rep

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train_report.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out_dir}/train_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
