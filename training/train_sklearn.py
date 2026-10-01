#!/usr/bin/env python3
"""Train a multi-task ONNX media-classifier bundle with scikit-learn.

This is the reference "train your own backend" pipeline.  It turns the canonical
``ocp-media-intents`` dataset (the precomputed 0/1 feature columns, NOT raw text)
into the **self-describing multi-head ONNX bundle** that
:meth:`ovos_media_classifier.onnx.OnnxMediaClassifier.from_path` loads unchanged.

Multi-task: one head per axis
-----------------------------
Every axis the dataset labels by construction gets its own ONNX head, so the
backend predicts each axis directly (and can soft-gate — trust an axis head even
when the leaf is uncertain) instead of deriving everything from the leaf:

  single-label heads   domain · media_type · playback_type · structure ·
                       explicitness · control_intent
  multi-label  heads   content_form_genres (adult/anime/animation/asmr — the
                       content-filter axis) · tags (namespaced genre:/mood:/era:
                       — genre/mood/era folded into one head) ·
                       qualifiers (black_and_white/silent/live/subtitled/…)

A head is **skipped** when its column is degenerate (a single class) on the
training data.  The bundle records exactly which heads it carries; the runtime
uses a head when present and derives the axis otherwise, so partial bundles (and
old 2-head ``domain``/``play`` bundles) still load.

The ladder
----------
Each head is trained on two feature sets and both are reported — the lift is the
headline result:

* **context-only** — keyword columns only (``kw_*`` / ``verb_*`` / ``mod_*`` /
  ``fmt_*`` / ``attr_*``); ``ner_*`` EXCLUDED — the "works with no registered
  entities" baseline a fresh install sees.
* **context+NER** — keyword columns **plus** the ``ner_*`` columns (the entities
  a populated NER would surface).

Several sklearn models are tried per (head, feature-set); the best by validation
macro-F1 is kept (ties broken toward the most compact model).  The exported
``feature_names`` is the exact column order the model was trained on, so the
runtime vectorizer reproduces it with no assumptions.

Run it (needs ``pip install ovos-media-classifier[train]``)::

    python -m training.train_sklearn                       # data/release → data/models
    python -m training.train_sklearn --data-dir data/release --out-dir data/models

If ``--data-dir`` has no split, the dataset is built first with
``python -m training.build_dataset``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ovos_media_classifier.constants import (
    DEFAULT_DOMAIN_THRESHOLD,
    DEFAULT_PLAY_THRESHOLD,
)
from ovos_media_classifier.features import _KEYWORD_VOCABS, VALUE_FEATURE_COLS
from ovos_media_classifier.intents import OCPDomain

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)

# Keyword (context) feature columns — the order the runtime extractor uses.
# Plain .voc presence flags + the per-value genre flags so the genre head can
# learn *which* value was named.
KEYWORD_COLS: List[str] = ([col for _voc, col in _KEYWORD_VOCABS]
                           + list(VALUE_FEATURE_COLS))

FEATURE_SETS = ("context", "context_ner")

# Default per-label threshold for the multi-label (sigmoid) heads.
DEFAULT_MULTILABEL_THRESHOLD = 0.5
# Cap the open-vocabulary ``content_genres`` head to its most frequent labels so
# it stays a tractable multi-label problem (the long tail collapses to "no tag").
CONTENT_GENRE_TOP_K = 40


# ---------------------------------------------------------------------------
# Head specs — declares each axis: its column, kind, and how to read labels.
# ---------------------------------------------------------------------------

def _json_list(v: object) -> List[str]:
    try:
        return [x for x in json.loads(v) if x]
    except Exception:
        return []


# (axis_name, column, kind)  kind in {"single", "multi"}.
# ``domain`` is special-cased (synthetic negatives); the rest read the column.
HEAD_SPECS: List[Tuple[str, str, str]] = [
    ("media_type", "media_type", "single"),
    ("playback_type", "playback_type", "single"),
    ("structure", "structure", "single"),
    ("explicitness", "explicitness", "single"),
    ("control_intent", "control_intent", "single"),
    ("content_form_genres", "content_form_genres", "multi"),
    # the mediavocab axes (the classifier emits mediavocab's own vocabulary)
    ("content_genres", "content_genres", "multi"),
    ("content_form", "content_form", "single"),
    ("programme_format", "programme_format", "single"),
    ("accessibility", "accessibility", "multi"),
    ("variant", "variant", "single"),
    ("picture_format", "picture_format", "multi"),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _read_split(data_dir: str, name: str) -> pd.DataFrame:
    pq = os.path.join(data_dir, f"{name}.parquet")
    csv = os.path.join(data_dir, f"{name}.csv")
    if os.path.isfile(pq):
        return pd.read_parquet(pq)
    if os.path.isfile(csv):
        return pd.read_csv(csv)
    raise FileNotFoundError(f"missing split: {pq} / {csv}")


def ensure_dataset(data_dir: str) -> None:
    if all(
        os.path.isfile(os.path.join(data_dir, f"{n}.parquet"))
        or os.path.isfile(os.path.join(data_dir, f"{n}.csv"))
        for n in ("train", "validation", "test")
    ):
        return
    print(f"No dataset under {data_dir}; building it …")
    subprocess.run(
        [sys.executable, "-m", "training.build_dataset", "--out-dir", data_dir],
        cwd=REPO_ROOT, check=True,
    )


def feature_columns(df: pd.DataFrame, feature_set: str) -> List[str]:
    kw = [c for c in KEYWORD_COLS if c in df.columns]
    if feature_set == "context":
        return kw
    ner = [c for c in df.columns if c.startswith("ner_")]
    return kw + ner


def _matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype="float32")


# ---------------------------------------------------------------------------
# Model zoo + selection
# ---------------------------------------------------------------------------

def _model_zoo() -> Dict[str, object]:
    """Candidate single-label classifiers tried for each (head, feature-set).

    ``LinearSVC`` is wrapped in ``CalibratedClassifierCV`` so it exposes
    ``predict_proba`` (the confidence the runtime thresholds on) and converts to
    an ONNX graph with a probability output.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB
    from sklearn.svm import LinearSVC

    return {
        "logreg": LogisticRegression(max_iter=400, C=4.0),
        "linsvc": CalibratedClassifierCV(LinearSVC(C=1.0), cv=3),
        "complement_nb": ComplementNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=120, n_jobs=-1, random_state=42),
        "hist_gbdt": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.15, random_state=42),
    }


# Prefer compact linear models on near-ties (a few-KiB logreg over a 40 MiB
# forest) so the shipped bundle stays tiny without losing accuracy.
_MODEL_PREFERENCE = ["logreg", "linsvc", "complement_nb", "hist_gbdt",
                     "random_forest"]
_TIE_EPS = 0.005


def _macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _fit_select_single(
    X_tr, y_tr, X_va, y_va,
) -> Tuple[str, object, float, float, Dict[str, Dict[str, float]]]:
    """Fit every candidate; return the best by validation macro-F1."""
    from sklearn.metrics import accuracy_score

    scores: Dict[str, Dict[str, float]] = {}
    fitted: Dict[str, object] = {}
    best_f1 = -1.0
    for name, model in _model_zoo().items():
        t0 = time.perf_counter()
        try:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
        except Exception as exc:  # noqa: BLE001
            scores[name] = {"error": str(exc)}
            continue
        f1 = _macro_f1(y_va, pred)
        acc = float(accuracy_score(y_va, pred))
        scores[name] = {"macro_f1": round(f1, 4), "accuracy": round(acc, 4),
                        "fit_predict_s": round(time.perf_counter() - t0, 1)}
        fitted[name] = model
        best_f1 = max(best_f1, f1)
    if not fitted:
        raise RuntimeError("every candidate model failed to fit")
    near = [n for n in fitted if scores[n]["macro_f1"] >= best_f1 - _TIE_EPS]
    near.sort(key=lambda n: _MODEL_PREFERENCE.index(n)
              if n in _MODEL_PREFERENCE else len(_MODEL_PREFERENCE))
    best = near[0]
    return best, fitted[best], scores[best]["macro_f1"], scores[best]["accuracy"], scores


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

def train_domain_head(train_df, val_df, cols, seed=42):
    """Binary ``ocp_play`` vs synthetic ``not_ocp`` (the empty feature vector).

    Every dataset row is ``ocp_play``; the negative class is the all-zero row
    (no keyword/NER evidence), so the head learns "any media evidence ⇒ OCP".
    """
    rng = np.random.RandomState(seed)

    def _xy(df):
        X_pos = _matrix(df, cols)
        y_pos = np.full(len(df), OCPDomain.OCP_PLAY.value, dtype=object)
        X_neg = np.zeros((len(df), len(cols)), dtype="float32")
        y_neg = np.full(len(df), OCPDomain.NOT_OCP.value, dtype=object)
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([y_pos, y_neg])
        idx = rng.permutation(len(y))
        return X[idx], y[idx]

    X_tr, y_tr = _xy(train_df)
    X_va, y_va = _xy(val_df)
    name, model, f1, acc, scores = _fit_select_single(X_tr, y_tr, X_va, y_va)
    return {"head": "domain", "kind": "single", "best_model": name,
            "val_macro_f1": round(f1, 4), "val_accuracy": round(acc, 4),
            "labels": sorted(set(y_tr)), "model": model,
            "classes": list(model.classes_), "candidates": scores}


def train_single_head(name, column, train_df, val_df, cols):
    """A single-label head over a dataset column (skip if degenerate)."""
    y_tr_all = train_df[column].astype(str).to_numpy()
    classes = sorted({c for c in y_tr_all if c != ""})
    if len(classes) < 2:
        return {"head": name, "kind": "single", "status": "skipped",
                "reason": f"degenerate column ({len(classes)} class)"}
    # rows with an empty label are not informative for this axis
    mask_tr = train_df[column].astype(str) != ""
    mask_va = val_df[column].astype(str) != ""
    X_tr = _matrix(train_df[mask_tr], cols)
    y_tr = train_df.loc[mask_tr, column].astype(str).to_numpy()
    X_va = _matrix(val_df[mask_va], cols)
    y_va = val_df.loc[mask_va, column].astype(str).to_numpy()
    if len(set(y_va)) < 2 or len(X_va) == 0:
        return {"head": name, "kind": "single", "status": "skipped",
                "reason": "no validation signal"}
    bm, model, f1, acc, scores = _fit_select_single(X_tr, y_tr, X_va, y_va)
    return {"head": name, "kind": "single", "best_model": bm,
            "val_macro_f1": round(f1, 4), "val_accuracy": round(acc, 4),
            "labels": sorted(set(y_tr)), "model": model,
            "classes": list(model.classes_), "candidates": scores,
            "n_train": int(len(X_tr))}


def train_multi_head(name, column, train_df, val_df, cols, top_k=None):
    """A multi-label (sigmoid) head over a JSON-list dataset column.

    OneVsRest logistic regression → per-label probabilities; the runtime keeps
    labels whose probability ≥ threshold.  ``top_k`` caps the label space to the
    most frequent labels (for the open-vocabulary ``content_genre``).
    """
    from collections import Counter
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    y_tr_lists = [_json_list(v) for v in train_df[column]]
    counts = Counter(g for lst in y_tr_lists for g in lst)
    if not counts:
        return {"head": name, "kind": "multi", "status": "skipped",
                "reason": "no labels present"}
    labels = [g for g, _ in counts.most_common(top_k)] if top_k \
        else sorted(counts)
    labels = sorted(labels)
    # A single-label column is degenerate as a multi-label head: a one-column
    # indicator makes OneVsRestClassifier emit a 1-D proba that the
    # multilabel-indicator scorer cannot read.  Skip it (the runtime derives the
    # axis instead), the same way a degenerate single-label head is skipped.
    if len(labels) < 2:
        return {"head": name, "kind": "multi", "status": "skipped",
                "reason": f"degenerate column ({len(labels)} label)"}
    mlb = MultiLabelBinarizer(classes=labels)

    def _Y(df):
        lists = [[g for g in _json_list(v) if g in set(labels)] for v in df[column]]
        return mlb.fit_transform(lists)

    Y_tr = _Y(train_df)
    Y_va = _Y(val_df)
    X_tr = _matrix(train_df, cols)
    X_va = _matrix(val_df, cols)

    base = LogisticRegression(max_iter=400, C=4.0)
    model = OneVsRestClassifier(base, n_jobs=-1)
    t0 = time.perf_counter()
    model.fit(X_tr, Y_tr)
    proba = model.predict_proba(X_va)
    pred = (proba >= DEFAULT_MULTILABEL_THRESHOLD).astype(int)
    macro = float(f1_score(Y_va, pred, average="macro", zero_division=0))
    micro = float(f1_score(Y_va, pred, average="micro", zero_division=0))
    return {"head": name, "kind": "multi", "best_model": "ovr_logreg",
            "val_macro_f1": round(macro, 4), "val_micro_f1": round(micro, 4),
            "labels": labels, "model": model, "classes": labels,
            "threshold": DEFAULT_MULTILABEL_THRESHOLD,
            "fit_s": round(time.perf_counter() - t0, 1),
            "n_labels": len(labels)}


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _to_onnx(model, n_features: int) -> bytes:
    """Convert a fitted sklearn classifier to a single-input ONNX graph.

    Input ``("input", FloatTensorType([None, n_features]))`` matches the runtime
    vectorizer.  ``zipmap=False`` makes the probability output a plain float
    tensor.  For multi-label ``OneVsRestClassifier`` the graph emits per-label
    probabilities (sigmoid-style), which the runtime thresholds.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    onx = convert_sklearn(
        model,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        options={id(model): {"zipmap": False}},
        target_opset=15,
    )
    return onx.SerializeToString()


def _index_map(classes) -> Dict[str, str]:
    return {str(i): str(c) for i, c in enumerate(classes)}


def export_bundle(out_dir, feature_names, feature_set, heads, extra_meta):
    """Write ``<axis>.onnx`` per trained head + ``meta.json`` into *out_dir*.

    Always writes the legacy ``play.onnx`` alias for the ``media_type`` head so
    old loaders that look for a play head keep working (back-compat).
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(feature_names)

    # Clear any stale head .onnx files from a previous run so the bundle dir only
    # ever contains the heads this run trained (a demoted head must not linger).
    for fn in os.listdir(out_dir):
        if fn.endswith(".onnx"):
            os.remove(os.path.join(out_dir, fn))

    head_meta: Dict[str, dict] = {}
    for axis, info in heads.items():
        if info.get("status") == "skipped" or "model" not in info:
            continue
        onnx_name = f"{axis}.onnx"
        with open(os.path.join(out_dir, onnx_name), "wb") as fh:
            fh.write(_to_onnx(info["model"], n))
        entry = {
            "onnx": onnx_name,
            "kind": info["kind"],
            "labels": _index_map(info["classes"]),
            "best_model": info.get("best_model"),
        }
        if info["kind"] == "multi":
            entry["threshold"] = info.get("threshold", DEFAULT_MULTILABEL_THRESHOLD)
        head_meta[axis] = entry

    # ---- back-compat: domain.onnx + play.onnx (old 2-head contract) ----
    if "domain" in head_meta:
        domain_labels = head_meta["domain"]["labels"]
    else:
        domain_labels = {}
    # play head = the media_type head; the old meta keyed it as ``play_labels``.
    play_axis = "media_type"
    play_labels = (head_meta.get(play_axis, {}).get("labels", {})
                   if play_axis in head_meta else {})
    if play_axis in head_meta:
        import shutil
        shutil.copyfile(os.path.join(out_dir, f"{play_axis}.onnx"),
                        os.path.join(out_dir, "play.onnx"))

    meta = {
        "feature_names": feature_names,
        "feature_set": feature_set,
        "heads": head_meta,
        # legacy keys so an old OnnxMediaClassifier.from_path still loads it
        "domain_labels": domain_labels,
        "play_labels": play_labels,
        "input_name": "input",
        "domain_threshold": DEFAULT_DOMAIN_THRESHOLD,
        "play_threshold": DEFAULT_PLAY_THRESHOLD,
        **extra_meta,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return out_dir


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _strip_models(heads: Dict[str, dict]) -> Dict[str, dict]:
    """A JSON-safe copy of the head reports (drop the live model objects)."""
    out = {}
    for k, v in heads.items():
        out[k] = {kk: vv for kk, vv in v.items() if kk != "model"}
    return out


def train(data_dir, out_dir, seed=42):
    ensure_dataset(data_dir)
    print(f"Loading splits from {data_dir} …")
    train_df = _read_split(data_dir, "train")
    val_df = _read_split(data_dir, "validation")
    print(f"  train={len(train_df):,}  validation={len(val_df):,}")

    summary: Dict[str, object] = {"data_dir": data_dir, "rungs": {}}
    bundles: Dict[str, str] = {}

    for feature_set in FEATURE_SETS:
        cols = feature_columns(train_df, feature_set)
        print(f"\n=== rung: {feature_set} ({len(cols)} features) ===")
        heads: Dict[str, dict] = {}

        print("  domain …")
        heads["domain"] = train_domain_head(train_df, val_df, cols, seed=seed)
        print(f"    macroF1={heads['domain']['val_macro_f1']:.4f} "
              f"({heads['domain']['best_model']})")

        for axis, column, kind in HEAD_SPECS:
            print(f"  {axis} ({kind}) …")
            if kind == "single":
                info = train_single_head(axis, column, train_df, val_df, cols)
            else:
                top_k = {"content_genres": CONTENT_GENRE_TOP_K}.get(axis)
                info = train_multi_head(axis, column, train_df, val_df, cols,
                                        top_k=top_k)
            heads[axis] = info
            if info.get("status") == "skipped":
                print(f"    skipped — {info['reason']}")
            else:
                print(f"    macroF1={info['val_macro_f1']:.4f}"
                      + (f" ({info['best_model']})" if 'best_model' in info else ""))

        bundle_dir = os.path.join(out_dir, feature_set)
        export_bundle(bundle_dir, cols, feature_set, heads, extra_meta={
            "trained_by": "training/train_sklearn.py",
            "n_train_rows": int(len(train_df)),
        })
        trained = [a for a, h in heads.items() if h.get("status") != "skipped"]
        size = sum(os.path.getsize(os.path.join(bundle_dir, f))
                   for f in os.listdir(bundle_dir))
        print(f"  → bundle {bundle_dir}  heads={trained}  ({size/1024:.0f} KiB)")
        bundles[feature_set] = bundle_dir
        summary["rungs"][feature_set] = {
            "n_features": len(cols), "bundle": bundle_dir,
            "bundle_bytes": size, "heads": _strip_models(heads),
        }

    summary["bundles"] = bundles
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train a multi-task ONNX media-classifier bundle (scikit-learn)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--data-dir", default=os.path.join("data", "release"))
    ap.add_argument("--out-dir", default=os.path.join("data", "models"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    data_dir = args.data_dir if os.path.isabs(args.data_dir) \
        else os.path.join(REPO_ROOT, args.data_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) \
        else os.path.join(REPO_ROOT, args.out_dir)

    summary = train(data_dir, out_dir, seed=args.seed)
    with open(os.path.join(out_dir, "train_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== ladder (validation macro-F1 per head) ===")
    axes = ["domain"] + [a for a, _c, _k in HEAD_SPECS]
    for fs, r in summary["rungs"].items():
        print(f"  [{fs}]")
        for a in axes:
            h = r["heads"].get(a, {})
            if h.get("status") == "skipped":
                print(f"    {a:22s} skipped ({h['reason']})")
            else:
                print(f"    {a:22s} macroF1={h.get('val_macro_f1', 0):.4f}")
    print(f"\nwrote {out_dir}/train_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
