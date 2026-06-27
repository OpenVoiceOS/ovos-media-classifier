"""Benchmark the multi-task classifier **ladder**, per axis, on the test split.

Evaluates the three rungs the project builds toward, on the real
``ocp-media-intents`` held-out test split:

    rules                → KeywordMediaClassifier (zero-ML, the default install)
    learned-context      → ONNX bundle trained on keyword features only (no NER)
    learned-context+NER  → ONNX bundle trained on keyword + ner_* features

and reports **every axis the multi-task heads predict**:

  accuracy        domain · media_type · playback_type · structure · explicitness
  macro-F1        content_form_genres · tags (genre:/mood:/era:) · qualifiers
  content filter  adult / hentai recall from the content_form_genres axis

The headline is the lift from rules → context-only → +NER across these axes.

The two ONNX bundles are loaded from ``data/models/<rung>/``; a missing bundle
is reported as ``unavailable`` rather than crashing.  Each backend predicts from
the **precomputed feature columns** in the split, so the comparison is
apples-to-apples on the identical rows.

Outputs::

    benchmarks/ladder_results.json   machine-readable
    benchmarks/ladder_results.md     the markdown tables

Run it (needs ``[onnx]``; the test split under ``data/release``)::

    python -m benchmarks.ladder
    python -m benchmarks.ladder --limit 5000      # quick smoke
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mediavocab import MediaType

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "release")
DEFAULT_MODELS_DIR = os.path.join(REPO_ROOT, "data", "models")
DEFAULT_TORCH_DIR = os.path.join(REPO_ROOT, "data", "models_torch")
DEFAULT_TEXT_DIR = os.path.join(REPO_ROOT, "data", "models_text")
RESULTS_JSON = os.path.join(HERE, "ladder_results.json")
RESULTS_MD = os.path.join(HERE, "ladder_results.md")

ADULT_GENRE = "adult"

# axis -> (dataset column, kind) for scoring.  single → accuracy + macro-F1;
# multi → multi-label macro-F1 over a JSON-list column.
SINGLE_AXES = [
    ("domain", "domain"),
    ("media_type", "media_type"),
    ("playback_type", "playback_type"),
    ("structure", "structure"),
    ("explicitness", "explicitness"),
]
MULTI_AXES = [
    ("content_form_genres", "content_form_genres"),
    # the namespaced descriptive axis — genre/mood/era folded into one head
    ("tags", "tags"),
    ("qualifiers", "qualifiers"),
]

# Capped (open-vocabulary) multi-label heads model only their top-K labels, so a
# fair macro-F1 is computed over **that modelled label space**, not over the
# thousands of distinct (mostly un-modelled, junk-tail) values present in the raw
# truth column.  Scoring uses the bundle's own label set (recorded in meta.json),
# applied uniformly across rungs so the comparison stays apples-to-apples.
CAPPED_MULTI_AXES = {"tags"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_test(data_dir: str, limit: int = 0) -> pd.DataFrame:
    pq = os.path.join(data_dir, "test.parquet")
    csv = os.path.join(data_dir, "test.csv")
    if os.path.isfile(pq):
        df = pd.read_parquet(pq)
    elif os.path.isfile(csv):
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(
            f"no test split in {data_dir}; run python -m training.build_dataset")
    if limit:
        df = df.head(limit).reset_index(drop=True)
    return df


def _json_list(raw: object) -> List[str]:
    try:
        return [g for g in json.loads(raw) if g]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)


def _macro_f1_single(truths: List[str], preds: List[str]) -> float:
    labels = sorted(set(truths) | set(preds))
    f1s = []
    for l in labels:
        tp = sum(1 for t, p in zip(truths, preds) if t == l and p == l)
        fp = sum(1 for t, p in zip(truths, preds) if t != l and p == l)
        fn = sum(1 for t, p in zip(truths, preds) if t == l and p != l)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _accuracy(truths: List[str], preds: List[str]) -> float:
    if not truths:
        return 0.0
    return sum(1 for t, p in zip(truths, preds) if t == p) / len(truths)


def _multilabel_macro_f1(truths: List[List[str]], preds: List[List[str]]) -> float:
    labels = sorted({l for row in truths for l in row}
                    | {l for row in preds for l in row})
    if not labels:
        return 0.0
    f1s = []
    for l in labels:
        tp = sum(1 for t, p in zip(truths, preds) if l in t and l in p)
        fp = sum(1 for t, p in zip(truths, preds) if l not in t and l in p)
        fn = sum(1 for t, p in zip(truths, preds) if l in t and l not in p)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

def _collect_rules(df: pd.DataFrame) -> Dict[str, object]:
    """KeywordMediaClassifier over the raw ``sentence`` — predicts each axis."""
    from ovos_media_classifier import KeywordMediaClassifier
    clf = KeywordMediaClassifier()
    pred: Dict[str, list] = {a: [] for a, _ in SINGLE_AXES}
    multi: Dict[str, list] = {a: [] for a, _ in MULTI_AXES}
    lat = []
    for sent, lang in zip(df["sentence"], df["lang"]):
        t0 = time.perf_counter()
        full = clf.classify_full(sent, lang)
        cform = clf.classify_content_form_genres(sent, lang)
        tags = clf.classify_tags(sent, lang)
        quals = clf.classify_qualifiers(sent, lang)
        expl = clf.classify_explicitness(sent, lang)
        lat.append((time.perf_counter() - t0) * 1000.0)
        pred["domain"].append(full.domain.value)
        pred["media_type"].append(full.media_type.value)
        pred["playback_type"].append(full.playback_type.value)
        pred["structure"].append(full.structure.value)
        pred["explicitness"].append(expl)
        multi["content_form_genres"].append(cform)
        multi["tags"].append(tags)
        multi["qualifiers"].append(quals)
    return {"pred": pred, "multi": multi, "lat": lat,
            "model_bytes": 0, "status": "available", "label_space": {}}


def _collect_onnx(df: pd.DataFrame, bundle_dir: str) -> Dict[str, object]:
    """Run an ONNX bundle over the test split; predict every axis.

    Two feature paths, both apples-to-apples on the same rows:

    * **categorical-only bundles** (the sklearn ladder) read the precomputed 0/1
      feature columns straight from the split — the fast path.
    * **text-bearing bundles** (the neural ``cat_text`` / ``cat_wordvec`` /
      ``cat_all`` variants) declare a char-hash / word-vector spec, so the row is
      built from the **raw utterance** through the runtime featurizer
      (``clf._row_for``) — numpy-only, exactly what production does. Latency is
      measured INCLUDING this featurization so the cost is honest.
    """
    from ovos_media_classifier.onnx import OnnxMediaClassifier

    clf = OnnxMediaClassifier.from_path(bundle_dir)
    # the runtime row is built from raw text when the bundle carries a numpy text
    # block (char-hash / word-vec) OR is a baked-vectorizer string-input pipeline
    has_text = (clf._text_spec is not None or clf._wordvec_spec is not None
                or clf._input_kind == "text")
    if not has_text:
        feat_names = clf._feature_names
        missing = [c for c in feat_names if c not in df.columns]
        if missing:
            raise ValueError(f"test split missing {len(missing)} feature cols "
                             f"(e.g. {missing[:3]})")
        X = df[feat_names].to_numpy(dtype="float32")

    pred: Dict[str, list] = {a: [] for a, _ in SINGLE_AXES}
    multi: Dict[str, list] = {a: [] for a, _ in MULTI_AXES}
    lat = []

    def _single(axis):
        h = clf._heads.get(axis)
        if h is None:
            return None
        out = h["session"].run(None, {"input": row})
        arr = next(np.asarray(o) for o in out
                   if np.asarray(o).dtype.kind in "fiu"
                   and np.asarray(o).reshape(1, -1).shape[-1] == len(h["labels"]))
        return h["labels"].get(int(np.argmax(arr.reshape(-1))), "")

    def _multi(axis):
        h = clf._heads.get(axis)
        if h is None:
            return []
        out = h["session"].run(None, {"input": row})
        prob = next((np.asarray(o) for o in out if np.asarray(o).dtype.kind == "f"
                     and np.asarray(o).reshape(1, -1).shape[-1] == len(h["labels"])),
                    None)
        if prob is None:
            return []
        v = prob.reshape(-1)
        return [h["labels"][i] for i in range(len(h["labels"]))
                if v[i] >= h["threshold"]]

    sentences = df["sentence"].astype(str).tolist()
    langs = df["lang"].astype(str).tolist()
    for i in range(len(df)):
        t0 = time.perf_counter()
        # build the feature row: precomputed columns (categorical) or the runtime
        # featurizer over the raw utterance (text-bearing bundles)
        row = (clf._row_for(sentences[i], langs[i]) if has_text
               else X[i:i + 1])
        dprob = clf._run(clf._domain_session, row)
        domain = clf._domain_labels.get(int(np.argmax(dprob)), "not_ocp")
        mt = _single("media_type") or "generic"
        pb = _single("playback_type") or "unknown"
        st = _single("structure") or "unknown"
        expl = _single("explicitness") or "clean"
        cform = _multi("content_form_genres")
        tags = _multi("tags")
        quals = _multi("qualifiers")
        lat.append((time.perf_counter() - t0) * 1000.0)
        pred["domain"].append(domain)
        pred["media_type"].append(mt)
        pred["playback_type"].append(pb)
        pred["structure"].append(st)
        pred["explicitness"].append(expl)
        multi["content_form_genres"].append(cform)
        multi["tags"].append(tags)
        multi["qualifiers"].append(quals)

    size = sum(os.path.getsize(os.path.join(bundle_dir, f))
               for f in os.listdir(bundle_dir))
    # the modelled label space for each capped multi-label head (for fair scoring)
    label_space = {axis: set(clf._heads[axis]["labels"].values())
                   for axis in CAPPED_MULTI_AXES if axis in clf._heads}
    return {"pred": pred, "multi": multi, "lat": lat,
            "model_bytes": size, "status": "available",
            "label_space": label_space}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_collected(df: pd.DataFrame, c: Dict[str, object]) -> Dict[str, object]:
    axes: Dict[str, dict] = {}
    for axis, col in SINGLE_AXES:
        truths = df[col].astype(str).tolist()
        preds = [str(x) for x in c["pred"][axis]]
        # only score rows that HAVE a label for that axis (mood/era are sparse)
        pairs = [(t, p) for t, p in zip(truths, preds) if t != ""]
        if not pairs:
            axes[axis] = {"n": 0, "accuracy": None, "macro_f1": None}
            continue
        t2, p2 = zip(*pairs)
        axes[axis] = {"n": len(pairs),
                      "accuracy": round(_accuracy(list(t2), list(p2)), 4),
                      "macro_f1": round(_macro_f1_single(list(t2), list(p2)), 4)}
    label_space = c.get("label_space", {})
    for axis, col in MULTI_AXES:
        truths = [_json_list(v) for v in df[col]]
        preds = c["multi"][axis]
        space = label_space.get(axis)
        if axis in CAPPED_MULTI_AXES and space:
            # score only over the head's modelled label space (the in-scope task)
            truths = [[g for g in row if g in space] for row in truths]
            preds = [[g for g in row if g in space] for row in preds]
        axes[axis] = {
            "macro_f1": round(_multilabel_macro_f1(truths, preds), 4),
            "labelled_rows": sum(1 for t in truths if t),
        }
        if axis in CAPPED_MULTI_AXES:
            axes[axis]["label_space_size"] = len(space) if space else None
    lat = c["lat"]
    return {
        "axes": axes,
        "content_filter": _cf_recall(df, c["multi"]["content_form_genres"]),
        "latency_ms_median": round(_percentile(lat, 0.5), 5),
        "latency_ms_p95": round(_percentile(lat, 0.95), 5),
        "model_bytes": c["model_bytes"],
        "status": "available",
    }


def _cf_recall(df: pd.DataFrame, form_preds: List[List[str]]) -> Dict[str, object]:
    """Adult/hentai recall from the content_form_genres axis + false-block rate."""
    from ovos_media_classifier import ContentFilter
    cf = ContentFilter()
    truth = [_json_list(v) for v in df["content_form_genres"]]
    intents = list(df["intent"]) if "intent" in df.columns else [""] * len(df)

    def blk(g):
        b, _ = cf.is_blocked(MediaType.GENERIC, g)
        return b

    a_tot = a_blk = h_tot = h_blk = na_tot = na_blk = 0
    for tg, pg, it in zip(truth, form_preds, intents):
        if ADULT_GENRE in tg:
            a_tot += 1
            if blk(pg):
                a_blk += 1
            if it == "hentai":
                h_tot += 1
                if blk(pg):
                    h_blk += 1
        else:
            na_tot += 1
            if blk(pg):
                na_blk += 1
    return {
        "adult_rows": a_tot, "adult_blocked": a_blk,
        "recall": round(a_blk / a_tot, 4) if a_tot else 0.0,
        "hentai_rows": h_tot, "hentai_blocked": h_blk,
        "hentai_recall": round(h_blk / h_tot, 4) if h_tot else 0.0,
        "non_adult_rows": na_tot, "non_adult_blocked": na_blk,
        "false_block_rate": round(na_blk / na_tot, 4) if na_tot else 0.0,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# The full ladder: rules → sklearn (categorical) → neural × feature sets.  Each
# entry is ``(display_name, bundle_dir_or_None)``; ``None`` is the rules backend.
# sklearn bundles live under ``--models-dir`` (data/models); the neural variants
# under ``--torch-dir`` (data/models_torch).  Missing bundles report
# ``unavailable`` rather than crash.
SKLEARN_RUNGS = [
    ("rules", None),
    ("sklearn context", "context"),
    ("sklearn context+NER", "context_ner"),
]
# self-contained TF-IDF→clf string-input pipelines (vectorizer baked in-graph) —
# the classic bag-of-n-grams representations + a linear/NB classifier
TEXT_RUNGS = [
    ("tfidf word(1,2)+linear", "tfidf_word12"),
    ("tfidf word(1,3)+linear", "tfidf_word13"),
    ("tfidf char(3,5)+linear", "tfidf_char35"),
]
# neural variant dir → display name (the head-to-head feature-set comparison)
TORCH_RUNGS = [
    ("neural cat", "cat"),
    ("neural cat+text(char-hash)", "cat_text"),
    ("neural cat+wordvec(skip)", "cat_wordvec"),
    ("neural cat+wordvec(cbow)", "cat_wordvec_cbow"),
    ("neural cat+all", "cat_all"),
    ("neural cat+all (deep)", "cat_all_deep"),
    ("neural cat+all (wide)", "cat_all_wide"),
]


def _ladder(models_dir: str, torch_dir: str, text_dir: str):
    """Resolve the rung list to ``(name, bundle_dir_or_None)`` absolute paths."""
    rungs = []
    for name, fs in SKLEARN_RUNGS:
        rungs.append((name, None if fs is None else os.path.join(models_dir, fs)))
    for name, variant in TEXT_RUNGS:
        rungs.append((name, os.path.join(text_dir, variant)))
    for name, variant in TORCH_RUNGS:
        rungs.append((name, os.path.join(torch_dir, variant)))
    return rungs


def run(data_dir: str, models_dir: str, torch_dir: str, text_dir: str,
        limit: int = 0) -> Dict[str, object]:
    df = load_test(data_dir, limit=limit)
    ladder = _ladder(models_dir, torch_dir, text_dir)

    # Collect raw predictions first so the capped multi-label label space (from a
    # trained bundle's meta) can be shared with the rules rung for a fair,
    # apples-to-apples macro-F1 on the in-scope task (see CAPPED_MULTI_AXES).
    collected: Dict[str, Dict[str, object]] = {}
    for name, bundle in ladder:
        if bundle is None:
            print(f"[{name}] rules backend on {len(df):,} rows …")
            collected[name] = _collect_rules(df)
            continue
        if not os.path.isfile(os.path.join(bundle, "meta.json")):
            collected[name] = {"status": "unavailable",
                               "reason": f"no bundle at {bundle}; "
                                         f"run the matching trainer"}
            print(f"[{name}] unavailable (no bundle)")
            continue
        print(f"[{name}] ONNX bundle {bundle} on {len(df):,} rows …")
        try:
            collected[name] = _collect_onnx(df, bundle)
        except Exception as e:  # noqa: BLE001
            collected[name] = {"status": "unavailable",
                               "reason": f"{type(e).__name__}: {e}"}
            print(f"  failed: {e}")

    shared_space: Dict[str, set] = {}
    for c in collected.values():
        for axis, sp in (c.get("label_space") or {}).items():
            if sp and len(sp) > len(shared_space.get(axis, set())):
                shared_space[axis] = sp

    rungs: Dict[str, object] = {}
    for name, c in collected.items():
        if c.get("status") == "unavailable":
            rungs[name] = c
            continue
        c.setdefault("label_space", {})
        for axis, sp in shared_space.items():
            c["label_space"].setdefault(axis, sp)
        rungs[name] = _score_collected(df, c)
    return {"n_test_rows": len(df), "data_dir": data_dir,
            "models_dir": models_dir, "torch_dir": torch_dir,
            "text_dir": text_dir,
            "rung_order": [n for n, _ in ladder], "rungs": rungs}


def write_json(report, path: str = RESULTS_JSON) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return path


def _fmt_bytes(n: int) -> str:
    if not n:
        return "—"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KiB"
    return f"{n / (1024 * 1024):.1f} MiB"


def write_md(report, path: str = RESULTS_MD) -> str:
    order = report.get("rung_order") or [n for n, _ in SKLEARN_RUNGS]
    avail = [(n, report["rungs"][n]) for n in order
             if report["rungs"].get(n, {}).get("status") == "available"]
    L: List[str] = []
    L.append("# ovos-media-classifier — the multi-task ladder\n")
    L.append(f"Held-out **test split**: {report['n_test_rows']:,} utterances. "
             "The ladder runs **rules → sklearn (categorical) → neural × feature "
             "set** (categorical → +char-hash text → +domain word-vectors → all) "
             "on the SAME rows. Neural rungs build their text features from the "
             "raw utterance at inference (numpy only); latency is measured "
             "including that featurization.\n")

    # per-axis accuracy table (single-label axes)
    L.append("## Single-label axes — accuracy\n")
    L.append("| axis | " + " | ".join(n for n, _ in avail) + " |")
    L.append("|" + "---|" * (1 + len(avail)))
    for axis, _col in SINGLE_AXES:
        cells = []
        for _n, b in avail:
            m = b["axes"].get(axis, {})
            acc = m.get("accuracy")
            cells.append(f"{acc:.3f}" if acc is not None else "–")
        L.append(f"| {axis} | " + " | ".join(cells) + " |")
    L.append("")

    # multi-label axes — macro-F1
    L.append("## Multi-label axes — macro-F1\n")
    L.append("| axis | " + " | ".join(n for n, _ in avail) + " |")
    L.append("|" + "---|" * (1 + len(avail)))
    for axis, _col in MULTI_AXES:
        cells = [f"{b['axes'].get(axis, {}).get('macro_f1', 0):.3f}"
                 for _n, b in avail]
        L.append(f"| {axis} | " + " | ".join(cells) + " |")
    L.append("")

    # content filter
    L.append("## Content filter (from the content_form_genres axis)\n")
    L.append("| rung | adult recall | hentai recall | false-block | "
             "median ms | p95 ms | size |")
    L.append("|---|---|---|---|---|---|---|")
    for n, b in avail:
        cf = b["content_filter"]
        L.append(f"| {n} | {cf['recall']:.3f} ({cf['adult_blocked']}/{cf['adult_rows']}) "
                 f"| {cf['hentai_recall']:.3f} | {cf['false_block_rate']:.3f} | "
                 f"{b['latency_ms_median']:.4f} | {b['latency_ms_p95']:.4f} | "
                 f"{_fmt_bytes(b['model_bytes'])} |")
    L.append("")

    unavailable = [n for n in order
                   if report["rungs"].get(n, {}).get("status") != "available"]
    if unavailable:
        L.append("## Unavailable rungs\n")
        for n in unavailable:
            L.append(f"- **{n}**: {report['rungs'][n].get('reason', '?')}")
        L.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--torch-dir", default=DEFAULT_TORCH_DIR)
    ap.add_argument("--text-dir", default=DEFAULT_TEXT_DIR)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    report = run(args.data_dir, args.models_dir, args.torch_dir, args.text_dir,
                 limit=args.limit)
    jp = write_json(report)
    mp = write_md(report)
    print()
    for n in report.get("rung_order", []):
        b = report["rungs"].get(n, {})
        if b.get("status") != "available":
            print(f"  {n:24s} unavailable")
            continue
        ax = b["axes"]
        print(f"  {n:24s} media_type={ax['media_type']['accuracy']} "
              f"form_genres_F1={ax['content_form_genres']['macro_f1']} "
              f"CFrecall={b['content_filter']['recall']}")
    print(f"wrote {jp}\nwrote {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
