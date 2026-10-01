"""Run all AVAILABLE classifier backends over the eval set and report metrics.

Computes, per backend:
  * accuracy (over mediavocab MediaType)
  * macro-F1 (unweighted mean of per-type F1)
  * per-type precision / recall / F1 / support
  * median + p95 inference latency (ms) and rows/sec
and, globally, content-filter recall over the adult-genre slice.

Backends that need optional deps or trained model files unavailable in this
environment are detected and recorded as ``"unavailable"`` (with the reason) —
they never crash the run.  The zero-dependency keyword backend is always
available.

Outputs::

    benchmarks/results.json   machine-readable
    benchmarks/results.md     human-readable table

Usage::

    python -m benchmarks.run            # metrics only
    python -m benchmarks.run --plots    # also (re)generate docs/benchmarks/*.png
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

from benchmarks.dataset import (
    EVAL_CSV,
    EvalRow,
    load_eval_csv,
    summary as dataset_summary,
    write_eval_csv,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RESULTS_JSON = os.path.join(HERE, "results.json")
RESULTS_MD = os.path.join(HERE, "results.md")


# ---------------------------------------------------------------------------
# Backend registry
#
# Each entry is a zero-arg loader returning an AbstractMediaClassifier, or
# raising if the backend is unavailable (missing dep / missing model file).
# Only the keyword backend is expected to be available in a bare checkout.
# ---------------------------------------------------------------------------

def _load_keyword():
    from ovos_media_classifier import KeywordMediaClassifier
    return KeywordMediaClassifier()


def _load_ahocorasick():
    # The Ahocorasick backend is a NER/exact-match classifier; without a
    # populated entity container or wordlist it has nothing to match on, so we
    # seed it from the same bundled keyword vocs to give a meaningful run.
    import os as _os
    from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
    from ovos_media_classifier.intents import OCPEntityLabel
    from benchmarks.dataset import LOCALE_DIR

    # Map each keyword voc to the closest *_KEYWORD entity label so hits resolve
    # to the right MediaType via NER_LABEL_TO_MEDIA_TYPE.
    voc_to_label = {
        "MusicKeyword": OCPEntityLabel.MUSIC_KEYWORD,
        "PodcastKeyword": OCPEntityLabel.PODCAST_KEYWORD,
        "RadioKeyword": OCPEntityLabel.RADIO_KEYWORD,
        "AudioBookKeyword": OCPEntityLabel.AUDIOBOOK_KEYWORD,
        "NewsKeyword": OCPEntityLabel.NEWS_KEYWORD,
        "MovieKeyword": OCPEntityLabel.MOVIE_KEYWORD,
        "TVKeyword": OCPEntityLabel.TV_KEYWORD,
        "IPTVKeyword": OCPEntityLabel.TV_KEYWORD,
        "SeriesKeyword": OCPEntityLabel.VIDEO_EPISODES_KEYWORD,
        "VideoKeyword": OCPEntityLabel.VIDEO_KEYWORD,
        "AudioKeyword": OCPEntityLabel.AUDIO_KEYWORD,
        "GameKeyword": OCPEntityLabel.GAME_KEYWORD,
        "AnimeKeyword": OCPEntityLabel.ANIME_KEYWORD,
        "CartoonKeyword": OCPEntityLabel.CARTOON_KEYWORD,
        "DocumentaryKeyword": OCPEntityLabel.DOCUMENTARY_KEYWORD,
        "TrailerKeyword": OCPEntityLabel.TRAILER_KEYWORD,
        "BehindTheScenesKeyword": OCPEntityLabel.BEHIND_THE_SCENES_KEYWORD,
        "ComicBookKeyword": OCPEntityLabel.VISUAL_STORY_KEYWORD,
        "ADKeyword": OCPEntityLabel.AUDIO_DESCRIPTION_KEYWORD,
        "ASMRKeyword": OCPEntityLabel.ASMR_KEYWORD,
        "AdultKeyword": OCPEntityLabel.ADULT_KEYWORD,
        "HentaiKeyword": OCPEntityLabel.HENTAI_KEYWORD,
        "MusicVideoKeyword": OCPEntityLabel.MUSIC_VIDEO_KEYWORD,
        "AudioDramaKeyword": OCPEntityLabel.RADIO_THEATRE_KEYWORD,
    }
    wordlists: Dict[str, List[str]] = {}
    en_dir = None
    for entry in os.listdir(LOCALE_DIR):
        if entry.lower() == "en-us":
            en_dir = os.path.join(LOCALE_DIR, entry)
            break
    if not en_dir:
        raise RuntimeError("en-us locale not found")
    for voc, label in voc_to_label.items():
        path = _os.path.join(en_dir, f"{voc}.voc")
        if _os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                words = [l.strip().lower() for l in fh if l.strip() and not l.startswith("#")]
            if words:
                wordlists.setdefault(label.value, []).extend(words)
    return AhocorasickMediaClassifier.from_wordlists(wordlists)


def _load_sklearn():
    model = os.environ.get("MEDIA_CLF_SKLEARN_MODEL")
    if not model or not os.path.isfile(model):
        raise FileNotFoundError(
            "no sklearn model (set MEDIA_CLF_SKLEARN_MODEL to a trained .joblib)"
        )
    from ovos_media_classifier.sklearn import SklearnMediaClassifier
    return SklearnMediaClassifier.from_path(model)


def _load_padatious():
    pdir = os.environ.get("MEDIA_CLF_PADATIOUS_DIR")
    if not pdir or not os.path.isdir(pdir):
        raise FileNotFoundError(
            "no padatious locale dir (set MEDIA_CLF_PADATIOUS_DIR); "
            "also needs the 'padatious' package"
        )
    from ovos_media_classifier.padatious import PadatiousMediaClassifier
    return PadatiousMediaClassifier.from_locale_dir(pdir, lang="en-us")


def _load_m2v():
    model = os.environ.get("MEDIA_CLF_M2V_MODEL")
    if not model or not os.path.exists(model):
        raise FileNotFoundError(
            "no Model2Vec model (set MEDIA_CLF_M2V_MODEL); also needs 'model2vec'"
        )
    from ovos_media_classifier.m2v import Model2VecMediaClassifier
    return Model2VecMediaClassifier.from_path(model)


def _load_guided():
    model = os.environ.get("MEDIA_CLF_GUIDED_MODEL")
    if not model or not os.path.exists(model):
        raise FileNotFoundError(
            "no GuidedEmbeddings ONNX model (set MEDIA_CLF_GUIDED_MODEL)"
        )
    from ovos_media_classifier.guided import GuidedEmbeddingsMediaClassifier
    return GuidedEmbeddingsMediaClassifier.from_path(model)


BACKENDS: Dict[str, Callable] = {
    "keyword": _load_keyword,
    "ahocorasick": _load_ahocorasick,
    "sklearn": _load_sklearn,
    "padatious": _load_padatious,
    "model2vec": _load_m2v,
    "guided_onnx": _load_guided,
}


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
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def evaluate_backend(clf, rows: List[EvalRow]) -> Dict[str, object]:
    """Run a classifier over all rows and compute metrics."""
    labels = sorted({r.expected_media_type for r in rows})
    label_idx = {l: i for i, l in enumerate(labels)}

    preds: List[str] = []
    truths: List[str] = []
    latencies_ms: List[float] = []

    for r in rows:
        t0 = time.perf_counter()
        mt, _ = clf.classify(r.utterance, r.lang)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        preds.append(mt.value if hasattr(mt, "value") else str(mt))
        truths.append(r.expected_media_type)

    # confusion: predictions may include types outside the eval label set
    pred_labels = sorted(set(preds) | set(labels))
    pidx = {l: i for i, l in enumerate(pred_labels)}
    confusion = [[0] * len(pred_labels) for _ in labels]

    tp: Dict[str, int] = {l: 0 for l in labels}
    fp: Dict[str, int] = {l: 0 for l in labels}
    fn: Dict[str, int] = {l: 0 for l in labels}
    support: Dict[str, int] = {l: 0 for l in labels}

    correct = 0
    for tr, pr in zip(truths, preds):
        support[tr] += 1
        confusion[label_idx[tr]][pidx[pr]] += 1
        if tr == pr:
            correct += 1
            tp[tr] += 1
        else:
            fn[tr] += 1
            if pr in fp:
                fp[pr] += 1

    per_type: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for l in labels:
        prec, rec, f1 = _prf(tp[l], fp[l], fn[l])
        per_type[l] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support[l],
        }
        f1s.append(f1)

    accuracy = correct / len(rows) if rows else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    total_s = sum(latencies_ms) / 1000.0

    return {
        "n": len(rows),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "latency_ms_median": round(_percentile(latencies_ms, 0.5), 4),
        "latency_ms_p95": round(_percentile(latencies_ms, 0.95), 4),
        "rows_per_sec": round(len(rows) / total_s, 1) if total_s else 0.0,
        "per_type": per_type,
        "labels": labels,
        "pred_labels": pred_labels,
        "confusion": confusion,
    }


def content_filter_recall(clf, rows: List[EvalRow]) -> Dict[str, object]:
    """Of the adult-genre rows, what fraction does the default ContentFilter block?"""
    from ovos_media_classifier import ContentFilter
    cf = ContentFilter()  # default policy: blocks 'adult'
    adult = [r for r in rows if "adult" in r.expected_genres]
    blocked = 0
    for r in adult:
        is_blocked, _reason = cf.check(clf, r.utterance, r.lang)
        if is_blocked:
            blocked += 1
    # false-positive check: how many non-adult rows get blocked?
    non_adult = [r for r in rows if "adult" not in r.expected_genres]
    fp = 0
    for r in non_adult:
        is_blocked, _ = cf.check(clf, r.utterance, r.lang)
        if is_blocked:
            fp += 1
    return {
        "adult_rows": len(adult),
        "adult_blocked": blocked,
        "recall": round(blocked / len(adult), 4) if adult else 0.0,
        "non_adult_rows": len(non_adult),
        "non_adult_blocked": fp,
        "false_block_rate": round(fp / len(non_adult), 4) if non_adult else 0.0,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_hf_test_rows(repo: str, split: str = "test", limit: int = 0) -> List[EvalRow]:
    """Load eval rows from the published HF dataset's test split.

    Maps the canonical schema (``sentence``/``mediavocab_type``/``genres``) onto
    :class:`EvalRow`, so the benchmark reports numbers on the *real* held-out
    test set instead of the small voc-derived eval set.
    """
    from datasets import load_dataset
    ds = load_dataset(repo, split=split)
    rows: List[EvalRow] = []
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        genres = [g for g in str(r.get("genres") or "").split(";") if g]
        rows.append(EvalRow(
            utterance=r["sentence"], lang=r.get("lang", "en-us"),
            expected_media_type=r["mediavocab_type"], expected_genres=genres,
        ))
    return rows


def run(rebuild_dataset: bool = False, hf_dataset: Optional[str] = None,
        hf_limit: int = 0) -> Dict[str, object]:
    if hf_dataset:
        rows = load_hf_test_rows(hf_dataset, limit=hf_limit)
    else:
        if rebuild_dataset or not os.path.isfile(EVAL_CSV):
            write_eval_csv()
        rows = load_eval_csv()
    ds = dataset_summary(rows)

    backends: Dict[str, object] = {}
    available: List[str] = []
    unavailable: List[str] = []

    for name, loader in BACKENDS.items():
        try:
            clf = loader()
        except Exception as e:  # noqa: BLE001 - record, never crash
            backends[name] = {"status": "unavailable", "reason": f"{type(e).__name__}: {e}"}
            unavailable.append(name)
            continue
        metrics = evaluate_backend(clf, rows)
        metrics["status"] = "available"
        metrics["content_filter"] = content_filter_recall(clf, rows)
        backends[name] = metrics
        available.append(name)

    return {
        "dataset": ds,
        "available_backends": available,
        "unavailable_backends": unavailable,
        "backends": backends,
    }


def write_json(report: Dict[str, object], path: str = RESULTS_JSON) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)
    return path


def write_md(report: Dict[str, object], path: str = RESULTS_MD) -> str:
    ds = report["dataset"]
    lines: List[str] = []
    lines.append("# ovos-media-classifier — benchmark results\n")
    lines.append(
        f"Eval set: **{ds['total']} utterances** across "
        f"{len(ds['by_lang'])} languages "
        f"({', '.join(f'{k}={v}' for k, v in sorted(ds['by_lang'].items()))}), "
        f"**{ds['adult_rows']} adult-genre rows**. "
        "Ground truth and utterances are derived from the bundled `.voc` keyword "
        "files (see `benchmarks/dataset.py`).\n"
    )
    lines.append(f"- Available backends: {', '.join(report['available_backends']) or 'none'}")
    lines.append(f"- Unavailable backends: {', '.join(report['unavailable_backends']) or 'none'}\n")

    # Summary table
    lines.append("## Backend summary\n")
    lines.append("| backend | status | accuracy | macro-F1 | median ms | p95 ms | rows/s | CF recall | false-block |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, b in report["backends"].items():
        if b.get("status") != "available":
            lines.append(f"| {name} | unavailable | – | – | – | – | – | – | – |")
            continue
        cf = b.get("content_filter", {})
        lines.append(
            f"| {name} | available | {b['accuracy']:.3f} | {b['macro_f1']:.3f} | "
            f"{b['latency_ms_median']:.4f} | {b['latency_ms_p95']:.4f} | {b['rows_per_sec']:.0f} | "
            f"{cf.get('recall', 0):.3f} ({cf.get('adult_blocked', 0)}/{cf.get('adult_rows', 0)}) | "
            f"{cf.get('false_block_rate', 0):.3f} |"
        )
    lines.append("")

    # Unavailable reasons
    if report["unavailable_backends"]:
        lines.append("## Unavailable backends\n")
        for name in report["unavailable_backends"]:
            lines.append(f"- **{name}**: {report['backends'][name]['reason']}")
        lines.append("")

    # Per-type table for each available backend
    for name, b in report["backends"].items():
        if b.get("status") != "available":
            continue
        lines.append(f"## Per-type metrics — `{name}`\n")
        lines.append("| media_type | precision | recall | f1 | support |")
        lines.append("|---|---|---|---|---|")
        for t, m in sorted(b["per_type"].items()):
            lines.append(
                f"| {t} | {m['precision']:.3f} | {m['recall']:.3f} | "
                f"{m['f1']:.3f} | {m['support']} |"
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plots", action="store_true", help="also generate PNG plots into docs/benchmarks/")
    ap.add_argument("--rebuild-dataset", action="store_true", help="regenerate eval_set.csv before running")
    ap.add_argument("--hf-dataset", default=None,
                    help="evaluate on this HF dataset's test split (e.g. TigreGotico/ocp-media-intents) instead of the bundled eval_set.csv")
    ap.add_argument("--hf-limit", type=int, default=0, help="cap HF test rows (0 = all)")
    args = ap.parse_args(argv)

    report = run(rebuild_dataset=args.rebuild_dataset,
                 hf_dataset=args.hf_dataset, hf_limit=args.hf_limit)
    jp = write_json(report)
    mp = write_md(report)

    print(f"dataset: {report['dataset']['total']} rows "
          f"({report['dataset']['adult_rows']} adult)")
    print(f"available: {report['available_backends']}")
    print(f"unavailable: {report['unavailable_backends']}")
    for name in report["available_backends"]:
        b = report["backends"][name]
        cf = b["content_filter"]
        print(f"  {name:12s} acc={b['accuracy']:.3f} macroF1={b['macro_f1']:.3f} "
              f"median={b['latency_ms_median']:.4f}ms p95={b['latency_ms_p95']:.4f}ms "
              f"rows/s={b['rows_per_sec']:.0f} CFrecall={cf['recall']:.3f}")
    print(f"wrote {jp}")
    print(f"wrote {mp}")

    if args.plots:
        from benchmarks import plots
        made = plots.generate_all(report)
        for p in made:
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
