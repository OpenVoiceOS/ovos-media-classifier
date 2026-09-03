#!/usr/bin/env python3
"""Hierarchical coarse-to-fine ``media_type`` classifier experiment.

Self-contained study: does predicting the **coarse axes first**
(``playback_type`` × ``structure``) and then constraining the fine
``media_type`` leaf to the compatible subset beat a flat multi-task
``media_type`` head — especially on the documented near-tie failures where two
leaves differ only on a coarse axis (``music``[audio] vs ``music_video``[video];
``book``[paged] vs ``interactive_fiction``[interactive])?

The taxonomy is near-deterministic in the *forward* direction
(``media_type`` → ``playback_type``/``structure`` via
:func:`mediavocab.infer_playback_type` / :func:`mediavocab.infer_structure`).
So the inverse — a (playback_type, structure) **group** → the set of leaves
compatible with it — is a clean mask we can apply to a flat leaf head, or use to
shard a cascade of per-group fine classifiers.

Three variants are built and compared on the **same held-out test split**, with
**features held constant** (the exact context / context+NER columns the sklearn
ladder uses — this isolates the STRUCTURE effect, not features):

1. ``flat``         — a plain ``media_type`` head (the control).
2. ``leaf_masking`` — predict ``playback_type`` + ``structure``, then mask the
   flat ``media_type`` probabilities to leaves compatible with the predicted
   coarse group and argmax over that subset. Cheapest hierarchical variant
   (reuses the SAME flat head, only post-processes its logits).
3. ``cascade``      — a coarse classifier over the (playback_type, structure)
   group, then a SEPARATE fine ``media_type`` classifier trained per group and
   applied only within that group's compatible leaves.

Per variant we report overall ``media_type`` accuracy + macro-F1, a focused
confusion breakdown on the near-tie pairs, and — honestly — the propagated-error
penalty: when the coarse prediction is WRONG, does the constraint HURT?

Run it (needs ``ovos-media-classifier[train]``; the split under ``data/release``)::

    python -m training.train_hierarchical                       # context-only + context+NER
    python -m training.train_hierarchical --feature-set context # one rung
    python -m training.train_hierarchical --export-onnx         # also export winner bundle

All artefacts land under gitignored ``data/`` (``data/hierarchical/``). Nothing
is published. ``ovos_media_classifier/onnx.py`` and the shared ladder benchmark
are NOT touched — the runtime integration path is documented in
``docs/hierarchical.md`` for wiring later.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mediavocab import MediaType, infer_playback_type
from mediavocab import infer_structure
from training.train_sklearn import (
    REPO_ROOT,
    ensure_dataset,
    feature_columns,
    _read_split,
)

# ---------------------------------------------------------------------------
# Taxonomy: forward leaf→group + inverse group→leaves (the constraint mask)
# ---------------------------------------------------------------------------

def _leaf_group(leaf: str) -> Tuple[str, str]:
    """The (playback_type, structure) coarse group a leaf belongs to."""
    mt = MediaType(leaf)
    return infer_playback_type(mt).value, infer_structure(mt).value


def build_taxonomy(leaves: List[str]) -> Tuple[
    Dict[str, Tuple[str, str]], Dict[Tuple[str, str], List[str]]
]:
    """forward leaf→group + inverse group→[leaves] over the observed leaves."""
    leaf_to_group: Dict[str, Tuple[str, str]] = {}
    group_to_leaves: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for leaf in leaves:
        g = _leaf_group(leaf)
        leaf_to_group[leaf] = g
        group_to_leaves[g].append(leaf)
    return leaf_to_group, dict(group_to_leaves)


# The documented near-tie pairs: leaves that differ ONLY on a coarse axis and
# that a flat head confuses.  Reported as a focused confusion breakdown.
NEAR_TIE_PAIRS: List[Tuple[str, str]] = [
    ("music", "music_video"),            # audio   vs video   (playback_type)
    ("book", "interactive_fiction"),     # paged   vs interactive
    ("comic", "interactive_fiction"),    # paged   vs interactive
    ("movie", "short_film"),             # both video/single — a coarse TIE (control)
    ("episodic_series", "tv"),           # episodic vs continuous (structure)
    ("podcast", "radio"),                # episodic vs continuous (structure)
    ("audiobook", "audio_drama"),        # single   vs episodic  (structure)
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _make_model():
    """The same compact linear head the sklearn ladder prefers (logreg, C=4)."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=400, C=4.0)


def _matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype="float32")


def _proba_frame(model, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """predict_proba + the class label order, as plain python strings."""
    proba = model.predict_proba(X)
    classes = [str(c) for c in model.classes_]
    return proba, classes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _scores(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
    }


def _pair_confusion(y_true: List[str], y_pred: List[str],
                    pairs: List[Tuple[str, str]]) -> Dict[str, dict]:
    """For each near-tie pair (a,b): how often true-a is called b and vice versa,
    plus the per-leaf recall on each member."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    out: Dict[str, dict] = {}
    for a, b in pairs:
        ma = yt == a
        mb = yt == b
        na, nb = int(ma.sum()), int(mb.sum())
        if na == 0 and nb == 0:
            continue
        a_to_b = int((yp[ma] == b).sum()) if na else 0
        b_to_a = int((yp[mb] == a).sum()) if nb else 0
        a_correct = int((yp[ma] == a).sum()) if na else 0
        b_correct = int((yp[mb] == b).sum()) if nb else 0
        out[f"{a}|{b}"] = {
            "n_a": na, "n_b": nb,
            "a_recall": round(a_correct / na, 4) if na else None,
            "b_recall": round(b_correct / nb, 4) if nb else None,
            "a_called_b": a_to_b, "b_called_a": b_to_a,
            "cross_leak": a_to_b + b_to_a,  # the confusion we want to fix
        }
    return out


# ---------------------------------------------------------------------------
# Variant 1 — flat baseline
# ---------------------------------------------------------------------------

def train_flat(train_df, test_df, cols) -> dict:
    Xtr, ytr = _matrix(train_df, cols), train_df["media_type"].astype(str).tolist()
    Xte = _matrix(test_df, cols)
    model = _make_model()
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    proba, classes = _proba_frame(model, Xte)
    preds = [classes[i] for i in proba.argmax(1)]
    return {"model": model, "classes": classes, "proba": proba,
            "preds": preds, "fit_s": round(fit_s, 1)}


# ---------------------------------------------------------------------------
# Variant 2 — inference leaf-masking (reuses the SAME flat head)
# ---------------------------------------------------------------------------

def train_coarse_heads(train_df, test_df, cols) -> dict:
    """Train the two coarse single-label heads (playback_type, structure)."""
    out = {}
    for axis in ("playback_type", "structure"):
        ytr = train_df[axis].astype(str)
        mask = ytr != ""
        m = _make_model()
        m.fit(_matrix(train_df[mask.values], cols), ytr[mask].tolist())
        proba, classes = _proba_frame(m, _matrix(test_df, cols))
        preds = [classes[i] for i in proba.argmax(1)]
        out[axis] = {"model": m, "classes": classes, "preds": preds}
    return out


def apply_leaf_masking(flat: dict, coarse: dict,
                       group_to_leaves: Dict[Tuple[str, str], List[str]],
                       leaves: List[str]) -> List[str]:
    """Mask flat media_type proba to the predicted coarse group's leaves.

    When the predicted (playback_type, structure) group has compatible leaves,
    restrict the argmax to them; otherwise fall back to the unmasked argmax (so
    an out-of-taxonomy coarse prediction never zeroes out every option).
    """
    proba = flat["proba"]
    classes = flat["classes"]
    cls_idx = {c: i for i, c in enumerate(classes)}
    pb = coarse["playback_type"]["preds"]
    st = coarse["structure"]["preds"]
    out: List[str] = []
    for i in range(proba.shape[0]):
        group = (pb[i], st[i])
        allowed = group_to_leaves.get(group)
        row = proba[i]
        if allowed:
            idxs = [cls_idx[l] for l in allowed if l in cls_idx]
            if idxs:
                best = idxs[int(np.argmax(row[idxs]))]
                out.append(classes[best])
                continue
        out.append(classes[int(np.argmax(row))])
    return out


# ---------------------------------------------------------------------------
# Variant 3 — cascade (coarse group head → per-group fine head)
# ---------------------------------------------------------------------------

def train_cascade(train_df, test_df, cols,
                  group_to_leaves) -> dict:
    """A coarse (playback_type, structure) group head + one fine media_type head
    per group (trained only on that group's rows / compatible leaves)."""
    # ---- coarse group label = "playback|structure" string ----
    def _group_str(df):
        return [f"{infer_playback_type(MediaType(mt)).value}|"
                f"{infer_structure(MediaType(mt)).value}"
                for mt in df["media_type"].astype(str)]

    gtr = _group_str(train_df)
    gte_true = _group_str(test_df)
    coarse = _make_model()
    coarse.fit(_matrix(train_df, cols), gtr)
    gproba, gclasses = _proba_frame(coarse, _matrix(test_df, cols))
    gpred = [gclasses[i] for i in gproba.argmax(1)]

    # ---- one fine head per group (skip degenerate single-leaf groups) ----
    fine: Dict[str, dict] = {}
    gtr_arr = np.asarray(gtr)
    ytr_mt = train_df["media_type"].astype(str).to_numpy()
    Xtr = _matrix(train_df, cols)
    for gstr in sorted(set(gtr)):
        sel = gtr_arr == gstr
        leaves_here = sorted(set(ytr_mt[sel]))
        if len(leaves_here) < 2:
            fine[gstr] = {"single": leaves_here[0]}  # only one possible leaf
            continue
        m = _make_model()
        m.fit(Xtr[sel], ytr_mt[sel])
        fine[gstr] = {"model": m, "classes": [str(c) for c in m.classes_]}

    # ---- inference: route each row to its predicted group's fine head ----
    Xte = _matrix(test_df, cols)
    preds: List[str] = []
    for i in range(len(test_df)):
        g = gpred[i]
        head = fine.get(g)
        if head is None:
            # group never seen in train — fall back to global argmax via any leaf
            preds.append("generic")
            continue
        if "single" in head:
            preds.append(head["single"])
            continue
        row = head["model"].predict_proba(Xte[i:i + 1])[0]
        preds.append(head["classes"][int(np.argmax(row))])
    return {"coarse": coarse, "fine": fine, "group_pred": gpred,
            "group_true": gte_true, "preds": preds,
            "group_classes": gclasses}


# ---------------------------------------------------------------------------
# Propagated-error analysis (the honest tradeoff)
# ---------------------------------------------------------------------------

def coarse_error_breakdown(test_df, coarse_pred_group, flat_preds,
                           hier_preds, leaf_to_group) -> dict:
    """When the predicted coarse group is RIGHT vs WRONG, what is media_type
    accuracy for the flat vs hierarchical predictions?  Quantifies whether the
    constraint helps (right group) or hurts (propagated error, wrong group)."""
    y_true = test_df["media_type"].astype(str).to_numpy()
    true_group = [leaf_to_group[l] for l in y_true]
    pred_group = coarse_pred_group  # list of (pb, st) tuples
    right = np.asarray([pg == tg for pg, tg in zip(pred_group, true_group)])

    def _acc(mask, preds):
        if mask.sum() == 0:
            return None
        p = np.asarray(preds)[mask]
        t = y_true[mask]
        return round(float((p == t).mean()), 4)

    return {
        "coarse_group_accuracy": round(float(right.mean()), 4),
        "n_right_group": int(right.sum()),
        "n_wrong_group": int((~right).sum()),
        "right_group": {
            "flat_acc": _acc(right, flat_preds),
            "hier_acc": _acc(right, hier_preds),
        },
        "wrong_group": {
            "flat_acc": _acc(~right, flat_preds),
            "hier_acc": _acc(~right, hier_preds),
        },
    }


# ---------------------------------------------------------------------------
# Orchestration for one feature set
# ---------------------------------------------------------------------------

def run_feature_set(train_df, test_df, feature_set: str) -> dict:
    cols = feature_columns(train_df, feature_set)
    leaves = sorted(set(train_df["media_type"].astype(str)))
    leaf_to_group, group_to_leaves = build_taxonomy(leaves)
    y_true = test_df["media_type"].astype(str).tolist()

    print(f"\n=== feature-set: {feature_set} ({len(cols)} features, "
          f"{len(leaves)} leaves, {len(group_to_leaves)} coarse groups) ===")

    # --- 1. flat ---
    print("  training flat media_type head …")
    flat = train_flat(train_df, test_df, cols)
    flat_scores = _scores(y_true, flat["preds"])
    print(f"    flat: acc={flat_scores['accuracy']:.4f} "
          f"macroF1={flat_scores['macro_f1']:.4f}")

    # --- 2. leaf-masking (reuses flat head + coarse heads) ---
    print("  training coarse heads (playback_type, structure) …")
    coarse = train_coarse_heads(train_df, test_df, cols)
    mask_preds = apply_leaf_masking(flat, coarse, group_to_leaves, leaves)
    mask_scores = _scores(y_true, mask_preds)
    coarse_group_pred = list(zip(coarse["playback_type"]["preds"],
                                 coarse["structure"]["preds"]))
    print(f"    leaf_masking: acc={mask_scores['accuracy']:.4f} "
          f"macroF1={mask_scores['macro_f1']:.4f}")

    # --- 3. cascade ---
    print("  training cascade (coarse group head + per-group fine heads) …")
    casc = train_cascade(train_df, test_df, cols, group_to_leaves)
    casc_scores = _scores(y_true, casc["preds"])
    casc_group_pred = [tuple(g.split("|")) for g in casc["group_pred"]]
    print(f"    cascade: acc={casc_scores['accuracy']:.4f} "
          f"macroF1={casc_scores['macro_f1']:.4f}")

    # --- per-variant near-tie confusion ---
    near = {
        "flat": _pair_confusion(y_true, flat["preds"], NEAR_TIE_PAIRS),
        "leaf_masking": _pair_confusion(y_true, mask_preds, NEAR_TIE_PAIRS),
        "cascade": _pair_confusion(y_true, casc["preds"], NEAR_TIE_PAIRS),
    }

    # --- propagated-error analysis ---
    prop = {
        "leaf_masking": coarse_error_breakdown(
            test_df, coarse_group_pred, flat["preds"], mask_preds, leaf_to_group),
        "cascade": coarse_error_breakdown(
            test_df, casc_group_pred, flat["preds"], casc["preds"], leaf_to_group),
    }

    return {
        "feature_set": feature_set,
        "n_features": len(cols),
        "n_test": len(test_df),
        "leaves": leaves,
        "groups": {f"{g[0]}|{g[1]}": v for g, v in group_to_leaves.items()},
        "variants": {
            "flat": flat_scores,
            "leaf_masking": mask_scores,
            "cascade": casc_scores,
        },
        "near_tie_confusion": near,
        "propagated_error": prop,
        "_artifacts": {  # live objects, stripped before JSON dump
            "flat": flat, "coarse": coarse, "cascade": casc,
            "leaf_to_group": leaf_to_group,
            "group_to_leaves": group_to_leaves, "cols": cols,
        },
    }


# ---------------------------------------------------------------------------
# ONNX export (optional, only if a variant wins) — leaf-masking bundle
# ---------------------------------------------------------------------------

def export_leaf_masking_bundle(out_dir, result) -> str:
    """Export a self-describing bundle for the leaf-masking variant.

    Reuses the SAME multi-head ONNX format the runtime already loads: a
    ``media_type`` head + ``playback_type`` + ``structure`` heads, plus a
    ``masking`` block in meta that records the group→leaves map so the runtime
    can apply the constraint with NO new code (the maps already power onnx.py's
    derive-fallback).  onnx.py is NOT modified here.
    """
    from training.train_sklearn import _to_onnx, _index_map

    os.makedirs(out_dir, exist_ok=True)
    art = result["_artifacts"]
    cols = art["cols"]
    n = len(cols)

    heads = {}
    # flat media_type head
    with open(os.path.join(out_dir, "media_type.onnx"), "wb") as fh:
        fh.write(_to_onnx(art["flat"]["model"], n))
    heads["media_type"] = {"onnx": "media_type.onnx", "kind": "single",
                           "labels": _index_map(art["flat"]["model"].classes_)}
    for axis in ("playback_type", "structure"):
        m = art["coarse"][axis]["model"]
        with open(os.path.join(out_dir, f"{axis}.onnx"), "wb") as fh:
            fh.write(_to_onnx(m, n))
        heads[axis] = {"onnx": f"{axis}.onnx", "kind": "single",
                       "labels": _index_map(m.classes_)}

    meta = {
        "feature_names": cols,
        "feature_set": result["feature_set"],
        "input_name": "input",
        "heads": heads,
        "variant": "leaf_masking",
        # the constraint the runtime applies: predicted (playback_type,structure)
        # group → allowed media_type leaves.  Same maps as onnx.py derive-fallback.
        "masking": {
            "group_to_leaves": {f"{g}": leaves
                                for g, leaves in result["groups"].items()},
            "note": ("argmax media_type over leaves compatible with the "
                     "predicted playback_type|structure group; fall back to "
                     "unmasked argmax when the group has no leaves."),
        },
        "trained_by": "training/train_hierarchical.py",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return out_dir


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _strip(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "_artifacts"}


def render_md(report: dict) -> str:
    L: List[str] = []
    L.append("# Hierarchical coarse-to-fine `media_type` — flat vs "
             "leaf-masking vs cascade\n")
    L.append(f"Held-out **test split**: {report['n_test']:,} utterances. "
             "Features held CONSTANT per rung (same columns the sklearn ladder "
             "uses) so this isolates the STRUCTURE effect, not features.\n")
    for fs, r in report["rungs"].items():
        L.append(f"## Feature set: `{fs}` ({r['n_features']} features)\n")
        L.append("### Overall `media_type`\n")
        L.append("| variant | accuracy | macro-F1 |")
        L.append("|---|---|---|")
        for v in ("flat", "leaf_masking", "cascade"):
            s = r["variants"][v]
            L.append(f"| {v} | {s['accuracy']:.4f} | {s['macro_f1']:.4f} |")
        L.append("")
        # near-tie confusion
        L.append("### Near-tie confusion — cross-leak (lower is better)\n")
        pairs = sorted(r["near_tie_confusion"]["flat"].keys())
        L.append("| pair | flat | leaf_masking | cascade |")
        L.append("|---|---|---|---|")
        for p in pairs:
            cells = []
            for v in ("flat", "leaf_masking", "cascade"):
                c = r["near_tie_confusion"][v].get(p, {})
                cells.append(str(c.get("cross_leak", "-")))
            L.append(f"| {p} | {cells[0]} | {cells[1]} | {cells[2]} |")
        L.append("")
        # propagated error
        L.append("### Propagated-error tradeoff (when the coarse group is wrong)\n")
        for v in ("leaf_masking", "cascade"):
            pe = r["propagated_error"][v]
            L.append(f"**{v}** — coarse-group accuracy "
                     f"{pe['coarse_group_accuracy']:.4f} "
                     f"(right={pe['n_right_group']:,}, wrong={pe['n_wrong_group']:,})\n")
            L.append("| subset | flat media_type acc | hier media_type acc |")
            L.append("|---|---|---|")
            rg, wg = pe["right_group"], pe["wrong_group"]
            L.append(f"| coarse RIGHT | {rg['flat_acc']} | {rg['hier_acc']} |")
            L.append(f"| coarse WRONG | {wg['flat_acc']} | {wg['hier_acc']} |")
            L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Hierarchical coarse-to-fine media_type experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--data-dir", default=os.path.join("data", "release"))
    ap.add_argument("--out-dir", default=os.path.join("data", "hierarchical"))
    ap.add_argument("--feature-set", choices=["context", "context_ner", "both"],
                    default="both")
    ap.add_argument("--export-onnx", action="store_true",
                    help="export the leaf-masking bundle for the winning rung")
    args = ap.parse_args(argv)

    data_dir = args.data_dir if os.path.isabs(args.data_dir) \
        else os.path.join(REPO_ROOT, args.data_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) \
        else os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ensure_dataset(data_dir)
    print(f"Loading splits from {data_dir} …")
    train_df = _read_split(data_dir, "train")
    test_df = _read_split(data_dir, "test")
    print(f"  train={len(train_df):,}  test={len(test_df):,}")

    feature_sets = (["context", "context_ner"] if args.feature_set == "both"
                    else [args.feature_set])
    report = {"data_dir": data_dir, "n_test": len(test_df), "rungs": {}}
    results = {}
    for fs in feature_sets:
        res = run_feature_set(train_df, test_df, fs)
        results[fs] = res
        report["rungs"][fs] = _strip(res)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    md = render_md(report)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    print(f"\nwrote {out_dir}/results.json + results.md")

    if args.export_onnx:
        # export the leaf-masking bundle for the most-confused rung (context)
        rung = "context" if "context" in results else feature_sets[0]
        bundle = os.path.join(out_dir, f"leaf_masking_{rung}")
        export_leaf_masking_bundle(bundle, results[rung])
        print(f"exported leaf-masking bundle → {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
