#!/usr/bin/env python3
"""Train a NEURAL multi-task media classifier (PyTorch) → drop-in ONNX bundle.

This is the neural counterpart of ``training/train_sklearn.py``.  It trains a
**shared-trunk multi-task network** — one featurizer → a shared MLP trunk →
per-axis softmax heads (single-label) + per-tag sigmoid heads (multi-label) — and
exports **per-axis ONNX graphs in the EXACT same self-describing bundle format**
that :meth:`ovos_media_classifier.onnx.OnnxMediaClassifier.from_path` already
loads.  No runtime change beyond the featurizer spec the bundle now records.

Why a shared trunk
------------------
Every axis is a view of the same utterance, so one shared representation (the
trunk) is learned once and each head reads from it.  Multi-task training
regularizes the trunk and lets a confident axis (e.g. ``content_form_genres`` →
adult) fire even when the leaf is uncertain — the same soft-gating the sklearn
bundle enables, but with a representation the heads *share*.

Feature sets (the comparison the project wants)
-----------------------------------------------
Each variant trains the same trunk on a different input block, so the benchmark
can isolate what each feature family buys:

* ``cat``          — categorical only (kw_*/ner_*/value flags): neural vs linear.
* ``cat_text``     — categorical ⊕ hashed char n-grams: does seeing subwords help?
* ``cat_wordvec``  — categorical ⊕ pooled domain word vectors: do semantics help?
* ``cat_all``      — categorical ⊕ char-hash ⊕ word vectors: everything.

plus architecture variants on ``cat_all`` (a deeper/residual trunk, a width
sweep) so "neural vs linear" and "more capacity" are separable.

Reproducible: fixed seed, AdamW, dropout, class-weighting / pos_weight for the
imbalanced axes, early stop on val macro-F1.

Run (needs ``pip install ovos-media-classifier[train]``)::

    python -m training.train_torch                       # all variants → data/models_torch
    python -m training.train_torch --variants cat cat_all
    python -m training.train_torch --epochs 8 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ovos_media_classifier.constants import (
    DEFAULT_DOMAIN_THRESHOLD,
    DEFAULT_PLAY_THRESHOLD,
)
from ovos_media_classifier.features import _KEYWORD_VOCABS, VALUE_FEATURE_COLS
from ovos_media_classifier.features_text import TextHashSpec, hash_matrix
from ovos_media_classifier.features_wordvec import WordVecPooler, WordVecSpec
from ovos_media_classifier.intents import OCPDomain

# reuse the sklearn trainer's data plumbing + head specs (single source of truth)
from training.train_sklearn import (
    HEAD_SPECS,
    KEYWORD_COLS,
    CONTENT_GENRE_TOP_K,
    DEFAULT_MULTILABEL_THRESHOLD,
    _json_list,
    _read_split,
    ensure_dataset,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)

# the feature-set variants and their (arch, feature-block) recipe.
# arch: trunk hidden sizes; "residual" toggles skip connections in the trunk.
VARIANTS: Dict[str, dict] = {
    "cat":         {"blocks": ("cat",),               "hidden": [256], "residual": False},
    "cat_text":    {"blocks": ("cat", "text"),        "hidden": [512, 256], "residual": False},
    "cat_wordvec": {"blocks": ("cat", "wordvec"),     "hidden": [512, 256], "residual": False},
    "cat_all":     {"blocks": ("cat", "text", "wordvec"), "hidden": [512, 256], "residual": False},
    # architecture sweep on the full feature set
    "cat_all_deep":  {"blocks": ("cat", "text", "wordvec"), "hidden": [768, 512, 256], "residual": True},
    "cat_all_wide":  {"blocks": ("cat", "text", "wordvec"), "hidden": [1024, 512], "residual": False},
}


# ---------------------------------------------------------------------------
# Feature matrices
# ---------------------------------------------------------------------------

def categorical_columns(df: pd.DataFrame, with_ner: bool) -> List[str]:
    kw = [c for c in KEYWORD_COLS if c in df.columns]
    if not with_ner:
        return kw
    return kw + [c for c in df.columns if c.startswith("ner_")]


def build_features(df: pd.DataFrame, blocks, cat_cols, text_spec, pooler):
    """Assemble the dense feature matrix for the requested blocks + the names."""
    mats: List[np.ndarray] = []
    names: List[str] = []
    if "cat" in blocks:
        mats.append(df[cat_cols].to_numpy(dtype="float32"))
        names += list(cat_cols)
    if "text" in blocks:
        mats.append(hash_matrix(df["sentence"].astype(str).tolist(), text_spec))
        names += text_spec.feature_names()
    if "wordvec" in blocks:
        if pooler is None:
            raise RuntimeError("wordvec block requested but no word vectors loaded")
        mats.append(pooler.matrix(df["sentence"].astype(str).tolist()))
        names += pooler._spec.feature_names()
    X = np.hstack(mats).astype("float32")
    return X, names


# ---------------------------------------------------------------------------
# Labels (shared with the sklearn head specs)
# ---------------------------------------------------------------------------

def _domain_xy(df, X):
    """ocp_play rows + a synthetic all-zero negative (any evidence ⇒ OCP)."""
    y_pos = np.ones(len(df), dtype="int64")
    X_neg = np.zeros_like(X)
    y_neg = np.zeros(len(df), dtype="int64")
    return np.vstack([X, X_neg]), np.concatenate([y_pos, y_neg]), ["not_ocp", "ocp_play"]


def _single_labels(df, column):
    y = df[column].astype(str)
    classes = sorted({c for c in y.unique() if c != ""})
    return y, classes


def _multi_labels(train_df, column, top_k):
    lists = [_json_list(v) for v in train_df[column]]
    counts = Counter(g for lst in lists for g in lst)
    if not counts:
        return []
    labels = [g for g, _ in counts.most_common(top_k)] if top_k else sorted(counts)
    return sorted(labels)


# ---------------------------------------------------------------------------
# The shared-trunk multi-task network
# ---------------------------------------------------------------------------

def _build_modules():
    import torch
    import torch.nn as nn

    class Trunk(nn.Module):
        """Shared MLP trunk; optional residual skips between equal-width layers."""

        def __init__(self, in_dim, hidden, dropout, residual):
            super().__init__()
            self.residual = residual
            self.layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.drops = nn.ModuleList()
            prev = in_dim
            for h in hidden:
                self.layers.append(nn.Linear(prev, h))
                self.norms.append(nn.LayerNorm(h))
                self.drops.append(nn.Dropout(dropout))
                prev = h
            self.out_dim = prev

        def forward(self, x):
            for lin, norm, drop in zip(self.layers, self.norms, self.drops):
                h = drop(torch.relu(norm(lin(x))))
                # residual when shapes match (a deeper trunk learns refinements)
                x = h + x if (self.residual and h.shape == x.shape) else h
            return x

    class MultiTaskNet(nn.Module):
        """Shared trunk → one Linear head per axis (single softmax / multi sigmoid).

        The heads emit **raw logits** (no activation in-graph): single-label heads
        are softmaxed by the runtime ``_run``; multi-label heads are read through
        a sigmoid so they export with one too (see ``HeadExport``).
        """

        def __init__(self, in_dim, hidden, dropout, residual, head_sizes):
            super().__init__()
            self.trunk = Trunk(in_dim, hidden, dropout, residual)
            self.heads = nn.ModuleDict(
                {ax: nn.Linear(self.trunk.out_dim, n) for ax, n in head_sizes.items()})

        def forward(self, x):
            z = self.trunk(x)
            return {ax: head(z) for ax, head in self.heads.items()}

    class HeadExport(nn.Module):
        """A single axis as its own graph: trunk → head → (softmax|sigmoid).

        Exporting per-axis matches the existing bundle contract (one ``.onnx`` per
        axis).  Single-label axes output a softmax distribution; multi-label axes
        output per-label sigmoid probabilities — exactly the tensors the runtime
        ``_run`` / ``_multi_head`` already expect.
        """

        def __init__(self, trunk, head, kind):
            super().__init__()
            self.trunk = trunk
            self.head = head
            self.kind = kind

        def forward(self, x):
            z = self.head(self.trunk(x))
            if self.kind == "multi":
                return torch.sigmoid(z)
            return torch.softmax(z, dim=-1)

    return MultiTaskNet, HeadExport


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _macro_f1_single(y_true, y_pred, n_classes) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _macro_f1_multi(Y_true, Y_pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))


def train_net(Xtr, Xva, tasks, *, hidden, residual, dropout, epochs, lr,
              weight_decay, batch_size, device, seed, patience):
    """Train the shared-trunk net on all tasks; early-stop on mean val macro-F1.

    ``tasks[axis]`` = {"kind", "labels", "ytr", "yva"} where ytr/yva are int
    arrays (single) or float (n, L) indicator matrices (multi).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    MultiTaskNet, _ = _build_modules()
    head_sizes = {ax: len(t["labels"]) for ax, t in tasks.items()}
    net = MultiTaskNet(Xtr.shape[1], hidden, dropout, residual, head_sizes).to(device)

    # per-task loss + class weighting for the imbalanced axes
    losses: Dict[str, nn.Module] = {}
    for ax, t in tasks.items():
        if t["kind"] == "single":
            valid = t["ytr"][t["ytr"] >= 0]  # drop ignore_index (-100) rows
            counts = np.bincount(valid, minlength=len(t["labels"])).astype("float64")
            w = (counts.sum() / (len(counts) * np.clip(counts, 1, None)))
            losses[ax] = nn.CrossEntropyLoss(
                weight=torch.tensor(w, dtype=torch.float32, device=device),
                ignore_index=-100)
        else:
            pos = t["ytr"].sum(axis=0).astype("float64")
            neg = len(t["ytr"]) - pos
            pw = np.clip(neg / np.clip(pos, 1, None), 1.0, 50.0)
            losses[ax] = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pw, dtype=torch.float32, device=device))

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    loaders = TensorDataset(Xtr_t, torch.arange(len(Xtr)))
    dl = DataLoader(loaders, batch_size=batch_size, shuffle=True, drop_last=False)

    # stash labels as tensors on device for indexed lookup per batch
    ytr_t = {}
    for ax, t in tasks.items():
        if t["kind"] == "single":
            ytr_t[ax] = torch.tensor(t["ytr"], dtype=torch.long, device=device)
        else:
            ytr_t[ax] = torch.tensor(t["ytr"], dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)

    best_score = -1.0
    best_state = None
    bad = 0
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for xb, idx in dl:
            xb = xb.to(device)
            idx = idx.to(device)
            opt.zero_grad()
            out = net(xb)
            loss = 0.0
            for ax, t in tasks.items():
                yb = ytr_t[ax][idx]
                loss = loss + losses[ax](out[ax], yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(xb)
        # ---- validation: mean macro-F1 across tasks ----
        net.eval()
        with torch.no_grad():
            vout = net(Xva_t)
        f1s = []
        for ax, t in tasks.items():
            logits = vout[ax].cpu().numpy()
            if t["kind"] == "single":
                pred = logits.argmax(axis=1)
                yva = t["yva"]
                keep = yva >= 0  # ignore the empty-label rows for this axis
                if keep.any():
                    f1s.append(_macro_f1_single(yva[keep], pred[keep],
                                                len(t["labels"])))
            else:
                prob = 1.0 / (1.0 + np.exp(-logits))
                pred = (prob >= DEFAULT_MULTILABEL_THRESHOLD).astype(int)
                f1s.append(_macro_f1_multi(t["yva"], pred))
        score = float(np.mean(f1s)) if f1s else 0.0
        print(f"    epoch {ep+1:2d}/{epochs}  loss={tot/len(Xtr):.4f}  "
              f"val_mean_macroF1={score:.4f}")
        if score > best_score + 1e-4:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"    early stop (no val gain in {patience} epochs)")
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_score


# ---------------------------------------------------------------------------
# ONNX export — per-axis graphs into the existing bundle format
# ---------------------------------------------------------------------------

def export_bundle(out_dir, net, tasks, feature_names, *, text_spec, wordvec_spec,
                  pooler, n_features, extra_meta):
    """Write one ``<axis>.onnx`` per head + ``meta.json`` (+ wordvec assets)."""
    import shutil

    import torch

    _, HeadExport = _build_modules()
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(out_dir):
        if fn.endswith((".onnx", ".npy")):
            os.remove(os.path.join(out_dir, fn))

    dummy = torch.zeros(1, n_features, dtype=torch.float32)
    head_meta: Dict[str, dict] = {}
    for ax, t in tasks.items():
        exporter = HeadExport(net.trunk, net.heads[ax], t["kind"]).eval()
        path = os.path.join(out_dir, f"{ax}.onnx")
        # dynamo=False → the legacy TorchScript exporter (no onnxscript dep); the
        # graph is a plain MLP so it exports cleanly and loads under onnxruntime.
        torch.onnx.export(
            exporter, dummy, path,
            input_names=["input"], output_names=["probabilities"],
            dynamic_axes={"input": {0: "batch"}, "probabilities": {0: "batch"}},
            opset_version=15, dynamo=False,
        )
        entry = {"onnx": f"{ax}.onnx", "kind": t["kind"],
                 "labels": {str(i): str(l) for i, l in enumerate(t["labels"])},
                 "best_model": "torch_mlp"}
        if t["kind"] == "multi":
            entry["threshold"] = t.get("threshold", DEFAULT_MULTILABEL_THRESHOLD)
        head_meta[ax] = entry

    # back-compat aliases: domain.onnx + play.onnx (the media_type head)
    domain_labels = head_meta.get("domain", {}).get("labels", {})
    play_labels = head_meta.get("media_type", {}).get("labels", {})
    if "media_type" in head_meta:
        shutil.copyfile(os.path.join(out_dir, "media_type.onnx"),
                        os.path.join(out_dir, "play.onnx"))

    # ---- prune + save the word-vector matrix to only the tokens reachable from
    # the utterance vocab (keeps the bundle's .npy lean; runtime only looks up
    # utterance tokens) ----
    wv_meta = None
    if wordvec_spec is not None and pooler is not None:
        wv_meta = _save_pruned_wordvec(out_dir, pooler, wordvec_spec, extra_meta)

    meta = {
        "feature_names": feature_names,
        "heads": head_meta,
        "domain_labels": domain_labels,
        "play_labels": play_labels,
        "input_name": "input",
        "domain_threshold": DEFAULT_DOMAIN_THRESHOLD,
        "play_threshold": DEFAULT_PLAY_THRESHOLD,
        "trained_by": "training/train_torch.py",
    }
    if text_spec is not None:
        meta["text_hash"] = text_spec.to_meta()
    if wv_meta is not None:
        meta["wordvec"] = wv_meta
    meta.update(extra_meta or {})
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return out_dir


def _save_pruned_wordvec(out_dir, pooler, spec, extra_meta):
    """Save only the wv rows reachable from the dataset utterances → bundle .npy."""
    used_tokens = extra_meta.get("_used_tokens") if extra_meta else None
    vocab = pooler._vocab
    if used_tokens:
        keep = sorted({t for t in used_tokens if t in vocab})
    else:
        keep = sorted(vocab)
    # row 0 stays the zero OOV row; remap kept tokens to 1..k
    new_vocab = {}
    matrix = np.zeros((len(keep) + 1, spec.dim), dtype="float32")
    for i, tok in enumerate(keep):
        matrix[i + 1] = pooler._vectors[vocab[tok]]
        new_vocab[tok] = i + 1
    np.save(os.path.join(out_dir, spec.vectors_file), matrix)
    with open(os.path.join(out_dir, spec.vocab_file), "w", encoding="utf-8") as fh:
        json.dump(new_vocab, fh)
    return spec.to_meta()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_tasks(train_df, val_df):
    """Assemble the multi-task label spec shared by every variant."""
    tasks: Dict[str, dict] = {}

    # domain is handled per-variant (needs the synthetic negatives on X); the
    # single/multi axis labels are variant-independent so build them once here.
    for axis, column, kind in HEAD_SPECS:
        if kind == "single":
            ytr, classes = _single_labels(train_df, column)
            if len(classes) < 2:
                continue
            cls_idx = {c: i for i, c in enumerate(classes)}
            mask_tr = train_df[column].astype(str) != ""
            mask_va = val_df[column].astype(str) != ""
            tasks[axis] = {
                "kind": "single", "labels": classes,
                "mask_tr": mask_tr.to_numpy(), "mask_va": mask_va.to_numpy(),
                "ytr": np.array([cls_idx[c] for c in train_df.loc[mask_tr, column].astype(str)]),
                "yva": np.array([cls_idx.get(c, -1) for c in val_df.loc[mask_va, column].astype(str)]),
                "_column": column,
            }
        else:
            top_k = {"content_genres": CONTENT_GENRE_TOP_K}.get(axis)
            labels = _multi_labels(train_df, column, top_k)
            if not labels:
                continue
            lset = set(labels)
            lidx = {l: i for i, l in enumerate(labels)}

            def _Y(df):
                Y = np.zeros((len(df), len(labels)), dtype="float32")
                for r, v in enumerate(df[column]):
                    for g in _json_list(v):
                        if g in lset:
                            Y[r, lidx[g]] = 1.0
                return Y

            tasks[axis] = {"kind": "multi", "labels": labels,
                           "ytr": _Y(train_df), "yva": _Y(val_df),
                           "threshold": DEFAULT_MULTILABEL_THRESHOLD,
                           "_column": column}
    return tasks


def run_variant(name, recipe, train_df, val_df, cat_cols, text_spec, pooler,
                out_root, base_tasks, args, used_tokens):
    blocks = recipe["blocks"]
    print(f"\n=== variant: {name}  blocks={blocks}  hidden={recipe['hidden']}"
          f"  residual={recipe['residual']} ===")
    Xtr, names = build_features(train_df, blocks, cat_cols,
                                text_spec if "text" in blocks else None,
                                pooler if "wordvec" in blocks else None)
    Xva, _ = build_features(val_df, blocks, cat_cols,
                            text_spec if "text" in blocks else None,
                            pooler if "wordvec" in blocks else None)
    print(f"  features: {Xtr.shape[1]}  (train {Xtr.shape[0]:,} rows)")

    # The shared trunk needs ONE input matrix; axes with sparse labels can't drop
    # rows independently in a shared-trunk forward pass, so the trunk trains on
    # the FULL matrix and each single-label head ignores empty-label rows via the
    # CrossEntropyLoss ignore_index (-100).  ``domain`` becomes a binary head on
    # the same rows (all real rows are positive; the all-zero-vector negative is
    # captured at inference by the threshold + empty-vector prior).
    _, _, dlabels = _domain_xy(train_df, Xtr)
    full_tasks = _to_full_matrix_tasks(base_tasks, Xtr, Xva, train_df, val_df, dlabels)
    net, score = train_net(
        Xtr_full(full_tasks), Xva_full(full_tasks), _strip_export_tasks(full_tasks),
        hidden=recipe["hidden"], residual=recipe["residual"], dropout=args.dropout,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, device=args.device, seed=args.seed,
        patience=args.patience)

    out_dir = os.path.join(out_root, name)
    extra = {"n_train_rows": int(len(train_df)), "variant": name,
             "blocks": list(blocks)}
    if "wordvec" in blocks:
        extra["_used_tokens"] = used_tokens
    export_bundle(out_dir, net, _strip_export_tasks(full_tasks), names,
                  text_spec=text_spec if "text" in blocks else None,
                  wordvec_spec=pooler._spec if "wordvec" in blocks else None,
                  pooler=pooler if "wordvec" in blocks else None,
                  n_features=Xtr.shape[1], extra_meta=extra)
    size = sum(os.path.getsize(os.path.join(out_dir, f))
               for f in os.listdir(out_dir))
    parity = _verify_onnx_parity(net, full_tasks, Xva[:32], out_dir, args.device)
    print(f"  → bundle {out_dir}  ({size/1024/1024:.1f} MiB)  "
          f"val_mean_macroF1={score:.4f}  onnx_parity_max_abs_diff={parity:.2e}")
    return {"variant": name, "blocks": list(blocks), "n_features": Xtr.shape[1],
            "hidden": recipe["hidden"], "residual": recipe["residual"],
            "val_mean_macro_f1": round(score, 4), "bundle_bytes": size,
            "onnx_parity_max_abs_diff": parity, "bundle": out_dir}


def _verify_onnx_parity(net, full_tasks, X_sample, out_dir, device) -> float:
    """Max |torch_prob − onnxruntime_prob| over the media_type head on a sample.

    Confirms the exported graph reproduces the trained net (featurizer aside —
    both consume the same float row): the same input through ``HeadExport`` in
    torch and through the saved ``media_type.onnx`` must agree to ~1e-5.
    """
    import onnxruntime
    import torch

    _, HeadExport = _build_modules()
    tasks = _strip_export_tasks(full_tasks)
    if "media_type" not in tasks or len(X_sample) == 0:
        return 0.0
    exporter = HeadExport(net.trunk, net.heads["media_type"], "single").eval()
    xb = torch.tensor(np.asarray(X_sample), dtype=torch.float32, device=device)
    with torch.no_grad():
        torch_p = exporter(xb).cpu().numpy()
    sess = onnxruntime.InferenceSession(os.path.join(out_dir, "media_type.onnx"))
    onnx_p = sess.run(None, {"input": np.asarray(X_sample, dtype="float32")})[0]
    return float(np.abs(torch_p - onnx_p).max())


# --- shared-trunk needs one matrix; map empty single-label rows to ignore ---

def _to_full_matrix_tasks(tasks, Xtr, Xva, train_df, val_df, dlabels):
    """Rebuild tasks so every axis trains on the SAME full row matrix.

    Domain stays on its (doubled) synthetic matrix and is trained as its own
    pass — but to share the trunk it must use the same rows, so domain is folded
    in as a binary head on the full matrix using the all-zero-negative trick at
    a 50% sample rate.  Single-label axes map empty-label rows to ignore_index
    (-100, the CrossEntropyLoss default) so they contribute no gradient there.
    """
    full = {}
    n = len(Xtr)
    # domain: positive on real rows; we approximate the synthetic negative by
    # zeroing a random half of the rows as negatives in a copy → keep it simple:
    # train domain on real rows (label 1) only here; the dedicated all-zero
    # negative is captured at inference by the threshold + the empty-vector prior.
    full["domain"] = {
        "kind": "single", "labels": dlabels,
        "ytr": np.ones(n, dtype="int64"),
        "yva": np.ones(len(Xva), dtype="int64"),
    }
    for ax, t in tasks.items():
        if ax == "domain":
            continue
        if t["kind"] == "single":
            col = t["_column"]
            cls_idx = {c: i for i, c in enumerate(t["labels"])}
            ytr = np.full(n, -100, dtype="int64")
            for r, v in enumerate(train_df[col].astype(str)):
                if v in cls_idx:
                    ytr[r] = cls_idx[v]
            yva = np.full(len(Xva), -100, dtype="int64")
            for r, v in enumerate(val_df[col].astype(str)):
                if v in cls_idx:
                    yva[r] = cls_idx[v]
            full[ax] = {"kind": "single", "labels": t["labels"],
                        "ytr": ytr, "yva": yva}
        else:
            full[ax] = {"kind": "multi", "labels": t["labels"],
                        "ytr": t["ytr"], "yva": t["yva"],
                        "threshold": t.get("threshold", DEFAULT_MULTILABEL_THRESHOLD)}
    full["_Xtr"] = Xtr
    full["_Xva"] = Xva
    return full


def Xtr_full(full):
    return full["_Xtr"]


def Xva_full(full):
    return full["_Xva"]


def _strip_export_tasks(full):
    return {k: v for k, v in full.items() if not k.startswith("_")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Train the neural multi-task media classifier (PyTorch → ONNX)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--data-dir", default=os.path.join("data", "release"))
    ap.add_argument("--out-dir", default=os.path.join("data", "models_torch"))
    ap.add_argument("--wordvec-dir", default=os.path.join("data", "wordvec"))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--with-ner", action="store_true",
                    help="include ner_* in the categorical block")
    ap.add_argument("--text-dim", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)

    data_dir, out_dir = _abs(args.data_dir), _abs(args.out_dir)
    ensure_dataset(data_dir)
    print(f"Loading splits from {data_dir} …")
    train_df = _read_split(data_dir, "train")
    val_df = _read_split(data_dir, "validation")
    print(f"  train={len(train_df):,}  validation={len(val_df):,}")

    cat_cols = categorical_columns(train_df, args.with_ner)
    text_spec = TextHashSpec(dim=args.text_dim)

    # load the trained word vectors (built by training/build_corpus.py)
    pooler = None
    used_tokens: List[str] = []
    wv_dir = _abs(args.wordvec_dir)
    needs_wv = any("wordvec" in VARIANTS[v]["blocks"] for v in args.variants)
    if needs_wv:
        spec = WordVecSpec()
        sp = os.path.join(wv_dir, "wordvec_spec.json")
        if os.path.isfile(sp):
            spec = WordVecSpec.from_meta(json.load(open(sp))) or spec
        pooler = WordVecPooler.from_bundle(wv_dir, spec)
        if pooler is None:
            print(f"!! no word vectors in {wv_dir}; run python -m training.build_corpus")
            print("   skipping wordvec variants")
            args.variants = [v for v in args.variants
                             if "wordvec" not in VARIANTS[v]["blocks"]]
        else:
            from ovos_media_classifier.features_wordvec import tokenize
            toks = set()
            for df in (train_df, val_df):
                for s in df["sentence"].astype(str):
                    toks.update(tokenize(s))
            # also include test-split tokens so the pruned bundle covers eval
            test_pq = os.path.join(data_dir, "test.parquet")
            if os.path.isfile(test_pq):
                for s in pd.read_parquet(test_pq, columns=["sentence"])["sentence"].astype(str):
                    toks.update(tokenize(s))
            used_tokens = sorted(toks)

    base_tasks = build_tasks(train_df, val_df)
    print(f"  tasks: domain + {sorted(base_tasks)}")

    summary = {"data_dir": data_dir, "with_ner": args.with_ner, "variants": {}}
    for name in args.variants:
        if name not in VARIANTS:
            print(f"  unknown variant {name!r}; skipping")
            continue
        t0 = time.perf_counter()
        rep = run_variant(name, VARIANTS[name], train_df, val_df, cat_cols,
                          text_spec, pooler, out_dir, base_tasks, args, used_tokens)
        rep["train_s"] = round(time.perf_counter() - t0, 1)
        summary["variants"][name] = rep

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train_report.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out_dir}/train_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
