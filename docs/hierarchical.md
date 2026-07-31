# Hierarchical coarse-to-fine `media_type`, experiment

Does predicting the **coarse axes first** (`playback_type` × `structure`) and
then constraining the fine `media_type` leaf to the compatible subset beat the
flat multi-task `media_type` head, especially on the documented near-tie
failures where two leaves differ only on a coarse axis?

This is a **structure** experiment, not a features experiment. Every variant is
trained on the *same* feature columns the sklearn ladder uses (`kw_*` / `verb_*`
/ `mod_*` / `fmt_*` / `attr_*`, plus `ner_*` for the `context_ner` rung) so the
only thing that changes is whether classification is flat or hierarchical.

Built and benchmarked by **`training/train_hierarchical.py`** (self-contained:
trains all three variants and scores them on the same held-out test split, then
optionally exports the leaf-masking ONNX bundle). Artefacts land under gitignored
`data/hierarchical/`. The runtime (`ovos_media_classifier/onnx.py`) and the
shared ladder benchmark are **not** touched, the integration path is documented
below for wiring later.

```
python -m training.train_hierarchical               # context + context+NER rungs
python -m training.train_hierarchical --export-onnx # also export the masking bundle
```

## The taxonomy is the lever

The forward map `media_type → (playback_type, structure)` is near-deterministic
(`mediavocab.infer_playback_type` + `ovos_media_classifier.axes.infer_structure`).
Inverting it gives, for each coarse **group**, the set of compatible leaves, the
constraint mask. Over the dataset leaves there are 9 groups:

| group (playback_type \| structure) | compatible leaves |
|---|---|
| `audio\|single` | audiobook, music, sound_effect |
| `audio\|episodic` | audio_drama |
| `audio\|continuous` | procedural_ambient, radio |
| `video\|single` | movie, music_video, short_film |
| `video\|episodic` | episodic_series |
| `video\|continuous` | tv |
| `paged\|single` | comic *(book maps here too)* |
| `interactive\|single` | game, interactive_fiction |
| `unknown\|collection` | playlist |

The hypothesised near-tie wins live across group boundaries:
`music`[audio] vs `music_video`[video], `book`[paged] vs
`interactive_fiction`[interactive], leaves a flat head could confuse but a mask
would forbid.

## Three variants

1. **flat**, one plain `media_type` logistic-regression head (the control;
   reproduces the sklearn-ladder flat number).
2. **leaf_masking**, predict `playback_type` + `structure`, then restrict the
   flat head's argmax to the predicted group's compatible leaves. Cheapest
   variant: it reuses the **same** flat head and only post-processes its logits.
   Falls back to the unmasked argmax when the predicted group has no leaves.
3. **cascade**, a coarse `(playback_type, structure)` group head, then a
   **separate** fine `media_type` head trained per group and applied only within
   that group's leaves (single-leaf groups skip the fine head).

All heads are the same compact `LogisticRegression(C=4)` the ladder prefers, so
size/latency are comparable.

## Results (held-out test split, 20,131 utterances)

### Overall `media_type`

| feature set | variant | accuracy | macro-F1 |
|---|---|---|---|
| context | flat | 0.8663 | 0.8690 |
| context | leaf_masking | 0.8661 | 0.8689 |
| context | cascade | 0.8665 | 0.8691 |
| context+NER | flat | 0.9525 | 0.9370 |
| context+NER | leaf_masking | 0.9519 | 0.9365 |
| context+NER | cascade | 0.9524 | 0.9369 |

The three variants are **within noise** of each other on both rungs (≤ 0.0006
spread). Hierarchy neither helps nor hurts the headline number.

### Near-tie confusion, cross-leak (count of true-a→pred-b + true-b→pred-a; lower is better)

| pair | flat (ctx) | mask (ctx) | cascade (ctx) | flat (+NER) | mask (+NER) |
|---|---|---|---|---|---|
| music \| music_video | 2 | 2 | 2 | 2 | 2 |
| book \| interactive_fiction | 0 | 0 | 0 | 0 | 0 |
| comic \| interactive_fiction | 60 | 59 | 59 | 16 | 16 |
| movie \| short_film | 48 | 48 | 48 | 48 | 48 |
| episodic_series \| tv | 0 | 0 | 0 | 0 | 0 |
| podcast \| radio | 0 | 0 | 0 | 0 | 0 |
| audiobook \| audio_drama | 0 | 0 | 0 | 0 | 0 |

The **premise does not survive contact with the data**: the flat head already
resolves the cross-coarse near-ties (music↔music_video = 2, book↔IF = 0). The
only non-trivial residual confusions are:

* `movie ↔ short_film` (48), both `video|single`, the **same coarse group**.
  No coarse axis distinguishes them, so masking and cascade are *structurally
  unable* to help. This is a within-group leaf ambiguity.
* `comic ↔ interactive_fiction` (60 → 59), these *are* in different groups
  (`paged|single` vs `interactive|single`), so masking should help. It barely
  moves (−1). The reason is propagation: the errors are the **coarse head**
  itself miscalling `paged` vs `interactive` on those rows, and when the coarse
  prediction is wrong the mask cannot rescue the leaf.

### Propagated-error tradeoff (the honest cost)

When the predicted coarse group is **right**, the constraint is harmless (it ties
flat, +0.0004). When the predicted coarse group is **wrong**, the constraint is
strictly **worse** than flat, because it forbids the correct leaf by construction:

| feature set | variant | coarse-group acc | flat acc \| coarse-RIGHT | hier acc \| coarse-RIGHT | flat acc \| coarse-WRONG | hier acc \| coarse-WRONG |
|---|---|---|---|---|---|---|
| context | leaf_masking | 0.9144 | 0.9468 | 0.9472 | 0.007 | **0.000** |
| context | cascade | 0.9148 | 0.9469 | 0.9473 | 0.0017 | **0.000** |
| context+NER | leaf_masking | 0.9588 | 0.9925 | 0.9929 | 0.0217 | **0.000** |
| context+NER | cascade | 0.9593 | 0.9929 | 0.9929 | 0.0024 | **0.000** |

The flat head occasionally **recovers** a row whose coarse axis it would have got
wrong (it scores 0.7–2.2 % on the coarse-wrong subset); the hierarchy scores a
hard **0** there. The tiny gain on the coarse-right subset (+0.04 %) and the
small loss on the coarse-wrong subset cancel almost exactly, which is why the
overall numbers are a wash.

## Verdict

**Hierarchy does not help here, and is not worth integrating into the runtime.**

* The flat `media_type` head, trained on the same categorical+NER features,
  *already* learns the coarse distinctions implicitly. The cross-coarse near-ties
  the hypothesis targeted (music↔music_video, book↔interactive_fiction) are
  effectively already solved by the flat head (cross-leak ≤ 2).
* The residual confusions are either **within-group** (movie↔short_film, no
  coarse axis separates them, so hierarchy is structurally powerless) or
  **coarse-head errors** (comic↔interactive_fiction, masking inherits the
  mistake instead of fixing it).
* The constraint has an asymmetric cost: it never adds more than +0.0004 on
  coarse-right rows but forces 0 % accuracy on coarse-wrong rows, removing the
  flat head's small self-recovery. Net effect across the split is ≈ 0.
* Cascade adds per-group models (more artefacts, more inference branches) for the
  same null result, strictly worse on a cost/benefit basis than leaf-masking,
  which is already a no-op.

The taxonomy is a good *fallback* (deriving a coarse axis from the leaf when a
head is missing, what `onnx.py` already does), but as a *forward constraint* on
a head that already has the signal it is redundant. **Recommendation: keep the
flat multi-task design; do not adopt masking or cascade in the runtime.** If a
future model regresses on the cross-coarse near-ties (e.g. a much smaller head
that loses the implicit coarse signal), leaf-masking is the cheap lever to
revisit, the bundle format already supports it (below).

## Runtime integration path (if ever needed)

`--export-onnx` writes a self-describing bundle to
`data/hierarchical/leaf_masking_context/` in the **same multi-head format the
runtime already loads**, a `media_type` head plus `playback_type` + `structure`
heads, with one extra `masking` block in `meta.json`:

```json
"masking": {
  "group_to_leaves": { "audio|single": ["audiobook", "music", "sound_effect"], ... },
  "note": "argmax media_type over leaves compatible with the predicted
           playback_type|structure group; fall back to unmasked argmax when the
           group has no leaves."
}
```

To apply masking at runtime **without** changing the leaf head, an
`OnnxMediaClassifier` subclass (or a flag) would, in `classify()`:

1. run the existing `playback_type` + `structure` heads (already present, `classify_playback_type` / `classify_structure`),
2. look up the compatible leaves with the **same** forward maps that already
   power the derive-fallback (`mediavocab.MEDIA_TYPE_TO_PLAYBACK_TYPE` +
   `ovos_media_classifier.axes.MEDIA_TYPE_TO_STRUCTURE`), build the inverse map
   once at load, no new data needed,
3. restrict the `media_type` head's probability argmax to that leaf set (falling
   back to the unmasked argmax when the predicted group is empty).

This needs **no change to `onnx.py`'s file format**, the `masking` block is
additive metadata an older loader ignores. Given the verdict above it is left
unwired; this document records how to wire it should a future head need it.

---
[← Routing eval](routing-eval.md) · [Home](index.md) · [Entity lists →](entity-lists.md)
