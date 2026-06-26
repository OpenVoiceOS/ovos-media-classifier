# The classifier model

This page is for an ML engineer who wants to understand *how the trained backend
works as a model*: what it sees, what it predicts, how each axis is learned, and
where the design hits its ceiling. It is the implementation companion to
[classification-model.md](classification-model.md) (which explains *why* the
output is a set of orthogonal axes) and [stable-api.md](stable-api.md) (the method
reference).

The runtime that loads a trained bundle is
[`OnnxMediaClassifier`](../ovos_media_classifier/onnx.py); the trainer is
[`training/train_sklearn.py`](../training/train_sklearn.py); the dataset that
supervises it is documented in [dataset.md](dataset.md).

---

## 1. Feature representation

The model never sees raw text. Each utterance is reduced to a **sparse 0/1
categorical feature dict** by
[`CategoricalFeatureExtractor`](../ovos_media_classifier/features.py): a feature
is present (value `"1"`) when its cue fired, absent otherwise. Two families of
columns make up the vector.

**Keyword / context columns** — one per bundled `.voc` file, word-boundary matched
through `ovos-spec-tools` (so `art` does not fire inside `start`). The stable
column menu is `_KEYWORD_VOCABS` in
[`features.py`](../ovos_media_classifier/features.py); the prefixes encode the cue
family:

| prefix | what fired | examples |
|---|---|---|
| `kw_*` | a media-type / form keyword | `kw_music`, `kw_movie`, `kw_anime`, `kw_adult`, `kw_hentai` |
| `verb_*` | a modality verb | `verb_audio` (listen), `verb_video` (watch), `verb_read`, `verb_tune` |
| `mod_*` | a structure / time modifier | `mod_episode`, `mod_season`, `mod_live`, `mod_continue`, `mod_latest` |
| `attr_*` | an attribute cue | `attr_topic` (*about …*), `attr_starring` |
| `fmt_*` | an explicit format hint | `fmt_audio_only`, `fmt_video_only` |

**NER-by-construction columns** — one `ner_<label>` per
[`OCPEntityLabel`](../ovos_media_classifier/intents.py). In the training data the
flag is set when a `{label}` slot filled that row (ground truth, see
[dataset.md](dataset.md)); at runtime a NER backend would set it when it tags an
entity of that label.

**Be honest about what these encode.** The `ner_*` columns record the entity
*label* that fired — that an `artist_name` was present — **not the entity *value
text*** (*which* artist). The model knows "a music artist appears here"; it cannot
read "Adele" vs "Metallica". This is the load-bearing limitation of the whole
representation and it recurs in §3 and §6.

`meta.json` in every bundle records the exact ordered `feature_names` the model
was trained on. At inference `OnnxMediaClassifier._vectorize` walks that list and
emits a dense `float32` row in precisely that order, so the runtime never assumes
a column layout — it reads it from the bundle.

---

## 2. Multi-task per-axis heads — the key design

A naive classifier predicts the leaf `MediaType` and *derives* every other axis
from it. The trained backend instead predicts **each axis with its own head**, so
an axis can be right even when the leaf is wrong. The trainer declares the heads
in `HEAD_SPECS` ([`train_sklearn.py`](../training/train_sklearn.py)); the runtime
runs whichever heads the bundle carries.

**Single-label heads** (argmax over their `labels`):

| head | axis | label space |
|---|---|---|
| `domain` | is this OCP at all | `ocp_play` / `not_ocp` |
| `media_type` | the leaf | the 17 `mediavocab.MediaType` leaves the dataset exercises |
| `playback_type` | modality | `audio` / `video` / `paged` / `interactive` |
| `structure` | temporal shape | `single` / `episodic` / `continuous` / `collection` |
| `explicitness` | clean vs adult | `clean` / `adult` |
| `control_intent` | transport control | `play` / `pause` / `next` / … (degenerate in a play-only bundle — skipped) |

**Multi-label heads** — a `OneVsRestClassifier` of logistic regressions emitting
per-label probabilities; the runtime keeps every label whose probability is
≥ a per-label `threshold` (default `0.5`, recorded in `meta.json`):

| head | axis | label space |
|---|---|---|
| `content_form_genres` | sensitive / content-form tags | `adult` / `anime` / `animation` / `asmr` |
| `tags` | the **namespaced descriptive axis** — genre, mood and era folded into one | `genre:rock` / `genre:action` / `mood:chill` / `era:1980s` / … (capped to the top-K most frequent, `TAGS_TOP_K = 80`) |
| `qualifiers` | result-narrowing filters | `black_and_white` / `silent` / `live` / `subtitled` / `dubbed` / `audio_described` / `trailer` / … |

**Axes vs tags.** The single-label heads above are the **axes** — one answer per
query. `tags` is the multi-label catch-all for the open-vocabulary *descriptive*
signals (genre, mood, era) that all live in slot *value text*: rather than three
starved single-label heads, they are one namespaced multi-label head. The
`classify_content_genres` / `classify_mood` / `classify_era` convenience methods
read the `genre:` / `mood:` / `era:` slice of this one head.

The `content_form_genres` head is the one the **content filter reads**. Because it
is its own head, it can flag `adult` **independently of the leaf** — a request can
be blocked even when the model is unsure whether it is a `movie` or an
`episodic_series`. That is what makes detect-to-block robust (a single leaf
mistake never unblocks adult content).

### Soft-gating and the derive fallback

Predicting axes separately enables **soft-gating**: `OnnxMediaClassifier` trusts
an axis head even when the leaf is uncertain, and the leaf head is domain-gated
(a non-`ocp_play` domain short-circuits to `GENERIC`) rather than vetoed by a hard
cascade. See `_play_label`, `_media_type_from_head` and `classify_full` in
[`onnx.py`](../ovos_media_classifier/onnx.py).

Every per-axis method follows the same pattern: **use the head when the bundle
carries it, else fall back to the inherited derive/empty default**. A head whose
column was degenerate on the training data is simply skipped at train time
(`train_single_head` returns `skipped`), its `.onnx` file is absent, and
`from_path` derives that axis instead. This is what lets **partial bundles** —
and old two-head `domain`/`play` bundles — load and run unchanged.

---

## 3. The ladder

Each head is trained and reported on a rising sequence of feature sets — the lift
from one rung to the next *is* the headline result.

1. **rules** — the deterministic bundled keyword classifier (no learning). The
   floor.
2. **context-only** — the keyword columns only (`kw_*` / `verb_*` / `mod_*` /
   `attr_*` / `fmt_*`); `ner_*` **excluded**. This is the "works with no
   registered entities" baseline a fresh install sees, before any skill has
   populated a NER store.
3. **context+NER** — the keyword columns **plus** the `ner_*` columns: the
   features a populated NER would surface.
4. **semantic** — *the documented next step, not yet implemented.* Sentence
   embeddings as features.

The two implemented learned rungs are the `FEATURE_SETS = ("context",
"context_ner")` in [`train_sklearn.py`](../training/train_sklearn.py);
`feature_columns` builds each column set. **Semantic embeddings are the next
lift** because a bag of cue-presence flags cannot read entity *value text* (§1) —
the only way to recover the signal that lives inside the slot value (which genre,
which mood, which decade) is to embed the surface string. §6 quantifies exactly
which axes are starved by this.

---

## 4. Self-describing bundle / retrain contract

A bundle is a **directory that fully describes itself**, so the runtime needs no
out-of-band knowledge of what it trained on:

```
<bundle>/
  ├── domain.onnx              # ocp_play / not_ocp
  ├── media_type.onnx          # the leaf mediavocab.MediaType
  ├── playback_type.onnx       # audio / video / paged / interactive
  ├── structure.onnx           # single / episodic / continuous / collection
  ├── content_form_genres.onnx # MULTI-LABEL adult/anime/animation/asmr  ← content filter
  ├── tags.onnx                # MULTI-LABEL genre:rock/mood:chill/era:1980s
  ├── qualifiers.onnx          # MULTI-LABEL black_and_white/silent/live/…
  ├── explicitness.onnx        # clean / adult  (when trained)
  ├── play.onnx                # back-compat alias of the media_type head
  └── meta.json
```

`meta.json` carries the ordered `feature_names`, the `input_name`, and a `heads`
manifest — one entry per axis naming its `.onnx` file, `kind` (`single`/`multi`),
index→label map, and (for multi-label heads) the `threshold`. It also keeps the
legacy `domain_labels` / `play_labels` keys so a pre-multihead loader still works.

[`OnnxMediaClassifier.from_path`](../ovos_media_classifier/onnx.py) loads whatever
heads exist: it iterates `meta["heads"]`, opens an `InferenceSession` per present
`.onnx`, and skips any whose file is missing (that axis falls back to derive). The
retrain contract is therefore: produce a bundle in this layout and any version of
the runtime consumes it. The reference producer is
[`training/train_sklearn.py`](../training/train_sklearn.py) (`export_bundle`),
installable via the `[train]` extra. End-to-end retraining — including adding a
*new* axis — is covered in [extending.md](extending.md).

---

## 5. Benchmark

Held-out **test split: 34,700 utterances**. Per axis, the lift across the three
implemented rungs (**rules → learned context-only → learned context+NER**) is the
result. (Source: [benchmarks/ladder_results.md](../benchmarks/ladder_results.md).)

### Single-label axes — accuracy

| axis | rules | learned-context | learned-context+NER |
|---|---|---|---|
| domain | 0.833 | 0.866 | 0.986 |
| media_type | 0.629 | 0.778 | 0.964 |
| playback_type | 0.702 | 0.895 | 0.988 |
| structure | 0.708 | 0.907 | 0.990 |
| explicitness | 0.988 | 0.989 | 0.997 |

### Multi-label axes — macro-F1

| axis | rules | learned-context | learned-context+NER |
|---|---|---|---|
| content_form_genres | 0.706 | 0.738 | 0.975 |
| tags | 0.000 | 0.547 | 0.581 |
| qualifiers | 0.000 | 0.746 | 0.906 |

The `tags` macro-F1 is scored over the head's **modelled label space** (its
top-`TAGS_TOP_K` namespaced labels) — the honest in-scope task, not over the
thousands of distinct raw genre values it cannot model (§6b). Folding genre, mood
and era into this one head lifts it from the ~0.00–0.10 the old three starved
single-label heads scored to **0.55 → 0.58**.

### Content filter (driven by the `content_form_genres` axis)

| rung | adult recall | hentai recall | false-block | median ms | p95 ms | size |
|---|---|---|---|---|---|---|
| rules | 0.481 (364/756) | 0.510 | 0.000 | 0.319 | 0.498 | — |
| learned-context | 0.481 (364/756) | 0.510 | 0.000 | 0.211 | 0.250 | 176 KiB |
| learned-context+NER | 0.922 (697/756) | 0.936 | 0.001 | 0.214 | 0.259 | 289 KiB |

The headline lifts (rules → context → context+NER): `media_type` accuracy
0.63 → 0.78 → 0.96; `playback_type` 0.70 → 0.90 → 0.99; `structure`
0.71 → 0.91 → 0.99; `content_form_genres` macro-F1 0.71 → 0.74 → 0.98;
`qualifiers` 0 → 0.75 → 0.91; `tags` 0 → 0.55 → 0.58; adult-block recall
0.48 → 0.48 → 0.92; hentai recall 0.51 → 0.51 → 0.94 — at a near-zero false-block
rate and sub-millisecond latency in a 289 KiB bundle. The **coherent bw/silent**
qualifier data (real black-and-white / silent titles) is what lifts `qualifiers`
recall on those filters.

---

## 6. Limitations

The benchmark above is honest about where the model is strong; it is just as
important to read where it is weak and *why*.

**(a) The bag-of-cue-presence ceiling.** The feature vector encodes *which cues
and which entity labels fired*, never the entity *value text* (§1). Any axis whose
ground truth lives in the slot value — not in a cue word — is fundamentally
under-determined by these features. This is a property of the representation, not
of the chosen estimator; a bigger model on the same features hits the same wall.

**(b) the `tags` axis (genre / mood / era) is starved.** The folded `tags` head
scores low and barely moves from context to context+NER, because its signal is
exactly the value text the features drop: the decade is *in* the year string, the
mood is *in* the activity phrase, the real genre is *in* the genre slot value.
Folding the three into one namespaced multi-label head (instead of three starved
single-label heads) keeps the axis count honest, but does not by itself recover
the signal — that is the direct motivation for the **semantic** rung (§3):
embedding the surface string is the only way to read which genre / mood / era was
named. The head is shipped so the rung is ready to train, not because the current
features predict it well.

**(c) Synthetic / degenerate label regions.** The `domain` head's negative class
is **synthetic** — the all-zero feature vector (no keyword or NER evidence)
labelled `not_ocp`; see `train_domain_head`. It learns "any media evidence ⇒ OCP",
which is the right prior but is not trained against real non-media utterances. And
because every dataset row is `ocp_play`, the `control_intent` column is constant,
so its head is skipped at train time and the `ocp_control` domain is **untrained /
degenerate** in this bundle.

**(d) The runtime feature path is keyword-only.** The shipped
`CategoricalFeatureExtractor` produces only the keyword columns — the NER
value-extraction path is not part of this release (`features.py` documents this).
So even a `context+NER` bundle only ever sees keyword features at runtime **unless
a NER backend is wired in to populate the `ner_*` columns**. The context+NER
numbers above are the model's *capability* given populated entities; the
out-of-the-box runtime sees the context-only behaviour until a NER store is
attached.

---

## See also

* [classification-model.md](classification-model.md) — why the output is
  orthogonal axes rather than a strict tree.
* [extending.md](extending.md) — add a backend, retrain a bundle, add a new axis
  end-to-end.
* [dataset.md](dataset.md) — the columns these heads are supervised on.
* [stable-api.md](stable-api.md) — the per-axis method reference.
* [benchmarks](../benchmarks/README.md) — the reproducible harness behind §5.
