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

| head | axis | kind | label space |
|---|---|---|---|
| `content_form_genres` | sensitive / content-form tags | multi | `adult` / `anime` / `animation` / `asmr` |
| `content_genres` | the real genre(s) | multi | `mediavocab.KNOWN_GENRES` (capped to the top-`CONTENT_GENRE_TOP_K = 40`) |
| `content_form` | experiential kind (`mediavocab.ContentForm`) | single | `trailer` / `teaser` / `behind_scenes` / `excerpt` / `supplement` / … |
| `programme_format` | structural format (`mediavocab.ProgrammeFormat`) | single | `documentary` / `news` / `concert` / `stand_up` / `sports` / … |
| `accessibility` | a11y assets (`mediavocab.AccessibilityKind`) | multi | `subtitles` / `audio_description` / `sign_language` / … |
| `variant` | work-level cut (`mediavocab.VariantKind`) | single | `directors` / `extended` / `remastered` / `colorized` / `fanedit` / … |

**mediavocab axes.** Every descriptive head emits **mediavocab's own vocabulary**
so the classifier and the resolver / providers share one taxonomy. The finer
classifier cues collapse onto mediavocab's set (making_of / bloopers /
deleted_scenes / featurette → `behind_scenes`; clip → `excerpt`; interview →
`supplement`). `black_and_white` / `silent` / `3d` are picture-presentation
attributes with no mediavocab home yet — they ride a classifier-local
`presentation` axis until Phase 2 adds a `PictureFormat` enum. `dubbed` likewise
has no mediavocab home yet (deferred to Phase 2). `mood` / `era` are dropped from
the taxonomy (not axiom-admissible; a release year feeds `Signals.year`).

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
  ├── content_genres.onnx      # MULTI-LABEL  mediavocab.KNOWN_GENRES
  ├── content_form.onnx        # SINGLE  trailer/teaser/behind_scenes/excerpt/…
  ├── programme_format.onnx    # SINGLE  documentary/news/concert/stand_up/…
  ├── accessibility.onnx       # MULTI-LABEL subtitles/audio_description/sign_language
  ├── variant.onnx             # SINGLE  directors/extended/remastered/colorized/…
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

### Building it locally + the (manual) publish step

The whole bundle is reproduced from source with three local commands; **all
artifacts stay local under the gitignored `data/`** — nothing is published
automatically:

```bash
python -m training.ingest_entities --relations   # flat pools + non-IMDb relations
python -m training.imdb_relations                 # IMDb relations + popularity weights
python -m training.build_dataset                  # → data/release/*
python -m training.train_sklearn                  # → data/models/{context,context_ner}/
python -m benchmarks.ladder                        # → benchmarks/ladder_results.{json,md}
```

Publishing the dataset / model bundle to the Hub is a **separate, manual,
explicitly-authorised step** — it is never run by the build. When authorised, the
dataset is pushed with `python -m training.build_dataset --push --repo
TigreGotico/ocp-media-intents [--private]`; a model bundle is uploaded by hand
from `data/models/`. Until then the bundle lives only in the local gitignored
`data/` tree.

---

## Pre-trained models (download)

The trained bundles are published as a private Hugging Face collection,
**[OpenVoiceOS / OCP Media Classification](https://huggingface.co/collections/OpenVoiceOS/ocp-media-classification-6a3f21707a3e55a062d0d8d8)**
(token-gated), one repo per approach so you can pick the cost/quality point:

| repo | approach | pick it for |
|---|---|---|
| `OpenVoiceOS/ovos-media-classifier-onnx-default` | sklearn, keyword + NER | **recommended default** — lean (289 KiB, 0.25 ms), best content-filter precision |
| `OpenVoiceOS/ovos-media-classifier-onnx-keyword` | sklearn, keyword-only | works with zero registered entities |
| `OpenVoiceOS/ovos-media-classifier-onnx-tfidf-char` | TF-IDF char n-grams → linear | best `media_type` (0.978), no entities needed |
| `OpenVoiceOS/ovos-media-classifier-onnx-tfidf-word` | TF-IDF word n-grams → linear | best `tags` / genre·mood·era (0.93 F1) |
| `OpenVoiceOS/ovos-media-classifier-onnx-neural-wordvec` | neural + domain word-vectors | best content-filter recall (adult/hentai 0.98) |
| `OpenVoiceOS/ovos-media-classifier-onnx-neural` | neural, all features | balanced across axes |

```python
from huggingface_hub import snapshot_download
from ovos_media_classifier.onnx import OnnxMediaClassifier
path = snapshot_download("OpenVoiceOS/ovos-media-classifier-onnx-default", token="<hf_token>")
clf = OnnxMediaClassifier.from_path(path)
```

All bundles are trained on [`TigreGotico/ocp-media-intents`](https://huggingface.co/datasets/TigreGotico/ocp-media-intents)
and run on `onnxruntime`+`numpy` only.

---

## 5. Benchmark

> **Note.** The numbers below were measured on the *pre-alignment* taxonomy
> (`tags` / `qualifiers` heads). The mediavocab-axes alignment (this change)
> replaces those with the `content_form` / `programme_format` / `accessibility` /
> `variant` / `content_genres` heads; the dataset schema is already aligned, but
> the model retrain is a separate planned step, so these figures will be
> re-measured after the retrain on the aligned taxonomy. The single-label
> media-axis rows (`domain` / `media_type` / `playback_type` / `structure` /
> `explicitness`) are unaffected.

Held-out **test split: 34,700 utterances**. Per axis, the lift across the three
implemented rungs (**rules → learned context-only → learned context+NER**) is the
result. (Source: [benchmarks/ladder_results.md](../benchmarks/ladder_results.md).)

### Single-label axes — accuracy

| axis | rules | learned-context | learned-context+NER |
|---|---|---|---|
| domain | 0.844 | 0.873 | 0.988 |
| media_type | 0.663 | 0.786 | 0.967 |
| playback_type | 0.717 | 0.895 | 0.990 |
| structure | 0.738 | 0.908 | 0.992 |
| explicitness | 0.989 | 0.989 | 0.998 |

### Multi-label axes — macro-F1

| axis (pre-alignment) | maps to (post-alignment) | rules | learned-context | learned-context+NER |
|---|---|---|---|---|
| content_form_genres | content_form_genres | 0.720 | 0.729 | 0.979 |
| tags (genre slice) | content_genres | 0.000 | 0.561 | 0.596 |
| qualifiers | content_form + accessibility + presentation | 0.000 | 0.780 | 0.945 |

The `content_genres` macro-F1 is scored over the head's **modelled label space**
(its top-`CONTENT_GENRE_TOP_K` ⊆ `KNOWN_GENRES` labels) — the honest in-scope
task, not over the thousands of distinct raw genre values it cannot model (§6b).
The `qualifiers` row aggregated cues that the alignment splits across the
`content_form` (trailer / behind_scenes / …), `accessibility` (audio_description)
and classifier-local `presentation` (bw / silent) axes; per-axis figures land
with the retrain.

### Content filter (driven by the `content_form_genres` axis)

| rung | adult recall | hentai recall | false-block | median ms | p95 ms | size |
|---|---|---|---|---|---|---|
| rules | 0.479 (350/731) | 0.453 | 0.001 | 0.509 | 0.822 | — |
| learned-context | 0.479 (350/731) | 0.453 | 0.000 | 0.283 | 0.519 | 380 KiB |
| learned-context+NER | 0.932 (681/731) | 0.937 | 0.001 | 0.261 | 0.382 | 289 KiB |

The headline lifts (rules → context → context+NER): `media_type` accuracy
0.66 → 0.79 → 0.97; `playback_type` 0.72 → 0.90 → 0.99; `structure`
0.74 → 0.91 → 0.99; `content_form_genres` macro-F1 0.72 → 0.73 → 0.98;
the pre-alignment `qualifiers`/`tags` heads 0 → 0.78 → 0.95 / 0 → 0.56 → 0.60
(re-measured per the new axes after the retrain); adult-block recall
0.48 → 0.48 → 0.93; hentai recall 0.45 → 0.45 → 0.94 — at a near-zero false-block
rate and sub-millisecond latency in a 289 KiB bundle. The saturated `.intent`
templates and the enriched `kw_*` keyword vocabularies lift the **rules floor**
itself (`media_type` 0.63 → 0.66, `structure` 0.71 → 0.74,
`content_form_genres` 0.71 → 0.72) — the deterministic backend reads more cues —
while the canonical UNIFIED entity sets (cross-deduplicated artists / performers)
and the **coherent bw/silent** qualifier data (real black-and-white / silent
titles) sharpen the learned heads, lifting `qualifiers` 0.91 → 0.95.

---

## 6. Limitations

The benchmark above is honest about where the model is strong; it is just as
important to read where it is weak and *why*.

**(a) The bag-of-cue-presence ceiling.** The feature vector encodes *which cues
and which entity labels fired*, never the entity *value text* (§1). Any axis whose
ground truth lives in the slot value — not in a cue word — is fundamentally
under-determined by these features. This is a property of the representation, not
of the chosen estimator; a bigger model on the same features hits the same wall.

**(b) the `content_genres` axis is starved.** The genre head scores low and
barely moves from context to context+NER, because its signal is exactly the value
text the features drop: the real genre is *in* the genre slot value, not in a cue
word. This is the direct motivation for the **semantic** rung (§3): embedding the
surface string is the only way to read which genre was named. The head is shipped
so the rung is ready to train, not because the current features predict it well.
(`mood` / `era` are no longer modelled axes — dropped in the mediavocab-axes
alignment; a release year feeds `Signals.year` directly.)

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

**(e) Near-tie leaves where the keyword default is already right.** A handful of
leaves share almost all of their cue features and differ only in a token the bag
under-weights — `music` vs `music_video` is the canonical case (both fire the
music keywords; only the *video* modality cue separates them); `book` vs
`interactive_fiction` is another (both fire `verb_read`). The trained
`media_type` head can confuse such pairs where the deterministic keyword
classifier, matching leaf-first on the more specific `music_video` voc chain, gets
them right. The aggregate `media_type` accuracy is high, but on these specific
near-ties the rules floor is not strictly dominated — which is exactly why the
backends are interchangeable behind one contract and the keyword default stays the
zero-config baseline rather than being retired.

**(f) The dataset is English-dominated.** Templates are built across the seven core
languages, but the `en-us` `.intent` / `.voc` set is by far the richest — its
alternations and lead-ins expand to the large majority of the rows, so the trained
bundle is strongest on `en-us` and thinner on the other locales (and on the many
languages with no templates at all). The fix is **more translated templates**, not
a model change: the `.intent` / `.voc` files are managed through
[ovos-localize](https://openvoiceos.github.io/ovos-localize), so adding or
translating a locale ([dataset.md](dataset.md#to-add-or-translate-templates)) lifts
that language's coverage with no code change. The keyword backend degrades
gracefully on a thin locale (it finds no axis vocab and falls back to leaf-only
matching), so a missing language is under-served, not broken.

---

## 7. Neural backend + richer text features (does seeing the value text help?)

§6(a–b) names the load-bearing limitation: the categorical features encode *which
cue/entity-label fired*, never the *value text*, so any axis whose ground truth
lives in the slot value is under-determined. This section is the experiment that
attacks that wall directly — two new feature families that **can** read the
surface string, a neural (PyTorch → ONNX) trainer that consumes them, and a
head-to-head benchmark against the sklearn ladder on the same held-out test split.

### 7.1 Two text feature families (numpy-only at runtime)

Both run at train time *and* inference from the same code, so a bundle stays
self-describing — the spec goes in `meta.json` and the runtime rebuilds the exact
vector in numpy (no torch, no gensim, no transformers):

* **Hashed character n-grams** —
  [`features_text.py`](../ovos_media_classifier/features_text.py). Char 3–5 grams
  of the utterance → a fixed `dim` (default 4096) via the signed hashing trick,
  L2-normalized. This *sees subwords*: `jazz`, `horror`, title fragments — the
  exact tokens the binary flags drop. Spec (`dim` / ngram range / analyzer) is
  recorded in `meta.json["text_hash"]`.
* **Trained domain word vectors** —
  [`features_wordvec.py`](../ovos_media_classifier/features_wordvec.py) +
  [`training/build_corpus.py`](../training/build_corpus.py). A `gensim` Word2Vec
  (skip-gram, dim 100) trained on the **full domain corpus**: every entity pool
  (~4.35 M artist / track / album / movie / tv / anime / book / podcast / game
  strings), the relational co-occurrence records (~1.17 M — each record's fields
  joined so an artist, its album and its genre share a window), and the 347 k
  utterances. The learned matrix captures media semantics the flags can't —
  `jazz ≈ swing, reggae`; `horror ≈ thriller, mystery`; `rock ≈ punk, pop`. An
  utterance is mean-pooled over its in-vocab token rows; the matrix is saved as a
  pruned `.npy` (only tokens reachable from the dataset utterances) + a token→row
  vocab in the bundle, and `meta.json["wordvec"]` records the pooling config.

The model input becomes `[categorical ⊕ char-hash ⊕ word-vectors]`, any subset
selectable per variant.

### 7.2 The neural net —
[`training/train_torch.py`](../training/train_torch.py)

A **shared-trunk multi-task** net: featurizer → shared MLP trunk (LayerNorm +
ReLU + dropout, optional residual skips) → one linear head per axis (softmax for
single-label, sigmoid for multi-label). AdamW, class-weighting / `pos_weight` for
the imbalanced axes, early-stop on mean val macro-F1, fixed seed. Each head exports
as its **own** ONNX graph into the *existing* bundle format, so
[`OnnxMediaClassifier`](../ovos_media_classifier/onnx.py) loads it unchanged — the
only addition is reading the featurizer spec from `meta.json` to build the
`txt_*` / `wv_*` blocks at runtime (categorical-only sklearn bundles are
untouched, back-compat). torch→onnxruntime round-trip parity is verified at export
(max |Δ| ≈ 1e-7 on the softmax outputs).

### 7.3 The comparison (held-out test split, 34 700 utterances)

Single-label **accuracy** / multi-label **macro-F1**, scored identically across
rungs. `cat` is categorical-only (the neural counterpart of `sklearn context`);
`+text` adds char-hash, `+wordvec` adds the trained word vectors, `+all` both,
`(deep)` / `(wide)` are arch sweeps on `+all`. Full table + content-filter +
latency + size in
[`benchmarks/ladder_results.md`](../benchmarks/ladder_results.md).

| axis | rules | sklearn ctx | sklearn ctx+NER | neural cat | neural cat+text | neural cat+wv | neural cat+all | cat+all (wide) |
|---|---|---|---|---|---|---|---|---|
| media_type (acc) | 0.663 | 0.786 | **0.967** | 0.787 | 0.972 | 0.965 | 0.975 | 0.974 |
| playback_type | 0.717 | 0.895 | 0.990 | 0.887 | 0.987 | 0.984 | 0.988 | 0.988 |
| structure | 0.738 | 0.908 | 0.992 | 0.833 | 0.985 | 0.978 | 0.986 | 0.986 |
| content_form_genres (F1) | 0.720 | 0.729 | **0.979** | 0.510 | 0.858 | 0.717 | 0.875 | 0.876 |
| tags (F1) | 0.000 | 0.561 | 0.596 | 0.511 | **0.800** | 0.528 | 0.762 | **0.840** |
| qualifiers (F1) | 0.000 | 0.780 | 0.945 | 0.561 | 0.964 | 0.730 | 0.952 | 0.956 |
| adult recall | 0.479 | 0.479 | 0.932 | 0.867 | 0.921 | **0.977** | 0.919 | 0.943 |
| median ms | 0.43 | 0.23 | 0.22 | 0.16 | 6.29 | 0.62 | 6.10 | 16.1 |
| bundle size | — | 380 KiB | 289 KiB | 1.3 MiB | 79 MiB | 42 MiB | 114 MiB | 204 MiB |

### 7.4 Findings — honest

**Does the char-hash text help? Emphatically yes, and it is the headline.** On the
realistic *no-NER* inputs, adding char-hash to the categorical block lifts exactly
the value-text-dependent axes §6(b) said were starved: `tags` 0.511 → **0.800**,
`media_type` 0.787 → 0.972, `content_form_genres` 0.510 → 0.858, `qualifiers`
0.561 → 0.964. Seeing subwords is what reads the genre / title / qualifier out of
the surface string. This is the direct answer to §6(a–b): the wall was the
representation, and a text-aware representation climbs it.

**Do the trained domain word-vectors help?** On the *value-text* axes that depend
on **semantics over an open vocabulary** they help most for the **content filter**:
`+wordvec` gives the best adult recall of any rung (**0.977**, beating even the
NER-oracle), because the embedding pulls unseen adult-domain titles/terms toward
the blocked region. They lift `media_type` to 0.965 on raw text alone. But on
`tags` (0.528) mean-pooling *underperforms* char-hash — pooling averages away the
specific token that names the decade/mood, which the order-preserving char-hash
keeps. So word-vectors buy **semantic generalization** (content safety, coarse
type) more than fine descriptive precision.

**Does neural beat sklearn?** *Not on the same features* — `neural cat` ≈
`sklearn context` on `media_type` (0.787 vs 0.786) and is **worse** on the
multi-label axes (content_form_genres 0.510 vs 0.729). A plain MLP buys nothing
over a calibrated linear model on the binary flags. Neural wins **only because it
unlocks the richer features**: a linear model cannot consume a 4096-dim hashed
block as usefully, and the trunk lets all axes share that representation. The
lift is the *features*, delivered through the net — not the net itself.

**The honest caveat about `sklearn context+NER`.** It tops `media_type` (0.967)
and `content_form_genres` (0.979) — but its `ner_*` columns are **ground-truth by
construction** (set from the slot that filled the row, §1); it is a near-oracle
that the runtime only realizes once a NER store is wired in. The neural text/wv
rungs reach comparable accuracy reading **only the raw utterance**, which is what a
fresh install actually sees — so for the out-of-the-box, no-NER deployment the
char-hash neural bundle is the strongest realistic option.

**Is it worth the size / latency?** That is the real tradeoff. Char-hash costs
~6 ms/utterance (vs 0.2 ms sklearn) and a 79–204 MiB bundle — fine for a server,
heavy for a Pi. Word-vectors are the **sweet spot for content safety**: 0.6 ms,
42 MiB, best adult recall. The artifacts stay **local** (gitignored `data/`); the
shipped default remains the lean zero-config keyword classifier, with these bundles
an opt-in for deployments that can pay for the accuracy.

---

## See also

* [classification-model.md](classification-model.md) — why the output is
  orthogonal axes rather than a strict tree.
* [extending.md](extending.md) — add a backend, retrain a bundle, add a new axis
  end-to-end.
* [dataset.md](dataset.md) — the columns these heads are supervised on.
* [stable-api.md](stable-api.md) — the per-axis method reference.
* [benchmarks](../benchmarks/README.md) — the reproducible harness behind §5.
