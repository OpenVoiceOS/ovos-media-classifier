# The multi-axis classification model

This page explains the classification model: **why media classification is a set
of orthogonal axes rather than one strict tree**, what each axis is, and how both
the keyword classifier and trained backends produce it.

It is the conceptual companion to the API docs ([stable-api.md](stable-api.md),
[taxonomy.md](taxonomy.md)). Read this when you want to understand *the shape of
the problem*; read those when you want method signatures.

---

## 1. The problem

When a user speaks to OVOS, the OCP pipeline has to turn one utterance into a
routing decision. _"put on the news"_, _"play Naruto"_, _"read me a chapter of
Dune"_, _"turn off the kitchen lights"_ are all different requests, and the
useful distinctions between them are not one-dimensional. _"a music video"_ is
**audio content** delivered as **video**; _"an audiobook"_ is a **book** delivered
as **audio**. Forcing those into a single label loses information that a
downstream skill needs.

So instead of one label, the classifier predicts a **point in a small product
space**. Each coordinate (axis) is a coarse, high-signal question; the combination
is the full answer, returned as one
[`MediaClassification`](#7-the-mediaclassification-result).

---

## 2. The axes

```
                         one utterance
                              │
        ┌──────────┬──────────┼───────────┬───────────────┐
        ▼          ▼          ▼           ▼               ▼
   Axis 0      Axis 1      Axis 2      Axis 3        orthogonal tags
   Domain      Modality    Structure   MediaType     genre + ContentForm
  (play/       (audio/     (single/    (the leaf:    (adult, anime,
   control/     video/      episodic/   movie,        animation, asmr…;
   not_media)   paged/      continuous/ podcast,      `adult` drives the
                interactive) collection) radio…)       content filter)
```

### Axis 0 — Domain · *is this even a media request?*

`OCPDomain`: `ocp_play` / `ocp_control` / `not_ocp`, mapped to a
`mediavocab.MediaType` sentinel where relevant:

| Domain | Meaning | Leaf sentinel |
|---|---|---|
| `ocp_play` | play some media — refine with the other axes | a real `MediaType` |
| `ocp_control` | control the player (pause/next/seek/…) | `MediaType.CONTROL` |
| `not_ocp` | not a media request at all | `MediaType.NOT_MEDIA` |

The crucial case is **`not_media`**. IoT / device control — _"turn off the
lights"_, _"set the thermostat to 21"_ — is **not media**. It resolves to
`not_media` / `not_ocp` and is handed back to the rest of the OVOS pipeline
(home-automation skills, etc.). The media classifier's job at this axis is to get
out of the way cleanly, not to mis-handle a non-media request.

### Axis 1 — Modality (`playback_type`) · *what physical type is it?*

This is `mediavocab.PlaybackType`: **`audio` / `video` / `paged` / `interactive`**
(plus `unknown`). It is the **coarse, high-confidence axis** — the "physical
type" of the thing being played. It is high-signal because the surface cues that
distinguish audio from video from a page are strong and language-robust ("watch"
vs "listen" vs "read" vs "play"), and because it is the axis a render backend
most directly cares about: an audio sink, a video surface, a paged reader, or an
interactive runtime.

### Axis 2 — Structure · *how is it organised in time?*

`Structure`: **`single` / `episodic` / `continuous` / `collection`** (plus
`unknown`). Orthogonal to modality — it is about temporal shape, not medium:

| Structure | One work that is… | Examples |
|---|---|---|
| `single` | one self-contained unit | a movie, a track, a book |
| `episodic` | a series of discrete instalments | a TV series, a podcast |
| `continuous` | an unbounded live/looping stream | radio, live TV, ambient |
| `collection` | an ordered set of works | a playlist |

Structure changes how a player behaves (resume-from-here, next-episode,
now-playing-from-a-list, never-ends) independently of whether the medium is audio
or video.

### Axis 3 — MediaType leaf · *the concrete type*

`mediavocab.MediaType` — the fine label (`movie`, `music`, `podcast`, `radio`,
`game`, `comic`, …). `mediavocab` is the source of truth for this taxonomy; this
package does not define its own (see [taxonomy.md](taxonomy.md)). The leaf is the
**most specific** and therefore the **least confidently** predicted coordinate —
which is exactly why it sits behind three coarser axes that can prune it.

### Orthogonal tags — genre + ContentForm

Not axes of the product space but free-standing labels attached to the result:

* **genre** — `mediavocab` genre tags (`anime`, `animation`, `asmr`, `adult`, …).
  The **`adult` genre is what the [content filter](content-filtering.md) blocks
  on by default.** Genres are deliberately orthogonal: _"play some hentai"_ is an
  `episodic_series` (leaf) tagged `["anime", "adult"]` — the type never carries
  the adult signal, the genre does.
* **ContentForm** — the editorial/origin distinction the leaf taxonomy folds away
  (a `documentary` and a `silent_movie` are both `MediaType.MOVIE`; a `cartoon`
  and an `anime` are both `episodic_series`). These survive as genre tags so the
  nuance is recoverable for ranking and filtering even though it is not its own
  leaf type. See `LABEL_TO_GENRES` in [taxonomy.md](taxonomy.md).

---

## 3. Why orthogonal axes, not a strict tree

An obvious alternative is a single decision tree: pick audio-vs-video, then
narrow, then narrow again, down to the leaf. Two failures make the tree the wrong
model:

### 3.1 Cross-product "double citizens"

A strict tree forces every leaf to live under exactly one parent. Real media
types belong under several at once:

| Type | Belongs to… | …and to | The tree has to pick one and lie |
|---|---|---|---|
| `music_video` | audio **content** | video **modality** | filed under "video", loses "it's music" |
| `audiobook` | book **origin** | audio **modality** | filed under "audio", loses "it's a book" |
| `audio_drama` | drama/theatre | audio + episodic | one parent can't hold both |

With orthogonal axes there is no contradiction: a music video is simply
`(playback_type=video, structure=single, leaf=music_video, genre=[])` and an
audiobook is `(playback_type=audio, structure=single, leaf=audiobook)`. Each axis
records one true fact; nothing has to be forced into a single bucket.

### 3.2 Unrecoverable coarse-error cascades

In a hard cascade, an early wrong turn is **fatal** — if the audio/video split
fires `video` for _"play the new Adele song"_, the entire audio subtree is
amputated and the leaf can never be `music`. The error at the top silently
caps accuracy at the bottom, and there is no signal that anything went wrong.

Orthogonal heads with **soft gating** avoid this: a coarse axis *down-weights* the
incompatible leaves rather than *deleting* them, and stays **top-k** instead of
top-1. If `playback_type` is `{video: 0.55, audio: 0.45}`, the leaf head still
gets to consider `music`; the coarse prediction shapes the prior, it does not
veto. A confident coarse axis prunes hard; an uncertain one prunes gently.

### 3.3 What the axes buy you

Predicting the coarse axes with independent heads and soft-gating the leaf is:

* **data-efficient** — the coarse axes have few classes and lots of examples per
  class, so they train well on modest data; each one prunes the leaf space, so
  the leaf head solves an easier conditioned problem.
* **robust** — no single early mistake is fatal (3.2); the result degrades into a
  still-useful partial answer instead of a wrong-everywhere one.
* **calibratable** — each axis gets **its own confidence threshold**. The OCP
  pipeline can demand high confidence on `domain` (don't steal a non-media query)
  while tolerating an uncertain leaf.
* **gracefully degradable** — if the leaf is uncertain, **route by modality**: an
  audio request with an unknown leaf can still go to an audio skill. A partial
  classification is actionable; a confidently-wrong leaf is not.

---

## 4. How the axes are produced

There are two production paths. They emit the **same** `MediaClassification`, so
a consumer never has to care which one ran.

### 4.1 Predict coarse-to-fine (the keyword classifier)

The bundled keyword classifier predicts the axes **coarse-to-fine**, in
descending order of signal strength, each from its own `.voc` evidence:

```
modality (PlaybackType)  ◀── VerbAudio / VerbVideo / VerbGame / VerbRead /
                             VerbTune + leaf-family keywords (score & pick best)
structure (Structure)    ◀── ModEpisode / ModSeason / ModLive / Series / Podcast …
        (orthogonal predicted axes — reported in their own right)

leaf MediaType           ◀── leaf-first: the specific-leaf `.voc` chain (most
                             specific first). No specific match + a confident
                             modality → DEFAULT leaf for the (modality, structure) cell.

domain + genres          ◀── classify_domain() / classify_genres()
```

The coarse axes (modality, structure) are predicted from their own voc evidence
and reported as **orthogonal axes** — they are NOT used as a hard gate on the leaf.
The leaf is chosen **leaf-first** (the specific-leaf voc chain): constraining the
leaf to the predicted axes regresses real-query macro-F1, because when modality
prediction is noisy (and on natural language it often is) a hard constraint
suppresses an otherwise-correct leaf. So a predicted axis and the leaf may
legitimately differ — _"watch the news"_ → a news-derived leaf **with** a predicted
**video** modality ("video news"); the player consumes the `playback_type` axis,
the leaf carries the content identity. The `(modality, structure)` **default leaf**
is used only when no specific leaf voc matched at all (audio+single→`music`,
video+continuous→`tv`, …).

> The keyword backend is a deterministic **floor** (~0.29 accuracy on the neutral
> HF split; ~0.99 on the synthetic set is vocabulary coverage, not generalization —
> see [benchmarks](../benchmarks/README.md)). Coarse-to-fine as a hard
> *search-space constraint* pays off in a **trained** classifier (where it prunes a
> probabilistic leaf distribution); in the deterministic keyword matcher it is
> value-neutral on real data, so the axes are predicted for output but the leaf is
> selected leaf-first.

This logic lives in `ovos_media_classifier/keyword.py` (`_predict_modality`,
`_predict_structure`, `_classify_leaf`); the per-axis
accessors (`classify_playback_type`, `classify_structure`, `classify_full`) are
overridden to report the **predicted** axes rather than deriving them from the
leaf. Where a locale lacks the axis vocab the prediction simply finds no evidence
and the classifier degrades to leaf-only matching (the en-us path is the
reference). The leaf-derived defaults (`infer_playback_type` / `infer_structure`
in `axes.py`) remain the fallback for that degraded path and for plugins that
model only the leaf.

The trade-off: the keyword model reads modality/structure from a fixed cue
vocabulary, so it still cannot capture every per-utterance override (e.g. _"play
the **album** Rumours"_ → `collection` needs an album/playlist cue voc). A trained
head with a genuine per-axis confidence is what closes that gap.

### 4.2 Predict each axis with its own head (trained plugins)

A trained classifier plugin (registered under `opm.media.classifier`, see
[external-plugins.md](external-plugins.md)) **MAY** predict each axis with its own
head and soft-gate the leaf, rather than deriving the axes. It does so by
overriding the axis methods on the contract:

```python
class MyTrainedClassifier(AbstractMediaClassifier):
    def classify(self, query, lang, valid_labels=None):
        ...                              # leaf head, conditioned on the coarse heads

    def classify_playback_type(self, query, lang):
        ...                              # dedicated modality head (audio/video/paged/interactive)

    def classify_structure(self, query, lang):
        ...                              # dedicated structure head

    def classify_full(self, query, lang):
        # combine the heads directly instead of deriving from the leaf,
        # soft-gating the leaf on the coarse predictions
        ...
```

A trained head can capture the per-utterance overrides the derivation cannot
(_"the **album**"_ → `collection`, _"the **trailer**"_ → `single`+short) and can
report a genuine per-axis confidence for the pipeline to threshold. The base-class
defaults (derive-from-leaf) are always available as a fallback, so a plugin only
overrides the axes it actually models.

Entity-driven strategies — the NER backend (Aho-Corasick exact match) and any
classifier that uses the user's known entities as categorical features — read from
the *same* **entity lists** (`label → list of strings`). That shared store, its
source specs and the perf/memory tradeoff are documented in
[entity-lists.md](entity-lists.md).

---

## 5. MediaType → default (playback_type, structure)

The defaults below are what the derive-from-leaf path produces. `playback_type`
comes from `mediavocab.infer_playback_type`; `structure` comes from
`MEDIA_TYPE_TO_STRUCTURE` in `ovos_media_classifier/axes.py`. This table is the
full enumeration of `mediavocab.MediaType`.

| MediaType | playback_type | structure |
|---|---|---|
| `movie` | `video` | `single` |
| `short_film` | `video` | `single` |
| `episodic_series` | `video` | `episodic` |
| `tv` | `video` | `continuous` |
| `music` | `audio` | `single` |
| `music_video` | `video` | `single` |
| `podcast` | `audio` | `episodic` |
| `audiobook` | `audio` | `single` |
| `audio_drama` | `audio` | `episodic` |
| `radio` | `audio` | `continuous` |
| `book` | `paged` | `single` |
| `comic` | `paged` | `single` |
| `game` | `interactive` | `single` |
| `interactive_fiction` | `interactive` | `single` |
| `sound_effect` | `audio` | `single` |
| `procedural_ambient` | `audio` | `continuous` |
| `playlist` | `unknown` | `collection` |
| `generic` | `unknown` | `unknown` |
| `not_media` | `unknown` | `unknown` |
| `control` | `unknown` | `unknown` |

Notes worth reading off the table:

* **`music_video`** is the canonical double-citizen — `video` modality, but its
  content is music. The tree can't express both at once; the axes do (the leaf
  name itself records the music side, the `playback_type` axis the video side).
* **`audiobook`** is `audio` modality with a **book** origin — again two truths,
  one per axis.
* **`playlist`** has no intrinsic modality (`unknown`) because a playlist can mix
  audio and video; its defining axis is `structure=collection`. A trained
  modality head can still resolve it per-utterance ("my workout *playlist*" →
  `audio`).
* The sentinels (`generic`, `not_media`, `control`) are `unknown` on the content
  axes by construction — there is no media to describe.

---

## 6. Structure is intrinsic-by-default, override-by-model

`Structure` is defined in this package (not `mediavocab`) because it is a
classification axis, not a vocabulary term. The default map in `axes.py` encodes
the *intrinsic* structure of each type:

```python
from ovos_media_classifier import Structure, infer_structure, MEDIA_TYPE_TO_STRUCTURE
from mediavocab import MediaType

infer_structure(MediaType.PODCAST)   # Structure.EPISODIC
infer_structure(MediaType.RADIO)     # Structure.CONTINUOUS
infer_structure(MediaType.PLAYLIST)  # Structure.COLLECTION
```

A trained model MAY override per utterance — _"play the **album** Rumours"_ is a
`music` leaf whose structure should be `collection`, not the `single` default.
That is precisely the kind of nuance a dedicated structure head adds on top of the
derived default (§4.2).

---

## 7. The `MediaClassification` result

All axes land in one dataclass (`ovos_media_classifier/axes.py`), returned by
`classify_full()`:

```python
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier()
result = clf.classify_full("play the breaking bad tv series", "en-us")

result.media_type      # <MediaType.EPISODIC_SERIES: 'episodic_series'>   (Axis 3)
result.playback_type   # <PlaybackType.VIDEO: 'video'>                    (Axis 1)
result.structure       # <Structure.EPISODIC: 'episodic'>                 (Axis 2)
result.domain          # <OCPDomain.OCP_PLAY: 'ocp_play'>                 (Axis 0)
result.genres          # []                                              (tags)
result.confidence      # 0.6
result.as_dict()
# {'media_type': 'episodic_series', 'playback_type': 'video', 'structure': 'episodic',
#  'domain': 'ocp_play', 'genres': [], 'confidence': 0.6, 'control_intent': None}
```

A non-media request collapses every content axis cleanly:

```python
clf.classify_full("what time is it", "en-us").as_dict()
# {'media_type': 'generic', 'playback_type': 'unknown', 'structure': 'unknown',
#  'domain': 'not_ocp', 'genres': [], 'confidence': 0.0, 'control_intent': None}
```

(The keyword classifier reports `not_ocp` / `generic` because no media keyword
matches; the content axes are `unknown` and the request is routed away from OCP.)

---

## 8. Multi-head training data

The axes are not just an inference-time convenience — the **dataset carries them
as columns** so backends can train one head per axis. The canonical
`TigreGotico/ocp-media-intents` dataset (built by `training/build_and_publish.py`)
ships, per row:

| Column | Axis | Derivation |
|---|---|---|
| `mediavocab_type` | Axis 3 (leaf) | `LABEL_TO_MEDIA_TYPE[media_label]` |
| `playback_type` | Axis 1 (modality) | `infer_playback_type(mediavocab_type)` |
| `structure` | Axis 2 (structure) | `infer_structure(mediavocab_type)` |
| `genres` | tags | `LABEL_TO_GENRES[media_label]` (`;`-joined; carries `adult`) |

Because every coarse column is derived from the same `infer_*` functions the
runtime uses (§4.1), a model trained on these columns and a model that derives at
runtime agree by construction on the defaults — and a multi-head model is free to
learn the per-utterance overrides on top. See [taxonomy.md](taxonomy.md) for the
label → type/genre projection and `training/README` for the dataset build.

---

## See also

* [taxonomy.md](taxonomy.md) — `mediavocab` enforcement and the raw label →
  type/genre projection (the leaf-axis machinery).
* [stable-api.md](stable-api.md) — `classify_full`, `classify_playback_type`,
  `classify_structure` and the rest of the contract.
* [content-filtering.md](content-filtering.md) — how the `adult` genre tag drives
  detect-to-block moderation.
* [external-plugins.md](external-plugins.md) — registering a trained,
  multi-head classifier under `opm.media.classifier`.
