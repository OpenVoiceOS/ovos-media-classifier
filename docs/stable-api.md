# Stable API

The public, stable surface is `load_media_classifier()`, the classification
methods on the returned object (the per-axis methods plus `classify_full`),
`ContentFilter`, and the `AbstractMediaClassifier` contract that the bundled
classifier and external plugins implement.

For *why* the output is split into orthogonal axes, see
[classification-model.md](classification-model.md); this page is the method
reference.

## The classifier object

`load_media_classifier(config=None, voc_match_func=None)` returns an
`AbstractMediaClassifier` — the bundled keyword classifier by default, or the
[NER / ONNX / external backend](backends.md) the config selects. Every classifier
exposes the same methods below.

### `classify(query, lang, valid_labels=None) -> (MediaType, float)`

Returns the most likely `mediavocab.MediaType` for an `ocp_play` request and a
confidence in `[0, 1]`. Returns `(MediaType.GENERIC, 0.0)` when nothing matches.

- `query` — the user utterance.
- `lang` — a BCP-47 language tag (e.g. `"en-us"`), assumed already standardised.
- `valid_labels` — optional list of `MediaType`; when given, only one of these is
  returned (otherwise `GENERIC`).

```python
clf.classify("play a movie", "en-us")        # (<MediaType.MOVIE: 'movie'>, 0.6)
```

### `classify_genres(query, lang) -> list[str]`

Returns the `mediavocab` genre tags implied by the query (default `[]`). Genres are
orthogonal to `MediaType` and are what the content filter blocks on. The keyword
classifier overrides this to surface genre from the matched `.voc` files.

```python
clf.classify_genres("play some anime", "en-us")   # ['anime']
```

### `classify_domain(query, lang) -> (OCPDomain, float)`

Returns the top-level domain: `OCP_PLAY`, `OCP_CONTROL`, or `NOT_OCP`, with a
confidence. The default implementation (used by the keyword classifier) derives the
domain from `classify()` (non-`GENERIC` type ⇒ `OCP_PLAY`, else `NOT_OCP`). A
backend with a dedicated domain or control head overrides it for better accuracy
and to detect `OCP_CONTROL`.

```python
clf.classify_domain("play a podcast", "en-us")    # (<OCPDomain.OCP_PLAY: 'ocp_play'>, 0.6)
```

### `is_ocp_query(query, lang) -> (bool, float)`

Returns whether the query targets OCP at all (play **or** control), with a
confidence. The default implementation delegates to `classify_domain()` (`True`
when the domain is not `NOT_OCP`).

```python
clf.is_ocp_query("what is the weather", "en-us")  # (False, 0.0)
```

## Multi-axis methods

These return the coarse axes of the [multi-axis
model](classification-model.md). On the bundled keyword classifier they are
**derived** from the leaf `MediaType`; a trained backend MAY override each to
predict it with its own head.

### `classify_playback_type(query, lang) -> PlaybackType`

Returns the modality axis as a `mediavocab.PlaybackType`
(`audio` / `video` / `paged` / `interactive` / `unknown`). Default: derived from
`classify()` via `mediavocab.infer_playback_type`.

```python
clf.classify_playback_type("play a podcast", "en-us")   # <PlaybackType.AUDIO: 'audio'>
```

### `classify_structure(query, lang) -> Structure`

Returns the structure axis as a `Structure`
(`single` / `episodic` / `continuous` / `collection` / `unknown`). Default:
derived from `classify()` via `infer_structure`.

```python
clf.classify_structure("put on the radio", "en-us")     # <Structure.CONTINUOUS: 'continuous'>
```

### The mediavocab descriptive axes

The classifier emits its descriptive signals in **mediavocab's own taxonomy**, so
`to_signals` is lossless and providers consume one shared vocabulary.

- `classify_content_form(query, lang) -> ContentForm | None` — the experiential
  kind: `trailer` / `teaser` / `behind_scenes` / `excerpt` / `supplement` / …
  (the parent `MediaType` stays the feature; `None` ⇒ a primary work).
- `classify_programme_format(query, lang) -> ProgrammeFormat | None` — the
  structural format: `documentary` / `news` / `concert` / `stand_up` / `sports` / …
- `classify_accessibility(query, lang) -> list[AccessibilityKind]` — requested
  accessibility assets: `subtitles` / `audio_description` / `sign_language` / …
- `classify_variant(query, lang) -> VariantKind | None` — the work-level cut:
  `directors` / `extended` / `remastered` / `colorized` / `fanedit` / …
- `classify_genres(query, lang) -> list[str]` — genre tags constrained to
  `mediavocab.KNOWN_GENRES` (the content filter reads `adult` / `anime` / … here).
- `classify_picture_format(query, lang) -> list[PictureFormat]` — the
  `mediavocab.PictureFormat` presentation attributes (`black_and_white` /
  `silent` / `3d`), multi-label.

```python
clf.classify_content_form("watch the Dune trailer", "en-us")  # <ContentForm.TRAILER: 'trailer'>
clf.classify_programme_format("a documentary about whales", "en-us")  # <ProgrammeFormat.DOCUMENTARY: 'documentary'>
```

### `classify_full(query, lang, player_status=None, ner_list=None) -> MediaClassification`

Returns **all axes at once** in a `MediaClassification` dataclass: `media_type`,
`playback_type`, `structure`, `domain`, `genres`, `confidence`. Default: runs
`classify`/`classify_domain`/`classify_genres` once and derives the coarse axes
from the leaf. A backend with dedicated heads SHOULD override this to predict the
axes directly and soft-gate the leaf.

The two minimal inputs are `query` and `lang`. Both context arguments are
**optional** (default `None` = context-free behaviour):

- `player_status` — a `PlayerStatus` (now-playing `media_type` + play/pause/stop).
  Enables relative / control follow-ups (*"next"* / *"pause"* → `OCP_CONTROL`
  even with no media keyword; *"play something else"* → a re-query biased to the
  current type) and a light type bias on ambiguous follow-ups — applied
  conservatively, never overriding a confident explicit route.
- `ner_list` — `{ner_label: [entity, …]}` of the entities the user actually has
  (skill-registered keywords + library). The entity context for NER matching and
  the embedding router's runtime injection, threaded per query so a caller passes
  the live list with no retraining.

```python
clf.classify_full("play the breaking bad tv series", "en-us").as_dict()
# {'media_type': 'episodic_series', 'playback_type': 'video', 'structure': 'episodic',
#  'domain': 'ocp_play', 'genres': [], 'confidence': 0.6, 'control_intent': None}

# context-aware: now-playing status + the user's live entity lists
clf.classify_full("play something else", "en-us",
                  player_status=status, ner_list={"artist_name": ["Radiohead"]})
```

See [contextual-classification.md](contextual-classification.md) for how
`player_status` and `ner_list` shape the result.

## The `AbstractMediaClassifier` contract

This ABC is both the interface the bundled keyword classifier implements and the
**plugin contract** for external classifiers discovered via the
`opm.media.classifier` entry-point group (see [external-plugins.md](external-plugins.md)).

| Method | Required? | Default behaviour |
|---|---|---|
| `classify` | **abstract** — must implement | — |
| `classify_domain` | optional override | derives from `classify()` |
| `is_ocp_query` | optional override | derives from `classify_domain()` |
| `classify_genres` | optional override | returns `[]` |
| `classify_playback_type` | optional override | derives from `classify()` |
| `classify_structure` | optional override | derives from `classify()` |
| `classify_content_form` | optional override | returns `None` |
| `classify_programme_format` | optional override | returns `None` |
| `classify_accessibility` | optional override | returns `[]` |
| `classify_variant` | optional override | returns `None` |
| `classify_picture_format` | optional override | returns `[]` (`mediavocab.PictureFormat`) |
| `classify_full` | optional override | combines the above (derive-from-leaf) |

Override guidance (for plugin authors — the bundled keyword classifier only
overrides `classify_genres()` and derives every coarse axis from the leaf):

- Override `classify_domain()` when you have a cheap domain head.
- Override `is_ocp_query()` when you also handle `ocp_control`, so control commands
  count as OCP queries.
- Override `classify_genres()` when you can surface genre signal so the
  [content filter](content-filtering.md) can block on it.
- Override `classify_playback_type()` / `classify_structure()` (and `classify_full()`)
  when you have dedicated coarse-axis heads — predict each axis directly and
  soft-gate the leaf instead of deriving the axes from it (see
  [classification-model.md](classification-model.md)).

## Return types

| Type | Source | Notes |
|---|---|---|
| `MediaType` | re-exported from `mediavocab` | string-Enum; the enforced public taxonomy (the leaf axis) |
| `PlaybackType` | `mediavocab` | string-Enum; the modality axis (`audio`/`video`/`paged`/`interactive`/`unknown`) |
| `Structure` | `ovos_media_classifier` | string-Enum; the structure axis (`single`/`episodic`/`continuous`/`collection`/`unknown`) |
| `OCPDomain` | `ovos_media_classifier` | `ocp_play` / `ocp_control` / `not_ocp` |
| `MediaClassification` | `ovos_media_classifier` | dataclass holding all axes + genres + confidence; `.as_dict()` for a plain dict |
| `ContentForm` | `mediavocab` | string-Enum; experiential kind (`trailer`/`behind_scenes`/`excerpt`/…) |
| `ProgrammeFormat` | `mediavocab` | string-Enum; structural format (`documentary`/`news`/`concert`/…) |
| `AccessibilityKind` | `mediavocab` | string-Enum; a11y asset (`subtitles`/`audio_description`/…) |
| `VariantKind` | `mediavocab` | string-Enum; work-level cut (`directors`/`extended`/`remastered`/…) |
| genre tags | `list[str]` | members of `mediavocab` `KNOWN_GENRES` |
| confidence | `float` | in `[0, 1]` |

`OCPControlIntent` and `OCPEntityLabel` are exported for backends and tooling but
are **internal** label spaces, not part of the public classification output —
that output is always `mediavocab.MediaType` + genres. The raw-label →
`(MediaType, genres)` maps (`LABEL_TO_MEDIA_TYPE`, `LABEL_TO_GENRES`,
`NER_LABEL_TO_MEDIA_TYPE`, `NER_LABEL_TO_GENRES`) are exported for the same use.
See [taxonomy.md](taxonomy.md).

## Content filter

`ContentFilter(config=None)` exposes:

- `check(classifier, query, lang="en-us") -> (bool, str)` — classify and apply the
  policy in one call.
- `is_blocked(media_type, genres=None) -> (bool, str)` — apply the policy to an
  already-computed result.

See [content-filtering.md](content-filtering.md).
