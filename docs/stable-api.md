# Stable API

The public, stable surface is `load_media_classifier()`, the four classification
methods on the returned object, `ContentFilter`, and the
`AbstractMediaClassifier` contract that the bundled classifier and external plugins
implement.

## The classifier object

`load_media_classifier(config=None, voc_match_func=None)` returns an
`AbstractMediaClassifier` — in this release the bundled keyword classifier, unless
`config["media_classifier_plugin"]` selects an external plugin. Every classifier
exposes the same four methods.

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
domain from `classify()` (non-`GENERIC` type ⇒ `OCP_PLAY`, else `NOT_OCP`). A plugin
with a dedicated domain or control head can override it for better accuracy and to
detect `OCP_CONTROL`.

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

Override guidance (for plugin authors — the bundled keyword classifier only
overrides `classify_genres()`):

- Override `classify_domain()` when you have a cheap domain head.
- Override `is_ocp_query()` when you also handle `ocp_control`, so control commands
  count as OCP queries.
- Override `classify_genres()` when you can surface genre signal so the
  [content filter](content-filtering.md) can block on it.

## Return types

| Type | Source | Notes |
|---|---|---|
| `MediaType` | re-exported from `mediavocab` | string-Enum; the enforced public taxonomy |
| `OCPDomain` | `ovos_media_classifier` | `ocp_play` / `ocp_control` / `not_ocp` |
| genre tags | `list[str]` | members of `mediavocab` `KNOWN_GENRES` |
| confidence | `float` | in `[0, 1]` |

`OCPPlayIntent`, `OCPControlIntent`, and `OCPEntityLabel` are exported for
backends and tooling but are **internal** label spaces, not part of the public
classification output — that output is always `mediavocab.MediaType` + genres. See
[taxonomy.md](taxonomy.md).

## Content filter

`ContentFilter(config=None)` exposes:

- `check(classifier, query, lang="en-us") -> (bool, str)` — classify and apply the
  policy in one call.
- `is_blocked(media_type, genres=None) -> (bool, str)` — apply the policy to an
  already-computed result.

See [content-filtering.md](content-filtering.md).
