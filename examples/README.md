# examples

Small, runnable, heavily-commented scripts that exercise the public
`ovos-media-classifier` API. Each file is standalone — run any of them with:

```bash
python examples/01_quickstart.py
```

They use only the bundled `.voc` keyword classifier, so there is nothing to
download and no model files: the examples run fully offline with just
`ovos-media-classifier` (and its `ovos-utils` + `mediavocab` dependencies)
installed.

> **Not shipped in the wheel.** This directory is a top-level folder, outside
> the `ovos_media_classifier` package. The wheel only includes
> `ovos_media_classifier*` (see `[tool.setuptools.packages.find]` in
> `pyproject.toml`), so these examples are *not* packaged — they live in the
> source checkout for learning and reference only.

## Index

| # | File | Demonstrates |
|---|------|--------------|
| 01 | [`01_quickstart.py`](01_quickstart.py) | `load_media_classifier()` then `classify()` on a few utterances → `(MediaType, confidence)`. The minimal end-to-end call. |
| 02 | [`02_multi_axis.py`](02_multi_axis.py) | `classify_full()` → the full `MediaClassification` (media_type, playback_type, structure, domain, genres, confidence) across music/movie/tv-show/anime/live-tv/podcast/radio/game, showing how the orthogonal axes prune together. |
| 03 | [`03_content_filter.py`](03_content_filter.py) | `ContentFilter().check(clf, query, lang)`: adult blocked by default, `allow_adult_content: true` lifting it, and a custom `blocked_genres`; plus the lower-level `is_blocked()`. |
| 04 | [`04_domain_and_is_ocp.py`](04_domain_and_is_ocp.py) | `classify_domain()` / `is_ocp_query()` separating media requests from non-media utterances (`"what time is it"`, `"turn on the lights"` → `not_ocp`). |
| 05 | [`05_writing_a_plugin.py`](05_writing_a_plugin.py) | A minimal in-file `AbstractMediaClassifier` subclass, the `opm.media.classifier` entry-point `pyproject.toml` snippet, and loading by name via `load_media_classifier({"media_classifier_plugin": "..."})`. |
| 06 | [`06_pipeline_voc_match.py`](06_pipeline_voc_match.py) | Pipeline mode: `load_media_classifier(voc_match_func=...)` supplying an external `.voc` matcher (what an OCP pipeline plugin does with its own `voc_match`). |
| 07 | [`07_playback_routing.py`](07_playback_routing.py) | Using `playback_type` (audio/video/paged/interactive) + `structure` to drive a small illustrative play-dispatcher. |
| 08 | [`08_genres_and_tags.py`](08_genres_and_tags.py) | `classify_genres()` surfacing `anime` / `animation` / `asmr` / `adult` tags — the orthogonal signal the content filter blocks on. |

## Public API touched

- `load_media_classifier(config=None, voc_match_func=None)` — factory.
- `AbstractMediaClassifier` — the classifier / plugin contract:
  `classify`, `classify_domain`, `is_ocp_query`, `classify_genres`,
  `classify_playback_type`, `classify_structure`, `classify_full`.
- `ContentFilter(config).check(clf, query, lang)` / `.is_blocked(media_type, genres)`.
- `MediaClassification` + `Structure` (`ovos_media_classifier.axes`).
- `MediaType` / `PlaybackType` (re-exported from `mediavocab`).
