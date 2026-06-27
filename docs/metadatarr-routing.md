# metadatarr-backed routing layers (the open-vocab gap)

The keyword (`.voc`) backend is a high-precision first pass on **explicit cues**
("play the *movie* X", "*watch* Y"). It abstains to GENERIC on a **bare real
title** it has no cue for — "attack on titan", "play interstellar", "stream the
lakers game". That open-vocab gap is filled by two metadatarr-backed layers.

> **Not circular.** A `MediaProvider` returns a playable **stream** for a title;
> **metadatarr returns METADATA** — given a title it resolves the canonical
> `medium` / external IDs. Routing "what is this?" through metadatarr to pick
> which stream-providers to call is a different system than the one that plays
> it.

The hybrid routes cheapest → most expensive, and **a later layer only ever fills
an abstention** — it never overrides a confident keyword route nor moves the
gate (adult-leak / false-hijack stay exactly at the keyword floor):

1. **keyword** — explicit cues; owns the gate + adult policy.
2. **embedding router + user library + Layer A offline gazetteer** — offline,
   fills keyword's abstains for common real titles.
3. **online metadatarr (Layer B)** — network, opt-in, last resort.

---

## Layer A — offline popularity gazetteer

A popularity-ranked gazetteer of the **most common real titles per media type**,
injected into the embedding router's entity matcher as a **default offline
library** (in addition to the user's personal library). It lets the hybrid route
a common bare title with **no network call and no user setup** — "play
interstellar" → MOVIE, "put on naruto" → EPISODIC_SERIES.

### Sources (TigreGotico HF metadata collections)

The authoritative builder (`data/build_gazetteer_hf.py`, needs the
`[huggingface]` extra) draws from the per-source, type-correct HF datasets — each
is a single media type with real popularity signals and built-in adult flags, so
there is **no cross-contamination** and adult content is excluded at the source:

| label | dataset | rank by | adult filter |
|---|---|---|---|
| `movie_title` | `media-metadata-imdb-titles` (+`-imdb-ratings`) | `num_votes` | `is_adult` |
| `tv_show_title` | imdb-titles (tvSeries) + `media-metadata-tvmaze-shows` | `num_votes` / `rating_average` | `is_adult` |
| `anime_title` | `media-metadata-anilist-anime` | `popularity` | `is_adult` |
| `game_title` | `media-metadata-steam-games` | `positive_reviews` | — |
| `artist_name` | `media-metadata-musicbrainz-artists` | (alpha) | — |
| `podcast_title` | `media-metadata-podcastindex-podcasts` | `episode_count` | `explicit` |
| `audiobook_title` | `media-metadata-librivox-audiobooks` | (alpha) | — |
| `book_title` | `media-metadata-openlibrary-books` | (alpha) | — |

(A `data/build_gazetteer.py` fallback builds from the local `data/entities/*.csv`
pools when HF is unavailable; those are noisier and require contamination
subtraction.)

The generated full gazetteer lands at `data/gazetteer.json` (gitignored); a small
top-N default ships in the wheel at `ovos_media_classifier/data/gazetteer.json`.

### Safety

* **No adult labels** — `adult_title` / `pornstar` / `hentai_title` are never in
  the gazetteer; adult routing stays exclusively on the keyword content-policy
  axis (0.0 leak).
* **Single-word common-word titles dropped** — a one-word title that is a common
  English word ("Legend", "Seven", "Serial") would hijack any utterance
  containing that word; these are filtered (multi-word and uncommon one-word
  titles like "Naruto" survive).
* **Ambiguity → abstain** — a title that fires several labels mapping to
  different media types ("Dune" movie/tv/book, "Watchmen" movie/comic) cannot be
  disambiguated, so the gazetteer abstains rather than guess a wrong leaf.
* **Gazetteer tier ≠ override** — gazetteer entities (lower precision than the
  user's own library) only fill keyword abstains; they never override a confident
  keyword route. The user's injected library still overrides.

### Live size MUST be bounded (latency)

The entity matcher is a word-boundary regex whose **per-query cost grows with the
injected title count**, so the LIVE gazetteer must stay bounded. Measured on the
routing eval (`benchmarks/routing_eval.py --latency-sweep`):

| cap / type | titles | median | p95 |
|---|---|---|---|
| 0 (off) | 0 | 0.48 ms | 0.84 ms |
| 500 | 4.5k | 0.87 ms | 2.6 ms |
| **1000 (default)** | **9k** | **1.3 ms** | **4.4 ms** |
| 5000 | 45k | 4.8 ms | 23 ms |
| 10000 | 120k | 19 ms | 83 ms |
| 50000 | 513k | 117 ms | 383 ms |
| 100000 | ~1M | 222 ms | 919 ms |

The knee is ~500–1000/type. The default cap is **1000/type** (p95 < 5 ms, good
common-title recall). The long tail is the **online layer's** job, not a giant
local set.

> **Live routing = small bounded set** (user library + ~1000/type gazetteer).
> **Offline tagging = large sets fine** (the full 1M-artist MusicBrainz set is
> productive for tagging entities in text to build datasets, NOT for live OCP).
> Configuring more than 50k live titles logs a latency warning.

### Config

| key | default | meaning |
|---|---|---|
| `media_classifier_gazetteer` | `true` | inject the default offline gazetteer |
| `media_classifier_gazetteer_size` | `1000` | per-type cap (`0`/null = no cap — offline only) |

---

## Layer B — online metadatarr (network, opt-in, OFF by default)

`MetadatarrMediaClassifier` resolves a bare title via `metadatarr.resolve` and
maps the resolved `medium` → `MediaType` (and `programme_format` /
`content_genres` / `year` / `playback_type` into the `Signals`). It is the
hybrid's **last layer**, consulted only when keyword + offline both abstain.

### Robustness

`classify` is wrapped in a wall-clock **timeout + try/except**: on timeout,
network failure, empty resolve, or low-confidence it returns GENERIC — it
**abstains, never raises, never blocks**. metadatarr is **lazy-imported** inside
the call so a runtime with the layer disabled never loads it.

### Latency — NOT for the live OCP pipeline

A real metadatarr resolve fans out to many providers and takes **~8–13 s** per
title. That is far above any live-pipeline budget, so Layer B is **off by
default** and is intended for **offline entity tagging / dataset prep**, or a
**long-running mission-critical agent** that can tolerate the latency — not the
interactive OCP pipeline. Keep it disabled for live OCP; rely on keyword + the
bounded offline gazetteer there.

### Config / install

| key | default | meaning |
|---|---|---|
| `media_classifier_online_metadatarr` | `false` | enable the online layer |
| `media_classifier_online_timeout` | `4.0` | per-query wall-clock budget (s) |
| `media_classifier_online_min_confidence` | `0.5` | min resolved confidence to route |

```
pip install ovos-media-classifier[online]    # adds the metadatarr dep
```
