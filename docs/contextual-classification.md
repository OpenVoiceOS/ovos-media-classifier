# Contextual classification — available media biases prediction

A media request is ambiguous in isolation. *"Play Halo"* could be a game, a
soundtrack, or a Halsey song; *"watch the office"* depends on whether you own the
series. The most reliable disambiguator is not a linguistic cue — it is **what
media you actually have**. This classifier makes that signal first-class:
prediction is **influenced and biased by the media available to you**.

The mechanism is **keyword feature slots**.

## Keyword feature slots

A *slot* is a named entity category mapped to a point on the
[taxonomy](classification-model.md) — a `mediavocab.MediaType` plus its playback
type and any genre tags:

| slot label | media type | playback | genres |
|---|---|---|---|
| `artist_name`, `track_name`, `album_name`, `music_genre` | `music` | audio | |
| `movie_title`, `movie_actor`, `movie_director` | `movie` | video | |
| `tv_show_title`, `anime_title`, `cartoon_title` | `episodic_series` | video | (`anime`) |
| `podcast_title`, `podcast_host` | `podcast` | audio | |
| `audiobook_title`, `audiobook_author` | `audiobook` | audio | |
| `radio_station`, `radio_genre` | `radio` | audio | |
| `game_title`, `game_platform` | `game` | interactive | |
| `hentai_title`, `adult_title`, `pornstar` | … | … | `adult` |

The full set (~85 slots) is `ovos_media_classifier.KEYWORD_FEATURE_SLOTS`
(`slots.py`), derived from the canonical taxonomy maps so there is one source of
truth. A slot is only a **definition** — it carries no entities. Entities are
filled in at runtime.

```python
from ovos_media_classifier import slot_for_label, slots_for_media_type
slot_for_label("artist_name")            # KeywordFeatureSlot(media_type=MUSIC, playback=audio, …)
slots_for_media_type(MediaType.MUSIC)    # every music slot
```

## Runtime filling — your library is the context

Slots are populated from whatever media you actually have, via
[entity lists](entity-lists.md) (`EntitiesContainer`):

- **Jellyfin** — `load_jellyfin(url, api_key)` fills `movie_title`, `tv_show_title`,
  `artist_name`, `album_name`, … from your library.
- **the \*arr stack** — `load_radarr` (movies), `load_sonarr` (series),
  `load_lidarr` (music) fill the matching slots from what you collect.
- **Music Assistant** — `load_music_assistant(url)`.
- **files / HF** — `.csv`/`.tsv`/`.jsonl`/HuggingFace for static rosters.
- **at runtime from the pipeline** — `container.add(label, entity)` as skills/
  providers announce what they can serve.

A slot you never fill contributes nothing; a slot full of *your* titles becomes a
strong, grounded signal.

## Two classifiers, one contract

The same slots drive both strategies:

1. **NER (exact match)** — a query containing a slot's entity resolves to that
   slot's media type directly. *"play Inception"* → `MOVIE` **because Inception is
   in your Radarr**, not because of a keyword guess. High confidence, zero
   ambiguity, language-agnostic. See [backends](backends.md).
2. **Guided embeddings (learned features)** — each slot is a categorical feature
   ("does the utterance contain a known `artist_name`?"). A model learns weights
   over the slots; at runtime the slots are filled with your available entities,
   so the features — and the prediction — are **biased toward the media you own**.
   This is the [`guided-categorical-embeddings`](backends.md) pattern: define the
   feature labels statically, fill the samples at runtime.

## Why this is the *right* constraint

The bundled keyword classifier predicts coarse axes (modality, structure) from
linguistic cues, but it does **not** use them to hard-constrain the media-type
leaf — empirically a guessed modality constraint *regressed* real-query accuracy
(see [the model](classification-model.md) §4.1). Available media is the opposite:
a **grounded** constraint. Where a noisy "watch → video" guess can suppress a
correct answer, "this title is in your library" rarely lies. Contextual,
inventory-driven bias is therefore the productive way to lift the keyword/NER
floor — and it is exactly what plugs the classifier into Jellyfin and the *arr
stack.
