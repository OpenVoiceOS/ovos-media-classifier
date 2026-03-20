# Language Support

This document describes which languages are supported by each backend,
how to verify coverage, and how to add a new language.

---

## Overview

`ovos-media-classifier` is a multilingual package.  Language support varies
by backend:

| Backend | How language is handled |
|---|---|
| Keyword | Per-language `.voc` files in `locale/<lang>/` |
| Padatious | Per-language `.intent` files in `locale/<lang>/` |
| AhocorasickNER | Language-agnostic (substring matching) |
| sklearn | Language-agnostic at inference; multilingual training data improves coverage |
| Model2Vec | Language-agnostic at inference; multilingual training data improves coverage |

---

## Bundled locale coverage

The bundled `locale/` directory (ported from `ovos-ocp-pipeline-plugin`) covers
13 language tags:

| Language tag | Language |
|---|---|
| `ca-es` | Catalan (Spain) |
| `da-dk` | Danish (Denmark) |
| `de-de` | German (Germany) |
| `en-us` | English (United States) |
| `es-es` | Spanish (Spain) |
| `eu` | Basque |
| `fr-fr` | French (France) |
| `gl-es` | Galician (Spain) |
| `it-it` | Italian (Italy) |
| `nl-nl` | Dutch (Netherlands) |
| `pl-pl` | Polish (Poland) |
| `pt-br` | Portuguese (Brazil) |
| `pt-pt` | Portuguese (Portugal) |

Each language directory contains the same set of vocabulary and intent files
(see the full list below).

---

## Files in each locale directory

### Vocabulary files (`.voc`) — used by the keyword backend

These are plain text files, one keyword or phrase per line.  The keyword
backend checks whether any entry appears as a substring in the lowercased query.

| File | What it detects |
|---|---|
| `MusicKeyword.voc` | Music queries |
| `MovieKeyword.voc` | Movie queries |
| `TVKeyword.voc` | TV show queries |
| `SeriesKeyword.voc` | Series / episode queries |
| `PodcastKeyword.voc` | Podcast queries |
| `RadioKeyword.voc` | Radio queries |
| `AudioBookKeyword.voc` | Audiobook queries |
| `NewsKeyword.voc` | News queries |
| `AudioKeyword.voc` | Generic audio queries |
| `VideoKeyword.voc` | Generic video queries |
| `AnimeKeyword.voc` | Anime queries |
| `CartoonKeyword.voc` | Cartoon queries |
| `DocumentaryKeyword.voc` | Documentary queries |
| `GameKeyword.voc` | Game queries |
| `ShortKeyword.voc` | Short film qualifier |
| `SilentKeyword.voc` | Silent movie qualifier |
| `BWKeyword.voc` | Black-and-white movie qualifier |
| `AudioDramaKeyword.voc` | Radio theatre / audio drama queries |
| `ASMRKeyword.voc` | ASMR queries |
| `ADKeyword.voc` | Audio description queries |
| `AdultKeyword.voc` | Adult content queries |
| `HentaiKeyword.voc` | Hentai queries |
| `ComicBookKeyword.voc` | Visual story / comic book queries |
| `TrailerKeyword.voc` | Trailer queries |
| `Play.voc` | Play trigger words |
| `Resume.voc` | Resume trigger words |
| `Alerts.voc` | Alert-related words |
| `Parrot.voc` | Parrot/repeat trigger words |
| `SoundIntents.voc` | Sound-only intent indicators |
| `audio_only.voc` | Audio-only mode keywords |
| `video_only.voc` | Video-only mode keywords |

### Intent files (`.intent`) — used by the padatious backend

Padatious-style utterance patterns with optional slots (`{query}`, `{artist}`,
etc.).  The padatious backend uses these as training samples.

| File | Intent / control action |
|---|---|
| `play.intent` | Generic play request |
| `pause.intent` | Pause playback |
| `resume.intent` | Resume playback |
| `next.intent` | Next track / episode |
| `prev.intent` | Previous track / episode |
| `media_stop.intent` | Stop playback |
| `open.intent` | Open media player |
| `like_song.intent` | Mark current song as liked |
| `play_favorites.intent` | Play favourites |
| `save_game.intent` | Save game state |
| `load_game.intent` | Load saved game |
| `read.intent` | Read / audiobook request |
| `featured.intent` | Play featured / recommended content |

### Dialog files (`.dialog`) — informational only

Dialog files contain spoken responses for the pipeline plugin.  They are not
used by the classifier itself but are included in the locale directory for
completeness.

| File | When used |
|---|---|
| `play.what.dialog` | Pipeline asks "what should I play?" |
| `just.one.moment.dialog` | Pipeline says "just a moment" |
| `cant.play.dialog` | Pipeline says it cannot play the requested media |
| `no.media.skills.dialog` | No OCP skill is available |

---

## How the keyword backend resolves languages

The `_VocMatcher` class tries locale directories in this order:

1. Exact BCP-47 tag: `locale/en-us/MusicKeyword.voc`
2. Base language: `locale/en/MusicKeyword.voc`

If neither exists, the vocabulary is empty and no keyword match is returned for
that vocab name.  There is **no English fallback** — a missing file means no
match, not a fallback to English.  This is intentional: returning English
keywords for a non-English query would cause false positives.

---

## Adding a new language

### Step 1 — Create the locale directory

```bash
mkdir ovos_media_classifier/locale/sv-se
```

### Step 2 — Add vocabulary files

Create a `.voc` file for each media type.  Each file should contain the most
common ways a Swedish speaker would refer to that media type, one per line:

```
# MusicKeyword.voc
musik
låt
spellista
```

Vocabulary entries can be:
- Single words: `musik`
- Phrases: `spela musik`
- They are matched as **substrings** (case-insensitive), so short entries will
  match more broadly.  Avoid entries so short they cause false positives.

### Step 3 — Add intent files (for padatious backend)

Create `.intent` files with padatious patterns:

```
# play.intent
spela {query}
sätt på {query}
```

Curly-brace slots (`{query}`) are optional entity captures — padatious learns
to generalise across the pattern.

### Step 4 — Test locally

```python
from ovos_media_classifier.keyword import KeywordMediaClassifier

clf = KeywordMediaClassifier()
print(clf.classify("spela lite jazz", "sv-se"))
# should return (MediaType.MUSIC, 0.6) if MusicKeyword.voc contains "jazz"
# or (MediaType.GENERIC, 0.0) if not yet populated
```

### Step 5 — Add training data (for ML backends)

To improve ML backend performance for your language, add a dataset source in
`gather_dataset.py`:

```python
_langs = [... "sv"]    # add "sv" to the lang-support-tracker list
```

Or provide a custom CSV source:

```python
_sv_csv = [
    "https://your-host.com/ovos-intents-sv.csv",
]
```

### Step 6 — Contribute

1. Open a pull request adding the locale directory.
2. The files will be included in the next release via `package_data` in
   `setup.py`.

---

## Language tag conventions

OVOS uses lowercase BCP-47 tags with a hyphen: `en-us`, `de-de`, `pt-br`.

The locale directory names follow this convention exactly.  The `_VocMatcher`
normalises the tag to lowercase before constructing the path.

The Basque language uses the tag `eu` (no region subtag) — this is the only
exception in the current locale set.

---

## ML backend language notes

### sklearn

The sklearn model is language-agnostic at inference time.  It operates on raw
text characters and n-grams without any language-specific preprocessing.
Training data is multilingual (see the source list in [TRAINING.md](TRAINING.md)).
Adding more data in an under-represented language and retraining will improve
accuracy for that language.

### Model2Vec

Static embedding models like `minishlab/potion-base-8M` are trained on
multilingual data.  Accuracy on European languages is generally good.  For
languages with different scripts or morphology (Arabic, Japanese, Chinese),
a language-specific base model may be needed.

### AhocorasickNER

Entirely language-agnostic — it performs substring matching regardless of
script or morphology.  For languages with rich inflection (Finnish, Turkish)
you may need to register multiple forms of the same entity.
