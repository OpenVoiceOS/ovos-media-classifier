# Content filtering

`ContentFilter` is a **detect-to-block** content-moderation / parental-control
layer. The classifier surfaces a `mediavocab` genre signal (e.g. `adult` for
adult/hentai/porn queries), and the filter decides whether the request is allowed.

This is a moderation capability, not a content provider: the package detects
sensitive media so OVOS can **refuse** it. By default `adult` is blocked.

## Quick example

```python
from ovos_media_classifier import load_media_classifier, ContentFilter

clf = load_media_classifier()
cf = ContentFilter()              # default policy: adult blocked

cf.check(clf, "play some porn", "en-us")   # -> (True,  'blocked genre: adult')
cf.check(clf, "play some music", "en-us")  # -> (False, '')
```

`check(classifier, query, lang)` is a convenience wrapper: it calls the
classifier's `classify()` and `classify_genres()` and applies the policy, returning
`(blocked: bool, reason: str)`. If you already have a media type + genres, call
`is_blocked(media_type, genres)` directly.

## Default policy

- `adult` is blocked by default. The top-level convenience flag
  `allow_adult_content` (default `false`) keeps it blocked; set it `true` to lift
  the adult block.
- Filtering is enabled by default; set `media_content_filter.enabled` to `false` to
  disable it entirely.

The genre signal comes from the taxonomy map `PLAY_INTENT_TO_GENRES` — adult,
hentai and adult-audio intents all carry the `adult` tag even though their public
`MediaType` is a generic `MOVIE` / `MUSIC` / `EPISODIC_SERIES`. The type alone never
carries the adult signal; the genre does. See [taxonomy.md](taxonomy.md).

## Configuration

Under `mycroft.conf` (the OCP config block):

```json
{
  "allow_adult_content": false,

  "media_content_filter": {
    "enabled": true,
    "blocked_genres": ["adult"],
    "blocked_media_types": []
  }
}
```

| Key | Default | Meaning |
|---|---|---|
| `allow_adult_content` | `false` | When `true`, removes `adult` from the blocklist |
| `media_content_filter.enabled` | `true` | Master on/off for the filter |
| `media_content_filter.blocked_genres` | `["adult"]` | `mediavocab` genre tags to block |
| `media_content_filter.blocked_media_types` | `[]` | `mediavocab.MediaType` values to block (string values or members) |

`blocked_genres` and `blocked_media_types` accept any `mediavocab` genre or
`MediaType` — an operator can, for instance, block the `game` media type or add
extra genres beyond `adult`. A request is blocked if any of its genres is in
`blocked_genres` **or** its media type is in `blocked_media_types`; the returned
reason names the matching genre or type.

```python
# block the whole "game" media type in addition to adult genres
cf = ContentFilter({"media_content_filter": {"blocked_media_types": ["game"]}})
```
