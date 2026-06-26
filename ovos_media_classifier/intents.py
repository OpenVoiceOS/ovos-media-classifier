"""OCP domain and entity-label enumerations.

These enums capture the label classes used across all classifier backends.

The classifier models media along the **real axes** only:

  domain (``OCPDomain``)  ×  type (``mediavocab.MediaType``)  +  genres

There is *no* separate per-media-type ``play intent`` enum: a play request is a
single domain (``OCP_PLAY``), the *what* is a ``mediavocab.MediaType``, and the
distinctions that are not types (anime / cartoon / asmr / adult …) are carried
as ``mediavocab`` **genre tags**.  Raw detection labels (``.voc`` / model-head
strings such as ``"music"``, ``"adult"``, ``"asmr"``) resolve straight to
``(MediaType, genres)`` via the ``LABEL_TO_*`` maps below — no intent layer.

OCPDomain
  ocp_play    → media playback request
  ocp_control → player control request  (use OCPControlIntent to further classify)
  not_ocp     → unrelated query

MediaType
  Canonical OCP media type taxonomy.  Authoritative definition lives here
  (ovos-media-classifier), not in ovos-utils.  ovos-utils keeps a
  backward-compatible copy for non-OCP OVOS components; integer values for
  shared types are identical, so they interoperate via int comparison.

OCPControlIntent
  One value per control action supported by the OCP pipeline.
  The string values match the padatious intent names (without the ".intent" suffix).
"""
from enum import Enum
from typing import Dict, List

# The canonical media taxonomy is owned by ``mediavocab`` (a str-Enum), not by
# this package.  We re-export it under the historical name ``MediaType`` so the
# public API stays ``(MediaType, confidence)`` while *enforcing* the shared
# vocabulary.  Raw detection labels map straight onto ``mediavocab.MediaType`` +
# genre tags at the boundary (see ``LABEL_TO_MEDIA_TYPE`` / ``LABEL_TO_GENRES``).
from mediavocab import MediaType
from mediavocab.taxonomy.genre import KNOWN_GENRES


class OCPDomain(str, Enum):
    """Top-level domain — tells whether the utterance targets OCP at all."""
    OCP_PLAY = "ocp_play"
    OCP_CONTROL = "ocp_control"
    NOT_OCP = "not_ocp"


class OCPControlIntent(str, Enum):
    """Fine-grained action labels for the ocp_control domain."""
    PLAY           = "play"
    NEXT           = "next"
    PREVIOUS       = "prev"
    PAUSE          = "pause"
    RESUME         = "resume"
    STOP           = "stop"
    OPEN           = "open"
    LIKE_SONG      = "like_song"
    PLAY_FAVORITES = "play_favorites"
    SAVE_GAME      = "save_game"
    LOAD_GAME      = "load_game"
    SHUFFLE        = "shuffle"        # "shuffle my playlist", "random order"
    REPEAT         = "repeat"         # "repeat this", "loop", "play again"
    SEEK_FORWARD   = "seek_forward"   # "skip 30 seconds", "fast forward"
    SEEK_BACKWARD  = "seek_backward"  # "go back a minute", "rewind"


class OCPEntityLabel(str, Enum):
    """Canonical AhocorasickNER entity label names for OCP classification.

    These are the labels that:
      1. The AhocorasickMediaClassifier is trained on (via NER feature extraction)
      2. OCP skills register at runtime via ``ovos.common_play.register_keyword``
         and ``ovos.common_play.announce`` bus messages
      3. The pipeline NER (in ocp_pipeline) uses to tag utterances for entity extraction

    The mapping from entity labels to ``mediavocab.MediaType`` (and any genre
    tags) is defined in ``NER_LABEL_TO_MEDIA_TYPE`` / ``NER_LABEL_TO_GENRES``
    below.

    Runtime population model
    ------------------------
    Skills populate the NER at runtime by registering their content:
      - A music skill registers known artist names under ``artist_name``
      - A movie skill registers known titles under ``movie_title``
      - A streaming service registers its name under the appropriate
        ``*_streaming_service`` label
    The AhocorasickMediaClassifier can then classify utterances based on
    which entity labels appear (i.e., what media the user actually has access to).

    Training vs runtime
    -------------------
    During training the NER is pre-seeded from HuggingFace datasets (see
    ``training.ner_datasets``).  At runtime it is
    populated incrementally as skills register their content.
    """

    # ---- Streaming service labels (registered by OCP skills on startup) ----
    MUSIC_STREAMING_SERVICE     = "music_streaming_service"
    MOVIE_STREAMING_SERVICE     = "movie_streaming_service"
    SHORTS_STREAMING_SERVICE    = "shorts_streaming_service"
    PODCAST_STREAMING_SERVICE   = "podcast_streaming_service"
    AUDIOBOOK_STREAMING_SERVICE = "audiobook_streaming_service"
    NEWS_PROVIDER               = "news_provider"
    TV_STREAMING_SERVICE        = "tv_streaming_service"
    RADIO_STREAMING_SERVICE     = "radio_streaming_service"
    ADULT_STREAMING_SERVICE     = "adult_streaming_service"

    # ---- Music entity labels (artist/track/album names registered by music skills) ----
    ARTIST_NAME   = "artist_name"
    TRACK_NAME    = "track_name"
    ALBUM_NAME    = "album_name"
    ALBUM_TYPE    = "album_type"
    MUSIC_GENRE   = "music_genre"
    RECORD_LABEL  = "record_label"

    # ---- Video entity labels (film/show names registered by video skills) ----
    MOVIE_TITLE        = "movie_title"
    MOVIE_ACTOR        = "movie_actor"
    MOVIE_DIRECTOR     = "movie_director"
    MOVIE_PRODUCER     = "movie_producer"
    MOVIE_WRITER       = "movie_writer"
    MOVIE_COMPOSER     = "movie_composer"
    MOVIE_STUDIO       = "movie_studio"
    VIDEO_GENRE        = "video_genre"
    TV_SHOW_TITLE      = "tv_show_title"
    ANIME_TITLE        = "anime_title"
    CARTOON_TITLE      = "cartoon_title"
    DOCUMENTARY_TITLE  = "documentary_title"
    TRAILER_TITLE      = "trailer_title"
    BTS_TITLE          = "bts_title"
    MUSIC_VIDEO_TITLE  = "music_video_title"
    VISUAL_STORY_TITLE = "visual_story_title"
    SILENT_MOVIE_TITLE = "silent_movie_title"
    BW_MOVIE_TITLE     = "bw_movie_title"
    HENTAI_TITLE       = "hentai_title"
    RADIO_DRAMA_TITLE  = "radio_drama_title"
    ADULT_TITLE        = "adult_title"
    PORNSTAR           = "pornstar"
    PORN_GENRE          = "porn_genre"

    # ---- TV / live stream entity labels ----
    TV_CHANNEL      = "tv_channel"
    YOUTUBE_CHANNEL = "youtube_channel"
    TV_GENRE        = "tv_genre"
    TV_NETWORK      = "tv_network"

    # ---- Other media entity labels ----
    PODCAST_TITLE    = "podcast_title"
    PODCAST_HOST     = "podcast_host"
    PODCAST_EPISODE  = "podcast_episode"
    PODCAST_GENRE    = "podcast_genre"
    AUDIOBOOK_TITLE  = "audiobook_title"
    AUDIOBOOK_AUTHOR = "audiobook_author"
    AUDIOBOOK_NARRATOR = "audiobook_narrator"
    BOOK_TITLE       = "book_title"
    BOOK_AUTHOR      = "book_author"
    BOOK_GENRE       = "book_genre"
    COMIC_TITLE      = "comic_title"
    COMIC_GENRE      = "comic_genre"
    SOUND_NAME       = "sound_name"
    AMBIENT_SOUND    = "ambient_sound"
    PLAYLIST_MOOD    = "playlist_mood"
    PLAYLIST_ACTIVITY = "playlist_activity"
    NEWS_CATEGORY    = "news_category"
    GAME_TITLE       = "game_title"
    GAME_GENRE       = "game_genre"
    GAME_PLATFORM    = "game_platform"
    ASMR_ARTIST      = "asmr_artist"
    ANIME_STUDIO     = "anime_studio"

    # ---- Radio entity labels ----
    RADIO_STATION    = "radio_station"
    RADIO_GENRE      = "radio_genre"

    # ---- Media-type keyword labels (generic vocabulary) ----
    # Raw media-type cue labels used when no specific entity is found but a
    # keyword strongly signals the media type.
    MUSIC_KEYWORD              = "music"
    PODCAST_KEYWORD            = "podcast"
    RADIO_KEYWORD              = "radio"
    AUDIOBOOK_KEYWORD          = "audiobook"
    BOOK_KEYWORD               = "book"
    PLAYLIST_KEYWORD           = "playlist"
    SOUND_EFFECT_KEYWORD       = "sound_effect"
    INTERACTIVE_FICTION_KEYWORD = "interactive_fiction"
    AMBIENT_KEYWORD            = "ambient"
    COMIC_KEYWORD              = "comic"
    NEWS_KEYWORD               = "news"
    MOVIE_KEYWORD              = "movie"
    TV_KEYWORD                 = "tv"
    TV_SHOW_KEYWORD            = "tv_show"
    VIDEO_KEYWORD              = "video"
    VIDEO_EPISODES_KEYWORD     = "video_episodes"
    AUDIO_KEYWORD              = "audio"
    GAME_KEYWORD               = "game"
    ANIME_KEYWORD              = "anime"
    CARTOON_KEYWORD            = "cartoon"
    DOCUMENTARY_KEYWORD        = "documentary"
    SHORT_FILM_KEYWORD         = "short_film"
    SILENT_MOVIE_KEYWORD       = "silent_movie"
    BW_MOVIE_KEYWORD           = "bw_movie"
    RADIO_THEATRE_KEYWORD      = "radio_theatre"
    VISUAL_STORY_KEYWORD       = "visual_story"
    ASMR_KEYWORD               = "asmr"
    AUDIO_DESCRIPTION_KEYWORD  = "audio_description"
    MUSIC_VIDEO_KEYWORD        = "music_video"
    TRAILER_KEYWORD            = "trailer"
    BEHIND_THE_SCENES_KEYWORD  = "behind_the_scenes"
    ADULT_KEYWORD              = "adult"
    ADULT_AUDIO_KEYWORD        = "adult_audio"
    HENTAI_KEYWORD             = "hentai"


# ---------------------------------------------------------------------------
# Raw detection label → (MediaType, genres)
#
# These are the raw label strings emitted by the ``.voc`` keyword backend, the
# trained model heads, and the training datasets (``"music"``, ``"movie"``,
# ``"adult"``, ``"asmr"``, ``"anime"`` …).  They resolve **directly** to a
# canonical ``mediavocab.MediaType`` plus any ``mediavocab`` genre tags — there
# is no per-media-type intent layer in between.
#
# Several labels collapse onto one ``MediaType`` (the taxonomy deliberately
# models distinctions like anime / cartoon / silent / documentary as *genre* or
# *content-form*, not as media types).  The nuance that the type map drops is
# carried by ``LABEL_TO_GENRES`` below so it survives for content filtering /
# ranking — most importantly the ``adult`` tag the content filter blocks on.
# ---------------------------------------------------------------------------

LABEL_TO_MEDIA_TYPE: Dict[str, MediaType] = {
    "music":              MediaType.MUSIC,
    "podcast":            MediaType.PODCAST,
    "radio":              MediaType.RADIO,
    "audiobook":          MediaType.AUDIOBOOK,
    "book":               MediaType.BOOK,
    "playlist":           MediaType.PLAYLIST,
    "sound_effect":       MediaType.SOUND_EFFECT,
    "interactive_fiction": MediaType.INTERACTIVE_FICTION,
    "ambient":            MediaType.PROCEDURAL_AMBIENT,
    "comic":              MediaType.COMIC,
    "news":               MediaType.RADIO,
    "movie":              MediaType.MOVIE,
    "tv":                 MediaType.TV,
    "tv_show":            MediaType.EPISODIC_SERIES,
    "video":              MediaType.MOVIE,
    "video_episodes":     MediaType.EPISODIC_SERIES,
    "audio":              MediaType.MUSIC,
    "game":               MediaType.GAME,
    "anime":              MediaType.EPISODIC_SERIES,
    "cartoon":            MediaType.EPISODIC_SERIES,
    "documentary":        MediaType.MOVIE,
    "short_film":         MediaType.SHORT_FILM,
    "silent_movie":       MediaType.MOVIE,
    "bw_movie":           MediaType.MOVIE,
    "radio_theatre":      MediaType.AUDIO_DRAMA,
    "visual_story":       MediaType.COMIC,
    "asmr":               MediaType.PROCEDURAL_AMBIENT,
    "audio_description":  MediaType.MOVIE,
    "music_video":        MediaType.MUSIC_VIDEO,
    "trailer":            MediaType.MOVIE,
    "behind_the_scenes":  MediaType.MOVIE,
    "adult":              MediaType.MOVIE,
    "adult_audio":        MediaType.MUSIC,
    "hentai":             MediaType.EPISODIC_SERIES,
    "generic":            MediaType.GENERIC,
}

# Raw label → genre tags.  Only labels that carry a genre signal appear here;
# everything else implies no genre.  Filtered to ``mediavocab`` KNOWN_GENRES so
# we never emit a tag the taxonomy does not recognise.
_RAW_LABEL_GENRES: Dict[str, List[str]] = {
    "anime":        ["anime"],
    "cartoon":      ["animation"],
    "asmr":         ["asmr"],
    "adult":        ["adult"],
    "adult_audio":  ["adult"],
    "hentai":       ["anime", "adult"],
}
LABEL_TO_GENRES: Dict[str, List[str]] = {
    label: [g for g in genres if g in KNOWN_GENRES]
    for label, genres in _RAW_LABEL_GENRES.items()
}


def genres_for_label(label: str) -> List[str]:
    """Return the mediavocab genre tags implied by a raw detection label."""
    return list(LABEL_TO_GENRES.get(label, []))


# ---------------------------------------------------------------------------
# NER entity label → (MediaType, genres)
#
# Covers every ``OCPEntityLabel`` value so AhocorasickMediaClassifier can map
# any entity hit straight to a media type (and genres where the entity carries
# one — e.g. an ``anime_title`` is an EPISODIC_SERIES tagged ``anime``, a
# ``pornstar`` is a MOVIE tagged ``adult``).  Keyed by the string value of the
# label (e.g. "artist_name") so callers can use raw NER output directly.
# ---------------------------------------------------------------------------

NER_LABEL_TO_MEDIA_TYPE: Dict[str, MediaType] = {
    # ---- Streaming service labels ----
    OCPEntityLabel.MUSIC_STREAMING_SERVICE.value:     MediaType.MUSIC,
    OCPEntityLabel.MOVIE_STREAMING_SERVICE.value:     MediaType.MOVIE,
    OCPEntityLabel.SHORTS_STREAMING_SERVICE.value:    MediaType.SHORT_FILM,
    OCPEntityLabel.PODCAST_STREAMING_SERVICE.value:   MediaType.PODCAST,
    OCPEntityLabel.AUDIOBOOK_STREAMING_SERVICE.value: MediaType.AUDIOBOOK,
    OCPEntityLabel.NEWS_PROVIDER.value:               MediaType.RADIO,
    OCPEntityLabel.TV_STREAMING_SERVICE.value:        MediaType.TV,
    OCPEntityLabel.RADIO_STREAMING_SERVICE.value:     MediaType.RADIO,
    OCPEntityLabel.ADULT_STREAMING_SERVICE.value:     MediaType.MOVIE,

    # ---- Music entity labels ----
    OCPEntityLabel.ARTIST_NAME.value:   MediaType.MUSIC,
    OCPEntityLabel.TRACK_NAME.value:    MediaType.MUSIC,
    OCPEntityLabel.ALBUM_NAME.value:    MediaType.MUSIC,
    OCPEntityLabel.ALBUM_TYPE.value:    MediaType.MUSIC,
    OCPEntityLabel.MUSIC_GENRE.value:   MediaType.MUSIC,
    OCPEntityLabel.RECORD_LABEL.value:  MediaType.MUSIC,

    # ---- Video entity labels ----
    OCPEntityLabel.MOVIE_TITLE.value:        MediaType.MOVIE,
    OCPEntityLabel.MOVIE_ACTOR.value:        MediaType.MOVIE,
    OCPEntityLabel.MOVIE_DIRECTOR.value:     MediaType.MOVIE,
    OCPEntityLabel.MOVIE_PRODUCER.value:     MediaType.MOVIE,
    OCPEntityLabel.MOVIE_WRITER.value:       MediaType.MOVIE,
    OCPEntityLabel.MOVIE_COMPOSER.value:     MediaType.MOVIE,
    OCPEntityLabel.MOVIE_STUDIO.value:       MediaType.MOVIE,
    OCPEntityLabel.VIDEO_GENRE.value:        MediaType.MOVIE,
    OCPEntityLabel.TV_SHOW_TITLE.value:      MediaType.EPISODIC_SERIES,
    OCPEntityLabel.ANIME_TITLE.value:        MediaType.EPISODIC_SERIES,
    OCPEntityLabel.CARTOON_TITLE.value:      MediaType.EPISODIC_SERIES,
    OCPEntityLabel.DOCUMENTARY_TITLE.value:  MediaType.MOVIE,
    OCPEntityLabel.TRAILER_TITLE.value:      MediaType.MOVIE,
    OCPEntityLabel.BTS_TITLE.value:          MediaType.MOVIE,
    OCPEntityLabel.MUSIC_VIDEO_TITLE.value:  MediaType.MUSIC_VIDEO,
    OCPEntityLabel.VISUAL_STORY_TITLE.value: MediaType.COMIC,
    OCPEntityLabel.SILENT_MOVIE_TITLE.value: MediaType.MOVIE,
    OCPEntityLabel.BW_MOVIE_TITLE.value:     MediaType.MOVIE,
    OCPEntityLabel.HENTAI_TITLE.value:       MediaType.EPISODIC_SERIES,
    OCPEntityLabel.RADIO_DRAMA_TITLE.value:  MediaType.AUDIO_DRAMA,
    OCPEntityLabel.ADULT_TITLE.value:        MediaType.MOVIE,
    OCPEntityLabel.PORNSTAR.value:           MediaType.MOVIE,
    OCPEntityLabel.PORN_GENRE.value:         MediaType.MOVIE,

    # ---- TV / live stream entity labels ----
    OCPEntityLabel.TV_CHANNEL.value:      MediaType.TV,
    OCPEntityLabel.YOUTUBE_CHANNEL.value: MediaType.EPISODIC_SERIES,
    OCPEntityLabel.TV_GENRE.value:        MediaType.EPISODIC_SERIES,
    OCPEntityLabel.TV_NETWORK.value:      MediaType.EPISODIC_SERIES,

    # ---- Other media entity labels ----
    OCPEntityLabel.PODCAST_TITLE.value:      MediaType.PODCAST,
    OCPEntityLabel.PODCAST_HOST.value:       MediaType.PODCAST,
    OCPEntityLabel.PODCAST_EPISODE.value:    MediaType.PODCAST,
    OCPEntityLabel.PODCAST_GENRE.value:      MediaType.PODCAST,
    OCPEntityLabel.AUDIOBOOK_TITLE.value:    MediaType.AUDIOBOOK,
    OCPEntityLabel.AUDIOBOOK_AUTHOR.value:   MediaType.AUDIOBOOK,
    OCPEntityLabel.AUDIOBOOK_NARRATOR.value: MediaType.AUDIOBOOK,
    OCPEntityLabel.BOOK_TITLE.value:         MediaType.BOOK,
    OCPEntityLabel.BOOK_AUTHOR.value:        MediaType.BOOK,
    OCPEntityLabel.BOOK_GENRE.value:         MediaType.BOOK,
    OCPEntityLabel.COMIC_TITLE.value:        MediaType.COMIC,
    OCPEntityLabel.COMIC_GENRE.value:        MediaType.COMIC,
    OCPEntityLabel.SOUND_NAME.value:         MediaType.SOUND_EFFECT,
    OCPEntityLabel.AMBIENT_SOUND.value:      MediaType.PROCEDURAL_AMBIENT,
    OCPEntityLabel.PLAYLIST_MOOD.value:      MediaType.PLAYLIST,
    OCPEntityLabel.PLAYLIST_ACTIVITY.value:  MediaType.PLAYLIST,
    OCPEntityLabel.NEWS_CATEGORY.value:      MediaType.RADIO,
    OCPEntityLabel.GAME_TITLE.value:         MediaType.GAME,
    OCPEntityLabel.GAME_GENRE.value:         MediaType.GAME,
    OCPEntityLabel.GAME_PLATFORM.value:      MediaType.GAME,
    OCPEntityLabel.ASMR_ARTIST.value:        MediaType.PROCEDURAL_AMBIENT,
    OCPEntityLabel.ANIME_STUDIO.value:       MediaType.EPISODIC_SERIES,

    # ---- Radio entity labels ----
    OCPEntityLabel.RADIO_STATION.value:      MediaType.RADIO,
    OCPEntityLabel.RADIO_GENRE.value:        MediaType.RADIO,

    # ---- Media-type keyword labels (generic vocabulary) ----
    OCPEntityLabel.MUSIC_KEYWORD.value:              MediaType.MUSIC,
    OCPEntityLabel.PODCAST_KEYWORD.value:            MediaType.PODCAST,
    OCPEntityLabel.RADIO_KEYWORD.value:              MediaType.RADIO,
    OCPEntityLabel.AUDIOBOOK_KEYWORD.value:          MediaType.AUDIOBOOK,
    OCPEntityLabel.BOOK_KEYWORD.value:               MediaType.BOOK,
    OCPEntityLabel.PLAYLIST_KEYWORD.value:           MediaType.PLAYLIST,
    OCPEntityLabel.SOUND_EFFECT_KEYWORD.value:       MediaType.SOUND_EFFECT,
    OCPEntityLabel.INTERACTIVE_FICTION_KEYWORD.value: MediaType.INTERACTIVE_FICTION,
    OCPEntityLabel.AMBIENT_KEYWORD.value:            MediaType.PROCEDURAL_AMBIENT,
    OCPEntityLabel.COMIC_KEYWORD.value:              MediaType.COMIC,
    OCPEntityLabel.NEWS_KEYWORD.value:               MediaType.RADIO,
    OCPEntityLabel.MOVIE_KEYWORD.value:              MediaType.MOVIE,
    OCPEntityLabel.TV_KEYWORD.value:                 MediaType.TV,
    OCPEntityLabel.TV_SHOW_KEYWORD.value:            MediaType.EPISODIC_SERIES,
    OCPEntityLabel.VIDEO_KEYWORD.value:              MediaType.MOVIE,
    OCPEntityLabel.VIDEO_EPISODES_KEYWORD.value:     MediaType.EPISODIC_SERIES,
    OCPEntityLabel.AUDIO_KEYWORD.value:              MediaType.MUSIC,
    OCPEntityLabel.GAME_KEYWORD.value:               MediaType.GAME,
    OCPEntityLabel.ANIME_KEYWORD.value:              MediaType.EPISODIC_SERIES,
    OCPEntityLabel.CARTOON_KEYWORD.value:            MediaType.EPISODIC_SERIES,
    OCPEntityLabel.DOCUMENTARY_KEYWORD.value:        MediaType.MOVIE,
    OCPEntityLabel.SHORT_FILM_KEYWORD.value:         MediaType.SHORT_FILM,
    OCPEntityLabel.SILENT_MOVIE_KEYWORD.value:       MediaType.MOVIE,
    OCPEntityLabel.BW_MOVIE_KEYWORD.value:           MediaType.MOVIE,
    OCPEntityLabel.RADIO_THEATRE_KEYWORD.value:      MediaType.AUDIO_DRAMA,
    OCPEntityLabel.VISUAL_STORY_KEYWORD.value:       MediaType.COMIC,
    OCPEntityLabel.ASMR_KEYWORD.value:               MediaType.PROCEDURAL_AMBIENT,
    OCPEntityLabel.AUDIO_DESCRIPTION_KEYWORD.value:  MediaType.MOVIE,
    OCPEntityLabel.MUSIC_VIDEO_KEYWORD.value:        MediaType.MUSIC_VIDEO,
    OCPEntityLabel.TRAILER_KEYWORD.value:            MediaType.MOVIE,
    OCPEntityLabel.BEHIND_THE_SCENES_KEYWORD.value:  MediaType.MOVIE,
    OCPEntityLabel.ADULT_KEYWORD.value:              MediaType.MOVIE,
    OCPEntityLabel.ADULT_AUDIO_KEYWORD.value:        MediaType.MUSIC,
    OCPEntityLabel.HENTAI_KEYWORD.value:             MediaType.EPISODIC_SERIES,
}

# NER entity label → genre tags.  Only entity labels that carry a genre signal
# appear; this is what lets the content filter block adult entities (pornstar,
# adult_title, …) and rank anime/asmr even though the entity's MediaType is a
# generic MOVIE / EPISODIC_SERIES.  Filtered to ``mediavocab`` KNOWN_GENRES.
_RAW_NER_LABEL_GENRES: Dict[str, List[str]] = {
    OCPEntityLabel.ANIME_TITLE.value:    ["anime"],
    OCPEntityLabel.ANIME_STUDIO.value:   ["anime"],
    OCPEntityLabel.ANIME_KEYWORD.value:  ["anime"],
    OCPEntityLabel.CARTOON_TITLE.value:   ["animation"],
    OCPEntityLabel.CARTOON_KEYWORD.value: ["animation"],
    OCPEntityLabel.ASMR_ARTIST.value:    ["asmr"],
    OCPEntityLabel.ASMR_KEYWORD.value:   ["asmr"],
    OCPEntityLabel.HENTAI_TITLE.value:   ["anime", "adult"],
    OCPEntityLabel.HENTAI_KEYWORD.value: ["anime", "adult"],
    OCPEntityLabel.ADULT_STREAMING_SERVICE.value: ["adult"],
    OCPEntityLabel.ADULT_TITLE.value:    ["adult"],
    OCPEntityLabel.PORNSTAR.value:       ["adult"],
    OCPEntityLabel.PORN_GENRE.value:     ["adult"],
    OCPEntityLabel.ADULT_KEYWORD.value:       ["adult"],
    OCPEntityLabel.ADULT_AUDIO_KEYWORD.value: ["adult"],
}
NER_LABEL_TO_GENRES: Dict[str, List[str]] = {
    label: [g for g in genres if g in KNOWN_GENRES]
    for label, genres in _RAW_NER_LABEL_GENRES.items()
}
