"""OCP domain and intent enumerations.

These enums capture every label class used across all classifier backends.

Hierarchy
---------
OCPDomain
  ocp_play    → media playback request  (use OCPPlayIntent to further classify)
  ocp_control → player control request  (use OCPControlIntent to further classify)
  not_ocp     → unrelated query

MediaType
  Canonical OCP media type taxonomy.  Authoritative definition lives here
  (ovos-media-classifier), not in ovos-utils.  ovos-utils keeps a
  backward-compatible copy for non-OCP OVOS components; integer values for
  shared types are identical, so they interoperate via int comparison.

OCPPlayIntent
  One value per MediaType that OCP can handle.  The string values match the
  training labels used in ovos-m2v-pipeline and in the padatious samples.

OCPControlIntent
  One value per control action supported by the OCP pipeline.
  The string values match the padatious intent names (without the ".intent" suffix).
"""
from enum import Enum
from typing import Dict, List

# The canonical media taxonomy is owned by ``mediavocab`` (a str-Enum), not by
# this package.  We re-export it under the historical name ``MediaType`` so the
# public API stays ``(MediaType, confidence)`` while *enforcing* the shared
# vocabulary.  The classifier keeps a richer internal *intent* label space
# (``OCPPlayIntent`` / ``OCPEntityLabel``) used to train models; the public
# output is mapped onto ``mediavocab.MediaType`` + genre tags at the boundary
# (see ``PLAY_INTENT_TO_MEDIA_TYPE`` / ``PLAY_INTENT_TO_GENRES``).
from mediavocab import MediaType
from mediavocab.taxonomy.genre import KNOWN_GENRES


class OCPDomain(str, Enum):
    """Top-level domain — tells whether the utterance targets OCP at all."""
    OCP_PLAY = "ocp_play"
    OCP_CONTROL = "ocp_control"
    NOT_OCP = "not_ocp"


class OCPPlayIntent(str, Enum):
    """Fine-grained media-type intent labels for the ocp_play domain."""
    MUSIC            = "music"
    PODCAST          = "podcast"
    RADIO            = "radio"
    AUDIOBOOK        = "audiobook"
    NEWS             = "news"
    MOVIE            = "movie"
    TV               = "tv"           # live IPTV / cable TV stream
    TV_SHOW          = "tv_show"      # episodic TV series (Breaking Bad, etc.)
    VIDEO            = "video"
    VIDEO_EPISODES   = "video_episodes"
    AUDIO            = "audio"
    GAME             = "game"
    ANIME            = "anime"
    CARTOON          = "cartoon"
    DOCUMENTARY      = "documentary"
    SHORT_FILM       = "short_film"
    SILENT_MOVIE     = "silent_movie"
    BW_MOVIE         = "bw_movie"
    RADIO_THEATRE    = "radio_theatre"
    VISUAL_STORY     = "visual_story"
    ASMR             = "asmr"
    AUDIO_DESCRIPTION = "audio_description"
    MUSIC_VIDEO      = "music_video"      # official music video
    TRAILER          = "trailer"
    BEHIND_THE_SCENES = "behind_the_scenes"
    ADULT            = "adult"
    ADULT_AUDIO      = "adult_audio"
    HENTAI           = "hentai"
    GENERIC          = "generic"


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

    The mapping from entity labels to OCPPlayIntent (and thus MediaType) is
    defined in ``NER_LABEL_TO_PLAY_INTENT`` below.

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
    # These mirror the OCPPlayIntent values and are used when no specific
    # entity is found but a keyword strongly signals the media type.
    MUSIC_KEYWORD              = "music"
    PODCAST_KEYWORD            = "podcast"
    RADIO_KEYWORD              = "radio"
    AUDIOBOOK_KEYWORD          = "audiobook"
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
# Canonical label → MediaType mapping
# Shared by all backends — keeps training labels and runtime values in sync.
# ---------------------------------------------------------------------------

# Fine-grained play intent → canonical ``mediavocab.MediaType``.  Several
# intents collapse onto one mediavocab type (the taxonomy deliberately models
# distinctions like anime / cartoon / silent / documentary as *genre* or
# *content-form*, not as media types) — the lost nuance is carried by
# ``PLAY_INTENT_TO_GENRES`` below so it survives for content filtering / ranking.
PLAY_INTENT_TO_MEDIA_TYPE: Dict[OCPPlayIntent, MediaType] = {
    OCPPlayIntent.MUSIC:              MediaType.MUSIC,
    OCPPlayIntent.PODCAST:            MediaType.PODCAST,
    OCPPlayIntent.RADIO:              MediaType.RADIO,
    OCPPlayIntent.AUDIOBOOK:          MediaType.AUDIOBOOK,
    OCPPlayIntent.NEWS:               MediaType.RADIO,
    OCPPlayIntent.MOVIE:              MediaType.MOVIE,
    OCPPlayIntent.TV:                 MediaType.TV,
    OCPPlayIntent.TV_SHOW:            MediaType.EPISODIC_SERIES,
    OCPPlayIntent.VIDEO:              MediaType.MOVIE,
    OCPPlayIntent.VIDEO_EPISODES:     MediaType.EPISODIC_SERIES,
    OCPPlayIntent.AUDIO:              MediaType.MUSIC,
    OCPPlayIntent.GAME:               MediaType.GAME,
    OCPPlayIntent.ANIME:             MediaType.EPISODIC_SERIES,
    OCPPlayIntent.CARTOON:            MediaType.EPISODIC_SERIES,
    OCPPlayIntent.DOCUMENTARY:        MediaType.MOVIE,
    OCPPlayIntent.SHORT_FILM:         MediaType.SHORT_FILM,
    OCPPlayIntent.SILENT_MOVIE:       MediaType.MOVIE,
    OCPPlayIntent.BW_MOVIE:           MediaType.MOVIE,
    OCPPlayIntent.RADIO_THEATRE:      MediaType.AUDIO_DRAMA,
    OCPPlayIntent.VISUAL_STORY:       MediaType.COMIC,
    OCPPlayIntent.ASMR:               MediaType.PROCEDURAL_AMBIENT,
    OCPPlayIntent.AUDIO_DESCRIPTION:  MediaType.MOVIE,
    OCPPlayIntent.MUSIC_VIDEO:        MediaType.MUSIC_VIDEO,
    OCPPlayIntent.TRAILER:            MediaType.MOVIE,
    OCPPlayIntent.BEHIND_THE_SCENES:  MediaType.MOVIE,
    OCPPlayIntent.ADULT:              MediaType.MOVIE,
    OCPPlayIntent.ADULT_AUDIO:        MediaType.MUSIC,
    OCPPlayIntent.HENTAI:             MediaType.EPISODIC_SERIES,
    OCPPlayIntent.GENERIC:            MediaType.GENERIC,
}

# Fine-grained play intent → genre tags (all members of ``mediavocab`` KNOWN_GENRES).
# These preserve the distinctions that collapse in the type map and, crucially,
# carry the ``adult`` signal the content filter blocks on by default.
_RAW_PLAY_INTENT_GENRES: Dict[OCPPlayIntent, List[str]] = {
    OCPPlayIntent.ANIME:        ["anime"],
    OCPPlayIntent.CARTOON:      ["animation"],
    OCPPlayIntent.ASMR:         ["asmr"],
    OCPPlayIntent.ADULT:        ["adult"],
    OCPPlayIntent.ADULT_AUDIO:  ["adult"],
    OCPPlayIntent.HENTAI:       ["anime", "adult"],
}
# enforce taxonomy: only emit genres mediavocab actually knows
PLAY_INTENT_TO_GENRES: Dict[OCPPlayIntent, List[str]] = {
    intent: [g for g in genres if g in KNOWN_GENRES]
    for intent, genres in _RAW_PLAY_INTENT_GENRES.items()
}

# Canonical reverse (one representative intent per mediavocab type), used by
# training/exploration tooling.  Explicit to avoid arbitrary last-wins collapse.
MEDIA_TYPE_TO_PLAY_INTENT: Dict[MediaType, OCPPlayIntent] = {
    MediaType.MUSIC:              OCPPlayIntent.MUSIC,
    MediaType.PODCAST:            OCPPlayIntent.PODCAST,
    MediaType.RADIO:              OCPPlayIntent.RADIO,
    MediaType.AUDIOBOOK:          OCPPlayIntent.AUDIOBOOK,
    MediaType.MOVIE:              OCPPlayIntent.MOVIE,
    MediaType.TV:                 OCPPlayIntent.TV,
    MediaType.EPISODIC_SERIES:    OCPPlayIntent.TV_SHOW,
    MediaType.GAME:               OCPPlayIntent.GAME,
    MediaType.SHORT_FILM:         OCPPlayIntent.SHORT_FILM,
    MediaType.AUDIO_DRAMA:        OCPPlayIntent.RADIO_THEATRE,
    MediaType.COMIC:              OCPPlayIntent.VISUAL_STORY,
    MediaType.PROCEDURAL_AMBIENT: OCPPlayIntent.ASMR,
    MediaType.MUSIC_VIDEO:        OCPPlayIntent.MUSIC_VIDEO,
    MediaType.GENERIC:            OCPPlayIntent.GENERIC,
}

# String form of intent labels → MediaType (used by backends that emit raw strings)
LABEL_TO_MEDIA_TYPE: Dict[str, MediaType] = {
    intent.value: mt for intent, mt in PLAY_INTENT_TO_MEDIA_TYPE.items()
}

# String form of intent labels → genre tags (raw model output → genres)
LABEL_TO_GENRES: Dict[str, List[str]] = {
    intent.value: genres for intent, genres in PLAY_INTENT_TO_GENRES.items()
}


def genres_for_label(label: str) -> List[str]:
    """Return the mediavocab genre tags implied by a raw play-intent label."""
    return list(LABEL_TO_GENRES.get(label, []))

# ---------------------------------------------------------------------------
# NER entity label → OCPPlayIntent mapping
# Covers every OCPEntityLabel value so AhocorasickMediaClassifier can map
# any entity hit to a media type.  Keyed by the string value of the label
# (e.g. "artist_name") so callers can use raw NER output directly.
# ---------------------------------------------------------------------------

NER_LABEL_TO_PLAY_INTENT: Dict[str, OCPPlayIntent] = {
    # ---- Streaming service labels ----
    OCPEntityLabel.MUSIC_STREAMING_SERVICE:     OCPPlayIntent.MUSIC,
    OCPEntityLabel.MOVIE_STREAMING_SERVICE:     OCPPlayIntent.MOVIE,
    OCPEntityLabel.SHORTS_STREAMING_SERVICE:    OCPPlayIntent.SHORT_FILM,
    OCPEntityLabel.PODCAST_STREAMING_SERVICE:   OCPPlayIntent.PODCAST,
    OCPEntityLabel.AUDIOBOOK_STREAMING_SERVICE: OCPPlayIntent.AUDIOBOOK,
    OCPEntityLabel.NEWS_PROVIDER:               OCPPlayIntent.NEWS,
    OCPEntityLabel.TV_STREAMING_SERVICE:        OCPPlayIntent.TV,
    OCPEntityLabel.RADIO_STREAMING_SERVICE:     OCPPlayIntent.RADIO,
    OCPEntityLabel.ADULT_STREAMING_SERVICE:     OCPPlayIntent.ADULT,

    # ---- Music entity labels ----
    OCPEntityLabel.ARTIST_NAME:   OCPPlayIntent.MUSIC,
    OCPEntityLabel.TRACK_NAME:    OCPPlayIntent.MUSIC,
    OCPEntityLabel.ALBUM_NAME:    OCPPlayIntent.MUSIC,
    OCPEntityLabel.ALBUM_TYPE:    OCPPlayIntent.MUSIC,
    OCPEntityLabel.MUSIC_GENRE:   OCPPlayIntent.MUSIC,
    OCPEntityLabel.RECORD_LABEL:  OCPPlayIntent.MUSIC,
    OCPEntityLabel.RADIO_STATION: OCPPlayIntent.RADIO,

    # ---- Video entity labels ----
    OCPEntityLabel.MOVIE_TITLE:       OCPPlayIntent.MOVIE,
    OCPEntityLabel.MOVIE_ACTOR:       OCPPlayIntent.MOVIE,
    OCPEntityLabel.MOVIE_DIRECTOR:    OCPPlayIntent.MOVIE,
    OCPEntityLabel.MOVIE_PRODUCER:    OCPPlayIntent.MOVIE,
    OCPEntityLabel.MOVIE_WRITER:      OCPPlayIntent.MOVIE,
    OCPEntityLabel.MOVIE_COMPOSER:    OCPPlayIntent.MOVIE,
    OCPEntityLabel.MOVIE_STUDIO:      OCPPlayIntent.MOVIE,
    OCPEntityLabel.VIDEO_GENRE:       OCPPlayIntent.VIDEO,
    OCPEntityLabel.TV_SHOW_TITLE:     OCPPlayIntent.TV_SHOW,
    OCPEntityLabel.ANIME_TITLE:       OCPPlayIntent.ANIME,
    OCPEntityLabel.CARTOON_TITLE:     OCPPlayIntent.CARTOON,
    OCPEntityLabel.DOCUMENTARY_TITLE: OCPPlayIntent.DOCUMENTARY,
    OCPEntityLabel.TRAILER_TITLE:     OCPPlayIntent.TRAILER,
    OCPEntityLabel.BTS_TITLE:         OCPPlayIntent.BEHIND_THE_SCENES,
    OCPEntityLabel.MUSIC_VIDEO_TITLE: OCPPlayIntent.MUSIC_VIDEO,
    OCPEntityLabel.VISUAL_STORY_TITLE: OCPPlayIntent.VISUAL_STORY,
    OCPEntityLabel.SILENT_MOVIE_TITLE: OCPPlayIntent.SILENT_MOVIE,
    OCPEntityLabel.BW_MOVIE_TITLE:     OCPPlayIntent.BW_MOVIE,
    OCPEntityLabel.HENTAI_TITLE:       OCPPlayIntent.HENTAI,
    OCPEntityLabel.RADIO_DRAMA_TITLE:  OCPPlayIntent.RADIO_THEATRE,
    OCPEntityLabel.ADULT_TITLE:        OCPPlayIntent.ADULT,
    OCPEntityLabel.PORNSTAR:           OCPPlayIntent.ADULT,
    OCPEntityLabel.PORN_GENRE:          OCPPlayIntent.ADULT,

    # ---- TV / live stream entity labels ----
    OCPEntityLabel.TV_CHANNEL:      OCPPlayIntent.TV,
    OCPEntityLabel.YOUTUBE_CHANNEL: OCPPlayIntent.VIDEO_EPISODES,
    OCPEntityLabel.TV_GENRE:        OCPPlayIntent.TV_SHOW,
    OCPEntityLabel.TV_NETWORK:      OCPPlayIntent.TV_SHOW,

    # ---- Other media entity labels ----
    OCPEntityLabel.PODCAST_TITLE:    OCPPlayIntent.PODCAST,
    OCPEntityLabel.PODCAST_HOST:     OCPPlayIntent.PODCAST,
    OCPEntityLabel.PODCAST_EPISODE:  OCPPlayIntent.PODCAST,
    OCPEntityLabel.PODCAST_GENRE:    OCPPlayIntent.PODCAST,
    OCPEntityLabel.AUDIOBOOK_TITLE:  OCPPlayIntent.AUDIOBOOK,
    OCPEntityLabel.AUDIOBOOK_AUTHOR: OCPPlayIntent.AUDIOBOOK,
    OCPEntityLabel.AUDIOBOOK_NARRATOR: OCPPlayIntent.AUDIOBOOK,
    OCPEntityLabel.NEWS_CATEGORY:    OCPPlayIntent.NEWS,
    OCPEntityLabel.GAME_TITLE:       OCPPlayIntent.GAME,
    OCPEntityLabel.GAME_GENRE:       OCPPlayIntent.GAME,
    OCPEntityLabel.GAME_PLATFORM:    OCPPlayIntent.GAME,
    OCPEntityLabel.ASMR_ARTIST:      OCPPlayIntent.ASMR,
    OCPEntityLabel.ANIME_STUDIO:     OCPPlayIntent.ANIME,

    # ---- Radio entity labels ----
    OCPEntityLabel.RADIO_STATION:    OCPPlayIntent.RADIO,
    OCPEntityLabel.RADIO_GENRE:      OCPPlayIntent.RADIO,

    # ---- Media-type keyword labels (generic vocabulary) ----
    OCPEntityLabel.MUSIC_KEYWORD:              OCPPlayIntent.MUSIC,
    OCPEntityLabel.PODCAST_KEYWORD:            OCPPlayIntent.PODCAST,
    OCPEntityLabel.RADIO_KEYWORD:              OCPPlayIntent.RADIO,
    OCPEntityLabel.AUDIOBOOK_KEYWORD:          OCPPlayIntent.AUDIOBOOK,
    OCPEntityLabel.NEWS_KEYWORD:               OCPPlayIntent.NEWS,
    OCPEntityLabel.MOVIE_KEYWORD:              OCPPlayIntent.MOVIE,
    OCPEntityLabel.TV_KEYWORD:                 OCPPlayIntent.TV,
    OCPEntityLabel.TV_SHOW_KEYWORD:            OCPPlayIntent.TV_SHOW,
    OCPEntityLabel.VIDEO_KEYWORD:              OCPPlayIntent.VIDEO,
    OCPEntityLabel.VIDEO_EPISODES_KEYWORD:     OCPPlayIntent.VIDEO_EPISODES,
    OCPEntityLabel.AUDIO_KEYWORD:              OCPPlayIntent.AUDIO,
    OCPEntityLabel.GAME_KEYWORD:               OCPPlayIntent.GAME,
    OCPEntityLabel.ANIME_KEYWORD:              OCPPlayIntent.ANIME,
    OCPEntityLabel.CARTOON_KEYWORD:            OCPPlayIntent.CARTOON,
    OCPEntityLabel.DOCUMENTARY_KEYWORD:        OCPPlayIntent.DOCUMENTARY,
    OCPEntityLabel.SHORT_FILM_KEYWORD:         OCPPlayIntent.SHORT_FILM,
    OCPEntityLabel.SILENT_MOVIE_KEYWORD:       OCPPlayIntent.SILENT_MOVIE,
    OCPEntityLabel.BW_MOVIE_KEYWORD:           OCPPlayIntent.BW_MOVIE,
    OCPEntityLabel.RADIO_THEATRE_KEYWORD:      OCPPlayIntent.RADIO_THEATRE,
    OCPEntityLabel.VISUAL_STORY_KEYWORD:       OCPPlayIntent.VISUAL_STORY,
    OCPEntityLabel.ASMR_KEYWORD:               OCPPlayIntent.ASMR,
    OCPEntityLabel.AUDIO_DESCRIPTION_KEYWORD:  OCPPlayIntent.AUDIO_DESCRIPTION,
    OCPEntityLabel.MUSIC_VIDEO_KEYWORD:        OCPPlayIntent.MUSIC_VIDEO,
    OCPEntityLabel.TRAILER_KEYWORD:            OCPPlayIntent.TRAILER,
    OCPEntityLabel.BEHIND_THE_SCENES_KEYWORD:  OCPPlayIntent.BEHIND_THE_SCENES,
    OCPEntityLabel.ADULT_KEYWORD:              OCPPlayIntent.ADULT,
    OCPEntityLabel.ADULT_AUDIO_KEYWORD:        OCPPlayIntent.ADULT_AUDIO,
    OCPEntityLabel.HENTAI_KEYWORD:             OCPPlayIntent.HENTAI,
}
