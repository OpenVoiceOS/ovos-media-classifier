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
from enum import Enum, IntEnum
from typing import Dict


class MediaType(IntEnum):
    """Canonical OCP media type taxonomy.

    Authoritative definition lives here (ovos-media-classifier), not in
    ovos-utils.  ovos-utils keeps a backward-compatible copy for non-OCP
    OVOS components; integer values for shared types are identical.
    """
    # ── Catch-all ──────────────────────────────────────────────────────
    GENERIC           = 0
    # ── Audio-only ─────────────────────────────────────────────────────
    AUDIO             = 1    # ambient / unclassified audio
    MUSIC             = 2
    AUDIOBOOK         = 4
    PODCAST           = 6
    RADIO             = 7    # live radio stream
    NEWS              = 8    # news broadcast / feed
    RADIO_THEATRE     = 16   # audio drama / radio play
    ASMR              = 23
    MUSIC_VIDEO       = 24   # official music video for a song
    AUDIO_DESCRIPTION = 12   # narrated film for the blind
    # ── Live streams ───────────────────────────────────────────────────
    TV                = 9    # live IPTV / cable TV stream
    # ── Video – episodic / online ──────────────────────────────────────
    VIDEO_EPISODES    = 19   # YouTube channels, online video series
    TV_SHOW           = 25   # episodic TV series (Breaking Bad, Naruto)
    # ── Video – film ───────────────────────────────────────────────────
    VIDEO             = 3    # generic / YouTube video
    MOVIE             = 10
    SHORT_FILM        = 17
    SILENT_MOVIE      = 18
    BLACK_WHITE_MOVIE = 20
    DOCUMENTARY       = 15
    TRAILER           = 11
    BEHIND_THE_SCENES = 14
    VISUAL_STORY      = 13   # animated comic / motion comic
    # ── Animation ──────────────────────────────────────────────────────
    ANIME             = 22
    CARTOON           = 21
    # ── Interactive ────────────────────────────────────────────────────
    GAME              = 5
    # ── Adult (content-filter group, high int values) ──────────────────
    ADULT             = 69
    HENTAI            = 70
    ADULT_AUDIO       = 71


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
    ``ovos_media_classifier.train.ner_datasets``).  At runtime it is
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

PLAY_INTENT_TO_MEDIA_TYPE: Dict[OCPPlayIntent, MediaType] = {
    OCPPlayIntent.MUSIC:              MediaType.MUSIC,
    OCPPlayIntent.PODCAST:            MediaType.PODCAST,
    OCPPlayIntent.RADIO:              MediaType.RADIO,
    OCPPlayIntent.AUDIOBOOK:          MediaType.AUDIOBOOK,
    OCPPlayIntent.NEWS:               MediaType.NEWS,
    OCPPlayIntent.MOVIE:              MediaType.MOVIE,
    OCPPlayIntent.TV:                 MediaType.TV,
    OCPPlayIntent.TV_SHOW:            MediaType.TV_SHOW,
    OCPPlayIntent.VIDEO:              MediaType.VIDEO,
    OCPPlayIntent.VIDEO_EPISODES:     MediaType.VIDEO_EPISODES,
    OCPPlayIntent.AUDIO:              MediaType.AUDIO,
    OCPPlayIntent.GAME:               MediaType.GAME,
    OCPPlayIntent.ANIME:              MediaType.ANIME,
    OCPPlayIntent.CARTOON:            MediaType.CARTOON,
    OCPPlayIntent.DOCUMENTARY:        MediaType.DOCUMENTARY,
    OCPPlayIntent.SHORT_FILM:         MediaType.SHORT_FILM,
    OCPPlayIntent.SILENT_MOVIE:       MediaType.SILENT_MOVIE,
    OCPPlayIntent.BW_MOVIE:           MediaType.BLACK_WHITE_MOVIE,
    OCPPlayIntent.RADIO_THEATRE:      MediaType.RADIO_THEATRE,
    OCPPlayIntent.VISUAL_STORY:       MediaType.VISUAL_STORY,
    OCPPlayIntent.ASMR:               MediaType.ASMR,
    OCPPlayIntent.AUDIO_DESCRIPTION:  MediaType.AUDIO_DESCRIPTION,
    OCPPlayIntent.MUSIC_VIDEO:        MediaType.MUSIC_VIDEO,
    OCPPlayIntent.TRAILER:            MediaType.TRAILER,
    OCPPlayIntent.BEHIND_THE_SCENES:  MediaType.BEHIND_THE_SCENES,
    OCPPlayIntent.ADULT:              MediaType.ADULT,
    OCPPlayIntent.ADULT_AUDIO:        MediaType.ADULT_AUDIO,
    OCPPlayIntent.HENTAI:             MediaType.HENTAI,
    OCPPlayIntent.GENERIC:            MediaType.GENERIC,
}

MEDIA_TYPE_TO_PLAY_INTENT: Dict[MediaType, OCPPlayIntent] = {
    v: k for k, v in PLAY_INTENT_TO_MEDIA_TYPE.items()
}

# String form of intent labels → MediaType (used by backends that emit raw strings)
LABEL_TO_MEDIA_TYPE: Dict[str, MediaType] = {
    intent.value: mt for intent, mt in PLAY_INTENT_TO_MEDIA_TYPE.items()
}

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
