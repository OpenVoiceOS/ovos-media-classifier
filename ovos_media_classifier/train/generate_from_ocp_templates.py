"""Generate training data by filling OpenVoiceOS/OCP_templates with real Wikidata entities.

This script uses two HuggingFace datasets:

  ``OpenVoiceOS/OCP_templates``
      3,409 sentence templates with labelled slots like
      ``{actor_name}``, ``{movie_name}``, ``{generic_play_verb}``, …
      Each row also carries ``binary_label``, ``playback_label``,
      ``adult_label``, and ``media_label``.

  ``Jarbas/WikidataMediaEntities``
      1.6 M entity strings keyed by ``entity_type``
      (movie_name, artist_name, game_name, anime_name, …).

Slots that reference real-world entities are filled from WikidataMediaEntities;
"literal" slots (verbs, keywords, genre adjectives, provider names) are filled
from compact hardcoded lists.

Output schema (same as gather_dataset.py)::

    lang, domain, intent, binary_label, playback_label, media_label, sentence

Run ``download_datasets.py`` first to pre-warm the HuggingFace cache.

Usage::

    python -m ovos_media_classifier.train.generate_from_ocp_templates
    python -m ovos_media_classifier.train.generate_from_ocp_templates --n 30 --output out.csv
    python -m ovos_media_classifier.train.generate_from_ocp_templates \\
        --media-labels movie music podcast \\
        --n 50 --dedup-against ocp_dataset.csv
"""

# TODO - entity lists
#  https://huggingface.co/datasets/TigreGotico/metal-archives-bands
#  https://huggingface.co/datasets/TigreGotico/metal-archives-tracks
#  https://huggingface.co/datasets/TigreGotico/jazz-music-archives
# https://huggingface.co/datasets/TigreGotico/prog-archives
# https://huggingface.co/datasets/TigreGotico/classic-composers
# https://huggingface.co/datasets/TigreGotico/trance_tracks
# https://huggingface.co/datasets/TigreGotico/movie_actors
# https://huggingface.co/datasets/TigreGotico/movie_directors
# https://huggingface.co/datasets/TigreGotico/movie_writers
# https://huggingface.co/datasets/TigreGotico/movie_producers
# https://huggingface.co/datasets/TigreGotico/movie_composers
#
#

from __future__ import annotations

import os
import random
import re
from typing import Optional

import pandas as pd

from ovos_media_classifier.train import get_hf_cache_dir, get_output_dir

random.seed(42)

# ---------------------------------------------------------------------------
# media_label → (ocp_domain, ocp_intent)
# ---------------------------------------------------------------------------

_LABEL_TO_OCP: dict[str, tuple[str, str]] = {
    "music":              ("ocp_play", "music"),
    "movie":              ("ocp_play", "movie"),
    "series":             ("ocp_play", "tv_show"),
    "tv":                 ("ocp_play", "tv"),
    "tv_show":            ("ocp_play", "tv_show"),
    "anime":              ("ocp_play", "anime"),
    "cartoon":            ("ocp_play", "cartoon"),
    "documentary":        ("ocp_play", "documentary"),
    "podcast":            ("ocp_play", "podcast"),
    "radio":              ("ocp_play", "radio"),
    "radio_theatre":      ("ocp_play", "radio_theatre"),
    "audiobook":          ("ocp_play", "audiobook"),
    "game":               ("ocp_play", "game"),
    "short_film":         ("ocp_play", "short_film"),
    "silent_movie":       ("ocp_play", "silent_movie"),
    "black_white_movie":  ("ocp_play", "bw_movie"),
    "asmr":               ("ocp_play", "asmr"),
    "audio_description":  ("ocp_play", "audio_description"),
    "hentai":             ("ocp_play", "hentai"),
    "porn":               ("ocp_play", "adult"),
    "adult_asmr":         ("ocp_play", "adult_audio"),
    "adult_game":         ("ocp_play", "adult"),
    "video":              ("ocp_play", "video"),
    "news":               ("ocp_play", "news"),
    "trailer":            ("ocp_play", "trailer"),
    "behind_the_scenes":  ("ocp_play", "behind_the_scenes"),
    "music_video":        ("ocp_play", "music_video"),
    "comic_book":         ("ocp_play", "visual_story"),
    "short_sound":        ("ocp_play", "audio"),
    "ambient_sounds":     ("ocp_play", "asmr"),
    "not_media":          ("not_ocp", "not_ocp"),
    # device-specific — skip (not content-based classification)
    "video_device":       None,
    "audio_device":       None,
    "game_device":        None,
}

_MEDIA_TYPE_TO_GENRE_SLOT: dict[str, str] = {
    "MOVIE": "movie_genre",
    "DOCUMENTARY": "movie_genre",
    "SHORT_FILM": "movie_genre",
    "SILENT_MOVIE": "movie_genre",
    "BLACK_WHITE_MOVIE": "movie_genre",
    "TRAILER": "movie_genre",
    "BEHIND_THE_SCENES": "movie_genre",
    "MUSIC": "music_genre",
    "MUSIC_VIDEO": "music_genre",
    "TV_SHOW": "series_genre",
    "ANIME": "anime_genre",
    "CARTOON": "cartoon_genre",
    "AUDIOBOOK": "audiobook_genre",
    "PODCAST": "podcast_genre",
    "GAME": "game_genre",
    "NEWS": "news_genre",
    "RADIO": "radio_genre",
}

# ---------------------------------------------------------------------------
# Wikidata entity_type → which template slot names it fills
# ---------------------------------------------------------------------------

_WIKIDATA_SLOT_MAP: dict[str, list[str]] = {
    "movie_name":        ["movie_name", "trailer_name", "behind_the_scenes_name"],
    "short_film_name":   ["short_film_name"],
    "silent_movie_name": ["silent_movie_name"],
    "bw_movie_name":     ["black_white_movie_name"],
    "artist_name":       ["music_artist_name", "artist_name"],
    "album_name":        ["album_name", "music_name"],
    "song_name":         ["music_name"],
    "game_name":         ["game_name", "adult_game_name"],
    "series_name":       ["series_name"],
    "anime_name":        ["anime_name"],
    "cartoon_name":      ["cartoon_name"],
    "book_name":         ["audiobook_name"],
    "book_author":       ["author_name"],
    "audiobook_narrator":["narrator_name"],
    "podcast_name":      ["podcast_name"],
    "podcaster":         ["podcast_host"],
    "documentary_name":  ["documentary_name"],
    "movie_director":    ["director_name"],
    "movie_actor":       ["actor_name", "character_name"],
    "music_genre":       ["music_genre"],
    "film_genre":        ["movie_genre"],
    "book_genre":        ["audiobook_genre"],
    "game_genre":        ["game_genre"],
    "news_provider":     ["news_provider", "news_name"],
    "country_name":      ["country_name"],
    "radio_drama_name":  ["radio_theatre_name"],
    "radio_drama_actor": ["radio_theatre_name"],
    "pornstar_name":     ["pornstar_name"],
    "hentai_name":       ["hentai_name"],
    "porn_film_name":    ["porn_name"],
    "youtube_channel":   ["video_channel", "video_name"],
    "tv_channel":        ["tv_name", "video_channel"],
    "tv_streaming_service": ["tv_provider", "series_provider"],
    "music_streaming_service": ["music_provider"],
    "podcast_streaming_service": ["podcast_provider"],
    "radio_streaming_service": ["radio_provider"],
    "audiobook_streaming_service": ["audiobook_provider"],
    "video_streaming_service": ["video_provider"],
    "movie_streaming_service": ["movie_provider", "trailer_provider",
                                "short_film_provider", "silent_movie_provider",
                                "black_white_movie_provider", "documentary_provider",
                                "behind_the_scenes_provider", "audio_description_provider"],
    "gaming_console_name": ["game_device_name", "game_device_provider",
                             "game_provider", "adult_game_provider"],
    "film_studio":       ["movie_provider"],
}

# ---------------------------------------------------------------------------
# Hardcoded literal/verb/genre/provider pools
# ---------------------------------------------------------------------------

_LITERAL_SLOTS: dict[str, list[str]] = {
    # --- Play verbs ---
    "generic_play_verb": [
        "play", "stream", "put on", "start", "queue up", "find me", "get me",
        "I want to watch", "I want to listen to", "show me", "give me", "load",
    ],
    "video_play_verb": [
        "watch", "play", "stream", "show me", "put on", "view", "display",
        "I want to watch", "find me",
    ],
    "audio_play_verb": [
        "play", "listen to", "stream", "put on", "hear", "start",
        "I want to listen to", "find me",
    ],
    "book_play_verb": [
        "read", "read me", "listen to", "play", "start reading", "narrate",
    ],
    "game_play_verb": [
        "play", "launch", "start", "open", "load", "run", "boot up",
    ],
    "device_play_verb": [
        "connect to", "open", "launch", "use", "switch to", "open on",
    ],
    "question_verb": [
        "what", "who", "which", "where", "when", "how",
    ],

    # --- Media type literals ---
    "movie_literal": [
        "movie", "film", "flick", "feature film", "picture", "motion picture",
    ],
    "audiobook_literal": [
        "audiobook", "audio book", "book", "audio edition",
    ],
    "podcast_literal": [
        "podcast", "show", "episode", "podcast episode", "audio show",
    ],
    "documentary_literal": [
        "documentary", "doc", "documentary film", "docu",
    ],
    "game_literal": [
        "game", "video game", "videogame",
    ],
    "cartoon_literal": [
        "cartoon", "animated series", "animation", "animated show",
    ],
    "anime_literal": [
        "anime", "japanese animation", "anime series",
    ],
    "radio_theatre_literal": [
        "radio theatre", "radio play", "audio drama", "radio drama",
    ],
    "asmr_literal": [
        "ASMR", "asmr", "ASMR video", "asmr audio", "asmr sounds",
    ],
    "silent_movie_literal": [
        "silent movie", "silent film", "silent picture",
    ],
    "short_film_literal": [
        "short film", "short", "short movie",
    ],
    "black_white_movie_literal": [
        "black and white movie", "black and white film", "b&w movie", "classic film",
    ],
    "audio_description_literal": [
        "audio description", "described video", "AD version", "audio described",
    ],
    "episode_literal": [
        "episode", "ep", "the next episode", "season premiere", "latest episode",
    ],
    "season_literal": [
        "season 1", "season 2", "the first season", "the latest season", "a new season",
    ],
    "series_literal": [
        "series", "TV show", "show", "season", "TV series",
    ],
    "music_literal": [
        "music", "song", "track", "tune", "album", "playlist",
    ],
    "trailer_literal": [
        "trailer", "preview", "teaser", "official trailer",
    ],
    "behind_the_scenes_literal": [
        "behind the scenes", "making of", "BTS", "extras", "featurette",
    ],
    "porn_literal": [
        "porn", "adult film", "adult video", "adult content",
    ],
    "hentai_literal": [
        "hentai", "adult anime",
    ],
    "adult_asmr_literal": [
        "adult ASMR", "erotic ASMR",
    ],
    "adult_game_literal": [
        "adult game", "mature game",
    ],
    "short_sound_literal": [
        "sound effect", "sound clip", "audio clip", "sound bite",
    ],
    "comic_book_literal": [
        "comic book", "comic", "graphic novel",
    ],
    "ambient_sounds_literal": [
        "ambient sounds", "ambient music", "background sounds", "white noise",
    ],
    "game_device_literal": [
        "game console", "gaming device", "gaming system",
    ],
    "video_device_literal": [
        "TV", "television", "display", "smart TV",
    ],
    "audio_device_literal": [
        "speaker", "headphones", "earbuds", "stereo",
    ],
    "visually_impaired_literal": [
        "audio description", "for the visually impaired", "described version",
    ],
    "news_literal": [
        "news", "news bulletin", "headlines", "latest news", "the news",
    ],
    "video_literal": [
        "video", "clip", "footage", "content",
    ],
    "radio_literal": [
        "radio", "radio station", "FM radio",
    ],
    "tv_literal": [
        "TV show", "television show", "TV series", "TV program",
    ],
    "ŋame_literal": [  # typo in original dataset
        "game", "video game",
    ],

    # --- Genre slots ---
    "movie_genre": [
        "action", "comedy", "horror", "sci-fi", "romance", "thriller", "drama",
        "animation", "fantasy", "adventure", "crime", "mystery", "western",
    ],
    "anime_genre": [
        "shonen", "shojo", "mecha", "isekai", "slice of life", "fantasy",
        "action", "romance", "horror", "psychological",
    ],
    "cartoon_genre": [
        "adventure", "comedy", "action", "educational", "family", "superhero",
    ],
    "game_genre": [
        "action", "RPG", "strategy", "puzzle", "first-person shooter", "sports",
        "adventure", "simulation", "platformer", "racing", "fighting",
    ],
    "asmr_genre": [
        "relaxing", "sleep", "tapping", "whispering", "nature sounds", "rain sounds",
        "crinkling", "soft-spoken",
    ],
    "porn_genre": [
        "adult", "mature",
    ],
    "documentary_genre": [
        "nature", "history", "crime", "science", "sports", "technology",
        "food", "travel", "music", "politics",
    ],
    "short_film_genre": [
        "drama", "comedy", "animated", "experimental", "horror",
    ],
    "silent_movie_genre": [
        "comedy", "drama", "adventure", "romance",
    ],
    "black_white_movie_genre": [
        "noir", "drama", "comedy", "western", "thriller", "romance",
    ],
    "audiobook_genre": [
        "fiction", "non-fiction", "fantasy", "thriller", "romance",
        "science fiction", "biography", "self-help", "history",
    ],
    "podcast_genre": [
        "technology", "comedy", "true crime", "news", "sports", "history",
        "science", "business", "health", "education",
    ],
    "radio_genre": [
        "news", "music", "talk", "sport", "classical", "jazz", "rock",
    ],
    "video_genre": [
        "gaming", "comedy", "music", "education", "news", "sports", "cooking",
    ],
    "series_genre": [
        "drama", "comedy", "crime", "sci-fi", "fantasy", "thriller",
        "documentary", "reality", "animation",
    ],
    "news_genre": [
        "world news", "local news", "sports news", "tech news", "business news",
        "entertainment news", "political news",
    ],
    "short_sound_genre": [
        "nature", "ambient", "city", "animal", "music",
    ],
    "ambient_sounds_genre": [
        "rain", "ocean waves", "forest", "white noise", "thunderstorm", "fireplace",
    ],
    "game_device_genre": [
        "action", "RPG", "sports", "racing",
    ],
    "video_device_genre": [
        "movie", "TV", "sports",
    ],
    "audio_device_genre": [
        "music", "podcast", "radio",
    ],
    "radio_theatre_genre": [
        "mystery", "comedy", "drama", "horror", "sci-fi", "adventure",
    ],
    "hentai_genre": [
        "hentai", "adult anime",
    ],
    "adult_asmr_genre": [
        "ASMR",
    ],
    "adult_game_genre": [
        "adult", "mature",
    ],
    "trailer_genre": [
        "action", "comedy", "horror", "sci-fi", "drama",
    ],
    "behind_the_scenes_genre": [
        "documentary", "interview", "featurette",
    ],
    "comic_book_genre": [
        "superhero", "horror", "sci-fi", "fantasy", "crime",
    ],

    # --- Provider slots ---
    "audiobook_provider": [
        "Audible", "Librivox", "Libro.fm", "Storytel", "Scribd",
    ],
    "radio_theatre_provider": [
        "BBC Radio 4", "Audible", "BBC Sounds", "Escape Pod",
    ],
    "cartoon_provider": [
        "Disney+", "Cartoon Network", "Nickelodeon", "Netflix", "Amazon Prime",
    ],
    "anime_provider": [
        "Crunchyroll", "Funimation", "Netflix", "Disney+", "Hidive",
    ],
    "game_provider": [
        "Steam", "Xbox", "PlayStation", "Nintendo Switch", "Epic Games Store",
    ],
    "music_provider": [
        "Spotify", "Apple Music", "Tidal", "Deezer", "YouTube Music", "Amazon Music",
    ],
    "movie_provider": [
        "Netflix", "Prime Video", "Disney+", "HBO Max", "Apple TV+", "Hulu",
    ],
    "video_provider": [
        "YouTube", "Vimeo", "Netflix", "Prime Video",
    ],
    "podcast_provider": [
        "Spotify", "Apple Podcasts", "Google Podcasts", "Overcast", "Pocket Casts",
    ],
    "short_film_provider": [
        "Vimeo", "YouTube", "Mubi",
    ],
    "silent_movie_provider": [
        "YouTube", "Internet Archive", "Mubi", "Fandor",
    ],
    "series_provider": [
        "Netflix", "HBO Max", "Amazon Prime", "Disney+", "Hulu", "Apple TV+",
    ],
    "black_white_movie_provider": [
        "Mubi", "TCM", "Amazon Prime", "YouTube", "Internet Archive",
    ],
    "documentary_provider": [
        "Netflix", "Disney+", "HBO Max", "BBC", "Discovery+",
    ],
    "asmr_provider": [
        "YouTube", "Spotify", "SoundCloud",
    ],
    "radio_provider": [
        "Spotify", "BBC", "iHeartRadio", "Pandora", "TuneIn",
    ],
    "tv_provider": [
        "Netflix", "HBO Max", "Disney+", "Amazon Prime", "Hulu",
    ],
    "trailer_provider": [
        "YouTube", "IMDb",
    ],
    "audio_description_provider": [
        "Netflix", "BBC", "Amazon Prime",
    ],
    "comic_book_provider": [
        "Marvel Unlimited", "ComiXology", "Kindle",
    ],
    "behind_the_scenes_provider": [
        "YouTube", "IMDb", "Netflix", "Disney+",
    ],
    "short_sound_provider": [
        "YouTube", "SoundCloud", "Freesound",
    ],
    "ambient_sounds_provider": [
        "YouTube", "Spotify", "Calm", "Headspace", "Noisli",
    ],
    "game_device_provider": [
        "Xbox", "PlayStation", "Nintendo Switch", "Steam",
    ],
    "video_device_provider": [
        "Netflix", "YouTube", "Disney+", "Prime Video",
    ],
    "audio_device_provider": [
        "Spotify", "Apple Music", "SoundCloud",
    ],
    "porn_provider": [
        "adult content site",
    ],
    "hentai_provider": [
        "anime streaming site",
    ],
    "adult_asmr_provider": [
        "adult content site",
    ],
    "adult_game_provider": [
        "adult game store",
    ],

    # --- Other ---
    "indoor_location": [
        "bedroom", "living room", "kitchen", "office", "bathroom", "gym",
    ],
    "video_device_name": [
        "TV", "laptop", "tablet", "projector", "monitor", "smart TV",
    ],
    "audio_device_name": [
        "speaker", "headphones", "earbuds", "stereo system",
    ],
    "game_device_name": [
        "Xbox", "PlayStation", "Nintendo Switch", "PC", "laptop",
    ],
    "music_playlist_name": [
        "my workout playlist", "chill vibes", "road trip mix", "study music", "favorites",
    ],
    "radio_name": [
        "BBC Radio 1", "NPR", "KEXP", "Radio Paradise", "SomaFM",
    ],
    "asmr_name": [
        "ASMR Darling", "Gibi ASMR", "WhispersRed", "Gentle Whispering",
    ],
    "ambient_sounds_name": [
        "rain sounds", "ocean waves", "forest ambience", "thunderstorm sounds",
    ],
    "short_sound_name": [
        "a bird chirping", "thunder", "ocean waves", "rain drops",
    ],
    "adult_asmr_name": [
        "adult ASMR",
    ],
    "comic_book_name": [
        "Batman", "Spider-Man", "Superman", "X-Men", "The Walking Dead",
    ],
    "video_channel": [
        "Linus Tech Tips", "Veritasium", "Kurzgesagt", "CGP Grey", "3Blue1Brown",
    ],
}


# ---------------------------------------------------------------------------
# Load Wikidata entities → per-slot pool
# ---------------------------------------------------------------------------

def load_wikidata_pools(hf_cache: str, max_per_type: int = 50000) -> dict[str, list[str]]:
    """Load Jarbas/WikidataMediaEntities and build per-slot-name pools."""
    print("Loading Jarbas/WikidataMediaEntities …", end="", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("Jarbas/WikidataMediaEntities", split="train", cache_dir=hf_cache)
    except Exception as e:
        print(f" FAILED: {e}")
        return {}

    # Group by entity_type
    by_type: dict[str, set] = {}
    for row in ds:
        etype = row["entity_type"]
        text = (row.get("text") or "").strip()
        if text and len(text) > 1:
            if etype not in by_type:
                by_type[etype] = set()
            if len(by_type[etype]) < max_per_type:
                by_type[etype].add(text)

    print(f" {sum(len(v) for v in by_type.values()):,} entities across {len(by_type)} types")

    # Map entity_type → slot name(s)
    pools: dict[str, list[str]] = {}
    for etype, slot_names in _WIKIDATA_SLOT_MAP.items():
        values = list(by_type.get(etype, []))
        for slot in slot_names:
            if slot not in pools:
                pools[slot] = []
            pools[slot].extend(values)

    # Deduplicate
    for slot in pools:
        pools[slot] = list(set(pools[slot]))

    return pools


# ---------------------------------------------------------------------------
# Entity pools from local sources and HuggingFace
# ---------------------------------------------------------------------------

_STREAMING_SERVICES: dict[str, list[str]] = {
    "music_streaming_service": [
        "Spotify", "Apple Music", "Tidal", "Deezer", "Qobuz", "Amazon Music",
        "YouTube Music", "SoundCloud", "Bandcamp", "Last.fm", "Pandora",
    ],
    "movie_streaming_service": [
        "Netflix", "Disney+", "Prime Video", "HBO Max", "Apple TV+",
        "Hulu", "Peacock", "Paramount+", "Crunchyroll", "Mubi",
    ],
    "podcast_streaming_service": [
        "Spotify", "Apple Podcasts", "Google Podcasts", "Overcast",
        "Pocket Casts", "Castbox",
    ],
    "radio_streaming_service": [
        "BBC Radio", "NPR", "Radio Paradise", "KEXP", "SomaFM",
        "iHeartRadio",
    ],
    "news_provider": [
        "BBC News", "CNN", "Reuters", "AP News", "The Guardian",
        "NPR News", "Al Jazeera",
    ],
    "audiobook_streaming_service": [
        "Audible", "Librivox", "Libro.fm", "Storytel",
    ],
    "tv_streaming_service": [
        "Netflix", "Disney+", "HBO Max", "Apple TV+", "Hulu",
        "Amazon Prime", "Peacock", "Paramount+",
    ],
}


def load_media_entity_pools(csv_path: str) -> dict[str, list[str]]:
    """Load entities from a wide-format CSV (output of generate_dataset_from_media.py)."""
    print(f"Loading media entities from {csv_path} …", end="", flush=True)
    pools: dict[str, list[str]] = {}

    def add_val(slot: str, val: str):
        if not val or not str(val).strip():
            return
        if slot not in pools:
            pools[slot] = []
        # Support pipe-separated values
        for v in val.split("|"):
            v = v.strip()
            if v and v not in pools[slot]:
                pools[slot].append(v)

    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # direct: row.ocp_label IS the slot name for the title
            if hasattr(row, "ocp_label") and hasattr(row, "title"):
                add_val(str(row.ocp_label), str(row.title))

            # expanded columns
            if hasattr(row, "actor"):
                add_val("movie_actor", str(row.actor))
            if hasattr(row, "director"):
                add_val("movie_director", str(row.director))
            if hasattr(row, "producer"):
                add_val("movie_producer", str(row.producer))
            if hasattr(row, "writer"):
                add_val("movie_writer", str(row.writer))
            if hasattr(row, "composer"):
                add_val("movie_composer", str(row.composer))
            if hasattr(row, "artist"):
                add_val("artist_name", str(row.artist))
            if hasattr(row, "author"):
                add_val("audiobook_author", str(row.author))

            # genre column mapping
            if hasattr(row, "genre") and hasattr(row, "media_type"):
                mtype = str(row.media_type)
                slot = _MEDIA_TYPE_TO_GENRE_SLOT.get(mtype)
                if slot:
                    add_val(slot, str(row.genre))

    except Exception as e:
        print(f" FAILED: {e}")
        return {}

    print(f" {sum(len(v) for v in pools.values()):,} entities across {len(pools)} slots")
    return pools


def load_local_templates(templates_dir: str) -> list[tuple[str, str]]:
    """Load all *_templates.csv files from a directory."""
    import glob
    templates: list[tuple[str, str]] = []
    files = glob.glob(os.path.join(templates_dir, "*_templates.csv"))
    print(f"Loading {len(files)} local template files from {templates_dir}")
    for f in files:
        media_label = os.path.basename(f).removesuffix("_templates.csv")
        try:
            df = pd.read_csv(f)
            # Schema: category, template
            for _, row in df.iterrows():
                if "template" in row:
                    templates.append((str(row["template"]), media_label))
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return templates


def load_ner_entity_pools(hf_cache: str) -> dict[str, list[str]]:
    """Load HuggingFace NER datasets as raw entity pools."""
    print("Loading HuggingFace NER datasets …", end="", flush=True)
    pools: dict[str, list[str]] = {}

    def add_val(slot: str, val: str):
        if not val or not str(val).strip():
            return
        if slot not in pools:
            pools[slot] = []
        if val not in pools[slot]:
            pools[slot].append(val)

    from datasets import load_dataset

    # Music datasets
    for ds_name in ["Jarbas/metal-archives-tracks", "Jarbas/metal-archives-bands",
                   "Jarbas/jazz-music-archives", "Jarbas/prog-archives",
                   "Jarbas/classic-composers", "Jarbas/trance_tracks"]:
        try:
            ds = load_dataset(ds_name, split="train", cache_dir=hf_cache)
            for entry in ds:
                if ds_name == "Jarbas/metal-archives-tracks":
                    add_val("artist_name", entry.get("band_name"))
                    add_val("track_name", entry.get("track_name"))
                    add_val("album_name", entry.get("album_name"))
                    add_val("album_type", entry.get("album_type"))
                elif ds_name == "Jarbas/metal-archives-bands":
                    add_val("artist_name", entry.get("name"))
                    add_val("music_genre", entry.get("genre"))
                    add_val("record_label", entry.get("label"))
                elif ds_name in ["Jarbas/jazz-music-archives", "Jarbas/prog-archives"]:
                    add_val("artist_name", entry.get("artist"))
                    add_val("music_genre", entry.get("genre"))
                elif ds_name == "Jarbas/classic-composers":
                    add_val("artist_name", entry.get("name"))
                elif ds_name == "Jarbas/trance_tracks":
                    add_val("artist_name", entry.get("ARTIST(S)"))
                    add_val("track_name", entry.get("TRACK"))
                    add_val("music_genre", entry.get("STYLE"))
        except Exception as e:
            print(f"\nWarning: failed to load {ds_name}: {e}")

    # Movie datasets
    for label, ds_name in [
        ("movie_actor",    "Jarbas/movie_actors"),
        ("movie_director", "Jarbas/movie_directors"),
        ("movie_producer", "Jarbas/movie_producers"),
        ("movie_writer",   "Jarbas/movie_writers"),
        ("movie_composer", "Jarbas/movie_composers"),
    ]:
        try:
            ds = load_dataset(ds_name, split="train", cache_dir=hf_cache)
            for entry in ds:
                add_val(label, entry.get("name"))
        except Exception as e:
            print(f"\nWarning: failed to load {ds_name}: {e}")

    print(f" {sum(len(v) for v in pools.values()):,} entities across {len(pools)} slots")
    return pools


# ---------------------------------------------------------------------------
# Template filling
# ---------------------------------------------------------------------------

_SLOT_RE = re.compile(r"\{([^}]+)\}")


def _fill_template(template: str, pools: dict[str, list[str]], rng: random.Random) -> Optional[str]:
    """Fill all slots in a template. Returns None if any required slot has no data."""
    slots = _SLOT_RE.findall(template)
    result = template
    for slot in slots:
        pool = pools.get(slot)
        if not pool:
            return None  # skip templates with missing slots
        value = rng.choice(pool)
        result = result.replace("{" + slot + "}", value, 1)
    return result


def generate_from_templates(
    media_labels: Optional[list[str]],
    n_per_template: int,
    hf_cache: str,
    dedup_against: Optional[str],
    include_negatives: bool,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)

    # Load templates
    print("Loading OpenVoiceOS/OCP_templates …", end="", flush=True)
    try:
        from datasets import load_dataset
        tpl_ds = load_dataset("OpenVoiceOS/OCP_templates", split="train", cache_dir=hf_cache)
        templates_df = tpl_ds.to_pandas()
    except Exception as e:
        print(f" FAILED: {e}")
        return pd.DataFrame(columns=["lang", "domain", "intent", "binary_label",
                                      "playback_label", "media_label", "sentence"])
    print(f" {len(templates_df)} templates")

    # Load all entity pools: literal → curated (keyword/synthetic) → Wikidata
    # Later sources override earlier ones when both supply the same slot,
    # so Wikidata (richest) wins on overlap.
    all_pools: dict[str, list[str]] = dict(_LITERAL_SLOTS)

    print("Loading curated entity pools …", end="", flush=True)
    curated_pools = load_curated_pools()
    print(f" {sum(len(v) for v in curated_pools.values()):,} values across {len(curated_pools)} slots")
    for slot, values in curated_pools.items():
        if slot in all_pools:
            all_pools[slot] = list(dict.fromkeys(all_pools[slot] + values))
        else:
            all_pools[slot] = values

    wikidata_pools = load_wikidata_pools(hf_cache)
    for slot, values in wikidata_pools.items():
        if slot in all_pools:
            all_pools[slot] = list(dict.fromkeys(all_pools[slot] + values))
        else:
            all_pools[slot] = values

    # Filter templates by requested media labels
    if media_labels:
        templates_df = templates_df[templates_df["media_label"].isin(media_labels)]
    if not include_negatives:
        templates_df = templates_df[templates_df["media_label"] != "not_media"]

    # Skip device-specific labels (no content-based intent)
    skip_labels = {k for k, v in _LABEL_TO_OCP.items() if v is None}
    templates_df = templates_df[~templates_df["media_label"].isin(skip_labels)]

    print(f"Filling {len(templates_df)} templates × {n_per_template} samples each …")

    # Load existing sentences for dedup
    existing: set[str] = set()
    if dedup_against and os.path.exists(dedup_against):
        df_ex = pd.read_csv(dedup_against)
        if "sentence" in df_ex.columns:
            existing = set(df_ex["sentence"].dropna().str.lower())
            print(f"Deduplicating against {len(existing):,} existing sentences")

    rows: list[tuple] = []
    skipped_labels: set[str] = set()

    # Playback label derivation
    from ovos_media_classifier.train.sources import AUDIO_INTENTS as _AUDIO, VIDEO_INTENTS as _VIDEO

    for _, tpl_row in templates_df.iterrows():
        template = str(tpl_row["template"])
        media_label = str(tpl_row["media_label"])
        ocp = _LABEL_TO_OCP.get(media_label)
        if ocp is None:
            skipped_labels.add(media_label)
            continue
        domain, intent = ocp

        # Use binary_label/playback_label from the template row when available,
        # otherwise derive them.
        if "binary_label" in tpl_row.index:
            binary_label = str(tpl_row["binary_label"])
        else:
            binary_label = "ocp" if domain in ("ocp_play", "ocp_control") else "not_ocp"

        if "playback_label" in tpl_row.index:
            playback_label = str(tpl_row["playback_label"])
        else:
            if domain != "ocp_play":
                playback_label = "undefined"
            elif intent in _AUDIO:
                playback_label = "audio"
            elif intent in _VIDEO:
                playback_label = "video"
            else:
                playback_label = "undefined"

        out_media_label = intent if domain == "ocp_play" else "not_ocp"

        seen_in_template: set[str] = set()
        attempts = 0
        while len(seen_in_template) < n_per_template and attempts < n_per_template * 5:
            attempts += 1
            sentence = _fill_template(template, all_pools, rng)
            if sentence is None:
                break  # missing slot data — skip this template
            sentence = sentence.lower().strip()
            if sentence and sentence not in existing and sentence not in seen_in_template:
                seen_in_template.add(sentence)
                rows.append(("en", domain, intent, binary_label, playback_label,
                             out_media_label, sentence))

    if skipped_labels:
        print(f"Skipped labels (no ocp mapping): {sorted(skipped_labels)}")

    df = pd.DataFrame(rows, columns=["lang", "domain", "intent", "binary_label",
                                      "playback_label", "media_label", "sentence"])
    df.drop_duplicates(subset=["sentence"], inplace=True)
    return df


def generate_from_local_templates(
    templates_dir: str,
    media_csv: Optional[str],
    n_per_template: int,
    hf_cache: str,
    dedup_against: Optional[str],
    seed: int,
) -> pd.DataFrame:
    """Generate utterances from local CSV templates filling slots with entities."""
    rng = random.Random(seed)

    # 1. Build slot pools in priority order
    all_pools: dict[str, list[str]] = dict(_LITERAL_SLOTS)

    # Merge OCPMediaNER._STREAMING_SERVICES
    for slot, values in _STREAMING_SERVICES.items():
        if slot in all_pools:
            all_pools[slot] = list(dict.fromkeys(all_pools[slot] + values))
        else:
            all_pools[slot] = values

    # Merge load_ner_entity_pools(hf_cache)
    ner_pools = load_ner_entity_pools(hf_cache)
    for slot, values in ner_pools.items():
        if slot in all_pools:
            all_pools[slot] = list(dict.fromkeys(all_pools[slot] + values))
        else:
            all_pools[slot] = values

    # Merge load_media_entity_pools(media_csv) if path provided
    if media_csv and os.path.exists(media_csv):
        media_pools = load_media_entity_pools(media_csv)
        for slot, values in media_pools.items():
            if slot in all_pools:
                all_pools[slot] = list(dict.fromkeys(values + all_pools[slot])) # Priority to media_csv
            else:
                all_pools[slot] = values

    # Merge load_wikidata_pools(hf_cache)
    wikidata_pools = load_wikidata_pools(hf_cache)
    for slot, values in wikidata_pools.items():
        if slot in all_pools:
            all_pools[slot] = list(dict.fromkeys(all_pools[slot] + values))
        else:
            all_pools[slot] = values

    # 2. Load local templates
    templates = load_local_templates(templates_dir)

    # 3. Fill templates
    print(f"Filling {len(templates)} local templates × {n_per_template} samples each …")

    # Load existing sentences for dedup
    existing: set[str] = set()
    if dedup_against and os.path.exists(dedup_against):
        df_ex = pd.read_csv(dedup_against)
        if "sentence" in df_ex.columns:
            existing = set(df_ex["sentence"].dropna().str.lower())
            print(f"Deduplicating against {len(existing):,} existing sentences")

    rows: list[tuple] = []
    from ovos_media_classifier.train.sources import AUDIO_INTENTS as _AUDIO, VIDEO_INTENTS as _VIDEO

    for template, media_label in templates:
        # derive domain/intent from filename stem
        intent = media_label
        domain = "ocp_play"

        binary_label = "ocp"
        if intent in _AUDIO:
            playback_label = "audio"
        elif intent in _VIDEO:
            playback_label = "video"
        else:
            playback_label = "undefined"

        out_media_label = intent

        seen_in_template: set[str] = set()
        attempts = 0
        while len(seen_in_template) < n_per_template and attempts < n_per_template * 5:
            attempts += 1
            sentence = _fill_template(template, all_pools, rng)
            if sentence is None:
                break
            sentence = sentence.lower().strip()
            if sentence and sentence not in existing and sentence not in seen_in_template:
                seen_in_template.add(sentence)
                rows.append(("en", domain, intent, binary_label, playback_label,
                             out_media_label, sentence))

    df = pd.DataFrame(rows, columns=["lang", "domain", "intent", "binary_label",
                                      "playback_label", "media_label", "sentence"])
    df.drop_duplicates(subset=["sentence"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Public convenience wrapper
# ---------------------------------------------------------------------------

def generate_all(
    n_per_template: int = 20,
    media_labels: Optional[list[str]] = None,
    dedup_against: Optional[str] = None,
    include_negatives: bool = False,
    seed: int = 42,
) -> "pd.DataFrame":
    """Generate utterances from all OCP Wikidata templates. Convenience wrapper."""
    hf_cache = get_hf_cache_dir()
    return generate_from_templates(
        media_labels=media_labels,
        n_per_template=n_per_template,
        hf_cache=hf_cache,
        dedup_against=dedup_against,
        include_negatives=include_negatives,
        seed=seed,
    )


