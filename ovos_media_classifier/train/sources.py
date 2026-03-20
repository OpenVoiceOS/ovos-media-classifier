"""Shared dataset source lists for all OCP training scripts.

Single source of truth for:
  - CSV_SOURCES     — URLs of raw intent CSV files to download
  - MUSIC_CSV_SOURCES — music-specific CSVs (forced to ocp_play:music)
  - GITHUB_CSV_LANGS  — language codes for per-lang GitHub CSVs
  - HF_DATASETS     — HuggingFace dataset names used by NER + synthetic generation
  - AUDIO_INTENTS   — set of OCPPlayIntent string values that are audio-only
  - VIDEO_INTENTS   — set of OCPPlayIntent string values that are video
  - SCHEMA_COLUMNS  — standard column order for all output CSVs
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard schema
# ---------------------------------------------------------------------------

SCHEMA_COLUMNS: list[str] = [
    "lang", "domain", "intent",
    "binary_label", "playback_label", "media_label",
    "sentence",
]

# ---------------------------------------------------------------------------
# Playback classification sets
# ---------------------------------------------------------------------------

AUDIO_INTENTS: frozenset[str] = frozenset({
    "music", "podcast", "radio", "audiobook", "news", "radio_theatre",
    "asmr", "audio_description", "adult_audio", "audio",
})

VIDEO_INTENTS: frozenset[str] = frozenset({
    "movie", "tv", "tv_show", "anime", "cartoon", "documentary",
    "short_film", "silent_movie", "bw_movie", "hentai", "adult",
    "video", "video_episodes", "visual_story", "game",
    "music_video", "trailer", "behind_the_scenes",
})

# ---------------------------------------------------------------------------
# Raw CSV sources (downloaded to csv/ cache)
# ---------------------------------------------------------------------------

# General OVOS intent datasets (HuggingFace)
CSV_SOURCES: list[str] = [
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-common-query-intents/resolve/main/common_query.csv",
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-intents-massive-subset/resolve/main/ovos_massive_subset.csv",
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-llm-augmented-intents/resolve/main/augmented.csv",
    # Weather — not_ocp negative examples
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-weather-intents/resolve/main/weather_intents_en.csv",
    # Multilingual test sets
    "https://huggingface.co/datasets/OpenVoiceOS/MT-intents-dataset-pt-PT/resolve/main/train_pt-PT.csv",
    "https://huggingface.co/datasets/OpenVoiceOS/MT-intents-dataset-pt-PT/resolve/main/test_pt-PT.csv",
    "https://huggingface.co/datasets/OpenVoiceOS/MT-intents-dataset-pt-PT/resolve/main/validation_pt-PT.csv",
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-intents-ilenia-testset-ca/resolve/main/test.csv",
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-intents-ilenia-testset-es/resolve/main/test.csv",
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-intents-ilenia-testset-nl/resolve/main/test.csv",
]

# Music query template datasets — always forced to ocp_play:music
MUSIC_CSV_SOURCES: list[str] = [
    "https://huggingface.co/datasets/Jarbas/music_queries_templates/resolve/main/music_templates.csv",
    "https://huggingface.co/datasets/Jarbas/music_queries_psytrance_tracks/resolve/main/mq_psy_tracks.csv",
    "https://huggingface.co/datasets/Jarbas/music_queries_metal_tracks/resolve/main/mq_ma_tracks.csv",
    "https://huggingface.co/datasets/Jarbas/music_queries_classical/resolve/main/mq_classical.csv",
    "https://huggingface.co/datasets/Jarbas/music_queries_prog/resolve/main/mq_prog.csv",
    "https://huggingface.co/datasets/Jarbas/music_queries_jazz/resolve/main/mq_jazz.csv",
    "https://huggingface.co/datasets/Jarbas/music_queries_metal_bands/resolve/main/mq_ma_bands.csv",
]

# Per-language GitHub intent datasets
GITHUB_CSV_LANGS: list[str] = ["en", "pt", "eu", "es", "gl", "nl", "fr", "de", "ca", "it", "da"]

GITHUB_CSV_SOURCES: list[str] = [
    f"https://raw.githubusercontent.com/OpenVoiceOS/lang-support-tracker/refs/heads/dev/skills/intents_{lang}.csv"
    for lang in GITHUB_CSV_LANGS
]

# Flat list of everything (used by download_datasets.py)
ALL_CSV_SOURCES: list[str] = CSV_SOURCES + MUSIC_CSV_SOURCES + GITHUB_CSV_SOURCES

# ---------------------------------------------------------------------------
# HuggingFace dataset names  (NER + generate_synthetic.py)
# ---------------------------------------------------------------------------

HF_DATASETS: list[tuple[str, str]] = [
    # Music entity datasets
    ("Jarbas/metal-archives-tracks",  "train"),
    ("Jarbas/metal-archives-bands",   "train"),
    ("Jarbas/jazz-music-archives",    "train"),
    ("Jarbas/prog-archives",          "train"),
    ("Jarbas/classic-composers",      "train"),
    ("Jarbas/trance_tracks",          "train"),
    # Movie entity datasets
    ("Jarbas/movie_actors",           "train"),
    ("Jarbas/movie_directors",        "train"),
    ("Jarbas/movie_producers",        "train"),
    ("Jarbas/movie_writers",          "train"),
    ("Jarbas/movie_composers",        "train"),
    # Unified Wikidata media entity dataset (1.6 M entities across all media types)
    ("Jarbas/WikidataMediaEntities",  "train"),
    # Official OCP sentence templates with slot labels
    ("OpenVoiceOS/OCP_templates",     "train"),
]
