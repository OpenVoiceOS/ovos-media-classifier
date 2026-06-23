#!/usr/bin/env python3
"""Generate per-language, per-media-type sentence templates with slot placeholders.

Slot names map 1:1 to ``OCPEntityLabel`` string values.  Templates are used by:

- ``generate_slot_literal_dataset.py`` — keeps slot names as literal text
- ``generate_slot_filled_dataset.py``  — replaces slots with real entities

Output: one CSV per ``(lang, media_type)`` pair under ``<output_dir>/<lang>/<media_type>.csv``
Schema: ``lang, intent, template, slots``  (slots = comma-separated slot names in template)

Usage::

    python -m training.generate_templates
    python -m training.generate_templates \\
        --langs en-us pt-pt de-de \\
        --output /tmp/templates
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

from training import get_cache_dir

# ---------------------------------------------------------------------------
# Supported languages (must match locale folder names)
# ---------------------------------------------------------------------------

ALL_LANGS: List[str] = [
    "ca-es", "da-dk", "de-de", "en-us", "es-es",
    "eu", "fr-fr", "gl-es", "it-it", "nl-nl",
    "pl-pl", "pt-br", "pt-pt",
]

# ---------------------------------------------------------------------------
# Template definitions per media type
# Each entry is (template_string, intent)
# Slot names MUST match OCPEntityLabel string values exactly.
# ---------------------------------------------------------------------------

# A template list is: list of (template, intent) tuples.
# We define templates by language.

_EN_TEMPLATES: List[Tuple[str, str]] = [
    # ── music ────────────────────────────────────────────────────────────────
    ("play {artist_name}", "music"),
    ("play some {artist_name}", "music"),
    ("play {track_name}", "music"),
    ("play {track_name} by {artist_name}", "music"),
    ("I want to listen to {artist_name}", "music"),
    ("I want to hear {track_name}", "music"),
    ("put on some {artist_name}", "music"),
    ("stream {artist_name}", "music"),
    ("play music by {artist_name}", "music"),
    ("play {artist_name}'s music", "music"),
    ("play some {music_genre} music", "music"),
    ("I'm in the mood for {music_genre}", "music"),
    ("play the album {album_name}", "music"),
    ("play {album_name} by {artist_name}", "music"),
    ("stream {track_name}", "music"),
    ("find {track_name} for me", "music"),
    ("put on {album_name}", "music"),
    ("I'd like to hear some {music_genre}", "music"),
    ("play a {music_genre} playlist", "music"),
    ("play {record_label} artists", "music"),

    # ── movie ─────────────────────────────────────────────────────────────────
    ("play {movie_title}", "movie"),
    ("play the movie {movie_title}", "movie"),
    ("I want to watch {movie_title}", "movie"),
    ("put on {movie_title}", "movie"),
    ("show me {movie_title}", "movie"),
    ("play a movie with {movie_actor}", "movie"),
    ("I want to watch a {movie_genre} movie", "movie"),
    ("play a movie directed by {movie_director}", "movie"),
    ("find me a movie with {movie_actor}", "movie"),
    ("stream {movie_title}", "movie"),
    ("play the {movie_genre} film {movie_title}", "movie"),
    ("watch {movie_title}", "movie"),
    ("play something with {movie_actor}", "movie"),
    ("play a movie produced by {movie_producer}", "movie"),
    ("play {movie_title} directed by {movie_director}", "movie"),

    # ── tv_show ───────────────────────────────────────────────────────────────
    ("play {tv_show_title}", "tv_show"),
    ("play the show {tv_show_title}", "tv_show"),
    ("I want to watch {tv_show_title}", "tv_show"),
    ("put on {tv_show_title}", "tv_show"),
    ("stream {tv_show_title}", "tv_show"),
    ("play the {tv_genre} show {tv_show_title}", "tv_show"),
    ("show me {tv_show_title}", "tv_show"),
    ("watch {tv_show_title}", "tv_show"),
    ("I'd like to watch {tv_show_title}", "tv_show"),
    ("play a {tv_genre} series", "tv_show"),

    # ── podcast ───────────────────────────────────────────────────────────────
    ("play {podcast_title}", "podcast"),
    ("play the podcast {podcast_title}", "podcast"),
    ("I want to listen to {podcast_title}", "podcast"),
    ("put on {podcast_title}", "podcast"),
    ("play the latest {podcast_title}", "podcast"),
    ("stream {podcast_title}", "podcast"),
    ("play a {podcast_genre} podcast", "podcast"),
    ("find me {podcast_title}", "podcast"),
    ("play podcasts by {podcast_host}", "podcast"),
    ("I want to hear {podcast_title}", "podcast"),

    # ── audiobook ─────────────────────────────────────────────────────────────
    ("play {audiobook_title}", "audiobook"),
    ("read {audiobook_title}", "audiobook"),
    ("play the audiobook {audiobook_title}", "audiobook"),
    ("I want to listen to {audiobook_title}", "audiobook"),
    ("play {audiobook_title} by {audiobook_author}", "audiobook"),
    ("find me {audiobook_title}", "audiobook"),
    ("play a book by {audiobook_author}", "audiobook"),
    ("stream the audiobook {audiobook_title}", "audiobook"),
    ("play {audiobook_title} narrated by {audiobook_narrator}", "audiobook"),
    ("I want to hear {audiobook_title}", "audiobook"),

    # ── radio ──────────────────────────────────────────────────────────────────
    ("play {radio_station}", "radio"),
    ("tune in to {radio_station}", "radio"),
    ("put on {radio_station}", "radio"),
    ("I want to listen to {radio_station}", "radio"),
    ("stream {radio_station}", "radio"),
    ("find {radio_station} radio", "radio"),
    ("play some {radio_genre} radio", "radio"),
    ("switch to {radio_station}", "radio"),
    ("turn on {radio_station}", "radio"),
    ("play {radio_station} radio station", "radio"),

    # ── news ───────────────────────────────────────────────────────────────────
    ("play news from {news_provider}", "news"),
    ("I want to hear news from {news_provider}", "news"),
    ("play the latest {news_category} news", "news"),
    ("stream news from {news_provider}", "news"),
    ("put on {news_provider}", "news"),
    ("I want to hear the news from {news_provider}", "news"),
    ("play {news_provider} news", "news"),
    ("find me news about {news_category}", "news"),
    ("play today's {news_category} headlines", "news"),
    ("stream {news_provider}", "news"),

    # ── game ───────────────────────────────────────────────────────────────────
    ("play {game_title}", "game"),
    ("start {game_title}", "game"),
    ("launch {game_title}", "game"),
    ("I want to play {game_title}", "game"),
    ("open {game_title}", "game"),
    ("play {game_title} on {game_platform}", "game"),
    ("I want to play a {game_genre} game", "game"),
    ("find me {game_title}", "game"),
    ("play {game_genre} games", "game"),
    ("start a game of {game_title}", "game"),

    # ── anime ──────────────────────────────────────────────────────────────────
    ("play {anime_title}", "anime"),
    ("play the anime {anime_title}", "anime"),
    ("I want to watch {anime_title}", "anime"),
    ("stream {anime_title}", "anime"),
    ("show me {anime_title}", "anime"),
    ("watch {anime_title}", "anime"),
    ("play anime by {anime_studio}", "anime"),
    ("find me {anime_title}", "anime"),
    ("I'd like to watch {anime_title}", "anime"),
    ("put on {anime_title}", "anime"),

    # ── documentary ────────────────────────────────────────────────────────────
    ("play {documentary_title}", "documentary"),
    ("play the documentary {documentary_title}", "documentary"),
    ("I want to watch {documentary_title}", "documentary"),
    ("show me {documentary_title}", "documentary"),
    ("stream {documentary_title}", "documentary"),
    ("watch the documentary {documentary_title}", "documentary"),
    ("play a documentary directed by {movie_director}", "documentary"),
    ("find me {documentary_title}", "documentary"),
    ("put on {documentary_title}", "documentary"),
    ("I'd like to watch {documentary_title}", "documentary"),

    # ── cartoon ────────────────────────────────────────────────────────────────
    ("play {cartoon_title}", "cartoon"),
    ("play the cartoon {cartoon_title}", "cartoon"),
    ("I want to watch {cartoon_title}", "cartoon"),
    ("show me {cartoon_title}", "cartoon"),
    ("stream {cartoon_title}", "cartoon"),
    ("put on {cartoon_title}", "cartoon"),
    ("watch {cartoon_title}", "cartoon"),
    ("find me {cartoon_title}", "cartoon"),
    ("I'd like to watch {cartoon_title}", "cartoon"),
    ("play a cartoon called {cartoon_title}", "cartoon"),

    # ── asmr ───────────────────────────────────────────────────────────────────
    ("play some ASMR", "asmr"),
    ("play ASMR by {asmr_artist}", "asmr"),
    ("I want to hear some ASMR", "asmr"),
    ("put on some ASMR", "asmr"),
    ("stream ASMR by {asmr_artist}", "asmr"),
    ("find me ASMR videos", "asmr"),
    ("play relaxing ASMR", "asmr"),
    ("I want to listen to ASMR", "asmr"),
    ("play ASMR content", "asmr"),
    ("stream some ASMR", "asmr"),

    # ── short_film ─────────────────────────────────────────────────────────────
    ("play the short film {short_film_title}", "short_film"),
    ("I want to watch {short_film_title}", "short_film"),
    ("stream {short_film_title}", "short_film"),
    ("show me the short {short_film_title}", "short_film"),
    ("play a short film directed by {movie_director}", "short_film"),
    ("find me {short_film_title}", "short_film"),
    ("watch the short {short_film_title}", "short_film"),
    ("put on {short_film_title}", "short_film"),

    # ── trailer ────────────────────────────────────────────────────────────────
    ("play the trailer for {trailer_title}", "trailer"),
    ("show me the trailer for {trailer_title}", "trailer"),
    ("I want to see the trailer for {trailer_title}", "trailer"),
    ("stream the trailer for {trailer_title}", "trailer"),
    ("play the {trailer_title} trailer", "trailer"),
    ("find me the trailer for {trailer_title}", "trailer"),
    ("watch the trailer for {trailer_title}", "trailer"),
    ("play a trailer with {movie_actor}", "trailer"),

    # ── music_video ────────────────────────────────────────────────────────────
    ("play the music video for {music_video_title}", "music_video"),
    ("play the music video by {artist_name}", "music_video"),
    ("show me the music video for {music_video_title}", "music_video"),
    ("I want to watch the music video {music_video_title}", "music_video"),
    ("stream the music video for {music_video_title}", "music_video"),
    ("play {artist_name}'s music video", "music_video"),
    ("find me the {music_video_title} music video", "music_video"),
    ("watch the official video for {music_video_title}", "music_video"),

    # ── tv (live) ──────────────────────────────────────────────────────────────
    ("put on {tv_channel}", "tv"),
    ("switch to {tv_channel}", "tv"),
    ("tune in to {tv_channel}", "tv"),
    ("I want to watch {tv_channel}", "tv"),
    ("turn on {tv_channel}", "tv"),
    ("stream {tv_channel}", "tv"),
    ("play live TV on {tv_channel}", "tv"),
    ("change to {tv_channel}", "tv"),
    ("watch {tv_channel}", "tv"),
    ("play {tv_channel} channel", "tv"),

    # ── video ──────────────────────────────────────────────────────────────────
    ("play a video", "video"),
    ("show me a video", "video"),
    ("I want to watch a video", "video"),
    ("play some videos", "video"),
    ("stream a video", "video"),
    ("play videos", "video"),

    # ── audio ──────────────────────────────────────────────────────────────────
    ("play some audio", "audio"),
    ("I want to listen to something", "audio"),
    ("play something", "audio"),
    ("stream some {music_genre} audio", "audio"),
    ("play some background {music_genre}", "audio"),
    ("put on some audio", "audio"),

    # ── silent_movie ───────────────────────────────────────────────────────────
    ("play the silent film {silent_movie_title}", "silent_movie"),
    ("show me {silent_movie_title}", "silent_movie"),
    ("I want to watch {silent_movie_title}", "silent_movie"),
    ("stream the silent movie {silent_movie_title}", "silent_movie"),
    ("play a silent film called {silent_movie_title}", "silent_movie"),
    ("watch {silent_movie_title}", "silent_movie"),

    # ── bw_movie ───────────────────────────────────────────────────────────────
    ("play the black and white film {bw_movie_title}", "bw_movie"),
    ("show me {bw_movie_title}", "bw_movie"),
    ("I want to watch {bw_movie_title}", "bw_movie"),
    ("stream the black and white movie {bw_movie_title}", "bw_movie"),
    ("play the classic film {bw_movie_title}", "bw_movie"),
    ("watch {bw_movie_title}", "bw_movie"),

    # ── visual_story ───────────────────────────────────────────────────────────
    ("play the visual story {visual_story_title}", "visual_story"),
    ("show me {visual_story_title}", "visual_story"),
    ("I want to watch {visual_story_title}", "visual_story"),
    ("stream {visual_story_title}", "visual_story"),

    # ── behind_the_scenes ──────────────────────────────────────────────────────
    ("play the behind the scenes for {bts_title}", "behind_the_scenes"),
    ("show me behind the scenes of {bts_title}", "behind_the_scenes"),
    ("I want to watch the making of {bts_title}", "behind_the_scenes"),
    ("stream the behind the scenes for {bts_title}", "behind_the_scenes"),
    ("play behind the scenes with {movie_actor}", "behind_the_scenes"),
    ("find behind the scenes content for {bts_title}", "behind_the_scenes"),

    # ── radio_theatre ──────────────────────────────────────────────────────────
    ("play the radio drama {radio_drama_title}", "radio_theatre"),
    ("I want to listen to {radio_drama_title}", "radio_theatre"),
    ("stream the radio play {radio_drama_title}", "radio_theatre"),
    ("play the audio drama {radio_drama_title}", "radio_theatre"),
    ("find me {radio_drama_title}", "radio_theatre"),
    ("put on the radio theatre {radio_drama_title}", "radio_theatre"),

    # ── audio_description ──────────────────────────────────────────────────────
    ("play audio description", "audio_description"),
    ("I want to listen to audio description", "audio_description"),
    ("enable audio description", "audio_description"),
    ("play audio described content", "audio_description"),

    # ── video_episodes ─────────────────────────────────────────────────────────
    ("play episodes of {tv_show_title}", "video_episodes"),
    ("show me episodes of {tv_show_title}", "video_episodes"),
    ("I want to watch episodes of {tv_show_title}", "video_episodes"),
    ("stream episodes of {tv_show_title}", "video_episodes"),
    ("play the next episode of {tv_show_title}", "video_episodes"),
    ("continue watching {tv_show_title}", "video_episodes"),
    ("play a {tv_genre} series episode", "video_episodes"),
    ("watch episodes of {tv_show_title}", "video_episodes"),
]

# ---------------------------------------------------------------------------
# Multilingual verb/phrase stubs
# These are used to build translated templates by verb-substitution.
# Keys match locale folder names.
# ---------------------------------------------------------------------------

# Per-language play/watch/listen verbs + object phrase patterns
# Format: {lang: {"play": [...], "watch": [...], "listen": [...], "stream": [...],
#                 "find": [...], "put_on": [...], "tune": [...], "start": [...]}}
_LANG_VERBS: Dict[str, Dict[str, List[str]]] = {
    "pt-pt": {
        "play":   ["reproduz", "toca", "coloca"],
        "watch":  ["quero ver", "mostrar"],
        "listen": ["quero ouvir", "ouvir"],
        "stream": ["transmite"],
        "find":   ["encontra"],
        "put_on": ["põe"],
        "tune":   ["sintoniza"],
        "start":  ["inicia", "começa"],
    },
    "pt-br": {
        "play":   ["reproduz", "toca", "coloca"],
        "watch":  ["quero ver", "mostrar"],
        "listen": ["quero ouvir", "ouvir"],
        "stream": ["transmite"],
        "find":   ["encontra"],
        "put_on": ["põe"],
        "tune":   ["sintoniza"],
        "start":  ["inicia", "começa"],
    },
    "es-es": {
        "play":   ["reproduce", "pon", "toca"],
        "watch":  ["quiero ver", "muéstrame"],
        "listen": ["quiero escuchar", "escuchar"],
        "stream": ["transmite"],
        "find":   ["encuentra"],
        "put_on": ["pon"],
        "tune":   ["sintoniza"],
        "start":  ["inicia", "comienza"],
    },
    "fr-fr": {
        "play":   ["joue", "mets", "lance"],
        "watch":  ["je veux regarder", "montre-moi"],
        "listen": ["je veux écouter", "écouter"],
        "stream": ["diffuse"],
        "find":   ["trouve"],
        "put_on": ["mets"],
        "tune":   ["syntonise"],
        "start":  ["démarre", "lance"],
    },
    "de-de": {
        "play":   ["spiele", "spiel", "starte"],
        "watch":  ["ich will sehen", "zeig mir"],
        "listen": ["ich will hören", "abspielen"],
        "stream": ["streame"],
        "find":   ["finde"],
        "put_on": ["leg auf"],
        "tune":   ["stimme ein"],
        "start":  ["starte", "öffne"],
    },
    "it-it": {
        "play":   ["riproduci", "metti", "avvia"],
        "watch":  ["voglio vedere", "mostrami"],
        "listen": ["voglio ascoltare", "ascoltare"],
        "stream": ["trasmetti"],
        "find":   ["trova"],
        "put_on": ["metti"],
        "tune":   ["sintonizza"],
        "start":  ["avvia", "inizia"],
    },
    "nl-nl": {
        "play":   ["speel", "zet op", "start"],
        "watch":  ["ik wil kijken naar", "toon me"],
        "listen": ["ik wil luisteren naar", "luisteren"],
        "stream": ["stream"],
        "find":   ["zoek"],
        "put_on": ["zet op"],
        "tune":   ["stem af op"],
        "start":  ["start", "open"],
    },
    "ca-es": {
        "play":   ["reprodueix", "posa", "inicia"],
        "watch":  ["vull veure", "mostra'm"],
        "listen": ["vull escoltar", "escoltar"],
        "stream": ["retransmet"],
        "find":   ["troba"],
        "put_on": ["posa"],
        "tune":   ["sintonitza"],
        "start":  ["inicia", "comença"],
    },
    "gl-es": {
        "play":   ["reproduce", "pon", "toca"],
        "watch":  ["quero ver", "mostrar"],
        "listen": ["quero ouvir", "escoitar"],
        "stream": ["transmite"],
        "find":   ["atopa"],
        "put_on": ["pon"],
        "tune":   ["sintoniza"],
        "start":  ["inicia", "comeza"],
    },
    "eu": {
        "play":   ["jarri", "erreproduzitu"],
        "watch":  ["ikusi nahi dut", "erakutsi"],
        "listen": ["entzun nahi dut", "entzun"],
        "stream": ["streaming egin"],
        "find":   ["bilatu"],
        "put_on": ["jarri"],
        "tune":   ["sintonizatu"],
        "start":  ["hasi", "ireki"],
    },
    "da-dk": {
        "play":   ["afspil", "sæt på", "start"],
        "watch":  ["jeg vil se", "vis mig"],
        "listen": ["jeg vil høre", "lytte til"],
        "stream": ["stream"],
        "find":   ["find"],
        "put_on": ["sæt på"],
        "tune":   ["stem ind på"],
        "start":  ["start", "åbn"],
    },
    "pl-pl": {
        "play":   ["odtwórz", "puść", "uruchom"],
        "watch":  ["chcę obejrzeć", "pokaż mi"],
        "listen": ["chcę posłuchać", "słuchać"],
        "stream": ["streamuj"],
        "find":   ["znajdź"],
        "put_on": ["włącz"],
        "tune":   ["nastaw"],
        "start":  ["uruchom", "rozpocznij"],
    },
}


def _build_translated_templates(lang: str) -> List[Tuple[str, str]]:
    """Build simplified translated templates using verb stubs for non-English languages.

    For each media type, generates a set of templates combining each verb form
    with the primary slot placeholder.

    Args:
        lang: BCP-47 language code.

    Returns:
        List of ``(template, intent)`` tuples.
    """
    verbs = _LANG_VERBS.get(lang, {})
    if not verbs:
        return []

    play_verbs = verbs.get("play", ["play"])
    watch_verbs = verbs.get("watch", ["watch"])
    listen_verbs = verbs.get("listen", ["listen"])
    stream_verbs = verbs.get("stream", ["stream"])
    start_verbs = verbs.get("start", ["start"])
    find_verbs = verbs.get("find", ["find"])
    tune_verbs = verbs.get("tune", ["tune in to"])
    put_on_verbs = verbs.get("put_on", ["put on"])

    # (verb_list, slot, intent) patterns
    patterns: List[Tuple[List[str], str, str]] = [
        (play_verbs,   "{artist_name}", "music"),
        (play_verbs,   "{track_name}", "music"),
        (listen_verbs, "{artist_name}", "music"),
        (play_verbs,   "{album_name}", "music"),
        (play_verbs,   "{movie_title}", "movie"),
        (watch_verbs,  "{movie_title}", "movie"),
        (play_verbs,   "{tv_show_title}", "tv_show"),
        (watch_verbs,  "{tv_show_title}", "tv_show"),
        (play_verbs,   "{podcast_title}", "podcast"),
        (listen_verbs, "{podcast_title}", "podcast"),
        (play_verbs,   "{audiobook_title}", "audiobook"),
        (listen_verbs, "{audiobook_title}", "audiobook"),
        (play_verbs,   "{radio_station}", "radio"),
        (tune_verbs,   "{radio_station}", "radio"),
        (play_verbs,   "{game_title}", "game"),
        (start_verbs,  "{game_title}", "game"),
        (play_verbs,   "{anime_title}", "anime"),
        (watch_verbs,  "{anime_title}", "anime"),
        (play_verbs,   "{documentary_title}", "documentary"),
        (watch_verbs,  "{documentary_title}", "documentary"),
        (play_verbs,   "{cartoon_title}", "cartoon"),
        (watch_verbs,  "{cartoon_title}", "cartoon"),
        (play_verbs,   "{tv_channel}", "tv"),
        (tune_verbs,   "{tv_channel}", "tv"),
        (play_verbs,   "{news_provider}", "news"),
        (listen_verbs, "{news_provider}", "news"),
        (play_verbs,   "{music_video_title}", "music_video"),
        (watch_verbs,  "{music_video_title}", "music_video"),
        (play_verbs,   "{radio_drama_title}", "radio_theatre"),
        (listen_verbs, "{radio_drama_title}", "radio_theatre"),
        (stream_verbs, "{tv_show_title}", "video_episodes"),
        (watch_verbs,  "{tv_show_title}", "video_episodes"),
        (play_verbs,   "{bts_title}", "behind_the_scenes"),
        (watch_verbs,  "{bts_title}", "behind_the_scenes"),
        (play_verbs,   "{silent_movie_title}", "silent_movie"),
        (watch_verbs,  "{silent_movie_title}", "silent_movie"),
        (play_verbs,   "{bw_movie_title}", "bw_movie"),
        (watch_verbs,  "{bw_movie_title}", "bw_movie"),
        (play_verbs,   "{visual_story_title}", "visual_story"),
        (find_verbs,   "{game_title}", "game"),
        (stream_verbs, "{movie_title}", "movie"),
        (stream_verbs, "{anime_title}", "anime"),
    ]

    result: List[Tuple[str, str]] = []
    for verb_list, slot, intent in patterns:
        for verb in verb_list:
            result.append((f"{verb} {slot}", intent))

    return result


def _extract_slots(template: str) -> str:
    """Return comma-separated slot names present in a template string."""
    slots = re.findall(r"\{(\w+)\}", template)
    return ",".join(slots)


def get_templates_dir(output_dir: Optional[str] = None) -> str:
    """Return the default templates output directory."""
    return output_dir or os.path.join(get_cache_dir(), "templates_new")


def generate_all(
    langs: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, int]:
    """Generate template CSVs for all requested languages.

    Args:
        langs: Language codes to generate (default: all 13 supported languages).
        output_dir: Root output directory (default: ``~/.cache/ovos-media-classifier/templates_new/``).

    Returns:
        ``{lang: row_count}`` summary dict.
    """
    if langs is None:
        langs = ALL_LANGS
    out_root = get_templates_dir(output_dir)
    summary: Dict[str, int] = {}

    for lang in langs:
        if lang == "en-us":
            templates = _EN_TEMPLATES
        else:
            templates = _build_translated_templates(lang)
            if not templates:
                print(f"  {lang}: no verb stubs defined, skipping")
                continue

        lang_dir = os.path.join(out_root, lang)
        os.makedirs(lang_dir, exist_ok=True)

        # Group by intent
        by_intent: Dict[str, List[str]] = {}
        for template, intent in templates:
            by_intent.setdefault(intent, []).append(template)

        total = 0
        for intent, tmpl_list in by_intent.items():
            path = os.path.join(lang_dir, f"{intent}.csv")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["lang", "intent", "template", "slots"])
                writer.writeheader()
                for tmpl in tmpl_list:
                    writer.writerow({
                        "lang":     lang,
                        "intent":   intent,
                        "template": tmpl,
                        "slots":    _extract_slots(tmpl),
                    })
            total += len(tmpl_list)

        # Also write a combined CSV for this language
        combined_path = os.path.join(lang_dir, "_all.csv")
        with open(combined_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["lang", "intent", "template", "slots"])
            writer.writeheader()
            for template, intent in templates:
                writer.writerow({
                    "lang":     lang,
                    "intent":   intent,
                    "template": template,
                    "slots":    _extract_slots(template),
                })

        print(f"  {lang}: {total} templates across {len(by_intent)} intents → {lang_dir}/")
        summary[lang] = total

    return summary


def load_templates(templates_dir: str,
                   langs: Optional[List[str]] = None) -> "pd.DataFrame":
    """Load all template CSVs from a directory into a single DataFrame.

    Args:
        templates_dir: Root directory produced by ``generate_all()``.
        langs: Filter to specific language codes (default: all subdirectories).

    Returns:
        DataFrame with columns ``lang, intent, template, slots``.
    """
    import pandas as pd
    frames = []
    for entry in os.scandir(templates_dir):
        if not entry.is_dir():
            continue
        if langs and entry.name not in langs:
            continue
        combined = os.path.join(entry.path, "_all.csv")
        if os.path.exists(combined):
            frames.append(pd.read_csv(combined))
    if not frames:
        return pd.DataFrame(columns=["lang", "intent", "template", "slots"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for generate_templates pipeline step."""
    parser = argparse.ArgumentParser(
        description="Generate per-language sentence template CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--langs", nargs="*", default=None, metavar="LANG",
                        help=f"Language codes (default: all). Available: {ALL_LANGS}")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: ~/.cache/ovos-media-classifier/templates_new/)")
    args = parser.parse_args()

    print("Generating templates …")
    summary = generate_all(langs=args.langs, output_dir=args.output)
    total = sum(summary.values())
    print(f"Done: {total} templates across {len(summary)} languages.")


if __name__ == "__main__":
    main()
