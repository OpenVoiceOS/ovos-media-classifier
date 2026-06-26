#!/usr/bin/env python3
"""Author the OVOS-INTENT-1 ``.intent`` / ``.voc`` media templates.

The training set is generated from **translatable template files** rather than a
hard-coded python list, so the user can manage and translate them through
ovos-localize.  Two artefact kinds are written under ``training/templates/``:

``vocab/<lang>/<VocName>.voc``
    Shared lead-in vocabularies (request openers grouped by playback family and
    register): ``lead_play_audio``, ``lead_watch``, ``LeadPolite`` …  Each line is a
    member phrase.  These are the componential *lead-ins*.

``<lang>/<intent>.intent``
    One file per media label (``music``, ``movie``, ``adult`` …).  Each line is
    an OVOS-INTENT-1 template that may use:

    * ``<VocName>``  — expands to every member of that ``.voc`` (the lead-ins);
    * ``(a|b|c)``    — inline alternation (the slot-pattern variants);
    * ``[word]``     — optional word;
    * ``{slot}``     — an opaque entity slot (an ``OCPEntityLabel`` name) left
      for the slot-filler.

``ovos_spec_tools.expand(template, vocabularies)`` turns one ``.intent`` line
into its full sample set; ``build_dataset.py`` then fills the ``{slot}``
placeholders from the real entity pools.

The lead-in × pattern structure mirrors the old componential generator: the
``<Lead*>`` references are the request-kind lead-ins and the inline
``(…|…)`` alternations are the slot-pattern variants.  To add or translate
templates, edit these files (or add a new ``<lang>/`` directory) — no code
change is needed.

Usage::

    python -m training.author_templates                 # (re)write en-us
    python -m training.author_templates --langs en-us pt-pt es-es
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(_HERE, "templates")

# ---------------------------------------------------------------------------
# Shared lead-in vocabularies, per language, grouped by playback family + kind.
# Referenced from the .intent files as <lead_play_audio> etc.
# ---------------------------------------------------------------------------
LEADINS: Dict[str, Dict[str, List[str]]] = {
    "en-us": {
        "lead_play_audio": [
            # bare imperatives
            "play", "put on", "start", "start playing", "stream", "queue up",
            "throw on", "spin up", "pull up", "fire up", "cue", "cue up",
            "play me", "play some", "give me", "find me", "search for",
            # first-person desire / mood
            "i want to listen to", "i want to hear", "i wanna hear",
            "i'd like to hear", "i'd like to listen to", "i feel like listening to",
            "i'm in the mood for", "i feel like", "let's hear", "let's listen to",
            "gimme some", "i'd like",
            # polite / question openers
            "can you play", "could you play", "please play", "can you put on",
            "could you put on", "can you find", "do you have", "got any",
            "how about",
        ],
        "lead_watch": [
            "watch", "play", "put on", "show me", "stream", "pull up",
            "throw on", "fire up", "start watching", "queue up",
            "i want to watch", "i wanna watch", "i'd like to watch",
            "i feel like watching", "i'm in the mood for", "i feel like",
            "let's watch", "can we watch", "could you play", "can you put on",
            "please put on", "how about", "how about we watch", "got any",
            "do you have", "find me", "show me",
        ],
        "lead_read": [
            "read", "read me", "read aloud", "read to me", "open",
            "i want to read", "i wanna read", "i'd like to read",
            "i feel like reading", "i'm in the mood for", "let's read",
            "can you read", "could you read", "please read", "can you open",
            "got any", "do you have", "find me",
        ],
        "lead_game": [
            "play", "launch", "open", "start", "boot up", "fire up", "load up",
            "start a game of", "let's play", "i want to play", "i wanna play",
            "i'd like to play", "can we play", "could you launch", "please start",
            "i feel like playing", "i'm in the mood for", "how about", "got any",
            "do you have",
        ],
        "lead_tune": [
            "tune in to", "put on", "switch to", "turn on", "play", "pull up",
            "i want to listen to", "i'd like to listen to", "i feel like listening to",
            "stream", "let's listen to", "can you put on", "could you tune in to",
            "please put on", "how about",
        ],
    },
    "pt-pt": {
        "lead_play_audio": ["reproduz", "toca", "coloca", "põe", "quero ouvir",
                          "gostava de ouvir", "apetece-me ouvir", "podes tocar",
                          "por favor toca", "estou com vontade de ouvir"],
        "lead_watch": ["vê", "reproduz", "mostra-me", "quero ver", "gostava de ver",
                      "apetece-me ver", "vamos ver", "podes pôr", "põe"],
        "lead_read": ["lê", "lê-me", "reproduz o audiolivro", "quero ouvir o audiolivro",
                     "podes ler"],
        "lead_game": ["joga", "abre", "inicia", "começa", "quero jogar", "vamos jogar"],
        "lead_tune": ["sintoniza", "põe", "muda para", "liga", "quero ouvir"],
    },
    "es-es": {
        "lead_play_audio": ["reproduce", "pon", "toca", "quiero escuchar",
                          "me apetece escuchar", "puedes poner", "por favor pon",
                          "estoy de humor para"],
        "lead_watch": ["ve", "reproduce", "muéstrame", "quiero ver", "me apetece ver",
                      "vamos a ver", "puedes poner", "pon"],
        "lead_read": ["lee", "léeme", "reproduce el audiolibro", "quiero escuchar el audiolibro"],
        "lead_game": ["juega", "abre", "inicia", "quiero jugar", "vamos a jugar"],
        "lead_tune": ["sintoniza", "pon", "cambia a", "enciende", "quiero escuchar"],
    },
    "fr-fr": {
        "lead_play_audio": ["joue", "mets", "lance", "je veux écouter",
                          "j'ai envie d'écouter", "peux-tu jouer", "s'il te plaît joue"],
        "lead_watch": ["regarde", "joue", "montre-moi", "je veux regarder",
                      "j'ai envie de regarder", "on regarde", "mets"],
        "lead_read": ["lis", "lis-moi", "joue le livre audio", "je veux écouter le livre audio"],
        "lead_game": ["joue", "lance", "ouvre", "démarre", "je veux jouer", "on joue à"],
        "lead_tune": ["syntonise", "mets", "change pour", "allume", "je veux écouter"],
    },
    "de-de": {
        "lead_play_audio": ["spiele", "spiel", "starte", "leg auf", "ich will hören",
                          "ich möchte hören", "kannst du spielen", "bitte spiele"],
        "lead_watch": ["schau", "spiele", "zeig mir", "ich will sehen",
                      "ich möchte sehen", "lass uns sehen", "leg auf"],
        "lead_read": ["lies", "lies mir", "spiele das hörbuch", "ich will das hörbuch hören"],
        "lead_game": ["spiele", "starte", "öffne", "ich will spielen", "lass uns spielen"],
        "lead_tune": ["schalte ein", "leg auf", "wechsle zu", "ich will hören"],
    },
    "it-it": {
        "lead_play_audio": ["riproduci", "metti", "avvia", "voglio ascoltare",
                          "ho voglia di ascoltare", "puoi mettere", "per favore metti"],
        "lead_watch": ["guarda", "riproduci", "mostrami", "voglio vedere",
                      "ho voglia di vedere", "guardiamo", "metti"],
        "lead_read": ["leggi", "leggimi", "riproduci l'audiolibro", "voglio ascoltare l'audiolibro"],
        "lead_game": ["gioca", "avvia", "apri", "voglio giocare", "giochiamo a"],
        "lead_tune": ["sintonizza", "metti", "cambia su", "accendi", "voglio ascoltare"],
    },
    "nl-nl": {
        "lead_play_audio": ["speel", "zet op", "start", "ik wil luisteren naar",
                          "ik heb zin in", "kun je spelen", "speel alsjeblieft"],
        "lead_watch": ["kijk", "speel", "toon me", "ik wil kijken naar",
                      "ik heb zin om te kijken", "laten we kijken", "zet op"],
        "lead_read": ["lees", "lees me voor", "speel het luisterboek", "ik wil het luisterboek horen"],
        "lead_game": ["speel", "start", "open", "ik wil spelen", "laten we spelen"],
        "lead_tune": ["stem af op", "zet op", "schakel over naar", "ik wil luisteren naar"],
    },
}

# Optional decorations as inline [..]/(..) so expansion stays in the grammar.
# en-us only — translated decorations would be added per locale.
_DECOR = {
    "en-us": {"pre": "[hey|ok|please]", "post": "[please|for me|right now|thanks]"},
}


# ---------------------------------------------------------------------------
# Per-intent template bodies (the variable middle).  These reuse <Lead*> refs
# for the request opener and (a|b) alternations for the slot-pattern variants.
# Bodies are language-independent in *structure*; only the lead-ins translate,
# so the same bodies are emitted for every language with that language's
# <Lead*> vocab.  (Slot names and a few English function words remain; locales
# refine them by editing the generated files.)
# ---------------------------------------------------------------------------
# Each entry: intent -> (lead_ref, [body templates])
BODIES: Dict[str, tuple] = {
    # MUSIC — the richest entity set ({artist_name}, {track_name}=song,
    # {album_name}, {music_genre}, {record_label}, {release_decade}).  Authored
    # across a deliberate 1→5 filled-slot range with many phrasings per combo.
    "music": ("lead_play_audio", [
        # ── 1 slot ──────────────────────────────────────────────────────────
        "{artist_name}", "some {artist_name}", "{artist_name}'s music",
        "music by {artist_name}", "songs by {artist_name}",
        "the best of {artist_name}", "anything by {artist_name}",
        "more {artist_name}", "{artist_name} radio", "the {artist_name} station",
        "the latest from {artist_name}", "{artist_name}'s newest",
        "{artist_name}'s greatest hits", "a song by {artist_name}",
        "{track_name}", "the song {track_name}", "the track {track_name}",
        "that song {track_name}", "{album_name}", "the album {album_name}",
        "songs from {album_name}", "the record {album_name}",
        "some {music_genre}", "some {music_genre} music", "a {music_genre} playlist",
        "a {music_genre} mix", "a {music_genre} song", "the {music_genre} station",
        "{record_label} artists", "music on {record_label}",
        # ── 2 slots ─────────────────────────────────────────────────────────
        "{track_name} by {artist_name}", "{artist_name}'s {track_name}",
        "the song {track_name} by {artist_name}",
        "{album_name} by {artist_name}", "{artist_name}'s {album_name}",
        "the album {album_name} by {artist_name}",
        "songs from {album_name} by {artist_name}",
        "a song featuring {artist_name}", "{track_name} featuring {artist_name}",
        "some {music_genre} by {artist_name}", "{artist_name}'s {music_genre}",
        "a {music_genre} song by {artist_name}", "{music_genre} from {artist_name}",
        "{album_name} on {record_label}", "{artist_name} on {record_label}",
        "the {music_genre} one called {track_name}",
        "some {music_genre} from {album_name}",
        "{music_genre} from the {release_decade}",
        "some {artist_name} from the {release_decade}",
        # ── 3 slots ─────────────────────────────────────────────────────────
        "{track_name} by {artist_name} from {album_name}",
        "{artist_name}'s {music_genre} song {track_name}",
        "{track_name} off {album_name} by {artist_name}",
        "the {music_genre} track {track_name} by {artist_name}",
        "{album_name} by {artist_name} on {record_label}",
        "some {music_genre} by {artist_name} from {album_name}",
        "{artist_name}'s {music_genre} from the {release_decade}",
        "the {music_genre} song {track_name} from the {release_decade}",
        # ── 4 slots ─────────────────────────────────────────────────────────
        "the {music_genre} track {track_name} by {artist_name} off {album_name}",
        "{track_name} by {artist_name} from {album_name} on {record_label}",
        "some {music_genre} by {artist_name} from {album_name} on {record_label}",
        "the {music_genre} song {track_name} by {artist_name} from the {release_decade}",
        # ── 5 slots ─────────────────────────────────────────────────────────
        "{track_name} by {artist_name} from {album_name}, a {music_genre} record from the {release_decade}",
        "the {music_genre} track {track_name} by {artist_name} off {album_name} on {record_label}",
        # ── question / mood / contextual forms ──────────────────────────────
        "do you have any {artist_name}", "got any {music_genre}",
        "do you have any {music_genre} by {artist_name}",
        "something {music_genre} for my workout",
        "some {music_genre} to help me focus",
        "a {music_genre} mix while i cook",
        "something by {artist_name} to relax to",
    ]),
    # MOVIE — full crew-role richness across a 1→5 filled-slot range.
    "movie": ("lead_watch", [
        # ── 1 slot ──
        "{movie_title}", "the movie {movie_title}", "the film {movie_title}",
        "a {movie_genre} movie", "a {movie_genre} film", "some {movie_genre} movie",
        "an old {movie_genre} movie", "a classic {movie_genre} movie",
        "a good {movie_genre} film", "something with {movie_actor}",
        "a movie with {movie_actor}", "a film starring {movie_actor}",
        "anything with {movie_actor}", "a movie directed by {movie_director}",
        "a film by {movie_director}", "a movie produced by {movie_producer}",
        "a movie written by {movie_writer}", "a film written by {movie_writer}",
        "a movie scored by {movie_composer}", "a film with music by {movie_composer}",
        # ── 2 slots ──
        "{movie_title} with {movie_actor}", "{movie_title} starring {movie_actor}",
        "{movie_title} directed by {movie_director}",
        "the film {movie_title} by {movie_director}",
        "a {movie_genre} film with {movie_actor}",
        "a {movie_genre} movie directed by {movie_director}",
        "a {movie_genre} film by {movie_director}",
        "a movie with {movie_actor} directed by {movie_director}",
        "a film starring {movie_actor} and {movie_director}",
        "a {movie_genre} movie from {release_year}",
        "a {movie_genre} film from {media_country}",
        # ── 3 slots ──
        "a {movie_genre} film directed by {movie_director} starring {movie_actor}",
        "{movie_title} directed by {movie_director} starring {movie_actor}",
        "a {movie_genre} movie with {movie_actor} from {release_year}",
        "a movie directed by {movie_director} written by {movie_writer} starring {movie_actor}",
        "a {movie_genre} film from {media_country} starring {movie_actor}",
        # ── 4 slots ──
        "{movie_title} directed by {movie_director} starring {movie_actor}, a {movie_genre} film",
        "a {movie_genre} film directed by {movie_director} starring {movie_actor} from {release_year}",
        # ── 5 slots ──
        "{movie_title} directed by {movie_director} starring {movie_actor}, a {movie_genre} film from {release_year}",
        # ── question / mood / contextual ──
        "do you have any {movie_genre} movies", "got a good {movie_genre} film",
        "do you have anything with {movie_actor}",
        "something {movie_genre} for movie night",
        "a {movie_genre} movie to watch with the kids",
    ]),
    "tv_show": ("lead_watch", [
        # ── 1 slot ──
        "{tv_show_title}", "the show {tv_show_title}", "the series {tv_show_title}",
        "an episode of {tv_show_title}", "a {tv_genre} show", "a {tv_genre} series",
        "the new season of {tv_show_title}", "the latest episode of {tv_show_title}",
        "something like {tv_show_title}", "a good {tv_genre} show",
        "a series starring {movie_actor}", "a show on {tv_network}",
        # ── 2 slots ──
        "{tv_show_title} starring {movie_actor}", "{tv_show_title} on {tv_network}",
        "a {tv_genre} show on {tv_network}", "the {tv_network} series {tv_show_title}",
        "a {tv_genre} series starring {movie_actor}",
        "a {tv_genre} show from {release_year}",
        "a {tv_genre} series from {media_country}",
        # ── 3 slots ──
        "a {tv_genre} show on {tv_network} starring {movie_actor}",
        "{tv_show_title} on {tv_network} starring {movie_actor}",
        "a {tv_genre} series from {release_year} on {tv_network}",
        # ── 4 slots ──
        "a {tv_genre} series on {tv_network} starring {movie_actor} from {release_year}",
        "{tv_show_title} on {tv_network} starring {movie_actor}, a {tv_genre} show",
        # ── 5 slots ──
        "{tv_show_title}, a {tv_genre} series on {tv_network} starring {movie_actor} from {release_year}",
        # ── question / mood / contextual ──
        "do you have any {tv_genre} shows", "got a good {tv_genre} series",
        "something {tv_genre} to binge", "a {tv_genre} show to fall asleep to",
    ]),
    "podcast": ("lead_play_audio", [
        # ── 1 slot ──
        "{podcast_title}", "the {podcast_title} podcast",
        "an episode of {podcast_title}", "the latest {podcast_title} episode",
        "a {podcast_genre} podcast", "the podcast with {podcast_host}",
        "{podcast_host}'s podcast", "a podcast about {podcast_genre}",
        "the podcast hosted by {podcast_host}", "a good {podcast_genre} podcast",
        # ── 2 slots ──
        "{podcast_title} hosted by {podcast_host}",
        "a {podcast_genre} podcast with {podcast_host}",
        "a {podcast_genre} podcast hosted by {podcast_host}",
        "the {podcast_title} episode with {podcast_host}",
        # ── 3 slots ──
        "a {podcast_genre} podcast called {podcast_title} with {podcast_host}",
        # ── question / mood / contextual ──
        "do you have any {podcast_genre} podcasts", "got a good {podcast_genre} podcast",
        "a {podcast_genre} podcast for my commute",
    ]),
    "radio": ("lead_tune", [
        "{radio_station}", "the radio station {radio_station}",
        "{radio_station} radio", "some {radio_genre} radio",
        "a {radio_genre} station", "live radio", "the {radio_genre} station",
        "the {radio_station} stream", "another {radio_genre} station",
        "some live radio", "a local radio station", "{radio_station} fm",
        "the fm station {radio_station}", "an internet radio station",
        "a {radio_genre} internet station", "a {radio_genre} fm station",
        "talk radio", "the {radio_genre} talk station", "news radio",
        "a music station", "a {radio_genre} music station",
        "my favorite radio station", "the morning show on {radio_station}",
        # 2 slots
        "the {radio_genre} station {radio_station}",
        "{radio_station}, a {radio_genre} station",
        "{radio_station} the {radio_genre} station",
        # question / mood / contextual
        "do you have any {radio_genre} radio", "got any {radio_genre} stations",
        "some {radio_genre} radio for the drive",
        "a {radio_genre} station to wake up to",
        "what {radio_genre} stations are there",
        "a {radio_genre} station for the background",
        "some live radio while i work", "a {radio_genre} station for my commute",
    ]),
    # AUDIOBOOK = play a NARRATION (audio).  Phrasing is audiobook-specific so it
    # stays distinct from BOOK (TTS-read text) below.
    "audiobook": ("lead_play_audio", [
        # ── 1 slot ──
        "the audiobook {audiobook_title}", "the audiobook of {audiobook_title}",
        "an audiobook by {audiobook_author}",
        "an audiobook narrated by {audiobook_narrator}",
        "a {audiobook_genre} audiobook", "the audiobook version of {audiobook_title}",
        "anything by {audiobook_author}", "a good {audiobook_genre} audiobook",
        # ── 2 slots ──
        "the audiobook {audiobook_title} by {audiobook_author}",
        "the audiobook {audiobook_title} narrated by {audiobook_narrator}",
        "an audiobook by {audiobook_author} narrated by {audiobook_narrator}",
        "a {audiobook_genre} audiobook by {audiobook_author}",
        "a {audiobook_genre} audiobook narrated by {audiobook_narrator}",
        # ── 3 slots ──
        "the audiobook {audiobook_title} by {audiobook_author} narrated by {audiobook_narrator}",
        "a {audiobook_genre} audiobook by {audiobook_author} narrated by {audiobook_narrator}",
        # ── question / mood / contextual ──
        "do you have any {audiobook_genre} audiobooks",
        "an audiobook to fall asleep to", "a {audiobook_genre} audiobook for the trip",
    ]),
    # BOOK = TTS-read a readable text.  The "read" lead-in + book cues route this
    # to MediaType.BOOK (paged), distinct from the audiobook narration above.
    "book": ("lead_read", [
        # ── 1 slot ──
        "{book_title}", "the book {book_title}", "the novel {book_title}",
        "a book by {book_author}", "anything by {book_author}",
        "a {book_genre} book", "a {book_genre} novel", "a book about {book_genre}",
        "a good {book_genre} book", "some {book_genre}",
        # ── 2 slots ──
        "{book_title} by {book_author}", "the novel {book_title} by {book_author}",
        "the book {book_title} by {book_author}",
        "the {book_genre} book {book_title}", "a {book_genre} book by {book_author}",
        "a {book_genre} novel by {book_author}",
        "the latest book by {book_author}", "another {book_genre} book",
        "a chapter of {book_title}", "the next chapter of {book_title}",
        "more of {book_title}", "a short story by {book_author}",
        # ── 3 slots ──
        "the {book_genre} book {book_title} by {book_author}",
        "the {book_genre} novel {book_title} by {book_author}",
        # ── question / mood / contextual ──
        "do you have any {book_genre} books", "got a good {book_genre} novel",
        "a {book_genre} book to read at bedtime",
        "do you have anything by {book_author}", "got a {book_genre} book",
        "a {book_genre} novel for the weekend", "something by {book_author} to relax",
    ]),
    "news": ("lead_play_audio", [
        "the news", "today's news", "the latest news", "the morning news",
        "a news briefing", "the headlines",
        "the news from {news_provider}", "{news_provider} news",
        "the {news_category} news", "{news_category} headlines",
        "the latest {news_category} news", "an update on {news_category}",
        "some {news_category} news",
        # 2 slots
        "the {news_category} headlines from {news_provider}",
        "the {news_category} news from {news_provider}",
        # question / contextual
        "what's the latest {news_category} news", "any {news_category} news",
        "the {news_category} headlines for my morning",
    ]),
    "anime": ("lead_watch", [
        # ── 1 slot ──
        "{anime_title}", "the anime {anime_title}", "an episode of {anime_title}",
        "some anime", "anime by {anime_studio}", "a {anime_studio} anime",
        "the new season of {anime_title}", "the next episode of {anime_title}",
        "a {anime_genre} anime", "some {anime_genre} anime", "a good anime",
        # ── 2 slots ──
        "the {anime_studio} anime {anime_title}", "{anime_title} by {anime_studio}",
        "a {anime_genre} anime by {anime_studio}",
        "a {anime_genre} anime from {anime_studio}",
        "{anime_title} from {anime_studio}",
        # ── 3 slots ──
        "the {anime_genre} anime {anime_title} by {anime_studio}",
        # ── question / mood / contextual ──
        "do you have any {anime_genre} anime", "got a good {anime_genre} anime",
        "some {anime_genre} anime to binge",
    ]),
    "cartoon": ("lead_watch", [
        "{cartoon_title}", "the cartoon {cartoon_title}", "some cartoons",
        "an episode of {cartoon_title}", "a kids cartoon", "the show {cartoon_title}",
        "a saturday morning cartoon", "an old cartoon", "the next episode of {cartoon_title}",
        "a classic cartoon", "a funny cartoon", "an animated show",
        "the animated series {cartoon_title}", "another episode of {cartoon_title}",
        "a cartoon show", "some classic cartoons", "a vintage cartoon",
        "the new {cartoon_title} episode", "a kids animated show",
        "the {cartoon_title} cartoon", "a cartoon for children",
        "an episode of that cartoon", "more {cartoon_title}",
        # question / mood / contextual
        "do you have any cartoons", "got any {cartoon_title} episodes",
        "a cartoon for the kids", "something animated for saturday morning",
        "a cartoon to keep the kids busy", "got a funny cartoon",
    ]),
    "documentary": ("lead_watch", [
        "{documentary_title}", "the documentary {documentary_title}",
        "a documentary", "a nature documentary", "some documentaries",
        "the {documentary_title} documentary", "a good documentary",
        "a history documentary", "a science documentary", "a true crime documentary",
        "a {movie_genre} documentary", "a documentary about {documentary_title}",
        # question / mood / contextual
        "do you have any documentaries", "got a good documentary",
        "a documentary to fall asleep to", "something educational to watch",
    ]),
    "short_film": ("lead_watch", [
        "{short_film_title}", "the short film {short_film_title}", "a short film",
        "some short films", "the short {short_film_title}",
        "an award-winning short film", "an animated short", "a {movie_genre} short film",
        "a short {movie_genre} film", "a quick short film", "a short movie",
        "an oscar-nominated short film", "a live action short",
        "the short film called {short_film_title}", "a festival short film",
        "an indie short film", "a student short film", "a {movie_genre} short",
        "another short film", "a short film by an indie director",
        "the short {movie_genre} film {short_film_title}",
        # question / mood / contextual
        "do you have any short films", "got a good short film",
        "a short film for a quick break", "something short to watch",
        "a short film while i wait", "got any {movie_genre} short films",
    ]),
    "silent_movie": ("lead_watch", [
        "{silent_movie_title}", "the silent movie {silent_movie_title}",
        "a silent film", "an old silent movie",
        "the silent film {silent_movie_title}", "a classic silent movie",
        "a {movie_genre} silent film", "an early silent film",
        "a silent comedy", "a silent era classic", "a silent picture",
        "an old silent picture", "the silent picture {silent_movie_title}",
        "a {movie_genre} silent movie", "a silent movie classic",
        "an early cinema classic", "a silent drama",
        "the classic silent film {silent_movie_title}", "another silent movie",
        "a silent film from the early days", "a vintage silent movie",
        # question / mood / contextual
        "do you have any silent films", "got a good silent movie",
        "a silent classic for the evening", "something from the silent era",
        "a silent comedy for tonight", "got any {movie_genre} silent films",
    ]),
    "bw_movie": ("lead_watch", [
        "{bw_movie_title}", "the black and white movie {bw_movie_title}",
        "a black and white film", "an old black and white movie",
        "the black and white film {bw_movie_title}",
        "a {movie_genre} black and white movie", "a classic black and white film",
        "an old monochrome movie", "a vintage black and white film",
        "a black and white classic", "a monochrome film",
        "the b&w movie {bw_movie_title}", "an old b&w film",
        "a {movie_genre} monochrome film", "a black and white drama",
        "a black and white {movie_genre} film", "an old hollywood black and white film",
        "another black and white movie", "a noir black and white film",
        "the classic black and white movie {bw_movie_title}",
        "a black and white movie from the old days",
        # question / mood / contextual
        "do you have any black and white films", "got a good black and white movie",
        "a black and white classic for tonight", "something old in black and white",
        "a black and white film for a cozy night", "got any {movie_genre} black and white films",
    ]),
    "game": ("lead_game", [
        # ── 1 slot ──
        "{game_title}", "the game {game_title}", "a {game_genre} game",
        "a game on {game_platform}", "a round of {game_title}",
        "a quick {game_genre} game", "some {game_genre}", "another {game_genre} game",
        "a good {game_genre} game", "a game like {game_title}",
        # ── 2 slots ──
        "the {game_genre} game {game_title}", "a {game_genre} game on {game_platform}",
        "{game_title} on {game_platform}", "{game_genre} on {game_platform}",
        "a {game_genre} game from {release_year}",
        "{game_title} from the {release_decade}", "a retro {game_genre} game",
        "an indie {game_genre} game", "a multiplayer {game_genre} game",
        "another round of {game_title}", "a co-op {game_genre} game",
        # ── 3 slots ──
        "the {game_genre} game {game_title} on {game_platform}",
        "a {game_genre} game on {game_platform} from {release_year}",
        # ── question / mood / contextual ──
        "do you have any {game_genre} games", "got a good {game_genre} game",
        "a {game_genre} game to unwind with", "a quick game before bed",
        "do you have {game_title}", "got any {game_genre} games on {game_platform}",
        "a {game_genre} game for the evening", "something quick to play",
    ]),
    "music_video": ("lead_watch", [
        "the music video for {music_video_title}", "{music_video_title}",
        "the {artist_name} music video", "a music video by {artist_name}",
        "the official video for {music_video_title}", "{artist_name}'s music video",
        "the {music_video_title} video", "a music video", "some music videos",
        "the latest {artist_name} music video", "another {artist_name} music video",
        "a {artist_name} video", "the official {music_video_title} video",
        "the new music video by {artist_name}", "some {artist_name} music videos",
        "a recent music video", "the {music_video_title} official video",
        "the music video {music_video_title}",
        # 2 slots
        "the music video for {music_video_title} by {artist_name}",
        "{artist_name}'s music video for {music_video_title}",
        "the official video for {music_video_title} by {artist_name}",
        # question / mood / contextual
        "do you have the music video for {music_video_title}",
        "got any {artist_name} music videos", "some music videos for the party",
        "do you have any music videos by {artist_name}",
        "a music video to vibe to", "got the video for {music_video_title}",
    ]),
    "trailer": ("lead_watch", [
        "the trailer for {trailer_title}", "{trailer_title} trailer",
        "a movie trailer", "the latest trailers", "the {trailer_title} teaser",
        "the official trailer for {trailer_title}", "a new trailer",
        "the teaser for {trailer_title}", "some movie trailers",
        "the {trailer_title} trailer", "the new {trailer_title} trailer",
        "a film trailer", "the {trailer_title} preview", "a game trailer",
        "the official teaser for {trailer_title}", "a tv show trailer",
        "the final trailer for {trailer_title}", "an upcoming movie trailer",
        "the trailer of {trailer_title}", "a {trailer_title} sneak peek",
        # question / mood / contextual
        "do you have the trailer for {trailer_title}",
        "got any new trailers", "a trailer to check out", "what trailers are new",
        "do you have the {trailer_title} trailer", "got the teaser for {trailer_title}",
        "some trailers to watch",
    ]),
    "behind_the_scenes": ("lead_watch", [
        "behind the scenes of {bts_title}", "the making of {bts_title}",
        "{bts_title} behind the scenes", "some behind the scenes footage",
        "the {bts_title} featurette", "the making-of for {bts_title}",
        "a behind the scenes video", "the bonus features for {bts_title}",
        "the {bts_title} bloopers", "some bonus footage",
        "the gag reel for {bts_title}", "a making-of documentary",
        "the behind the scenes of {bts_title}", "the {bts_title} extras",
        "some deleted scenes from {bts_title}", "the {bts_title} commentary",
        "an on-set featurette", "the {bts_title} making of",
        "behind the scenes footage of {bts_title}", "the {bts_title} bonus content",
        # question / mood / contextual
        "do you have the making of {bts_title}",
        "got any behind the scenes for {bts_title}",
        "some behind the scenes to watch",
        "do you have the bloopers for {bts_title}",
        "got the featurette for {bts_title}", "some bonus footage to watch",
    ]),
    "radio_theatre": ("lead_play_audio", [
        "the radio drama {radio_drama_title}", "{radio_drama_title}",
        "an audio drama", "a radio play", "some radio theatre",
        "the audio drama {radio_drama_title}", "a radio drama",
        "an old time radio show", "the radio play {radio_drama_title}",
        "a {radio_genre} radio drama", "some old time radio",
        # question / mood / contextual
        "do you have any radio dramas", "got a good audio drama",
        "an audio drama for the evening", "a radio play to wind down to",
    ]),
    "asmr": ("lead_play_audio", [
        "some asmr", "asmr", "relaxing asmr", "asmr by {asmr_artist}",
        "{asmr_artist}'s asmr", "asmr to fall asleep to",
        "some {asmr_artist} asmr", "calming asmr", "an asmr video",
        "whispered asmr", "tingly asmr", "an asmr session",
        "some {asmr_artist} whispering", "soft spoken asmr",
        # question / mood / contextual
        "do you have any asmr", "got any {asmr_artist} asmr",
        "some asmr to help me sleep", "relaxing asmr to wind down",
    ]),
    "audio_description": ("lead_watch", [
        "{movie_title} with audio description",
        "the audio described version of {movie_title}",
        "a movie with audio description", "the described version of {movie_title}",
        "a described {movie_genre} movie", "{movie_title} described",
        "a {movie_genre} movie with audio description",
        "an audio described film", "the described {movie_title}",
        "{movie_title} with narration", "an accessible version of {movie_title}",
        "the narrated version of {movie_title}", "a film with audio description",
        "a {movie_genre} film with narration", "{movie_title} for the blind",
        "an audio described {movie_genre} film", "the described film {movie_title}",
        "a movie with descriptive narration", "an accessible {movie_genre} movie",
        # 2 slots
        "a {movie_genre} film with audio description starring {movie_actor}",
        "{movie_title} with audio description starring {movie_actor}",
        # question / mood / contextual
        "do you have {movie_title} with audio description",
        "got any audio described movies", "a described movie for movie night",
        "an audio described film for the evening",
        "do you have an accessible version of {movie_title}",
        "got any {movie_genre} movies with audio description",
    ]),
    "tv": ("lead_watch", [
        "{tv_channel}", "the channel {tv_channel}", "live tv", "{tv_channel} live",
        "whatever's on {tv_channel}", "a {tv_genre} channel", "channel {tv_channel}",
        "what's on {tv_channel}", "some live tv", "the {tv_channel} stream",
        "another {tv_genre} channel", "live {tv_genre} tv",
        "the news channel", "a sports channel", "a movie channel",
        "the {tv_channel} live feed", "live {tv_genre} channel",
        "whatever is on tv", "regular tv", "broadcast tv",
        "the {tv_genre} live channel", "a different channel",
        # 2 slots
        "the {tv_genre} channel {tv_channel}", "{tv_channel}, a {tv_genre} channel",
        # question / mood / contextual
        "do you have {tv_channel}", "got any {tv_genre} channels",
        "some live tv for the background", "a {tv_genre} channel to put on",
        "what's on right now", "got a {tv_genre} channel",
        "some live tv while i cook", "put something on tv",
    ]),
    "audio": ("lead_play_audio", [
        "something", "something to listen to", "some audio", "anything",
        "something good", "something nice", "something to fill the silence",
        "whatever", "anything you like", "something in the background",
        "something chill", "a bit of audio",
        # question / mood / contextual
        "do you have anything to listen to", "got anything good",
        "something to listen to while i cook", "something to help me relax",
    ]),
    "video": ("lead_watch", [
        "a video", "some videos", "something to watch",
        "a video about {documentary_title}", "something interesting", "any video",
        "a random video", "another video", "a short video", "some clips",
        "a video to watch", "anything to watch", "a clip", "a funny video",
        "a viral video", "a how-to video", "a tutorial video", "an explainer video",
        "a video about anything", "a quick clip", "a recommended video",
        "a trending video", "a video on {documentary_title}",
        # question / mood / contextual
        "do you have any videos", "got something to watch",
        "a video to pass the time", "something quick to watch",
        "got a funny video", "a video while i eat lunch",
        "something interesting to watch", "a short video to fill the time",
    ]),
    "video_episodes": ("lead_watch", [
        "episodes of {tv_show_title}", "the next episode of {tv_show_title}",
        "more episodes of {tv_show_title}", "a web series", "a video series",
        "an online video series", "the latest episode of {tv_show_title} online",
        "the next episode", "more of that web series",
        "the new episodes of {tv_show_title}", "another episode of {tv_show_title}",
        "a video podcast series", "the {tv_show_title} web series",
        "the latest video episode", "an episode of that web series",
        "a youtube series", "the next part of {tv_show_title}",
        "more video episodes", "the {tv_show_title} video series",
        "a streaming web series", "an episode from {tv_show_title}",
        "the newest episode of {tv_show_title}", "an online series episode",
        # question / mood / contextual
        "do you have more episodes of {tv_show_title}",
        "got the next episode of {tv_show_title}",
        "some episodes to binge", "a web series for the evening",
        "do you have the latest {tv_show_title} episode",
        "got a good web series", "some video episodes to watch",
    ]),
    "visual_story": ("lead_watch", [
        "{visual_story_title}", "the motion comic {visual_story_title}",
        "a visual story", "the visual story {visual_story_title}",
        "a motion comic", "an interactive story", "an animated story",
        "the animated comic {visual_story_title}", "a digital comic story",
        "the {visual_story_title} motion comic", "some visual stories",
        "an illustrated story", "the illustrated story {visual_story_title}",
        "a webtoon", "the webtoon {visual_story_title}", "a story slideshow",
        "an animated comic", "a narrated picture story", "a picture story",
        "the digital story {visual_story_title}", "another visual story",
        "a kids visual story", "a motion comic episode",
        # question / mood / contextual
        "do you have any visual stories", "got the motion comic {visual_story_title}",
        "a visual story for bedtime", "an animated story to watch",
        "do you have the webtoon {visual_story_title}", "got any visual stories",
        "a visual story to wind down with",
    ]),
    # ── playlist (PLAYLIST) ─────────────────────────────────────────────────
    "playlist": ("lead_play_audio", [
        "my playlist", "my liked songs", "my favorites", "my favourites",
        "a {playlist_mood} playlist", "a {playlist_mood} mix",
        "my {playlist_activity} mix", "my {playlist_activity} playlist",
        "my saved songs", "the {playlist_mood} playlist", "my daily mix",
        "my discover weekly", "a fresh playlist", "my top songs",
        "a {playlist_activity} playlist", "the {playlist_activity} mix",
        # 2 slots — separated by a literal word so expansion is valid
        "a {playlist_mood} playlist for {playlist_activity}",
        "my {playlist_activity} mix for when i feel {playlist_mood}",
        "a {playlist_mood} mix to {playlist_activity}",
        # question / mood / contextual
        "do you have a {playlist_mood} playlist", "got a {playlist_activity} playlist",
        "a {playlist_mood} playlist for the morning",
        "a {playlist_activity} playlist to keep me going",
    ]),
    # ── sound_effect (SOUND_EFFECT) ─────────────────────────────────────────
    "sound_effect": ("lead_play_audio", [
        "a {sound_name} sound", "a {sound_name} sound effect",
        "the sound of {sound_name}", "a {sound_name} noise",
        "the {sound_name} sound effect", "some {sound_name} sounds",
        "a {sound_name} sfx", "the {sound_name} noise", "{sound_name} sounds",
        "the {sound_name} sfx", "a clip of a {sound_name}", "a {sound_name} clip",
        "the sound effect of {sound_name}", "a recording of a {sound_name}",
        # question / mood / contextual
        "do you have a {sound_name} sound", "got a {sound_name} sound effect",
        "a {sound_name} sound for my video", "the sound of a {sound_name} please",
    ]),
    # ── interactive_fiction (INTERACTIVE_FICTION) ───────────────────────────
    "interactive_fiction": ("lead_game", [
        "a text adventure", "some interactive fiction",
        "a choose your own adventure", "a choose your own adventure game",
        "the interactive story {game_title}", "an interactive novel",
        "a text adventure game", "an interactive fiction game",
        "the text adventure {game_title}", "a story game", "a branching story game",
        "the interactive fiction {game_title}", "a parser game",
        "a {game_genre} text adventure", "an old text adventure",
        "a interactive story", "a gamebook", "the gamebook {game_title}",
        "an if game", "a classic text adventure", "the story game {game_title}",
        "a {game_genre} interactive fiction game", "an adventure game with choices",
        "the branching story {game_title}", "another text adventure",
        "a text based game", "the interactive novel {game_title}",
        # question / mood / contextual
        "do you have any text adventures", "got a good choose your own adventure",
        "a text adventure to play tonight", "an interactive story for bedtime",
        "do you have the text adventure {game_title}",
        "got any {game_genre} text adventures", "an interactive story to wind down with",
    ]),
    # ── ambient (PROCEDURAL_AMBIENT, non-asmr) ──────────────────────────────
    "ambient": ("lead_play_audio", [
        "{ambient_sound}", "some {ambient_sound}", "{ambient_sound} sounds",
        "white noise", "brown noise", "pink noise", "rain sounds",
        "ocean waves", "nature sounds", "forest sounds", "thunderstorm sounds",
        "focus music", "sleep sounds", "some ambient music", "some ambient noise",
        "the sound of {ambient_sound}", "a {ambient_sound} soundscape",
        "a loop of {ambient_sound}", "gentle {ambient_sound}",
        "some {ambient_sound} for sleep", "calming {ambient_sound}",
        # question / mood / contextual
        "do you have any {ambient_sound}", "got some {ambient_sound} sounds",
        "some {ambient_sound} to help me sleep", "{ambient_sound} while i work",
    ]),
    # ── comic / manga (COMIC, read) ─────────────────────────────────────────
    "comic": ("lead_read", [
        "the manga {comic_title}", "the comic {comic_title}", "{comic_title}",
        "a {comic_genre} manga", "a {comic_genre} comic",
        "the comic book {comic_title}", "a graphic novel",
        "some manga", "a {comic_genre} graphic novel", "a webcomic",
        "the next chapter of {comic_title}", "the latest {comic_title} chapter",
        "a comic book", "the manga series {comic_title}", "a {comic_genre} webcomic",
        "the graphic novel {comic_title}", "a chapter of {comic_title}",
        "some {comic_genre} manga", "a digital comic", "the webcomic {comic_title}",
        "a {comic_genre} manga series", "more of {comic_title}",
        "an issue of {comic_title}", "a superhero comic",
        # 2 slots
        "the {comic_genre} manga {comic_title}", "the {comic_genre} comic {comic_title}",
        "a {comic_genre} comic called {comic_title}",
        "the {comic_genre} graphic novel {comic_title}",
        # question / mood / contextual
        "do you have any {comic_genre} manga", "got a good {comic_genre} comic",
        "a {comic_genre} comic to read tonight", "some manga to read",
        "do you have the manga {comic_title}", "got any {comic_genre} graphic novels",
        "a {comic_genre} comic for the weekend",
    ]),
}

# Adult / NSFW — retained ONLY as content-filter (detect-to-block) signal.
# These DO carry real entity slots ({pornstar}, {adult_title}, {hentai_name},
# {adult_streaming_service}) so the slice is diverse and learnable; every line
# is labelled adult/adult_audio/hentai so the ``adult`` genre reaches the filter.
ADULT_BODIES: Dict[str, tuple] = {
    "adult": ("lead_watch", [
        # named / provider forms
        "some adult videos", "an adult movie", "porn", "some porn", "nsfw videos",
        "an adult film", "some adult content", "{pornstar}", "a video with {pornstar}",
        "something with {pornstar}", "a scene with {pornstar}", "{adult_title}",
        "the adult film {adult_title}", "{pornstar} videos", "porn with {pornstar}",
        "a {pornstar} scene", "adult videos on {adult_streaming_service}",
        "porn from {adult_streaming_service}", "{adult_title} with {pornstar}",
    ]),
    "adult_audio": ("lead_play_audio", [
        "some adult audio", "an adult audio story", "nsfw audio",
        "an adult audiobook", "an erotic audio story", "adult audio with {pornstar}",
        "an erotic audiobook", "a steamy audio story", "an explicit audio story",
        "some erotic audio", "an adult podcast", "a sensual audio story",
        "an 18+ audio story", "an explicit audiobook", "a mature audio story",
        "some nsfw audio content", "an erotic audio drama", "a spicy audio story",
        "an adult audio scene", "erotic audio with {pornstar}",
        "an erotic audio session", "an adult radio drama",
        "another erotic audio story",
        # question / mood / contextual
        "do you have any adult audio", "got an erotic audio story",
        "some erotic audio for the night", "an adult audio story to relax to",
    ]),
    "hentai": ("lead_watch", [
        # descriptive forms — fire detection without a named title
        "some hentai", "an adult anime", "nsfw anime", "a hentai episode",
        "an explicit anime", "an 18+ anime", "an uncensored hentai",
        "hentai on {adult_streaming_service}", "more hentai", "an ecchi anime",
        "a lewd anime", "an adult animated series", "an x-rated anime",
        "some uncensored hentai", "a hentai series", "an explicit hentai",
        "an adult cartoon", "a mature anime",
        # named forms — real hentai titles / studios (detect-to-block)
        "{hentai_title}", "the hentai {hentai_title}",
        "an episode of {hentai_title}", "the anime {hentai_title}",
        "hentai by {hentai_studio}", "a {hentai_studio} hentai",
        "adult anime {hentai_title}", "{hentai_title} on {adult_streaming_service}",
        "the {hentai_studio} hentai {hentai_title}", "more {hentai_title}",
        "another episode of {hentai_title}", "a hentai by {hentai_studio}",
    ]),
}

# DESCRIPTIVE adult templates (en-us) — detection must fire on a DESCRIPTION, not
# only a named performer, so the content filter cannot be evaded by avoiding a
# name.  These slot-fill real physical-attribute pools mined from the performer
# datasets and are labelled adult (detect-to-block training rows ONLY).
ADULT_DESCRIPTIVE_BODIES: Dict[str, tuple] = {
    "adult": ("lead_watch", [
        "porn with a {adult_hair_color} haired performer",
        "an adult video with a {adult_hair_color} haired star",
        "porn with {adult_eye_color} eyes",
        "a {adult_ethnicity} porn star", "some {adult_ethnicity} porn",
        "a {adult_ethnicity} adult video",
        "an adult film with a {adult_ethnicity} performer",
        "porn with a {adult_body_type} performer",
        "an adult video with a {adult_body_type} star",
        "a porn performer with {adult_marking}",
        "an adult star with {adult_marking}",
        "a {adult_hair_color} haired {adult_ethnicity} porn star",
        "an adult scene with a {adult_eye_color} eyed performer",
    ]),
    "adult_audio": ("lead_play_audio", [
        "an erotic story with a {adult_ethnicity} narrator",
    ]),
}

# Attribute-bearing variants (year / decade / country) appended to normal types
# so the surface text covers descriptive media requests.  These reuse real
# attribute pools mined from the dataset columns.
ATTRIBUTE_BODIES: Dict[str, tuple] = {
    "movie": ("lead_watch", [
        "a movie from {release_year}", "some {media_country} cinema",
        "a {media_country} film", "a movie from the {release_decade}",
        "a {movie_genre} film from the {release_decade}",
    ]),
    "music": ("lead_play_audio", [
        "some music from the {release_decade}", "a {release_year} hit",
        "some {media_country} music", "a hit from the {release_decade}",
        "a {music_genre} playlist from the {release_decade}",
    ]),
    "tv_show": ("lead_watch", [
        "a show from the {release_decade}", "the {tv_network} show {tv_show_title}",
        "a {media_country} series", "a {tv_genre} series from the {release_decade}",
    ]),
    "anime": ("lead_watch", [
        "an anime from the {release_decade}", "an anime from {release_year}",
    ]),
    "game": ("lead_game", [
        "a game from {release_year}", "a {game_genre} game from the {release_decade}",
    ]),
}


# ---------------------------------------------------------------------------
# CONFUSABLES — cross-type templates where a foreign entity of type X appears
# but the CORRECT media_type is Y, driven by context words.  The slot is still
# filled with the real foreign entity (so ``ner_<foreign_label>=1``), but the
# row is labelled with the contextually-correct type — that mismatch is the
# hard training signal that teaches disambiguation.  The disambiguating context
# words (soundtrack / theme / trailer / making of / documentary / audiobook of /
# music video / about) are keyword features (see SoundtrackKeyword.voc,
# TrailerKeyword.voc, … and ``CategoricalFeatureExtractor``).
# ---------------------------------------------------------------------------
CONFUSABLE_BODIES: Dict[str, tuple] = {
    # movie_title / movie_composer present, but it is MUSIC (the soundtrack)
    "music": ("lead_play_audio", [
        "the {movie_title} soundtrack", "the soundtrack to {movie_title}",
        "the soundtrack from {movie_title}", "music from {movie_title}",
        "the theme song from {movie_title}", "the theme from {movie_title}",
        "{movie_composer}'s score", "the score by {movie_composer}",
        "the {movie_title} theme", "songs from {movie_title}",
    ]),
    # foreign title/person present, but it is the TRAILER / BTS / MUSIC_VIDEO /
    # DOCUMENTARY / AUDIOBOOK / PODCAST — context word decides the type.
    "trailer": ("lead_watch", [
        "the {movie_title} trailer", "the trailer for {movie_title}",
        "the trailer of {tv_show_title}", "the teaser for {movie_title}",
    ]),
    "behind_the_scenes": ("lead_watch", [
        "behind the scenes of {movie_title}", "the making of {movie_title}",
        "{movie_title} behind the scenes", "the making of {tv_show_title}",
    ]),
    "music_video": ("lead_watch", [
        "{artist_name}'s music video", "the music video by {artist_name}",
        "the official video for {track_name}", "the {track_name} music video",
    ]),
    "documentary": ("lead_watch", [
        "a documentary about {movie_actor}", "a documentary about {artist_name}",
        "the {artist_name} documentary", "a documentary on {movie_director}",
        "the documentary about {movie_title}",
    ]),
    # foreign movie title, but the AUDIOBOOK narration is meant (audio).
    "audiobook": ("lead_play_audio", [
        "the audiobook of {movie_title}", "the audiobook version of {movie_title}",
        "{movie_title} as an audiobook",
    ]),
    # foreign movie title, but the BOOK (readable text) is meant — "read"+"novel".
    "book": ("lead_read", [
        "the {movie_title} novel", "the novel {movie_title}",
        "the book {movie_title}", "{movie_title} the novel",
    ]),
    "podcast": ("lead_play_audio", [
        "a podcast about {artist_name}", "a podcast about {movie_title}",
        "the podcast about {movie_actor}",
    ]),
    # adult content-filter confusable: a mainstream actor name in an adult ask
    "adult": ("lead_watch", [
        "a porn parody of {movie_title}", "an adult parody of {tv_show_title}",
    ]),
}


def _voc_name_to_filename(name: str) -> str:
    return f"{name}.voc"


def write_vocab(lang: str) -> int:
    leads = LEADINS.get(lang)
    if not leads:
        return 0
    voc_dir = os.path.join(TEMPLATES_DIR, "vocab", lang)
    os.makedirs(voc_dir, exist_ok=True)
    for name, members in leads.items():
        path = os.path.join(voc_dir, _voc_name_to_filename(name))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(members) + "\n")
    return len(leads)


# Target expanded-template count per media type.  The decor (`pre`/`post`)
# multiplies a body's expansions by ~20, so it is the high-leverage balancing
# knob: rich types decorate FEWER of their bodies and thin types decorate ALL of
# them, bringing every type's expanded count into the same band regardless of how
# many bodies / how big the shared lead-in vocab is.  This keeps deep phrasing
# variety in the rich types (lots of bare bodies) without inflating their
# expanded-sample count past the thin tail.
BALANCE_TARGET = 16000
# decor multiplies one body's expansions by this (len(pre alternation) ×
# len(post alternation), each counting the empty option).
_DECOR_FACTOR = 20


def _decorate_quota(n_bodies: int, lead_count: int, decorate: bool) -> int:
    """How many bodies should receive the decorated variant to land in band."""
    if not decorate or lead_count <= 0:
        return 0
    # bare lines already contribute ~lead_count * n_bodies expansions.
    bare = lead_count * n_bodies
    remaining = BALANCE_TARGET - bare
    if remaining <= 0:
        return 0
    quota = remaining // (lead_count * _DECOR_FACTOR)
    return max(0, min(n_bodies, int(quota)))


def _take_decor(idx: int, total: int, quota: int) -> bool:
    """Evenly select `quota` of `total` indices (stride sampling)."""
    if quota >= total:
        return True
    if quota <= 0:
        return False
    # idx is decorated iff it lands on an evenly spaced grid of `quota` points.
    return (idx * quota) % total < quota


def _emit_lines(lead_ref: str, bodies: List[str], decor: dict,
                lead_count: int = 0) -> List[str]:
    lines: List[str] = []
    pre = decor.get("pre", "") if decor else ""
    post = decor.get("post", "") if decor else ""
    can_decorate = bool(pre or post)
    quota = _decorate_quota(len(bodies), lead_count, can_decorate)
    for i, body in enumerate(bodies):
        # bare lead-in + body (always — this is the core variation)
        lines.append(f"<{lead_ref}> {body}".strip())
        # decorate a representative, evenly-spaced subset of bodies up to `quota`.
        if can_decorate and _take_decor(i, len(bodies), quota):
            deco = " ".join(p for p in (pre, f"<{lead_ref}>", body, post) if p)
            lines.append(deco.strip())
    # dedup, preserve order
    seen, out = set(), []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out


def write_intents(lang: str) -> Dict[str, int]:
    lang_dir = os.path.join(TEMPLATES_DIR, lang)
    os.makedirs(lang_dir, exist_ok=True)
    decor = _DECOR.get(lang, {})
    leads = LEADINS.get(lang, {})
    counts: Dict[str, int] = {}
    # gather (lead_ref, [bodies]) per intent from the natural + adult sets, then
    # append the confusable bodies to the SAME intent file (correct label).
    per_intent: Dict[str, List[tuple]] = {}
    # confusables carry English context words (soundtrack / trailer / making of …)
    # so they are authored for en-us only; translators add localized confusables.
    srcs = [BODIES, ADULT_BODIES]
    if lang == "en-us":
        srcs.append(CONFUSABLE_BODIES)
        srcs.append(ATTRIBUTE_BODIES)
        srcs.append(ADULT_DESCRIPTIVE_BODIES)
    for src in srcs:
        for intent, (lead_ref, bodies) in src.items():
            per_intent.setdefault(intent, []).append((lead_ref, bodies))
    for intent, groups in per_intent.items():
        # Merge all bodies that share a lead_ref so the decorate quota (the
        # balancing knob) is computed against the WHOLE body set for this intent
        # — otherwise each source dict would be balanced in isolation.
        by_lead: Dict[str, List[str]] = {}
        seen_body: set = set()
        for lead_ref, bodies in groups:
            for b in bodies:
                key = (lead_ref, b)
                if key in seen_body:
                    continue
                seen_body.add(key)
                by_lead.setdefault(lead_ref, []).append(b)
        lines: List[str] = []
        seen: set = set()
        for lead_ref, bodies in by_lead.items():
            lead_count = len(leads.get(lead_ref, []))
            for ln in _emit_lines(lead_ref, bodies, decor, lead_count):
                if ln not in seen:
                    seen.add(ln)
                    lines.append(ln)
        path = os.path.join(lang_dir, f"{intent}.intent")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        counts[intent] = len(lines)
    return counts


def author(langs: List[str]) -> None:
    for lang in langs:
        nv = write_vocab(lang)
        ci = write_intents(lang)
        print(f"  {lang}: {nv} lead-in vocabs, "
              f"{len(ci)} intents, {sum(ci.values())} template lines")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Author .intent/.voc media templates",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--langs", nargs="*", default=list(LEADINS.keys()),
                    help=f"languages to author (default: {list(LEADINS.keys())})")
    args = ap.parse_args()
    print("Authoring templates …")
    author(args.langs)
    print(f"Done → {TEMPLATES_DIR}")


if __name__ == "__main__":
    main()
