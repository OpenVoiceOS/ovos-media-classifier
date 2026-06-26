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
        "lead_play_audio": ["play", "put on", "start", "start playing", "stream",
                          "queue up", "throw on", "play me", "play some",
                          "i want to listen to", "i want to hear",
                          "i'd like to hear", "i feel like listening to",
                          "can you play", "could you play", "please play",
                          "i'm in the mood for", "let's hear", "gimme some"],
        "lead_watch": ["watch", "play", "put on", "show me", "stream", "pull up",
                      "start watching", "i want to watch", "i'd like to watch",
                      "i feel like watching", "let's watch", "can we watch",
                      "could you play", "please put on", "how about we watch"],
        "lead_read": ["read", "read me", "read aloud", "read to me",
                     "i want to read", "i'd like to read", "can you read",
                     "could you read", "please read"],
        "lead_game": ["play", "launch", "open", "start", "boot up", "start a game of",
                     "let's play", "i want to play", "can we play", "i feel like playing"],
        "lead_tune": ["tune in to", "put on", "switch to", "turn on", "play",
                     "i want to listen to", "stream"],
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
    "music": ("lead_play_audio", [
        "{artist_name}", "some {artist_name}", "{artist_name}'s music",
        "music by {artist_name}", "songs by {artist_name}",
        "the best of {artist_name}", "anything by {artist_name}",
        "{track_name}", "the song {track_name}", "{track_name} by {artist_name}",
        "the album {album_name}", "{album_name}", "{album_name} by {artist_name}",
        "some {music_genre}", "some {music_genre} music", "a {music_genre} playlist",
        "a {music_genre} mix", "{record_label} artists",
        # entity-role richness
        "a song featuring {artist_name}", "{track_name} featuring {artist_name}",
        "music on {record_label}", "{album_name} on {record_label}",
        "the latest from {artist_name}", "a {music_genre} song by {artist_name}",
    ]),
    "movie": ("lead_watch", [
        "{movie_title}", "the movie {movie_title}", "the film {movie_title}",
        "a {movie_genre} movie", "a {movie_genre} film",
        "something with {movie_actor}", "a movie with {movie_actor}",
        "a film starring {movie_actor}", "a movie directed by {movie_director}",
        "a film by {movie_director}", "a movie produced by {movie_producer}",
        "{movie_title} with {movie_actor}", "an old {movie_genre} movie",
        "a classic {movie_genre} movie",
        # full entity-role richness (every crew role)
        "{movie_title} starring {movie_actor}",
        "{movie_title} directed by {movie_director}",
        "a film produced by {movie_producer}",
        "a movie written by {movie_writer}",
        "a film written by {movie_writer}",
        "a movie scored by {movie_composer}",
        "the film {movie_title} by {movie_director}",
        "a {movie_genre} film with {movie_actor}",
        "a film with music by {movie_composer}",
    ]),
    "tv_show": ("lead_watch", [
        "{tv_show_title}", "the show {tv_show_title}", "the series {tv_show_title}",
        "an episode of {tv_show_title}", "a {tv_genre} show", "a {tv_genre} series",
        "the new season of {tv_show_title}", "the latest episode of {tv_show_title}",
        "something like {tv_show_title}",
        # entity-role richness
        "{tv_show_title} starring {movie_actor}", "a series starring {movie_actor}",
        "{tv_show_title} on {tv_network}", "a {tv_genre} show on {tv_network}",
        "the {tv_network} series {tv_show_title}",
    ]),
    "podcast": ("lead_play_audio", [
        "{podcast_title}", "the {podcast_title} podcast",
        "an episode of {podcast_title}", "the latest {podcast_title} episode",
        "a {podcast_genre} podcast", "the podcast with {podcast_host}",
        "{podcast_host}'s podcast", "a podcast about {podcast_genre}",
        # entity-role richness
        "the podcast hosted by {podcast_host}",
        "{podcast_title} hosted by {podcast_host}",
        "a {podcast_genre} podcast with {podcast_host}",
    ]),
    "radio": ("lead_tune", [
        "{radio_station}", "the radio station {radio_station}",
        "{radio_station} radio", "some {radio_genre} radio",
        "a {radio_genre} station", "live radio", "the {radio_genre} station",
    ]),
    # AUDIOBOOK = play a NARRATION (audio).  Phrasing is audiobook-specific so it
    # stays distinct from BOOK (TTS-read text) below.
    "audiobook": ("lead_play_audio", [
        "the audiobook {audiobook_title}", "the audiobook of {audiobook_title}",
        "the audiobook {audiobook_title} by {audiobook_author}",
        "the audiobook {audiobook_title} narrated by {audiobook_narrator}",
        "an audiobook by {audiobook_author}",
        "an audiobook narrated by {audiobook_narrator}",
        "an audiobook by {audiobook_author} narrated by {audiobook_narrator}",
        "a {audiobook_genre} audiobook", "the audiobook version of {audiobook_title}",
    ]),
    # BOOK = TTS-read a readable text.  The "read" lead-in + book cues route this
    # to MediaType.BOOK (paged), distinct from the audiobook narration above.
    "book": ("lead_read", [
        "{book_title}", "the book {book_title}", "the novel {book_title}",
        "{book_title} by {book_author}", "a book by {book_author}",
        "a {book_genre} book", "a {book_genre} novel", "anything by {book_author}",
        "the {book_genre} book {book_title}", "the novel {book_title} by {book_author}",
        "a book about {book_genre}",
    ]),
    "news": ("lead_play_audio", [
        "the news", "today's news", "the latest news",
        "the news from {news_provider}", "{news_provider} news",
        "the {news_category} news", "{news_category} headlines",
        "the {news_category} headlines from {news_provider}",
    ]),
    "anime": ("lead_watch", [
        "{anime_title}", "the anime {anime_title}", "an episode of {anime_title}",
        "some anime", "anime by {anime_studio}", "a {anime_studio} anime",
        "the new season of {anime_title}", "the next episode of {anime_title}",
        # entity-role richness
        "the {anime_studio} anime {anime_title}",
        "{anime_title} by {anime_studio}",
    ]),
    "cartoon": ("lead_watch", [
        "{cartoon_title}", "the cartoon {cartoon_title}", "some cartoons",
        "an episode of {cartoon_title}", "a kids cartoon", "the show {cartoon_title}",
    ]),
    "documentary": ("lead_watch", [
        "{documentary_title}", "the documentary {documentary_title}",
        "a documentary", "a nature documentary", "some documentaries",
        "the {documentary_title} documentary", "a good documentary",
    ]),
    "short_film": ("lead_watch", [
        "{short_film_title}", "the short film {short_film_title}", "a short film",
        "some short films", "the short {short_film_title}",
        "an award-winning short film",
    ]),
    "silent_movie": ("lead_watch", [
        "{silent_movie_title}", "the silent movie {silent_movie_title}",
        "a silent film", "an old silent movie",
        "the silent film {silent_movie_title}", "a classic silent movie",
    ]),
    "bw_movie": ("lead_watch", [
        "{bw_movie_title}", "the black and white movie {bw_movie_title}",
        "a black and white film", "an old black and white movie",
        "the black and white film {bw_movie_title}",
    ]),
    "game": ("lead_game", [
        "{game_title}", "the game {game_title}", "a {game_genre} game",
        "a game on {game_platform}", "a round of {game_title}",
        "a quick {game_genre} game", "the {game_genre} game {game_title}",
        "some {game_genre}",
    ]),
    "music_video": ("lead_watch", [
        "the music video for {music_video_title}", "{music_video_title}",
        "the {artist_name} music video", "a music video by {artist_name}",
        "the official video for {music_video_title}", "{artist_name}'s music video",
    ]),
    "trailer": ("lead_watch", [
        "the trailer for {trailer_title}", "{trailer_title} trailer",
        "a movie trailer", "the latest trailers", "the {trailer_title} teaser",
        "the official trailer for {trailer_title}",
    ]),
    "behind_the_scenes": ("lead_watch", [
        "behind the scenes of {bts_title}", "the making of {bts_title}",
        "{bts_title} behind the scenes", "some behind the scenes footage",
        "the {bts_title} featurette",
    ]),
    "radio_theatre": ("lead_play_audio", [
        "the radio drama {radio_drama_title}", "{radio_drama_title}",
        "an audio drama", "a radio play", "some radio theatre",
        "the audio drama {radio_drama_title}",
    ]),
    "asmr": ("lead_play_audio", [
        "some asmr", "asmr", "relaxing asmr", "asmr by {asmr_artist}",
        "{asmr_artist}'s asmr", "asmr to fall asleep to",
        "some {asmr_artist} asmr", "calming asmr", "an asmr video",
    ]),
    "audio_description": ("lead_watch", [
        "{movie_title} with audio description",
        "the audio described version of {movie_title}",
        "a movie with audio description", "the described version of {movie_title}",
        "a described {movie_genre} movie",
    ]),
    "tv": ("lead_watch", [
        "{tv_channel}", "the channel {tv_channel}", "live tv", "{tv_channel} live",
        "whatever's on {tv_channel}", "a {tv_genre} channel", "channel {tv_channel}",
        "what's on {tv_channel}", "some live tv",
    ]),
    "audio": ("lead_play_audio", [
        "something", "something to listen to", "some audio", "anything",
        "something good", "something nice", "something to fill the silence",
    ]),
    "video": ("lead_watch", [
        "a video", "some videos", "something to watch",
        "a video about {documentary_title}", "something interesting", "any video",
    ]),
    "video_episodes": ("lead_watch", [
        "episodes of {tv_show_title}", "the next episode of {tv_show_title}",
        "more episodes of {tv_show_title}", "a web series", "a video series",
        "an online video series", "the latest episode of {tv_show_title} online",
        "the next episode", "more of that web series",
    ]),
    "visual_story": ("lead_watch", [
        "{visual_story_title}", "the motion comic {visual_story_title}",
        "a visual story", "the visual story {visual_story_title}",
        "a motion comic", "an interactive story",
    ]),
    # ── playlist (PLAYLIST) ─────────────────────────────────────────────────
    "playlist": ("lead_play_audio", [
        "my playlist", "my liked songs", "my favorites", "my favourites",
        "a {playlist_mood} playlist", "a {playlist_mood} mix",
        "my {playlist_activity} mix", "my {playlist_activity} playlist",
        "a {playlist_mood} {playlist_activity} playlist", "my saved songs",
        "the {playlist_mood} playlist", "my daily mix",
    ]),
    # ── sound_effect (SOUND_EFFECT) ─────────────────────────────────────────
    "sound_effect": ("lead_play_audio", [
        "a {sound_name} sound", "a {sound_name} sound effect",
        "the sound of {sound_name}", "a {sound_name} noise",
        "the {sound_name} sound effect", "some {sound_name} sounds",
    ]),
    # ── interactive_fiction (INTERACTIVE_FICTION) ───────────────────────────
    "interactive_fiction": ("lead_game", [
        "a text adventure", "some interactive fiction",
        "a choose your own adventure", "a choose your own adventure game",
        "the interactive story {game_title}", "an interactive novel",
        "a text adventure game",
    ]),
    # ── ambient (PROCEDURAL_AMBIENT, non-asmr) ──────────────────────────────
    "ambient": ("lead_play_audio", [
        "{ambient_sound}", "some {ambient_sound}", "{ambient_sound} sounds",
        "white noise", "brown noise", "pink noise", "rain sounds",
        "ocean waves", "nature sounds", "forest sounds", "thunderstorm sounds",
        "focus music", "sleep sounds", "some ambient music", "some ambient noise",
    ]),
    # ── comic / manga (COMIC, read) ─────────────────────────────────────────
    "comic": ("lead_read", [
        "the manga {comic_title}", "the comic {comic_title}", "{comic_title}",
        "a {comic_genre} manga", "a {comic_genre} comic",
        "the comic book {comic_title}", "a graphic novel",
        "the {comic_genre} manga {comic_title}", "some manga",
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
        "an erotic audiobook", "a steamy audio story",
    ]),
    "hentai": ("lead_watch", [
        # descriptive forms — fire detection without a named title
        "some hentai", "an adult anime", "nsfw anime", "a hentai episode",
        "an explicit anime", "an 18+ anime", "an uncensored hentai",
        "hentai on {adult_streaming_service}",
        # named forms — real hentai titles / studios (detect-to-block)
        "{hentai_title}", "the hentai {hentai_title}",
        "an episode of {hentai_title}", "the anime {hentai_title}",
        "hentai by {hentai_studio}", "a {hentai_studio} hentai",
        "adult anime {hentai_title}", "{hentai_title} on {adult_streaming_service}",
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
        "a {movie_genre} movie from {release_year}",
        "a movie from {release_year}", "a {release_decade} {movie_genre} film",
        "some {media_country} cinema", "a {media_country} film",
        "a {movie_genre} film from {media_country}",
    ]),
    "music": ("lead_play_audio", [
        "a {release_decade} {music_genre} playlist", "some {release_decade} music",
        "{music_genre} from {release_decade}", "a {release_year} hit",
        "some {media_country} music",
    ]),
    "tv_show": ("lead_watch", [
        "a {tv_genre} show from {release_year}", "a {release_decade} {tv_genre} series",
        "the {tv_network} show {tv_show_title}", "a {media_country} series",
    ]),
    "anime": ("lead_watch", [
        "a {release_decade} anime", "an anime from {release_year}",
    ]),
    "game": ("lead_game", [
        "a {game_genre} game from {release_year}", "a {release_decade} {game_genre} game",
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


def _emit_lines(lead_ref: str, bodies: List[str], decor: dict) -> List[str]:
    lines: List[str] = []
    pre = decor.get("pre", "") if decor else ""
    post = decor.get("post", "") if decor else ""
    for body in bodies:
        # bare lead-in + body
        lines.append(f"<{lead_ref}> {body}".strip())
        # one decorated variant per body keeps the file compact yet varied
        if pre or post:
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
        lines: List[str] = []
        seen: set = set()
        for lead_ref, bodies in groups:
            for ln in _emit_lines(lead_ref, bodies, decor):
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
