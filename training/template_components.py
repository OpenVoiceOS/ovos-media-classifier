"""Componential template generation: lead-in × slot-pattern × qualifier.

A flat hand-written template list saturates quickly. This module instead defines,
per intent, the orthogonal pieces of a spoken media request and combines them, so
a modest set of lead-ins × slot patterns × optional qualifiers yields thousands of
diverse `{slot}`-bearing templates. The output is a list of ``(template, intent)``
tuples in the same shape as ``generate_templates._EN_TEMPLATES``;
``generate_slot_filled_dataset`` fills the ``{slot}`` placeholders from the entity
pools.

The result is **balanced two ways**:

* **across request kinds** — lead-ins are tagged by kind (imperative / casual /
  polite / want / indirect) and each intent samples evenly across kinds, so no
  intent is all-imperative or all-question.
* **across taxonomy types** — every intent targets the same number of templates
  (``per_intent_target``), drawing from its own pattern set, so music does not
  swamp game or visual_story.

Slot names must match entity-pool labels produced by ``gather_entities`` (e.g.
``artist_name``, ``movie_title``).
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Lead-ins (request verb / opener) tagged by KIND, grouped by playback family.
# Balancing draws evenly across kinds so every request style is represented.
# ──────────────────────────────────────────────────────────────────────────────
_LEADINS: Dict[str, Dict[str, List[str]]] = {
    "audio": {
        "imperative": ["play", "put on", "start", "start playing", "stream",
                       "queue up", "throw on", "spin", "blast", "fire up",
                       "play some", "put on some", "throw on some", "play me",
                       "play me some", "play my"],
        "casual": ["i wanna hear", "i wanna listen to", "gimme", "gimme some",
                   "lemme hear", "let's hear", "let me hear", "let's listen to",
                   "i need some", "throw on"],
        "polite": ["can you play", "could you play", "would you play",
                   "can you put on", "would you mind playing", "please play",
                   "play ... please", "mind playing"],
        "want": ["i want to listen to", "i want to hear", "i'd like to hear",
                 "i'd like to listen to", "i feel like listening to",
                 "i feel like hearing"],
        "indirect": ["i'm in the mood for", "how about", "what about",
                     "set the mood with", "let's get into", "in the mood for"],
    },
    "video": {
        "imperative": ["watch", "play", "play the", "put on", "show me",
                       "stream", "pull up", "start", "start watching",
                       "throw on", "play me", "find me"],
        "casual": ["i wanna watch", "let's watch", "let me watch",
                   "lemme watch", "wanna watch", "let's put on"],
        "polite": ["can we watch", "can you put on", "could you play",
                   "would you put on", "would you mind putting on",
                   "please put on", "mind putting on"],
        "want": ["i want to watch", "i'd like to watch", "i feel like watching",
                 "i'm in the mood to watch", "i'd love to watch"],
        "indirect": ["how about we watch", "what about", "let's see",
                     "in the mood for", "how about"],
    },
    "read": {
        "imperative": ["read", "read me", "play", "play the audiobook",
                       "put on the audiobook", "start the audiobook", "narrate"],
        "casual": ["lemme listen to the audiobook of", "let's listen to the audiobook of"],
        "polite": ["can you read", "could you read", "would you read",
                   "can you play the audiobook"],
        "want": ["i want to listen to the audiobook", "i'd like the audiobook",
                 "i feel like an audiobook"],
        "indirect": ["i'm in the mood for an audiobook", "how about the audiobook"],
    },
    "game": {
        "imperative": ["play", "launch", "open", "start", "boot up",
                       "start a game of", "start playing", "fire up"],
        "casual": ["let's play", "lemme play", "wanna play", "let's play a game of"],
        "polite": ["can we play", "could we play", "can you start", "would you launch"],
        "want": ["i want to play", "i wanna play", "i feel like playing",
                 "i'd like to play"],
        "indirect": ["i'm in the mood for a game", "how about a game of",
                     "what about playing"],
    },
}

# Optional decorations producing natural noise. Empty weighted to keep bare forms.
SUFFIXES: List[str] = [
    "", "", "", "", "", " please", " for me", " right now", " again",
    " will you", " thanks", " on shuffle", " next", " real quick", " would you",
]
PREFIXES: List[str] = [
    "", "", "", "", "", "", "hey ", "ok ", "um ", "yeah ", "alright ", "please ",
]

# ──────────────────────────────────────────────────────────────────────────────
# Per-intent slot patterns (the variable middle). `family` picks the lead-in set.
# Thin intents are padded so every taxonomy type can reach the balanced target.
# ──────────────────────────────────────────────────────────────────────────────
PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "music": {"family": "audio", "patterns": [
        "{artist_name}", "some {artist_name}", "{artist_name}'s music",
        "music by {artist_name}", "songs by {artist_name}", "the best of {artist_name}",
        "{artist_name}'s greatest hits", "{track_name}", "the song {track_name}",
        "{track_name} by {artist_name}", "the album {album_name}", "{album_name}",
        "{album_name} by {artist_name}", "some {music_genre}", "some {music_genre} music",
        "a {music_genre} playlist", "{music_genre} from the 80s", "the latest from {artist_name}",
        "{artist_name} on shuffle", "a {music_genre} mix", "{record_label} artists",
        "that song {track_name}", "anything by {artist_name}", "a {music_genre} song",
    ]},
    "movie": {"family": "video", "patterns": [
        "{movie_title}", "the movie {movie_title}", "the film {movie_title}",
        "a {movie_genre} movie", "a {movie_genre} film", "something with {movie_actor}",
        "a movie with {movie_actor}", "a film starring {movie_actor}",
        "a movie directed by {movie_director}", "a film by {movie_director}",
        "{movie_title} with {movie_actor}", "the {movie_genre} film {movie_title}",
        "a movie produced by {movie_producer}", "an old {movie_genre} movie",
        "a good {movie_genre} flick", "{movie_actor}'s latest movie",
        "a classic {movie_genre} movie", "that movie {movie_title}",
    ]},
    "tv_show": {"family": "video", "patterns": [
        "{tv_show_title}", "the show {tv_show_title}", "the series {tv_show_title}",
        "an episode of {tv_show_title}", "the next episode of {tv_show_title}",
        "a {tv_genre} show", "a {tv_genre} series", "something like {tv_show_title}",
        "the new season of {tv_show_title}", "{tv_show_title} season one",
        "the latest episode of {tv_show_title}", "a good {tv_genre} series",
        "that show {tv_show_title}", "more {tv_show_title}",
    ]},
    "podcast": {"family": "audio", "patterns": [
        "the {podcast_title} podcast", "{podcast_title}", "an episode of {podcast_title}",
        "the latest {podcast_title} episode", "a {podcast_genre} podcast",
        "the podcast with {podcast_host}", "{podcast_host}'s podcast",
        "the newest episode of {podcast_title}", "a podcast about {podcast_genre}",
        "the {podcast_title} show", "that {podcast_genre} podcast",
        "a new episode of {podcast_title}",
    ]},
    "radio": {"family": "audio", "patterns": [
        "{radio_station}", "the radio station {radio_station}", "{radio_station} radio",
        "some {radio_genre} radio", "a {radio_genre} station", "live radio",
        "the {radio_genre} station", "tune in to {radio_station}",
        "the station {radio_station}", "{radio_genre} radio", "some radio",
    ]},
    "audiobook": {"family": "read", "patterns": [
        "{audiobook_title}", "the audiobook {audiobook_title}",
        "{audiobook_title} by {audiobook_author}", "a book by {audiobook_author}",
        "{audiobook_title} narrated by {audiobook_narrator}", "the book {audiobook_title}",
        "anything by {audiobook_author}", "the novel {audiobook_title}",
        "{audiobook_author}'s book", "a book narrated by {audiobook_narrator}",
        "the audiobook of {audiobook_title}", "the story {audiobook_title}",
    ]},
    "news": {"family": "audio", "patterns": [
        "the news", "today's news", "the latest news", "the news from {news_provider}",
        "{news_provider} news", "the {news_category} news", "a news briefing",
        "what's happening in the news", "the headlines", "{news_category} headlines",
        "my news briefing", "the {news_category} headlines from {news_provider}",
    ]},
    "anime": {"family": "video", "patterns": [
        "{anime_title}", "the anime {anime_title}", "an episode of {anime_title}",
        "some anime", "anime by {anime_studio}", "a {anime_studio} anime",
        "the new season of {anime_title}", "the next episode of {anime_title}",
        "that anime {anime_title}", "more {anime_title}",
    ]},
    "cartoon": {"family": "video", "patterns": [
        "{cartoon_title}", "the cartoon {cartoon_title}", "some cartoons",
        "an episode of {cartoon_title}", "a kids cartoon", "the show {cartoon_title}",
        "that cartoon {cartoon_title}", "more {cartoon_title}", "some {cartoon_title}",
    ]},
    "documentary": {"family": "video", "patterns": [
        "{documentary_title}", "the documentary {documentary_title}", "a documentary",
        "a documentary about {documentary_title}", "a nature documentary",
        "some documentaries", "the {documentary_title} documentary",
        "a good documentary", "that documentary {documentary_title}",
    ]},
    "short_film": {"family": "video", "patterns": [
        "{short_film_title}", "the short film {short_film_title}", "a short film",
        "some short films", "the short {short_film_title}", "an award-winning short film",
        "that short film {short_film_title}", "a quick short film",
    ]},
    "silent_movie": {"family": "video", "patterns": [
        "{silent_movie_title}", "the silent movie {silent_movie_title}",
        "a silent film", "an old silent movie", "the silent film {silent_movie_title}",
        "a classic silent movie", "some silent cinema",
    ]},
    "bw_movie": {"family": "video", "patterns": [
        "{bw_movie_title}", "the black and white movie {bw_movie_title}",
        "a black and white film", "an old black and white movie",
        "the black and white film {bw_movie_title}", "a classic black and white movie",
        "some old black and white cinema",
    ]},
    "game": {"family": "game", "patterns": [
        "{game_title}", "the game {game_title}", "a {game_genre} game",
        "a game on {game_platform}", "a round of {game_title}", "something fun to play",
        "a quick {game_genre} game", "the {game_genre} game {game_title}",
        "that game {game_title}", "a {game_platform} game", "some {game_genre}",
        "a game of {game_title}",
    ]},
    "music_video": {"family": "video", "patterns": [
        "the music video for {music_video_title}", "{music_video_title}",
        "the {artist_name} music video", "a music video by {artist_name}",
        "the official video for {music_video_title}", "{artist_name}'s music video",
        "the music video {music_video_title}", "a {artist_name} video",
    ]},
    "trailer": {"family": "video", "patterns": [
        "the trailer for {trailer_title}", "{trailer_title} trailer", "a movie trailer",
        "the latest trailers", "the {trailer_title} teaser", "the official trailer for {trailer_title}",
        "some new trailers", "that {trailer_title} trailer",
    ]},
    "behind_the_scenes": {"family": "video", "patterns": [
        "behind the scenes of {bts_title}", "the making of {bts_title}",
        "{bts_title} behind the scenes", "some behind the scenes footage",
        "the {bts_title} featurette", "behind the scenes footage of {bts_title}",
        "a making-of for {bts_title}",
    ]},
    "radio_theatre": {"family": "audio", "patterns": [
        "the radio drama {radio_drama_title}", "{radio_drama_title}", "an audio drama",
        "a radio play", "some radio theatre", "the audio drama {radio_drama_title}",
        "a radio play of {radio_drama_title}", "that radio drama {radio_drama_title}",
    ]},
    "asmr": {"family": "audio", "patterns": [
        "some asmr", "asmr", "relaxing asmr", "asmr by {asmr_artist}",
        "{asmr_artist}'s asmr", "asmr to fall asleep to", "some {asmr_artist} asmr",
        "calming asmr", "an asmr video",
    ]},
    "audio_description": {"family": "video", "patterns": [
        "{movie_title} with audio description", "the audio described version of {movie_title}",
        "a movie with audio description", "{movie_title} described", "describe {movie_title} for me",
        "the described version of {movie_title}", "a described {movie_genre} movie",
    ]},
    "tv": {"family": "video", "patterns": [
        "{tv_channel}", "the channel {tv_channel}", "live tv", "{tv_channel} live",
        "whatever's on {tv_channel}", "a {tv_genre} channel", "channel {tv_channel}",
        "what's on {tv_channel}", "the {tv_genre} channel", "some live tv",
    ]},
    "audio": {"family": "audio", "patterns": [
        "something", "something to listen to", "some audio", "anything",
        "something good", "whatever", "something nice", "anything good",
        "whatever you like", "something to fill the silence",
    ]},
    "video": {"family": "video", "patterns": [
        "a video", "some videos", "something to watch", "a video about {documentary_title}",
        "whatever's good", "something", "anything to watch", "a good video",
        "something interesting", "any video",
    ]},
    "video_episodes": {"family": "video", "patterns": [
        "a web series", "the next episode", "some online videos", "a video series",
        "the next episode of my series", "an online video series", "more of that web series",
        "the latest episode online",
    ]},
    "visual_story": {"family": "video", "patterns": [
        "{visual_story_title}", "the motion comic {visual_story_title}", "a visual story",
        "the visual story {visual_story_title}", "a motion comic", "that visual story {visual_story_title}",
        "an interactive story",
    ]},
}

# Adult / NSFW intents — retained ONLY as content-filter (detect-to-block) signal.
ADULT_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "adult": {"family": "video", "patterns": [
        "some adult videos", "an adult movie", "porn", "some porn", "nsfw videos",
        "an adult film", "some adult content",
    ]},
    "adult_audio": {"family": "audio", "patterns": [
        "some adult audio", "an adult audio story", "nsfw audio", "an adult audiobook",
    ]},
    "hentai": {"family": "video", "patterns": [
        "some hentai", "an adult anime", "nsfw anime", "a hentai episode",
    ]},
}

_DET_END = ("some", "the", "a", "an", "my")
_DET_START = ("the ", "a ", "an ", "some ", "my ")


def _grammatical(lead: str, pat: str) -> bool:
    """Reject double-determiner collisions (e.g. "play some" + "the album")."""
    last = lead.split()[-1].lower()
    return not (last in _DET_END and pat.lower().startswith(_DET_START))


def _render(lead: str, pat: str, rnd: random.Random, decorate: bool) -> str:
    """Render a lead-in + pattern into a template, handling the '...' slot and decoration."""
    if "..." in lead:                       # e.g. "play ... please" → "play {pat} please"
        core = lead.replace("...", pat).strip()
    else:
        core = f"{lead} {pat}".strip()
    if decorate:
        core = f"{rnd.choice(PREFIXES)}{core}{rnd.choice(SUFFIXES)}".strip()
    return " ".join(core.split())


def generate_componential_templates(
        per_intent_target: int = 500,
        decorate_ratio: float = 0.45,
        seed: int = 42,
        include_adult: bool = True) -> List[Tuple[str, str]]:
    """Combine lead-ins × slot patterns per intent, balanced across kinds + intents.

    @param per_intent_target: templates to keep per intent (balanced across the
        request kinds; intents with a smaller combinatorial space yield fewer).
    @param decorate_ratio: fraction of kept templates that get a prefix/suffix.
    @param seed: RNG seed for reproducible sampling.
    @param include_adult: include the adult/NSFW content-filter intents.
    @return: list of ``(template, intent)`` tuples, deduped.
    """
    rnd = random.Random(seed)
    out: List[Tuple[str, str]] = []
    specs = dict(PATTERNS)
    if include_adult:
        specs.update(ADULT_PATTERNS)

    for intent, spec in specs.items():
        kinds = _LEADINS[spec["family"]]
        # Build candidate cores grouped by request kind.
        by_kind: Dict[str, List[str]] = defaultdict(list)
        seen: set = set()
        for kind, leads in kinds.items():
            for lead in leads:
                for pat in spec["patterns"]:
                    if not _grammatical(lead, pat):
                        continue
                    core = _render(lead, pat, rnd, decorate=False)
                    if core and core not in seen:
                        seen.add(core)
                        by_kind[kind].append(core)
        for v in by_kind.values():
            rnd.shuffle(v)
        # Round-robin across kinds → even spread of request styles per intent.
        picks: List[str] = []
        kind_order = list(by_kind.keys())
        idx = {k: 0 for k in kind_order}
        while len(picks) < per_intent_target and any(idx[k] < len(by_kind[k]) for k in kind_order):
            for k in kind_order:
                if idx[k] < len(by_kind[k]):
                    picks.append(by_kind[k][idx[k]])
                    idx[k] += 1
                    if len(picks) >= per_intent_target:
                        break
        # Apply decoration to a fraction (adds register/politeness noise).
        for core in picks:
            if rnd.random() < decorate_ratio:
                core = f"{rnd.choice(PREFIXES)}{core}{rnd.choice(SUFFIXES)}".strip()
                core = " ".join(core.split())
            out.append((core, intent))
    # Final dedup (decoration may collide).
    seen_all: set = set()
    deduped: List[Tuple[str, str]] = []
    for t, i in out:
        key = (t, i)
        if key not in seen_all:
            seen_all.add(key)
            deduped.append((t, i))
    return deduped


if __name__ == "__main__":
    tmpls = generate_componential_templates()
    by_intent: Dict[str, int] = defaultdict(int)
    for _, intent in tmpls:
        by_intent[intent] += 1
    print(f"generated {len(tmpls)} templates across {len(by_intent)} intents")
    lo = min(by_intent.values()); hi = max(by_intent.values())
    print(f"balance: min {lo} / max {hi} per intent")
    for intent, n in sorted(by_intent.items()):
        print(f"  {intent:20s} {n}")
    print("\nsamples:")
    for t, i in tmpls[:20]:
        print(f"  [{i}] {t}")
