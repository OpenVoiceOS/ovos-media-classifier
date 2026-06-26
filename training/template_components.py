"""Componential template generation: lead-in × slot-pattern × qualifier.

A flat hand-written template list saturates quickly. This module instead defines,
per intent, the orthogonal pieces of a spoken media request and combines them, so
a few dozen lead-ins × a few dozen slot patterns × optional qualifiers yields
thousands of diverse `{slot}`-bearing templates. The output is a list of
``(template, intent)`` tuples in the same shape as ``generate_templates._EN_TEMPLATES``;
``generate_slot_filled_dataset`` fills the ``{slot}`` placeholders from the entity
pools.

Slot names must match entity-pool labels produced by ``gather_entities`` (e.g.
``artist_name``, ``movie_title``). Patterns are grouped by intent; lead-ins are
shared within a playback family (audio vs video vs read vs game) so each request
verb composes with every pattern.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Lead-ins (the request verb / opener), grouped by how the media is consumed.
# Each composes with every slot pattern of an intent in that family.
# ──────────────────────────────────────────────────────────────────────────────
AUDIO_LEADINS: List[str] = [
    "play", "play some", "play me", "play me some", "put on", "put on some",
    "start", "start playing", "stream", "queue up", "throw on", "throw on some",
    "i want to listen to", "i wanna listen to", "i want to hear", "i wanna hear",
    "i'd like to listen to", "i feel like listening to", "i'm in the mood for",
    "let's listen to", "let's hear", "can you play", "could you play",
    "can you put on", "would you play", "play something by", "give me",
    "give me some", "gimme some", "how about", "let me hear", "i need some",
    "set the mood with", "fire up", "spin", "blast", "play my",
]

VIDEO_LEADINS: List[str] = [
    "watch", "play", "play the", "put on", "show me", "stream",
    "i want to watch", "i wanna watch", "i'd like to watch", "i feel like watching",
    "let's watch", "can we watch", "can you put on", "could you play",
    "would you put on", "let me watch", "i'm in the mood to watch",
    "pull up", "start", "start watching", "throw on", "find me", "play me",
]

READ_LEADINS: List[str] = [
    "read", "read me", "play", "play the audiobook", "put on the audiobook",
    "i want to listen to the audiobook", "start the audiobook", "narrate",
    "can you read", "let's listen to the audiobook of", "stream the audiobook of",
]

GAME_LEADINS: List[str] = [
    "play", "let's play", "i want to play", "i wanna play", "start a game of",
    "launch", "open", "can we play", "let's play a game of", "boot up",
    "i feel like playing", "start playing",
]

# Optional decorations. The empty string keeps the bare form; the rest add the
# kind of natural noise real users produce. Applied as prefix or suffix.
SUFFIXES: List[str] = [
    "", "", "", "", "",                      # weight the bare form
    " please", " for me", " right now", " again", " will you", " thanks",
    " on shuffle", " next", " real quick", " would you",
]
PREFIXES: List[str] = [
    "", "", "", "", "", "",                   # weight the bare form
    "hey ", "ok ", "um ", "yeah ", "alright ", "please ",
]

# ──────────────────────────────────────────────────────────────────────────────
# Per-intent slot patterns (the variable middle). `family` picks the lead-in set.
# ──────────────────────────────────────────────────────────────────────────────
PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "music": {"family": "audio", "patterns": [
        "{artist_name}", "some {artist_name}", "{artist_name}'s music",
        "music by {artist_name}", "songs by {artist_name}",
        "the best of {artist_name}", "{artist_name}'s greatest hits",
        "{track_name}", "the song {track_name}", "{track_name} by {artist_name}",
        "the album {album_name}", "{album_name}", "{album_name} by {artist_name}",
        "some {music_genre}", "some {music_genre} music", "a {music_genre} playlist",
        "{music_genre} from the 80s", "the latest from {artist_name}",
        "{artist_name} on shuffle", "a {music_genre} mix", "{record_label} artists",
        "that song {track_name}", "anything by {artist_name}",
    ]},
    "movie": {"family": "video", "patterns": [
        "{movie_title}", "the movie {movie_title}", "the film {movie_title}",
        "a {movie_genre} movie", "a {movie_genre} film", "something with {movie_actor}",
        "a movie with {movie_actor}", "a film starring {movie_actor}",
        "a movie directed by {movie_director}", "a film by {movie_director}",
        "{movie_title} with {movie_actor}", "the {movie_genre} film {movie_title}",
        "a movie produced by {movie_producer}", "an old {movie_genre} movie",
        "a good {movie_genre} flick", "{movie_actor}'s latest movie",
    ]},
    "tv_show": {"family": "video", "patterns": [
        "{tv_show_title}", "the show {tv_show_title}", "the series {tv_show_title}",
        "an episode of {tv_show_title}", "the next episode of {tv_show_title}",
        "a {tv_genre} show", "a {tv_genre} series", "something like {tv_show_title}",
        "the new season of {tv_show_title}", "{tv_show_title} season one",
    ]},
    "podcast": {"family": "audio", "patterns": [
        "the {podcast_title} podcast", "{podcast_title}", "an episode of {podcast_title}",
        "the latest {podcast_title} episode", "a {podcast_genre} podcast",
        "the podcast with {podcast_host}", "{podcast_host}'s podcast",
        "the newest episode of {podcast_title}", "a podcast about {podcast_genre}",
    ]},
    "radio": {"family": "audio", "patterns": [
        "{radio_station}", "the radio station {radio_station}", "{radio_station} radio",
        "some {radio_genre} radio", "a {radio_genre} station", "live radio",
        "the {radio_genre} station", "tune in to {radio_station}",
    ]},
    "audiobook": {"family": "read", "patterns": [
        "{audiobook_title}", "the audiobook {audiobook_title}",
        "{audiobook_title} by {audiobook_author}", "a book by {audiobook_author}",
        "{audiobook_title} narrated by {audiobook_narrator}",
        "the book {audiobook_title}", "anything by {audiobook_author}",
    ]},
    "news": {"family": "audio", "patterns": [
        "the news", "today's news", "the latest news", "the news from {news_provider}",
        "{news_provider} news", "the {news_category} news", "a news briefing",
        "what's happening in the news", "the headlines", "{news_category} headlines",
    ]},
    "anime": {"family": "video", "patterns": [
        "{anime_title}", "the anime {anime_title}", "an episode of {anime_title}",
        "some anime", "anime by {anime_studio}", "a {anime_studio} anime",
        "the new season of {anime_title}",
    ]},
    "cartoon": {"family": "video", "patterns": [
        "{cartoon_title}", "the cartoon {cartoon_title}", "some cartoons",
        "an episode of {cartoon_title}", "a kids cartoon",
    ]},
    "documentary": {"family": "video", "patterns": [
        "{documentary_title}", "the documentary {documentary_title}",
        "a documentary", "a documentary about {documentary_title}",
        "a nature documentary", "some documentaries",
    ]},
    "short_film": {"family": "video", "patterns": [
        "{short_film_title}", "the short film {short_film_title}", "a short film",
        "some short films",
    ]},
    "silent_movie": {"family": "video", "patterns": [
        "{silent_movie_title}", "the silent movie {silent_movie_title}",
        "a silent film", "an old silent movie",
    ]},
    "bw_movie": {"family": "video", "patterns": [
        "{bw_movie_title}", "the black and white movie {bw_movie_title}",
        "a black and white film", "an old black and white movie",
    ]},
    "game": {"family": "game", "patterns": [
        "{game_title}", "the game {game_title}", "a {game_genre} game",
        "a game on {game_platform}", "a round of {game_title}",
        "something fun to play", "a {game_genre} game",
    ]},
    "music_video": {"family": "video", "patterns": [
        "the music video for {music_video_title}", "{music_video_title}",
        "the {artist_name} music video", "a music video by {artist_name}",
        "the official video for {music_video_title}",
    ]},
    "trailer": {"family": "video", "patterns": [
        "the trailer for {trailer_title}", "{trailer_title} trailer",
        "a movie trailer", "the latest trailers", "the {trailer_title} teaser",
    ]},
    "behind_the_scenes": {"family": "video", "patterns": [
        "behind the scenes of {bts_title}", "the making of {bts_title}",
        "{bts_title} behind the scenes", "some behind the scenes footage",
    ]},
    "radio_theatre": {"family": "audio", "patterns": [
        "the radio drama {radio_drama_title}", "{radio_drama_title}",
        "an audio drama", "a radio play", "some radio theatre",
    ]},
    "asmr": {"family": "audio", "patterns": [
        "some asmr", "asmr", "relaxing asmr", "asmr by {asmr_artist}",
        "{asmr_artist}'s asmr", "asmr to fall asleep to",
    ]},
    "audio_description": {"family": "video", "patterns": [
        "{movie_title} with audio description", "the audio described version of {movie_title}",
        "a movie with audio description",
    ]},
    "tv": {"family": "video", "patterns": [
        "{tv_channel}", "the channel {tv_channel}", "live tv", "{tv_channel} live",
        "whatever's on {tv_channel}", "a {tv_genre} channel",
    ]},
    "audio": {"family": "audio", "patterns": [
        "something", "something to listen to", "some audio", "anything",
        "something good", "whatever",
    ]},
    "video": {"family": "video", "patterns": [
        "a video", "some videos", "something to watch", "a video about {documentary_title}",
        "whatever's good", "something",
    ]},
    "video_episodes": {"family": "video", "patterns": [
        "a web series", "the next episode", "some online videos", "a video series",
    ]},
    "visual_story": {"family": "video", "patterns": [
        "{visual_story_title}", "the motion comic {visual_story_title}",
        "a visual story",
    ]},
}

# Adult / NSFW intents — retained ONLY as content-filter (detect-to-block) signal.
ADULT_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "adult": {"family": "video", "patterns": [
        "some adult videos", "an adult movie", "porn", "some porn", "nsfw videos",
    ]},
    "adult_audio": {"family": "audio", "patterns": [
        "some adult audio", "an adult audio story", "nsfw audio",
    ]},
    "hentai": {"family": "video", "patterns": [
        "some hentai", "an adult anime", "nsfw anime",
    ]},
}


_DET_END = ("some", "the", "a", "an", "my")
_DET_START = ("the ", "a ", "an ", "some ", "my ")


def _grammatical(lead: str, pat: str) -> bool:
    """Reject double-determiner collisions (e.g. "play some" + "the album")."""
    last = lead.split()[-1].lower()
    return not (last in _DET_END and pat.lower().startswith(_DET_START))


def _decorate(core: str, rnd: random.Random) -> str:
    """Apply an optional prefix and/or suffix to a core request string."""
    prefix = rnd.choice(PREFIXES)
    suffix = rnd.choice(SUFFIXES)
    return f"{prefix}{core}{suffix}".strip()


def generate_componential_templates(
        per_intent_cap: int = 600,
        decorate_ratio: float = 0.5,
        seed: int = 42,
        include_adult: bool = True) -> List[Tuple[str, str]]:
    """Combine lead-ins × slot patterns (× optional decoration) per intent.

    @param per_intent_cap: max templates kept per intent (sampled for spread).
    @param decorate_ratio: fraction of combinations that also get a prefix/suffix
        variant (in addition to the bare lead-in+pattern form).
    @param seed: RNG seed for reproducible sampling.
    @param include_adult: include the adult/NSFW content-filter intents.
    @return: list of ``(template, intent)`` tuples, deduped.
    """
    rnd = random.Random(seed)
    leadins = {"audio": AUDIO_LEADINS, "video": VIDEO_LEADINS,
               "read": READ_LEADINS, "game": GAME_LEADINS}
    out: List[Tuple[str, str]] = []
    specs = dict(PATTERNS)
    if include_adult:
        specs.update(ADULT_PATTERNS)

    for intent, spec in specs.items():
        family_leadins = leadins[spec["family"]]
        seen: set = set()
        combos: List[str] = []
        for pat in spec["patterns"]:
            for lead in family_leadins:
                if not _grammatical(lead, pat):
                    continue
                core = f"{lead} {pat}".strip()
                if core not in seen:
                    seen.add(core)
                    combos.append(core)
                # a decorated variant for some combinations
                if rnd.random() < decorate_ratio:
                    dec = _decorate(core, rnd)
                    if dec not in seen:
                        seen.add(dec)
                        combos.append(dec)
        rnd.shuffle(combos)
        for core in combos[:per_intent_cap]:
            out.append((core, intent))
    return out


if __name__ == "__main__":
    tmpls = generate_componential_templates()
    by_intent: Dict[str, int] = {}
    for _, intent in tmpls:
        by_intent[intent] = by_intent.get(intent, 0) + 1
    print(f"generated {len(tmpls)} templates across {len(by_intent)} intents")
    for intent, n in sorted(by_intent.items(), key=lambda kv: -kv[1]):
        print(f"  {intent:20s} {n}")
    print("\nsamples:")
    for t, i in tmpls[:25]:
        print(f"  [{i}] {t}")
