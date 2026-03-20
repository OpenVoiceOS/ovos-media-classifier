"""Augment low-data OCP media intent classes via a local OpenAI-compatible LLM.

Automatically identifies which intent classes fall below ``--target`` and builds
a priority work queue (most under-represented first).  Progress is resume-safe:
restart the script and it picks up where it left off.

The script calls any server that exposes ``POST /v1/chat/completions`` — works
with llama.cpp, Ollama, vLLM, LM Studio, etc.

Prompts are tuned per-intent and optionally inject real entity names from
``Jarbas/WikidataMediaEntities`` so the model produces grounded utterances
("play a movie starring Cate Blanchett" rather than generic filler).

Multiple workers run concurrently (one thread per worker), enabling load
distribution across several llama.cpp instances or different models.

Usage::

    # single worker at the default URL
    python -m ovos_media_classifier.train.llm_augment

    # 4 concurrent workers at the same server
    python -m ovos_media_classifier.train.llm_augment --workers 4

    # two separate servers / models
    python -m ovos_media_classifier.train.llm_augment \\
        --worker http://host1:8080 gemma-3n \\
        --worker http://host2:8080 llama3

    # only augment specific intents
    python -m ovos_media_classifier.train.llm_augment \\
        --intents radio audio visual_story adult_audio

    # target 5000 per intent, batch of 30, dry-run first
    python -m ovos_media_classifier.train.llm_augment \\
        --target 5000 --batch 30 --dry-run

    # also generate negatives (not_ocp)
    python -m ovos_media_classifier.train.llm_augment --negatives

Tuning constants::

    API_URL              – default base URL
    MODEL                – default model
    DEFAULT_WORKER_COUNT – concurrent workers when --workers/--worker not given
    TEMPERATURE_MIN/MAX  – temperature sampled uniformly per request
    MAX_TOKENS           – upper bound on tokens per response
    BATCH_SIZE           – utterances requested per call
    REQUEST_TIMEOUT      – seconds before an HTTP request times out
    MAX_RETRIES          – failures before dropping an intent for the run
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from ovos_media_classifier.train import get_hf_cache_dir, get_output_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default worker pool — used when no --worker flags are supplied on the CLI.
# Add or remove entries to point at additional LLM servers.
DEFAULT_WORKERS: list[dict] = [
    {"url": "http://192.168.1.200:8000",  "model": "Qwen3-8B-GGUF"},
    {"url": "http://192.168.1.200:8103",  "model": "gemma-3n-E4B-it-GGUF"},
]
TEMPERATURE_MIN:      float = 0.6
TEMPERATURE_MAX:      float = 1.0
MAX_TOKENS:           int   = 2048
BATCH_SIZE:           int   = 25
REQUEST_TIMEOUT:      int   = 120
MAX_RETRIES:          int   = 3
NUM_SEED_EXAMPLES:    int   = 10    # real examples included in each prompt
DEFAULT_TARGET:       int   = 5000  # utterances to reach per intent

# ---------------------------------------------------------------------------
# Intents targeted by default (lowest coverage in balanced_dataset)
# ---------------------------------------------------------------------------

LOW_DATA_INTENTS = [
    # ── Brand-new intents — ZERO existing training data anywhere ──────────
    "tv",                  # MediaType.TV (9) — live IPTV / cable TV stream
    "music_video",         # MediaType.MUSIC_VIDEO — official music videos
    "trailer",             # MediaType.TRAILER — movie / show trailers
    "behind_the_scenes",   # MediaType.BEHIND_THE_SCENES — making-of / featurettes
    # ── Genuinely sparse (< 2k examples in balanced dataset) ─────────────
    "audio",               # MediaType.AUDIO — ambient/background audio
    "visual_story",        # MediaType.VISUAL_STORY — comics/graphic novels
    "adult_audio",         # MediaType.ADULT_AUDIO
    "radio",               # MediaType.RADIO — live radio
    "hentai",              # MediaType.HENTAI
    "adult",               # MediaType.ADULT
    "asmr",                # MediaType.ASMR
    "audio_description",   # MediaType.AUDIO_DESCRIPTION
    "silent_movie",        # MediaType.SILENT_MOVIE
    "bw_movie",            # MediaType.BLACK_WHITE_MOVIE
    "short_film",          # MediaType.SHORT_FILM
    "news",                # MediaType.NEWS
    "radio_theatre",       # MediaType.RADIO_THEATRE
    "cartoon",             # MediaType.CARTOON
    "anime",               # MediaType.ANIME
    "documentary",         # MediaType.DOCUMENTARY
    "game",                # MediaType.GAME
    "tv_show",             # MediaType.TV_SHOW (25) — episodic series (Breaking Bad)
    "video_episodes",      # MediaType.VIDEO_EPISODES (19) — YouTube channels/online series
    "podcast",             # MediaType.PODCAST
]

# ---------------------------------------------------------------------------
# Per-intent system prompts
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """\
You are a dataset generator for a voice assistant media classifier.
Generate natural spoken utterances a real user would say to a smart speaker to request media playback.

Rules:
- One utterance per line. No numbering, bullets, quotes, or explanations.
- Sound like real speech to a voice assistant, not search-engine queries.
- Vary length: ~30% very short (2–5 words), ~45% medium (6–12 words), ~25% longer.
- Vary register freely: casual ("chuck on some jazz"), polite ("could you please play…"),
  direct ("play X"), indirect ("I feel like watching something scary"), conversational.
- Vary syntax: imperatives, questions, statements, fragments.
- Name specific real-world entities (artists, titles, actors, shows, games…).
- Do NOT repeat or paraphrase the seed examples. Use them only for style/tone reference.
- Do NOT output anything except the utterances themselves.
"""

def _label_system(label_name: str, definition: str, distinctions: str = "") -> str:
    s = _SYSTEM_BASE
    s += f"\n\nLabel: {label_name}\nDefinition: {definition}"
    if distinctions:
        s += f"\nDistinct from: {distinctions}"
    return s


_INTENT_SYSTEM: dict[str, str] = {
    "music": _label_system(
        "MUSIC",
        "User wants to listen to music — any genre, artist, song, album, or mood-based request.",
        "podcast (talking show), radio (live broadcast), audiobook (narrated book), audio (ambient sounds)",
    ),
    "movie": _label_system(
        "MOVIE",
        "User wants to watch a feature-length film. Includes all genres and eras unless "
        "the request is specifically for a silent or black-and-white film.",
        "tv_show (episodic series), short_film (< ~40 min), silent_movie (no dialogue), "
        "bw_movie (specifically black & white), documentary (factual film), anime (Japanese animation)",
    ),
    "tv": _label_system(
        "TV  (= MediaType.TV = live IPTV stream)",
        "User wants to watch a LIVE television broadcast — a TV channel streaming in real time. "
        "This is NOT a request to watch a specific episode or series. "
        "Examples: 'put on BBC One', 'stream CNN', 'I want live TV', 'tune to ESPN'.",
        "tv_show (episodic series like Breaking Bad — on-demand, NOT live), "
        "video_episodes (YouTube channel or creator content), "
        "movie (single film), documentary (on-demand doc), radio (audio-only broadcast)",
    ),
    "tv_show": _label_system(
        "TV_SHOW  (= MediaType.TV_SHOW = episodic TV series)",
        "User wants to watch a specific TV series, episodic show, season, or episode — "
        "on-demand, not live. Includes dramas, comedies, reality TV, mini-series, and "
        "continuation requests ('next episode', 'continue', 'resume').",
        "tv (live TV channel stream — NOT a named series), "
        "video_episodes (YouTube channel or creator), "
        "movie (single film), anime (Japanese animation), cartoon (animated Western show)",
    ),
    "video_episodes": _label_system(
        "VIDEO_EPISODES  (= MediaType.VIDEO_EPISODES = YouTube / online video channel)",
        "User wants to watch content from a specific YouTube channel, online creator, "
        "or web-based video series — NOT broadcast TV, NOT a traditional TV series. "
        "Examples: 'play Linus Tech Tips', 'put on MrBeast', 'show me Kurzgesagt episodes'.",
        "tv (live broadcast TV channel), "
        "tv_show (traditional TV series like Breaking Bad), "
        "video (single generic clip — not a channel/series)",
    ),
    "anime": _label_system(
        "ANIME",
        "User wants to watch anime — Japanese animated series or films (not Western cartoons). "
        "Covers all anime genres: shonen, shojo, isekai, mecha, slice of life, etc.",
        "cartoon (Western animation), movie (live-action or CG film), tv_show (live-action series)",
    ),
    "cartoon": _label_system(
        "CARTOON",
        "User wants to watch a Western cartoon or animated show "
        "(American, European, etc. — NOT Japanese anime). "
        "Includes both kids' cartoons and adult animation.",
        "anime (Japanese animation), movie (animated film, e.g. Pixar), tv_show (live-action series)",
    ),
    "documentary": _label_system(
        "DOCUMENTARY",
        "User wants to watch a documentary — factual film or series about real events, "
        "people, nature, history, science, crime, sports, or society.",
        "movie (fictional narrative film), news (live news broadcast), podcast (audio only), "
        "tv_show (scripted episodic series)",
    ),
    "podcast": _label_system(
        "PODCAST",
        "User wants to listen to a podcast — a recorded audio programme with episodes, "
        "hosted by one or more people, covering a specific topic or format.",
        "radio (live broadcast station), audiobook (narrated book), news (news bulletin), "
        "music (music playback), radio_theatre (scripted audio drama)",
    ),
    "radio": _label_system(
        "RADIO",
        "User wants to listen to a radio station — live or internet radio broadcasting "
        "continuously. May be music radio, talk radio, news radio, or sports radio.",
        "podcast (pre-recorded episodes), music (on-demand music), news (single news bulletin), "
        "radio_theatre (scripted drama programme)",
    ),
    "radio_theatre": _label_system(
        "RADIO_THEATRE",
        "User wants to listen to a scripted radio drama, audio play, or old-time radio show. "
        "This is acted, scripted fiction performed for radio — not news, not podcasts.",
        "podcast (unscripted talk show), radio (live broadcast station), "
        "audiobook (single narrator reading a book)",
    ),
    "audiobook": _label_system(
        "AUDIOBOOK",
        "User wants to listen to a narrated book — fiction or non-fiction read aloud. "
        "Includes requests by title, author, genre, or narrator.",
        "podcast (hosted talk show), radio_theatre (scripted drama), music (music playback), "
        "news (news bulletin)",
    ),
    "news": _label_system(
        "NEWS",
        "User wants to listen to a news broadcast, bulletin, or news audio programme. "
        "Includes morning briefings, headline summaries, and news podcasts.",
        "radio (continuous radio station), podcast (non-news show), "
        "documentary (factual film, not live news)",
    ),
    "game": _label_system(
        "GAME",
        "User wants to play a video game on any platform (PC, console, mobile). "
        "Includes launching a specific game, browsing by genre, or asking to play something.",
        "movie (watching a film), video (watching gameplay video), "
        "tv_show (watching a show — even gaming shows)",
    ),
    "short_film": _label_system(
        "SHORT_FILM",
        "User wants to watch a short film — typically under 40 minutes, standalone narrative. "
        "Distinct from feature films because the user explicitly wants something short.",
        "movie (feature-length film), video (generic online clip), "
        "documentary (factual content)",
    ),
    "silent_movie": _label_system(
        "SILENT_MOVIE",
        "User wants to watch a silent film — specifically movies from the pre-sound era "
        "(before ~1930) with no synchronized spoken dialogue. "
        "Includes Chaplin, Keaton, expressionist cinema, etc.",
        "bw_movie (black-and-white but may have sound), movie (any film), "
        "movie (colour or sound films from later eras)",
    ),
    "bw_movie": _label_system(
        "BW_MOVIE",
        "User wants to watch a black-and-white film — classic Hollywood sound films, "
        "film noir, golden-age cinema. The user is asking for the aesthetic 'black and white', "
        "not necessarily silent.",
        "silent_movie (no dialogue, pre-1930), movie (any film including colour), "
        "documentary (factual)",
    ),
    "asmr": _label_system(
        "ASMR",
        "User wants to listen to ASMR content — autonomous sensory meridian response audio "
        "designed to produce relaxation or tingling. Includes trigger-based sounds, "
        "roleplays, and sleep ASMR.",
        "audio (generic ambient sound — not specifically ASMR), "
        "music (music playback), podcast (talk show)",
    ),
    "audio_description": _label_system(
        "AUDIO_DESCRIPTION",
        "User wants content with audio description (AD) — an additional narration track "
        "that describes visual elements for blind or visually impaired viewers. "
        "The user may say 'described version', 'AD version', 'audio described', etc.",
        "audiobook (reading a book), movie (standard film without AD), "
        "audio (generic sound content)",
    ),
    "video": _label_system(
        "VIDEO",
        "User wants to watch a generic online video — YouTube clips, vlogs, tutorials, "
        "compilations, gaming videos, etc. Not a specific named media type.",
        "movie (feature film), tv_show (broadcast series), documentary (factual film), "
        "short_film (standalone narrative short), game (playing a game), "
        "music_video (official music video for a song), video_episodes (a channel/series)",
    ),
    "music_video": _label_system(
        "MUSIC_VIDEO  (= MediaType.MUSIC_VIDEO = official music video)",
        "User wants to watch an official music video for a specific song or artist — "
        "the visual accompaniment released alongside a song. "
        "Examples: 'watch the Bohemian Rhapsody video', 'play the music video for Blinding Lights'.",
        "music (audio-only music), video (generic online clip), "
        "audio (ambient sound, not music)",
    ),
    "trailer": _label_system(
        "TRAILER  (= MediaType.TRAILER = movie or show trailer)",
        "User wants to watch a trailer, teaser, or preview for a film or TV series. "
        "Examples: 'show the trailer for Dune', 'play the Avengers teaser', "
        "'I want to see the preview for Oppenheimer'.",
        "movie (watching the full film, not the trailer), "
        "tv_show (watching the full series), behind_the_scenes (making-of content)",
    ),
    "behind_the_scenes": _label_system(
        "BEHIND_THE_SCENES  (= MediaType.BEHIND_THE_SCENES = making-of / featurette)",
        "User wants to watch making-of content, featurettes, cast interviews, "
        "bloopers, or production documentaries for a film or show. "
        "Examples: 'show the making of Dune', 'watch the featurette for The Batman', "
        "'I want behind the scenes from Stranger Things'.",
        "movie (watching the film itself), trailer (promotional preview), "
        "documentary (standalone factual documentary, not tied to a specific production)",
    ),
    "audio": _label_system(
        "AUDIO",
        "User wants to play generic ambient or background audio — NOT music, podcasts, radio, "
        "or ASMR. Covers white noise, nature sounds, binaural beats, sleep sounds, "
        "meditation audio, and atmospheric soundscapes.",
        "music (songs/artists), podcast (hosted show), radio (broadcast station), "
        "asmr (ASMR-specific content)",
    ),
    "visual_story": _label_system(
        "VISUAL_STORY",
        "User wants to read or view a comic book, graphic novel, or manga. "
        "The request is about sequential visual storytelling — not a film adaptation.",
        "movie (film version of a comic), anime (animated adaptation), "
        "audiobook (prose book read aloud)",
    ),
    "adult": _label_system(
        "ADULT",
        "User wants to watch adult/pornographic video content. "
        "Keep generated utterances direct and clinical.",
        "hentai (adult Japanese animation), adult_audio (audio-only adult content), "
        "movie (mainstream film)",
    ),
    "adult_audio": _label_system(
        "ADULT_AUDIO",
        "User wants to listen to adult audio content — erotic audiobooks, erotic ASMR, "
        "dirty talk audio, or erotic fiction read aloud. Audio only, not video.",
        "adult (adult video), hentai (adult animation), asmr (non-adult ASMR), "
        "audiobook (mainstream book)",
    ),
    "hentai": _label_system(
        "HENTAI",
        "User wants to watch hentai — adult Japanese animated content. "
        "Keep generated utterances direct and clinical.",
        "anime (non-adult Japanese animation), adult (live-action adult video), "
        "adult_audio (audio-only adult content)",
    ),
    "not_ocp": _label_system(
        "NOT_OCP",
        "Utterance is NOT a media playback request. "
        "Includes: information queries about media, smart home commands, reminders, "
        "weather, shopping, calendar, general knowledge, anything except 'play X now'.",
        "Any intent above (all of which ARE playback requests). "
        "Key test: would this cause media to start playing? If yes → wrong class.",
    ),
}

# ---------------------------------------------------------------------------
# Per-intent user prompt templates
# Placeholders: {n}  {seeds}  {entity_block}
# ---------------------------------------------------------------------------

_INTENT_USER: dict[str, str] = {

"music": """\
Generate {n} natural voice commands to play music.

Entity pool — use freely and mix with the extras below:
Artists: The Beatles, Pink Floyd, Led Zeppelin, Queen, David Bowie, Bob Dylan, Jimi Hendrix,
  The Rolling Stones, Nirvana, Radiohead, Taylor Swift, Beyoncé, Drake, Kendrick Lamar,
  Kanye West, The Weeknd, Billie Eilish, Adele, Ed Sheeran, Bruno Mars, Rihanna, Ariana Grande,
  Frank Sinatra, Nina Simone, Ella Fitzgerald, Miles Davis, John Coltrane, Chet Baker,
  Louis Armstrong, Duke Ellington, Herbie Hancock, Thelonious Monk, Charlie Parker,
  Johnny Cash, Dolly Parton, Willie Nelson, Hank Williams, Merle Haggard,
  Metallica, Black Sabbath, Iron Maiden, Slayer, System of a Down, Tool, Pantera,
  Daft Punk, Aphex Twin, Kraftwerk, Brian Eno, Massive Attack, Portishead,
  Tupac Shakur, Notorious B.I.G., Nas, Eminem, Wu-Tang Clan, NWA,
  Bach, Beethoven, Mozart, Chopin, Debussy, Brahms, Vivaldi, Tchaikovsky,
  Bob Marley, Peter Tosh, The Clash, The Cure, Joy Division, Depeche Mode
Songs: Bohemian Rhapsody, Hotel California, Stairway to Heaven, Imagine, Smells Like Teen Spirit,
  Purple Rain, Like a Rolling Stone, Yesterday, Superstition, Born to Run, God Save the Queen,
  Blinding Lights, Shape of You, Rolling in the Deep, Uptown Funk, Bad Guy, God's Plan, HUMBLE.
Albums: Dark Side of the Moon, Abbey Road, Thriller, Rumours, OK Computer, Nevermind,
  Random Access Memories, To Pimp a Butterfly, Lemonade, 1989, Led Zeppelin IV, Pet Sounds
Genres: jazz, blues, classical, hip-hop, rock, heavy metal, punk, indie, folk, country, reggae,
  soul, funk, R&B, EDM, techno, house, trance, ambient, lo-fi, bossa nova, K-pop, opera, gospel,
  drum and bass, dubstep, progressive rock, psychedelic, grunge, bluegrass, afrobeats
{entity_block}
Coverage — generate examples across ALL these patterns:
- by artist: "play The Beatles", "put on some Miles Davis", "I want to hear Nirvana"
- by song: "play Bohemian Rhapsody", "put on Blinding Lights", "find Stairway to Heaven"
- by album: "play Dark Side of the Moon", "queue up Abbey Road"
- by genre/mood: "play some jazz", "chill lo-fi hip-hop", "put on something upbeat", "sad music please"
- by activity: "workout playlist", "music to study to", "dinner music", "sleep music", "driving music"
- by provider: "play Drake on Spotify", "stream The Beatles on Apple Music"
- vague: "play something", "some music please", "I want to listen to music", "music"

Seed examples (style reference — do NOT repeat):
{seeds}

Output {n} utterances, one per line:""",

"movie": """\
Generate {n} natural voice commands to watch a movie.

Entity pool:
Films: The Godfather, The Shawshank Redemption, Pulp Fiction, Schindler's List, The Dark Knight,
  Fight Club, Forrest Gump, Inception, The Matrix, Goodfellas, Interstellar, Parasite, Spirited Away,
  Blade Runner, Alien, Star Wars, Jaws, Jurassic Park, The Lion King, Toy Story, WALL-E,
  Avengers: Endgame, Black Panther, No Country for Old Men, Mad Max: Fury Road, Get Out,
  Hereditary, Midsommar, Everything Everywhere All at Once, Oppenheimer, Dune, The Whale
Actors: Tom Hanks, Meryl Streep, Jack Nicholson, Robert De Niro, Al Pacino, Morgan Freeman,
  Denzel Washington, Cate Blanchett, Leonardo DiCaprio, Brad Pitt, Timothée Chalamet,
  Scarlett Johansson, Natalie Portman, Viola Davis, Ryan Gosling, Florence Pugh, Zendaya,
  Charlize Theron, Samuel L. Jackson, Christian Bale, Anthony Hopkins, Daniel Day-Lewis
Directors: Christopher Nolan, Quentin Tarantino, Martin Scorsese, Steven Spielberg,
  Stanley Kubrick, Francis Ford Coppola, Alfred Hitchcock, Ridley Scott, James Cameron,
  David Fincher, Wes Anderson, Greta Gerwig, Denis Villeneuve, Bong Joon-ho, Hayao Miyazaki,
  Akira Kurosawa, Pedro Almodóvar, Wong Kar-wai, Guillermo del Toro
Genres: action, comedy, drama, thriller, horror, sci-fi, romance, animation, documentary,
  fantasy, western, war, crime, mystery, noir, heist, superhero, musical, biopic
{entity_block}
Coverage:
- by title: "play Inception", "put on The Godfather", "I want to watch Parasite"
- by actor: "a Tom Hanks movie", "something with Meryl Streep", "show me a Leonardo DiCaprio film"
- by director: "play a Christopher Nolan film", "something by Tarantino"
- by genre/mood: "play a horror movie", "I want something scary", "comedy movie please", "action film"
- by era: "an 80s movie", "something from the 90s", "a classic film"
- by occasion: "movie night", "something for date night", "family movie"
- continuation: "resume the movie", "play the next one", "what else has she been in"
- vague: "find me something to watch", "play a movie", "I want to watch something"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"tv": """\
Generate {n} natural voice commands to watch LIVE television — a live TV channel or broadcast.
This label is MediaType.TV = live IPTV stream. Do NOT generate requests for specific named series.

Live TV channels:
UK: BBC One, BBC Two, BBC Four, ITV, Channel 4, Channel 5, Sky One, Sky Atlantic, Dave, E4
US: CNN, Fox News, MSNBC, NBC, CBS, ABC, ESPN, ESPN 2, TNT, TBS, FX, Comedy Central,
  Cartoon Network, Disney Channel, Nickelodeon, MTV, VH1, Bravo, Lifetime, Hallmark
Sports: Sky Sports, BT Sport, beIN Sports, Eurosport, NFL Network, NBA TV, MLB Network
News: BBC News Channel, Sky News, Al Jazeera, France 24, Deutsche Welle TV, Bloomberg TV, CNBC
European: RAI Uno, TVE, ARD, ZDF, France 2, RTL, M6
IPTV providers: Pluto TV, Plex TV, TVHeadend, Jellyfin Live TV, Samsung TV Plus, Tubi
Genres: news, sport, entertainment, kids, documentary, music
{entity_block}
Coverage — ALL examples must be about LIVE TV (not named series):
- by channel: "put on BBC One", "stream CNN", "I want ESPN live", "switch to ITV", "turn on Fox"
- by genre: "live sports channel", "a news channel please", "live kids TV"
- IPTV: "open Pluto TV", "play live channels on Plex", "tune to a live stream"
- vague: "turn on the TV", "play live TV", "I want to watch something live",
  "put on the telly", "live stream please", "stream live television"

⚠ DO NOT generate requests for specific series like "play Breaking Bad" — those are tv_show.

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"tv_show": """\
Generate {n} natural voice commands to watch an episodic TV series (on-demand, NOT live).
This label is MediaType.TV_SHOW = TV series episodes streamed on demand.

Entity pool:
Drama/Thriller: Breaking Bad, Better Call Saul, Succession, The Wire, The Sopranos, Ozark,
  Fargo, Peaky Blinders, True Detective, Mindhunter, Hannibal, Dexter, Yellowstone,
  Mad Men, House of Cards, Six Feet Under, Deadwood, The West Wing, Narcos, The Crown,
  Downton Abbey, Band of Brothers, Chernobyl, The Last of Us, Severance, The Bear,
  The White Lotus, Industry, Billions, Suits, Yellowjackets, Killing Eve, Fleabag
Sci-fi/Fantasy: Westworld, Black Mirror, Dark, Stranger Things, Battlestar Galactica,
  Firefly, Twin Peaks, The X-Files, Doctor Who, The Mandalorian, Squid Game, House of the Dragon,
  The Boys, The Witcher, Shadow and Bone, Rings of Power, Andor
Comedy/Sitcom: Friends, Seinfeld, The Office, Parks and Recreation, 30 Rock, Community,
  Brooklyn Nine-Nine, Abbott Elementary, Arrested Development, It's Always Sunny,
  What We Do in the Shadows, Futurama, South Park
International: Money Heist, Dark, 1899, Lupin, Fauda, Emily in Paris, Bridgerton
Genres: drama, comedy, thriller, crime, sci-fi, fantasy, horror, sitcom, procedural,
  limited series, miniseries, medical drama, legal drama, period drama
{entity_block}
Coverage:
- by title: "play Breaking Bad", "put on Succession", "I want to watch Fargo"
- continuation: "next episode of Severance", "continue The Wire", "resume Peaky Blinders",
  "where was I in Ozark", "keep playing The Bear", "play the next one"
- by season/episode: "season 3 of True Detective", "episode 4 of Chernobyl",
  "start The Sopranos from the beginning", "The Office season 2"
- by genre: "a good crime drama series", "funny sitcom please", "a thriller miniseries"
- binge: "I want to binge something tonight", "something I can marathon this weekend"
- vague: "put on a series", "show me a good show", "find me a TV show to watch"

⚠ DO NOT generate requests for live TV channels like "put on CNN" — those are tv (live).

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"video_episodes": """\
Generate {n} natural voice commands to watch content from a specific YouTube channel,
online video creator, or web-based video series.
This label is MediaType.VIDEO_EPISODES = YouTube channels / online video content.

Entity pool:
Tech / Science: Linus Tech Tips, Veritasium, Vsauce, Kurzgesagt, CGP Grey, Tom Scott,
  3Blue1Brown, Mark Rober, Wendover Productions, Real Engineering, Practical Engineering,
  ElectroBOOM, NileRed, SciShow, CrashCourse, TED-Ed, PBS Space Time
Entertainment / Gaming: MrBeast, PewDiePie, Markiplier, Jacksepticeye, Ninja, Dream,
  Pokimane, Technoblade, SSSniperWolf, Valkyrae, xQc, Ludwig, Moistcr1TiKaL, Penguinz0
Cooking / Lifestyle: Binging with Babish, Joshua Weissman, Bon Appétit, Tasty,
  Internet Shaquille, Adam Ragusea, Ethan Chlebowski, J. Kenji López-Alt
Commentary / Education: Hasan Piker, H.Bomberguy, Shaun, Philosophy Tube, Contrapoints,
  Lindsay Ellis, Some More News, SecondThought, Knowing Better
Platforms: YouTube, Nebula, Curiosity Stream, Dropout, Patreon video, Floatplane
{entity_block}
Coverage — ALL examples must be about a channel or creator, NOT a traditional TV series:
- by channel/creator: "play Linus Tech Tips", "put on MrBeast", "show me Kurzgesagt episodes"
- by creator + topic: "Veritasium science videos", "MrBeast challenges", "Tom Scott geography"
- continuation: "next Kurzgesagt video", "continue Binging with Babish", "more MrBeast"
- by platform: "something on Nebula", "a Curiosity Stream series", "Dropout show"
- vague: "play my subscriptions", "show me my YouTube feed", "something from YouTube"

⚠ DO NOT generate "play Breaking Bad" — that is tv_show. This is for creator channels/series.

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"anime": """\
Generate {n} natural voice commands to watch anime.

Entity pool:
Titles: Attack on Titan, Demon Slayer, One Piece, Naruto, Naruto Shippuden, Dragon Ball Z,
  My Hero Academia, Hunter x Hunter, Fullmetal Alchemist Brotherhood, Death Note, Steins;Gate,
  Cowboy Bebop, Neon Genesis Evangelion, Sword Art Online, Tokyo Ghoul, JoJo's Bizarre Adventure,
  One Punch Man, Mob Psycho 100, Vinland Saga, Chainsaw Man, Spy x Family, Jujutsu Kaisen,
  Made in Abyss, Re:Zero, Your Lie in April, Clannad, Violet Evergarden, A Silent Voice,
  Your Name, Spirited Away, Princess Mononoke, Akira, Ghost in the Shell, Berserk,
  Trigun, Samurai Champloo, Bleach, Fairy Tail, Black Clover, Haikyuu, Kuroko's Basketball
Genres: shonen, shojo, isekai, mecha, slice of life, fantasy, action, romance, horror,
  psychological, sports, mystery, sci-fi, seinen, josei, shounen-ai, magical girl
{entity_block}
Coverage:
- by title: "play Attack on Titan", "put on Demon Slayer", "I want to watch Cowboy Bebop"
- by genre: "some isekai anime", "slice of life anime", "a good mecha series"
- continuation: "next episode of One Piece", "continue Hunter x Hunter"
- vague: "play some anime", "I want to watch anime", "any anime"
- by mood: "something action packed", "a sad anime", "something funny"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"cartoon": """\
Generate {n} natural voice commands to watch a cartoon or animated show.

Entity pool:
Shows: SpongeBob SquarePants, Avatar: The Last Airbender, The Legend of Korra, Gravity Falls,
  The Simpsons, Futurama, Family Guy, Rick and Morty, Adventure Time, Steven Universe,
  Regular Show, The Amazing World of Gumball, Teen Titans, Teen Titans Go,
  Batman: The Animated Series, Justice League, X-Men: The Animated Series,
  DuckTales, Darkwing Duck, Garfield, Looney Tunes, Tom and Jerry, Scooby-Doo,
  Bob's Burgers, BoJack Horseman, Archer, Animaniacs, The Owl House, Amphibia,
  Hilda, Big Mouth, South Park, King of the Hill, Beavis and Butt-Head,
  Rugrats, Hey Arnold, Rocko's Modern Life, Invader Zim, Fairly OddParents,
  Danny Phantom, Phineas and Ferb, Gravity Falls, Bluey, Peppa Pig
{entity_block}
Coverage:
- by title: "play SpongeBob", "put on Avatar", "I want to watch Gravity Falls"
- by genre/audience: "a kids cartoon", "something for adults", "a superhero cartoon"
- vague: "put on a cartoon", "some cartoons please", "I want to watch something animated"
- continuation: "next episode of Rick and Morty", "keep playing Futurama"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"documentary": """\
Generate {n} natural voice commands to watch a documentary.

Entity pool:
Titles: Planet Earth, Blue Planet, Our Planet, Life, Frozen Planet, Planet Earth II,
  Free Solo, Meru, The Alpinist, Valley Uprising,
  Making a Murderer, The Jinx, Tiger King, The Staircase, Amanda Knox,
  The Last Dance, Icarus, Formula 1: Drive to Survive, Senna, Diego Maradona,
  Seaspiracy, Blackfish, Cowspiracy, An Inconvenient Truth, Before the Flood,
  Super Size Me, Fast Food Nation, Food Inc., Fed Up,
  13th, I Am Not Your Negro, Won't You Be My Neighbor?, RBG,
  Jiro Dreams of Sushi, Street Food, Chef's Table, Ugly Delicious,
  Amy, 20 Feet from Stardom, Searching for Sugar Man, Shut Up and Sing,
  My Octopus Teacher, Crip Camp, 13th, Fire of Love, All That Breathes
Topics: nature, wildlife, space, history, crime, politics, food, sport, music,
  technology, climate, social justice, biography, war, science, art, culture
{entity_block}
Coverage:
- by title: "play Planet Earth", "put on The Last Dance", "I want to watch Blackfish"
- by topic: "a nature documentary", "crime documentary", "something about space"
- by subject: "documentary about Michael Jordan", "a food doc", "climate change documentary"
- vague: "play a documentary", "I want to watch something informative", "find me a good doc"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"podcast": """\
Generate {n} natural voice commands to listen to a podcast.

Entity pool:
Shows: Serial, This American Life, Radiolab, 99% Invisible, Freakonomics Radio,
  The Joe Rogan Experience, Lex Fridman Podcast, How I Built This, Masters of Scale,
  Crime Junkie, My Favorite Murder, Casefile, Last Podcast on the Left, True Crime Garage,
  The Daily, Up First, Planet Money, Fresh Air, Throughline, Embedded,
  Stuff You Should Know, Stuff You Missed in History Class, No Such Thing as a Fish,
  TED Talks Daily, Hidden Brain, Invisibilia, On Being, Armchair Expert,
  SmartLess, Conan O'Brien Needs a Friend, My Brother My Brother and Me,
  The Tim Ferriss Show, Diary of a CEO, Call Her Daddy, The Moth, Risk!,
  Welcome to Night Vale, Wolf 359, Limetown, The Black Tapes, Bubble,
  Hardcore History, Revolutions, Revolutions, History Extra, In Our Time,
  Darknet Diaries, Reply All, 99% Invisible, StartUp, Gimlet, Wondery
Topics: true crime, comedy, tech, history, science, business, self-help, politics,
  sports, culture, storytelling, interview, fiction, horror, news
{entity_block}
Coverage:
- by show name: "play Serial", "put on the Joe Rogan podcast", "I want to hear Radiolab"
- by host: "the Lex Fridman podcast", "something with Malcolm Gladwell"
- by topic: "a true crime podcast", "tech podcast", "comedy podcast please"
- continuation: "resume my podcast", "next episode of Serial", "continue Hardcore History"
- vague: "play a podcast", "some podcasts please", "put on something to listen to"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"radio": """\
Generate {n} natural voice commands to listen to a radio station.

Entity pool:
Stations: BBC Radio 1, BBC Radio 2, BBC Radio 3, BBC Radio 4, BBC Radio 5 Live,
  BBC Radio 6 Music, BBC World Service, NPR, WBEZ, KCRW, KEXP, WNYC,
  Heart FM, Capital FM, Kiss FM, Absolute Radio, TalkSPORT, LBC, Classic FM,
  Radio Nova, NRJ, Fun Radio, Skyrock, Europe 1, France Inter, France Musique,
  Radio Nacional España, COPE, Cadena SER, Europa FM, Onda Cero,
  Bayern 3, SWR3, HR3, NDR 2, Deutschlandfunk,
  Radio 2 (Netherlands), Radio 538, 3FM,
  iHeartRadio, SiriusXM, Pandora, Radio Paradise, AccuRadio,
  Radio Mirchi, All India Radio, Radio City
Genres: pop, rock, classical, jazz, talk, news, sports, country, R&B, electronic
{entity_block}
Coverage:
- by station name: "play BBC Radio 6", "put on NPR", "stream KEXP"
- by genre: "jazz radio", "classical radio station", "talk radio", "sports radio"
- by location: "local radio", "French radio", "German radio station"
- vague: "play the radio", "some radio please", "turn on the radio"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"radio_theatre": """\
Generate {n} natural voice commands to listen to a radio drama or audio play.

Entity pool:
Shows: The Hitchhiker's Guide to the Galaxy (BBC Radio), Cabin Pressure, The Goon Show,
  I'm Sorry I'll Read That Again, Just a Minute, The Archers, Desert Island Discs,
  Sherlock Holmes BBC Radio, War of the Worlds (Orson Welles), Sorry Wrong Number, Suspense,
  The Adventures of Superman, Jack Benny Program, Fibber McGee and Molly,
  Escape, Dimension X, X Minus One, Inner Sanctum Mysteries, The Shadow,
  The Lone Ranger, Gunsmoke, Have Gun Will Travel, Dragnet, Lux Radio Theatre,
  Old Time Radio, Radio Mystery Theater, Quiet Riot Girl, Welcome to Night Vale
Providers: BBC Radio 4, BBC Sounds, Audible, Escape Pod, Old Time Radio Researchers Group
Genres: mystery, comedy, horror, sci-fi, adventure, drama, thriller, western, crime
{entity_block}
Coverage:
- by show: "play Cabin Pressure", "put on The Goon Show", "I want The Hitchhiker's Guide"
- by genre: "a radio mystery", "old time radio comedy", "horror radio play"
- by era: "old time radio", "classic radio shows", "1940s radio drama"
- by provider: "BBC Radio 4 drama", "something from Audible"
- vague: "play a radio drama", "audio play please", "a radio play"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"audiobook": """\
Generate {n} natural voice commands to listen to an audiobook.

Entity pool:
Books: Harry Potter, The Lord of the Rings, Dune, Foundation, A Song of Ice and Fire,
  The Wheel of Time, The Kingkiller Chronicle, 1984, Brave New World, Fahrenheit 451,
  Animal Farm, To Kill a Mockingbird, The Great Gatsby, Moby Dick, Crime and Punishment,
  The Hitchhiker's Guide to the Galaxy, Good Omens, Sapiens, A Brief History of Time,
  Thinking Fast and Slow, Atomic Habits, The Power of Habit, Man's Search for Meaning,
  The Alchemist, The Name of the Wind, Words of Radiance, Ender's Game, Project Hail Mary,
  The Martian, Recursion, Dark Matter, Gone Girl, The Girl with the Dragon Tattoo,
  And Then There Were None, Murder on the Orient Express, Big Little Lies
Authors: Stephen King, Agatha Christie, J.R.R. Tolkien, Frank Herbert, Brandon Sanderson,
  Neil Gaiman, Terry Pratchett, Douglas Adams, George R.R. Martin, Robert Jordan,
  James Patterson, Michael Connelly, Lee Child, John Grisham, Toni Morrison,
  Cormac McCarthy, Kazuo Ishiguro, Haruki Murakami, Yuval Noah Harari
Narrators: Jim Dale, Stephen Fry, Frank Muller, Roy Dotrice, Scott Brick, Kate Reading,
  Michael Kramer, January LaVoy, Wil Wheaton, George Guidall, Mark Bramhall
Genres: fiction, thriller, mystery, sci-fi, fantasy, romance, biography, history,
  self-help, business, non-fiction, horror, literary fiction
{entity_block}
Coverage:
- by title: "read me Harry Potter", "play Dune audiobook", "I want to listen to Sapiens"
- by author: "a Stephen King book", "audiobook by Agatha Christie", "something by Neil Gaiman"
- by narrator: "the Jim Dale narration", "read by Stephen Fry"
- by genre: "a thriller audiobook", "sci-fi book", "something non-fiction"
- vague: "play an audiobook", "read me a book", "I want something to listen to while I drive"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"news": """\
Generate {n} natural voice commands to listen to news audio.

Entity pool:
Providers: BBC News, CNN, NPR, Al Jazeera, Reuters, The Guardian, Associated Press,
  Sky News, CBS News, NBC News, ABC News, Bloomberg, Financial Times, The Economist,
  France 24, Deutsche Welle, Euronews, RFI, RTVE, NHK World, ABC Australia,
  The Daily (NYT podcast), Up First (NPR), Today in Focus (Guardian),
  Global News Podcast (BBC), Monocle 24
Topics: world news, local news, sports news, tech news, business news, politics,
  entertainment news, science news, health news, climate news, breaking news
Times: morning briefing, evening news, headlines, daily update, weekly roundup
{entity_block}
Coverage:
- generic: "play the news", "give me the headlines", "what's happening in the world"
- by provider: "BBC News please", "put on NPR", "play Al Jazeera"
- by topic: "sports news", "tech news today", "business headlines"
- by time: "morning news", "evening bulletin", "tonight's news", "latest headlines"
- by format: "a news podcast", "news briefing", "five-minute news update"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"game": """\
Generate {n} natural voice commands to play a video game.

Entity pool:
Games: The Legend of Zelda, Minecraft, The Witcher 3, Elden Ring, Red Dead Redemption 2,
  Grand Theft Auto V, Call of Duty, FIFA, Cyberpunk 2077, God of War, Spider-Man, Halo,
  Mario Kart, Super Mario Odyssey, The Last of Us, Ghost of Tsushima, Horizon Zero Dawn,
  Dark Souls, Bloodborne, Sekiro, Stardew Valley, Among Us, Fortnite, Valorant,
  League of Legends, Dota 2, Overwatch, Counter-Strike, Apex Legends, Rocket League,
  Final Fantasy, Dragon Age, Mass Effect, Baldur's Gate 3, Diablo IV, World of Warcraft,
  Civilization VI, Crusader Kings III, Portal, Half-Life, Bioshock, Skyrim, Fallout 4,
  Celeste, Hollow Knight, Hades, Dead Cells, Cuphead, It Takes Two, A Way Out
Platforms: PlayStation, Xbox, Nintendo Switch, Steam, PC, Epic Games, Game Pass
Genres: RPG, FPS, strategy, puzzle, platformer, open world, racing, sports, survival,
  simulation, roguelike, fighting, adventure, MMORPG, battle royale
{entity_block}
Coverage:
- by title: "launch Minecraft", "start The Witcher 3", "play Elden Ring"
- by genre: "play an RPG", "I want to play a puzzle game", "open world game"
- by platform: "something on Game Pass", "a PlayStation game", "Steam game"
- vague: "play a game", "I want to game", "find me something to play"
- continuation: "resume my game", "continue where I left off"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"short_film": """\
Generate {n} natural voice commands to watch a short film.

Entity pool:
Providers: Vimeo, YouTube, Mubi, Criterion Channel, Short of the Week, Sundance Now
Genres: drama, comedy, animation, experimental, horror, documentary, thriller, romance, sci-fi
Awards context: Oscar-winning short, Sundance selection, Cannes short, animated short
{entity_block}
Coverage (maximise structural variation — this is a low-data intent):
- generic: "play a short film", "find me a short", "I want something short"
- by genre: "a short horror film", "animated short film", "short comedy"
- by platform: "a short on Vimeo", "short film on YouTube"
- by award/context: "an Oscar-winning short", "a student film", "indie short"
- framing: "I don't have much time, play something short", "something under 20 minutes",
  "quick watch", "a brief film please", "a mini movie"
- rhetorical: "got any short films?", "what short films do you have?"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"silent_movie": """\
Generate {n} natural voice commands to watch a silent film.

Entity pool:
Films: City Lights, The General, Modern Times, The Kid, The Gold Rush, The Circus,
  Nosferatu, Metropolis, The Cabinet of Dr. Caligari, Sunrise: A Song of Two Humans,
  Safety Last!, The Navigator, Sherlock Jr., Steamboat Bill Jr., Our Hospitality,
  The Birth of a Nation, Intolerance, The Wind, Greed, The Big Parade, Wings
Actors: Charlie Chaplin, Buster Keaton, Harold Lloyd, Mary Pickford, Lillian Gish,
  Douglas Fairbanks, Rudolph Valentino, Lon Chaney, Louise Brooks, Emil Jannings
Eras/terms: silent era, pre-talkie, early cinema, 1920s film, 1910s film
{entity_block}
Coverage (maximise variation):
- generic: "play a silent movie", "I want to watch a silent film", "silent cinema"
- by actor: "a Charlie Chaplin film", "Buster Keaton comedy", "Harold Lloyd movie"
- by title: "play Nosferatu", "put on Metropolis", "City Lights please"
- by genre: "silent comedy", "silent horror", "a silent drama"
- by era: "an old silent movie", "something from the 1920s", "early cinema"
- framing: "something without dialogue", "a pre-talkie film", "black and white silent"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"bw_movie": """\
Generate {n} natural voice commands to watch a black-and-white movie.

Entity pool:
Films: Casablanca, Citizen Kane, Some Like It Hot, Sunset Boulevard, The Maltese Falcon,
  Double Indemnity, Laura, Out of the Past, 12 Angry Men, High Noon, The Philadelphia Story,
  It Happened One Night, All About Eve, A Streetcar Named Desire, The African Queen,
  Key Largo, The Treasure of the Sierra Madre, White Heat, Touch of Evil, The Third Man,
  Bicycle Thieves, Rome Open City, 8½, The 400 Blows, Breathless, Seven Samurai,
  Ikiru, Rashomon, Wild Strawberries, The Seventh Seal, Persona
Actors: Humphrey Bogart, Ingrid Bergman, Cary Grant, Katharine Hepburn, Clark Gable,
  Bette Davis, James Cagney, Barbara Stanwyck, Henry Fonda, Spencer Tracy, Jimmy Stewart,
  Fred Astaire, Ginger Rogers, Marilyn Monroe, Audrey Hepburn, Gregory Peck,
  Marlon Brando, James Dean, Grace Kelly, Kim Novak
Genres: film noir, drama, comedy, western, thriller, romance, war film
Eras: classic Hollywood, golden age of cinema, 1940s, 1950s, Italian neorealism, French New Wave
{entity_block}
Coverage:
- by title: "play Casablanca", "put on Citizen Kane", "I want to watch Some Like It Hot"
- by actor: "a Humphrey Bogart film", "something with Cary Grant", "a Bette Davis movie"
- by genre: "a film noir", "classic Hollywood comedy", "a 1940s drama"
- by era: "an old classic film", "black and white from the 50s", "golden age Hollywood"
- generic: "play a black and white movie", "find me a classic", "old movie please"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"asmr": """\
Generate {n} natural voice commands to listen to ASMR content.

Entity pool:
Triggers: tapping, whispering, soft-spoken, crinkling, scratching, page turning,
  hair brushing, ear cleaning, scalp massage, slime sounds, keyboard typing,
  rain sounds, ocean waves, fire crackling, wind chimes, eating sounds (mukbang),
  gloves, glass tapping, wood tapping, metal tapping, sand sounds, water sounds
Purposes: sleep, relaxation, focus, anxiety relief, studying, meditation, tingles
Roleplays: doctor roleplay, spa roleplay, hair salon, library ASMR, bookstore ASMR
Creators: Gibi ASMR, ASMR Darling, Gentle Whispering, SAS-ASMR, Tingting ASMR, Latte
{entity_block}
Coverage (maximise variety — this is a low-data intent):
- by trigger: "tapping ASMR", "whispering ASMR", "rain sounds to sleep to", "page turning sounds"
- by purpose: "ASMR for sleep", "relaxing ASMR", "focus ASMR", "anxiety ASMR"
- by roleplay: "doctor ASMR roleplay", "spa ASMR", "hair salon roleplay"
- by creator: "play Gibi ASMR", "ASMR Darling please"
- generic: "play some ASMR", "I want to relax with ASMR", "ASMR please", "tingles"
- framing: "something relaxing to fall asleep to", "soft sounds please", "calming audio"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"audio_description": """\
Generate {n} natural voice commands to play content with audio description (AD).
Audio description is an additional narration track describing on-screen action for blind/low-vision viewers.

Entity pool:
AD terminology: audio description, audio described, described version, narrated version,
  accessible version, AD track, visual description, commentary track for blind viewers
Content types: movie with audio description, TV show, documentary, sports with commentary,
  Netflix AD, BBC with audio description, Disney+ described version
{entity_block}
Coverage (maximise variety — highly specialised intent):
- explicit request: "play with audio description", "I need the described version",
  "turn on audio description", "enable AD", "the accessible version"
- for a specific title: "Inception with audio description", "play Planet Earth described",
  "The Crown with AD please"
- generic: "find me an audio described movie", "something with narration for the blind",
  "a film with audio description", "accessible content please"
- framing: "I'm visually impaired, enable audio description", "I need audio description on",
  "narrated version please", "described audio track"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"video": """\
Generate {n} natural voice commands to watch a generic video.

Entity pool:
Channels/creators: MrBeast, PewDiePie, Markiplier, Linus Tech Tips, Veritasium, Vsauce,
  Kurzgesagt, CGP Grey, Tom Scott, 3Blue1Brown, Mark Rober, Wendover Productions,
  Bon Appétit, Babish Culinary Universe, Binging with Babish, Tasty, Joshua Weissman,
  Pewdiepie, Ninja, Pokimane, Dream, Technoblade, SSSniperWolf, Jacksepticeye
Platforms: YouTube, TikTok, Vimeo, Twitch, Dailymotion, Rumble
Topics: gaming, cooking, travel, comedy sketches, tutorials, unboxing, vlogs,
  tech reviews, music videos, sports highlights, fails compilation, reaction videos,
  DIY, science experiments, fashion, beauty, fitness, language learning
{entity_block}
Coverage:
- by channel: "play MrBeast", "put on Veritasium", "I want Kurzgesagt"
- by topic: "gaming video", "cooking tutorial", "travel vlog", "comedy video"
- by platform: "a YouTube video", "something on Twitch", "TikTok videos"
- by type: "unboxing video", "a compilation", "music video", "reaction video"
- vague: "play a video", "show me something on YouTube", "find me a video", "something on YouTube"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"music_video": """\
Generate {n} natural voice commands to watch an official music video for a song or artist.

Entity pool:
Songs with iconic videos: Thriller, Bad, Billie Jean, Beat It (Michael Jackson),
  Bohemian Rhapsody (Queen), November Rain, Welcome to the Jungle (Guns N' Roses),
  Smells Like Teen Spirit, Heart-Shaped Box (Nirvana), Hurt (Nine Inch Nails / Johnny Cash),
  Virtual Insanity (Jamiroquai), Bitter Sweet Symphony (The Verve),
  Take On Me (a-ha), Total Eclipse of the Heart (Bonnie Tyler),
  99 Red Balloons (Nena), Come As You Are (Nirvana),
  Blinding Lights, Save Your Tears (The Weeknd),
  Bad Guy (Billie Eilish), Old Town Road (Lil Nas X),
  Telephone (Lady Gaga ft. Beyoncé), Single Ladies (Beyoncé),
  Happy (Pharrell), Uptown Funk (Mark Ronson ft. Bruno Mars),
  God's Plan, Hotline Bling (Drake), HUMBLE. (Kendrick Lamar),
  WAP (Cardi B), Shape of You (Ed Sheeran), Rolling in the Deep (Adele)
Artists: Taylor Swift, Beyoncé, Rihanna, Lady Gaga, Katy Perry, Ariana Grande,
  Dua Lipa, Harry Styles, The Weeknd, Post Malone, BTS, Blackpink, Twice,
  Eminem, Jay-Z, Kanye West, Childish Gambino, Kendrick Lamar
{entity_block}
Coverage (maximise variety — ZERO existing training data):
- by song + "video": "play the music video for Bohemian Rhapsody", "show me the Thriller video"
- by artist + "video": "Beyoncé music videos", "put on some Taylor Swift videos"
- by song alone (video context clear): "play Blinding Lights video", "show Bad Guy by Billie Eilish"
- with provider: "Beyoncé music video on YouTube", "official video for Shape of You"
- vague: "play a music video", "put on some music videos", "show me music videos"
- rhetorical: "got any music videos?", "what music videos do you have?"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"trailer": """\
Generate {n} natural voice commands to watch a movie or TV show trailer, teaser, or preview.

Entity pool:
Films: Dune, Oppenheimer, Barbie, The Batman, Top Gun: Maverick, Spider-Man: No Way Home,
  Avengers: Endgame, Black Panther: Wakanda Forever, Thor: Love and Thunder,
  Doctor Strange in the Multiverse of Madness, Mission: Impossible – Dead Reckoning,
  Indiana Jones and the Dial of Destiny, John Wick: Chapter 4, Fast X,
  Transformers: Rise of the Beasts, Aquaman and the Lost Kingdom,
  The Marvels, Wonka, Poor Things, Killers of the Flower Moon, Napoleon
TV Shows: House of the Dragon, The Last of Us, Andor, Rings of Power, Stranger Things 5,
  The Crown, Wednesday, Emily in Paris, Squid Game 2, Yellowstone, 1923, Tulsa King
Trailer types: trailer, teaser, official trailer, extended trailer, final trailer,
  international trailer, Super Bowl spot, theatrical trailer
{entity_block}
Coverage (ZERO existing training data — maximise structural variety):
- explicit "trailer": "show the trailer for Dune", "play the Oppenheimer trailer",
  "I want to see the Top Gun trailer"
- "teaser": "play the teaser for Avengers", "show me the teaser for The Last of Us"
- "preview": "show the preview for Barbie", "I want to see a preview"
- specific trailer type: "the official trailer for Dune", "the final trailer for Endgame"
- by show/film alone (trailer context): "Stranger Things 5 trailer", "The Batman teaser"
- vague: "play a trailer", "show me some trailers", "what trailers are available"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"behind_the_scenes": """\
Generate {n} natural voice commands to watch behind-the-scenes, making-of,
featurette, or cast/crew interview content for a film or TV series.

Entity pool:
Films: Dune, The Lord of the Rings, Harry Potter, Star Wars, The Dark Knight,
  Avengers: Endgame, Mad Max: Fury Road, Interstellar, Inception, Parasite,
  Everything Everywhere All at Once, Get Out, Hereditary, Oppenheimer
TV Shows: Game of Thrones, Breaking Bad, Stranger Things, The Mandalorian,
  House of the Dragon, The Last of Us, Succession, Peaky Blinders
BTS content types: making of, behind the scenes, featurette, cast interview,
  director's cut, production diary, set visit, bloopers, gag reel, deleted scenes,
  VFX breakdown, stunt choreography, costume design, production design
{entity_block}
Coverage (ZERO existing training data — maximise structural variety):
- "making of": "show the making of Dune", "play the making of Interstellar",
  "I want to watch how they made The Lord of the Rings"
- "behind the scenes": "behind the scenes of Game of Thrones",
  "show me behind the scenes footage from The Dark Knight"
- "featurette": "play the featurette for Mad Max", "show the Stranger Things featurette"
- cast/crew: "watch the cast interviews for Succession", "director's commentary for Inception"
- bloopers: "show me the bloopers from The Office", "play the gag reel for Friends"
- VFX / production: "how did they make the VFX in Endgame", "stunt breakdown for John Wick"
- vague: "show me some behind the scenes content", "making-of videos please"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"audio": """\
Generate {n} natural voice commands to play generic ambient or background audio
(NOT music, podcasts, or radio — sounds, noise, atmosphere, functional audio).

Entity pool:
Types: white noise, brown noise, pink noise, binaural beats, isochronic tones,
  rain sounds, thunderstorm, ocean waves, forest sounds, fire crackling, wind,
  river stream, birds chirping, crickets, waterfall, gentle stream,
  coffee shop ambience, city street sounds, library sounds, train sounds,
  guided meditation, breathing exercises, body scan meditation, sleep hypnosis,
  sleep story, power nap audio, 432 Hz, 528 Hz, solfeggio frequencies,
  deep focus sounds, lo-fi study atmosphere, womb sounds, fan noise
Purposes: sleep, focus, relax, meditate, study, mask tinnitus, anxiety relief
{entity_block}
Coverage (maximise variety — low-data intent):
- by type: "white noise please", "play rain sounds", "brown noise", "ocean wave sounds"
- by purpose: "something to help me sleep", "focus sounds", "meditation audio", "relaxing sounds"
- ambient: "coffee shop ambience", "fire crackling sounds", "forest atmosphere"
- functional: "binaural beats for studying", "432 Hz audio", "solfeggio tones"
- generic: "play some ambient sounds", "background noise please", "put on some sounds",
  "I want some calming audio", "play something relaxing"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"visual_story": """\
Generate {n} natural voice commands to read a comic book, graphic novel, or manga.

Entity pool:
Titles — Western: Batman, Superman, Spider-Man, X-Men, The Avengers, Iron Man, Captain America,
  Thor, Guardians of the Galaxy, Wonder Woman, The Flash, Aquaman, Black Panther,
  Watchmen, The Dark Knight Returns, Preacher, Transmetropolitan, Y: The Last Man,
  Saga, Sandman, Hellboy, Spawn, Fables, Invincible, The Boys, From Hell,
  V for Vendetta, Maus, Persepolis, Fun Home, Jimmy Corrigan, Love and Rockets
Titles — Manga: Berserk, Vagabond, Vinland Saga, One Piece, Naruto, Dragon Ball,
  Fullmetal Alchemist, Akira, Ghost in the Shell, Neon Genesis Evangelion,
  Blade of the Immortal, Lone Wolf and Cub, JoJo's Bizarre Adventure, Oyasumi Punpun,
  A Silent Voice, Your Lie in April, Tokyo Ghoul, Chainsaw Man, Jujutsu Kaisen, Spy x Family
Publishers: Marvel, DC Comics, Image Comics, Dark Horse, IDW, BOOM! Studios, Valiant,
  Viz Media, Kodansha, Shueisha, Drawn & Quarterly, Fantagraphics
Genres: superhero, horror, sci-fi, fantasy, crime noir, biography, romance,
  slice of life, action, political, literary, shonen, shojo, seinen, josei
{entity_block}
Coverage:
- by title: "read me Watchmen", "play Saga", "I want to read Berserk", "open Maus"
- by character: "a Batman comic", "Spider-Man", "X-Men graphic novel"
- by publisher: "something from Image Comics", "a Marvel comic", "DC graphic novel"
- by genre/medium: "superhero comic", "manga please", "a graphic novel", "read me a comic"
- by type: "a horror graphic novel", "slice of life manga", "a classic comic"
- vague: "read me a comic", "comic book please", "I want to read a graphic novel"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"adult": """\
Generate {n} natural voice commands to watch adult/pornographic video content.
Keep language direct and clinical — these are real voice assistant training samples.

Entity pool:
Providers: Pornhub, xVideos, RedTube, YouPorn, xHamster, OnlyFans, AVN, Brazzers, Bang Bros
Genres: heterosexual, lesbian, gay, BDSM, fetish, amateur, compilation, hentai, cosplay
{entity_block}
Coverage:
- by provider: "open Pornhub", "play something on xVideos", "OnlyFans content"
- by genre: "play adult content", "lesbian porn", "gay porn", "amateur video"
- vague: "play some adult content", "adult video please", "I want to watch porn",
  "play something explicit", "adult stuff"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"adult_audio": """\
Generate {n} natural voice commands to listen to adult audio content or erotic audio.
Keep language direct and clinical.

Entity pool:
Types: erotic audiobook, erotic fiction audio, erotic ASMR, JOI audio, dirty talk audio,
  sexual fantasy audio, erotic short story, adult podcast, erotic roleplay audio
Providers: Dipsea, Emjoy, Ferly, Coral, Audible (erotic section), Patreon audio
{entity_block}
Coverage (maximise variety — very low data intent):
- by type: "erotic ASMR", "dirty talk audio", "adult audio please", "erotic fiction"
- by provider: "play Dipsea", "something on Emjoy", "adult audio from Patreon"
- generic: "adult audio content", "erotic audio please", "play something sexy to listen to",
  "I want to hear adult content", "play erotic audio"
- framing: "something sexy to listen to", "adult audiobook", "erotic story please"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"hentai": """\
Generate {n} natural voice commands to watch hentai (adult Japanese animation).
Keep language direct and clinical.

Entity pool:
Well-known titles: Bible Black, La Blue Girl, Urotsukidoji, Demon Beast Invasion,
  Bondage Fairies, Words Worth, Kite, Mezzo Forte, Sex Demon Queen, Ogenki Clinic
Genres: vanilla, tentacle, magical girl, romance, school, fantasy, monster
Providers: Hanime, Fakku, MangaGamer, Nutaku
{entity_block}
Coverage:
- generic: "play some hentai", "I want to watch hentai", "hentai please"
- by genre: "vanilla hentai", "romance hentai", "fantasy hentai"
- by provider: "something on Hanime", "play from Fakku"
- by title: "play Bible Black", "I want to watch Kite"

Seed examples:
{seeds}

Output {n} utterances, one per line:""",

"not_ocp": """\
Generate {n} voice assistant queries about media or general topics that are NOT playback requests.
The classifier must NOT trigger media playback for these.

CRITICAL distinction:
  PLAY request (wrong class): "play Inception", "put on Breaking Bad", "stream some jazz"
  NOT-OCP (correct class):   "who directed Inception?", "how many seasons does Breaking Bad have?",
                               "what genre is jazz?", "is Spotify down?"

Generate a mix of:
1. Media information queries (NOT play):
   - "who directed The Godfather"
   - "what year did Bohemian Rhapsody come out"
   - "how many episodes does Succession have"
   - "is there a sequel to Dune"
   - "who plays Walter White"
   - "what genre is Radiohead"
   - "best movies of 2023"
   - "recommend me a good podcast"

2. Smart-home / device control:
   - "turn off the lights"
   - "set the thermostat to 22"
   - "lock the front door"
   - "is the washing machine done"

3. Reminders / calendar / timers:
   - "set an alarm for 7am"
   - "remind me to call mom at noon"
   - "what's on my calendar tomorrow"
   - "start a 10 minute timer"

4. General knowledge / web queries:
   - "what's the weather like"
   - "how tall is the Eiffel Tower"
   - "translate hello into French"
   - "what time is it in Tokyo"

5. Shopping / other assistant tasks:
   - "add milk to my shopping list"
   - "order more coffee"
   - "call a taxi"

{entity_block}
Seed examples:
{seeds}

Output {n} utterances, one per line:""",
}

# ---------------------------------------------------------------------------
# Static fallback entity pools (used when Wikidata has no data for an intent)
# ---------------------------------------------------------------------------

_FALLBACK_ENTITIES: dict[str, list[str]] = {
    "tv": [
        "BBC One", "BBC Two", "BBC Four", "ITV", "Channel 4", "Channel 5",
        "Sky One", "Sky Atlantic", "Dave", "E4",
        "CNN", "Fox News", "MSNBC", "NBC", "CBS", "ABC",
        "ESPN", "ESPN 2", "TNT", "TBS", "FX", "Comedy Central",
        "Cartoon Network", "Disney Channel", "Nickelodeon", "MTV", "VH1",
        "Sky Sports", "BT Sport", "beIN Sports", "Eurosport",
        "NFL Network", "NBA TV", "MLB Network",
        "BBC News Channel", "Sky News", "Al Jazeera", "France 24",
        "Deutsche Welle TV", "Bloomberg TV", "CNBC",
        "RAI Uno", "TVE", "ARD", "ZDF", "France 2", "RTL", "M6",
        "Pluto TV", "Plex TV", "Samsung TV Plus", "Tubi live",
    ],
    "music_video": [
        "Thriller music video", "Bad Guy music video", "Blinding Lights video",
        "Bohemian Rhapsody video", "Smells Like Teen Spirit video",
        "Take On Me music video", "Single Ladies music video",
        "Happy Pharrell video", "God's Plan Drake video", "WAP music video",
        "Virtual Insanity music video", "Hurt Johnny Cash video",
        "Old Town Road music video", "Shape of You video",
        "Rolling in the Deep video", "Telephone Lady Gaga video",
        "Uptown Funk video", "HUMBLE. Kendrick Lamar video",
        "Taylor Swift music video", "Beyoncé music video", "BTS music video",
        "Blackpink music video", "Dua Lipa music video", "Harry Styles video",
    ],
    "trailer": [
        "Dune trailer", "Oppenheimer trailer", "Barbie trailer",
        "The Batman trailer", "Top Gun Maverick trailer",
        "Avengers Endgame trailer", "Spider-Man No Way Home trailer",
        "The Last of Us trailer", "House of the Dragon teaser",
        "Stranger Things 5 trailer", "Rings of Power trailer",
        "Andor trailer", "John Wick 4 trailer", "Mission Impossible trailer",
        "Indiana Jones Dial of Destiny trailer",
        "official movie trailer", "movie teaser", "film preview",
        "extended trailer", "final trailer", "Super Bowl movie spot",
        "theatrical trailer", "international trailer",
    ],
    "behind_the_scenes": [
        "making of Dune", "making of The Lord of the Rings",
        "making of Harry Potter", "making of Star Wars",
        "making of The Dark Knight", "making of Avengers Endgame",
        "behind the scenes Game of Thrones", "behind the scenes Breaking Bad",
        "behind the scenes Stranger Things", "behind the scenes The Mandalorian",
        "Dune featurette", "Succession featurette", "Peaky Blinders featurette",
        "cast interview Succession", "director commentary Inception",
        "production diary The Last of Us", "set visit Stranger Things",
        "bloopers The Office", "gag reel Friends", "deleted scenes",
        "VFX breakdown Avengers", "stunt breakdown John Wick",
        "costume design Mad Max", "production design Dune",
    ],
    "video_episodes": [
        "Linus Tech Tips", "MrBeast", "Markiplier", "Jacksepticeye",
        "Veritasium", "Vsauce", "Kurzgesagt", "CGP Grey", "Tom Scott",
        "3Blue1Brown", "Mark Rober", "Wendover Productions",
        "Binging with Babish", "Joshua Weissman", "Bon Appétit", "Tasty",
        "SciShow", "CrashCourse", "TED-Ed", "PBS Space Time",
        "PewDiePie", "Ninja", "Dream", "Pokimane", "xQc", "Ludwig",
        "Hasan Piker", "Philosophy Tube", "Contrapoints", "H.Bomberguy",
        "Curiosity Stream series", "Nebula original", "Dropout show",
    ],
    "asmr": [
        "tapping ASMR", "whispering ASMR", "crinkling ASMR", "scratching ASMR",
        "rain sounds ASMR", "ocean waves ASMR", "fire crackling ASMR", "wind sounds ASMR",
        "ear cleaning ASMR", "scalp massage ASMR", "hair brushing ASMR", "page turning ASMR",
        "keyboard typing ASMR", "sand sounds ASMR", "mukbang ASMR", "eating sounds ASMR",
        "glass tapping ASMR", "wood tapping ASMR", "metal tapping ASMR",
        "soft-spoken ASMR", "sleep ASMR", "relaxing ASMR", "focus ASMR",
        "doctor roleplay ASMR", "spa roleplay ASMR", "hair salon ASMR", "library ASMR",
        "Gibi ASMR", "ASMR Darling", "Gentle Whispering ASMR", "SAS-ASMR",
        "Tingting ASMR", "Latte ASMR", "WhispersRed ASMR", "ASMR Glow",
    ],
    "audio": [
        "white noise", "brown noise", "pink noise", "binaural beats", "isochronic tones",
        "rain sounds", "thunderstorm sounds", "ocean waves", "forest sounds", "fire crackling",
        "river stream", "birds chirping", "crickets at night", "waterfall sounds",
        "coffee shop ambience", "library ambience", "busy café sounds", "city street sounds",
        "guided meditation", "breathing exercise", "body scan meditation", "sleep hypnosis",
        "sleep story", "power nap audio", "432 Hz frequency", "528 Hz solfeggio",
        "deep focus sounds", "lo-fi study atmosphere", "fan noise", "womb sounds",
        "delta waves", "theta waves", "alpha waves", "gamma waves",
    ],
    "audio_description": [
        "audio described movie", "audio described TV show", "audio description track",
        "described version", "AD version", "accessible version", "narrated version",
        "visually impaired friendly content", "blind-accessible movie",
        "audio description service", "described audio",
    ],
    "visual_story": [
        "Batman", "Superman", "Spider-Man", "X-Men", "Avengers", "Iron Man",
        "Captain America", "Thor", "Wonder Woman", "The Flash", "Black Panther",
        "Watchmen", "The Dark Knight Returns", "Saga", "Sandman", "Preacher",
        "Y: The Last Man", "Hellboy", "Invincible", "The Boys", "From Hell",
        "V for Vendetta", "Maus", "Persepolis", "Fun Home", "Fables",
        "Berserk", "Vagabond", "Vinland Saga", "Akira", "Ghost in the Shell",
        "Lone Wolf and Cub", "JoJo's Bizarre Adventure", "Oyasumi Punpun",
        "Marvel Comics", "DC Comics", "Image Comics", "Dark Horse Comics",
        "Viz Media manga", "shonen manga", "seinen manga", "shojo manga",
    ],
    "radio_theatre": [
        "The Hitchhiker's Guide to the Galaxy BBC Radio",
        "Cabin Pressure", "The Goon Show", "Just a Minute", "The Archers",
        "Sherlock Holmes BBC Radio", "War of the Worlds Orson Welles",
        "Sorry Wrong Number", "Suspense old time radio", "The Shadow",
        "The Lone Ranger radio", "Gunsmoke radio", "Dragnet radio",
        "Jack Benny Program", "Fibber McGee and Molly", "Escape radio",
        "Dimension X", "X Minus One", "Inner Sanctum Mysteries",
        "BBC Radio 4 drama", "BBC Sounds audio drama", "Audible originals",
    ],
    "adult_audio": [
        "erotic ASMR", "dirty talk audio", "erotic audiobook", "erotic fiction audio",
        "adult audio story", "JOI audio", "sexual fantasy audio", "erotic roleplay audio",
        "Dipsea app", "Emjoy app", "Ferly app", "Coral app",
        "erotic short story audio", "sensual audio", "romantic audio",
    ],
    "short_film": [
        "short film on Vimeo", "short film on YouTube", "Sundance short film",
        "Oscar-winning short film", "animated short", "live-action short",
        "student short film", "indie short film", "short drama", "short comedy",
        "short horror film", "experimental short", "documentary short",
        "Cannes short film", "BAFTA short", "Pixar short", "Aardman short",
    ],
    "silent_movie": [
        "City Lights", "The General", "Modern Times", "The Kid", "The Gold Rush",
        "Nosferatu", "Metropolis", "The Cabinet of Dr. Caligari", "Safety Last",
        "The Navigator", "Sherlock Jr.", "Sunrise: A Song of Two Humans",
        "Charlie Chaplin", "Buster Keaton", "Harold Lloyd", "Mary Pickford",
        "Lillian Gish", "Douglas Fairbanks", "Lon Chaney",
        "silent comedy", "silent horror", "silent drama", "expressionist cinema",
        "pre-talkie film", "early cinema", "1920s movie",
    ],
    "bw_movie": [
        "Casablanca", "Citizen Kane", "Some Like It Hot", "Sunset Boulevard",
        "The Maltese Falcon", "Double Indemnity", "12 Angry Men", "High Noon",
        "The African Queen", "Key Largo", "Touch of Evil", "The Third Man",
        "Bicycle Thieves", "Seven Samurai", "Rashomon", "Wild Strawberries", "The 400 Blows",
        "Humphrey Bogart", "Ingrid Bergman", "Cary Grant", "Katharine Hepburn",
        "Bette Davis", "James Cagney", "Henry Fonda", "Marilyn Monroe", "Gregory Peck",
        "film noir", "classic Hollywood", "golden age of cinema", "1940s film", "1950s drama",
    ],
}

# ---------------------------------------------------------------------------
# Entity pools — loaded lazily on first use
# ---------------------------------------------------------------------------

_ENTITY_POOL_CACHE: dict[str, list[str]] = {}


def _get_entity_pool(intent: str, hf_cache: str) -> list[str]:
    """Return a list of real-world entity strings relevant to this intent."""
    if intent in _ENTITY_POOL_CACHE:
        return _ENTITY_POOL_CACHE[intent]

    # Map intent → entity_types to fetch from WikidataMediaEntities
    _INTENT_ENTITY_TYPES: dict[str, list[str]] = {
        "music":              ["artist_name", "song_name", "album_name", "music_genre"],
        "movie":              ["movie_name", "movie_actor", "movie_director", "film_genre"],
        "tv":                 ["tv_channel"],           # live IPTV — channel names only
        "tv_show":            ["series_name"],           # episodic series — show titles
        "video_episodes":     ["youtube_channel"],       # online creators / channels
        "music_video":        ["music_video_name", "artist_name", "song_name"],
        "trailer":            ["movie_name", "trailer_name"],
        "behind_the_scenes":  ["movie_name", "series_name"],
        "anime":              ["anime_name"],
        "cartoon":            ["cartoon_name"],
        "documentary":        ["documentary_name"],
        "podcast":            ["podcast_name", "podcaster"],
        "radio":              ["radio_streaming_service"],
        "radio_theatre":      ["radio_drama_name"],
        "audiobook":          ["book_name", "book_author"],
        "news":               ["news_provider"],
        "game":               ["game_name", "game_genre", "gaming_console_name"],
        "short_film":         ["short_film_name"],
        "silent_movie":       ["silent_movie_name"],
        "bw_movie":           ["bw_movie_name"],
        "asmr":               [],
        "audio_description":  [],
        "video":              ["youtube_channel", "tv_channel"],
        "audio":              [],
        "visual_story":       [],
        "adult":              ["pornstar_name", "porn_genre"],
        "adult_audio":        [],
        "hentai":             ["hentai_name"],
        "not_ocp":            ["movie_name", "series_name", "artist_name"],
    }

    entity_types = _INTENT_ENTITY_TYPES.get(intent, [])
    if not entity_types:
        _ENTITY_POOL_CACHE[intent] = []
        return []

    try:
        from datasets import load_dataset
        ds = load_dataset("Jarbas/WikidataMediaEntities", split="train", cache_dir=hf_cache)
    except Exception as e:
        logger.warning(f"Could not load WikidataMediaEntities: {e}")
        _ENTITY_POOL_CACHE[intent] = []
        return []

    pool: list[str] = []
    for row in ds:
        if row["entity_type"] in entity_types:
            text = (row.get("text") or "").strip()
            if text and len(text) > 1:
                pool.append(text)

    random.shuffle(pool)
    pool = pool[:5000]  # cap per intent to avoid slow prompts
    _ENTITY_POOL_CACHE[intent] = pool
    logger.info(f"Loaded {len(pool)} entities for intent={intent}")
    return pool


def _build_entity_block(intent: str, hf_cache: str, n_entities: int = 15) -> str:
    """Build an 'Additional entities' block to inject into the prompt.

    Uses Wikidata when available, falls back to _FALLBACK_ENTITIES for
    intents with no Wikidata coverage (asmr, audio, audio_description, etc.).
    """
    pool = _get_entity_pool(intent, hf_cache)
    if not pool:
        pool = _FALLBACK_ENTITIES.get(intent, [])
    if not pool:
        return ""
    sample = random.sample(pool, min(n_entities, len(pool)))
    return "Additional entities to use naturally (mix into your utterances):\n" + \
           "\n".join(f"  {e}" for e in sample) + "\n"


# ---------------------------------------------------------------------------
# API + parsing
# ---------------------------------------------------------------------------

def _call_api(url: str, model: str, system: str, user: str,
              temperature: float, max_tokens: int, timeout: int) -> Optional[str]:
    endpoint = url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        logger.warning("Request timed out")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        logger.warning(f"Unexpected response format: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP error: {e}")
        return None


def _parse_sentences(text: str) -> list[str]:
    """Extract one utterance per line, strip markers and quotes."""
    results = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Strip bullets / numbering
        for pfx in ("- ", "* ", "• "):
            if line.startswith(pfx):
                line = line[len(pfx):]
                break
        if len(line) > 2 and line[0].isdigit() and line[1] in (".", ")"):
            line = line[2:].lstrip()
        elif len(line) > 3 and line[:2].isdigit() and line[2] in (".", ")"):
            line = line[3:].lstrip()
        line = line.strip('"\'').strip()
        if line and len(line) > 3:
            results.append(line.lower())
    return results


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(csv_path: str) -> dict[str, list[str]]:
    """Return {intent: [sentence, ...]} index from a dataset CSV."""
    index: dict[str, list[str]] = {}
    if not os.path.exists(csv_path):
        return index
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            intent = row.get("intent", "").strip()
            sent   = row.get("sentence", "").strip()
            if intent and sent:
                index.setdefault(intent, []).append(sent)
    return index


def _load_already_generated(csv_path: str) -> dict[str, set[str]]:
    """Return {intent: set_of_sentences} already written to the output CSV."""
    result: dict[str, set[str]] = {}
    if not os.path.exists(csv_path):
        return result
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            intent = row.get("intent", "").strip()
            sent   = row.get("sentence", "").strip()
            if intent and sent:
                result.setdefault(intent, set()).add(sent)
    return result


def _append_rows(csv_path: str, rows: list[dict], write_header: bool) -> None:
    fieldnames = ["lang", "domain", "intent", "binary_label", "playback_label",
                  "media_label", "sentence", "model"]
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Batch worker (runs in a thread — stateless except for item dict)
# ---------------------------------------------------------------------------

from ovos_media_classifier.train.sources import AUDIO_INTENTS as _AUDIO_INTENTS, VIDEO_INTENTS as _VIDEO_INTENTS


def _domain_for(intent: str) -> str:
    return "not_ocp" if intent == "not_ocp" else "ocp_play"


def _binary_label_for(intent: str) -> str:
    return "not_ocp" if intent == "not_ocp" else "ocp"


def _playback_label_for(intent: str) -> str:
    if intent == "not_ocp":
        return "undefined"
    if intent in _AUDIO_INTENTS:
        return "audio"
    if intent in _VIDEO_INTENTS:
        return "video"
    return "undefined"


def _media_label_for(intent: str) -> str:
    return "not_ocp" if intent == "not_ocp" else intent


def _do_batch(item: dict, worker: dict, hf_cache: str,
              sentence_index: dict[str, list[str]], dry_run: bool) -> tuple[list[str], str]:
    rng        = random.Random()
    intent     = item["intent"]
    batch      = min(item["remaining"], BATCH_SIZE)
    model_name = worker["model"]

    # Sample seed examples from the real dataset pool
    pool  = sentence_index.get(intent, item["examples"])
    seeds = rng.sample(pool, min(NUM_SEED_EXAMPLES, len(pool))) if pool else []
    seed_block = "\n".join(f"  - {s}" for s in seeds) if seeds else "  (none available)"

    entity_block = _build_entity_block(intent, hf_cache)
    temperature  = rng.uniform(TEMPERATURE_MIN, TEMPERATURE_MAX)

    system = _INTENT_SYSTEM.get(intent, _SYSTEM_BASE)
    user_tmpl = _INTENT_USER.get(intent)
    if user_tmpl is None:
        logger.warning(f"No prompt defined for intent={intent!r} — skipping")
        return [], model_name

    user = user_tmpl.format(n=batch, seeds=seed_block, entity_block=entity_block)

    logger.info(
        f"[{worker['_id']}] intent={intent}  batch={batch}  "
        f"remaining={item['remaining']}  temp={temperature:.3f}"
    )

    if dry_run:
        logger.info(f"[{worker['_id']}] [dry-run] System:\n{system}\n\nUser:\n{user}\n")
        return [f"<dry-run {intent} {j+1}>" for j in range(batch)], model_name

    text = _call_api(
        url=worker["url"], model=model_name,
        system=system, user=user,
        temperature=temperature, max_tokens=MAX_TOKENS,
        timeout=REQUEST_TIMEOUT,
    )
    if text is None:
        return [], model_name

    sentences  = _parse_sentences(text)
    seed_set   = {s.lower() for s in seeds} | item["generated_this_run"]
    seen: set[str] = set()
    new: list[str] = []
    for s in sentences:
        if s not in seed_set and s not in seen:
            new.append(s)
            seen.add(s)
    return new, model_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    workers:    list[dict],
    dataset:    str,
    output:     str,
    intents:    list[str],
    target:     int,
    dry_run:    bool,
    negatives:  bool,
    temp_min:   float,
    temp_max:   float,
    hf_cache:   str,
) -> None:
    global TEMPERATURE_MIN, TEMPERATURE_MAX
    TEMPERATURE_MIN, TEMPERATURE_MAX = temp_min, temp_max

    logger.info(
        f"Workers: {len(workers)}  —  "
        + "  |  ".join(f"[{w['_id']}] {w['model']} @ {w['url']}" for w in workers)
    )

    if negatives and "not_ocp" not in intents:
        intents = intents + ["not_ocp"]

    # Load seed dataset
    sentence_index = _load_dataset(dataset)
    logger.info(
        f"Seed dataset: {sum(len(v) for v in sentence_index.values()):,} sentences "
        f"across {len(sentence_index)} intents  ({dataset})"
    )

    # Resume: load already-generated sentences
    already_generated = _load_already_generated(output)
    total_already = sum(len(v) for v in already_generated.values())
    if total_already:
        logger.info(
            f"Resuming: {total_already:,} sentences already written "
            f"for {len(already_generated)} intents"
        )

    # Build work queue
    work: list[dict] = []
    skipped = 0
    for intent in intents:
        existing_count = len(sentence_index.get(intent, [])) + \
                         len(already_generated.get(intent, set()))
        needed = max(0, target - existing_count)
        already_set = already_generated.get(intent, set())
        # Account for already-generated in this output file
        remaining = max(0, target - len(sentence_index.get(intent, [])) - len(already_set))
        if remaining == 0:
            skipped += 1
            logger.info(f"  {intent}: already at target ({existing_count} ≥ {target}) — skip")
            continue
        work.append({
            "intent":             intent,
            "remaining":          remaining,
            "examples":           sentence_index.get(intent, [])[:50],
            "generated_this_run": set(already_set),
            "failures":           0,
        })
        logger.info(
            f"  {intent}: {existing_count} existing → need {remaining} more"
        )

    logger.info(f"Work queue: {len(work)} intents  ({skipped} already at target)")
    if not work:
        logger.info("Nothing to do.")
        return

    # Sort: most under-represented first
    rng = random.Random()
    rng.shuffle(work)
    work.sort(key=lambda x: x["remaining"], reverse=True)

    need_header   = not os.path.exists(output)
    total_written = 0
    total_failed  = 0
    round_num     = 0

    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        while work:
            round_num += 1
            logger.info(f"--- Round {round_num}: {len(work)} intents ---")

            futures = {
                executor.submit(_do_batch, item, workers[i % len(workers)],
                                hf_cache, sentence_index, dry_run): item
                for i, item in enumerate(work)
            }

            still_working: list[dict] = []
            for future in as_completed(futures):
                item   = futures[future]
                intent = item["intent"]
                new_sents, model_name = future.result()

                if not new_sents:
                    item["failures"] += 1
                    if item["failures"] >= MAX_RETRIES:
                        logger.warning(f"Giving up on {intent} after {MAX_RETRIES} failures")
                        total_failed += 1
                    else:
                        logger.warning(f"No sentences for {intent}; retry ({item['failures']}/{MAX_RETRIES})")
                        still_working.append(item)
                    continue

                item["failures"] = 0
                item["generated_this_run"].update(new_sents)

                rows = [{"lang": "en", "domain": _domain_for(intent),
                         "intent": intent,
                         "binary_label": _binary_label_for(intent),
                         "playback_label": _playback_label_for(intent),
                         "media_label": _media_label_for(intent),
                         "sentence": s, "model": model_name}
                        for s in new_sents]
                _append_rows(output, rows, write_header=need_header)
                need_header = False
                total_written += len(rows)
                item["remaining"] = max(0, item["remaining"] - len(new_sents))

                logger.info(
                    f"  {intent}  wrote={len(new_sents)}  "
                    f"remaining={item['remaining']}  total={total_written:,}"
                )
                if item["remaining"] > 0:
                    still_working.append(item)

            work = still_working
            rng.shuffle(work)
            work.sort(key=lambda x: x["remaining"], reverse=True)

    logger.info("=" * 60)
    logger.info(
        f"Done.  Written: {total_written:,}  "
        f"Skipped: {skipped}  Failed: {total_failed}"
    )
    logger.info(f"Output → {output}")


if __name__ == "__main__":
    out_dir = get_output_dir()

    parser = argparse.ArgumentParser(
        description="Augment low-data OCP media intents via a local LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _default_url   = DEFAULT_WORKERS[0]["url"]
    _default_model = DEFAULT_WORKERS[0]["model"]
    parser.add_argument(
        "--url", default=_default_url,
        help=f"Single-host URL used with --workers N (default: {_default_url})",
    )
    parser.add_argument(
        "--model", default=_default_model,
        help=f"Model for --url (default: {_default_model})",
    )
    parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Spawn N workers all pointing at --url/--model. "
             "Mutually exclusive with --worker.",
    )
    parser.add_argument(
        "--worker", nargs=2, metavar=("URL", "MODEL"), action="append",
        help="Add a worker with explicit URL+model. Repeatable. "
             "If neither --worker nor --workers given, uses DEFAULT_WORKERS list.",
    )
    parser.add_argument(
        "--intents", nargs="+", metavar="INTENT",
        default=LOW_DATA_INTENTS,
        help="Intents to augment (default: all low-data intents)",
    )
    parser.add_argument(
        "--negatives", action="store_true",
        help="Also generate not_ocp negative examples",
    )
    parser.add_argument(
        "--target", type=int, default=DEFAULT_TARGET,
        help=f"Target sentence count per intent (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--batch", type=int, default=BATCH_SIZE,
        help=f"Utterances per API call (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--dataset", metavar="CSV",
        default=os.path.join(out_dir, "balanced_dataset.csv"),
        help="Seed dataset CSV (default: balanced_dataset.csv)",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        default=os.path.join(out_dir, "llm_augmented.csv"),
        help="Output CSV path (default: llm_augmented.csv in output dir)",
    )
    parser.add_argument(
        "--temp-min", type=float, default=TEMPERATURE_MIN,
        help=f"Temperature lower bound (default: {TEMPERATURE_MIN})",
    )
    parser.add_argument(
        "--temp-max", type=float, default=TEMPERATURE_MAX,
        help=f"Temperature upper bound (default: {TEMPERATURE_MAX})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without calling the API",
    )
    args = parser.parse_args()

    BATCH_SIZE = args.batch

    if args.worker:
        worker_list = [{"url": u, "model": m, "_id": f"w{i}"}
                       for i, (u, m) in enumerate(args.worker)]
    elif args.workers is not None:
        worker_list = [{"url": args.url, "model": args.model, "_id": f"w{i}"}
                       for i in range(args.workers)]
    else:
        worker_list = [{"url": w["url"], "model": w["model"], "_id": f"w{i}"}
                       for i, w in enumerate(DEFAULT_WORKERS)]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    main(
        workers=worker_list,
        dataset=args.dataset,
        output=args.output,
        intents=args.intents,
        target=args.target,
        dry_run=args.dry_run,
        negatives=args.negatives,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        hf_cache=get_hf_cache_dir(),
    )
