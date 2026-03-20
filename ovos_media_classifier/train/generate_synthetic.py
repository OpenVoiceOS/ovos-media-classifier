"""Generate synthetic OCP training data from sentence templates + real-world entities.

Entities are loaded from the local HuggingFace dataset cache (populated by
``download_datasets.py``).  Curated fallback lists are used for media types that
don't have a dedicated HF dataset (TV shows, anime, podcasts, etc.).

Each output row has the same schema as ``gather_dataset.py``::

    lang, domain, intent, sentence

Templates use ``{slot}`` placeholders.  Multi-slot templates (e.g.
``"play {track} by {artist}"``) are filled by randomly pairing values from each
slot's entity list.

Usage::

    python -m ovos_media_classifier.train.generate_synthetic
    python -m ovos_media_classifier.train.generate_synthetic --max-per-intent 2000 --output my_synthetic.csv
    python -m ovos_media_classifier.train.generate_synthetic --skip-hf --output synthetic_curated.csv

    # dedup against existing dataset
    python -m ovos_media_classifier.train.generate_synthetic \\
        --dedup-against output/ocp_dataset.csv \\
        --output output/synthetic.csv
"""
from __future__ import annotations

import csv
import glob
import os
import random
import re
from typing import Callable, Optional

import pandas as pd

from ovos_media_classifier.train import get_hf_cache_dir, get_output_dir
from ovos_media_classifier.train.sources import AUDIO_INTENTS as _AUDIO_INTENTS_SRC, VIDEO_INTENTS as _VIDEO_INTENTS_SRC

random.seed(42)

# ---------------------------------------------------------------------------
# Templates
# Each entry: (template_string, [required_slot_names])
# Slots are filled from per-intent entity dicts loaded at runtime.
# ---------------------------------------------------------------------------

_MUSIC_TEMPLATES: list[tuple[str, list[str]]] = [
    # By artist
    ("play {artist}", ["artist"]),
    ("play some {artist}", ["artist"]),
    ("I want to listen to {artist}", ["artist"]),
    ("can you play {artist}", ["artist"]),
    ("put on some {artist}", ["artist"]),
    ("I'd like to hear {artist}", ["artist"]),
    ("stream {artist}", ["artist"]),
    ("play music by {artist}", ["artist"]),
    ("play a song by {artist}", ["artist"]),
    ("play something by {artist}", ["artist"]),
    ("I feel like listening to {artist}", ["artist"]),
    ("I'm in the mood for {artist}", ["artist"]),
    ("find me some {artist}", ["artist"]),
    ("play {artist}'s music", ["artist"]),
    ("I want to hear {artist}", ["artist"]),
    # By track
    ("play {track}", ["track"]),
    ("play the song {track}", ["track"]),
    ("I want to hear {track}", ["track"]),
    ("find {track} for me", ["track"]),
    ("I want to listen to {track}", ["track"]),
    # By track + artist
    ("play {track} by {artist}", ["track", "artist"]),
    ("I want to listen to {track} by {artist}", ["track", "artist"]),
    ("play {track} from {artist}", ["track", "artist"]),
    ("find me {track} by {artist}", ["track", "artist"]),
    ("put on {track} by {artist}", ["track", "artist"]),
    # By genre
    ("play some {genre} music", ["genre"]),
    ("I want to listen to some {genre}", ["genre"]),
    ("put on some {genre}", ["genre"]),
    ("I'm in the mood for {genre}", ["genre"]),
    ("play a {genre} playlist", ["genre"]),
    ("find me some {genre} music", ["genre"]),
    ("play {genre} for me", ["genre"]),
    ("I'd like some {genre}", ["genre"]),
    ("play {genre}", ["genre"]),
    ("stream some {genre}", ["genre"]),
    ("put on {genre} music", ["genre"]),
    # By album
    ("play the album {album}", ["album"]),
    ("I want to hear {album}", ["album"]),
    ("put on {album}", ["album"]),
    ("play {album} by {artist}", ["album", "artist"]),
    ("stream the album {album}", ["album"]),
]

_MOVIE_TEMPLATES: list[tuple[str, list[str]]] = [
    # By actor
    ("play a movie with {actor}", ["actor"]),
    ("play a movie starring {actor}", ["actor"]),
    ("I want to watch a movie with {actor}", ["actor"]),
    ("find a movie with {actor}", ["actor"]),
    ("show me a film starring {actor}", ["actor"]),
    ("I'd like to watch a movie with {actor}", ["actor"]),
    ("put on a movie with {actor}", ["actor"]),
    ("stream a film starring {actor}", ["actor"]),
    ("watch a movie with {actor} in it", ["actor"]),
    ("find me something starring {actor}", ["actor"]),
    ("I want to see {actor} in a film", ["actor"]),
    # By director
    ("play a film by {director}", ["director"]),
    ("I want to watch a {director} film", ["director"]),
    ("show me a {director} movie", ["director"]),
    ("find a movie by {director}", ["director"]),
    ("I'd like to watch something by {director}", ["director"]),
    ("play the latest {director} film", ["director"]),
    ("stream a film directed by {director}", ["director"]),
    ("play a movie directed by {director}", ["director"]),
    ("I want to watch something directed by {director}", ["director"]),
    ("find me a {director} movie", ["director"]),
    # By writer
    ("play a movie written by {writer}", ["writer"]),
    ("find a film written by {writer}", ["writer"]),
    ("I want to watch a film written by {writer}", ["writer"]),
    # By producer
    ("play a film produced by {producer}", ["producer"]),
    ("find a movie produced by {producer}", ["producer"]),
    # Actor + director combination
    ("play a {director} film with {actor}", ["director", "actor"]),
    ("I want to watch a {director} movie starring {actor}", ["director", "actor"]),
    # Generic
    ("play a movie", []),
    ("watch a film", []),
    ("I want to watch a movie", []),
    ("find me something to watch", []),
    ("play a good movie", []),
]

_PODCAST_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play a podcast about {topic}", ["topic"]),
    ("find me a podcast about {topic}", ["topic"]),
    ("I want to listen to a podcast about {topic}", ["topic"]),
    ("put on a {topic} podcast", ["topic"]),
    ("I'd like a podcast on {topic}", ["topic"]),
    ("stream a podcast about {topic}", ["topic"]),
    ("find a podcast on {topic}", ["topic"]),
    ("play something about {topic}", ["topic"]),
    ("I want to hear a podcast on {topic}", ["topic"]),
    ("find me a {topic} podcast", ["topic"]),
    ("play the {show} podcast", ["show"]),
    ("I want to listen to {show}", ["show"]),
    ("play an episode of {show}", ["show"]),
    ("find the {show} podcast", ["show"]),
    ("put on {show}", ["show"]),
    ("stream {show}", ["show"]),
    ("play some podcasts", []),
    ("I want to hear some podcasts", []),
    ("find me something to listen to", []),
]

_RADIO_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play {station}", ["station"]),
    ("tune into {station}", ["station"]),
    ("put on {station}", ["station"]),
    ("I want to listen to {station}", ["station"]),
    ("play {station} radio", ["station"]),
    ("stream {station}", ["station"]),
    ("find {station}", ["station"]),
    ("turn on {station}", ["station"]),
    ("switch to {station}", ["station"]),
    ("can you play {station}", ["station"]),
    ("play {genre} radio", ["genre"]),
    ("I want to listen to {genre} radio", ["genre"]),
    ("find me a {genre} radio station", ["genre"]),
    ("put on some {genre} radio", ["genre"]),
    ("stream some {genre} radio", ["genre"]),
    ("play the radio", []),
    ("turn on the radio", []),
    ("find me a radio station", []),
]

_TV_SHOW_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play {show}", ["show"]),
    ("put on {show}", ["show"]),
    ("I want to watch {show}", ["show"]),
    ("find {show} for me", ["show"]),
    ("play an episode of {show}", ["show"]),
    ("stream {show}", ["show"]),
    ("I want to watch an episode of {show}", ["show"]),
    ("play season 1 of {show}", ["show"]),
    ("continue watching {show}", ["show"]),
    ("can you play {show}", ["show"]),
    ("show me {show}", ["show"]),
    ("I'd like to watch {show}", ["show"]),
    ("watch {show}", ["show"]),
    ("I want to see {show}", ["show"]),
    ("play the next episode of {show}", ["show"]),
    ("find me something to watch", []),
    ("play a TV show", []),
]

_AUDIOBOOK_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play {title}", ["title"]),
    ("read me {title}", ["title"]),
    ("I want to listen to {title}", ["title"]),
    ("play the audiobook {title}", ["title"]),
    ("find the audiobook {title}", ["title"]),
    ("I want to hear {title}", ["title"]),
    ("play {title} by {author}", ["title", "author"]),
    ("read {title} to me", ["title"]),
    ("find me an audiobook by {author}", ["author"]),
    ("play a book by {author}", ["author"]),
    ("I'd like to listen to {author}", ["author"]),
    ("read me something by {author}", ["author"]),
    ("play an audiobook by {author}", ["author"]),
    ("find an audiobook by {author}", ["author"]),
    ("play an audiobook", []),
    ("I want to listen to a book", []),
    ("read me a story", []),
]

_NEWS_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play the news", []),
    ("I want to hear the news", []),
    ("give me the latest news", []),
    ("what's in the news today", []),
    ("read me the headlines", []),
    ("what's the latest news", []),
    ("play the morning news", []),
    ("play today's news", []),
    ("catch me up on the news", []),
    ("stream the news", []),
    ("I'd like to hear the news", []),
    ("play the evening news", []),
    ("play news from {provider}", ["provider"]),
    ("I want to hear the news from {provider}", ["provider"]),
    ("stream news from {provider}", ["provider"]),
    ("play {provider} news", ["provider"]),
    ("find me the {provider} news", ["provider"]),
    ("what's {provider} saying today", ["provider"]),
]

_ANIME_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play {title}", ["title"]),
    ("watch {title}", ["title"]),
    ("I want to watch {title}", ["title"]),
    ("put on {title}", ["title"]),
    ("play an episode of {title}", ["title"]),
    ("stream {title}", ["title"]),
    ("find {title} for me", ["title"]),
    ("I'd like to watch {title}", ["title"]),
    ("play the anime {title}", ["title"]),
    ("show me {title}", ["title"]),
    ("play some anime", []),
    ("I want to watch some anime", []),
    ("find me a good anime", []),
    ("I want to watch anime", []),
]

_CARTOON_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play {title}", ["title"]),
    ("put on {title}", ["title"]),
    ("I want to watch {title}", ["title"]),
    ("stream {title}", ["title"]),
    ("find {title} for me", ["title"]),
    ("play a {title} episode", ["title"]),
    ("I'd like to watch {title}", ["title"]),
    ("show me {title}", ["title"]),
    ("play some cartoons", []),
    ("I want to watch some cartoons", []),
    ("play cartoons", []),
    ("find me a cartoon", []),
]

_DOCUMENTARY_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play a documentary about {topic}", ["topic"]),
    ("I want to watch a documentary about {topic}", ["topic"]),
    ("find me a documentary about {topic}", ["topic"]),
    ("show me a documentary on {topic}", ["topic"]),
    ("stream a documentary about {topic}", ["topic"]),
    ("I'd like to watch a documentary about {topic}", ["topic"]),
    ("find a documentary on {topic}", ["topic"]),
    ("play a {topic} documentary", ["topic"]),
    ("I want to learn about {topic}", ["topic"]),
    ("show me something about {topic}", ["topic"]),
    ("play a nature documentary", []),
    ("play a science documentary", []),
    ("I want to watch some documentaries", []),
    ("find me a good documentary", []),
    ("play a documentary", []),
]

_GAME_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play {title}", ["title"]),
    ("I want to play {title}", ["title"]),
    ("launch {title}", ["title"]),
    ("start {title}", ["title"]),
    ("open {title}", ["title"]),
    ("I'd like to play {title}", ["title"]),
    ("find {title}", ["title"]),
    ("load {title}", ["title"]),
    ("boot up {title}", ["title"]),
    ("play a game", []),
    ("I want to play a video game", []),
    ("start a game", []),
]

_SHORT_FILM_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play a short film", []),
    ("I want to watch a short film", []),
    ("find me a short film", []),
    ("show me a short", []),
    ("play a short", []),
    ("I'd like to watch something short", []),
    ("play a short movie", []),
    ("find a short film for me", []),
    ("I want to see a short film", []),
    ("stream a short film", []),
]

_SILENT_MOVIE_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play a silent film", []),
    ("I want to watch a silent movie", []),
    ("find me a silent film", []),
    ("play some silent cinema", []),
    ("I'd like to watch a silent movie", []),
    ("stream a silent film", []),
    ("play a classic silent movie", []),
    ("show me a silent film", []),
]

_BW_MOVIE_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play a black and white movie", []),
    ("I want to watch a black and white film", []),
    ("find me a black and white movie", []),
    ("play a classic film", []),
    ("I'd like to watch a black and white film", []),
    ("stream a black and white movie", []),
    ("play an old black and white movie", []),
    ("show me something in black and white", []),
    ("play a classic black and white film", []),
]

_ASMR_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play some ASMR", []),
    ("I want to listen to ASMR", []),
    ("find me some ASMR", []),
    ("put on some ASMR", []),
    ("play ASMR for relaxation", []),
    ("I'd like to hear some ASMR", []),
    ("stream some ASMR", []),
    ("play a relaxing ASMR video", []),
    ("play ASMR sounds", []),
    ("I want to relax with ASMR", []),
]

_GENERIC_TEMPLATES: list[tuple[str, list[str]]] = [
    ("play something", []),
    ("play some media", []),
    ("I want to listen to something", []),
    ("play something random", []),
    ("find me something to listen to", []),
    ("play something for me", []),
    ("put something on", []),
    ("I want to hear something", []),
    ("play", []),
    ("find something to play", []),
    ("play anything", []),
]

# ---------------------------------------------------------------------------
# Curated entity lists for types without dedicated HF datasets
# ---------------------------------------------------------------------------

_PODCAST_TOPICS = [
    "technology", "science", "history", "politics", "comedy", "true crime",
    "sports", "business", "health", "fitness", "music", "film", "art",
    "culture", "philosophy", "economics", "psychology", "education", "news",
    "travel", "food", "finance", "investing", "parenting", "relationships",
    "meditation", "self-improvement", "entrepreneurship", "artificial intelligence",
    "space", "environment", "gaming", "literature", "architecture", "design",
    "fashion", "cryptocurrency", "cybersecurity", "medicine", "wildlife",
    "astronomy", "mythology", "folklore", "language", "mathematics",
    "social justice", "climate change", "mental health", "addiction",
    "personal finance", "real estate", "marketing", "leadership",
    "stoicism", "Buddhism", "mindfulness", "productivity", "creativity",
]

_PODCAST_SHOWS = [
    "Serial", "This American Life", "Radiolab", "Freakonomics Radio",
    "How I Built This", "The Daily", "Planet Money", "Stuff You Should Know",
    "Crime Junkie", "My Favorite Murder", "Hardcore History",
    "Lex Fridman Podcast", "Hidden Brain", "TED Talks Daily",
    "Philosophize This", "Darknet Diaries", "Syntax", "Talk Python to Me",
    "The Changelog", "Python Bytes", "Software Engineering Daily",
    "Wait Wait Don't Tell Me", "Fresh Air", "Invisibilia",
    "Conan O'Brien Needs a Friend", "SmartLess", "Armchair Expert",
    "Call Her Daddy", "Stuff You Missed in History Class",
    "No Such Thing as a Fish", "Hello Internet", "Cortex",
    "The Tim Ferriss Show", "Masters of Scale", "How I Built This",
    "WorkLife with Adam Grant", "The Knowledge Project",
    "Huberman Lab", "Found My Fitness", "The Drive", "Rich Roll Podcast",
    "Revisionist History", "Stuff They Don't Want You to Know",
    "Conspiracy Theories", "Casefile", "Last Podcast on the Left",
]

_RADIO_STATIONS = [
    "BBC Radio 1", "BBC Radio 2", "BBC Radio 3", "BBC Radio 4", "BBC Radio 6 Music",
    "NPR", "NPR Music", "KEXP", "Radio Paradise", "SomaFM",
    "iHeartRadio", "WNYC", "WBUR", "KCRW", "KPCC",
    "Jazz 24", "Classical KING FM", "181.fm", "AccuRadio",
    "DI.FM", "Digitally Imported", "KQED", "WAMU", "WBGO",
    "Radio Swiss Jazz", "Radio Swiss Classic", "Radio Swiss Pop",
    "France Inter", "FIP", "France Culture", "France Musique",
    "Deutschlandfunk", "Bayern 3", "WDR 2",
    "Radio Nacional de España", "Radio 3 España",
    "Antena 3 Portugal", "Rádio Renascença",
    "RNE", "RTP Antena 1", "RTP Antena 3",
    "Triple J", "ABC Classic FM", "ABC RN",
]

_RADIO_GENRES = [
    "jazz", "classical", "rock", "pop", "country", "hip hop", "R&B",
    "electronic", "ambient", "news", "talk", "sports", "indie", "metal",
    "reggae", "blues", "soul", "folk", "punk", "alternative", "dance",
    "house", "techno", "trance", "drum and bass", "world music",
]

_TV_SHOWS = [
    "Breaking Bad", "Game of Thrones", "The Wire", "The Sopranos", "Mad Men",
    "True Detective", "Sherlock", "Doctor Who", "Black Mirror", "Stranger Things",
    "The Crown", "Narcos", "Peaky Blinders", "Fleabag", "Chernobyl",
    "Westworld", "Better Call Saul", "Ozark", "Succession", "The Boys",
    "The Mandalorian", "Ted Lasso", "Severance", "The Bear",
    "House of the Dragon", "Andor", "The Rings of Power", "Yellowstone",
    "The Last of Us", "Wednesday", "Only Murders in the Building",
    "Abbott Elementary", "Barry", "Euphoria", "Squid Game",
    "Dark", "Money Heist", "Lupin", "Emily in Paris",
    "How I Met Your Mother", "Friends", "Seinfeld",
    "The Office", "Parks and Recreation", "Community",
    "30 Rock", "Arrested Development", "Curb Your Enthusiasm",
    "BoJack Horseman", "Rick and Morty", "Futurama", "Archer",
    "It's Always Sunny in Philadelphia", "What We Do in the Shadows",
    "Schitt's Creek", "The Good Place", "Brooklyn Nine-Nine",
    "Band of Brothers", "The Pacific", "Generation Kill",
    "Rome", "Deadwood", "Carnivàle", "Six Feet Under",
    "Battlestar Galactica", "Lost", "24", "Prison Break",
    "Dexter", "Homeland", "House of Cards", "The Americans",
    "Fargo", "Mr. Robot", "Mindhunter", "Wentworth",
]

_AUDIOBOOK_TITLES = [
    "Harry Potter and the Philosopher's Stone", "The Hitchhiker's Guide to the Galaxy",
    "Dune", "The Lord of the Rings", "Foundation",
    "Ender's Game", "The Martian", "1984", "Brave New World",
    "The Great Gatsby", "To Kill a Mockingbird", "Sapiens",
    "Thinking Fast and Slow", "Atomic Habits", "The Alchemist",
    "Ready Player One", "The Name of the Wind", "Mistborn",
    "The Way of Kings", "Words of Radiance", "A Game of Thrones",
    "Clean Code", "The Pragmatic Programmer",
    "Designing Data-Intensive Applications", "Zero to One",
    "The Lean Startup", "Outliers", "Freakonomics",
    "The Power of Habit", "Deep Work", "Digital Minimalism",
    "Man's Search for Meaning", "The Power of Now",
    "Rich Dad Poor Dad", "The 4-Hour Workweek",
    "Meditations", "Stoic Joy", "Letters from a Stoic",
    "The Art of War", "Thinking in Systems", "Antifragile",
    "The Black Swan", "Guns Germs and Steel",
    "The Selfish Gene", "A Brief History of Time",
    "Cosmos", "The Gene", "The Emperor of All Maladies",
]

_AUDIOBOOK_AUTHORS = [
    "J.K. Rowling", "Douglas Adams", "Frank Herbert", "J.R.R. Tolkien",
    "Isaac Asimov", "Orson Scott Card", "Andy Weir", "George Orwell",
    "Aldous Huxley", "F. Scott Fitzgerald", "Harper Lee", "Yuval Noah Harari",
    "Daniel Kahneman", "James Clear", "Paulo Coelho", "Ernest Cline",
    "Patrick Rothfuss", "Brandon Sanderson", "George R.R. Martin",
    "Malcolm Gladwell", "Charles Duhigg", "Cal Newport",
    "Viktor Frankl", "Stephen King", "Dean Koontz", "John Grisham",
    "Agatha Christie", "Arthur Conan Doyle", "Mark Twain", "Charles Dickens",
    "Leo Tolstoy", "Fyodor Dostoevsky", "Franz Kafka", "Ernest Hemingway",
    "Scott Fitzgerald", "William Faulkner", "Toni Morrison",
    "Nassim Nicholas Taleb", "Jared Diamond", "Richard Dawkins",
    "Carl Sagan", "Stephen Hawking", "Siddhartha Mukherjee",
]

_NEWS_PROVIDERS = [
    "BBC", "CNN", "Reuters", "NPR", "The Guardian", "Al Jazeera", "AP",
    "The New York Times", "The Washington Post", "Sky News", "ABC News",
    "CBS News", "NBC News", "Fox News", "MSNBC", "Bloomberg", "Vice News",
    "Deutsche Welle", "France 24", "CGTN", "Euronews",
    "The Financial Times", "The Economist", "The Atlantic", "Vox",
    "Axios", "Politico", "The Hill",
]

_ANIME_TITLES = [
    "Naruto", "One Piece", "Dragon Ball Z", "Attack on Titan",
    "Fullmetal Alchemist Brotherhood", "Death Note", "Hunter x Hunter",
    "Demon Slayer", "My Hero Academia", "Sword Art Online",
    "Re:Zero", "Overlord", "Black Clover", "Fairy Tail",
    "Bleach", "Neon Genesis Evangelion", "Cowboy Bebop",
    "Steins Gate", "Your Lie in April", "A Silent Voice",
    "Spirited Away", "Princess Mononoke", "My Neighbor Totoro",
    "Howl's Moving Castle", "Akira", "Ghost in the Shell",
    "Erased", "Made in Abyss", "Violet Evergarden",
    "Vinland Saga", "Jujutsu Kaisen", "Chainsaw Man",
    "Spy x Family", "Mob Psycho 100", "One Punch Man",
    "Dr. Stone", "The Promised Neverland", "Dororo",
    "Golden Kamuy", "Berserk", "Trigun", "Outlaw Star",
    "Sailor Moon", "Dragon Ball", "Yu Yu Hakusho",
    "Inuyasha", "Ruroni Kenshin", "Hajime no Ippo",
]

_CARTOON_TITLES = [
    "SpongeBob SquarePants", "Tom and Jerry", "Looney Tunes", "The Simpsons",
    "Family Guy", "South Park", "Avatar the Last Airbender", "Gravity Falls",
    "Steven Universe", "Adventure Time", "Regular Show", "The Legend of Korra",
    "Teen Titans", "Batman the Animated Series", "Ducktales", "Darkwing Duck",
    "Rugrats", "Hey Arnold", "Dexter's Laboratory", "The Powerpuff Girls",
    "Johnny Bravo", "Scooby-Doo", "Yogi Bear", "The Flintstones",
    "The Jetsons", "Chip n Dale", "Animaniacs", "Pinky and the Brain",
    "Beavis and Butt-Head", "King of the Hill", "Futurama",
    "Archer", "Bob's Burgers", "American Dad", "Phineas and Ferb",
    "Star vs the Forces of Evil", "We Bare Bears",
    "Over the Garden Wall", "The Owl House", "Amphibia",
    "Bojack Horseman", "Primal", "Invincible",
]

_DOCUMENTARY_TOPICS = [
    "nature", "wildlife", "space", "oceans", "climate change", "history",
    "World War II", "the Cold War", "ancient civilizations", "technology",
    "artificial intelligence", "social media", "food", "cooking", "wine",
    "sports", "football", "basketball", "cycling", "true crime",
    "science", "physics", "biology", "evolution", "psychology",
    "mental health", "economics", "capitalism", "health", "medicine",
    "art", "architecture", "animals", "big cats", "primates", "marine life",
    "ancient Egypt", "ancient Rome", "medieval Europe", "the Renaissance",
    "the Vietnam War", "the Civil Rights Movement", "the Holocaust",
    "the Amazon rainforest", "Antarctica", "the Sahara", "the ocean floor",
    "sharks", "whales", "elephants", "wolves", "gorillas",
    "serial killers", "cults", "organized crime", "drug trafficking",
    "nuclear power", "renewable energy", "deforestation", "plastic pollution",
    "the space race", "Mars exploration", "black holes", "dark matter",
]

_MUSIC_ARTISTS = [
    # Rock / classic rock
    "The Beatles", "Led Zeppelin", "Pink Floyd", "The Rolling Stones", "Queen",
    "AC/DC", "Black Sabbath", "Deep Purple", "Aerosmith", "Guns N Roses",
    "Nirvana", "Pearl Jam", "Soundgarden", "Alice in Chains", "Foo Fighters",
    "Red Hot Chili Peppers", "Radiohead", "Oasis", "The Cure", "Depeche Mode",
    # Pop
    "Michael Jackson", "Madonna", "Prince", "Whitney Houston", "Elton John",
    "David Bowie", "Freddie Mercury", "Beyoncé", "Taylor Swift", "Adele",
    "Ed Sheeran", "Rihanna", "Lady Gaga", "Katy Perry", "Justin Bieber",
    "Ariana Grande", "Billie Eilish", "Dua Lipa", "Harry Styles", "The Weeknd",
    # Hip-hop / R&B
    "Kendrick Lamar", "Jay-Z", "Kanye West", "Eminem", "Drake",
    "Nas", "Notorious B.I.G.", "Tupac Shakur", "Ice Cube", "Snoop Dogg",
    "Frank Ocean", "Tyler the Creator", "J Cole", "Childish Gambino", "Travis Scott",
    "Cardi B", "Nicki Minaj", "Lauryn Hill", "Missy Elliott", "Mary J. Blige",
    # Electronic / dance
    "Daft Punk", "The Chemical Brothers", "Aphex Twin", "Boards of Canada",
    "Massive Attack", "Portishead", "Gorillaz", "LCD Soundsystem",
    "Kraftwerk", "Brian Eno", "Moby", "Fatboy Slim", "The Prodigy",
    # Jazz
    "Miles Davis", "John Coltrane", "Thelonious Monk", "Charlie Parker",
    "Billie Holiday", "Ella Fitzgerald", "Louis Armstrong", "Duke Ellington",
    "Dave Brubeck", "Chet Baker", "Bill Evans", "Herbie Hancock",
    # Classical
    "Ludwig van Beethoven", "Johann Sebastian Bach", "Wolfgang Amadeus Mozart",
    "Frédéric Chopin", "Pyotr Tchaikovsky", "Claude Debussy",
    "Gustav Mahler", "Franz Schubert", "Johannes Brahms", "Richard Wagner",
    # Metal
    "Metallica", "Iron Maiden", "Slayer", "Megadeth", "Pantera",
    "Black Sabbath", "Judas Priest", "Motörhead", "Ozzy Osbourne",
    "Tool", "System of a Down", "Rage Against the Machine", "Deftones",
    # Country
    "Johnny Cash", "Dolly Parton", "Willie Nelson", "Hank Williams",
    "Merle Haggard", "Waylon Jennings", "George Strait", "Garth Brooks",
    "Kenny Rogers", "Loretta Lynn", "Patsy Cline",
    # Soul / funk
    "James Brown", "Aretha Franklin", "Marvin Gaye", "Stevie Wonder",
    "Al Green", "Otis Redding", "Ray Charles", "Sam Cooke",
    "Sly and the Family Stone", "Parliament Funkadelic", "Curtis Mayfield",
    # Indie / alternative
    "The Strokes", "Arctic Monkeys", "Interpol", "Vampire Weekend",
    "Bon Iver", "Fleet Foxes", "Sufjan Stevens", "The National",
    "Arcade Fire", "Animal Collective", "Beach House", "Tame Impala",
    "Mac DeMarco", "Car Seat Headrest", "Japanese Breakfast",
]

_MUSIC_GENRES = [
    "rock", "classic rock", "indie rock", "alternative rock", "punk rock",
    "heavy metal", "thrash metal", "black metal", "death metal", "doom metal",
    "pop", "indie pop", "synth pop", "dream pop", "electro pop",
    "hip hop", "rap", "trap", "lo-fi hip hop", "old school hip hop",
    "R&B", "soul", "funk", "neo soul", "new jack swing",
    "jazz", "bebop", "cool jazz", "fusion jazz", "smooth jazz",
    "classical", "baroque", "romantic", "contemporary classical",
    "electronic", "techno", "house", "trance", "drum and bass",
    "ambient", "downtempo", "chillout", "lo-fi", "vaporwave",
    "country", "bluegrass", "folk", "americana", "outlaw country",
    "blues", "delta blues", "electric blues", "Chicago blues",
    "reggae", "ska", "dancehall", "dub",
    "world music", "afrobeat", "bossa nova", "salsa", "flamenco",
    "gospel", "spiritual", "Christian music",
    "opera", "musical theatre",
    "psychedelic rock", "progressive rock", "krautrock",
    "post-punk", "new wave", "goth rock", "shoegaze",
    "grunge", "emo", "post-hardcore", "screamo",
    "disco", "new wave", "80s music", "90s music",
    "relaxing music", "focus music", "workout music", "meditation music",
]

_MUSIC_TRACKS = [
    "Bohemian Rhapsody", "Stairway to Heaven", "Hotel California",
    "Smells Like Teen Spirit", "Purple Haze", "Johnny B. Goode",
    "Like a Rolling Stone", "Imagine", "What's Going On",
    "Superstition", "Higher Ground", "Billie Jean", "Thriller",
    "Purple Rain", "When Doves Cry", "Kiss",
    "Shape of You", "Rolling in the Deep", "Someone Like You",
    "Bad Guy", "Old Town Road", "Blinding Lights",
    "Lose Yourself", "Rap God", "HUMBLE.",
    "God's Plan", "Hotline Bling", "Started From the Bottom",
    "All of Me", "Stay With Me", "Thinking Out Loud",
    "Watermelon Sugar", "As It Was", "Golden",
    "Levitating", "Physical", "Don't Start Now",
    "Clocks", "The Scientist", "Yellow", "Fix You",
    "Mr. Brightside", "Somebody Told Me", "Human",
    "Seven Nation Army", "Fell in Love with a Girl",
    "Come as You Are", "Heart-Shaped Box", "About a Girl",
    "Black", "Yellow Ledbetter", "Even Flow",
]

_MUSIC_ALBUMS = [
    "Abbey Road", "Sgt. Pepper's Lonely Hearts Club Band", "Revolver",
    "Dark Side of the Moon", "Wish You Were Here", "Animals",
    "Led Zeppelin IV", "Physical Graffiti",
    "Nevermind", "In Utero", "Bleach",
    "OK Computer", "Kid A", "The Bends",
    "The Chronic", "Illmatic", "Ready to Die",
    "My Beautiful Dark Twisted Fantasy", "The Blueprint", "Reasonable Doubt",
    "Kind of Blue", "A Love Supreme", "Birth of the Cool",
    "Thriller", "Off the Wall", "Bad",
    "Purple Rain", "Sign o the Times",
    "What's Going On", "Songs in the Key of Life",
    "Rumours", "Tusk",
    "Born to Run", "Nebraska", "The River",
    "Tapestry", "Tapestry",
    "Back in Black", "Highway to Hell",
    "Master of Puppets", "Ride the Lightning", "The Black Album",
    "folklore", "evermore", "1989",
    "25", "21", "19",
    "÷", "×", "+",
    "After Hours", "Dawn FM",
    "When We All Fall Asleep Where Do We Go",
    "Future Nostalgia",
]

_MOVIE_ACTORS = [
    "Tom Hanks", "Meryl Streep", "Leonardo DiCaprio", "Cate Blanchett",
    "Denzel Washington", "Natalie Portman", "Brad Pitt", "Angelina Jolie",
    "Robert De Niro", "Jodie Foster", "Al Pacino", "Julia Roberts",
    "Morgan Freeman", "Sandra Bullock", "Anthony Hopkins", "Kate Winslet",
    "Daniel Day-Lewis", "Charlize Theron", "Joaquin Phoenix", "Viola Davis",
    "Tom Cruise", "Nicole Kidman", "Matt Damon", "Cate Blanchett",
    "Will Smith", "Halle Berry", "Johnny Depp", "Helena Bonham Carter",
    "Russell Crowe", "Julianne Moore", "Edward Norton", "Gwyneth Paltrow",
    "Christian Bale", "Hilary Swank", "Jake Gyllenhaal", "Reese Witherspoon",
    "Ryan Gosling", "Emma Stone", "Timothée Chalamet", "Zendaya",
    "Pedro Pascal", "Florence Pugh", "Andrew Garfield", "Ana de Armas",
    "Paul Mescal", "Carey Mulligan", "Barry Keoghan", "Saoirse Ronan",
    "Arnold Schwarzenegger", "Sylvester Stallone", "Bruce Willis", "Harrison Ford",
    "Clint Eastwood", "Jack Nicholson", "Dustin Hoffman", "Gene Hackman",
    "Marlon Brando", "James Dean", "Audrey Hepburn", "Grace Kelly",
    "Humphrey Bogart", "Ingrid Bergman", "Clark Gable", "Marilyn Monroe",
]

_MOVIE_DIRECTORS = [
    "Steven Spielberg", "Martin Scorsese", "Christopher Nolan", "Quentin Tarantino",
    "Alfred Hitchcock", "Stanley Kubrick", "Francis Ford Coppola", "Ridley Scott",
    "James Cameron", "Peter Jackson", "David Fincher", "Paul Thomas Anderson",
    "Wes Anderson", "Sofia Coppola", "Guillermo del Toro", "Denis Villeneuve",
    "Darren Aronofsky", "Terrence Malick", "Coen Brothers", "David Lynch",
    "Tim Burton", "Spike Lee", "Ang Lee", "Woody Allen",
    "Clint Eastwood", "Ron Howard", "Oliver Stone", "Michael Mann",
    "Werner Herzog", "Akira Kurosawa", "Federico Fellini", "Ingmar Bergman",
    "Jean-Luc Godard", "François Truffaut", "Pedro Almodóvar", "Wong Kar-wai",
    "Park Chan-wook", "Bong Joon-ho", "Hayao Miyazaki", "Takeshi Kitano",
    "Andrei Tarkovsky", "Satyajit Ray", "Abbas Kiarostami",
    "Ari Aster", "Jordan Peele", "Greta Gerwig", "Kelly Reichardt",
    "Barry Jenkins", "Ryan Coogler", "Damien Chazelle", "Noah Baumbach",
]

_GAME_TITLES = [
    "Minecraft", "Fortnite", "Among Us", "Roblox",
    "Grand Theft Auto V", "Red Dead Redemption 2",
    "The Witcher 3", "Cyberpunk 2077", "Elden Ring",
    "Dark Souls", "Sekiro", "Bloodborne",
    "The Legend of Zelda Breath of the Wild", "Super Mario Bros",
    "Mario Kart", "Pokemon", "Call of Duty", "Halo", "Doom",
    "Counter-Strike", "Valorant", "Overwatch", "Apex Legends",
    "League of Legends", "Dota 2", "Hearthstone", "Stardew Valley",
    "Animal Crossing", "Terraria", "Portal", "Portal 2",
    "Half-Life 2", "Mass Effect", "Dragon Age", "Skyrim", "Fallout",
    "God of War", "Spider-Man", "Horizon Zero Dawn",
    "The Last of Us", "Uncharted", "Ghost of Tsushima",
    "FIFA", "NBA 2K", "Madden NFL", "Rocket League",
    "Cuphead", "Hollow Knight", "Celeste", "Hades",
    "Dead Cells", "Ori and the Blind Forest", "Undertale",
    "Disco Elysium", "Divinity Original Sin 2", "Baldur's Gate 3",
]


# ---------------------------------------------------------------------------
# Entity loaders — read from the local HuggingFace cache
# ---------------------------------------------------------------------------

def _load_hf_safe(dataset_name: str, split: str, hf_cache: str):
    """Load a HuggingFace dataset from the local cache, return empty list on failure."""
    try:
        from datasets import load_dataset
        return load_dataset(dataset_name, split=split, cache_dir=hf_cache)
    except Exception as exc:
        print(f"  [warn] could not load {dataset_name}: {exc}")
        return []


def load_music_entities(hf_cache: str) -> dict[str, list[str]]:
    artists, tracks, genres, albums = set(), set(), set(), set()

    for entry in _load_hf_safe("Jarbas/metal-archives-tracks", "train", hf_cache):
        if entry.get("band_name"):
            artists.add(entry["band_name"])
        if entry.get("track_name"):
            tracks.add(entry["track_name"])
        if entry.get("album_name"):
            albums.add(entry["album_name"])

    for entry in _load_hf_safe("Jarbas/metal-archives-bands", "train", hf_cache):
        if entry.get("name"):
            artists.add(entry["name"])
        if entry.get("genre"):
            for g in entry["genre"].split(","):
                genres.add(g.strip())

    for entry in _load_hf_safe("Jarbas/jazz-music-archives", "train", hf_cache):
        if entry.get("artist"):
            artists.add(entry["artist"])
        if entry.get("genre"):
            genres.add(entry["genre"])

    for entry in _load_hf_safe("Jarbas/prog-archives", "train", hf_cache):
        if entry.get("artist"):
            artists.add(entry["artist"])
        if entry.get("genre"):
            genres.add(entry["genre"])

    for entry in _load_hf_safe("Jarbas/classic-composers", "train", hf_cache):
        if entry.get("name"):
            artists.add(entry["name"])

    for entry in _load_hf_safe("Jarbas/trance_tracks", "train", hf_cache):
        if entry.get("ARTIST(S)"):
            artists.add(entry["ARTIST(S)"])
        if entry.get("TRACK"):
            tracks.add(entry["TRACK"])
        if entry.get("STYLE"):
            genres.add(entry["STYLE"])

    return {
        "artist": [a for a in artists if a and len(a) > 1],
        "track":  [t for t in tracks  if t and len(t) > 1],
        "genre":  [g for g in genres  if g and len(g) > 1],
        "album":  [a for a in albums  if a and len(a) > 1],
    }


def load_movie_entities(hf_cache: str) -> dict[str, list[str]]:
    result: dict[str, set] = {
        "actor": set(), "director": set(), "producer": set(),
        "writer": set(), "composer": set(),
    }
    for label, dataset_name in [
        ("actor",    "Jarbas/movie_actors"),
        ("director", "Jarbas/movie_directors"),
        ("producer", "Jarbas/movie_producers"),
        ("writer",   "Jarbas/movie_writers"),
        ("composer", "Jarbas/movie_composers"),
    ]:
        for entry in _load_hf_safe(dataset_name, "train", hf_cache):
            if entry.get("name"):
                result[label].add(entry["name"])
    return {k: list(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Per-intent configuration
# ---------------------------------------------------------------------------

def _identity_entities(_hf_cache: str) -> dict[str, list[str]]:
    return {}


# ---------------------------------------------------------------------------
# CSV template loading
# ---------------------------------------------------------------------------

def load_templates_from_csv(templates_dir: str) -> dict[str, list[tuple[str, list[str]]]]:
    """Load templates from {templates_dir}/*.csv.

    Each CSV file is expected to have columns: category,template
    where category is a programmatic identifier (unused) and template
    contains {slot} placeholders.

    Returns a dict mapping intent name (filename stem, e.g. "music" from "music.csv")
    to a list of (template_string, [required_slot_names]) tuples.
    """
    result = {}
    for csv_path in glob.glob(os.path.join(templates_dir, "*.csv")):
        intent = os.path.splitext(os.path.basename(csv_path))[0]  # e.g. "music"
        templates = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tmpl = row.get("template", "").strip() if row else ""
                    if not tmpl:
                        continue
                    # Parse slot names from {slot} and {{slot}} patterns
                    slots = re.findall(r'\{(\w+)\}', tmpl.replace('{{', '{').replace('}}', '}'))
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_slots = []
                    for s in slots:
                        if s not in seen:
                            seen.add(s)
                            unique_slots.append(s)
                    templates.append((tmpl, unique_slots))
        except Exception as e:
            print(f"  Warning: Failed to load {csv_path}: {e}")
            continue
        if templates:
            result[intent] = templates
    return result


_INTENT_CONFIGS: dict[str, dict] = {
    "music": {
        "domain": "ocp_play",
        "templates": _MUSIC_TEMPLATES,
        "load_entities": load_music_entities,
        # HF datasets extend these at runtime; curated lists ensure --skip-hf still works
        "static_slots": {
            "artist": _MUSIC_ARTISTS,
            "genre":  _MUSIC_GENRES,
            "track":  _MUSIC_TRACKS,
            "album":  _MUSIC_ALBUMS,
        },
    },
    "movie": {
        "domain": "ocp_play",
        "templates": _MOVIE_TEMPLATES,
        "load_entities": load_movie_entities,
        # HF datasets extend these at runtime; curated lists work offline too
        "static_slots": {
            "actor":    _MOVIE_ACTORS,
            "director": _MOVIE_DIRECTORS,
        },
    },
    "podcast": {
        "domain": "ocp_play",
        "templates": _PODCAST_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {"topic": _PODCAST_TOPICS, "show": _PODCAST_SHOWS},
    },
    "radio": {
        "domain": "ocp_play",
        "templates": _RADIO_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {"station": _RADIO_STATIONS, "genre": _RADIO_GENRES},
    },
    "tv_show": {
        "domain": "ocp_play",
        "templates": _TV_SHOW_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {"show": _TV_SHOWS},
    },
    "audiobook": {
        "domain": "ocp_play",
        "templates": _AUDIOBOOK_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {"title": _AUDIOBOOK_TITLES, "author": _AUDIOBOOK_AUTHORS},
    },
    "news": {
        "domain": "ocp_play",
        "templates": _NEWS_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {"provider": _NEWS_PROVIDERS},
    },
    "anime": {
        "domain": "ocp_play",
        "templates": _ANIME_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {"title": _ANIME_TITLES},
    },
    "cartoon": {
        "domain": "ocp_play",
        "templates": _CARTOON_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {"title": _CARTOON_TITLES},
    },
    "documentary": {
        "domain": "ocp_play",
        "templates": _DOCUMENTARY_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {"topic": _DOCUMENTARY_TOPICS},
    },
    "game": {
        "domain": "ocp_play",
        "templates": _GAME_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {"title": _GAME_TITLES},
    },
    "short_film": {
        "domain": "ocp_play",
        "templates": _SHORT_FILM_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {},
    },
    "silent_movie": {
        "domain": "ocp_play",
        "templates": _SILENT_MOVIE_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {},
    },
    "bw_movie": {
        "domain": "ocp_play",
        "templates": _BW_MOVIE_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {},
    },
    "asmr": {
        "domain": "ocp_play",
        "templates": _ASMR_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {},
    },
    "generic": {
        "domain": "ocp_play",
        "templates": _GENERIC_TEMPLATES,
        "load_entities": _identity_entities,
        "static_slots": {},
    },
    "music_video": {
        "domain": "ocp_play",
        "templates": [
            ("play the music video for {track}", ["track"]),
            ("show me the {track} music video", ["track"]),
            ("I want to watch the official video for {track}", ["track"]),
            ("play the music video for {track} by {artist}", ["track", "artist"]),
            ("find the {track} video by {artist}", ["track", "artist"]),
            ("play {artist} music videos", ["artist"]),
            ("show me a music video by {artist}", ["artist"]),
            ("I want to watch {artist} videos", ["artist"]),
            ("play a music video", []),
            ("show me music videos", []),
            ("find me a good music video", []),
        ],
        "load_entities": _identity_entities,
        "static_slots": {
            "artist": _MUSIC_ARTISTS,
            "track": _MUSIC_TRACKS,
        },
    },
    "trailer": {
        "domain": "ocp_play",
        "templates": [
            ("play the trailer for {movie}", ["movie"]),
            ("show me the {movie} trailer", ["movie"]),
            ("I want to watch the {movie} trailer", ["movie"]),
            ("find the official trailer for {movie}", ["movie"]),
            ("play the teaser for {movie}", ["movie"]),
            ("play the trailer for a {director} film", ["director"]),
            ("play a movie trailer", []),
            ("show me some trailers", []),
            ("find the latest trailers", []),
        ],
        "load_entities": load_movie_entities,
        "static_slots": {},
    },
    "behind_the_scenes": {
        "domain": "ocp_play",
        "templates": [
            ("show me the making of {movie}", ["movie"]),
            ("I want to watch the making of {movie}", ["movie"]),
            ("play the featurette for {movie}", ["movie"]),
            ("find behind the scenes footage for {movie}", ["movie"]),
            ("show me how {movie} was made", ["movie"]),
            ("play the cast interview for {movie}", ["movie"]),
            ("play behind the scenes content", []),
            ("show me a making of documentary", []),
            ("find me some featurettes", []),
            ("play some bloopers", []),
        ],
        "load_entities": load_movie_entities,
        "static_slots": {},
    },
    "tv": {
        "domain": "ocp_play",
        "templates": [
            ("play {channel}", ["channel"]),
            ("stream {channel}", ["channel"]),
            ("put on {channel}", ["channel"]),
            ("I want to watch {channel}", ["channel"]),
            ("tune to {channel}", ["channel"]),
            ("switch to {channel}", ["channel"]),
            ("play live TV", []),
            ("stream live television", []),
            ("put on the news channel", []),
            ("I want to watch live TV", []),
        ],
        "load_entities": _identity_entities,
        "static_slots": {
            "channel": [
                "CNN", "BBC One", "BBC Two", "BBC News", "ITV", "Channel 4",
                "Fox News", "MSNBC", "NBC", "CBS", "ABC", "PBS",
                "Discovery Channel", "National Geographic", "History Channel",
                "Eurosport", "Sky Sports", "ESPN",
                "Arte", "ZDF", "ARD", "France 2", "RAI 1",
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_intent(
    intent: str,
    config: dict,
    hf_cache: str,
    max_samples: int,
    skip_hf: bool,
    existing: Optional[set] = None,
    lang: str = "en",
) -> list[tuple[str, str, str, str]]:
    """Generate (lang, domain, intent, sentence) rows for one intent."""
    # Load entities from HF datasets
    if skip_hf:
        hf_entities: dict[str, list[str]] = {}
    else:
        hf_entities = config["load_entities"](hf_cache)

    # Merge static slot lists with HF-loaded ones
    slots: dict[str, list[str]] = {}
    for slot_name, static_vals in config["static_slots"].items():
        merged = list(static_vals)
        if slot_name in hf_entities:
            merged = list(set(merged + hf_entities[slot_name]))
        slots[slot_name] = [v for v in merged if v and str(v).strip()]

    for slot_name, hf_vals in hf_entities.items():
        if slot_name not in slots:
            slots[slot_name] = [v for v in hf_vals if v and str(v).strip()]

    domain = config["domain"]
    templates = config["templates"]
    existing = existing or set()
    results: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()

    # Split budget roughly across templates
    templates_with_slots = [t for t in templates if t[1]]
    templates_no_slots = [t for t in templates if not t[1]]

    # Templates that have slots generate most of the variety
    per_template = max(10, max_samples // max(len(templates), 1))

    def _fill(template: str, required: list[str]) -> Optional[str]:
        try:
            kw = {s: random.choice(slots[s]) for s in required if slots.get(s)}
            if len(kw) < len(required):
                return None  # a required slot has no data
            return template.format(**kw)
        except (KeyError, IndexError):
            return None

    # Slot-bearing templates: oversample then cap
    for tmpl, required in templates_with_slots:
        if not all(slots.get(s) for s in required):
            continue
        count = 0
        attempts = 0
        max_attempts = per_template * 5
        while count < per_template and attempts < max_attempts:
            attempts += 1
            sentence = _fill(tmpl, required)
            if sentence and sentence not in seen and sentence not in existing:
                seen.add(sentence)
                results.append((lang, domain, intent, sentence))
                count += 1
        if len(results) >= max_samples:
            break

    # No-slot templates: add each once
    for tmpl, _ in templates_no_slots:
        sentence = tmpl
        if sentence not in seen and sentence not in existing:
            seen.add(sentence)
            results.append((lang, domain, intent, sentence))

    # Shuffle and cap
    random.shuffle(results)
    return results[:max_samples]


def generate_all(
    max_per_intent: int = 5000,
    intents: Optional[list[str]] = None,
    skip_hf: bool = False,
    dedup_against: Optional[str] = None,
    lang: str = "en",
    templates_dir: Optional[str] = None,
) -> pd.DataFrame:
    hf_cache = get_hf_cache_dir()

    existing: set[str] = set()
    if dedup_against and os.path.exists(dedup_against):
        df_ex = pd.read_csv(dedup_against)
        if "sentence" in df_ex.columns:
            existing = set(df_ex["sentence"].dropna().str.lower())
            print(f"Deduplicating against {len(existing):,} existing sentences")

    selected = {k: v for k, v in _INTENT_CONFIGS.items()
                if intents is None or k in intents}

    # Load and merge CSV templates if provided
    if templates_dir and os.path.isdir(templates_dir):
        csv_templates = load_templates_from_csv(templates_dir)
        for intent, tmpl_list in csv_templates.items():
            if intent in selected:
                selected[intent]["templates"].extend(tmpl_list)
            # else: silently skip unknown intents (may be a foreign-lang artifact)

    _AUDIO = _AUDIO_INTENTS_SRC
    _VIDEO = _VIDEO_INTENTS_SRC

    all_rows: list[tuple[str, str, str, str, str, str, str]] = []
    for intent, config in selected.items():
        print(f"  Generating {intent} …", end="", flush=True)
        rows = _generate_intent(intent, config, hf_cache, max_per_intent,
                                skip_hf=skip_hf, existing=existing, lang=lang)
        print(f" {len(rows):,}")
        for lang_v, domain, intent_v, sentence in rows:
            binary_label = "ocp" if domain in ("ocp_play", "ocp_control") else "not_ocp"
            if domain != "ocp_play":
                playback_label = "undefined"
            elif intent_v in _AUDIO:
                playback_label = "audio"
            elif intent_v in _VIDEO:
                playback_label = "video"
            else:
                playback_label = "undefined"
            media_label = intent_v if domain == "ocp_play" else "not_ocp"
            all_rows.append((lang_v, domain, intent_v, binary_label,
                             playback_label, media_label, sentence))

    df = pd.DataFrame(all_rows, columns=["lang", "domain", "intent", "binary_label",
                                          "playback_label", "media_label", "sentence"])
    df["lang"] = lang
    return df


# ---------------------------------------------------------------------------
