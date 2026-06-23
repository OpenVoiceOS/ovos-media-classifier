"""Generate keyword-based training utterances from entity lists + sentence templates.

Complements generate_from_ocp_templates.py (which fills official templates with
Wikidata) and generate_synthetic.py (custom templates + HF entities).  This
script is fully offline, using only hardcoded entity lists drawn from general
knowledge, so it works with zero network access.

Intended to fill data gaps for low-coverage intents:
  audio, asmr, audio_description, visual_story, radio_theatre,
  short_film, silent_movie, bw_movie, adult_audio

Output schema (identical to gather_dataset.py)::

    lang, domain, intent, binary_label, playback_label, media_label, sentence

Usage::

    python -m training.generate_keyword_csv
    python -m training.generate_keyword_csv --intents audio asmr --n 2000
    python -m training.generate_keyword_csv --dedup-against ocp_dataset.csv
"""
from __future__ import annotations

import argparse
import os
import random
import re
from itertools import product
from typing import Optional

import pandas as pd

from training import get_output_dir
from training.sources import AUDIO_INTENTS as _AUDIO_INTENTS, VIDEO_INTENTS as _VIDEO_INTENTS

random.seed(42)


def _labels(domain: str, intent: str) -> tuple[str, str, str]:
    binary = "ocp" if domain in ("ocp_play", "ocp_control") else "not_ocp"
    if domain != "ocp_play":
        playback = "undefined"
    elif intent in _AUDIO_INTENTS:
        playback = "audio"
    elif intent in _VIDEO_INTENTS:
        playback = "video"
    else:
        playback = "undefined"
    media = intent if domain == "ocp_play" else "not_ocp"
    return binary, playback, media


# ---------------------------------------------------------------------------
# Play verbs (shared across many intents)
# ---------------------------------------------------------------------------

_PLAY_VERBS = [
    "play", "stream", "put on", "start", "queue up", "launch",
    "find me", "get me", "show me", "load", "open",
    "I want to watch", "I want to listen to", "I'd like to watch",
    "I'd like to hear", "can you play", "please play", "could you play",
    "I feel like watching", "I feel like listening to",
]
_PLAY_VERBS_AUDIO = [
    "play", "stream", "put on", "start", "queue up",
    "find me", "get me", "load",
    "I want to listen to", "I'd like to hear", "can you play",
    "please play", "could you play", "let me hear",
]
_PLAY_VERBS_VIDEO = [
    "play", "stream", "put on", "start", "queue up", "show me",
    "find me", "get me", "load", "open",
    "I want to watch", "I'd like to watch", "can you play",
    "please put on", "could you show me",
]

# ---------------------------------------------------------------------------
# Entity lists per intent
# ---------------------------------------------------------------------------

_ENTITIES: dict[str, dict[str, list[str]]] = {

"music": {
    "artist": [
        "The Beatles", "Pink Floyd", "Led Zeppelin", "Queen", "David Bowie",
        "Bob Dylan", "Jimi Hendrix", "The Rolling Stones", "Nirvana", "Radiohead",
        "Taylor Swift", "Beyoncé", "Drake", "Kendrick Lamar", "Kanye West",
        "The Weeknd", "Billie Eilish", "Adele", "Ed Sheeran", "Bruno Mars",
        "Rihanna", "Ariana Grande", "Frank Sinatra", "Nina Simone", "Ella Fitzgerald",
        "Miles Davis", "John Coltrane", "Chet Baker", "Louis Armstrong", "Duke Ellington",
        "Herbie Hancock", "Thelonious Monk", "Charlie Parker", "Bill Evans",
        "Johnny Cash", "Dolly Parton", "Willie Nelson", "Hank Williams", "Merle Haggard",
        "Metallica", "Black Sabbath", "Iron Maiden", "Slayer", "System of a Down",
        "Tool", "Pantera", "Judas Priest", "Motörhead", "AC/DC",
        "Daft Punk", "Aphex Twin", "Kraftwerk", "Brian Eno", "Massive Attack",
        "Portishead", "Boards of Canada", "Burial", "The Chemical Brothers",
        "Tupac Shakur", "Notorious B.I.G.", "Nas", "Eminem", "Wu-Tang Clan",
        "Jay-Z", "50 Cent", "Lil Wayne", "Chance the Rapper", "Tyler the Creator",
        "Bob Marley", "Peter Tosh", "Burning Spear", "Toots and the Maytals",
        "Bach", "Beethoven", "Mozart", "Chopin", "Debussy", "Brahms",
        "Vivaldi", "Handel", "Tchaikovsky", "Mahler", "Stravinsky",
        "The Clash", "The Cure", "Joy Division", "New Order", "Depeche Mode",
        "Talking Heads", "Blondie", "Patti Smith", "The Velvet Underground",
        "Fleetwood Mac", "Stevie Wonder", "Marvin Gaye", "Otis Redding", "James Brown",
        "Aretha Franklin", "Ray Charles", "B.B. King", "Muddy Waters", "Robert Johnson",
        "Coldplay", "U2", "R.E.M.", "Pearl Jam", "Soundgarden", "Alice in Chains",
        "Rage Against the Machine", "Red Hot Chili Peppers", "Foo Fighters", "Weezer",
        "Post Malone", "Lil Nas X", "Bad Bunny", "J Balvin", "Daddy Yankee",
        "Rosalía", "C. Tangana", "Karol G", "Maluma", "Nicki Minaj",
        "Cardi B", "Megan Thee Stallion", "SZA", "H.E.R.", "Lizzo",
        "Harry Styles", "Dua Lipa", "Olivia Rodrigo", "Sabrina Carpenter", "Chappell Roan",
    ],
    "song": [
        "Bohemian Rhapsody", "Hotel California", "Stairway to Heaven", "Imagine",
        "Smells Like Teen Spirit", "Purple Rain", "Like a Rolling Stone", "Yesterday",
        "Johnny B. Goode", "Superstition", "Born to Run", "God Save the Queen",
        "Blinding Lights", "Shape of You", "Rolling in the Deep", "Uptown Funk",
        "Bad Guy", "God's Plan", "HUMBLE.", "Money Trees", "Alright",
        "Come As You Are", "Black Hole Sun", "Creep", "Paranoid Android",
        "Under the Bridge", "Californication", "Mr. Brightside", "Take On Me",
        "Don't Stop Me Now", "Bohemian Rhapsody", "We Will Rock You", "Radio Ga Ga",
        "Dancing Queen", "Waterloo", "Fernando", "Gimme Gimme Gimme",
        "Enter Sandman", "Master of Puppets", "Nothing Else Matters", "One",
        "Wish You Were Here", "Comfortably Numb", "Another Brick in the Wall",
        "Riders on the Storm", "Light My Fire", "People Are Strange",
        "Here Comes the Sun", "Let It Be", "Hey Jude", "Come Together",
        "Lose Yourself", "Stan", "The Real Slim Shady", "Without Me",
        "Gold Digger", "Stronger", "Power", "All Falls Down",
        "Hotline Bling", "Started From the Bottom", "One Dance", "Passionfruit",
        "Redbone", "This Is America", "Childish Gambino",
        "Drivers License", "good 4 u", "deja vu", "traitor",
        "As It Was", "Watermelon Sugar", "Adore You",
        "Levitating", "Don't Start Now", "Physical",
        "Anti-Hero", "Shake It Off", "Love Story", "You Belong With Me",
        "Crazy in Love", "Halo", "Single Ladies", "Lemonade",
        "Umbrella", "We Found Love", "Diamonds", "Work",
    ],
    "album": [
        "Dark Side of the Moon", "Abbey Road", "Thriller", "Rumours", "OK Computer",
        "Nevermind", "Random Access Memories", "To Pimp a Butterfly", "Lemonade",
        "1989", "Folklore", "Evermore", "Led Zeppelin IV", "Pet Sounds",
        "What's Going On", "Purple Rain", "Born to Run", "Highway 61 Revisited",
        "Blonde", "My Beautiful Dark Twisted Fantasy", "Watch the Throne",
        "The Blueprint", "The Marshall Mathers LP", "The Slim Shady LP",
        "Enter the Wu-Tang", "Illmatic", "Reasonable Doubt", "The College Dropout",
        "good kid m.A.A.d city", "Damn", "Ctrl", "When We All Fall Asleep",
        "Future Nostalgia", "Happier Than Ever", "30", "Multiply", "÷",
        "No Line on the Horizon", "Achtung Baby", "The Joshua Tree",
        "Disintegration", "Closer", "Violator", "Ultra",
        "Blue Lines", "Dummy", "Mezzanine", "Protection",
        "Appetite for Destruction", "Use Your Illusion", "Chinese Democracy",
        "Master of Puppets", "Ride the Lightning", "And Justice for All",
        "Black Album", "Load", "St. Anger", "Death Magnetic",
        "Paranoid", "Heaven and Hell", "Mob Rules", "Seventh Son of a Seventh Son",
    ],
    "genre": [
        "jazz", "blues", "classical", "hip-hop", "rap", "rock", "heavy metal",
        "death metal", "black metal", "thrash metal", "doom metal", "punk rock",
        "indie rock", "folk", "country", "bluegrass", "reggae", "ska", "dub",
        "soul", "funk", "R&B", "gospel", "pop", "EDM", "techno", "house",
        "trance", "ambient", "lo-fi", "drum and bass", "dubstep", "grime", "drill",
        "alternative rock", "progressive rock", "psychedelic rock", "grunge",
        "bossa nova", "samba", "flamenco", "K-pop", "J-pop", "opera",
        "afrobeats", "afropop", "salsa", "cumbia", "reggaeton", "Latin pop",
        "new wave", "post-punk", "shoegaze", "dream pop", "math rock",
        "bedroom pop", "synthpop", "vaporwave", "chillwave", "tropical house",
    ],
    "mood": [
        "upbeat", "chill", "sad", "happy", "energetic", "relaxing", "melancholic",
        "romantic", "motivating", "angry", "nostalgic", "focus", "party",
    ],
    "activity": [
        "working out", "studying", "sleeping", "cooking", "driving", "running",
        "yoga", "meditation", "dinner", "party", "road trip", "morning routine",
    ],
},

"movie": {
    "title": [
        "The Godfather", "The Shawshank Redemption", "Pulp Fiction", "Schindler's List",
        "The Dark Knight", "Fight Club", "Forrest Gump", "Inception", "The Matrix",
        "Goodfellas", "Interstellar", "Parasite", "Spirited Away", "The Silence of the Lambs",
        "Casablanca", "2001: A Space Odyssey", "Apocalypse Now", "Blade Runner",
        "Alien", "Star Wars", "Jaws", "Jurassic Park", "The Lion King", "Toy Story",
        "WALL-E", "Finding Nemo", "Up", "Coco", "Soul", "Encanto",
        "Avengers: Endgame", "Black Panther", "Spider-Man: No Way Home",
        "No Country for Old Men", "There Will Be Blood", "The Social Network",
        "Mad Max: Fury Road", "Get Out", "Hereditary", "Midsommar", "Us",
        "Everything Everywhere All at Once", "Oppenheimer", "Barbie", "Dune",
        "Top Gun: Maverick", "Avatar", "Titanic", "The Lord of the Rings",
        "Harry Potter", "The Hunger Games", "The Revenant", "Birdman",
        "Moonlight", "12 Years a Slave", "The Shape of Water", "Nomadland",
        "Whiplash", "La La Land", "The Grand Budapest Hotel", "Knives Out",
        "Glass Onion", "Triangle of Sadness", "Tár", "The Banshees of Inisherin",
        "Past Lives", "Poor Things", "American Fiction", "Anatomy of a Fall",
        "Saltburn", "Priscilla", "May December", "Ferrari", "Maestro",
        "Killers of the Flower Moon", "The Zone of Interest", "Society of the Snow",
    ],
    "actor": [
        "Tom Hanks", "Meryl Streep", "Jack Nicholson", "Robert De Niro", "Al Pacino",
        "Morgan Freeman", "Denzel Washington", "Cate Blanchett", "Leonardo DiCaprio",
        "Brad Pitt", "Timothée Chalamet", "Scarlett Johansson", "Natalie Portman",
        "Viola Davis", "Ryan Gosling", "Florence Pugh", "Zendaya", "Austin Butler",
        "Charlize Theron", "Samuel L. Jackson", "Christian Bale", "Anthony Hopkins",
        "Daniel Day-Lewis", "Marlon Brando", "Dustin Hoffman", "Gene Hackman",
        "Robert Duvall", "Javier Bardem", "Benicio del Toro", "Idris Elba",
        "Chiwetel Ejiofor", "Michael Fassbender", "Tom Hardy", "Gary Oldman",
        "Edward Norton", "Joaquin Phoenix", "Philip Seymour Hoffman", "Heath Ledger",
        "Michelle Pfeiffer", "Julianne Moore", "Nicole Kidman", "Kate Blanchett",
        "Kate Winslet", "Saoirse Ronan", "Emma Stone", "Jennifer Lawrence",
        "Halle Berry", "Lupita Nyong'o", "Taraji P. Henson", "Octavia Spencer",
    ],
    "director": [
        "Christopher Nolan", "Quentin Tarantino", "Martin Scorsese", "Steven Spielberg",
        "Stanley Kubrick", "Francis Ford Coppola", "Alfred Hitchcock", "Ridley Scott",
        "James Cameron", "David Fincher", "Wes Anderson", "Greta Gerwig",
        "Denis Villeneuve", "Bong Joon-ho", "Hayao Miyazaki", "Akira Kurosawa",
        "Pedro Almodóvar", "Wong Kar-wai", "Guillermo del Toro", "Jordan Peele",
        "Ari Aster", "Robert Eggers", "Paul Thomas Anderson", "Terrence Malick",
        "David Lynch", "Tim Burton", "Spike Lee", "Sofia Coppola", "Darren Aronofsky",
        "Yorgos Lanthimos", "Ruben Östlund", "Todd Field", "Todd Phillips",
        "Sam Raimi", "Zack Snyder", "J.J. Abrams", "Ron Howard",
    ],
    "genre": [
        "action", "comedy", "drama", "thriller", "horror", "sci-fi", "romance",
        "animation", "fantasy", "western", "war", "crime", "mystery", "noir",
        "heist", "superhero", "musical", "biopic", "adventure", "psychological thriller",
        "slasher", "body horror", "found footage", "mockumentary", "dark comedy",
    ],
},

"tv_show": {
    "channel": [
        "BBC One", "BBC Two", "BBC Four", "ITV", "Channel 4", "Channel 5", "Sky One",
        "CNN", "Fox News", "MSNBC", "NBC", "CBS", "ABC", "HBO", "Showtime",
        "ESPN", "Sky Sports", "BT Sport", "beIN Sports", "Eurosport",
        "National Geographic", "Discovery Channel", "History Channel", "Animal Planet",
        "Cartoon Network", "Disney Channel", "Nickelodeon", "Nick Jr.",
        "MTV", "VH1", "Comedy Central", "TNT", "TBS", "FX",
        "Al Jazeera", "France 24", "Deutsche Welle TV", "Euronews",
        "RAI Uno", "TVE", "ARD", "ZDF", "France 2",
        "Bloomberg Television", "CNBC", "Sky News", "BBC News Channel",
        "TLC", "Bravo", "E!", "Lifetime", "Hallmark Channel",
    ],
    "genre": [
        "news", "sports", "entertainment", "kids", "documentary",
        "music", "cooking", "home", "lifestyle", "comedy",
    ],
},

"video_episodes": {
    "title": [
        "Breaking Bad", "Game of Thrones", "The Wire", "The Sopranos", "Succession",
        "Better Call Saul", "Chernobyl", "Band of Brothers", "True Detective",
        "Fargo", "Black Mirror", "Westworld", "Severance", "Squid Game",
        "The Last of Us", "Yellowstone", "Ozark", "Peaky Blinders", "Narcos",
        "The Crown", "Downton Abbey", "The Mandalorian", "House of Cards",
        "Mad Men", "Six Feet Under", "Deadwood", "The West Wing", "Rome",
        "Friends", "Seinfeld", "The Office", "Parks and Recreation", "30 Rock",
        "Arrested Development", "Community", "Brooklyn Nine-Nine", "Abbott Elementary",
        "It's Always Sunny in Philadelphia", "What We Do in the Shadows",
        "Stranger Things", "Dark", "Mindhunter", "Hannibal", "Dexter",
        "American Horror Story", "The Haunting of Hill House", "Euphoria",
        "The Bear", "Andor", "House of the Dragon", "The Rings of Power",
        "The Boys", "Invincible", "Arcane",
        "The White Lotus", "Fleabag", "Killing Eve", "Normal People",
        "Industry", "Billions", "Suits", "Yellowjackets",
        "The Witcher", "Shadow and Bone", "Bridgerton", "Emily in Paris",
        "Money Heist", "1899", "Lupin", "Fauda",
        "Grey's Anatomy", "Criminal Minds", "NCIS", "CSI", "Law & Order",
        "House MD", "ER", "Scrubs", "Lost", "24",
        "The X-Files", "Twin Peaks", "Battlestar Galactica", "Firefly",
        "Babylon 5", "Star Trek", "Star Trek: The Next Generation",
        "Doctor Who", "Sherlock", "Luther", "Line of Duty",
    ],
    "genre": [
        "drama", "comedy", "thriller", "crime", "sci-fi", "fantasy", "horror",
        "reality", "sitcom", "procedural", "limited series", "miniseries",
        "anthology", "true crime series", "soap opera", "medical drama",
        "legal drama", "period drama",
    ],
},

"anime": {
    "title": [
        "Attack on Titan", "Demon Slayer", "One Piece", "Naruto", "Naruto Shippuden",
        "Dragon Ball Z", "Dragon Ball Super", "My Hero Academia", "Hunter x Hunter",
        "Fullmetal Alchemist Brotherhood", "Death Note", "Steins;Gate", "Cowboy Bebop",
        "Neon Genesis Evangelion", "Sword Art Online", "Tokyo Ghoul",
        "JoJo's Bizarre Adventure", "One Punch Man", "Mob Psycho 100",
        "Vinland Saga", "Chainsaw Man", "Spy x Family", "Jujutsu Kaisen",
        "Made in Abyss", "Re:Zero", "Your Lie in April", "Clannad", "Anohana",
        "Violet Evergarden", "A Silent Voice", "Your Name", "Weathering With You",
        "Spirited Away", "Princess Mononoke", "Nausicaä", "Castle in the Sky",
        "Akira", "Ghost in the Shell", "Berserk", "Trigun", "Outlaw Star",
        "Samurai Champloo", "Bleach", "Fairy Tail", "Black Clover",
        "Haikyuu", "Kuroko's Basketball", "Yuri on Ice", "Free!",
        "Attack on Titan Final Season", "Ranking of Kings", "Frieren",
        "Oshi no Ko", "Bocchi the Rock", "Lycoris Recoil", "Cyberpunk Edgerunners",
    ],
    "genre": [
        "shonen", "shojo", "isekai", "mecha", "slice of life", "fantasy",
        "action", "romance", "horror", "psychological", "sports", "mystery",
        "sci-fi", "seinen", "josei", "magical girl", "harem", "reverse harem",
        "dark fantasy", "historical", "military", "school life", "comedy",
    ],
},

"cartoon": {
    "title": [
        "SpongeBob SquarePants", "Avatar: The Last Airbender", "The Legend of Korra",
        "Gravity Falls", "The Simpsons", "Futurama", "Family Guy", "Rick and Morty",
        "Adventure Time", "Steven Universe", "Regular Show", "The Amazing World of Gumball",
        "Teen Titans", "Teen Titans Go", "Batman: The Animated Series",
        "Justice League", "Justice League Unlimited", "X-Men: The Animated Series",
        "X-Men '97", "DuckTales", "Darkwing Duck", "Garfield", "Looney Tunes",
        "Tom and Jerry", "Scooby-Doo", "Bob's Burgers", "BoJack Horseman",
        "Archer", "Animaniacs", "The Owl House", "Amphibia", "Hilda",
        "Big Mouth", "Human Resources", "South Park", "King of the Hill",
        "Beavis and Butt-Head", "Daria", "Rugrats", "Hey Arnold", "Rocko's Modern Life",
        "Invader Zim", "Fairly OddParents", "Danny Phantom", "Phineas and Ferb",
        "Bluey", "Peppa Pig", "Paw Patrol", "Doc McStuffins",
        "She-Ra", "She-Ra and the Princesses of Power", "Castlevania",
        "The Dragon Prince", "Blood of Zeus",
        "Primal", "Invincible", "Solar Opposites",
    ],
},

"documentary": {
    "title": [
        "Planet Earth", "Blue Planet", "Our Planet", "Life", "Frozen Planet",
        "Planet Earth II", "Seven Worlds One Planet", "Dynasties",
        "Free Solo", "Meru", "The Alpinist", "Valley Uprising", "14 Peaks",
        "Making a Murderer", "The Jinx", "Tiger King", "The Staircase",
        "Amanda Knox", "Don't F**k with Cats", "The Vow", "Keep Sweet",
        "The Last Dance", "Icarus", "Formula 1: Drive to Survive", "Senna",
        "Diego Maradona", "Pelé", "Fire of Love",
        "Seaspiracy", "Blackfish", "Cowspiracy", "An Inconvenient Truth",
        "Before the Flood", "David Attenborough: A Life on Our Planet",
        "Super Size Me", "Food Inc.", "Fed Up", "Forks Over Knives",
        "13th", "I Am Not Your Negro", "Won't You Be My Neighbor?", "RBG",
        "Jiro Dreams of Sushi", "Street Food", "Chef's Table", "Ugly Delicious",
        "Amy", "20 Feet from Stardom", "Searching for Sugar Man", "Shut Up and Sing",
        "My Octopus Teacher", "Crip Camp", "All That Breathes",
        "The Rescue", "Fauci", "We Need to Talk About Cosby", "Allen v. Farrow",
        "Wild Wild Country", "The Keepers", "The Confession Tapes", "Evil Genius",
    ],
    "topic": [
        "nature", "wildlife", "ocean", "space", "history", "crime", "politics",
        "food", "sport", "music", "technology", "climate change", "social justice",
        "biography", "war", "science", "art", "culture", "true crime",
        "business", "health", "environment", "animals", "cooking",
    ],
},

"podcast": {
    "show": [
        "Serial", "This American Life", "Radiolab", "99% Invisible", "Freakonomics Radio",
        "The Joe Rogan Experience", "Lex Fridman Podcast", "How I Built This",
        "Masters of Scale", "StartUp Podcast", "Crime Junkie", "My Favorite Murder",
        "Casefile", "Last Podcast on the Left", "True Crime Garage", "Morbid",
        "And That's Why We Drink", "The Daily", "Up First", "Planet Money", "Fresh Air",
        "Throughline", "Hidden Brain", "Invisibilia", "On Being", "Armchair Expert",
        "SmartLess", "Conan O'Brien Needs a Friend", "My Brother My Brother and Me",
        "Stuff You Should Know", "Stuff You Missed in History Class",
        "No Such Thing as a Fish", "The Bugle",
        "The Tim Ferriss Show", "Diary of a CEO", "The Knowledge Project",
        "The James Altucher Show", "The GaryVee Audio Experience",
        "Call Her Daddy", "Anything Goes with Emma Chamberlain", "The Moth",
        "Risk!", "The Moth Radio Hour", "Story Collider",
        "Welcome to Night Vale", "Wolf 359", "Limetown", "Bubble",
        "Hardcore History", "Revolutions", "History Extra", "In Our Time",
        "Darknet Diaries", "Reply All", "Gimlet", "Wondery presents",
        "Huberman Lab", "Found My Fitness", "Feel Better Live More",
        "Maintenance Phase", "You're Wrong About", "Conspirituality",
        "Unlocking Us", "We Can Do Hard Things", "The Happiness Lab",
    ],
    "topic": [
        "true crime", "comedy", "tech", "history", "science", "business",
        "self-help", "politics", "sports", "culture", "storytelling", "interview",
        "fiction", "horror", "news", "health", "psychology", "economics",
        "philosophy", "religion", "mystery", "language", "travel",
    ],
},

"radio": {
    "station": [
        "BBC Radio 1", "BBC Radio 2", "BBC Radio 3", "BBC Radio 4", "BBC Radio 5 Live",
        "BBC Radio 6 Music", "BBC World Service", "NPR", "WBEZ", "KCRW", "KEXP", "WNYC",
        "Heart FM", "Capital FM", "Kiss FM", "Absolute Radio", "TalkSPORT", "LBC",
        "Classic FM", "Magic FM", "Smooth Radio",
        "Radio Nova", "NRJ", "Fun Radio", "Skyrock", "Europe 1", "France Inter",
        "France Musique", "RMC", "RTL",
        "Radio Nacional España", "COPE", "Cadena SER", "Europa FM", "Onda Cero",
        "Bayern 3", "SWR3", "HR3", "NDR 2", "Deutschlandfunk", "Radio Eins",
        "Radio 2 Netherlands", "Radio 538", "3FM", "Radio 1 Belgium",
        "iHeartRadio", "SiriusXM", "Pandora Radio", "Radio Paradise", "AccuRadio",
        "Radio Mirchi", "All India Radio", "Radio City India",
        "RNE Radio Nacional", "Radio Nacional Argentina",
    ],
    "genre": [
        "pop radio", "rock radio", "jazz radio", "classical radio", "talk radio",
        "news radio", "sports radio", "country radio", "R&B radio", "hip-hop radio",
        "electronic radio", "oldies radio", "Christian radio", "public radio",
    ],
},

"radio_theatre": {
    "show": [
        "The Hitchhiker's Guide to the Galaxy BBC Radio",
        "Cabin Pressure", "The Goon Show", "I'm Sorry I'll Read That Again",
        "Just a Minute", "The Archers", "Desert Island Discs",
        "Sherlock Holmes BBC Radio", "War of the Worlds Orson Welles",
        "Sorry Wrong Number", "Suspense old time radio", "The Shadow",
        "The Lone Ranger radio", "Gunsmoke radio", "Dragnet radio",
        "Escape radio show", "Dimension X", "X Minus One",
        "Inner Sanctum Mysteries", "Jack Benny Program",
        "Fibber McGee and Molly", "Lux Radio Theatre",
        "Radio Mystery Theater", "The Adventures of Superman radio",
        "Dick Tracy radio", "The Green Hornet radio",
        "Old Harry's Game", "The Quangas", "Mark Steel's in Town",
    ],
    "genre": [
        "mystery", "comedy", "horror", "sci-fi", "adventure", "drama",
        "thriller", "western", "crime", "romance", "historical drama",
        "old time radio", "classic radio",
    ],
},

"audiobook": {
    "title": [
        "Harry Potter", "The Lord of the Rings", "Dune", "Foundation", "Neuromancer",
        "A Song of Ice and Fire", "Game of Thrones", "The Wheel of Time",
        "The Kingkiller Chronicle", "The Name of the Wind", "Words of Radiance",
        "1984", "Brave New World", "Fahrenheit 451", "Animal Farm",
        "To Kill a Mockingbird", "Of Mice and Men", "The Great Gatsby",
        "Moby Dick", "Crime and Punishment", "War and Peace", "Anna Karenina",
        "The Hitchhiker's Guide to the Galaxy", "Good Omens", "Discworld",
        "Sapiens", "A Brief History of Time", "Cosmos", "The Selfish Gene",
        "Thinking Fast and Slow", "Atomic Habits", "The Power of Habit",
        "Man's Search for Meaning", "The Subtle Art of Not Giving a F*ck",
        "The Alchemist", "The Little Prince",
        "Gone Girl", "Girl on the Train", "The Girl with the Dragon Tattoo",
        "And Then There Were None", "Murder on the Orient Express",
        "Big Little Lies", "The Lovely Bones", "Gone with the Wind",
        "Ender's Game", "Ready Player One", "Project Hail Mary",
        "The Martian", "Recursion", "Dark Matter", "The Three-Body Problem",
        "Lessons in Chemistry", "Tomorrow and Tomorrow and Tomorrow",
        "Demon Copperhead", "Trust", "The Covenant of Water",
        "Fourth Wing", "A Court of Thorns and Roses", "The Midnight Library",
    ],
    "author": [
        "Stephen King", "Agatha Christie", "J.R.R. Tolkien", "Frank Herbert",
        "Brandon Sanderson", "Neil Gaiman", "Terry Pratchett", "Douglas Adams",
        "George R.R. Martin", "Robert Jordan", "Patrick Rothfuss",
        "James Patterson", "Michael Connelly", "Lee Child", "John Grisham",
        "Toni Morrison", "Cormac McCarthy", "Kazuo Ishiguro", "Haruki Murakami",
        "Yuval Noah Harari", "Malcolm Gladwell", "Michael Lewis", "Naomi Klein",
        "Brené Brown", "James Clear", "Ryan Holiday", "Tim Ferriss",
        "Dan Brown", "John Sandford", "Lisa Gardner", "Karin Slaughter",
        "Colleen Hoover", "Sarah J. Maas", "Taylor Jenkins Reid",
        "Celeste Ng", "Jojo Moyes", "Nicholas Sparks", "Nora Roberts",
    ],
    "narrator": [
        "Jim Dale", "Stephen Fry", "Frank Muller", "Roy Dotrice",
        "Scott Brick", "Kate Reading", "Michael Kramer", "January LaVoy",
        "Wil Wheaton", "George Guidall", "Mark Bramhall", "Jefferson Mays",
        "Nick Offerman", "Tina Fey", "Trevor Noah", "Michelle Obama",
    ],
    "genre": [
        "thriller", "mystery", "sci-fi", "fantasy", "romance", "historical fiction",
        "literary fiction", "horror", "biography", "history", "self-help",
        "business", "philosophy", "science", "true crime", "adventure",
    ],
},

"news": {
    "provider": [
        "BBC News", "CNN", "NPR", "Al Jazeera", "Reuters", "Associated Press",
        "Sky News", "CBS News", "NBC News", "ABC News", "Bloomberg", "Financial Times",
        "France 24", "Deutsche Welle", "Euronews", "RFI", "RTVE", "NHK World",
        "The Guardian", "The New York Times", "The Washington Post",
        "The Daily by New York Times", "Up First NPR", "Today in Focus Guardian",
        "Global News Podcast BBC", "Monocle 24", "Fox News", "MSNBC",
        "Democracy Now", "The Intercept", "Axios", "Politico", "Vox",
    ],
    "topic": [
        "world news", "local news", "sports news", "tech news", "business news",
        "politics", "entertainment news", "science news", "health news",
        "climate news", "breaking news", "international news", "US news",
        "UK news", "European news", "Middle East news", "Asian news",
    ],
    "time": [
        "morning news", "evening news", "tonight's headlines", "morning briefing",
        "daily bulletin", "weekend news", "latest headlines", "breaking news",
        "lunchtime news", "hourly news update", "five-minute news",
    ],
},

"game": {
    "title": [
        "The Legend of Zelda", "The Legend of Zelda: Breath of the Wild",
        "The Legend of Zelda: Tears of the Kingdom", "Minecraft", "The Witcher 3",
        "Elden Ring", "Red Dead Redemption 2", "Grand Theft Auto V", "Call of Duty",
        "FIFA", "EA Sports FC", "Cyberpunk 2077", "God of War", "God of War Ragnarök",
        "Spider-Man", "Spider-Man 2", "Halo", "Halo Infinite",
        "Mario Kart", "Super Mario Odyssey", "Super Mario Bros", "Super Mario 64",
        "The Last of Us", "The Last of Us Part II", "Ghost of Tsushima",
        "Horizon Zero Dawn", "Horizon Forbidden West",
        "Dark Souls", "Dark Souls III", "Bloodborne", "Sekiro", "Demon's Souls",
        "Stardew Valley", "Among Us", "Fortnite", "PUBG",
        "Valorant", "League of Legends", "Dota 2", "Overwatch", "Overwatch 2",
        "Counter-Strike", "Counter-Strike 2", "Apex Legends", "Rocket League",
        "Final Fantasy XIV", "Final Fantasy XVI", "Dragon Age", "Mass Effect",
        "Baldur's Gate 3", "Diablo IV", "World of Warcraft", "Starcraft II",
        "Civilization VI", "Crusader Kings III", "Europa Universalis IV",
        "Cities: Skylines", "Planet Coaster", "The Sims 4",
        "Portal 2", "Half-Life: Alyx", "Bioshock Infinite", "Skyrim",
        "Fallout 4", "Fallout: New Vegas", "Celeste", "Hollow Knight",
        "Hades", "Dead Cells", "Cuphead", "Ori and the Blind Forest",
        "It Takes Two", "A Way Out", "Divinity: Original Sin 2",
        "Disco Elysium", "Planescape: Torment", "Pillars of Eternity",
        "Resident Evil Village", "Resident Evil 4 Remake", "Silent Hill 2",
        "Alien: Isolation", "Five Nights at Freddy's", "Phasmophobia",
    ],
    "platform": [
        "PlayStation", "PS5", "PS4", "Xbox", "Xbox Series X", "Nintendo Switch",
        "PC", "Steam", "Epic Games Store", "Game Pass", "PlayStation Now",
        "GOG", "Origin", "Uplay", "Battle.net",
    ],
    "genre": [
        "RPG", "action RPG", "FPS", "strategy", "puzzle", "platformer",
        "open world", "racing", "sports", "survival", "simulation",
        "roguelike", "fighting", "adventure", "MMORPG", "battle royale",
        "stealth", "horror", "metroidvania", "soulslike", "city builder",
        "turn-based", "real-time strategy", "visual novel",
    ],
},

"short_film": {
    "genre": [
        "drama", "comedy", "animation", "experimental", "horror", "documentary",
        "thriller", "romance", "sci-fi", "fantasy", "action", "mockumentary",
    ],
    "platform": [
        "Vimeo", "YouTube", "Mubi", "Criterion Channel", "Short of the Week",
        "Sundance Now", "FilmFreeway", "Pexels Films",
    ],
    "context": [
        "Oscar-winning short", "Sundance short", "Cannes short",
        "animated short", "student short film", "indie short",
        "BAFTA short", "Pixar short", "Aardman short",
    ],
},

"silent_movie": {
    "title": [
        "City Lights", "The General", "Modern Times", "The Kid", "The Gold Rush",
        "The Circus", "Nosferatu", "Metropolis", "The Cabinet of Dr. Caligari",
        "Sunrise: A Song of Two Humans", "Safety Last!", "The Navigator",
        "Sherlock Jr.", "The Phantom of the Carriage", "Battleship Potemkin",
        "The Passion of Joan of Arc", "A Trip to the Moon", "The Birth of a Nation",
        "Intolerance", "Greed", "The Crowd", "Flesh and the Devil",
    ],
    "actor": [
        "Charlie Chaplin", "Buster Keaton", "Harold Lloyd", "Lillian Gish",
        "Mary Pickford", "Rudolph Valentino", "Douglas Fairbanks", "Joan Crawford",
        "Clara Bow", "Louise Brooks", "Lon Chaney",
    ],
    "director": [
        "F.W. Murnau", "Fritz Lang", "D.W. Griffith", "Sergei Eisenstein",
        "Georges Méliès", "King Vidor", "Cecil B. DeMille", "Erich von Stroheim",
    ],
},

"bw_movie": {
    "title": [
        "12 Angry Men", "Psycho", "It Happened One Night", "Some Like It Hot",
        "Double Indemnity", "The Third Man", "Citizen Kane", "Casablanca",
        "Roman Holiday", "To Kill a Mockingbird", "The Maltese Falcon",
        "Sunset Boulevard", "All About Eve", "The Grapes of Wrath",
        "It's a Wonderful Life", "The Night of the Hunter", "Rebecca",
    ],
    "actor": [
        "Humphrey Bogart", "Cary Grant", "Clark Gable", "Katharine Hepburn",
        "Audrey Hepburn", "Marilyn Monroe", "James Stewart", "Bette Davis",
        "Henry Fonda", "Marlon Brando", "Grace Kelly", "Ingrid Bergman",
        "Spencer Tracy", "Joan Fontaine", "Gary Cooper", "Barbara Stanwyck",
    ],
    "director": [
        "Alfred Hitchcock", "Orson Welles", "John Ford", "Howard Hawks",
        "Billy Wilder", "Frank Capra", "William Wyler", "Elia Kazan",
        "Fritz Lang", "Jean Renoir", "Vittorio De Sica",
    ],
},

"audio": {
    "sound": [
        "rain sounds", "white noise", "ocean waves", "thunderstorm",
        "forest sounds", "train sounds", "cafe noise", "bird songs",
        "pink noise", "brown noise", "fan noise", "crackling fireplace",
        "jungle sounds", "city traffic noise", "airplane cabin noise",
    ],
    "purpose": [
        "for sleeping", "for studying", "for focus", "for relaxation",
        "to block out noise", "for meditation", "to help me sleep",
        "for reading", "to calm down",
    ],
},

"asmr": {
    "trigger": [
        "tapping", "whispering", "scratching", "page turning",
        "keyboard typing", "hair brushing", "eating sounds", "roleplay",
        "crinkling", "wood soup", "mic brushing", "water sounds",
        "inaudible whispering", "personal attention", "makeup roleplay",
    ],
    "creator": [
        "Gibi ASMR", "FrivolousFox", "ASMR Darling", "Latte ASMR",
        "Goodnight Moon", "Gentle Whispering", "Frenda", "Ephemeral Rift",
        "Tingting ASMR", "GwenGwiz", "ASMR Bakery", "WhispersRed",
    ],
},

"audio_description": {
    "content": [
        "a movie with audio description", "the visually impaired track",
        "a described video", "descriptive audio", "an audio described show",
        "a movie for the blind", "the descriptive video service track",
    ],
},

"visual_story": {
    "title": [
        "Batman", "Spider-Man", "The Sandman", "Watchmen", "Maus", "Saga",
        "X-Men", "The Avengers", "Invincible", "The Walking Dead", "Superman",
        "Wonder Woman", "Scott Pilgrim", "V for Vendetta", "Sin City",
        "Hellboy", "Spawn", "Teenage Mutant Ninja Turtles", "The Flash",
        "Justice League", "Deadpool", "Calvin and Hobbes", "Garfield",
        "Peanuts", "Persepolis", "Bone", "Y: The Last Man"
    ],
    "type": [
        "comic", "comic book", "graphic novel", "webcomic", "motion comic",
        "digital comic", "visual story", "comic strip", "manga", "manhwa"
    ],
    "creator": [
        "Stan Lee", "Jack Kirby", "Alan Moore", "Neil Gaiman", "Frank Miller",
        "Brian K. Vaughan", "Todd McFarlane", "Jim Lee", "Art Spiegelman",
        "Marjane Satrapi", "Robert Kirkman", "Bill Watterson", "Charles Schulz"
    ]
},

"adult_audio": {
    "type": [
        "erotica", "spicy audiobook", "steamy romance", "nsfw audio",
        "adult audio story", "romantic audio", "sensual audio",
        "erotic fiction", "adult stories",
    ],
}

}


# ---------------------------------------------------------------------------
# Sentence Templates (Mapping Intents to Utterance Structures)
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, list[str]] = {
    "music": [
        "{play_verb_audio} {song} by {artist}",
        "{play_verb_audio} the song {song} by {artist}",
        "{play_verb_audio} the album {album} by {artist}",
        "{play_verb_audio} {album} by {artist}",
        "{play_verb_audio} some {genre} music",
        "{play_verb_audio} {genre}",
        "{play_verb_audio} {artist}",
        "{play_verb_audio} songs by {artist}",
        "{play_verb_audio} music by {artist}",
        "{play_verb_audio} the latest album by {artist}",
        "{play_verb_audio} some {mood} music",
        "{play_verb_audio} a {mood} playlist",
        "{play_verb_audio} {mood} songs",
        "{play_verb_audio} music for {activity}",
        "{play_verb_audio} a playlist for {activity}",
        "{play_verb_audio} {genre} for {activity}",
        "{play_verb_audio} {song}",
        "{play_verb_audio} the track {song}",
        "I want to hear {song} by {artist}",
        "Can you put on {album} from {artist}",
        "Shuffle {artist}",
        "Play {artist} on shuffle",
    ],
    "movie": [
        "{play_verb_video} the movie {title}",
        "{play_verb_video} the film {title}",
        "{play_verb_video} {title}",
        "{play_verb_video} a {genre} movie",
        "{play_verb_video} a {genre} film",
        "{play_verb_video} some {genre} movies",
        "{play_verb_video} a movie starring {actor}",
        "{play_verb_video} a film with {actor}",
        "{play_verb_video} the {genre} movie with {actor}",
        "{play_verb_video} a movie directed by {director}",
        "{play_verb_video} a {director} film",
        "{play_verb_video} the new {director} movie",
        "{play_verb_video} a {genre} film directed by {director}",
        "I feel like watching {title}",
        "Put on the movie {title}",
    ],
    "tv_show": [
        "{play_verb_video} {channel}",
        "{play_verb_video} channel {channel}",
        "turn on {channel}",
        "switch to {channel}",
        "put {channel} on the tv",
        "{play_verb_video} some {genre} on tv",
        "{play_verb_video} live tv on {channel}",
        "watch {channel} live",
        "change the channel to {channel}",
    ],
    "video_episodes": [
        "{play_verb_video} the show {title}",
        "{play_verb_video} the series {title}",
        "{play_verb_video} {title}",
        "{play_verb_video} an episode of {title}",
        "{play_verb_video} the next episode of {title}",
        "{play_verb_video} the {genre} series {title}",
        "{play_verb_video} the {genre} show {title}",
        "{play_verb_video} a {genre} series",
        "I'd like to watch the show {title}",
        "put on {title}",
        "continue watching {title}",
    ],
    "anime": [
        "{play_verb_video} the anime {title}",
        "{play_verb_video} {title}",
        "{play_verb_video} an episode of {title}",
        "{play_verb_video} the next episode of the anime {title}",
        "{play_verb_video} some {genre} anime",
        "{play_verb_video} a {genre} anime series",
        "I want to watch the anime {title}",
        "put on the anime {title}",
    ],
    "cartoon": [
        "{play_verb_video} the cartoon {title}",
        "{play_verb_video} {title}",
        "{play_verb_video} an episode of {title}",
        "{play_verb_video} the next episode of {title}",
        "put on {title} for the kids",
        "play the {title} cartoon",
    ],
    "documentary": [
        "{play_verb_video} the documentary {title}",
        "{play_verb_video} {title}",
        "{play_verb_video} a documentary about {topic}",
        "{play_verb_video} a {topic} documentary",
        "{play_verb_video} a {topic} doc",
        "I want to watch a documentary on {topic}",
        "show me a nature documentary like {title}",
        "put on a documentary",
    ],
    "podcast": [
        "{play_verb_audio} the podcast {show}",
        "{play_verb_audio} {show}",
        "{play_verb_audio} the {show} podcast",
        "{play_verb_audio} the latest episode of {show}",
        "{play_verb_audio} an episode of {show}",
        "{play_verb_audio} a podcast about {topic}",
        "{play_verb_audio} a {topic} podcast",
        "I want to listen to the {show} podcast",
        "play the next episode of {show}",
    ],
    "radio": [
        "{play_verb_audio} the radio station {station}",
        "{play_verb_audio} {station}",
        "{play_verb_audio} {station} radio",
        "tune in to {station}",
        "turn the radio to {station}",
        "{play_verb_audio} some {genre} on the radio",
        "{play_verb_audio} {genre} radio",
        "listen to {station}",
        "start playing {station}",
    ],
    "radio_theatre": [
        "{play_verb_audio} the radio play {show}",
        "{play_verb_audio} the audio drama {show}",
        "{play_verb_audio} {show}",
        "{play_verb_audio} a {genre} radio drama",
        "{play_verb_audio} a {genre} radio play",
        "put on some old time radio",
        "I want to listen to the radio show {show}",
        "play the classic radio drama {show}",
    ],
    "audiobook": [
        "{play_verb_audio} the audiobook {title}",
        "{play_verb_audio} the book {title}",
        "{play_verb_audio} {title} by {author}",
        "{play_verb_audio} the audiobook {title} by {author}",
        "read {title} narrated by {narrator}",
        "{play_verb_audio} {title} narrated by {narrator}",
        "{play_verb_audio} a {genre} audiobook",
        "{play_verb_audio} a book by {author}",
        "{play_verb_audio} an audiobook by {author}",
        "resume the audiobook {title}",
    ],
    "news": [
        "{play_verb_audio} the {time} from {provider}",
        "{play_verb_audio} the {time}",
        "{play_verb_audio} the {topic} from {provider}",
        "{play_verb_audio} the {topic}",
        "what's the latest {topic} from {provider}",
        "what's the latest {topic}",
        "give me the {time}",
        "{play_verb_audio} the news from {provider}",
        "{play_verb_audio} {provider} news",
        "what is the news today",
        "read me the {time}",
    ],
    "game": [
        "{play_verb} the game {title}",
        "{play_verb} {title}",
        "{play_verb} {title} on {platform}",
        "I want to play {title} on my {platform}",
        "start {title}",
        "launch {title}",
        "{play_verb} a {genre} game",
        "let's play {title}",
        "open {title} on {platform}",
    ],
    "short_film": [
        "{play_verb_video} a {genre} short film",
        "{play_verb_video} a {genre} short",
        "{play_verb_video} a {context} on {platform}",
        "{play_verb_video} a {context}",
        "show me a short {genre} movie",
        "{play_verb_video} a short film",
    ],
    "silent_movie": [
        "{play_verb_video} the silent film {title}",
        "{play_verb_video} the silent movie {title}",
        "{play_verb_video} a silent movie starring {actor}",
        "{play_verb_video} a silent film with {actor}",
        "{play_verb_video} a silent film directed by {director}",
        "I want to watch the silent film {title}",
        "put on {title}",
    ],
    "bw_movie": [
        "{play_verb_video} the black and white movie {title}",
        "{play_verb_video} the classic film {title}",
        "{play_verb_video} a black and white movie starring {actor}",
        "{play_verb_video} an old {director} movie",
        "I want to watch a classic black and white film with {actor}",
        "{play_verb_video} the black and white film {title}",
    ],
    "audio": [
        "{play_verb_audio} {sound}",
        "{play_verb_audio} {sound} {purpose}",
        "I need {sound} {purpose}",
        "put on some {sound}",
        "start {sound}",
        "{play_verb_audio} some {sound} {purpose}",
        "turn on {sound}",
    ],
    "asmr": [
        "{play_verb_audio} some ASMR",
        "{play_verb_audio} {trigger} ASMR",
        "{play_verb_audio} an ASMR video by {creator}",
        "{play_verb_audio} {creator} ASMR",
        "I want to listen to {creator} doing {trigger}",
        "put on some {trigger} ASMR by {creator}",
        "{play_verb_audio} ASMR for sleeping",
    ],
    "audio_description": [
        "{play_verb_audio} {content}",
        "turn on {content}",
        "enable {content}",
        "I need {content} for this movie",
        "switch to {content}",
    ],
    "visual_story": [
        "read the {type} {title}",
        "read {title}",
        "open the {title} {type}",
        "show me the {title} {type}",
        "{play_verb_video} the {title} motion comic",
        "{play_verb_video} the {type} {title}",
        "I want to read {title}",
        "I want to read the {title} {type}",
        "read the {type} by {creator}",
        "show me a {type} by {creator}",
        "{play_verb_video} some {type}s",
        "open a {type}",
        "read the latest issue of {title}",
        "show me the {title} comic book",
    ],
    "adult_audio": [
        "{play_verb_audio} some {type}",
        "{play_verb_audio} a {type}",
        "put on a {type}",
        "I want to listen to {type}",
    ],
}

# ---------------------------------------------------------------------------
# Generation Logic
# ---------------------------------------------------------------------------

def _fill(tmpl: str, entities: dict[str, list[str]], rng: random.Random) -> Optional[str]:
    keys = re.findall(r'\{([^\}]+)\}', tmpl)
    kwargs = {}
    for k in keys:
        if k == "play_verb":
            kwargs[k] = rng.choice(_PLAY_VERBS)
        elif k == "play_verb_audio":
            kwargs[k] = rng.choice(_PLAY_VERBS_AUDIO)
        elif k == "play_verb_video":
            kwargs[k] = rng.choice(_PLAY_VERBS_VIDEO)
        elif k in entities:
            kwargs[k] = rng.choice(entities[k])
        else:
            return None  # Missing entity mapping
    return tmpl.format(**kwargs)


def generate_intent(intent: str, n: int, seen: set[str], seed: int) -> list[tuple]:
    rng = random.Random(f"{seed}_{intent}")
    domain = "ocp_play"
    binary, playback, media = _labels(domain, intent)

    templates = _TEMPLATES.get(intent, [])
    entities = _ENTITIES.get(intent, {})

    rows = []
    attempts = 0
    max_attempts = n * 10

    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        tmpl = rng.choice(templates)
        sentence = _fill(tmpl, entities, rng)
        if sentence is None:
            continue

        sentence = sentence.lower().strip()
        # Clean punctuation to enforce keyword-style format
        sentence = "".join(c for c in sentence if c not in ".,?!")
        sentence = " ".join(sentence.split())

        if not sentence or sentence in seen:
            continue

        seen.add(sentence)
        rows.append(("en", domain, intent, binary, playback, media, sentence))

    return rows


def generate_all(
    intents: Optional[list[str]] = None,
    n: int = 3000,
    dedup_against: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    existing: set[str] = set()
    if dedup_against and os.path.exists(dedup_against):
        df_ex = pd.read_csv(dedup_against)
        if "sentence" in df_ex.columns:
            existing = set(df_ex["sentence"].dropna().str.lower())
            print(f"Deduplicating against {len(existing):,} existing sentences")

    selected = intents or list(_TEMPLATES.keys())
    all_rows: list[tuple] = []

    for intent in selected:
        if intent not in _TEMPLATES:
            print(f"  [skip] {intent} — no templates defined")
            continue
        rows = generate_intent(intent, n, existing, seed)
        existing.update(r[6] for r in rows)
        print(f"  {intent:<20} {len(rows):>5} utterances")
        all_rows.extend(rows)

    cols = ["lang", "domain", "intent", "binary_label", "playback_label", "media_label", "sentence"]
    return pd.DataFrame(all_rows, columns=cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic OCP keyword data")
    parser.add_argument("--intents", nargs="+", help="Specific intents to generate")
    parser.add_argument("--n", type=int, default=3000, help="Number of utterances per intent")
    parser.add_argument("--dedup-against", type=str, help="Path to existing CSV to deduplicate against")
    args = parser.parse_args()

    print("Generating keyword templates...")
    df = generate_all(intents=args.intents, n=args.n, dedup_against=args.dedup_against)

    out_dir = get_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_keywords.csv")

    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} total rows to {out_path}")