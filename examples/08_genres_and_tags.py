#!/usr/bin/env python3
"""08 — Genres and tags: the orthogonal signal.

Genres are orthogonal to ``MediaType``. They carry the nuance that the type
taxonomy deliberately collapses: "anime" and "cartoon" both map to the
``episodic_series`` MediaType, but the genre tags keep them distinct
(``anime`` vs ``animation``). Crucially, genres also carry the ``adult`` signal
the content filter blocks on by default (see example 03).

``classify_genres(query, lang) -> list[str]`` returns ``mediavocab`` genre tags
implied by the winning keyword intent. Every emitted tag is a real member of
``mediavocab``'s ``KNOWN_GENRES`` — the package enforces the shared taxonomy, so
you never get an ad-hoc string.

Note how the *public* MediaType can look generic (a hentai query is just
``episodic_series``) while the genres surface the real, filterable nature
(``['anime', 'adult']``).

Run:  python examples/08_genres_and_tags.py
"""
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier()

# Utterances chosen to surface the genre tags the keyword backend knows:
# anime, animation, asmr, adult.
utterances = [
    "play anime",          # ['anime']
    "watch a cartoon",     # ['animation']
    "play asmr sounds",    # ['asmr']
    "play porn",           # ['adult']
    "play hentai",         # ['anime', 'adult']  <- two orthogonal tags at once
    "play some music",     # []  no genre signal
    "play the movie",      # []
]

print("utterance".ljust(24), "MediaType".ljust(18), "genres")
print("-" * 60)
for utt in utterances:
    media_type, _ = clf.classify(utt, "en-us")
    genres = clf.classify_genres(utt, "en-us")
    print(f"{utt:<24} {media_type.value:<18} {genres}")

print()
print("The MediaType alone is not enough to moderate content: 'play hentai'")
print("looks like a plain episodic_series until you read its genres.")
