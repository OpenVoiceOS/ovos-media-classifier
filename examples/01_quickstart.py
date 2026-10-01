#!/usr/bin/env python3
"""01 — Quickstart: load the classifier and classify a few utterances.

``load_media_classifier()`` with no arguments returns the bundled ``.voc``
keyword classifier: zero ML dependencies, no model files, fully offline. It
reads the keyword vocabularies shipped under
``ovos_media_classifier/locale/<lang>/*.voc`` directly off disk.

The core call is::

    media_type, confidence = clf.classify(utterance, lang)

``media_type`` is a ``mediavocab.MediaType`` (a str-Enum) and ``confidence`` is
a float in [0, 1]. A keyword match returns ~0.4-0.7 depending on how specific
the match is; no match returns ``(MediaType.GENERIC, 0.0)``.

Run:  python examples/01_quickstart.py
"""
from ovos_media_classifier import load_media_classifier

# No config -> bundled keyword classifier. Nothing to download, no model files.
clf = load_media_classifier()

# A grab-bag of utterances. The keyword classifier matches on substrings of the
# (lowercased) utterance, so each of these contains a media keyword.
utterances = [
    "play some music",
    "play the movie inception",
    "watch live tv channel",
    "play a podcast",
    "read me an audiobook",
    "what time is it",        # NOT a media request -> GENERIC, 0.0
]

print("utterance".ljust(32), "MediaType", "  confidence")
print("-" * 56)
for utt in utterances:
    # classify() is the one method every backend must implement.
    media_type, conf = clf.classify(utt, "en-us")
    # media_type.value is the canonical string ("music", "movie", ...).
    print(f"{utt:<32} {media_type.value:<10} {conf}")
