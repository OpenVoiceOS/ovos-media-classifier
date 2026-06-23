#!/usr/bin/env python3
"""03 — Content filtering: block sensitive (adult) media by default.

The *primary driver* for recognising sensitive media is to let OVOS **block
such requests by default** — this is content moderation / parental control, not
a content provider. The classifier surfaces a ``mediavocab`` genre signal
(``adult`` for adult/hentai/porn queries) and :class:`ContentFilter` decides
whether the request is allowed.

``ContentFilter(config).check(clf, query, lang)`` returns ``(blocked, reason)``:
it classifies the query with *clf*, pulls the genre tags, and applies the
blocklist.

Default policy: ``adult`` is blocked. The operator lifts it with the top-level
``allow_adult_content: true`` flag, or customises the blocklist under
``media_content_filter``.

Run:  python examples/03_content_filter.py
"""
from ovos_media_classifier import load_media_classifier, ContentFilter

clf = load_media_classifier()


def show(title, config, queries):
    print(title)
    cf = ContentFilter(config)
    for q in queries:
        blocked, reason = cf.check(clf, q, "en-us")
        verdict = "BLOCKED" if blocked else "allowed"
        print(f"    {q:<26} -> {verdict:<8} {reason}")
    print()


# 1) Default policy — no config. Adult is blocked, everything else allowed.
show(
    "1) Default policy (adult blocked):",
    config=None,
    queries=["play porn", "play hentai", "play some music", "play a movie"],
)

# 2) Operator opt-in — the top-level allow_adult_content flag lifts the block.
show(
    "2) allow_adult_content: true (adult now allowed):",
    config={"allow_adult_content": True},
    queries=["play porn", "play hentai"],
)

# 3) Custom blocklist — block anime too (any mediavocab genre tag works). Here
#    we drop the default adult block and instead block the 'anime' genre, so a
#    hentai query (genres=['anime', 'adult']) is still caught via 'anime'.
show(
    "3) Custom blocked_genres ['anime'] (adult lifted, anime blocked):",
    config={
        "allow_adult_content": True,
        "media_content_filter": {"blocked_genres": ["anime"]},
    },
    queries=["play anime", "play hentai", "play porn"],
)

# 4) You can also call the lower-level is_blocked() directly if you already have
#    a (media_type, genres) pair and don't want to re-classify.
mt, _ = clf.classify("play porn", "en-us")
genres = clf.classify_genres("play porn", "en-us")
blocked, reason = ContentFilter().is_blocked(mt, genres)
print(f"4) is_blocked({mt.value}, {genres}) -> {blocked} ({reason})")
