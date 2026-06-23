#!/usr/bin/env python3
"""06 — Pipeline mode: supplying an external ``.voc`` matcher.

The keyword classifier works in two modes:

1. **Standalone** (examples 01-04): ``load_media_classifier()`` reads the
   bundled ``.voc`` locale files off disk itself.

2. **Pipeline**: the OCP pipeline plugin already owns the locale files and a
   ``voc_match`` method (from ``OVOSAbstractApplication``). Instead of having
   the classifier re-load vocab, you pass that function in via
   ``load_media_classifier(voc_match_func=...)`` and the classifier just calls
   it. This avoids loading the vocab twice and lets the pipeline's own resource
   resolution win.

The matcher signature is::

    voc_match(phrase: str, vocab_name: str, *, lang: str) -> bool

returning True if any keyword in ``<vocab_name>.voc`` appears in ``phrase``.

Below we fake a tiny in-memory matcher so the example is self-contained; in a
real skill you would pass ``self.voc_match``.

Run:  python examples/06_pipeline_voc_match.py
"""
from ovos_media_classifier import load_media_classifier

# A stand-in for the pipeline's vocab store. In production these come from the
# pipeline plugin's locale/ resources; here we hand-roll a few so the example
# needs nothing on disk. Keys are vocab_name -> list of keywords.
FAKE_VOCAB = {
    "MusicKeyword": ["music", "song", "tune"],
    "MovieKeyword": ["movie", "film"],
    "RadioKeyword": ["radio"],
    "PodcastKeyword": ["podcast"],
    # everything else is intentionally empty -> never matches
}


def my_voc_match(phrase: str, vocab_name: str, *, lang: str) -> bool:
    """An external matcher exactly like a pipeline's ``voc_match``.

    Note the keyword-only ``lang`` argument — the classifier always calls the
    matcher as ``voc_match(phrase, vocab, lang=lang)``.
    """
    phrase = phrase.lower()
    return any(kw in phrase for kw in FAKE_VOCAB.get(vocab_name, ()))


# Hand the matcher to the factory. No bundled locale files are read at all —
# only the keywords our matcher knows about can ever fire.
clf = load_media_classifier(voc_match_func=my_voc_match)

for utt in [
    "play some music",       # MusicKeyword -> music
    "play my favourite tune",  # 'tune' is in our fake vocab -> music
    "play the film",         # MovieKeyword -> movie
    "play the radio",        # RadioKeyword -> radio
    "play a podcast",        # PodcastKeyword -> podcast
    "play an audiobook",     # not in our tiny vocab -> GENERIC
]:
    media_type, conf = clf.classify(utt, "en-us")
    print(f"{utt:<26} -> {media_type.value:<10} conf={conf}")

print()
print("Only the vocabularies our matcher knows fire; 'audiobook' is GENERIC")
print("because we never taught the fake matcher that keyword.")
