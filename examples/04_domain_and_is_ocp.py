#!/usr/bin/env python3
"""04 — Domain routing: is this even a media request?

Before deciding *which* media type, the OCP pipeline needs to know whether the
utterance targets OCP at all. Two methods answer that:

* ``classify_domain(query, lang) -> (OCPDomain, conf)`` — the top-level domain:
  ``OCP_PLAY`` (a playback request), ``OCP_CONTROL`` (a player control like
  pause/next), or ``NOT_OCP`` (unrelated).
* ``is_ocp_query(query, lang) -> (bool, conf)`` — a convenience boolean derived
  from the domain (True for play or control, False for not_ocp).

For the keyword classifier, the default domain head derives from
``classify()``: a non-GENERIC media type implies ``OCP_PLAY``; otherwise
``NOT_OCP``. (A backend with a dedicated domain head — padatious, M2V — would
override ``classify_domain`` and could also report ``OCP_CONTROL``.)

This is exactly the gate the pipeline uses to *not* steal "what time is it" or
"turn on the lights" away from other skills.

Run:  python examples/04_domain_and_is_ocp.py
"""
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier()

# Mix of media requests and non-media utterances.
utterances = [
    "play some music",        # media -> OCP_PLAY
    "play the movie inception",
    "play a podcast",
    "what time is it",        # not media -> NOT_OCP
    "turn on the lights",     # not media -> NOT_OCP
    "what is the weather",    # not media -> NOT_OCP
]

print("utterance".ljust(28), "is_ocp", " domain", "       conf")
print("-" * 60)
for utt in utterances:
    is_ocp, conf = clf.is_ocp_query(utt, "en-us")
    domain, dconf = clf.classify_domain(utt, "en-us")
    print(f"{utt:<28} {str(is_ocp):<6} {domain.value:<12} {conf}")

print()
print("Only the media requests are claimed by OCP; the rest fall through to")
print("the other OVOS pipelines (weather, IoT, datetime, ...).")
