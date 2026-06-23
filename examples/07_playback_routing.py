#!/usr/bin/env python3
"""07 — Playback routing with playback_type + structure.

Once a request is classified, *something* has to actually play it. The two
coarse axes are exactly what you route on:

* **playback_type** (``mediavocab.PlaybackType``): audio / video / paged /
  interactive — which output device / surface to use.
* **structure**: single / episodic / continuous / collection — how to advance
  through the content (one-shot vs next-episode vs a never-ending stream vs a
  playlist queue).

This example wires a tiny illustrative dispatcher that picks a "player" from
those two axes alone, without caring about the fine-grained leaf MediaType.
That is the point of orthogonal axes: a generic player only needs the modality
and the structure.

Run:  python examples/07_playback_routing.py
"""
from mediavocab import PlaybackType
from ovos_media_classifier import load_media_classifier, Structure

clf = load_media_classifier()


def route(playback_type: PlaybackType, structure: Structure) -> str:
    """Decide how to play, from the coarse axes only (illustrative)."""
    # interactive content (games) is its own world
    if playback_type == PlaybackType.INTERACTIVE:
        return "launch game engine (interactive session)"
    # paged content (comics/books) goes to a reader surface
    if playback_type == PlaybackType.PAGED:
        return "open reader UI (paged content)"

    surface = "video player" if playback_type == PlaybackType.VIDEO else "audio player"

    # structure decides how we advance through the content
    if structure == Structure.CONTINUOUS:
        return f"{surface}: tune live stream (no queue, never ends)"
    if structure == Structure.EPISODIC:
        return f"{surface}: resume next episode + build episode queue"
    if structure == Structure.COLLECTION:
        return f"{surface}: enqueue the whole collection / playlist"
    # SINGLE / UNKNOWN
    return f"{surface}: play one item then stop"


utterances = [
    "play some music",            # audio  + single
    "play the movie inception",   # video  + single
    "play the tv show breaking bad",  # video  + episodic
    "watch live tv channel",      # video  + continuous (live)
    "play the radio",             # audio  + continuous
    "play a podcast",             # audio  + episodic
    "start a game",               # interactive
]

for utt in utterances:
    full = clf.classify_full(utt, "en-us")
    decision = route(full.playback_type, full.structure)
    print(
        f"{utt:<32} "
        f"[{full.playback_type.value:<11} | {full.structure.value:<10}] "
        f"-> {decision}"
    )
