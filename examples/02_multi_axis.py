#!/usr/bin/env python3
"""02 — Multi-axis classification with ``classify_full()``.

Media classification is naturally coarse-to-fine. Rather than a brittle hard
hierarchy, ``ovos-media-classifier`` exposes a few **orthogonal axes** and
combines them into one :class:`MediaClassification`:

* **domain**        — is this a media request at all (ocp_play / ocp_control / not_ocp)
* **playback_type** — modality: audio / video / paged / interactive
                      (``mediavocab.PlaybackType``)
* **structure**     — single / episodic / continuous / collection
* **media_type**    — the concrete ``mediavocab.MediaType`` leaf
* **genres**        — orthogonal tags (``adult`` drives content filtering)

For the keyword classifier the coarse axes are *derived* from the predicted
``MediaType`` (cheap and exact). A trained plugin MAY predict each axis with its
own head and soft-gate the leaf instead.

Notice how the axes prune together: a "tv show" and an "anime" both collapse to
the ``episodic_series`` leaf and an ``episodic`` structure, while a live "tv
channel" is ``continuous`` and a "game" is ``interactive``.

Run:  python examples/02_multi_axis.py
"""
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier()

# One representative utterance per broad media family.
utterances = [
    "play some music",
    "play the movie inception",
    "play the tv show breaking bad",
    "play anime",
    "watch live tv channel",
    "play a podcast",
    "play the radio",
    "start a game",
]

for utt in utterances:
    # classify_full() returns a MediaClassification dataclass bundling every
    # axis. .as_dict() flattens the enums to their string .value for printing.
    clf_result = clf.classify_full(utt, "en-us")
    d = clf_result.as_dict()
    print(f"> {utt}")
    print(
        f"    media_type   = {d['media_type']}\n"
        f"    playback_type= {d['playback_type']}   "
        f"(audio/video/paged/interactive)\n"
        f"    structure    = {d['structure']}   "
        f"(single/episodic/continuous/collection)\n"
        f"    domain       = {d['domain']}\n"
        f"    genres       = {d['genres']}\n"
        f"    confidence   = {d['confidence']}"
    )
    print()
