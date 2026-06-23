#!/usr/bin/env python3
"""05 — Writing your own classifier plugin.

Richer classification strategies (trained ONNX models, NER from media servers,
…) are NOT in this release. They land later as independent, separately-reviewed
plugins discovered through the ``opm.media.classifier`` entry-point group.

A plugin is any subclass of :class:`AbstractMediaClassifier`. The only required
method is ``classify()``; the base class derives ``classify_domain``,
``is_ocp_query``, ``classify_playback_type``, ``classify_structure`` and
``classify_full`` from it. Override ``classify_genres`` if your backend can
surface genre signal (so the content filter can block on it), and override
``classify_domain`` if you have a cheap dedicated domain head.

To ship it as a real, installable plugin, add an entry point to your package's
``pyproject.toml`` (shown in the comment below) and OVOS will load it by name
via ``load_media_classifier({"media_classifier_plugin": "..."})``.

Run:  python examples/05_writing_a_plugin.py
"""
from typing import List, Optional, Tuple

from mediavocab import MediaType
from ovos_media_classifier import AbstractMediaClassifier


class HashtagMediaClassifier(AbstractMediaClassifier):
    """A toy classifier that reads an explicit ``#type`` hashtag in the query.

    Purely illustrative — it shows the contract, not a serious strategy. A real
    plugin would wrap an ONNX model, a padatious container, an Aho-Corasick NER,
    etc. ``config`` is whatever OVOS passes through from ``mycroft.conf``.
    """

    # Plugins are loaded as ``clazz(config=...)``; accept (and ignore) it.
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        # map hashtag -> (MediaType, genre tags)
        self._table = {
            "#music": (MediaType.MUSIC, []),
            "#movie": (MediaType.MOVIE, []),
            "#anime": (MediaType.EPISODIC_SERIES, ["anime"]),
        }

    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        for tag, (mtype, _genres) in self._table.items():
            if tag in query.lower():
                # respect valid_labels gating when the caller passes it
                if valid_labels is None or mtype in valid_labels:
                    return mtype, 0.95
        # Nothing matched -> the canonical "not a media request" answer.
        return MediaType.GENERIC, 0.0

    def classify_genres(self, query: str, lang: str) -> List[str]:
        for tag, (_mtype, genres) in self._table.items():
            if tag in query.lower():
                return list(genres)
        return []


# --- registering it as a real plugin -------------------------------------
# In YOUR package's pyproject.toml, declare the entry point so OVOS can find it:
#
#     [project.entry-points."opm.media.classifier"]
#     my-hashtag-classifier = "my_pkg.module:HashtagMediaClassifier"
#
# Once installed, the factory loads it by name:
#
#     from ovos_media_classifier import load_media_classifier
#     clf = load_media_classifier(
#         {"media_classifier_plugin": "my-hashtag-classifier"}
#     )
#
# (If the named plugin can't be loaded, the factory logs a warning and falls
#  back to the bundled keyword classifier — playback never hard-fails.)


if __name__ == "__main__":
    # We instantiate directly here since the plugin isn't pip-installed in this
    # example; the result is identical to what load_media_classifier() returns.
    clf = HashtagMediaClassifier()

    for utt in ["play #music please", "watch #movie tonight", "#anime marathon",
                "play something"]:
        media_type, conf = clf.classify(utt, "en-us")
        genres = clf.classify_genres(utt, "en-us")
        # base-class derived axes work for free:
        full = clf.classify_full(utt, "en-us").as_dict()
        print(f"{utt:<22} -> {media_type.value:<16} conf={conf} "
              f"genres={genres} playback={full['playback_type']}")
