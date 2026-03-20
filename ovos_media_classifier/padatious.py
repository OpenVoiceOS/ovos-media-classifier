"""Padatious / padacioso based media classifier.

Uses an IntentContainer trained on samples for each OCPPlayIntent label.
An optional second container handles domain classification
(ocp_play / ocp_control / not_ocp).

This mirrors how the OCP pipeline plugin registers and trains play/control
intents via padatious, but applied to media-type classification:
  - each OCPPlayIntent (music, movie, podcast, …) gets its own intent
  - an optional domain container distinguishes ocp_play / ocp_control / not_ocp

Dependency: ``ovos-padatious`` (fast, recommended) or ``padacioso`` (pure-python fallback)

    pip install ovos-media-classifier[padatious]   # installs ovos-padatious
    # OR
    pip install padacioso                           # pure-python, no extra deps

Usage::

    from ovos_media_classifier.padatious import PadatiousMediaClassifier

    clf = PadatiousMediaClassifier.from_samples(
        play_samples={
            "music":   ["play {query} music", "put on some {genre}",
                        "I want to listen to {artist}"],
            "podcast": ["play the {name} podcast", "subscribe to {show}"],
            "movie":   ["watch {title}", "play the movie {name}"],
        },
        domain_samples={
            "ocp_play":    ["play {query}", "watch {thing}", "put on {media}"],
            "ocp_control": ["pause", "resume", "next track", "stop the music"],
            "not_ocp":     ["set an alarm", "what is the weather", "tell me a joke"],
        },
    )
    media_type, conf = clf.classify("play some jazz", "en-us")
    is_ocp, conf    = clf.is_ocp_query("pause the music", "en-us")

Training samples format
-----------------------
Both ``play_samples`` and ``domain_samples`` accept lists of padatious-style
patterns (curly braces for optional entity slots).

Loading from locale files
-------------------------
Use :meth:`from_locale_dir` to load ``.intent`` files from a directory tree
in the same layout as the OCP pipeline plugin::

    locale/en-us/music.intent
    locale/en-us/movie.intent
    ...
"""
import os
from typing import Dict, List, Optional, Tuple

from ovos_utils.log import LOG

from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.intents import MediaType
from ovos_media_classifier.intents import (
    OCPDomain,
    OCPPlayIntent,
    LABEL_TO_MEDIA_TYPE,
)


def _make_container(cache_dir: Optional[str] = None):
    """Return a padatious or padacioso IntentContainer."""
    try:
        from ovos_padatious import IntentContainer
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            container = IntentContainer(cache_dir)
        else:
            container = IntentContainer()
        return container, True  # (container, is_padatious)
    except ImportError:
        pass
    try:
        from padacioso import IntentContainer
        LOG.warning(
            "ovos-padatious not available, falling back to padacioso. "
            "Intent matching will be significantly slower."
        )
        return IntentContainer(), False
    except ImportError:
        raise ImportError(
            "No padatious implementation found. Install one of:\n"
            "  pip install ovos-padatious   (fast, recommended)\n"
            "  pip install padacioso        (pure-python fallback)"
        )


class PadatiousMediaClassifier(AbstractMediaClassifier):
    """Media classifier backed by padatious / padacioso IntentContainers.

    Args:
        play_container:   IntentContainer trained on OCPPlayIntent samples.
        domain_container: Optional IntentContainer trained on OCPDomain samples.
                          When provided, classify_domain() uses it directly
                          instead of deriving from the play head.
        play_threshold:   Minimum confidence for the play container.
        domain_threshold: Minimum confidence for the domain container.
    """

    def __init__(
        self,
        play_container,
        domain_container=None,
        play_threshold: float = 0.5,
        domain_threshold: float = 0.5,
    ) -> None:
        self._play = play_container
        self._domain = domain_container
        self._play_thresh = play_threshold
        self._domain_thresh = domain_threshold

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_samples(
        cls,
        play_samples: Dict[str, List[str]],
        domain_samples: Optional[Dict[str, List[str]]] = None,
        play_cache_dir: Optional[str] = None,
        domain_cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "PadatiousMediaClassifier":
        """Train from in-memory sample dictionaries.

        Args:
            play_samples:   ``{intent_name: [sample, ...]}`` for media types.
                            Intent names should be OCPPlayIntent values
                            ("music", "movie", …) or any string.
            domain_samples: ``{domain_name: [sample, ...]}`` for domain head.
                            Domain names should be OCPDomain values
                            ("ocp_play", "ocp_control", "not_ocp").
            play_cache_dir:   Cache dir for the play container (padatious only).
            domain_cache_dir: Cache dir for the domain container.
        """
        play_c, is_padatious = _make_container(play_cache_dir)
        for name, samples in play_samples.items():
            play_c.add_intent(name, samples)
        if is_padatious:
            play_c.train()

        domain_c = None
        if domain_samples:
            domain_c, is_padatious_d = _make_container(domain_cache_dir)
            for name, samples in domain_samples.items():
                domain_c.add_intent(name, samples)
            if is_padatious_d:
                domain_c.train()

        return cls(play_c, domain_container=domain_c, **kwargs)

    @classmethod
    def from_locale_dir(
        cls,
        locale_dir: str,
        lang: str,
        domain_dir: Optional[str] = None,
        play_cache_dir: Optional[str] = None,
        domain_cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "PadatiousMediaClassifier":
        """Load from ``.intent`` files in a locale directory.

        Expected structure::

            {locale_dir}/{lang}/music.intent
            {locale_dir}/{lang}/movie.intent
            ...

        Each file is named ``<intent>.intent`` and contains one sample per line.

        Args:
            locale_dir: Root directory containing per-language subdirs.
            lang:       BCP-47 language tag (e.g. "en-us").
            domain_dir: Optional separate locale dir for domain intent files.
        """
        lang_dir = os.path.join(locale_dir, lang.lower())
        if not os.path.isdir(lang_dir):
            raise FileNotFoundError(
                f"No locale directory found for lang {lang!r} at {lang_dir!r}"
            )

        play_samples: Dict[str, List[str]] = {}
        for fname in os.listdir(lang_dir):
            if not fname.endswith(".intent"):
                continue
            intent_name = fname[:-len(".intent")]
            with open(os.path.join(lang_dir, fname), encoding="utf-8") as fh:
                lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
            if lines:
                play_samples[intent_name] = lines

        domain_samples = None
        if domain_dir:
            dom_lang_dir = os.path.join(domain_dir, lang.lower())
            if os.path.isdir(dom_lang_dir):
                domain_samples = {}
                for fname in os.listdir(dom_lang_dir):
                    if not fname.endswith(".intent"):
                        continue
                    domain_name = fname[:-len(".intent")]
                    with open(os.path.join(dom_lang_dir, fname), encoding="utf-8") as fh:
                        lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
                    if lines:
                        domain_samples[domain_name] = lines

        return cls.from_samples(
            play_samples,
            domain_samples=domain_samples,
            play_cache_dir=play_cache_dir,
            domain_cache_dir=domain_cache_dir,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_intent(container, query: str) -> Tuple[Optional[str], float]:
        """Run calc_intent and normalise padatious/padacioso result shapes."""
        try:
            match = container.calc_intent(query)
        except Exception as e:
            LOG.error(f"PadatiousMediaClassifier calc_intent failed: {e}")
            return None, 0.0

        if match is None:
            return None, 0.0

        # padatious returns an IntentMatch object with .name / .conf
        if hasattr(match, "name"):
            return match.name, float(match.conf or 0.0)

        # padacioso may return a dict
        if isinstance(match, dict):
            return match.get("name"), float(match.get("conf", 0.0))

        return None, 0.0

    # ------------------------------------------------------------------
    # AbstractMediaClassifier implementation
    # ------------------------------------------------------------------

    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        if self._domain is None:
            return super().classify_domain(query, lang)

        name, conf = self._calc_intent(self._domain, query)
        if name is None or conf < self._domain_thresh:
            return OCPDomain.NOT_OCP, conf
        try:
            return OCPDomain(name), conf
        except ValueError:
            return OCPDomain.NOT_OCP, conf

    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        name, conf = self._calc_intent(self._play, query)

        if name is None or conf < self._play_thresh:
            return MediaType.GENERIC, 0.0

        media_type = LABEL_TO_MEDIA_TYPE.get(name, MediaType.GENERIC)

        if valid_labels is not None and media_type not in valid_labels:
            return MediaType.GENERIC, 0.0

        return media_type, conf
