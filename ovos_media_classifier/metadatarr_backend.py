"""Layer B — **online metadatarr** media classifier (network, last resort).

The keyword backend abstains on bare real titles (the open-vocab gap), and the
Layer A offline gazetteer only covers the popularity head of the metadatarr
catalogues.  For a long-tail real title that neither has a cue for, this backend
asks metadatarr **what the title is** — and that is *not* circular:

* a :class:`MediaProvider` returns a playable **stream** for a title;
* **metadatarr returns METADATA** — given a title it resolves the canonical
  ``medium`` / external IDs.

So routing "what is this?" through metadatarr to *pick which stream-providers to
call* resolves the type from a different system than the one that plays it.

This backend is **opt-in** (the ``[online]`` extra + a config flag, default OFF)
because every classify is a network round-trip.  It is wired as the **last** /
most-expensive layer of the hybrid: consulted only when the cheaper keyword +
offline layers abstain.

Robustness contract
-------------------
``classify`` is wrapped in a wall-clock timeout + ``try/except``.  On timeout,
network failure, an empty resolve, or a low-confidence record it returns
``(GENERIC, 0.0)`` — it **abstains, never raises, never blocks** the pipeline.
metadatarr is **lazy-imported** inside the call so a runtime with the backend
disabled never pulls it in.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from ovos_utils.log import LOG

from ovos_media_classifier.base import AbstractMediaClassifier
from ovos_media_classifier.intents import MediaType, OCPDomain

#: default wall-clock budget for one online resolve (seconds)
DEFAULT_TIMEOUT_S: float = 4.0
#: minimum resolved confidence to trust a route (else abstain)
DEFAULT_MIN_CONFIDENCE: float = 0.5

# mediavocab sentinels that mean "no confident route" (abstain).
_ABSTAIN_MEDIA = {MediaType.GENERIC, MediaType.NOT_MEDIA}


def _run_with_timeout(fn, timeout_s: float):
    """Run *fn()* with a wall-clock *timeout_s*, returning ``None`` on timeout.

    Uses a daemon thread so a hung network call can never block the pipeline:
    the worker is abandoned (it cannot be force-killed in CPython) but the
    caller returns promptly with ``None``.
    """
    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout_s)
        except cf.TimeoutError:
            LOG.debug("metadatarr resolve timed out → abstain")
            return None
        except Exception as e:  # noqa: BLE001 - never raise out of classify
            LOG.debug(f"metadatarr resolve failed → abstain: {e}")
            return None


class MetadatarrMediaClassifier(AbstractMediaClassifier):
    """Online classifier that routes a title via ``metadatarr.resolve``.

    Args:
        timeout_s: per-query wall-clock budget; on expiry the backend abstains.
        min_confidence: minimum resolved match confidence to trust the route.
        max_workers: metadatarr provider fan-out width (passed to ``resolve``).
    """

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S,
                 min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                 max_workers: int = 8) -> None:
        self.timeout_s = float(timeout_s)
        self.min_confidence = float(min_confidence)
        self.max_workers = int(max_workers)
        # cache the resolved record per (query, lang) so the several axis calls
        # the pipeline makes for one utterance share a single network round-trip.
        self._last_key: Optional[Tuple[str, str]] = None
        self._last_result = None

    # ------------------------------------------------------------------
    # metadatarr resolve (lazy import, timeout-guarded, never raises)
    # ------------------------------------------------------------------

    def _resolve(self, query: str, lang: str):
        """Return the metadatarr ``ResolveResult`` (cached), or ``None``.

        Lazy-imports metadatarr so a disabled runtime never loads it; guards the
        network call with a timeout; abstains (``None``) on any failure.
        """
        if not query or not query.strip():
            return None
        key = (query, lang)
        if key == self._last_key:
            return self._last_result

        def _call():
            from metadatarr.resolve import resolve  # lazy: only when enabled
            from mediavocab import Signals
            sig = Signals(title=query,
                          language=(lang.split("-")[0] if lang else None))
            return resolve(sig, max_workers=self.max_workers)

        result = _run_with_timeout(_call, self.timeout_s)
        self._last_key, self._last_result = key, result
        return result

    def _best_confidence(self, result) -> float:
        accepted = getattr(result, "accepted", None) or []
        if not accepted:
            return 0.0
        return max(float(getattr(m, "confidence", 0.0) or 0.0) for m in accepted)

    def _best_accepted_with_medium(self, result):
        """The highest-confidence accepted match whose signals carry a medium.

        ``result.signals`` is ``None`` when the accepted matches conflict on a
        minor field (e.g. one provider omits ``medium`` and trips the consolidator
        to ``signals=None``).  But the strongest accepted match still carries a
        confident medium — that is the route — so we fall back to it rather than
        abstaining on a merge conflict.
        """
        accepted = sorted(
            (getattr(result, "accepted", None) or []),
            key=lambda m: float(getattr(m, "confidence", 0.0) or 0.0),
            reverse=True)
        for m in accepted:
            sig = getattr(m, "signals", None)
            if sig is not None and getattr(sig, "medium", None) is not None:
                return m
        return None

    def _resolved_signals(self, query: str, lang: str):
        """The best confident ``Signals`` of a resolve (with a medium), else ``None``."""
        result = self._resolve(query, lang)
        if result is None:
            return None
        if self._best_confidence(result) < self.min_confidence:
            return None
        # prefer the merged record; fall back to the strongest accepted match
        # that carries a medium when the merge conflicted to signals=None.
        sig = getattr(result, "signals", None)
        if sig is not None and getattr(sig, "medium", None) is not None:
            return sig
        best = self._best_accepted_with_medium(result)
        return getattr(best, "signals", None) if best is not None else None

    # ------------------------------------------------------------------
    # AbstractMediaClassifier
    # ------------------------------------------------------------------

    def classify(self, query: str, lang: str,
                 valid_labels: Optional[List[MediaType]] = None
                 ) -> Tuple[MediaType, float]:
        """Resolve ``media_type`` from metadatarr; GENERIC on any failure/low-conf."""
        result = self._resolve(query, lang)
        if result is None or self._best_confidence(result) < self.min_confidence:
            return MediaType.GENERIC, 0.0
        sig = self._resolved_signals(query, lang)
        if sig is None:
            return MediaType.GENERIC, 0.0
        medium = sig.medium
        try:
            media_type = MediaType(getattr(medium, "value", medium))
        except (ValueError, TypeError):
            return MediaType.GENERIC, 0.0
        if media_type in _ABSTAIN_MEDIA:
            return MediaType.GENERIC, 0.0
        if valid_labels is not None and media_type not in valid_labels:
            return MediaType.GENERIC, 0.0
        best = self._best_accepted_with_medium(result)
        conf = float(getattr(best, "confidence", 0.0) or 0.0) if best else \
            self._best_confidence(result)
        return media_type, conf

    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        """A confident resolved leaf infers OCP_PLAY; else NOT_OCP (abstain).

        This backend is a leaf-router only — in the hybrid the keyword gate is
        authoritative, so this conservative default is never the gate of record.
        """
        media_type, conf = self.classify(query, lang)
        if media_type != MediaType.GENERIC:
            return OCPDomain.OCP_PLAY, conf
        return OCPDomain.NOT_OCP, 0.0

    def classify_playback_type(self, query: str, lang: str):
        """metadatarr's resolved ``playback_type`` when set, else derive from leaf."""
        from mediavocab import infer_playback_type, PlaybackType
        sig = self._resolved_signals(query, lang)
        if sig is not None:
            pb = getattr(sig, "playback_type", None)
            if pb is not None and pb != PlaybackType.UNKNOWN:
                return pb
        return super().classify_playback_type(query, lang)

    def classify_genres(self, query: str, lang: str) -> List[str]:
        """Resolved ``content_genres`` (⊆ KNOWN_GENRES), else none."""
        from mediavocab.taxonomy.genre import KNOWN_GENRES
        sig = self._resolved_signals(query, lang)
        if sig is None:
            return []
        genres = getattr(sig, "content_genres", None) or []
        out: List[str] = []
        for g in genres:
            tag = getattr(g, "value", g)
            if tag in KNOWN_GENRES:
                out.append(tag)
        return out

    def classify_programme_format(self, query: str, lang: str):
        sig = self._resolved_signals(query, lang)
        return getattr(sig, "programme_format", None) if sig is not None else None

    def classify_content_form(self, query: str, lang: str):
        sig = self._resolved_signals(query, lang)
        return getattr(sig, "content_form", None) if sig is not None else None

    def to_signals(self, query: str, lang: str = "en-us"):
        """Provider-ready ``Signals`` enriched with metadatarr's resolved record.

        When metadatarr resolves the title confidently, its merged ``Signals``
        (medium / year / programme_format / content_genres / playback_type) are
        the richest available context for the providers; otherwise we fall back
        to the base behaviour (which abstains to a bare title query).
        """
        sig = self._resolved_signals(query, lang)
        if sig is None:
            return super().to_signals(query, lang)
        # carry the resolved year/medium etc., but keep the user's spoken title.
        try:
            return sig.model_copy(update={"title": query or sig.title})
        except Exception:  # noqa: BLE001 - defensive; never break the pipeline
            return super().to_signals(query, lang)
