from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

from ovos_media_classifier.intents import MediaType, OCPDomain, OCPControlIntent


class AbstractMediaClassifier(ABC):
    """Classifies a natural-language query into a ``mediavocab.MediaType`` + domain.

    This is also the **plugin contract** for external classifiers discovered via
    the ``opm.media.classifier`` entry-point group: a 3rd-party package ships a
    subclass and the factory loads it by name.

    Implementors must provide:
      classify()        — (mediavocab.MediaType, confidence) for the ocp_play domain
      classify_domain() — (OCPDomain, confidence); default derives from classify()
      is_ocp_query()    — (bool, confidence); default derives from classify_domain()
      classify_genres() — optional list[str] of mediavocab genre tags (default [])

    Subclasses that have a cheap domain head (e.g. M2V, padatious with a
    separate domain container) should override classify_domain() directly.
    Subclasses that also handle ocp_control (padatious) may override
    is_ocp_query() to return True for control intents too.  Subclasses that can
    surface genre signal (keyword, guided, m2v) should override
    classify_genres() so the content filter can block on it.
    """

    @abstractmethod
    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        """Return the most likely MediaType and a confidence in [0, 1].

        Args:
            query: User utterance.
            lang:  BCP-47 language tag (should already be standardised).
            valid_labels: When provided, only return one of these types.
                          Return (GENERIC, 0.0) when nothing matches.
        """

    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        """Classify the top-level OCP domain: ocp_play / ocp_control / not_ocp.

        The default implementation first consults ``classify_control`` — a
        transport-control intent (pause / stop / next / …) routes to
        ``OCP_CONTROL``.  Otherwise it delegates to ``classify``: a non-GENERIC
        MediaType infers ``OCP_PLAY``; else ``NOT_OCP``.

        Subclasses with a dedicated domain head (M2V, padatious domain
        container, sklearn domain model) should override this for better
        accuracy.
        """
        media_type, conf = self.classify(query, lang)
        if self.classify_control(query, lang) is not None:
            return OCPDomain.OCP_CONTROL, conf
        if media_type != MediaType.GENERIC:
            return OCPDomain.OCP_PLAY, conf
        return OCPDomain.NOT_OCP, 0.0

    def classify_control(
        self, query: str, lang: str
    ) -> Optional[OCPControlIntent]:
        """Classify a player **control** action (the ocp_control domain).

        Returns the :class:`~ovos_media_classifier.intents.OCPControlIntent` the
        utterance targets (pause / stop / next / shuffle / seek / …), or ``None``
        when the query is not a transport-control request.

        The default returns ``None`` — only backends that model control intents
        (the keyword backend via its ``Ctrl*.voc`` files, padatious, trained
        heads) override this.  ``classify_domain`` consults it to decide between
        ``OCP_PLAY`` and ``OCP_CONTROL``.
        """
        return None

    def is_ocp_query(self, query: str, lang: str) -> Tuple[bool, float]:
        """Return (True, conf) if the query targets OCP (play or control).

        The default implementation delegates to classify_domain().
        """
        domain, conf = self.classify_domain(query, lang)
        return domain != OCPDomain.NOT_OCP, conf

    def classify_genres(self, query: str, lang: str) -> List[str]:
        """Return mediavocab genre tags implied by the query (default: none).

        Genres are orthogonal to ``MediaType`` and are what the content filter
        blocks on (e.g. ``adult``).  Backends that can cheaply surface genre
        (keyword, guided, m2v) should override this.
        """
        return []

    # ------------------------------------------------------------------
    # Multi-axis output (OVOS-MEDIA-CLASSIFY) — orthogonal coarse axes.
    # Defaults derive the coarse axes from the predicted MediaType; a trained
    # backend MAY override to predict each axis with its own head.
    # ------------------------------------------------------------------

    def classify_playback_type(self, query: str, lang: str):
        """Return the ``mediavocab.PlaybackType`` (audio/video/paged/interactive).

        Default: derived from the classified ``MediaType``.
        """
        from mediavocab import infer_playback_type
        media_type, _ = self.classify(query, lang)
        return infer_playback_type(media_type)

    def classify_structure(self, query: str, lang: str):
        """Return the :class:`~ovos_media_classifier.axes.Structure`
        (single/episodic/continuous/collection).  Default: derived from MediaType."""
        from ovos_media_classifier.axes import infer_structure
        media_type, _ = self.classify(query, lang)
        return infer_structure(media_type)

    # ------------------------------------------------------------------
    # Extended multi-task axes (OVOS-MEDIA-CLASSIFY).
    #
    # Each is a sub-task a trained backend MAY predict with its own head; the
    # defaults here keep untrained backends (the keyword default) working by
    # deriving from the cheaper axes or returning empty.  See docs/model.md.
    # ------------------------------------------------------------------

    def classify_content_form_genres(self, query: str, lang: str) -> List[str]:
        """Sensitive / content-form genre tags (``adult`` / ``anime`` /
        ``animation`` / ``asmr``) — **this is what the content filter reads**.

        Default: delegates to :meth:`classify_genres` (the keyword backend
        surfaces exactly these form tags).  A trained backend SHOULD override
        with a dedicated multi-label head so it can flag ``adult`` even when it
        is unsure of the exact ``MediaType`` (more robust blocking).
        """
        return self.classify_genres(query, lang)

    def classify_content_genres(self, query: str, lang: str) -> List[str]:
        """The **real** genre(s) — rock/jazz/action/comedy/horror/… (multi-label).

        Orthogonal to the content-form tags.  Default: ``[]`` (the keyword
        backend does not model open-vocabulary genre); a trained backend
        overrides with its ``content_genre`` head.
        """
        return []

    def classify_mood(self, query: str, lang: str) -> Optional[str]:
        """The mood / activity (chill/workout/study/party/sleep/…) or ``None``.

        Default: ``None`` — a trained backend overrides with its ``mood`` head.
        """
        return None

    def classify_era(self, query: str, lang: str) -> Optional[str]:
        """The release era / decade (e.g. ``"1980s"``) or ``None``.

        Default: ``None`` — a trained backend overrides with its ``era`` head.
        """
        return None

    def classify_explicitness(self, query: str, lang: str) -> str:
        """``"adult"`` when an adult content-form tag is present, else ``"clean"``.

        Default: derived from :meth:`classify_content_form_genres`.
        """
        return "adult" if "adult" in self.classify_content_form_genres(query, lang) \
            else "clean"

    def classify_qualifiers(self, query: str, lang: str) -> List[str]:
        """Result-narrowing qualifiers (multi-label): ``black_and_white`` /
        ``silent`` / ``live`` / ``subtitled`` / ``dubbed`` / ``audio_described`` /
        ``trailer`` / … — strong filter signals that are *not* media types.

        Default: ``[]`` — backends that surface them (keyword via ``BWKeyword`` /
        ``SilentKeyword`` / …, trained ``qualifiers`` head) override this.
        """
        return []

    def classify_control_intent(self, query: str, lang: str):
        """The :class:`~ovos_media_classifier.intents.OCPControlIntent`, or ``None``.

        Default: delegates to :meth:`classify_control`.
        """
        return self.classify_control(query, lang)

    def classify_full(self, query: str, lang: str):
        """Return the full multi-axis :class:`~ovos_media_classifier.axes.MediaClassification`.

        Combines the leaf ``MediaType``, the derived coarse axes
        (``playback_type`` + ``structure``), the ``domain`` and ``genres`` into
        one result.  Backends with dedicated heads SHOULD override to predict the
        axes directly (and soft-gate the leaf) rather than deriving them.
        """
        from ovos_media_classifier.axes import classification_from_media_type
        media_type, conf = self.classify(query, lang)
        domain, _ = self.classify_domain(query, lang)
        genres = self.classify_genres(query, lang)
        return classification_from_media_type(media_type, domain, genres, conf)

    def to_signals(self, query: str, lang: str = "en-us"):
        """Build a provider-ready :class:`mediavocab.Signals` from the query.

        This is the classifier's primary output for the OCP pipeline: *all* the
        NLP (classification + the coarse axes) lives here, and the pipeline
        forwards the returned ``Signals`` straight to ``MediaProvider.search``
        without doing any parsing of its own.

        The base populates the classification axes (``medium`` /
        ``playback_type`` / ``content_genres``) plus the raw ``title``; backends
        that extract entities (artist / year / season / episode) should override
        to enrich the ``Signals`` further.

        ``content_genres`` carries both the **content-form** tags (``adult`` /
        ``anime`` / …) and the **real** genres (rock / action / …) so a provider
        can filter on either.  The result-narrowing **qualifiers**
        (``black_and_white`` / ``silent`` / ``subtitled`` / …) are joined into
        the ``edition`` filter field — a strong cut on the candidate set.
        """
        from mediavocab import Signals, MediaType
        full = self.classify_full(query, lang)
        sentinels = {MediaType.GENERIC, MediaType.NOT_MEDIA, MediaType.CONTROL}

        # form genres (the content-filter axis) + the real genre(s)
        genres = list(dict.fromkeys(
            list(full.genres) + list(self.classify_content_genres(query, lang))))
        qualifiers = self.classify_qualifiers(query, lang)

        return Signals.as_query(
            title=query or None,
            medium=full.media_type if full.media_type not in sentinels else None,
            playback_type=full.playback_type,
            content_genres=genres,
            edition=", ".join(qualifiers) or None,
            language=(lang.split("-")[0] if lang else None),
        )
