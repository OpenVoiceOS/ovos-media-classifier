from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

from mediavocab.taxonomy import (
    ContentForm,
    ProgrammeFormat,
    AccessibilityKind,
    VariantKind,
    PictureFormat,
    KNOWN_GENRES,
)

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

    mediavocab axes
    ---------------
    The descriptive output is expressed in **mediavocab's own taxonomy** so the
    classifier emits the same vocabulary the resolver / providers consume:

      * :meth:`classify_content_form`   → ``mediavocab.ContentForm | None``
        (primary / trailer / teaser / behind_scenes / excerpt / supplement / …)
      * :meth:`classify_programme_format` → ``mediavocab.ProgrammeFormat | None``
        (documentary / news / concert / stand_up / talk_show / sports / …)
      * :meth:`classify_accessibility`  → ``list[mediavocab.AccessibilityKind]``
        (subtitles / audio_description / sign_language / …)
      * :meth:`classify_variant`        → ``mediavocab.VariantKind | None``
        (directors / extended / remastered / colorized / fanedit / …)
      * :meth:`classify_genres`         → ``list[str] ⊆ mediavocab.KNOWN_GENRES``
        (the content filter reads ``adult`` / ``anime`` / … from here)

      * :meth:`classify_picture_format` → ``list[mediavocab.PictureFormat]``
        (black_and_white / silent / 3d / … — technical presentation attributes)
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
        """Return ``mediavocab.KNOWN_GENRES`` tags implied by the query (default: none).

        Genres are orthogonal to ``MediaType`` and are what the content filter
        blocks on (e.g. ``adult``).  Output is constrained to
        ``mediavocab.KNOWN_GENRES`` so the classifier never emits a tag the
        taxonomy does not recognise.  Backends that can cheaply surface genre
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
        """Return the :class:`mediavocab.Structure`
        (single/episodic/continuous/collection).  Default: derived from MediaType."""
        from mediavocab import infer_structure
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

    def classify_content_form(self, query: str, lang: str) -> Optional[ContentForm]:
        """The :class:`mediavocab.ContentForm` (trailer / teaser / behind_scenes
        / excerpt / supplement / …) or ``None`` for a primary work.

        Default: ``None`` — backends that surface supplementary-content cues
        (keyword via ``TrailerKeyword`` / ``BloopersKeyword`` / …, trained
        ``content_form`` head) override this.  The parent ``MediaType`` stays the
        feature (MOVIE / EPISODIC); the experiential kind rides this axis.
        """
        return None

    def classify_programme_format(
        self, query: str, lang: str
    ) -> Optional[ProgrammeFormat]:
        """The :class:`mediavocab.ProgrammeFormat` (documentary / news / concert
        / stand_up / talk_show / sports / …) or ``None``.

        Structural programme format, orthogonal to genre and media type.
        Default: ``None`` — backends that surface format cues override this.
        """
        return None

    def classify_accessibility(
        self, query: str, lang: str
    ) -> List[AccessibilityKind]:
        """The requested :class:`mediavocab.AccessibilityKind` assets
        (subtitles / audio_description / sign_language / …), multi-label.

        Default: ``[]`` — backends that surface accessibility cues override this.
        """
        return []

    def classify_variant(self, query: str, lang: str) -> Optional[VariantKind]:
        """The :class:`mediavocab.VariantKind` (directors / extended / remastered
        / colorized / fanedit / …) or ``None`` for the canonical cut.

        Default: ``None`` — backends that surface variant cues override this.
        """
        return None

    def classify_picture_format(self, query: str, lang: str) -> List[PictureFormat]:
        """The :class:`mediavocab.PictureFormat` presentation attributes
        (``black_and_white`` / ``silent`` / ``3d`` / …), multi-label.

        A technical Release attribute (T6); routing-family (A6).  Default:
        ``[]`` — backends that surface presentation cues override this.
        """
        return []

    def classify_explicitness(self, query: str, lang: str) -> str:
        """``"adult"`` when an adult content-form tag is present, else ``"clean"``.

        Default: derived from :meth:`classify_content_form_genres`.
        """
        return "adult" if "adult" in self.classify_content_form_genres(query, lang) \
            else "clean"

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

        Every descriptive axis is emitted in **mediavocab's own vocabulary** so
        the signals are lossless and directly comparable:

          * ``medium``           ← the leaf ``MediaType``
          * ``playback_type``    ← the modality axis
          * ``content_genres``   ← ``classify_genres`` (⊆ ``KNOWN_GENRES``)
          * ``content_form``     ← ``classify_content_form`` (trailer / supplement / …)
          * ``programme_format`` ← ``classify_programme_format`` (documentary / news / …)
          * ``variant_kind``     ← ``classify_variant`` (directors / remastered / …)
          * ``accessibility``    ← ``classify_accessibility`` (subtitles / dubbed / …)
          * ``picture_format``   ← ``classify_picture_format`` (black_and_white / silent / 3d)

        Every descriptive axis the classifier predicts now has a
        :class:`mediavocab.Signals` field, so ``to_signals`` is lossless.
        ``picture_format`` is single-valued on ``Signals``; the first predicted
        format wins when the backend surfaces several.

        Backends that extract entities (artist / year / season / episode) should
        override to enrich the ``Signals`` further.
        """
        from mediavocab import Signals, MediaType
        full = self.classify_full(query, lang)
        sentinels = {MediaType.GENERIC, MediaType.NOT_MEDIA, MediaType.CONTROL}

        # genres constrained to KNOWN_GENRES (the content-filter axis)
        genres = [g for g in dict.fromkeys(full.genres) if g in KNOWN_GENRES]

        picture_formats = self.classify_picture_format(query, lang)

        return Signals.as_query(
            title=query or None,
            medium=full.media_type if full.media_type not in sentinels else None,
            playback_type=full.playback_type,
            content_genres=genres,
            content_form=self.classify_content_form(query, lang),
            programme_format=self.classify_programme_format(query, lang),
            variant_kind=self.classify_variant(query, lang),
            accessibility=self.classify_accessibility(query, lang),
            picture_format=picture_formats[0] if picture_formats else None,
            language=(lang.split("-")[0] if lang else None),
        )
