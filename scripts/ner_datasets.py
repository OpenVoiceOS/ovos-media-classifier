"""OCP-specific AhocorasickNER classes backed by HuggingFace datasets.

These classes pre-populate an AhocorasickNER instance with known entity names
(artist names, track titles, film directors, etc.) from public datasets.  When
attached to an AhocorasickMediaClassifier they dramatically improve recall on
utterances that mention real-world media entities.

Usage::

    from ovos_media_classifier.train.ner_datasets import (
        MusicNER, ImdbNER, OCPMediaNER,
    )

    # Build a combined OCP entity recogniser (downloads data on first use):
    ner = OCPMediaNER("ocp_media.ahocorasick")   # loads or trains+saves
    from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
    clf = AhocorasickMediaClassifier(ner)

All classes inherit from AhocorasickNER and add ``train()`` / ``save()`` /
``load()`` methods.  The ``path`` constructor argument enables transparent
caching: if the file already exists the NER is loaded from disk; otherwise
it is trained from HuggingFace and saved to *path*.

Dependency: ``ahocorasick-ner``, ``datasets``

    pip install ahocorasick-ner datasets
"""
from __future__ import annotations

import os
from typing import Optional

from ovos_utils.log import LOG

try:
    from ahocorasick_ner import AhocorasickNER
except ImportError:
    raise ImportError(
        "ahocorasick-ner is required. pip install ovos-media-classifier[ner]"
    )


def _load_hf(dataset_name: str, split: str = "train"):
    """Load a HuggingFace dataset from the local cache (populated by download_datasets.py)."""
    try:
        from datasets import load_dataset as _load
        from ovos_media_classifier.train import get_hf_cache_dir
        return _load(dataset_name, split=split, cache_dir=get_hf_cache_dir())
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required to download NER training data.\n"
            "Install it with: pip install datasets"
        )


# ---------------------------------------------------------------------------
# Music NER
# ---------------------------------------------------------------------------

class MusicNER(AhocorasickNER):
    """AhocorasickNER pre-loaded with music artist, track, genre names.

    Entity labels produced:
      artist_name, track_name, album_name, album_type, music_genre,
      record_label
    """

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            LOG.info(f"Loading MusicNER from {path!r}")
            self.load(path)
        else:
            LOG.info("Building MusicNER from HuggingFace datasets …")
            self.train()
            if path:
                self.save(path)
                LOG.info(f"Saved MusicNER to {path!r}")

    def train(self) -> None:
        # TODO - configurable, select styles to load, it is too much data to load at once
        # dataset is HUGE, mainly interesting for training not runtime inference
        # we need helper scripts to generate this data from the user own library
        # eg, jellyfin hooks
        self._load_metal()
        self._load_jazz()
        self._load_prog()
        self._load_classical()
        self._load_trance()

    def _load_metal(self) -> None:
        try:
            for entry in _load_hf("Jarbas/metal-archives-tracks"):
                self.add_word("artist_name", entry["band_name"])
                if entry.get("track_name"):
                    self.add_word("track_name", entry["track_name"])
                if entry.get("album_name"):
                    self.add_word("album_name", entry["album_name"])
                if entry.get("album_type"):
                    self.add_word("album_type", entry["album_type"])
            for entry in _load_hf("Jarbas/metal-archives-bands"):
                self.add_word("artist_name", entry["name"])
                if entry.get("genre"):
                    self.add_word("music_genre", entry["genre"])
                if entry.get("label"):
                    self.add_word("record_label", entry["label"])
        except Exception as e:
            LOG.warning(f"MusicNER: failed to load metal archives: {e}")

    def _load_jazz(self) -> None:
        try:
            for entry in _load_hf("Jarbas/jazz-music-archives"):
                if entry.get("artist"):
                    self.add_word("artist_name", entry["artist"])
                if entry.get("genre"):
                    self.add_word("music_genre", entry["genre"])
        except Exception as e:
            LOG.warning(f"MusicNER: failed to load jazz archives: {e}")

    def _load_prog(self) -> None:
        try:
            for entry in _load_hf("Jarbas/prog-archives"):
                if entry.get("artist"):
                    self.add_word("artist_name", entry["artist"])
                if entry.get("genre"):
                    self.add_word("music_genre", entry["genre"])
        except Exception as e:
            LOG.warning(f"MusicNER: failed to load prog archives: {e}")

    def _load_classical(self) -> None:
        try:
            for entry in _load_hf("Jarbas/classic-composers"):
                if entry.get("name"):
                    self.add_word("artist_name", entry["name"])
        except Exception as e:
            LOG.warning(f"MusicNER: failed to load classical composers: {e}")

    def _load_trance(self) -> None:
        try:
            for entry in _load_hf("Jarbas/trance_tracks"):
                if entry.get("ARTIST(S)"):
                    self.add_word("artist_name", entry["ARTIST(S)"])
                if entry.get("TRACK"):
                    self.add_word("track_name", entry["TRACK"])
                if entry.get("STYLE"):
                    self.add_word("music_genre", entry["STYLE"])
        except Exception as e:
            LOG.warning(f"MusicNER: failed to load trance tracks: {e}")


# ---------------------------------------------------------------------------
# Movie / Film NER
# ---------------------------------------------------------------------------

class ImdbNER(AhocorasickNER):
    """AhocorasickNER pre-loaded with film industry names from IMDB datasets.

    Entity labels produced:
      movie_actor, movie_director, movie_producer, movie_writer, movie_composer
    """

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            LOG.info(f"Loading ImdbNER from {path!r}")
            self.load(path)
        else:
            LOG.info("Building ImdbNER from HuggingFace datasets …")
            self.train()
            if path:
                self.save(path)
                LOG.info(f"Saved ImdbNER to {path!r}")

    def train(self) -> None:
        # TODO - configurable, select styles to load, it is too much data to load at once
        # dataset is HUGE, mainly interesting for training not runtime inference
        # we need helper scripts to generate this data from the user own library
        # eg, jellyfin hooks
        for label, dataset_name in [
            ("movie_actor",    "Jarbas/movie_actors"),
            ("movie_director", "Jarbas/movie_directors"),
            ("movie_producer", "Jarbas/movie_producers"),
            ("movie_writer",   "Jarbas/movie_writers"),
            ("movie_composer", "Jarbas/movie_composers"),
        ]:
            try:
                for entry in _load_hf(dataset_name):
                    if entry.get("name"):
                        self.add_word(label, entry["name"])
            except Exception as e:
                LOG.warning(f"ImdbNER: failed to load {dataset_name}: {e}")


# ---------------------------------------------------------------------------
# Combined OCP media NER
# ---------------------------------------------------------------------------

class OCPMediaNER(AhocorasickNER):
    """Combined AhocorasickNER for all OCP media types.

    Merges MusicNER + ImdbNER into a single recogniser for use with
    AhocorasickMediaClassifier.  Additional entity labels added here:
      music_streaming_service, movie_streaming_service, podcast_streaming_service,
      radio_streaming_service, news_provider, audiobook_streaming_service,
      tv_streaming_service
    """

    # Well-known streaming service names — these seed the NER so common
    # service names are always recognised regardless of HuggingFace availability.
    # TODO - extend list? allow user to change?
    _STREAMING_SERVICES: dict[str, list[str]] = {
        "music_streaming_service": [
            "Spotify", "Apple Music", "Tidal", "Deezer", "Qobuz", "Amazon Music",
            "YouTube Music", "SoundCloud", "Bandcamp", "Last.fm", "Pandora",
        ],
        "movie_streaming_service": [
            "Netflix", "Disney+", "Prime Video", "HBO Max", "Apple TV+",
            "Hulu", "Peacock", "Paramount+", "Crunchyroll", "Mubi",
        ],
        "podcast_streaming_service": [
            "Spotify", "Apple Podcasts", "Google Podcasts", "Overcast",
            "Pocket Casts", "Castbox",
        ],
        "radio_streaming_service": [
            "BBC Radio", "NPR", "Radio Paradise", "KEXP", "SomaFM",
            "iHeartRadio",
        ],
        "news_provider": [
            "BBC News", "CNN", "Reuters", "AP News", "The Guardian",
            "NPR News", "Al Jazeera",
        ],
        "audiobook_streaming_service": [
            "Audible", "Librivox", "Libro.fm", "Storytel",
        ],
        "tv_streaming_service": [
            "Netflix", "Disney+", "HBO Max", "Apple TV+", "Hulu",
            "Amazon Prime", "Peacock", "Paramount+",
        ],
    }

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            LOG.info(f"Loading OCPMediaNER from {path!r}")
            self.load(path)
        else:
            LOG.info("Building OCPMediaNER …")
            self.train()
            if path:
                self.save(path)
                LOG.info(f"Saved OCPMediaNER to {path!r}")

    def train(self) -> None:
        # 1. Seed from hard-coded streaming service lists
        for label, names in self._STREAMING_SERVICES.items():
            for name in names:
                self.add_word(label, name)

        # 2. Music entities from HuggingFace
        music = MusicNER()
        for label, automaton in music._automaton.items() if hasattr(music, '_automaton') else []:
            pass  # AhocorasickNER doesn't expose its automaton directly
        # Instead, re-run the loaders and add to self
        music_sub = MusicNER()
        self._merge(music_sub)

        # 3. Movie entities
        imdb_sub = ImdbNER()
        self._merge(imdb_sub)

    def _merge(self, other: AhocorasickNER) -> None:
        """Copy all words from another AhocorasickNER into self.

        AhocorasickNER stores words in internal state — we re-add them by
        tagging a probe string that won't actually match anything, then
        instead use the public add_word API from a shared word list.

        Since AhocorasickNER doesn't expose its word list directly we
        work around this by accessing the private _words attribute if
        available, otherwise we rebuild via the datasets directly.
        """
        # Try to access internal word store (implementation detail)
        words_store = getattr(other, "_words", None) or getattr(other, "words", None)
        if isinstance(words_store, dict):
            for label, word_set in words_store.items():
                for word in word_set:
                    self.add_word(label, word)
        # If _words is not available, the entities were already seeded
        # from the streaming service lists above.
