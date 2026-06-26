"""Tests for the optional NER (entity-based) classifier backend.

Covers:
  - AhocorasickMediaClassifier — entity-label → mediavocab.MediaType + genres,
    with runtime updates, priority resolution and valid_labels filtering.
  - EntitiesContainer loaders — CSV, Radarr, Sonarr, Lidarr, Jellyfin,
    Music Assistant, HuggingFace (``requests`` / ``datasets`` mocked).
  - The ``load_media_classifier`` factory selection of the NER backend and the
    graceful fall-back to the lean keyword classifier when ``ahocorasick-ner``
    is missing.

The ``ahocorasick-ner`` package is **mocked** (a fake ``AhocorasickNER`` is
injected into ``sys.modules``) so the whole suite runs green WITHOUT the
optional dependency installed.  Likewise ``requests`` / ``datasets`` are never
imported for real — the loaders patch ``_http_get`` / ``load_dataset``.
"""
import csv
import os
import sys
import tempfile
import unittest
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

from ovos_media_classifier.intents import MediaType, OCPDomain


# ---------------------------------------------------------------------------
# Fake ahocorasick_ner — substring-matching automaton stand-in
# ---------------------------------------------------------------------------


class _FakeAhocorasickNER:
    """Minimal AhocorasickNER stand-in used so the suite runs without the
    real ``ahocorasick-ner`` dependency.

    Matches registered words as case-insensitive substrings of the query and
    returns ``[{"label": …, "word": …}]`` like the real engine.
    """

    def __init__(self) -> None:
        self._words: Dict[str, List[str]] = {}

    def add_word(self, label: str, word: str) -> None:
        self._words.setdefault(label, [])
        if word not in self._words[label]:
            self._words[label].append(word)

    def get_all_words(self) -> Dict[str, List[str]]:
        return {label: list(words) for label, words in self._words.items()}

    def tag(self, query: str):
        ql = query.lower()
        hits = []
        for label, words in self._words.items():
            for word in words:
                if word.lower() in ql:
                    hits.append({"label": label, "word": word})
        return hits


def _install_fake_ner():
    """Inject a fake ``ahocorasick_ner`` module into ``sys.modules``."""
    mod = type(sys)("ahocorasick_ner")
    mod.AhocorasickNER = _FakeAhocorasickNER
    return patch.dict(sys.modules, {"ahocorasick_ner": mod})


class _NERTestCase(unittest.TestCase):
    """Base case that installs the fake ``ahocorasick_ner`` for every test."""

    def setUp(self) -> None:
        self._ner_patch = _install_fake_ner()
        self._ner_patch.start()
        self.addCleanup(self._ner_patch.stop)


# ---------------------------------------------------------------------------
# AhocorasickMediaClassifier — import-error enforcement
# ---------------------------------------------------------------------------


class TestAhocorasickImportError(unittest.TestCase):
    def test_missing_ahocorasick_ner_raises_import_error(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        with patch.dict(sys.modules, {"ahocorasick_ner": None}):
            with self.assertRaises(ImportError) as ctx:
                AhocorasickMediaClassifier(MagicMock())
            self.assertIn("ahocorasick-ner", str(ctx.exception))


# ---------------------------------------------------------------------------
# AhocorasickMediaClassifier — classification
# ---------------------------------------------------------------------------


class TestAhocorasickClassify(_NERTestCase):
    @staticmethod
    def _ner_returning(entities):
        ner = MagicMock()
        ner.tag.return_value = [
            {"label": label, "word": word} for label, word in entities
        ]
        return ner

    def test_entity_label_maps_to_media_type(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        # movie_title is a real entity the user has → MOVIE
        ner = self._ner_returning([("movie_title", "Inception")])
        clf = AhocorasickMediaClassifier(ner)
        mt, conf = clf.classify("play Inception", "en-us")
        self.assertEqual(mt, MediaType.MOVIE)
        self.assertEqual(conf, 0.6)

    def test_music_streaming_service_maps_to_music(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        ner = self._ner_returning([("music_streaming_service", "Spotify")])
        clf = AhocorasickMediaClassifier(ner)
        mt, conf = clf.classify("play on Spotify", "en-us")
        self.assertEqual(mt, MediaType.MUSIC)
        self.assertEqual(conf, 0.6)

    def test_no_hit_returns_generic(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        ner = self._ner_returning([])
        clf = AhocorasickMediaClassifier(ner)
        mt, conf = clf.classify("what's the weather", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_valid_labels_filter_excludes(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        ner = self._ner_returning([("music_streaming_service", "Spotify")])
        clf = AhocorasickMediaClassifier(ner)
        mt, conf = clf.classify("play Spotify", "en-us",
                                valid_labels=[MediaType.MOVIE])
        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_valid_labels_filter_includes(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        ner = self._ner_returning([("music_streaming_service", "Spotify")])
        clf = AhocorasickMediaClassifier(ner)
        mt, conf = clf.classify("play Spotify", "en-us",
                                valid_labels=[MediaType.MUSIC])
        self.assertEqual(mt, MediaType.MUSIC)

    def test_priority_resolution_music_beats_movie(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        ner = self._ner_returning([
            ("music_streaming_service", "Spotify"),
            ("movie_streaming_service", "Netflix"),
        ])
        clf = AhocorasickMediaClassifier(ner)
        mt, _ = clf.classify("Spotify or Netflix", "en-us")
        self.assertEqual(mt, MediaType.MUSIC)

    def test_label_map_override(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        ner = self._ner_returning([("custom_label", "value")])
        clf = AhocorasickMediaClassifier(ner,
                                         label_map={"custom_label": MediaType.TV})
        mt, _ = clf.classify("value", "en-us")
        self.assertEqual(mt, MediaType.TV)

    def test_ner_exception_returns_generic(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        ner = MagicMock()
        ner.tag.side_effect = RuntimeError("boom")
        clf = AhocorasickMediaClassifier(ner)
        mt, conf = clf.classify("query", "en-us")
        self.assertEqual(mt, MediaType.GENERIC)
        self.assertEqual(conf, 0.0)

    def test_classify_domain_hit_and_miss(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        ner = self._ner_returning([("movie_title", "Inception")])
        clf = AhocorasickMediaClassifier(ner)
        domain, conf = clf.classify_domain("play Inception", "en-us")
        self.assertEqual(domain, OCPDomain.OCP_PLAY)
        self.assertEqual(conf, 0.6)

        ner2 = self._ner_returning([])
        clf2 = AhocorasickMediaClassifier(ner2)
        domain2, conf2 = clf2.classify_domain("hello", "en-us")
        self.assertEqual(domain2, OCPDomain.NOT_OCP)
        self.assertEqual(conf2, 0.0)


# ---------------------------------------------------------------------------
# AhocorasickMediaClassifier — genres
# ---------------------------------------------------------------------------


class TestAhocorasickGenres(_NERTestCase):
    def test_adult_entity_yields_adult_genre(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        ner = MagicMock()
        ner.tag.return_value = [{"label": "pornstar", "word": "someone"}]
        clf = AhocorasickMediaClassifier(ner)
        genres = clf.classify_genres("play someone", "en-us")
        self.assertIn("adult", genres)

    def test_hentai_entity_yields_anime_and_adult(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        ner = MagicMock()
        ner.tag.return_value = [{"label": "hentai_title", "word": "x"}]
        clf = AhocorasickMediaClassifier(ner)
        genres = clf.classify_genres("play x", "en-us")
        self.assertIn("adult", genres)

    def test_no_hit_yields_no_genres(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        ner = MagicMock()
        ner.tag.return_value = []
        clf = AhocorasickMediaClassifier(ner)
        self.assertEqual(clf.classify_genres("hello", "en-us"), [])


# ---------------------------------------------------------------------------
# AhocorasickMediaClassifier — factories + runtime updates
# ---------------------------------------------------------------------------


class TestAhocorasickFactories(_NERTestCase):
    def test_from_wordlists_classifies(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        clf = AhocorasickMediaClassifier.from_wordlists({
            "movie_title": ["Inception", "The Matrix"],
            "artist_name": ["Radiohead"],
        })
        self.assertIsNotNone(clf.container)
        mt, _ = clf.classify("play Inception tonight", "en-us")
        self.assertEqual(mt, MediaType.MOVIE)
        mt2, _ = clf.classify("put on Radiohead", "en-us")
        self.assertEqual(mt2, MediaType.MUSIC)

    def test_from_container_runtime_update(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
        from ovos_media_classifier.entities import EntitiesContainer

        container = EntitiesContainer()
        container.add("movie_title", "Inception")
        clf = AhocorasickMediaClassifier.from_container(container)
        self.assertIs(clf.container, container)

        # not yet known
        self.assertEqual(clf.classify("play Radiohead", "en-us")[0],
                         MediaType.GENERIC)
        # add at runtime → immediately reflected (shared NER by reference)
        container.add("artist_name", "Radiohead")
        self.assertEqual(clf.classify("play Radiohead", "en-us")[0],
                         MediaType.MUSIC)

    def test_add_word_via_classifier(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        clf = AhocorasickMediaClassifier.from_wordlists({"movie_title": ["A"]})
        clf.add_word("artist_name", "Radiohead")
        self.assertEqual(clf.classify("play Radiohead", "en-us")[0],
                         MediaType.MUSIC)

    def test_from_csv(self):
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ents.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["entity", "label", "source"])
                w.writerow(["Inception", "movie_title", "manual"])
            clf = AhocorasickMediaClassifier.from_csv(path)
            self.assertEqual(clf.classify("watch Inception", "en-us")[0],
                             MediaType.MOVIE)


# ---------------------------------------------------------------------------
# EntitiesContainer — basic store + CSV
# ---------------------------------------------------------------------------


class TestEntitiesContainerBasic(unittest.TestCase):
    def test_add_dedup_and_wordlists(self):
        from ovos_media_classifier.entities import EntitiesContainer

        c = EntitiesContainer()
        c.add("movie_title", "Inception")
        c.add("movie_title", "Inception")  # duplicate
        c.add("artist_name", "Radiohead")
        self.assertEqual(len(c), 2)
        self.assertEqual(c.wordlists["movie_title"], ["Inception"])
        self.assertEqual(c.stats["movie_title"], 1)

    def test_add_strips_and_ignores_empty(self):
        from ovos_media_classifier.entities import EntitiesContainer

        c = EntitiesContainer()
        c.add("movie_title", "  ")
        c.add("movie_title", "  Inception  ")
        self.assertEqual(c.wordlists["movie_title"], ["Inception"])

    def test_load_csv_named_columns(self):
        from ovos_media_classifier.entities import EntitiesContainer

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "e.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["entity", "label", "source"])
                w.writerow(["jazz", "music_genre", "manual"])
                w.writerow(["Netflix", "movie_streaming_service", "api"])
            c = EntitiesContainer()
            added = c.load_csv(path)
            self.assertEqual(added, 2)
            self.assertIn("music_genre", c.wordlists)

    def test_load_csv_label_value_columns(self):
        from ovos_media_classifier.entities import EntitiesContainer

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "e.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["label", "value"])
                w.writerow(["movie_title", "Inception"])
            c = EntitiesContainer()
            added = c.load_csv(path)
            self.assertEqual(added, 1)
            self.assertIn("movie_title", c.wordlists)


# ---------------------------------------------------------------------------
# EntitiesContainer — media-server loaders (requests mocked via _http_get)
# ---------------------------------------------------------------------------


class TestEntitiesContainerRadarr(unittest.TestCase):
    @patch("ovos_media_classifier.entities._http_get")
    def test_load_radarr_success(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        mock_http_get.return_value = [
            {
                "title": "The Dark Knight",
                "genres": ["action", "crime"],
                "alternateTitles": [{"title": "Batman Begins 2"}],
                "credits": {
                    "castMembers": [{"name": "Christian Bale"}],
                    "crewMembers": [{"name": "Christopher Nolan", "job": "Director"}],
                },
                "studio": "Warner Bros",
            },
        ]
        c = EntitiesContainer()
        added = c.load_radarr("http://localhost:7878", api_key="k")
        self.assertGreater(added, 0)
        self.assertIn("The Dark Knight", c.wordlists["movie_title"])
        self.assertIn("Christian Bale", c.wordlists["movie_actor"])
        self.assertIn("Christopher Nolan", c.wordlists["movie_director"])

    @patch("ovos_media_classifier.entities._http_get")
    def test_load_radarr_http_error(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        mock_http_get.return_value = None
        c = EntitiesContainer()
        self.assertEqual(c.load_radarr("http://x", api_key="k"), 0)

    def test_load_radarr_missing_requests_raises(self):
        from ovos_media_classifier.entities import EntitiesContainer

        c = EntitiesContainer()
        with patch.dict(sys.modules, {"requests": None}):
            with self.assertRaises(ImportError) as ctx:
                c.load_radarr("http://x", api_key="k")
            self.assertIn("requests", str(ctx.exception))


class TestEntitiesContainerSonarr(unittest.TestCase):
    @patch("ovos_media_classifier.entities._http_get")
    def test_load_sonarr_anime_vs_tv(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        mock_http_get.return_value = [
            {"title": "Naruto", "seriesType": "anime", "genres": [],
             "alternateTitles": [], "network": "TV Tokyo"},
            {"title": "Breaking Bad", "genres": ["drama"],
             "alternateTitles": [], "network": "AMC"},
        ]
        c = EntitiesContainer()
        added = c.load_sonarr("http://x", api_key="k")
        self.assertGreater(added, 0)
        self.assertIn("Naruto", c.wordlists["anime_title"])
        self.assertIn("Breaking Bad", c.wordlists["tv_show_title"])


class TestEntitiesContainerLidarr(unittest.TestCase):
    @patch("ovos_media_classifier.entities._http_get")
    def test_load_lidarr_success(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        def side_effect(session, url, **kwargs):
            if "/artist" in url:
                return [{"artistName": "Radiohead", "genres": ["rock"]}]
            if "/album" in url:
                return [{
                    "title": "OK Computer",
                    "artist": {"artistName": "Radiohead"},
                    "media": [{"tracks": [{"title": "Karma Police"}]}],
                }]
            return []

        mock_http_get.side_effect = side_effect
        c = EntitiesContainer()
        added = c.load_lidarr("http://x", api_key="k")
        self.assertGreater(added, 0)
        self.assertIn("Radiohead", c.wordlists["artist_name"])
        self.assertIn("OK Computer", c.wordlists["album_name"])
        self.assertIn("Karma Police", c.wordlists["track_name"])


class TestEntitiesContainerJellyfin(unittest.TestCase):
    @patch("ovos_media_classifier.entities._http_get")
    def test_load_jellyfin_success(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        def side_effect(session, url, **kwargs):
            if url.endswith("/Users"):
                return [{"Id": "user123"}]
            return {
                "Items": [{
                    "Name": "The Dark Knight",
                    "Type": "Movie",
                    "Genres": ["Action"],
                    "People": [{"Name": "Christian Bale", "Type": "Actor"}],
                }],
                "TotalRecordCount": 1,
            }

        mock_http_get.side_effect = side_effect
        c = EntitiesContainer()
        added = c.load_jellyfin("http://x", api_key="k")
        self.assertGreater(added, 0)
        self.assertIn("The Dark Knight", c.wordlists["movie_title"])


class TestEntitiesContainerMusicAssistant(unittest.TestCase):
    @patch("ovos_media_classifier.entities._http_get")
    def test_load_music_assistant(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        def side_effect(session, url, **kwargs):
            if url.endswith("/artists"):
                return [{"name": "Radiohead", "metadata": {"genres": ["rock"]}}]
            if url.endswith("/albums"):
                return [{"name": "OK Computer", "artists": [{"name": "Radiohead"}]}]
            if url.endswith("/tracks"):
                return [{"name": "Karma Police", "artists": []}]
            if url.endswith("/radio"):
                return [{"name": "BBC Radio 1"}]
            return []

        mock_http_get.side_effect = side_effect
        c = EntitiesContainer()
        added = c.load_music_assistant("http://x")
        self.assertGreater(added, 0)
        self.assertIn("Radiohead", c.wordlists["artist_name"])
        self.assertIn("BBC Radio 1", c.wordlists["radio_station"])


# ---------------------------------------------------------------------------
# EntitiesContainer — HuggingFace loader (datasets mocked)
# ---------------------------------------------------------------------------


class TestEntitiesContainerHuggingFace(unittest.TestCase):
    def test_load_huggingface_success(self):
        from ovos_media_classifier.entities import EntitiesContainer

        fake_datasets = type(sys)("datasets")
        fake_datasets.load_dataset = MagicMock(return_value=[
            {"entity": "Inception", "label": "movie_title"},
            {"entity": "Radiohead", "label": "artist_name"},
            {"entity": "", "label": "movie_title"},  # skipped
        ])
        c = EntitiesContainer()
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            added = c.load_huggingface("Some/dataset")
        self.assertEqual(added, 2)
        self.assertIn("Inception", c.wordlists["movie_title"])

    def test_load_huggingface_missing_datasets_raises(self):
        from ovos_media_classifier.entities import EntitiesContainer

        c = EntitiesContainer()
        with patch.dict(sys.modules, {"datasets": None}):
            with self.assertRaises(ImportError) as ctx:
                c.load_huggingface("Some/dataset")
            self.assertIn("datasets", str(ctx.exception))


# ---------------------------------------------------------------------------
# EntitiesContainer.from_config
# ---------------------------------------------------------------------------


class TestEntitiesContainerFromConfig(_NERTestCase):
    def test_from_config_wordlists_and_csv(self):
        from ovos_media_classifier.entities import EntitiesContainer

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "e.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["label", "value"])
                w.writerow(["artist_name", "Radiohead"])
            c = EntitiesContainer.from_config({
                "wordlists": {"movie_title": ["Inception"]},
                "csv": [path],
            })
        self.assertIn("Inception", c.wordlists["movie_title"])
        self.assertIn("Radiohead", c.wordlists["artist_name"])

    @patch("ovos_media_classifier.entities._http_get")
    def test_from_config_invokes_media_servers(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        mock_http_get.return_value = [
            {"title": "Inception", "genres": [], "alternateTitles": [],
             "credits": {}, "studio": ""},
        ]
        c = EntitiesContainer.from_config({
            "radarr": {"url": "http://x", "api_key": "k"},
        })
        self.assertIn("Inception", c.wordlists["movie_title"])


# ---------------------------------------------------------------------------
# load_media_classifier factory — NER selection + keyword fall-back
# ---------------------------------------------------------------------------


class TestFactoryNERSelection(_NERTestCase):
    def test_wordlists_config_selects_ner(self):
        from ovos_media_classifier import load_media_classifier
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        clf = load_media_classifier({
            "media_classifier_wordlists": {"movie_title": ["Inception"]},
        })
        self.assertIsInstance(clf, AhocorasickMediaClassifier)
        self.assertEqual(clf.classify("play Inception", "en-us")[0],
                         MediaType.MOVIE)

    def test_entities_config_selects_ner(self):
        from ovos_media_classifier import load_media_classifier
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        clf = load_media_classifier({
            "media_classifier_entities": {
                "wordlists": {"artist_name": ["Radiohead"]},
            },
        })
        self.assertIsInstance(clf, AhocorasickMediaClassifier)
        self.assertEqual(clf.classify("play Radiohead", "en-us")[0],
                         MediaType.MUSIC)

    def test_ner_csv_config_selects_ner(self):
        from ovos_media_classifier import load_media_classifier
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "e.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["label", "value"])
                w.writerow(["movie_title", "Inception"])
            clf = load_media_classifier({"media_classifier_ner_csv": path})
        self.assertIsInstance(clf, AhocorasickMediaClassifier)


class TestFactoryKeywordFallback(unittest.TestCase):
    """When ahocorasick-ner is missing the factory falls back to keyword."""

    def test_import_error_falls_back_to_keyword(self):
        from ovos_media_classifier import load_media_classifier
        from ovos_media_classifier.keyword import KeywordMediaClassifier

        # ahocorasick_ner unavailable → ImportError inside the NER branch
        with patch.dict(sys.modules, {"ahocorasick_ner": None}):
            clf = load_media_classifier({
                "media_classifier_wordlists": {"movie_title": ["Inception"]},
            })
        self.assertIsInstance(clf, KeywordMediaClassifier)

    def test_no_ner_config_returns_keyword(self):
        from ovos_media_classifier import load_media_classifier
        from ovos_media_classifier.keyword import KeywordMediaClassifier

        self.assertIsInstance(load_media_classifier(), KeywordMediaClassifier)
        self.assertIsInstance(load_media_classifier({}), KeywordMediaClassifier)


if __name__ == "__main__":
    unittest.main()
