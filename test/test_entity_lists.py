"""Tests for source-agnostic **entity lists** (label → list of strings).

These cover the *list-loading* machinery only — building the set of entity
lists from files / inline dicts / HuggingFace datasets — which is independent
of the Aho-Corasick matcher.  They therefore run WITHOUT the optional
``ahocorasick-ner`` dependency: only ``EntitiesContainer`` (a pure data store)
is exercised, and ``datasets`` / ``requests`` are mocked.

The matcher-coupled paths (``from_container`` / ``classify`` / the factory) are
covered in ``test_ner.py``.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class _Tmp:
    """Context helper writing a file with given content, yielding its path."""

    def __init__(self, name, content):
        self.name = name
        self.content = content
        self._dir = None

    def __enter__(self):
        self._dir = tempfile.TemporaryDirectory()
        path = os.path.join(self._dir.name, self.name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.content)
        return path

    def __exit__(self, *exc):
        self._dir.cleanup()


# ---------------------------------------------------------------------------
# load_tsv
# ---------------------------------------------------------------------------


class TestLoadTSV(unittest.TestCase):
    def test_load_tsv_named_columns(self):
        from ovos_media_classifier.entities import EntitiesContainer

        content = "entity\tlabel\tsource\n" \
                  "The Dark Knight\tmovie_title\tradarr\n" \
                  "Radiohead\tartist_name\tmanual\n"
        with _Tmp("e.tsv", content) as path:
            c = EntitiesContainer()
            added = c.load_tsv(path)
        self.assertEqual(added, 2)
        self.assertIn("The Dark Knight", c.wordlists["movie_title"])
        self.assertIn("Radiohead", c.wordlists["artist_name"])

    def test_load_tsv_label_value_columns(self):
        from ovos_media_classifier.entities import EntitiesContainer

        content = "label\tvalue\nmovie_title\tInception\n"
        with _Tmp("e.tsv", content) as path:
            c = EntitiesContainer()
            added = c.load_tsv(path)
        self.assertEqual(added, 1)
        self.assertIn("Inception", c.wordlists["movie_title"])

    def test_load_tsv_keeps_commas_in_values(self):
        from ovos_media_classifier.entities import EntitiesContainer

        content = "label\tvalue\nalbum_name\tDjango, Unchained\n"
        with _Tmp("e.tsv", content) as path:
            c = EntitiesContainer()
            c.load_tsv(path)
        self.assertIn("Django, Unchained", c.wordlists["album_name"])


# ---------------------------------------------------------------------------
# load_jsonl (both shapes)
# ---------------------------------------------------------------------------


class TestLoadJSONL(unittest.TestCase):
    def test_per_entity_rows(self):
        from ovos_media_classifier.entities import EntitiesContainer

        content = "\n".join([
            json.dumps({"label": "movie_title", "entity": "Inception"}),
            json.dumps({"label": "artist_name", "entity": "Radiohead"}),
            json.dumps({"label": "movie_title", "entity": ""}),  # skipped
        ])
        with _Tmp("e.jsonl", content) as path:
            c = EntitiesContainer()
            added = c.load_jsonl(path)
        self.assertEqual(added, 2)
        self.assertIn("Inception", c.wordlists["movie_title"])
        self.assertIn("Radiohead", c.wordlists["artist_name"])

    def test_per_entity_value_key(self):
        from ovos_media_classifier.entities import EntitiesContainer

        content = json.dumps({"label": "track_name", "value": "Karma Police"})
        with _Tmp("e.jsonl", content) as path:
            c = EntitiesContainer()
            c.load_jsonl(path)
        self.assertIn("Karma Police", c.wordlists["track_name"])

    def test_list_rows(self):
        from ovos_media_classifier.entities import EntitiesContainer

        content = "\n".join([
            json.dumps({"artist_name": ["Radiohead", "Bjork"]}),
            json.dumps({"movie_title": "The Matrix"}),  # single string ok
        ])
        with _Tmp("e.jsonl", content) as path:
            c = EntitiesContainer()
            added = c.load_jsonl(path)
        self.assertEqual(added, 3)
        self.assertIn("Radiohead", c.wordlists["artist_name"])
        self.assertIn("Bjork", c.wordlists["artist_name"])
        self.assertIn("The Matrix", c.wordlists["movie_title"])

    def test_mixed_shapes_and_blank_lines(self):
        from ovos_media_classifier.entities import EntitiesContainer

        content = "\n".join([
            json.dumps({"label": "movie_title", "entity": "Inception"}),
            "",  # blank line ignored
            json.dumps({"artist_name": ["Radiohead"]}),
        ])
        with _Tmp("e.jsonl", content) as path:
            c = EntitiesContainer()
            added = c.load_jsonl(path)
        self.assertEqual(added, 2)

    def test_malformed_line_skipped(self):
        from ovos_media_classifier.entities import EntitiesContainer

        content = "\n".join([
            "{not valid json",
            json.dumps({"label": "movie_title", "entity": "Inception"}),
        ])
        with _Tmp("e.jsonl", content) as path:
            c = EntitiesContainer()
            added = c.load_jsonl(path)
        self.assertEqual(added, 1)
        self.assertIn("Inception", c.wordlists["movie_title"])


# ---------------------------------------------------------------------------
# load_source dispatch by extension / shape
# ---------------------------------------------------------------------------


class TestLoadSourceDispatch(unittest.TestCase):
    def test_dispatch_csv(self):
        from ovos_media_classifier.entities import EntitiesContainer

        content = "label,value\nmovie_title,Inception\n"
        with _Tmp("e.csv", content) as path:
            c = EntitiesContainer()
            c.load_source(path)
        self.assertIn("Inception", c.wordlists["movie_title"])

    def test_dispatch_tsv(self):
        from ovos_media_classifier.entities import EntitiesContainer

        content = "label\tvalue\nartist_name\tRadiohead\n"
        with _Tmp("e.tsv", content) as path:
            c = EntitiesContainer()
            c.load_source(path)
        self.assertIn("Radiohead", c.wordlists["artist_name"])

    def test_dispatch_jsonl(self):
        from ovos_media_classifier.entities import EntitiesContainer

        content = json.dumps({"movie_title": ["Dune"]})
        with _Tmp("e.jsonl", content) as path:
            c = EntitiesContainer()
            c.load_source(path)
        self.assertIn("Dune", c.wordlists["movie_title"])

    def test_dispatch_inline_dict(self):
        from ovos_media_classifier.entities import EntitiesContainer

        c = EntitiesContainer()
        added = c.load_source({"artist_name": ["Radiohead", "Bjork"]})
        self.assertEqual(added, 2)
        self.assertIn("Bjork", c.wordlists["artist_name"])

    def test_dispatch_inline_single_string(self):
        from ovos_media_classifier.entities import EntitiesContainer

        c = EntitiesContainer()
        c.load_source({"movie_title": "Inception"})
        self.assertIn("Inception", c.wordlists["movie_title"])

    def test_dispatch_huggingface_dict(self):
        from ovos_media_classifier.entities import EntitiesContainer

        fake_datasets = type(sys)("datasets")
        fake_datasets.load_dataset = MagicMock(return_value=[
            {"entity": "Inception", "label": "movie_title"},
        ])
        c = EntitiesContainer()
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            added = c.load_source({"dataset": "Some/dataset"})
        self.assertEqual(added, 1)
        self.assertIn("Inception", c.wordlists["movie_title"])

    @patch("ovos_media_classifier.entities._http_get")
    def test_dispatch_media_server_single_key(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        mock_http_get.return_value = [
            {"title": "Inception", "genres": [], "alternateTitles": [],
             "credits": {}, "studio": ""},
        ]
        c = EntitiesContainer()
        added = c.load_source({"radarr": {"url": "http://x", "api_key": "k"}})
        self.assertGreater(added, 0)
        self.assertIn("Inception", c.wordlists["movie_title"])

    @patch("ovos_media_classifier.entities._http_get")
    def test_dispatch_media_server_type_hint(self, mock_http_get):
        from ovos_media_classifier.entities import EntitiesContainer

        mock_http_get.return_value = [
            {"title": "Inception", "genres": [], "alternateTitles": [],
             "credits": {}, "studio": ""},
        ]
        c = EntitiesContainer()
        added = c.load_source(
            {"type": "radarr", "url": "http://x", "api_key": "k"})
        self.assertGreater(added, 0)
        self.assertIn("Inception", c.wordlists["movie_title"])

    def test_unknown_extension_raises(self):
        from ovos_media_classifier.entities import EntitiesContainer

        c = EntitiesContainer()
        with self.assertRaises(ValueError):
            c.load_source("/data/library.parquet")

    def test_unknown_spec_type_raises(self):
        from ovos_media_classifier.entities import EntitiesContainer

        c = EntitiesContainer()
        with self.assertRaises(ValueError):
            c.load_source(12345)  # not a str/dict


# ---------------------------------------------------------------------------
# from_sources / load_lists
# ---------------------------------------------------------------------------


class TestFromSources(unittest.TestCase):
    def test_from_sources_mixed(self):
        from ovos_media_classifier.entities import EntitiesContainer

        csv_content = "label,value\nmovie_title,Inception\n"
        jsonl_content = json.dumps({"artist_name": ["Radiohead"]})
        with _Tmp("e.csv", csv_content) as csv_path, \
                _Tmp("e.jsonl", jsonl_content) as jsonl_path:
            c = EntitiesContainer.from_sources([
                csv_path,
                jsonl_path,
                {"album_name": ["OK Computer"]},
            ])
        self.assertIn("Inception", c.wordlists["movie_title"])
        self.assertIn("Radiohead", c.wordlists["artist_name"])
        self.assertIn("OK Computer", c.wordlists["album_name"])

    def test_load_lists_skips_bad_spec(self):
        from ovos_media_classifier.entities import EntitiesContainer

        c = EntitiesContainer()
        # one good, one missing file → bad spec is logged & skipped
        added = c.load_lists([
            {"artist_name": ["Radiohead"]},
            "/nonexistent/path.csv",
        ])
        self.assertEqual(added, 1)
        self.assertIn("Radiohead", c.wordlists["artist_name"])

    def test_load_lists_returns_total_added(self):
        from ovos_media_classifier.entities import EntitiesContainer

        c = EntitiesContainer()
        added = c.load_lists([
            {"artist_name": ["Radiohead"]},
            {"movie_title": ["Inception", "Dune"]},
        ])
        self.assertEqual(added, 3)


# ---------------------------------------------------------------------------
# from_config with entity_lists (+ back-compat with structured keys)
# ---------------------------------------------------------------------------


class TestFromConfigEntityLists(unittest.TestCase):
    def test_entity_lists_key(self):
        from ovos_media_classifier.entities import EntitiesContainer

        csv_content = "label,value\nmovie_title,Inception\n"
        with _Tmp("e.csv", csv_content) as csv_path:
            c = EntitiesContainer.from_config({
                "entity_lists": [
                    csv_path,
                    {"artist_name": ["Radiohead"]},
                ],
            })
        self.assertIn("Inception", c.wordlists["movie_title"])
        self.assertIn("Radiohead", c.wordlists["artist_name"])

    def test_entity_lists_alongside_structured_keys(self):
        from ovos_media_classifier.entities import EntitiesContainer

        c = EntitiesContainer.from_config({
            "entity_lists": [{"artist_name": ["Radiohead"]}],
            "wordlists": {"movie_title": ["Inception"]},
        })
        # both the new entity_lists path and the legacy wordlists key merge
        self.assertIn("Radiohead", c.wordlists["artist_name"])
        self.assertIn("Inception", c.wordlists["movie_title"])

    def test_entity_lists_with_jsonl_and_tsv(self):
        from ovos_media_classifier.entities import EntitiesContainer

        tsv_content = "label\tvalue\nalbum_name\tOK Computer\n"
        jsonl_content = json.dumps({"track_name": ["Karma Police"]})
        with _Tmp("e.tsv", tsv_content) as tsv_path, \
                _Tmp("e.jsonl", jsonl_content) as jsonl_path:
            c = EntitiesContainer.from_config({
                "entity_lists": [tsv_path, jsonl_path],
            })
        self.assertIn("OK Computer", c.wordlists["album_name"])
        self.assertIn("Karma Police", c.wordlists["track_name"])


# ---------------------------------------------------------------------------
# Factory: media_classifier_entities with entity_lists routes to the NER backend
# ---------------------------------------------------------------------------


class _FakeAhocorasickNER:
    """Minimal AhocorasickNER stand-in (substring matcher) — see test_ner.py."""

    def __init__(self):
        self._words = {}

    def add_word(self, label, word):
        self._words.setdefault(label, [])
        if word not in self._words[label]:
            self._words[label].append(word)

    def get_all_words(self):
        return {l: list(w) for l, w in self._words.items()}

    def tag(self, query):
        ql = query.lower()
        return [{"label": l, "word": w}
                for l, words in self._words.items()
                for w in words if w.lower() in ql]


class TestFactoryEntityLists(unittest.TestCase):
    def setUp(self):
        mod = type(sys)("ahocorasick_ner")
        mod.AhocorasickNER = _FakeAhocorasickNER
        self._p = patch.dict(sys.modules, {"ahocorasick_ner": mod})
        self._p.start()
        self.addCleanup(self._p.stop)

    def test_factory_entity_lists_selects_ner(self):
        from ovos_media_classifier import load_media_classifier
        from ovos_media_classifier.ahocorasick import AhocorasickMediaClassifier
        from ovos_media_classifier.intents import MediaType

        jsonl = json.dumps({"artist_name": ["Radiohead"]})
        with _Tmp("e.jsonl", jsonl) as path:
            clf = load_media_classifier({
                "media_classifier_entities": {
                    "entity_lists": [
                        path,
                        {"movie_title": ["Inception"]},
                    ],
                },
            })
        self.assertIsInstance(clf, AhocorasickMediaClassifier)
        self.assertEqual(clf.classify("play Radiohead", "en-us")[0],
                         MediaType.MUSIC)
        self.assertEqual(clf.classify("watch Inception", "en-us")[0],
                         MediaType.MOVIE)


if __name__ == "__main__":
    unittest.main()
