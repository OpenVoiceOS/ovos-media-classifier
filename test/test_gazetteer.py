"""Tests for Layer A — the offline popularity gazetteer.

Covers building from entity-pool CSVs (popularity ranking + adult exclusion +
movie-contamination subtraction), loading with the per-type cap, defensive
adult-stripping on load, and the offline routing it enables when injected into
the embedding router (no network).
"""
import json
import os
import tempfile
import unittest

import numpy as np

from ovos_media_classifier import gazetteer as gz
from ovos_media_classifier.embedding import GENERIC, EmbeddingMediaClassifier
from ovos_media_classifier.intents import MediaType

# reuse the fake-head helpers from the embedding tests
from test.test_embedding import _router


def _write_entities(d):
    """Write a tiny entity-pool fixture and return the dir."""
    os.makedirs(d, exist_ok=True)

    def w(name, rows, header="value"):
        with open(os.path.join(d, f"{name}.csv"), "w", encoding="utf-8") as fh:
            fh.write(header + "\n")
            for r in rows:
                fh.write(r + "\n")

    # movie pool + a contaminant ("Fight Club") that also leaks into anime
    w("movie_title", ["Interstellar", "Fight Club", "ab"])  # "ab" too short
    w("tv_show_title", ["Stranger Things", "Breaking Bad"])
    w("anime_title", ["Cowboy Bebop", "Fight Club"])  # Fight Club = contaminant
    w("artist_name", ["Radiohead", "Beethoven"])
    w("game_title", ["The Legend of Zelda"])
    # adult pools must be IGNORED entirely
    w("adult_title", ["Some Adult Movie"])
    # imdb votes ranks the movie pool
    with open(os.path.join(d, "_imdb_votes.csv"), "w", encoding="utf-8") as fh:
        fh.write("value,num_votes\n")
        fh.write("Interstellar,2548027\n")
        fh.write("Fight Club,2000000\n")
    return d


class TestGazetteerBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ent = _write_entities(os.path.join(self.tmp, "entities"))

    def test_build_ranks_and_excludes_adult(self):
        gaz = gz.build_gazetteer(self.ent, top_n=100)
        # adult label NEVER present
        self.assertNotIn("adult_title", gaz)
        self.assertNotIn("pornstar", gaz)
        # real labels present
        self.assertIn("movie_title", gaz)
        self.assertIn("artist_name", gaz)

    def test_movie_pool_vote_ranked(self):
        gaz = gz.build_gazetteer(self.ent, top_n=100)
        # Interstellar (more votes) ranks before Fight Club; "ab" dropped (short)
        self.assertEqual(gaz["movie_title"][0], "Interstellar")
        self.assertNotIn("ab", gaz["movie_title"])

    def test_movie_contamination_subtracted_from_anime(self):
        gaz = gz.build_gazetteer(self.ent, top_n=100)
        # "Fight Club" is in movie_title.csv → subtracted from anime
        self.assertIn("Cowboy Bebop", gaz["anime_title"])
        self.assertNotIn("Fight Club", gaz["anime_title"])

    def test_top_n_cap(self):
        gaz = gz.build_gazetteer(self.ent, top_n=1)
        for vals in gaz.values():
            self.assertLessEqual(len(vals), 1)


class TestGazetteerLoad(unittest.TestCase):
    def test_load_applies_cap_and_strips_adult(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "gaz.json")
        # a file that (defensively) contains an adult label + an over-long list
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "movie_title": ["A", "B", "C"],
                "adult_title": ["Bad"],  # must be stripped on load
            }, fh)
        loaded = gz.load_default_gazetteer(top_n=2, path=path)
        self.assertNotIn("adult_title", loaded)
        self.assertEqual(loaded["movie_title"], ["A", "B"])

    def test_load_missing_file_returns_empty(self):
        self.assertEqual(gz.load_default_gazetteer(path="/no/such/file.json"), {})

    def test_bundled_default_exists_and_has_no_adult(self):
        # the wheel ships a default; routing must work offline out of the box
        gaz = gz.load_default_gazetteer(top_n=None)
        self.assertTrue(gaz, "bundled gazetteer should be non-empty")
        self.assertFalse([l for l in gaz if "adult" in l or "porn" in l
                          or "hentai" in l])


class TestGazetteerInjectionOffline(unittest.TestCase):
    """Injecting the gazetteer lets the router route a bare title with NO network."""

    def test_injected_gazetteer_routes_bare_title(self):
        # head abstains; gazetteer-injected anime entity fires → EPISODIC_SERIES
        clf = _router([0.3, 0.3, 0.4], ["music", "movie", GENERIC],
                      entity_labels=["anime_title", "movie_title"])
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "gaz.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"anime_title": ["Cowboy Bebop"]}, fh)
        n = clf.register_default_gazetteer(path=path)
        self.assertEqual(n, 1)
        mt, conf = clf.classify("play cowboy bebop", "en-us")
        self.assertEqual(mt, MediaType.EPISODIC_SERIES)
        self.assertGreater(conf, 0.0)

    def test_no_gazetteer_no_route(self):
        clf = _router([0.3, 0.3, 0.4], ["music", "movie", GENERIC],
                      entity_labels=["anime_title"])
        # without injection the title is unknown → abstain (safe)
        self.assertEqual(clf.classify("play cowboy bebop", "en-us")[0],
                         MediaType.GENERIC)


if __name__ == "__main__":
    unittest.main()
