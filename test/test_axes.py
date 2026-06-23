"""Tests for the multi-axis classification output (Structure + playback_type)."""
import unittest

from mediavocab import MediaType, PlaybackType

from ovos_media_classifier import (
    Structure,
    MediaClassification,
    infer_structure,
    load_media_classifier,
)
from ovos_media_classifier.intents import OCPDomain


class TestStructureAxis(unittest.TestCase):
    def test_single_types(self):
        for mt in (MediaType.MOVIE, MediaType.MUSIC, MediaType.AUDIOBOOK,
                   MediaType.GAME, MediaType.SHORT_FILM):
            self.assertEqual(infer_structure(mt), Structure.SINGLE)

    def test_episodic_types(self):
        for mt in (MediaType.EPISODIC_SERIES, MediaType.PODCAST, MediaType.AUDIO_DRAMA):
            self.assertEqual(infer_structure(mt), Structure.EPISODIC)

    def test_continuous_types(self):
        for mt in (MediaType.RADIO, MediaType.TV, MediaType.PROCEDURAL_AMBIENT):
            self.assertEqual(infer_structure(mt), Structure.CONTINUOUS)

    def test_collection_type(self):
        self.assertEqual(infer_structure(MediaType.PLAYLIST), Structure.COLLECTION)

    def test_every_media_type_has_a_structure(self):
        for mt in MediaType:
            self.assertIsInstance(infer_structure(mt), Structure)


class TestKeywordClassifierAxes(unittest.TestCase):
    def setUp(self):
        self.clf = load_media_classifier()

    def test_playback_type_derived(self):
        # "play a podcast" -> PODCAST -> audio
        self.assertEqual(self.clf.classify_playback_type("play a podcast", "en-us"),
                         PlaybackType.AUDIO)
        # "watch a movie" -> MOVIE -> video
        self.assertEqual(self.clf.classify_playback_type("watch a movie", "en-us"),
                         PlaybackType.VIDEO)

    def test_structure_derived(self):
        self.assertEqual(self.clf.classify_structure("play a movie", "en-us"),
                         Structure.SINGLE)
        self.assertEqual(self.clf.classify_structure("i want to watch a tv show", "en-us"),
                         Structure.EPISODIC)
        self.assertEqual(self.clf.classify_structure("play the radio", "en-us"),
                         Structure.CONTINUOUS)

    def test_classify_full_combines_axes(self):
        res = self.clf.classify_full("i want to watch an anime", "en-us")
        self.assertIsInstance(res, MediaClassification)
        self.assertEqual(res.media_type, MediaType.EPISODIC_SERIES)
        self.assertEqual(res.playback_type, PlaybackType.VIDEO)
        self.assertEqual(res.structure, Structure.EPISODIC)
        self.assertIn("anime", res.genres)
        self.assertEqual(res.domain, OCPDomain.OCP_PLAY)
        d = res.as_dict()
        self.assertEqual(d["playback_type"], "video")
        self.assertEqual(d["structure"], "episodic")

    def test_classify_full_non_media(self):
        res = self.clf.classify_full("what time is it", "en-us")
        self.assertEqual(res.media_type, MediaType.GENERIC)
        self.assertEqual(res.structure, Structure.UNKNOWN)


if __name__ == "__main__":
    unittest.main()


class TestToSignals(unittest.TestCase):
    """The classifier emits a provider-ready mediavocab.Signals."""

    def setUp(self):
        self.clf = load_media_classifier()

    def test_to_signals_carries_axes(self):
        from mediavocab import Signals, PlaybackType, MediaType
        s = self.clf.to_signals("i want to watch an anime", "en-us")
        self.assertIsInstance(s, Signals)
        self.assertEqual(s.medium, MediaType.EPISODIC_SERIES)
        self.assertEqual(s.playback_type, PlaybackType.VIDEO)
        self.assertIn("anime", s.content_genres)
        self.assertEqual(s.title, "i want to watch an anime")
        self.assertEqual(s.language, "en")

    def test_to_signals_non_media_has_no_medium(self):
        s = self.clf.to_signals("what time is it", "en-us")
        self.assertIsNone(s.medium)
