import unittest
from ovos_media_classifier import (KEYWORD_FEATURE_SLOTS, slot_for_label,
                                    slots_for_media_type, KeywordFeatureSlot)
from mediavocab import MediaType, PlaybackType


class TestKeywordFeatureSlots(unittest.TestCase):
    def test_slots_built(self):
        self.assertGreater(len(KEYWORD_FEATURE_SLOTS), 50)
        self.assertTrue(all(isinstance(s, KeywordFeatureSlot) for s in KEYWORD_FEATURE_SLOTS))

    def test_known_slots(self):
        self.assertEqual(slot_for_label("artist_name").media_type, MediaType.MUSIC)
        self.assertEqual(slot_for_label("artist_name").playback_type, PlaybackType.AUDIO)
        self.assertEqual(slot_for_label("movie_title").media_type, MediaType.MOVIE)
        self.assertEqual(slot_for_label("game_title").playback_type, PlaybackType.INTERACTIVE)

    def test_genre_survives_slot(self):
        # content-filter signal carried by the slot
        self.assertIn("adult", slot_for_label("hentai_title").genres)

    def test_slots_for_media_type(self):
        music = {s.label for s in slots_for_media_type(MediaType.MUSIC)}
        self.assertIn("artist_name", music)
        self.assertIn("album_name", music)

    def test_unknown_label_raises(self):
        with self.assertRaises(KeyError):
            slot_for_label("not_a_real_slot")


if __name__ == "__main__":
    unittest.main()
