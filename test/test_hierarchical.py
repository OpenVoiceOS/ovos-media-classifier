"""Tests for the hierarchical coarse-to-fine keyword classifier.

These prove the *constraint* property the refactor is about: the high-signal
coarse axes (modality, then structure) are predicted FIRST and constrain the
leaf candidate set, so a stray specific keyword cannot win when it disagrees
with the predicted modality/structure.

The cases use the bundled ``en-us`` locale (the reference path), exercising the
real ``.voc`` files end-to-end.
"""
import unittest

from mediavocab import MediaType, PlaybackType

from ovos_media_classifier import KeywordMediaClassifier
from ovos_media_classifier.axes import Structure
from ovos_media_classifier.intents import OCPDomain


class TestModalityConstrainsLeaf(unittest.TestCase):
    """The modality axis (audio vs video) gates which leaf can win."""

    def setUp(self):
        self.clf = KeywordMediaClassifier()  # bundled en-us locale

    def test_listen_to_the_news_is_audio_radio(self):
        # "news" is an audio leaf; "listen" reinforces audio -> RADIO
        mt, _ = self.clf.classify("listen to the news", "en-us")
        self.assertEqual(mt, MediaType.RADIO)
        self.assertEqual(self.clf.classify_playback_type("listen to the news", "en-us"),
                         PlaybackType.AUDIO)

    def test_watch_the_news_is_video_tv(self):
        # SAME "news" leaf, but the explicit video verb "watch" flips the
        # modality to video, constraining the leaf to a video type -> TV.
        mt, _ = self.clf.classify("watch the news", "en-us")
        self.assertEqual(mt, MediaType.TV)
        self.assertEqual(self.clf.classify_playback_type("watch the news", "en-us"),
                         PlaybackType.VIDEO)

    def test_strong_audio_verb_overrides_weak_video_leaf(self):
        # "video" is a (weak) video leaf, but two audio cues ("listen" +
        # "podcast") outscore it: the audio modality wins and a video leaf
        # cannot be returned.
        mt = self.clf.classify_playback_type("listen to the podcast video recap", "en-us")
        self.assertEqual(mt, PlaybackType.AUDIO)

    def test_watch_constrains_audio_leaf_away(self):
        # "radio" is an audio leaf; "watch live" forces video -> must NOT be RADIO
        mt, _ = self.clf.classify("watch live radio show", "en-us")
        self.assertEqual(self.clf.classify_playback_type("watch live radio show", "en-us"),
                         PlaybackType.VIDEO)
        self.assertNotEqual(mt, MediaType.RADIO)


class TestStructureAxis(unittest.TestCase):
    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_episode_cue_is_episodic(self):
        self.assertEqual(self.clf.classify_structure("play the next episode", "en-us"),
                         Structure.EPISODIC)

    def test_series_keyword_is_episodic(self):
        self.assertEqual(self.clf.classify_structure("watch a tv show", "en-us"),
                         Structure.EPISODIC)

    def test_season_cue_is_episodic_series_leaf(self):
        mt, _ = self.clf.classify("watch the new season", "en-us")
        self.assertEqual(mt, MediaType.EPISODIC_SERIES)
        self.assertEqual(self.clf.classify_structure("watch the new season", "en-us"),
                         Structure.EPISODIC)

    def test_live_cue_is_continuous(self):
        self.assertEqual(self.clf.classify_structure("watch live tv", "en-us"),
                         Structure.CONTINUOUS)

    def test_tune_in_is_continuous_radio(self):
        mt, _ = self.clf.classify("tune in to the radio", "en-us")
        self.assertEqual(mt, MediaType.RADIO)
        self.assertEqual(self.clf.classify_structure("tune in to the radio", "en-us"),
                         Structure.CONTINUOUS)

    def test_live_channel_is_continuous_tv(self):
        mt, _ = self.clf.classify("watch the live channel", "en-us")
        self.assertEqual(mt, MediaType.TV)
        self.assertEqual(self.clf.classify_structure("watch the live channel", "en-us"),
                         Structure.CONTINUOUS)


class TestDefaultLeafFallback(unittest.TestCase):
    """When the coarse axes are confident but no specific leaf voc matches,
    the (modality, structure) DEFAULT leaf is emitted."""

    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_audio_verb_alone_defaults_to_music(self):
        # "listen" -> audio, no structure cue -> single -> default MUSIC
        mt, _ = self.clf.classify("listen to something nice", "en-us")
        self.assertEqual(mt, MediaType.MUSIC)

    def test_video_verb_alone_defaults_to_movie(self):
        mt, _ = self.clf.classify("watch something good", "en-us")
        self.assertEqual(mt, MediaType.MOVIE)

    def test_audio_verb_plus_live_defaults_to_radio(self):
        # audio + continuous, but no radio leaf voc word -> default RADIO
        mt, _ = self.clf.classify("tune in to my favourite station", "en-us")
        self.assertEqual(mt, MediaType.RADIO)


class TestPredictedAxesNotDerived(unittest.TestCase):
    """classify_full / classify_playback_type / classify_structure must PREDICT
    the axes top-down, not derive them from the leaf."""

    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_classify_full_predicts_video_for_watch_the_news(self):
        res = self.clf.classify_full("watch the news", "en-us")
        self.assertEqual(res.media_type, MediaType.TV)
        self.assertEqual(res.playback_type, PlaybackType.VIDEO)
        self.assertEqual(res.structure, Structure.CONTINUOUS)
        self.assertEqual(res.domain, OCPDomain.OCP_PLAY)

    def test_classify_full_predicts_audio_for_listen_to_the_news(self):
        res = self.clf.classify_full("listen to the news", "en-us")
        self.assertEqual(res.media_type, MediaType.RADIO)
        self.assertEqual(res.playback_type, PlaybackType.AUDIO)
        self.assertEqual(res.structure, Structure.CONTINUOUS)

    def test_non_media_is_unknown_axes(self):
        res = self.clf.classify_full("what time is it", "en-us")
        self.assertEqual(res.media_type, MediaType.GENERIC)
        self.assertEqual(res.playback_type, PlaybackType.UNKNOWN)
        self.assertEqual(res.structure, Structure.UNKNOWN)
        self.assertEqual(res.domain, OCPDomain.NOT_OCP)


class TestAdultSafetyPreserved(unittest.TestCase):
    """Adult precedence must survive the hierarchical rewrite: adult/hentai/porn
    cues still surface the ``adult`` genre so the content filter blocks them."""

    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_porn_surfaces_adult_genre(self):
        genres = self.clf.classify_genres("play porn", "en-us")
        self.assertIn("adult", genres)

    def test_adult_anime_is_blockable_and_episodic_series(self):
        # "porn" is in the en-us AdultKeyword voc ("adult" itself is not); with
        # an anime cue this resolves to the hentai intent: EPISODIC_SERIES but
        # tagged BOTH anime and adult so the content filter blocks it.
        mt, _ = self.clf.classify("watch porn anime", "en-us")
        self.assertEqual(mt, MediaType.EPISODIC_SERIES)
        genres = self.clf.classify_genres("watch porn anime", "en-us")
        self.assertIn("anime", genres)
        self.assertIn("adult", genres)

    def test_hentai_surfaces_adult_and_anime(self):
        genres = self.clf.classify_genres("play hentai", "en-us")
        self.assertIn("adult", genres)
        self.assertIn("anime", genres)


class TestGracefulDegradation(unittest.TestCase):
    """A locale without the axis vocab must not crash — it degrades to
    leaf-only matching."""

    def test_leaf_only_when_no_axis_vocab(self):
        # A matcher that only knows leaf vocs (no Verb*/Mod* axis files).
        def _voc_match(phrase, vocab, **kw):
            return vocab == "MusicKeyword"

        clf = KeywordMediaClassifier(_voc_match)
        mt, _ = clf.classify("play music", "xx-xx")
        self.assertEqual(mt, MediaType.MUSIC)
        # axes still resolvable (fall back to the leaf's intrinsic modality)
        self.assertEqual(clf.classify_playback_type("play music", "xx-xx"),
                         PlaybackType.AUDIO)


if __name__ == "__main__":
    unittest.main()
