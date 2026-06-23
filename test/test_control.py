"""Tests for OCP *control*-action classification.

control vs play is the domain axis — so it is classification and lives here in
``ovos-media-classifier`` (the lean keyword backend via its ``Ctrl*.voc`` files),
not in the pipeline.
"""
import unittest

from ovos_media_classifier import (
    KeywordMediaClassifier,
    OCPDomain,
    OCPControlIntent,
)
from ovos_media_classifier.axes import MediaClassification


class TestClassifyControl(unittest.TestCase):
    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def _ctrl(self, utt, lang="en-us"):
        return self.clf.classify_control(utt, lang)

    def _dom(self, utt, lang="en-us"):
        dom, _ = self.clf.classify_domain(utt, lang)
        return dom

    # ---- transport controls -> OCP_CONTROL + the right action --------------

    def test_pause(self):
        self.assertEqual(self._dom("pause"), OCPDomain.OCP_CONTROL)
        self.assertEqual(self._ctrl("pause"), OCPControlIntent.PAUSE)

    def test_pause_with_object(self):
        self.assertEqual(self._dom("pause the video"), OCPDomain.OCP_CONTROL)
        self.assertEqual(self._ctrl("pause the video"), OCPControlIntent.PAUSE)

    def test_stop_the_music(self):
        # the media noun is the *object* of the control, not a new play request
        self.assertEqual(self._dom("stop the music"), OCPDomain.OCP_CONTROL)
        self.assertEqual(self._ctrl("stop the music"), OCPControlIntent.STOP)

    def test_next_track(self):
        self.assertEqual(self._dom("next track"), OCPDomain.OCP_CONTROL)
        self.assertEqual(self._ctrl("next track"), OCPControlIntent.NEXT)

    def test_skip(self):
        self.assertEqual(self._dom("skip"), OCPDomain.OCP_CONTROL)
        self.assertEqual(self._ctrl("skip"), OCPControlIntent.NEXT)

    def test_previous(self):
        self.assertEqual(self._ctrl("previous"), OCPControlIntent.PREVIOUS)

    def test_go_back_a_track(self):
        # "go back" (no duration) is a track change, not a seek
        self.assertEqual(self._ctrl("go back a track"), OCPControlIntent.PREVIOUS)

    def test_shuffle_my_playlist(self):
        self.assertEqual(self._dom("shuffle my playlist"), OCPDomain.OCP_CONTROL)
        self.assertEqual(self._ctrl("shuffle my playlist"), OCPControlIntent.SHUFFLE)

    def test_repeat(self):
        self.assertEqual(self._ctrl("loop this"), OCPControlIntent.REPEAT)
        self.assertEqual(self._ctrl("play again"), OCPControlIntent.REPEAT)

    def test_rewind_seconds_is_seek_backward(self):
        self.assertEqual(self._dom("rewind 30 seconds"), OCPDomain.OCP_CONTROL)
        self.assertEqual(self._ctrl("rewind 30 seconds"),
                         OCPControlIntent.SEEK_BACKWARD)

    def test_go_back_seconds_is_seek_backward(self):
        # the trailing duration disambiguates "go back" from PREVIOUS
        self.assertEqual(self._ctrl("go back 30 seconds"),
                         OCPControlIntent.SEEK_BACKWARD)

    def test_fast_forward_is_seek_forward(self):
        self.assertEqual(self._ctrl("fast forward"), OCPControlIntent.SEEK_FORWARD)

    def test_skip_ahead_seconds_is_seek_forward(self):
        self.assertEqual(self._ctrl("skip ahead 10 seconds"),
                         OCPControlIntent.SEEK_FORWARD)

    def test_open_player(self):
        self.assertEqual(self._ctrl("open the player"), OCPControlIntent.OPEN)

    def test_save_game(self):
        self.assertEqual(self._dom("save the game"), OCPDomain.OCP_CONTROL)
        self.assertEqual(self._ctrl("save the game"), OCPControlIntent.SAVE_GAME)

    def test_load_game(self):
        self.assertEqual(self._ctrl("load last game"), OCPControlIntent.LOAD_GAME)

    def test_like_song(self):
        self.assertEqual(self._ctrl("i like this song"), OCPControlIntent.LIKE_SONG)

    # ---- bare play / resume / continue -> CONTROL (resume current) ----------

    def test_resume(self):
        self.assertEqual(self._dom("resume"), OCPDomain.OCP_CONTROL)
        self.assertEqual(self._ctrl("resume"), OCPControlIntent.RESUME)

    def test_continue(self):
        self.assertEqual(self._ctrl("continue"), OCPControlIntent.RESUME)

    def test_bare_play_is_control_play(self):
        # bare "play" with no media is resume-current, a control action
        self.assertEqual(self._dom("play"), OCPDomain.OCP_CONTROL)
        self.assertEqual(self._ctrl("play"), OCPControlIntent.PLAY)

    # ---- the "play <media>" ambiguity: PLAY, never control ------------------

    def test_play_some_jazz_is_play_not_control(self):
        # an explicit play verb + content is a *new play request*, not control
        self.assertEqual(self._dom("play some jazz"), OCPDomain.OCP_PLAY)
        self.assertIsNone(self._ctrl("play some jazz"))

    def test_play_the_news_is_play(self):
        self.assertEqual(self._dom("play the news"), OCPDomain.OCP_PLAY)
        self.assertIsNone(self._ctrl("play the news"))

    def test_listen_to_jazz_is_play(self):
        self.assertEqual(self._dom("listen to jazz"), OCPDomain.OCP_PLAY)

    # ---- non-media stays NOT_OCP -------------------------------------------

    def test_non_media_not_ocp(self):
        self.assertEqual(self._dom("what time is it"), OCPDomain.NOT_OCP)
        self.assertIsNone(self._ctrl("what time is it"))


class TestClassifyFullControl(unittest.TestCase):
    """``classify_full`` carries the control action on ``control_intent``."""

    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_control_intent_field_populated(self):
        res = self.clf.classify_full("pause", "en-us")
        self.assertIsInstance(res, MediaClassification)
        self.assertEqual(res.domain, OCPDomain.OCP_CONTROL)
        self.assertEqual(res.control_intent, OCPControlIntent.PAUSE)

    def test_control_intent_in_as_dict(self):
        res = self.clf.classify_full("next track", "en-us")
        d = res.as_dict()
        self.assertEqual(d["domain"], "ocp_control")
        self.assertEqual(d["control_intent"], "next")

    def test_play_has_no_control_intent(self):
        res = self.clf.classify_full("play some jazz", "en-us")
        self.assertEqual(res.domain, OCPDomain.OCP_PLAY)
        self.assertIsNone(res.control_intent)
        self.assertIsNone(res.as_dict()["control_intent"])

    def test_not_ocp_has_no_control_intent(self):
        res = self.clf.classify_full("set a timer", "en-us")
        self.assertEqual(res.domain, OCPDomain.NOT_OCP)
        self.assertIsNone(res.control_intent)


class TestAbstractDefault(unittest.TestCase):
    """The base ``classify_control`` defaults to ``None``."""

    def test_base_default_none(self):
        from ovos_media_classifier.base import AbstractMediaClassifier
        from ovos_media_classifier import MediaType

        class _Dummy(AbstractMediaClassifier):
            def classify(self, query, lang, valid_labels=None):
                return MediaType.MOVIE, 1.0

        self.assertIsNone(_Dummy().classify_control("pause", "en-us"))


if __name__ == "__main__":
    unittest.main()
