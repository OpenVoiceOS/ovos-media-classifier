"""Tests for context-aware classification (``classify_full`` minimal context).

Covers the two standalone context inputs:

* ``player_status`` — now-playing media_type + transport state → relative
  control ("next"/"pause" → OCP_CONTROL), the "play something else" re-query,
  the light follow-up bias, and the conservative guarantees (explicit route wins,
  no hijack of unrelated speech, no effect when stopped / no context);
* ``ner_list`` — the available-entity context threaded per-query into the
  embedding router's entity stream (injection routing, no retrain).
"""
import unittest

from mediavocab import MediaType

from ovos_media_classifier import KeywordMediaClassifier
from ovos_media_classifier.context import PlayerStatus, PlayerState
from ovos_media_classifier.intents import OCPDomain, OCPControlIntent


def _playing(media_type=MediaType.MUSIC):
    return PlayerStatus(now_playing=media_type, state=PlayerState.PLAYING)


class TestPlayerStatusControl(unittest.TestCase):
    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_next_with_active_session_is_control(self):
        r = self.clf.classify_full("next", "en-us", player_status=_playing())
        self.assertEqual(r.domain, OCPDomain.OCP_CONTROL)
        self.assertEqual(r.control_intent, OCPControlIntent.NEXT)

    def test_pause_with_active_session_is_control(self):
        r = self.clf.classify_full("pause", "en-us", player_status=_playing())
        self.assertEqual(r.domain, OCPDomain.OCP_CONTROL)
        self.assertEqual(r.control_intent, OCPControlIntent.PAUSE)


class TestPlayerStatusRelative(unittest.TestCase):
    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_play_something_else_requeries_current_type(self):
        r = self.clf.classify_full("play something else", "en-us",
                                   player_status=_playing(MediaType.MUSIC))
        self.assertEqual(r.domain, OCPDomain.OCP_PLAY)
        self.assertEqual(r.media_type, MediaType.MUSIC)

    def test_play_something_else_video_session(self):
        r = self.clf.classify_full("play something else", "en-us",
                                   player_status=_playing(MediaType.MOVIE))
        self.assertEqual(r.media_type, MediaType.MOVIE)


class TestPlayerStatusConservative(unittest.TestCase):
    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_explicit_route_wins_over_context(self):
        # music is playing, but an explicit "play a movie" must stay MOVIE
        r = self.clf.classify_full("play a movie", "en-us",
                                   player_status=_playing(MediaType.MUSIC))
        self.assertEqual(r.media_type, MediaType.MOVIE)

    def test_unrelated_speech_not_hijacked(self):
        r = self.clf.classify_full("what time is it", "en-us",
                                   player_status=_playing())
        self.assertEqual(r.domain, OCPDomain.NOT_OCP)
        self.assertEqual(r.media_type, MediaType.GENERIC)

    def test_no_context_is_baseline(self):
        with_ctx = self.clf.classify_full("play something else", "en-us")
        # without context "play something else" has no concrete leaf
        self.assertEqual(with_ctx.media_type, MediaType.GENERIC)

    def test_stopped_session_has_no_effect(self):
        ps = PlayerStatus(now_playing=MediaType.MUSIC, state=PlayerState.STOPPED)
        r = self.clf.classify_full("play something else", "en-us",
                                   player_status=ps)
        self.assertEqual(r.media_type, MediaType.GENERIC)

    def test_paused_session_is_active(self):
        ps = PlayerStatus(now_playing=MediaType.MOVIE, state=PlayerState.PAUSED)
        self.assertTrue(ps.is_active)
        r = self.clf.classify_full("play something else", "en-us",
                                   player_status=ps)
        self.assertEqual(r.media_type, MediaType.MOVIE)


class TestPlayerStatusFromDict(unittest.TestCase):
    def test_from_dict_roundtrip(self):
        ps = PlayerStatus.from_dict({"now_playing": "music", "state": "playing"})
        self.assertEqual(ps.now_playing, MediaType.MUSIC)
        self.assertEqual(ps.state, PlayerState.PLAYING)

    def test_from_dict_tolerant(self):
        # garbage values degrade safely, never raise
        ps = PlayerStatus.from_dict({"now_playing": "not_a_type", "state": "??"})
        self.assertIsNone(ps.now_playing)
        self.assertEqual(ps.state, PlayerState.STOPPED)

    def test_from_dict_none(self):
        self.assertIsNone(PlayerStatus.from_dict(None))
        self.assertIsNone(PlayerStatus.from_dict({}))


class TestNerListInjectionRouting(unittest.TestCase):
    """``ner_list`` threads the available entities into the router per-query."""

    def _hybrid(self):
        from ovos_media_classifier.embedding import HybridMediaClassifier
        from test.test_embedding import _router
        # a router head that abstains (entity context must drive routing)
        router = _router([0.34, 0.33, 0.33], ["music", "movie", "GENERIC"],
                         entity_labels=["anime_title", "movie_title"])
        return HybridMediaClassifier(KeywordMediaClassifier(), router)

    def test_ner_list_routes_bare_title(self):
        clf = self._hybrid()
        # without the entity context the keyword "watch" cue gives MOVIE
        base = clf.classify_full("watch attack on titan", "en-us")
        self.assertEqual(base.media_type, MediaType.MOVIE)
        # injecting the user's anime library routes to EPISODIC_SERIES
        ner = {"anime_title": ["Attack on Titan"]}
        ctx = clf.classify_full("watch attack on titan", "en-us", ner_list=ner)
        self.assertEqual(ctx.media_type, MediaType.EPISODIC_SERIES)

    def test_keyword_ner_list_inert(self):
        # the keyword backend has no entity stream: ner_list must not raise / change
        clf = KeywordMediaClassifier()
        r = clf.classify_full("play some music", "en-us",
                              ner_list={"artist_name": ["Whoever"]})
        self.assertEqual(r.media_type, MediaType.MUSIC)


if __name__ == "__main__":
    unittest.main()
