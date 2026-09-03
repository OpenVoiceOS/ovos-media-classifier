"""Tests for the negative-evidence guards on the keyword backend.

Three ways a non-media utterance used to be routed as media, and one way the
reported axes could contradict each other:

* smart-home commands ("turn off the lights") matched a bare control phrase;
* the entity context (``ner_list``) was trusted to be well-formed;
* intra-word punctuation ("por-n") slipped past the content keywords;
* a coarse modality tie ("a video game") outvoted the resolved leaf.
"""
import unittest

from mediavocab import MediaType, PlaybackType

from ovos_media_classifier import (
    KeywordMediaClassifier,
    OCPDomain,
    OCPControlIntent,
)
from ovos_media_classifier.content_filter import ContentFilter


class TestIoTNegativeEvidence(unittest.TestCase):
    """A smart-home object with no media evidence is not a media request."""

    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_iot_commands_abstain_completely(self):
        for utt in ("turn off the lights", "turn on the lights",
                    "turn off the heater", "turn off wifi",
                    "shut off the porch light",
                    "switch off the thermostat"):
            with self.subTest(utt):
                self.assertIsNone(self.clf.classify_control(utt, "en-us"))
                self.assertEqual(self.clf.classify_domain(utt, "en-us"),
                                 (OCPDomain.NOT_OCP, 0.0))
                self.assertEqual(self.clf.classify(utt, "en-us"),
                                 (MediaType.GENERIC, 0.0))
                self.assertFalse(self.clf.is_ocp_query(utt, "en-us")[0])

    def test_iot_words_in_other_languages(self):
        for utt, lang in (("apaga las luces", "es-es"),
                          ("mach das licht aus", "de-de"),
                          ("desliga as luzes", "pt-pt"),
                          ("éteins la lumière", "fr-fr"),
                          ("spegni le luci", "it-it"),
                          ("doe het licht uit", "nl-nl")):
            with self.subTest(utt):
                self.assertEqual(self.clf.classify_domain(utt, lang),
                                 (OCPDomain.NOT_OCP, 0.0))

    def test_an_unambiguous_media_verb_outranks_the_iot_word(self):
        # a play request over an IoT-sounding object is still a play request
        for utt in ("play doorbell sounds", "play fan noise"):
            with self.subTest(utt):
                domain, conf = self.clf.classify_domain(utt, "en-us")
                self.assertEqual(domain, OCPDomain.OCP_PLAY)
                self.assertGreater(conf, 0.0)
        # and a strong control phrase still controls
        for utt, action in (("stop light show", OCPControlIntent.STOP),
                            ("stop the alarm", OCPControlIntent.STOP),
                            ("pause the vacuum", OCPControlIntent.PAUSE)):
            with self.subTest(utt):
                self.assertEqual(self.clf.classify_control(utt, "en-us"), action)
                self.assertEqual(self.clf.classify_domain(utt, "en-us")[0],
                                 OCPDomain.OCP_CONTROL)

    def test_ambient_audio_is_media_only_under_a_play_verb(self):
        # "play X sounds" asks for a soundscape...
        for utt in ("play doorbell sounds", "play fan noise"):
            with self.subTest(utt):
                self.assertEqual(self.clf.classify_domain(utt, "en-us")[0],
                                 OCPDomain.OCP_PLAY)
        # ...while the same noun under a control verb mutes an appliance
        for utt in ("turn off alarm sounds", "turn off notification sounds",
                    "silence the alarm sounds", "turn off doorbell sounds"):
            with self.subTest(utt):
                self.assertIsNone(self.clf.classify_control(utt, "en-us"))
                self.assertEqual(self.clf.classify_domain(utt, "en-us"),
                                 (OCPDomain.NOT_OCP, 0.0))
        # "light show" is a full media object, so control verbs still control
        self.assertEqual(self.clf.classify_control("stop light show", "en-us"),
                         OCPControlIntent.STOP)

    def test_entity_context_reaches_the_control_axis(self):
        # a title the user has, spelled like the smart-home object: the request
        # is a control on that title, not a refusal
        result = self.clf.classify_full("turn off the lights", "en-us",
                                        ner_list={"song": "lights"})
        self.assertNotEqual(result.domain, OCPDomain.NOT_OCP)
        self.assertEqual(result.domain, OCPDomain.OCP_CONTROL)
        self.assertEqual(result.control_intent, OCPControlIntent.STOP)
        self.assertEqual(
            self.clf.classify_control("turn off the lights", "en-us",
                                      ner_list={"song": ["lights"]}),
            OCPControlIntent.STOP)
        # without the entity context it stays a smart-home command
        self.assertEqual(self.clf.classify_full("turn off the lights", "en-us").domain,
                         OCPDomain.NOT_OCP)

    def test_media_keyword_beats_the_iot_word(self):
        # "turn off the music": the object is media, so this stays a control
        self.assertEqual(self.clf.classify_control("turn off the music", "en-us"),
                         OCPControlIntent.STOP)
        # an IoT word next to a media word is media, with damped confidence
        media_type, conf = self.clf.classify("play the lights soundtrack", "en-us")
        self.assertEqual(media_type, MediaType.MUSIC)
        self.assertGreater(conf, 0.0)

    def test_known_entity_rescues_a_title_spelled_like_an_iot_object(self):
        ner = {"song_name": ["lights"]}
        result = self.clf.classify_full("play lights", "en-us", ner_list=ner)
        self.assertEqual(result.domain, OCPDomain.OCP_PLAY)
        # without the entity context there is no media evidence at all
        self.assertEqual(
            self.clf.classify_full("turn off the lights", "en-us").domain,
            OCPDomain.NOT_OCP)

    def test_malformed_entity_context_is_coerced_not_fatal(self):
        for ner in (123, "lights", [("song_name", "lights")],
                    {"song_name": None}, {"song_name": [None, 1]},
                    {"song_name": 7}):
            with self.subTest(repr(ner)):
                # an IoT word puts the entity context on the hot path
                result = self.clf.classify_full("turn off the lights", "en-us",
                                                ner_list=ner)
                self.assertEqual(result.domain, OCPDomain.NOT_OCP)

    def test_a_bare_string_entity_counts_as_one_entity(self):
        # callers routinely pass a single entity unwrapped; iterating the
        # string character by character would never match it
        from ovos_media_classifier.keyword import _normalized_ner_list
        self.assertEqual(_normalized_ner_list({"song_name": "lights"}),
                         {"song_name": ["lights"]})
        self.assertEqual(_normalized_ner_list(123), {})
        self.assertEqual(_normalized_ner_list({"song_name": [None, 1, "x"]}),
                         {"song_name": ["x"]})
        self.assertTrue(self.clf._has_media_evidence(
            "play lights", "en-us", _normalized_ner_list({"song_name": "lights"})))
        self.assertEqual(
            self.clf.classify_full("play lights", "en-us",
                                   ner_list={"song_name": "lights"}).domain,
            OCPDomain.OCP_PLAY)

    def test_config_blacklist_extends_the_voc(self):
        # deployments name their smart-home entities freely
        clf = KeywordMediaClassifier(
            config={"media_classifier_blacklist": ["hallway sconce"]})
        self.assertTrue(clf._has_iot_object("turn off the hallway sconce", "en-us"))
        self.assertFalse(
            self.clf._has_iot_object("turn off the hallway sconce", "en-us"))
        self.assertEqual(clf.classify_domain("turn off the hallway sconce", "en-us"),
                         (OCPDomain.NOT_OCP, 0.0))
        # the collision damping follows the config entry too
        self.assertEqual(
            clf.classify("play the hallway sconce playlist", "en-us"),
            (MediaType.PLAYLIST, 0.3))
        self.assertEqual(
            self.clf.classify("play the hallway sconce playlist", "en-us"),
            (MediaType.PLAYLIST, 0.6))


class TestWeakControlPhrases(unittest.TestCase):
    """An ambiguous control phrase needs a bare utterance or a media object."""

    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_weak_phrase_alone_is_still_a_control(self):
        for utt in ("shut up", "be quiet please", "silence", "quiet please"):
            with self.subTest(utt):
                self.assertEqual(self.clf.classify_control(utt, "en-us"),
                                 OCPControlIntent.STOP)

    def test_weak_phrase_over_a_non_media_object_is_not_a_control(self):
        for utt in ("turn off the oven", "shut off the sprinklers"):
            with self.subTest(utt):
                self.assertIsNone(self.clf.classify_control(utt, "en-us"))

    def test_weak_phrase_over_an_unknown_object_is_not_a_control(self):
        # nothing on any list: the phrase governs a real object, so it is not a
        # media command even though the object is unrecognised
        for utt in ("turn off the giraffe", "turn off my homework",
                    "shut off the printer", "quiet the neighbor"):
            with self.subTest(utt):
                self.assertIsNone(self.clf.classify_control(utt, "en-us"))
                self.assertEqual(self.clf.classify_domain(utt, "en-us"),
                                 (OCPDomain.NOT_OCP, 0.0))

    def test_strong_controls_are_untouched(self):
        for utt, action in (("stop the music", OCPControlIntent.STOP),
                            ("stop", OCPControlIntent.STOP),
                            ("pause", OCPControlIntent.PAUSE),
                            ("next track", OCPControlIntent.NEXT)):
            with self.subTest(utt):
                self.assertEqual(self.clf.classify_control(utt, "en-us"), action)
                self.assertEqual(self.clf.classify_domain(utt, "en-us")[0],
                                 OCPDomain.OCP_CONTROL)


class TestPunctuationCannotBypassTheContentFilter(unittest.TestCase):
    """Matching folds punctuation, so an infix hyphen/dot is not a disguise."""

    def setUp(self):
        self.clf = KeywordMediaClassifier()
        self.filter = ContentFilter()

    def test_punctuated_adult_keyword_still_flags_adult(self):
        for utt in ("play porn video please", "play por-n videos", "play p.orn",
                    "play por\u2013n video", "play por\u200bn video"):
            with self.subTest(utt):
                genres = self.clf.classify_genres(utt, "en-us")
                self.assertIn("adult", genres)
                media_type, _ = self.clf.classify(utt, "en-us")
                self.assertTrue(self.filter.is_blocked(media_type, genres))

    def test_word_boundaries_survive_the_folding(self):
        # "sexfilm" is one word: it must not leak the MOVIE keyword "film"
        self.assertFalse(self.clf._match("sexfilm", "MovieKeyword", "de-de"))
        # a hyphenated title is not a keyword
        self.assertEqual(self.clf.classify("play blade-runner", "en-us")[0],
                         MediaType.GENERIC)
        self.assertEqual(self.clf.classify("play some music", "en-us"),
                         (MediaType.MUSIC, 0.6))


class TestLeafAndModalityAgree(unittest.TestCase):
    """The reported playback type never contradicts the resolved leaf."""

    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_video_game_is_interactive(self):
        result = self.clf.classify_full("play a video game", "en-us")
        self.assertEqual(result.media_type, MediaType.GAME)
        self.assertEqual(result.playback_type, PlaybackType.INTERACTIVE)
        self.assertEqual(self.clf.classify_playback_type("play a video game", "en-us"),
                         PlaybackType.INTERACTIVE)

    def test_unambiguous_modalities_are_unchanged(self):
        for utt, media_type, playback in (
                ("watch a movie", MediaType.MOVIE, PlaybackType.VIDEO),
                ("play some music", MediaType.MUSIC, PlaybackType.AUDIO)):
            with self.subTest(utt):
                result = self.clf.classify_full(utt, "en-us")
                self.assertEqual(result.media_type, media_type)
                self.assertEqual(result.playback_type, playback)


if __name__ == "__main__":
    unittest.main()
