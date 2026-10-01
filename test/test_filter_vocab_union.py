"""The adult filter must not be weaker on a translated box (T-6805).

Six shipped locales ship a three-entry `AdultKeyword.voc` and do not block an
adult request that English blocks. The two locales that ship NO file are
protected, because an absent file falls back to English and a present one
suppresses that fallback. Adding a partial translation therefore makes a
deployment less safe than leaving it untranslated.

Blocking is asymmetric: a false block is a minor annoyance a user rephrases
around, a missed block is the failure the component exists to prevent. So the
per-language vocabulary is a union WITH the fallback, not a replacement for it.
"""
import os
import unittest

import ovos_media_classifier
from ovos_media_classifier import ContentFilter, load_media_classifier

LOCALE = os.path.join(os.path.dirname(ovos_media_classifier.__file__), "locale")
SHIPPED = sorted(d for d in os.listdir(LOCALE)
                 if os.path.isdir(os.path.join(LOCALE, d)))


class TestFilterIsNotWeakerWhenTranslated(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clf = load_media_classifier()
        cls.cf = ContentFilter()

    def test_every_shipped_locale_blocks_the_english_term(self):
        """A user on any box can type an English term; every locale must block it."""
        missed = [l for l in SHIPPED
                  if not self.cf.check(self.clf, "play some porn", l)[0]]
        self.assertEqual([], missed,
                         f"these locales do not block an English adult request: {missed}")

    def test_a_translated_locale_still_blocks_its_own_terms(self):
        """The union must not lose the per-language vocabulary it is added to."""
        for utt in ("speel porno", "speel wat pornografie"):
            with self.subTest(utterance=utt):
                blocked, _ = self.cf.check(self.clf, utt, "nl-nl")
                self.assertTrue(blocked, f"{utt!r} must still block on nl-nl")

    def test_a_locale_with_no_vocabulary_is_still_protected(self):
        """pl-pl and eu-es ship no AdultKeyword.voc and must stay protected."""
        for lang in ("pl-pl", "eu-es"):
            with self.subTest(lang=lang):
                self.assertTrue(
                    self.cf.check(self.clf, "play some porn", lang)[0])

    def test_a_clean_request_is_not_blocked_anywhere(self):
        """The control: widening the vocabulary must not block ordinary media."""
        for lang in ("en-us", "nl-nl", "de-de"):
            with self.subTest(lang=lang):
                for utt in ("play some music", "speel muziek"):
                    self.assertFalse(
                        self.cf.check(self.clf, utt, lang)[0],
                        f"{utt!r} must not be blocked on {lang}")


if __name__ == "__main__":
    unittest.main()
