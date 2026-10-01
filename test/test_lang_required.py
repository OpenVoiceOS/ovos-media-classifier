"""The public surface must never guess the language (T-6670).

`classify` has always required `lang` and `docs/stable-api.md` documents it that
way.  `to_signals` and `ContentFilter.check` defaulted it to "en-us", so the
same page promised two different contracts for one parameter.  These tests hold
the surface to the stricter one.
"""
import inspect
import unittest

from ovos_media_classifier import ContentFilter
from ovos_media_classifier.base import AbstractMediaClassifier


class TestLanguageIsRequired(unittest.TestCase):

    def test_classifier_methods_do_not_default_lang(self):
        for name in ("classify", "classify_genres", "to_signals"):
            method = getattr(AbstractMediaClassifier, name, None)
            if method is None:
                continue
            params = inspect.signature(method).parameters
            if "lang" not in params:
                continue
            with self.subTest(method=name):
                self.assertIs(
                    params["lang"].default, inspect.Parameter.empty,
                    f"{name} must not default lang")

    def test_content_filter_check_does_not_default_lang(self):
        lang = inspect.signature(ContentFilter.check).parameters["lang"]
        self.assertIs(lang.default, inspect.Parameter.empty,
                      "ContentFilter.check must not default lang")

    def test_omitting_lang_raises(self):
        with self.assertRaises(TypeError):
            ContentFilter.check(object(), object(), "play something")


if __name__ == "__main__":
    unittest.main()
