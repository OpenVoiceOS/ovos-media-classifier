"""Regression tests for the routing-harm fixes (content policy + precedence).

These exercise the *bundled* locale ``.voc`` files through a real
:class:`KeywordMediaClassifier` (no mocked matcher), so they pin the shipping
default behaviour the harm-weighted routing eval surfaced:

  - the adult / content-policy lexicon flags every common adult term (en + de /
    es / pt multilingual) so adult never leaks to a clean provider;
  - clean media requests are NOT over-flagged as adult;
  - the German / Danish play-verb ("spiel" / "spil") no longer collides with the
    GAME leaf when it is merely the leading transport verb;
  - a Portuguese "read the book" request routes to BOOK, not AUDIOBOOK.
"""
import unittest

from ovos_media_classifier import KeywordMediaClassifier, MediaType


class TestAdultContentPolicy(unittest.TestCase):
    """Every common adult term must surface the ``adult`` genre (block signal)."""

    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def _adult(self, utt, lang="en-us"):
        genres = self.clf.classify_content_form_genres(utt, lang)
        return "adult" in {str(g).lower() for g in genres}

    def test_en_adult_terms_blocked(self):
        terms = [
            "play some porn", "play a porno", "play some pornos",
            "play a porno film", "play some pornhub", "i want to watch xxx",
            "show me x-rated stuff", "something nsfw", "show me some nudes",
            "show me a nude scene", "show me a striptease",
            "put on a strip tease", "put on some erotica", "play erotic videos",
            "play hardcore stuff", "find a camgirl", "play onlyfans leaks",
            "put on a sex tape", "play a sextape",
        ]
        for t in terms:
            with self.subTest(term=t):
                self.assertTrue(self._adult(t), f"adult NOT flagged: {t!r}")

    def test_en_hentai_terms_blocked(self):
        for t in ("put on some hentai", "show me ecchi", "play rule34",
                  "play some rule 34"):
            with self.subTest(term=t):
                genres = {str(g).lower()
                          for g in self.clf.classify_content_form_genres(t, "en-us")}
                self.assertIn("adult", genres, f"adult NOT flagged: {t!r}")
                self.assertIn("anime", genres, f"anime NOT flagged: {t!r}")

    def test_de_adult_terms_blocked(self):
        for t in ("spiel irgendwas pornos", "mach einen sexfilm an",
                  "spiel einen pornofilm", "zeig mir nacktbilder",
                  "spiel etwas erotik"):
            with self.subTest(term=t):
                self.assertTrue(self._adult(t, "de-de"), f"adult NOT flagged: {t!r}")

    def test_es_adult_terms_blocked(self):
        for t in ("pon algo de porno", "quiero ver porno",
                  "pon una pelicula porno", "muéstrame desnudos"):
            with self.subTest(term=t):
                self.assertTrue(self._adult(t, "es-es"), f"adult NOT flagged: {t!r}")

    def test_pt_adult_terms_blocked(self):
        for t in ("põe lá uma pornografia", "quero ver pornô",
                  "põe um filme porno"):
            with self.subTest(term=t):
                self.assertTrue(self._adult(t, "pt-pt"), f"adult NOT flagged: {t!r}")

    def test_clean_content_not_overflagged(self):
        """Clean media requests must NOT be flagged adult (no over-block)."""
        for t, lang in [
            ("play some jazz", "en-us"),
            ("watch the matrix", "en-us"),
            ("play the harry potter audiobook", "en-us"),
            ("spiel jazzmusik", "de-de"),
            ("pon los simpson", "es-es"),
        ]:
            with self.subTest(term=t):
                self.assertFalse(self._adult(t, lang), f"clean over-flagged: {t!r}")


class TestPlayVerbGameCollision(unittest.TestCase):
    """A leading play verb spelled like the GAME noun must not route to GAME."""

    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_de_play_verb_not_game(self):
        for utt in ("spiel jazzmusik", "spiel die simpsons",
                    "spiele die simpsons"):
            with self.subTest(utt=utt):
                mt, _ = self.clf.classify(utt, "de-de")
                self.assertNotEqual(mt, MediaType.GAME,
                                    f"play verb mis-routed to GAME: {utt!r}")

    def test_da_play_verb_not_game(self):
        mt, _ = self.clf.classify("spil noget musik", "da-dk")
        self.assertNotEqual(mt, MediaType.GAME)

    def test_real_game_requests_still_route_to_game(self):
        for utt, lang in [("play a game", "en-us"), ("launch the game", "en-us"),
                          ("spiel ein spiel", "de-de"),
                          ("spiele ein videospiel", "de-de"),
                          ("spil et spil", "da-dk")]:
            with self.subTest(utt=utt):
                mt, _ = self.clf.classify(utt, lang)
                self.assertEqual(mt, MediaType.GAME,
                                 f"real game request lost: {utt!r}")


class TestBookAudiobookPrecedence(unittest.TestCase):
    """A read-the-book request must route to BOOK, not AUDIOBOOK (pt fix)."""

    def setUp(self):
        self.clf = KeywordMediaClassifier()

    def test_pt_read_book_is_book(self):
        mt, _ = self.clf.classify("lê-me o livro o principezinho", "pt-pt")
        self.assertEqual(mt, MediaType.BOOK)

    def test_pt_audiobook_keyword_still_audiobook(self):
        mt, _ = self.clf.classify("põe o audiolivro o principezinho", "pt-pt")
        self.assertEqual(mt, MediaType.AUDIOBOOK)


if __name__ == "__main__":
    unittest.main()
