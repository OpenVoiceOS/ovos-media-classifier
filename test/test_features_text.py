"""Unit tests for the numpy-only text featurizers (char-hash + word-vector pool).

These run with no torch / gensim — they are the runtime-side featurizers the
neural bundles reproduce, so they must be deterministic and self-describing.
"""
import unittest

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False

from ovos_media_classifier.features_text import TextHashSpec, hash_matrix, hash_vector
from ovos_media_classifier.features_wordvec import (
    WordVecPooler,
    WordVecSpec,
    tokenize,
)


@unittest.skipUnless(_HAVE_NUMPY, "numpy not installed")
class TestTextHash(unittest.TestCase):
    def test_dim_and_l2_norm(self):
        spec = TextHashSpec(dim=256)
        v = hash_vector("play some jazz", spec)
        self.assertEqual(v.shape, (256,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_empty_is_zero(self):
        spec = TextHashSpec(dim=128)
        v = hash_vector("   ", spec)
        self.assertEqual(float(np.abs(v).sum()), 0.0)

    def test_deterministic_reproducible(self):
        # the same text → byte-identical vector (no seed/state)
        spec = TextHashSpec(dim=512)
        a = hash_vector("horror movie", spec)
        b = hash_vector("horror movie", spec)
        np.testing.assert_array_equal(a, b)

    def test_different_text_different_vector(self):
        spec = TextHashSpec(dim=512)
        a = hash_vector("jazz", spec)
        b = hash_vector("metal", spec)
        self.assertFalse(np.allclose(a, b))

    def test_spec_roundtrips_through_meta(self):
        spec = TextHashSpec(dim=2048, ngram_min=2, ngram_max=4, analyzer="char")
        meta = spec.to_meta()
        back = TextHashSpec.from_meta(meta)
        self.assertEqual(back, spec)
        self.assertEqual(len(spec.feature_names()), 2048)
        self.assertTrue(spec.feature_names()[0].startswith("txt_"))

    def test_from_meta_none(self):
        self.assertIsNone(TextHashSpec.from_meta(None))

    def test_word_analyzer(self):
        spec = TextHashSpec(dim=256, analyzer="word", ngram_min=1, ngram_max=2)
        v = hash_vector("play some jazz", spec)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_matrix_stacks_rows(self):
        spec = TextHashSpec(dim=64)
        m = hash_matrix(["jazz", "rock", ""], spec)
        self.assertEqual(m.shape, (3, 64))
        self.assertEqual(float(np.abs(m[2]).sum()), 0.0)  # empty row stays zero


@unittest.skipUnless(_HAVE_NUMPY, "numpy not installed")
class TestWordVecPool(unittest.TestCase):
    def _pooler(self, normalize=True, pooling="mean"):
        spec = WordVecSpec(dim=4, normalize=normalize, pooling=pooling)
        vectors = np.array([
            [0, 0, 0, 0],     # row 0 — zero OOV row
            [1, 0, 0, 0],     # jazz
            [0, 1, 0, 0],     # rock
            [0, 0, 1, 0],     # play
        ], dtype="float32")
        vocab = {"jazz": 1, "rock": 2, "play": 3}
        return WordVecPooler(vectors, vocab, spec)

    def test_mean_pool_in_vocab(self):
        p = self._pooler(normalize=False)
        v = p.pool("play jazz")            # rows 3 + 1 → mean
        np.testing.assert_allclose(v, [0.5, 0.0, 0.5, 0.0], atol=1e-6)

    def test_oov_tokens_skipped(self):
        p = self._pooler(normalize=False)
        v = p.pool("play unknownword")     # only "play" is in vocab
        np.testing.assert_allclose(v, [0.0, 0.0, 1.0, 0.0], atol=1e-6)

    def test_all_oov_is_zero(self):
        p = self._pooler()
        v = p.pool("nothing here matches")
        self.assertEqual(float(np.abs(v).sum()), 0.0)

    def test_normalize(self):
        p = self._pooler(normalize=True)
        v = p.pool("play jazz")
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_tokenize(self):
        self.assertEqual(tokenize("Play, some JAZZ!"), ["play", "some", "jazz"])

    def test_spec_meta_roundtrip(self):
        spec = WordVecSpec(dim=100, pooling="sum", normalize=False)
        self.assertEqual(WordVecSpec.from_meta(spec.to_meta()), spec)
        self.assertEqual(len(spec.feature_names()), 100)


if __name__ == "__main__":
    unittest.main()
