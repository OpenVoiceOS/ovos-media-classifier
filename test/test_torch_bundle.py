"""A torch-exported ONNX bundle loads via ``OnnxMediaClassifier`` and classifies.

These tests exercise the **neural bundle** end-to-end with *real* onnxruntime +
numpy (unlike ``test_onnx.py``, which mocks onnxruntime): a tiny shared-trunk net
is built in torch, exported to the self-describing per-axis bundle format, and
loaded by :meth:`OnnxMediaClassifier.from_path`.  The bundle declares a char-hash
text featurizer (and a word-vector block in one case), so loading also exercises
the runtime numpy featurization (``txt_*`` / ``wv_*`` blocks).

The whole module **skips** when torch / onnxruntime / onnx are not installed (the
neural path lives in the ``[train]`` extra; runtime is ``[onnx]``), mirroring the
existing "skip if backend absent" convention.
"""
import json
import os
import tempfile
import unittest

try:  # the neural export path needs torch + onnx; the runtime needs onnxruntime
    import numpy as np
    import onnxruntime  # noqa: F401
    import torch  # noqa: F401
    import torch.nn as nn

    _HAVE_TORCH = True
except Exception:  # pragma: no cover - env-dependent
    _HAVE_TORCH = False

from ovos_media_classifier.intents import MediaType, OCPDomain


def _build_tiny_bundle(out_dir, with_wordvec=False):
    """Export a 2-axis (domain + media_type) torch net to a bundle dir.

    The trunk is deliberately tiny; correctness — not accuracy — is the point.
    The net is biased so a non-empty feature row argmaxes to ``ocp_play`` /
    ``music`` (idx chosen below), letting the test assert a sane classification.
    """
    from ovos_media_classifier.features import _KEYWORD_VOCABS
    from ovos_media_classifier.features_text import TextHashSpec

    text_spec = TextHashSpec(dim=64, ngram_min=3, ngram_max=4)
    cat_cols = [c for _v, c in _KEYWORD_VOCABS]
    feature_names = cat_cols + text_spec.feature_names()
    n_features = len(feature_names)

    wv_spec = None
    if with_wordvec:
        from ovos_media_classifier.features_wordvec import WordVecSpec
        wv_spec = WordVecSpec(dim=8)
        feature_names = feature_names + wv_spec.feature_names()
        n_features = len(feature_names)

    torch.manual_seed(0)
    trunk = nn.Sequential(nn.Linear(n_features, 16), nn.ReLU())
    domain_head = nn.Linear(16, 2)   # idx 0 not_ocp, idx 1 ocp_play
    mt_head = nn.Linear(16, 3)       # idx 0 music, 1 movie, 2 generic

    class _Export(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.trunk, self.head = trunk, head

        def forward(self, x):
            return torch.softmax(self.head(self.trunk(x)), dim=-1)

    os.makedirs(out_dir, exist_ok=True)
    dummy = torch.zeros(1, n_features)
    for name, head in (("domain", domain_head), ("media_type", mt_head)):
        torch.onnx.export(
            _Export(head), dummy, os.path.join(out_dir, f"{name}.onnx"),
            input_names=["input"], output_names=["probabilities"],
            dynamic_axes={"input": {0: "batch"}}, opset_version=15, dynamo=False)
    # play.onnx alias of media_type (back-compat contract)
    import shutil
    shutil.copyfile(os.path.join(out_dir, "media_type.onnx"),
                    os.path.join(out_dir, "play.onnx"))

    meta = {
        "feature_names": feature_names,
        "input_name": "input",
        "domain_labels": {"0": "not_ocp", "1": "ocp_play"},
        "play_labels": {"0": "music", "1": "movie", "2": "generic"},
        "heads": {
            "domain": {"onnx": "domain.onnx", "kind": "single",
                       "labels": {"0": "not_ocp", "1": "ocp_play"}},
            "media_type": {"onnx": "media_type.onnx", "kind": "single",
                           "labels": {"0": "music", "1": "movie", "2": "generic"}},
        },
        "text_hash": text_spec.to_meta(),
        "trained_by": "test",
    }
    if with_wordvec:
        # a 3-token vocab + (4, 8) matrix (row 0 = zero OOV row)
        vocab = {"jazz": 1, "play": 2, "music": 3}
        mat = np.zeros((4, 8), dtype="float32")
        mat[1:] = np.eye(3, 8, dtype="float32")
        np.save(os.path.join(out_dir, wv_spec.vectors_file), mat)
        with open(os.path.join(out_dir, wv_spec.vocab_file), "w") as fh:
            json.dump(vocab, fh)
        meta["wordvec"] = wv_spec.to_meta()

    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    return out_dir


@unittest.skipUnless(_HAVE_TORCH, "torch/onnxruntime not installed ([train] extra)")
class TestTorchBundle(unittest.TestCase):
    def test_torch_bundle_loads_and_carries_text_spec(self):
        from ovos_media_classifier.features_text import TextHashSpec
        from ovos_media_classifier.onnx import OnnxMediaClassifier

        with tempfile.TemporaryDirectory() as d:
            _build_tiny_bundle(d)
            clf = OnnxMediaClassifier.from_path(d)

        # the bundle's char-hash spec is read back + the txt_* block is wired
        self.assertIsInstance(clf._text_spec, TextHashSpec)
        self.assertEqual(clf._text_spec.dim, 64)
        self.assertEqual(len(clf._txt_idx), 64)
        # categorical columns are the non-text remainder
        self.assertTrue(all(not n.startswith("txt_") for n in clf._cat_names))

    def test_runtime_featurization_builds_text_block(self):
        """The runtime row has the char-hash block filled from the raw text."""
        from ovos_media_classifier.onnx import OnnxMediaClassifier

        with tempfile.TemporaryDirectory() as d:
            _build_tiny_bundle(d)
            clf = OnnxMediaClassifier.from_path(d)
            row = clf._row_for("play some relaxing jazz music", "en-us")

        self.assertEqual(row.shape, (1, len(clf._feature_names)))
        # the text block is non-zero (the hashed n-grams fired) and L2-ish scaled
        txt = row[0, clf._txt_idx[0]:clf._txt_idx[0] + len(clf._txt_idx)]
        self.assertGreater(float(np.abs(txt).sum()), 0.0)

    def test_torch_bundle_classifies_sanely(self):
        """Loads with real onnxruntime and returns a valid MediaType + domain."""
        from ovos_media_classifier.onnx import OnnxMediaClassifier

        with tempfile.TemporaryDirectory() as d:
            _build_tiny_bundle(d)
            clf = OnnxMediaClassifier.from_path(d)

            mt, conf = clf.classify("play some jazz", "en-us")
            self.assertIsInstance(mt, MediaType)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

            domain, _ = clf.classify_domain("play some jazz", "en-us")
            self.assertIsInstance(domain, OCPDomain)

            full = clf.classify_full("play some jazz", "en-us")
            self.assertIsInstance(full.media_type, MediaType)

    def test_torch_bundle_with_wordvec_block(self):
        """A bundle declaring a word-vector block loads + pools in numpy."""
        from ovos_media_classifier.features_wordvec import WordVecSpec
        from ovos_media_classifier.onnx import OnnxMediaClassifier

        with tempfile.TemporaryDirectory() as d:
            _build_tiny_bundle(d, with_wordvec=True)
            clf = OnnxMediaClassifier.from_path(d)

            self.assertIsInstance(clf._wordvec_spec, WordVecSpec)
            self.assertIsNotNone(clf._wordvec_pooler)
            self.assertEqual(len(clf._wv_idx), 8)
            row = clf._row_for("play jazz music", "en-us")
            wv = row[0, clf._wv_idx[0]:clf._wv_idx[0] + len(clf._wv_idx)]
            # jazz/play/music are all in vocab → the pooled block is non-zero
            self.assertGreater(float(np.abs(wv).sum()), 0.0)
            # and it still classifies
            mt, _ = clf.classify("play jazz music", "en-us")
            self.assertIsInstance(mt, MediaType)


if __name__ == "__main__":
    unittest.main()
