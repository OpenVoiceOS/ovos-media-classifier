"""A baked-vectorizer TF-IDF text-pipeline bundle loads + classifies from raw text.

These exercise the ``input_kind == "text"`` bundle path with *real* onnxruntime:
a ``sklearn`` ``Pipeline(TfidfVectorizer → classifier)`` is exported whole with
``skl2onnx`` so the vectorizer is **in the ONNX graph** — the bundle takes the
raw utterance string and needs **no** python featurization at runtime.  We build a
tiny one, load it via :meth:`OnnxMediaClassifier.from_path`, and assert it
classifies sanely.

Skips when scikit-learn / skl2onnx / onnxruntime are absent (the ``[train]`` /
``[onnx]`` extras), like the other backend tests.
"""
import json
import os
import shutil
import tempfile
import unittest

try:
    import numpy as np
    import onnxruntime  # noqa: F401
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    _HAVE_SKL = True
except Exception:  # pragma: no cover - env-dependent
    _HAVE_SKL = False

from ovos_media_classifier.intents import MediaType, OCPDomain


def _pipeline_onnx(texts, labels, analyzer="word", ngram=(1, 2)):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=1)),
        ("clf", LogisticRegression(max_iter=300)),
    ])
    pipe.fit(texts, labels)
    onx = convert_sklearn(
        pipe, initial_types=[("input", StringTensorType([None, 1]))],
        options={id(pipe): {"zipmap": False}}, target_opset=15)
    return pipe, onx.SerializeToString()


def _build_text_bundle(out_dir, analyzer="word"):
    """A 2-head (domain + media_type) baked-TF-IDF string-input bundle."""
    music = ["play some jazz", "play rock music", "put on a song", "play jazz"]
    movie = ["watch a horror movie", "play the film", "show me a movie"]
    book = ["read the book", "read me a novel", "open the book"]
    none_ = ["", "what time is it", "turn off the lights"]

    mt_texts = music + movie + book
    mt_labels = (["music"] * len(music) + ["movie"] * len(movie)
                 + ["book"] * len(book))
    d_texts = mt_texts + none_
    d_labels = (["ocp_play"] * len(mt_texts) + ["not_ocp"] * len(none_))

    os.makedirs(out_dir, exist_ok=True)
    d_pipe, d_onnx = _pipeline_onnx(d_texts, d_labels, analyzer)
    mt_pipe, mt_onnx = _pipeline_onnx(mt_texts, mt_labels, analyzer)
    with open(os.path.join(out_dir, "domain.onnx"), "wb") as fh:
        fh.write(d_onnx)
    with open(os.path.join(out_dir, "media_type.onnx"), "wb") as fh:
        fh.write(mt_onnx)
    shutil.copyfile(os.path.join(out_dir, "media_type.onnx"),
                    os.path.join(out_dir, "play.onnx"))

    def _imap(classes):
        return {str(i): str(c) for i, c in enumerate(classes)}

    meta = {
        "input_kind": "text",
        "feature_names": [],
        "input_name": "input",
        "domain_labels": _imap(d_pipe.classes_),
        "play_labels": _imap(mt_pipe.classes_),
        "heads": {
            "domain": {"onnx": "domain.onnx", "kind": "single",
                       "labels": _imap(d_pipe.classes_)},
            "media_type": {"onnx": "media_type.onnx", "kind": "single",
                           "labels": _imap(mt_pipe.classes_)},
        },
        "trained_by": "test",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    return out_dir


@unittest.skipUnless(_HAVE_SKL, "scikit-learn/skl2onnx/onnxruntime not installed")
class TestTextPipelineBundle(unittest.TestCase):
    def test_input_kind_text_loads(self):
        from ovos_media_classifier.onnx import OnnxMediaClassifier

        with tempfile.TemporaryDirectory() as d:
            _build_text_bundle(d)
            clf = OnnxMediaClassifier.from_path(d)

        self.assertEqual(clf._input_kind, "text")
        # the runtime model input for a text bundle is the raw (1,1) string array
        row = clf._row_for("play some jazz", "en-us")
        self.assertEqual(row.shape, (1, 1))
        self.assertEqual(row[0, 0], "play some jazz")

    def test_classifies_from_raw_text(self):
        """No python featurization — the baked vectorizer reads the string."""
        from ovos_media_classifier.onnx import OnnxMediaClassifier

        with tempfile.TemporaryDirectory() as d:
            _build_text_bundle(d)
            clf = OnnxMediaClassifier.from_path(d)

            mt, conf = clf.classify("play some jazz music", "en-us")
            self.assertEqual(mt, MediaType.MUSIC)
            self.assertGreater(conf, 0.0)

            mt2, _ = clf.classify("watch a horror movie", "en-us")
            self.assertEqual(mt2, MediaType.MOVIE)

            domain, _ = clf.classify_domain("play some jazz", "en-us")
            self.assertEqual(domain, OCPDomain.OCP_PLAY)

    def test_char_analyzer_bundle(self):
        """analyzer='char' also exports + loads (only char_wb is unconvertible)."""
        from ovos_media_classifier.onnx import OnnxMediaClassifier

        with tempfile.TemporaryDirectory() as d:
            _build_text_bundle(d, analyzer="char")
            clf = OnnxMediaClassifier.from_path(d)
            mt, _ = clf.classify("read the book", "en-us")
            self.assertIsInstance(mt, MediaType)


if __name__ == "__main__":
    unittest.main()
