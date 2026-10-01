"""Optional ONNX trained-classifier backend — config selection + fallback.

The default ``ovos-media-classifier`` install ships only the lean ``.voc``
keyword classifier (core deps: ``ovos-utils`` + ``mediavocab``).  The ONNX
trained backend is **opt-in**:

    1. install the extra:  ``pip install ovos-media-classifier[onnx]``
    2. point the config at a model bundle directory via
       ``media_classifier_onnx_model``.

A bundle is a self-describing directory::

    <bundle>/
      ├── domain.onnx   # domain head  (ocp_play / ocp_control / not_ocp)
      ├── play.onnx     # play head    (fine-grained media-type label)
      └── meta.json     # {feature_names, domain_labels, play_labels, ...}

If the extra is not installed, or no/invalid bundle is configured, the factory
logs a warning and transparently falls back to the keyword classifier — so this
example runs out of the box on a lean install (no model present → keyword).

Run::

    python examples/onnx_backend.py
    python examples/onnx_backend.py /path/to/model_bundle
"""
import sys

from ovos_media_classifier import load_media_classifier
from ovos_media_classifier.keyword import KeywordMediaClassifier


def main(model_path: str = None) -> None:
    # When model_path is None / missing, from_path() fails and the factory
    # gracefully falls back to the keyword classifier.
    config = {}
    if model_path:
        config["media_classifier_onnx_model"] = model_path

    clf = load_media_classifier(config)

    backend = (
        "keyword (.voc fallback)"
        if isinstance(clf, KeywordMediaClassifier)
        else "ONNX trained backend"
    )
    print(f"selected backend: {backend}\n")

    for query in ("play some jazz", "play a documentary about whales",
                  "what time is it"):
        result = clf.classify_full(query, "en-us")
        print(f"{query!r:45} -> {result.as_dict()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
