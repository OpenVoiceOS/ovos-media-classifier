# ovos-media-classifier — benchmark results

Eval set: **30000 utterances** across 11 languages (ca-es=2803, da-dk=309, de-de=416, en-us=23765, es-es=503, eu-es=71, fr-fr=169, gl-es=608, it-it=288, nl-nl=234, pt-pt=834), **9 adult-genre rows**. Ground truth and utterances are derived from the bundled `.voc` keyword files (see `benchmarks/dataset.py`).

- Available backends: keyword, ahocorasick
- Unavailable backends: sklearn, padatious, model2vec, guided_onnx

## Backend summary

| backend | status | accuracy | macro-F1 | median ms | p95 ms | rows/s | CF recall | false-block |
|---|---|---|---|---|---|---|---|---|
| keyword | available | 0.293 | 0.448 | 0.0132 | 0.0197 | 78378 | 0.667 (6/9) | 0.000 |
| ahocorasick | available | 0.176 | 0.233 | 0.0016 | 0.0052 | 439834 | 0.222 (2/9) | 0.000 |
| sklearn | unavailable | – | – | – | – | – | – | – |
| padatious | unavailable | – | – | – | – | – | – | – |
| model2vec | unavailable | – | – | – | – | – | – | – |
| guided_onnx | unavailable | – | – | – | – | – | – | – |

## Unavailable backends

- **sklearn**: FileNotFoundError: no sklearn model (set MEDIA_CLF_SKLEARN_MODEL to a trained .joblib)
- **padatious**: FileNotFoundError: no padatious locale dir (set MEDIA_CLF_PADATIOUS_DIR); also needs the 'padatious' package
- **model2vec**: FileNotFoundError: no Model2Vec model (set MEDIA_CLF_M2V_MODEL); also needs 'model2vec'
- **guided_onnx**: FileNotFoundError: no GuidedEmbeddings ONNX model (set MEDIA_CLF_GUIDED_MODEL)

## Per-type metrics — `keyword`

| media_type | precision | recall | f1 | support |
|---|---|---|---|---|
| audio_drama | 0.917 | 0.548 | 0.686 | 365 |
| audiobook | 0.652 | 0.387 | 0.485 | 2187 |
| comic | 0.952 | 0.079 | 0.145 | 509 |
| episodic_series | 0.852 | 0.338 | 0.484 | 3221 |
| game | 0.573 | 0.590 | 0.582 | 166 |
| generic | 0.015 | 0.463 | 0.029 | 596 |
| movie | 0.749 | 0.658 | 0.700 | 4899 |
| music | 0.556 | 0.149 | 0.234 | 8455 |
| music_video | 0.997 | 0.369 | 0.538 | 1868 |
| not_media | 0.000 | 0.000 | 0.000 | 6158 |
| podcast | 0.888 | 0.648 | 0.749 | 676 |
| procedural_ambient | 1.000 | 0.828 | 0.906 | 29 |
| radio | 0.717 | 0.715 | 0.716 | 804 |
| short_film | 0.667 | 0.318 | 0.430 | 63 |
| tv | 0.019 | 1.000 | 0.037 | 4 |

## Per-type metrics — `ahocorasick`

| media_type | precision | recall | f1 | support |
|---|---|---|---|---|
| audio_drama | 0.923 | 0.526 | 0.670 | 365 |
| audiobook | 0.929 | 0.162 | 0.276 | 2187 |
| comic | 0.987 | 0.149 | 0.259 | 509 |
| episodic_series | 0.841 | 0.339 | 0.484 | 3221 |
| game | 0.000 | 0.000 | 0.000 | 166 |
| generic | 0.021 | 0.804 | 0.040 | 596 |
| movie | 0.715 | 0.386 | 0.502 | 4899 |
| music | 0.393 | 0.037 | 0.069 | 8455 |
| music_video | 0.995 | 0.199 | 0.331 | 1868 |
| not_media | 0.000 | 0.000 | 0.000 | 6158 |
| podcast | 0.878 | 0.640 | 0.741 | 676 |
| procedural_ambient | 0.000 | 0.000 | 0.000 | 29 |
| radio | 0.246 | 0.086 | 0.127 | 804 |
| short_film | 0.000 | 0.000 | 0.000 | 63 |
| tv | 0.000 | 0.000 | 0.000 | 4 |
