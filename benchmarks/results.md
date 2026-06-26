# ovos-media-classifier — benchmark results

Eval set: **1571 utterances** across 3 languages (de-de=292, en-us=1026, pt-pt=253), **124 adult-genre rows**. Ground truth and utterances are derived from the bundled `.voc` keyword files (see `benchmarks/dataset.py`).

- Available backends: keyword, ahocorasick
- Unavailable backends: sklearn, padatious, model2vec, guided_onnx

## Backend summary

| backend | status | accuracy | macro-F1 | median ms | p95 ms | rows/s | CF recall | false-block |
|---|---|---|---|---|---|---|---|---|
| keyword | available | 0.992 | 0.990 | 0.0440 | 0.0925 | 7044 | 1.000 (124/124) | 0.000 |
| ahocorasick | available | 0.692 | 0.783 | 0.0035 | 0.0063 | 283081 | 0.629 (78/124) | 0.000 |
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
| audio_drama | 1.000 | 1.000 | 1.000 | 54 |
| audiobook | 0.918 | 1.000 | 0.957 | 90 |
| comic | 1.000 | 0.865 | 0.927 | 96 |
| episodic_series | 1.000 | 1.000 | 1.000 | 238 |
| game | 1.000 | 1.000 | 1.000 | 48 |
| movie | 1.000 | 1.000 | 1.000 | 432 |
| music | 1.000 | 1.000 | 1.000 | 192 |
| music_video | 1.000 | 1.000 | 1.000 | 74 |
| podcast | 1.000 | 1.000 | 1.000 | 33 |
| procedural_ambient | 1.000 | 1.000 | 1.000 | 27 |
| radio | 1.000 | 1.000 | 1.000 | 135 |
| tv | 1.000 | 1.000 | 1.000 | 152 |

## Per-type metrics — `ahocorasick`

| media_type | precision | recall | f1 | support |
|---|---|---|---|---|
| audio_drama | 1.000 | 0.778 | 0.875 | 54 |
| audiobook | 1.000 | 0.467 | 0.636 | 90 |
| comic | 1.000 | 1.000 | 1.000 | 96 |
| episodic_series | 1.000 | 0.714 | 0.833 | 238 |
| game | 1.000 | 0.625 | 0.769 | 48 |
| movie | 0.987 | 0.676 | 0.802 | 432 |
| music | 0.938 | 0.781 | 0.852 | 192 |
| music_video | 1.000 | 0.649 | 0.787 | 74 |
| podcast | 1.000 | 0.636 | 0.778 | 33 |
| procedural_ambient | 1.000 | 0.333 | 0.500 | 27 |
| radio | 1.000 | 0.822 | 0.902 | 135 |
| tv | 1.000 | 0.500 | 0.667 | 152 |
