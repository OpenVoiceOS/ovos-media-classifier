# ovos-media-classifier — benchmark results

Eval set: **875 utterances** across 3 languages (de-de=277, en-us=360, pt-pt=238), **64 adult-genre rows**. Ground truth and utterances are derived from the bundled `.voc` keyword files (see `benchmarks/dataset.py`).

- Available backends: keyword, ahocorasick
- Unavailable backends: sklearn, padatious, model2vec, guided_onnx

## Backend summary

| backend | status | accuracy | macro-F1 | median ms | p95 ms | rows/s | CF recall | false-block |
|---|---|---|---|---|---|---|---|---|
| keyword | available | 0.983 | 0.981 | 0.0247 | 0.0363 | 7889 | 1.000 (64/64) | 0.000 |
| ahocorasick | available | 0.495 | 0.628 | 0.0029 | 0.0047 | 410471 | 0.469 (30/64) | 0.000 |
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
| audio_drama | 1.000 | 1.000 | 1.000 | 24 |
| audiobook | 0.932 | 0.976 | 0.954 | 84 |
| comic | 0.938 | 0.833 | 0.882 | 36 |
| episodic_series | 1.000 | 1.000 | 1.000 | 148 |
| game | 1.000 | 1.000 | 1.000 | 18 |
| movie | 1.000 | 0.969 | 0.984 | 222 |
| music | 0.918 | 1.000 | 0.957 | 78 |
| music_video | 1.000 | 1.000 | 1.000 | 56 |
| podcast | 1.000 | 1.000 | 1.000 | 21 |
| procedural_ambient | 1.000 | 1.000 | 1.000 | 27 |
| radio | 1.000 | 1.000 | 1.000 | 45 |
| tv | 1.000 | 1.000 | 1.000 | 116 |

## Per-type metrics — `ahocorasick`

| media_type | precision | recall | f1 | support |
|---|---|---|---|---|
| audio_drama | 1.000 | 0.750 | 0.857 | 24 |
| audiobook | 1.000 | 0.286 | 0.444 | 84 |
| comic | 1.000 | 0.667 | 0.800 | 36 |
| episodic_series | 1.000 | 0.540 | 0.702 | 148 |
| game | 0.000 | 0.000 | 0.000 | 18 |
| movie | 0.967 | 0.531 | 0.686 | 222 |
| music | 0.828 | 0.615 | 0.706 | 78 |
| music_video | 1.000 | 0.536 | 0.698 | 56 |
| podcast | 1.000 | 1.000 | 1.000 | 21 |
| procedural_ambient | 1.000 | 0.333 | 0.500 | 27 |
| radio | 1.000 | 0.467 | 0.636 | 45 |
| tv | 1.000 | 0.345 | 0.513 | 116 |
