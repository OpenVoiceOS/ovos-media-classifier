# ovos-media-classifier — harm-weighted routing eval

Out-of-distribution, hand-curated eval: **222 cases** across 4 languages (de-de=15, en-us=176, es-es=16, pt-pt=15); **23 adult** cases, **171 abstain-ok** cases.

Composition by category: content_policy=21, control=28, conversational=36, gate_negative=41, keywordless=21, media=59, noise=3, playback_divergent=13.

**The metric is mis-route rate (confident-wrong), and GENERIC/abstain is scored as SAFE.** A confident-wrong route prunes the correct provider (harm); an abstain still lets every provider search (harmless).

## Headline

**mis-route** is the harm (GENERIC=safe); **resolved** is the open-vocab win (abstain_ok cases turned into the CORRECT confident route). A clean win over a cheaper layer is mis-route NOT WORSE *and* resolved HIGHER.

| backend | mis-route | resolved | adult-leak | false-hijack | false-miss | control recall | latency med/p95 |
|---|---|---|---|---|---|---|---|
| keyword | **0.050** (7/139) | **0.318** (28/88) | **0.000** (0/23) | 0.208 (10/48) | 0.103 (18/174) | 0.516 (16/31) | – |
| onnx:models/context | **0.101** (14/139) | **0.375** (33/88) | **0.043** (1/23) | 0.167 (8/48) | 0.397 (69/174) | 0.000 (0/31) | – |
| onnx:models/context_ner | **0.166** (23/139) | **0.250** (22/88) | **0.000** (0/23) | 0.146 (7/48) | 0.466 (81/174) | 0.000 (0/31) | – |
| onnx:models_text/tfidf_char35 | **0.374** (52/139) | **0.511** (45/88) | **0.609** (14/23) | 0.854 (41/48) | 0.081 (14/174) | 0.000 (0/31) | – |
| onnx:models_text/tfidf_word12 | **0.389** (54/139) | **0.500** (44/88) | **0.609** (14/23) | 0.667 (32/48) | 0.132 (23/174) | 0.000 (0/31) | – |
| onnx:models_text/tfidf_word13 | **0.396** (55/139) | **0.500** (44/88) | **0.652** (15/23) | 0.688 (33/48) | 0.115 (20/174) | 0.000 (0/31) | – |
| onnx:models_torch/cat | **0.345** (48/139) | **0.489** (43/88) | **0.000** (0/23) | 1.000 (48/48) | 0.000 (0/174) | 0.000 (0/31) | – |
| onnx:models_torch/cat_all | **0.360** (50/139) | **0.500** (44/88) | **0.087** (2/23) | 1.000 (48/48) | 0.000 (0/174) | 0.000 (0/31) | – |
| onnx:models_torch/cat_all_deep | **0.338** (47/139) | **0.511** (45/88) | **0.000** (0/23) | 1.000 (48/48) | 0.000 (0/174) | 0.000 (0/31) | – |
| onnx:models_torch/cat_all_wide | **0.352** (49/139) | **0.489** (43/88) | **0.087** (2/23) | 1.000 (48/48) | 0.000 (0/174) | 0.000 (0/31) | – |
| onnx:models_torch/cat_text | **0.281** (39/139) | **0.591** (52/88) | **0.043** (1/23) | 1.000 (48/48) | 0.000 (0/174) | 0.000 (0/31) | – |
| onnx:models_torch/cat_wordvec | **0.295** (41/139) | **0.580** (51/88) | **0.043** (1/23) | 1.000 (48/48) | 0.000 (0/174) | 0.000 (0/31) | – |
| onnx:models_torch/cat_wordvec_cbow | **0.374** (52/139) | **0.489** (43/88) | **0.087** (2/23) | 1.000 (48/48) | 0.000 (0/174) | 0.000 (0/31) | – |
| embedding-router:embedding_router | **0.000** (0/139) | **0.000** (0/88) | **1.000** (23/23) | 0.000 (0/48) | 1.000 (174/174) | 0.000 (0/31) | – |
| hybrid:embedding_router | **0.050** (7/139) | **0.318** (28/88) | **0.000** (0/23) | 0.208 (10/48) | 0.103 (18/174) | 0.516 (16/31) | – |
| hybrid+gazetteer:embedding_router | **0.050** (7/139) | **0.534** (47/88) | **0.000** (0/23) | 0.208 (10/48) | 0.103 (18/174) | 0.516 (16/31) | – |
| hybrid+inject:embedding_router | **0.029** (4/139) | **0.625** (55/88) | **0.000** (0/23) | 0.208 (10/48) | 0.103 (18/174) | 0.516 (16/31) | – |

## Slice: conversational (36 cases)

Mis-route / resolved / adult-leak scored over ONLY this category (the realism slice the conversational/ASR training targets).

| backend | mis-route | resolved | adult-leak |
|---|---|---|---|
| keyword | **0.120** (3/25) | **0.400** (6/15) | **0.000** (0/2) |
| onnx:models/context | **0.240** (6/25) | **0.467** (7/15) | **0.000** (0/2) |
| onnx:models/context_ner | **0.320** (8/25) | **0.333** (5/15) | **0.000** (0/2) |
| onnx:models_text/tfidf_char35 | **0.280** (7/25) | **0.533** (8/15) | **0.500** (1/2) |
| onnx:models_text/tfidf_word12 | **0.200** (5/25) | **0.667** (10/15) | **0.500** (1/2) |
| onnx:models_text/tfidf_word13 | **0.200** (5/25) | **0.667** (10/15) | **1.000** (2/2) |
| onnx:models_torch/cat | **0.280** (7/25) | **0.600** (9/15) | **0.000** (0/2) |
| onnx:models_torch/cat_all | **0.200** (5/25) | **0.667** (10/15) | **0.000** (0/2) |
| onnx:models_torch/cat_all_deep | **0.200** (5/25) | **0.667** (10/15) | **0.000** (0/2) |
| onnx:models_torch/cat_all_wide | **0.240** (6/25) | **0.600** (9/15) | **0.000** (0/2) |
| onnx:models_torch/cat_text | **0.200** (5/25) | **0.667** (10/15) | **0.000** (0/2) |
| onnx:models_torch/cat_wordvec | **0.200** (5/25) | **0.667** (10/15) | **0.000** (0/2) |
| onnx:models_torch/cat_wordvec_cbow | **0.200** (5/25) | **0.667** (10/15) | **0.000** (0/2) |
| embedding-router:embedding_router | **0.000** (0/25) | **0.000** (0/15) | **1.000** (2/2) |
| hybrid:embedding_router | **0.120** (3/25) | **0.400** (6/15) | **0.000** (0/2) |
| hybrid+gazetteer:embedding_router | **0.120** (3/25) | **0.600** (9/15) | **0.000** (0/2) |
| hybrid+inject:embedding_router | **0.040** (1/25) | **0.667** (10/15) | **0.000** (0/2) |

## Open-vocab cases each layer closes

Each row is a play-case the cheaper layer abstained on or mis-routed that the richer layer routes correctly.

### hybrid -> hybrid+gazetteer  (19 fixed)

| utterance | expected | was | now |
|---|---|---|---|
| put on the matrix | movie | generic | **movie** |
| play some miles davis | music | generic | **music** |
| stick on bluey for the kids | episodic_series | generic | **episodic_series** |
| throw on breaking bad | episodic_series | generic | **episodic_series** |
| can you play the office | episodic_series | generic | **episodic_series** |
| i feel like listening to the beatles | music | generic | **music** |
| play the joe rogan experience | podcast | generic | **podcast** |
| play frozen for my daughter | movie | generic | **movie** |
| play seinfeld | episodic_series | generic | **episodic_series** |
| play interstellar | movie | generic | **movie** |
| play radiohead | music | generic | **music** |
| play the witcher | episodic_series | generic | **episodic_series** |
| play sherlock | episodic_series | generic | **episodic_series** |
| play stranger things | episodic_series | generic | **episodic_series** |
| binge the mandalorian | episodic_series | generic | **episodic_series** |
| play me something by hans zimmer | music | generic | **music** |
| can ya throw on the office | episodic_series | generic | **episodic_series** |
| cmon play breaking bad already | episodic_series | generic | **episodic_series** |
| can you just pop on stranger things | episodic_series | generic | **episodic_series** |

## Routing axes — confident-wrong vs abstain (safe)

| backend | media_type wrong | media_type abstain | playback wrong | playback abstain | adult over-flag |
|---|---|---|---|---|---|
| keyword | 0.050 (7/139) | 0.396 (55/139) | 0.029 (4/139) | 0.403 (56/139) | 0.000 (0/199) |
| onnx:models/context | 0.101 (14/139) | 0.324 (45/139) | 0.266 (37/139) | 0.014 (2/139) | 0.000 (0/199) |
| onnx:models/context_ner | 0.166 (23/139) | 0.410 (57/139) | 0.230 (32/139) | 0.079 (11/139) | 0.000 (0/199) |
| onnx:models_text/tfidf_char35 | 0.374 (52/139) | 0.000 (0/139) | 0.252 (35/139) | 0.022 (3/139) | 0.000 (0/199) |
| onnx:models_text/tfidf_word12 | 0.389 (54/139) | 0.043 (6/139) | 0.209 (29/139) | 0.014 (2/139) | 0.000 (0/199) |
| onnx:models_text/tfidf_word13 | 0.396 (55/139) | 0.029 (4/139) | 0.237 (33/139) | 0.022 (3/139) | 0.000 (0/199) |
| onnx:models_torch/cat | 0.345 (48/139) | 0.000 (0/139) | 0.266 (37/139) | 0.072 (10/139) | 0.015 (3/199) |
| onnx:models_torch/cat_all | 0.360 (50/139) | 0.000 (0/139) | 0.216 (30/139) | 0.022 (3/139) | 0.045 (9/199) |
| onnx:models_torch/cat_all_deep | 0.338 (47/139) | 0.000 (0/139) | 0.194 (27/139) | 0.022 (3/139) | 0.065 (13/199) |
| onnx:models_torch/cat_all_wide | 0.352 (49/139) | 0.000 (0/139) | 0.194 (27/139) | 0.022 (3/139) | 0.055 (11/199) |
| onnx:models_torch/cat_text | 0.281 (39/139) | 0.000 (0/139) | 0.209 (29/139) | 0.022 (3/139) | 0.055 (11/199) |
| onnx:models_torch/cat_wordvec | 0.295 (41/139) | 0.000 (0/139) | 0.201 (28/139) | 0.022 (3/139) | 0.105 (21/199) |
| onnx:models_torch/cat_wordvec_cbow | 0.374 (52/139) | 0.000 (0/139) | 0.209 (29/139) | 0.022 (3/139) | 0.126 (25/199) |
| embedding-router:embedding_router | 0.000 (0/139) | 1.000 (139/139) | 0.000 (0/139) | 1.000 (139/139) | 0.000 (0/199) |
| hybrid:embedding_router | 0.050 (7/139) | 0.396 (55/139) | 0.029 (4/139) | 0.403 (56/139) | 0.000 (0/199) |
| hybrid+gazetteer:embedding_router | 0.050 (7/139) | 0.259 (36/139) | 0.029 (4/139) | 0.266 (37/139) | 0.000 (0/199) |
| hybrid+inject:embedding_router | 0.029 (4/139) | 0.230 (32/139) | 0.036 (5/139) | 0.237 (33/139) | 0.000 (0/199) |
