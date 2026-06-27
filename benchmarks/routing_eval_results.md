# ovos-media-classifier — harm-weighted routing eval

Out-of-distribution, hand-curated eval: **186 cases** across 4 languages (de-de=15, en-us=140, es-es=16, pt-pt=15); **21 adult** cases, **145 abstain-ok** cases.

Composition by category: content_policy=21, control=28, gate_negative=41, keywordless=21, media=59, noise=3, playback_divergent=13.

**The metric is mis-route rate (confident-wrong), and GENERIC/abstain is scored as SAFE.** A confident-wrong route prunes the correct provider (harm); an abstain still lets every provider search (harmless).

## Headline

**mis-route** is the harm (GENERIC=safe); **resolved** is the open-vocab win (abstain_ok cases turned into the CORRECT confident route). A clean win over a cheaper layer is mis-route NOT WORSE *and* resolved HIGHER.

| backend | mis-route | resolved | adult-leak | false-hijack | false-miss | control recall | latency med/p95 |
|---|---|---|---|---|---|---|---|
| keyword | **0.035** (4/114) | **0.301** (22/73) | **0.000** (0/21) | 0.227 (10/44) | 0.113 (16/142) | 0.500 (14/28) | – |
| onnx:models/context | **0.079** (9/114) | **0.329** (24/73) | **0.048** (1/21) | 0.182 (8/44) | 0.437 (62/142) | 0.000 (0/28) | – |
| onnx:models/context_ner | **0.132** (15/114) | **0.233** (17/73) | **0.000** (0/21) | 0.159 (7/44) | 0.486 (69/142) | 0.000 (0/28) | – |
| onnx:models_text/tfidf_char35 | **0.395** (45/114) | **0.507** (37/73) | **0.619** (13/21) | 0.841 (37/44) | 0.099 (14/142) | 0.000 (0/28) | – |
| onnx:models_text/tfidf_word12 | **0.430** (49/114) | **0.466** (34/73) | **0.619** (13/21) | 0.659 (29/44) | 0.155 (22/142) | 0.000 (0/28) | – |
| onnx:models_text/tfidf_word13 | **0.439** (50/114) | **0.466** (34/73) | **0.619** (13/21) | 0.682 (30/44) | 0.141 (20/142) | 0.000 (0/28) | – |
| onnx:models_torch/cat | **0.360** (41/114) | **0.466** (34/73) | **0.000** (0/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) | – |
| onnx:models_torch/cat_all | **0.395** (45/114) | **0.466** (34/73) | **0.095** (2/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) | – |
| onnx:models_torch/cat_all_deep | **0.368** (42/114) | **0.479** (35/73) | **0.000** (0/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) | – |
| onnx:models_torch/cat_all_wide | **0.377** (43/114) | **0.466** (34/73) | **0.095** (2/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) | – |
| onnx:models_torch/cat_text | **0.298** (34/114) | **0.575** (42/73) | **0.048** (1/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) | – |
| onnx:models_torch/cat_wordvec | **0.316** (36/114) | **0.562** (41/73) | **0.048** (1/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) | – |
| onnx:models_torch/cat_wordvec_cbow | **0.412** (47/114) | **0.452** (33/73) | **0.095** (2/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) | – |
| embedding-router:embedding_router | **0.000** (0/114) | **0.000** (0/73) | **1.000** (21/21) | 0.000 (0/44) | 1.000 (142/142) | 0.000 (0/28) | – |
| hybrid:embedding_router | **0.035** (4/114) | **0.301** (22/73) | **0.000** (0/21) | 0.227 (10/44) | 0.113 (16/142) | 0.500 (14/28) | – |
| hybrid+gazetteer:embedding_router | **0.035** (4/114) | **0.520** (38/73) | **0.000** (0/21) | 0.227 (10/44) | 0.113 (16/142) | 0.500 (14/28) | – |
| hybrid+inject:embedding_router | **0.026** (3/114) | **0.616** (45/73) | **0.000** (0/21) | 0.227 (10/44) | 0.113 (16/142) | 0.500 (14/28) | – |

## Open-vocab cases each layer closes

Each row is a play-case the cheaper layer abstained on or mis-routed that the richer layer routes correctly.

### hybrid -> hybrid+gazetteer  (16 fixed)

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

## Routing axes — confident-wrong vs abstain (safe)

| backend | media_type wrong | media_type abstain | playback wrong | playback abstain | adult over-flag |
|---|---|---|---|---|---|
| keyword | 0.035 (4/114) | 0.430 (49/114) | 0.026 (3/114) | 0.439 (50/114) | 0.000 (0/165) |
| onnx:models/context | 0.079 (9/114) | 0.377 (43/114) | 0.289 (33/114) | 0.009 (1/114) | 0.000 (0/165) |
| onnx:models/context_ner | 0.132 (15/114) | 0.439 (50/114) | 0.246 (28/114) | 0.053 (6/114) | 0.000 (0/165) |
| onnx:models_text/tfidf_char35 | 0.395 (45/114) | 0.000 (0/114) | 0.272 (31/114) | 0.018 (2/114) | 0.000 (0/165) |
| onnx:models_text/tfidf_word12 | 0.430 (49/114) | 0.053 (6/114) | 0.228 (26/114) | 0.009 (1/114) | 0.000 (0/165) |
| onnx:models_text/tfidf_word13 | 0.439 (50/114) | 0.035 (4/114) | 0.254 (29/114) | 0.018 (2/114) | 0.000 (0/165) |
| onnx:models_torch/cat | 0.360 (41/114) | 0.000 (0/114) | 0.281 (32/114) | 0.044 (5/114) | 0.012 (2/165) |
| onnx:models_torch/cat_all | 0.395 (45/114) | 0.000 (0/114) | 0.237 (27/114) | 0.018 (2/114) | 0.049 (8/165) |
| onnx:models_torch/cat_all_deep | 0.368 (42/114) | 0.000 (0/114) | 0.210 (24/114) | 0.018 (2/114) | 0.067 (11/165) |
| onnx:models_torch/cat_all_wide | 0.377 (43/114) | 0.000 (0/114) | 0.219 (25/114) | 0.018 (2/114) | 0.061 (10/165) |
| onnx:models_torch/cat_text | 0.298 (34/114) | 0.000 (0/114) | 0.219 (25/114) | 0.018 (2/114) | 0.067 (11/165) |
| onnx:models_torch/cat_wordvec | 0.316 (36/114) | 0.000 (0/114) | 0.219 (25/114) | 0.018 (2/114) | 0.121 (20/165) |
| onnx:models_torch/cat_wordvec_cbow | 0.412 (47/114) | 0.000 (0/114) | 0.210 (24/114) | 0.018 (2/114) | 0.145 (24/165) |
| embedding-router:embedding_router | 0.000 (0/114) | 1.000 (114/114) | 0.000 (0/114) | 1.000 (114/114) | 0.000 (0/165) |
| hybrid:embedding_router | 0.035 (4/114) | 0.430 (49/114) | 0.026 (3/114) | 0.439 (50/114) | 0.000 (0/165) |
| hybrid+gazetteer:embedding_router | 0.035 (4/114) | 0.289 (33/114) | 0.026 (3/114) | 0.298 (34/114) | 0.000 (0/165) |
| hybrid+inject:embedding_router | 0.026 (3/114) | 0.246 (28/114) | 0.035 (4/114) | 0.254 (29/114) | 0.000 (0/165) |
