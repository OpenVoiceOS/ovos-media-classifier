# ovos-media-classifier — harm-weighted routing eval

Out-of-distribution, hand-curated eval: **186 cases** across 4 languages (de-de=15, en-us=140, es-es=16, pt-pt=15); **21 adult** cases, **145 abstain-ok** cases.

Composition by category: content_policy=21, control=28, gate_negative=41, keywordless=21, media=59, noise=3, playback_divergent=13.

**The metric is mis-route rate (confident-wrong), and GENERIC/abstain is scored as SAFE.** A confident-wrong route prunes the correct provider (harm); an abstain still lets every provider search (harmless).

## Headline

| backend | mis-route | adult-leak | false-hijack | false-miss | control recall |
|---|---|---|---|---|---|
| keyword | **0.070** (8/114) | **0.429** (9/21) | 0.227 (10/44) | 0.204 (29/142) | 0.500 (14/28) |
| onnx:models/context | **0.097** (11/114) | **0.429** (9/21) | 0.182 (8/44) | 0.479 (68/142) | 0.000 (0/28) |
| onnx:models/context_ner | **0.149** (17/114) | **0.429** (9/21) | 0.159 (7/44) | 0.528 (75/142) | 0.000 (0/28) |
| onnx:models_text/tfidf_char35 | **0.395** (45/114) | **0.619** (13/21) | 0.841 (37/44) | 0.099 (14/142) | 0.000 (0/28) |
| onnx:models_text/tfidf_word12 | **0.430** (49/114) | **0.619** (13/21) | 0.659 (29/44) | 0.155 (22/142) | 0.000 (0/28) |
| onnx:models_text/tfidf_word13 | **0.439** (50/114) | **0.619** (13/21) | 0.682 (30/44) | 0.141 (20/142) | 0.000 (0/28) |
| onnx:models_torch/cat | **0.430** (49/114) | **0.381** (8/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) |
| onnx:models_torch/cat_all | **0.430** (49/114) | **0.238** (5/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) |
| onnx:models_torch/cat_all_deep | **0.404** (46/114) | **0.238** (5/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) |
| onnx:models_torch/cat_all_wide | **0.447** (51/114) | **0.286** (6/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) |
| onnx:models_torch/cat_text | **0.351** (40/114) | **0.238** (5/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) |
| onnx:models_torch/cat_wordvec | **0.360** (41/114) | **0.286** (6/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) |
| onnx:models_torch/cat_wordvec_cbow | **0.465** (53/114) | **0.333** (7/21) | 1.000 (44/44) | 0.000 (0/142) | 0.000 (0/28) |

## Routing axes — confident-wrong vs abstain (safe)

| backend | media_type wrong | media_type abstain | playback wrong | playback abstain | adult over-flag |
|---|---|---|---|---|---|
| keyword | 0.070 (8/114) | 0.465 (53/114) | 0.061 (7/114) | 0.474 (54/114) | 0.000 (0/165) |
| onnx:models/context | 0.097 (11/114) | 0.430 (49/114) | 0.351 (40/114) | 0.009 (1/114) | 0.000 (0/165) |
| onnx:models/context_ner | 0.149 (17/114) | 0.491 (56/114) | 0.263 (30/114) | 0.053 (6/114) | 0.000 (0/165) |
| onnx:models_text/tfidf_char35 | 0.395 (45/114) | 0.000 (0/114) | 0.272 (31/114) | 0.018 (2/114) | 0.000 (0/165) |
| onnx:models_text/tfidf_word12 | 0.430 (49/114) | 0.053 (6/114) | 0.228 (26/114) | 0.009 (1/114) | 0.000 (0/165) |
| onnx:models_text/tfidf_word13 | 0.439 (50/114) | 0.035 (4/114) | 0.254 (29/114) | 0.018 (2/114) | 0.000 (0/165) |
| onnx:models_torch/cat | 0.430 (49/114) | 0.000 (0/114) | 0.342 (39/114) | 0.044 (5/114) | 0.012 (2/165) |
| onnx:models_torch/cat_all | 0.430 (49/114) | 0.000 (0/114) | 0.254 (29/114) | 0.018 (2/114) | 0.049 (8/165) |
| onnx:models_torch/cat_all_deep | 0.404 (46/114) | 0.000 (0/114) | 0.219 (25/114) | 0.018 (2/114) | 0.067 (11/165) |
| onnx:models_torch/cat_all_wide | 0.447 (51/114) | 0.000 (0/114) | 0.246 (28/114) | 0.018 (2/114) | 0.061 (10/165) |
| onnx:models_torch/cat_text | 0.351 (40/114) | 0.000 (0/114) | 0.237 (27/114) | 0.018 (2/114) | 0.067 (11/165) |
| onnx:models_torch/cat_wordvec | 0.360 (41/114) | 0.000 (0/114) | 0.246 (28/114) | 0.018 (2/114) | 0.121 (20/165) |
| onnx:models_torch/cat_wordvec_cbow | 0.465 (53/114) | 0.000 (0/114) | 0.228 (26/114) | 0.018 (2/114) | 0.145 (24/165) |
