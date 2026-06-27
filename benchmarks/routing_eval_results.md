# ovos-media-classifier — harm-weighted routing eval

Out-of-distribution, hand-curated eval: **186 cases** across 4 languages (de-de=15, en-us=140, es-es=16, pt-pt=15); **21 adult** cases, **145 abstain-ok** cases.

Composition by category: content_policy=21, control=28, gate_negative=41, keywordless=21, media=59, noise=3, playback_divergent=13.

**The metric is mis-route rate (confident-wrong), and GENERIC/abstain is scored as SAFE.** A confident-wrong route prunes the correct provider (harm); an abstain still lets every provider search (harmless).

## Headline

| backend | mis-route | adult-leak | false-hijack | false-miss | control recall |
|---|---|---|---|---|---|
| keyword | **0.035** (4/114) | **0.000** (0/21) | 0.227 (10/44) | 0.113 (16/142) | 0.500 (14/28) |
| embedding-router:embedding_router | **0.000** (0/114) | **1.000** (21/21) | 0.000 (0/44) | 1.000 (142/142) | 0.000 (0/28) |
| hybrid:embedding_router | **0.035** (4/114) | **0.000** (0/21) | 0.227 (10/44) | 0.113 (16/142) | 0.500 (14/28) |
| hybrid+inject:embedding_router | **0.026** (3/114) | **0.000** (0/21) | 0.227 (10/44) | 0.113 (16/142) | 0.500 (14/28) |

## Routing axes — confident-wrong vs abstain (safe)

| backend | media_type wrong | media_type abstain | playback wrong | playback abstain | adult over-flag |
|---|---|---|---|---|---|
| keyword | 0.035 (4/114) | 0.430 (49/114) | 0.026 (3/114) | 0.439 (50/114) | 0.000 (0/165) |
| embedding-router:embedding_router | 0.000 (0/114) | 1.000 (114/114) | 0.000 (0/114) | 1.000 (114/114) | 0.000 (0/165) |
| hybrid:embedding_router | 0.035 (4/114) | 0.430 (49/114) | 0.026 (3/114) | 0.439 (50/114) | 0.000 (0/165) |
| hybrid+inject:embedding_router | 0.026 (3/114) | 0.246 (28/114) | 0.035 (4/114) | 0.254 (29/114) | 0.000 (0/165) |
