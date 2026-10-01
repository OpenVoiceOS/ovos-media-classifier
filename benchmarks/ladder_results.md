# ovos-media-classifier — the multi-task ladder

Held-out **test split**: 34,700 utterances. The ladder runs **rules → sklearn (categorical) → neural × feature set** (categorical → +char-hash text → +domain word-vectors → all) on the SAME rows. Neural rungs build their text features from the raw utterance at inference (numpy only); latency is measured including that featurization.

## Single-label axes — accuracy

| axis | rules | sklearn context | sklearn context+NER |
|---|---|---|---|
| domain | 0.878 | 0.873 | 0.989 |
| media_type | 0.675 | 0.789 | 0.962 |
| playback_type | 0.724 | 0.899 | 0.988 |
| structure | 0.746 | 0.902 | 0.988 |
| explicitness | 0.989 | 0.990 | 0.998 |
| content_form | 0.751 | 0.984 | 0.997 |
| programme_format | 0.896 | 0.993 | 0.998 |

## Multi-label axes — macro-F1

| axis | rules | sklearn context | sklearn context+NER |
|---|---|---|---|
| content_form_genres | 0.731 | 0.750 | 0.973 |
| content_genres | 0.000 | 0.556 | 0.586 |
| picture_format | 0.878 | 0.878 | 0.965 |

## Content filter (from the content_form_genres axis)

| rung | adult recall | hentai recall | false-block | median ms | p95 ms | size |
|---|---|---|---|---|---|---|
| rules | 0.581 (436/751) | 0.510 | 0.002 | 0.5889 | 0.9650 | — |
| sklearn context | 0.578 (434/751) | 0.503 | 0.001 | 0.1684 | 0.2017 | 97 KiB |
| sklearn context+NER | 0.904 (679/751) | 0.903 | 0.000 | 0.2498 | 0.3947 | 2.1 MiB |
