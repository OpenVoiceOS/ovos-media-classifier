# ovos-media-classifier — the multi-task ladder

Held-out **test split**: 27,125 utterances. Per axis, the lift from **rules → learned (context-only) → learned (context+NER)** is the headline result.

## Single-label axes — accuracy

| axis | rules | learned-context | learned-context+NER |
|---|---|---|---|
| domain | 0.898 | 0.856 | 0.989 |
| media_type | 0.703 | 0.789 | 0.985 |
| playback_type | 0.769 | 0.907 | 0.991 |
| structure | 0.755 | 0.878 | 0.995 |
| explicitness | 0.986 | 0.986 | 0.998 |
| mood | 0.000 | 0.090 | 0.079 |
| era | 0.000 | 0.098 | 0.098 |

## Multi-label axes — macro-F1

| axis | rules | learned-context | learned-context+NER |
|---|---|---|---|
| content_form_genres | 0.685 | 0.686 | 0.982 |
| content_genre | 0.000 | 0.000 | 0.000 |
| qualifiers | 0.000 | 0.746 | 0.918 |

## Content filter (from the content_form_genres axis)

| rung | adult recall | hentai recall | false-block | median ms | p95 ms | size |
|---|---|---|---|---|---|---|
| rules | 0.490 (356/727) | 0.433 | 0.000 | 0.3085 | 0.4946 | — |
| learned-context | 0.490 (356/727) | 0.433 | 0.000 | 0.1823 | 0.2091 | 65 KiB |
| learned-context+NER | 0.912 (663/727) | 0.979 | 0.000 | 0.1852 | 0.2145 | 154 KiB |
