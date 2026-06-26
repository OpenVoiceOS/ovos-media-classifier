# ovos-media-classifier — the multi-task ladder

Held-out **test split**: 34,700 utterances. Per axis, the lift from **rules → learned (context-only) → learned (context+NER)** is the headline result.

## Single-label axes — accuracy

| axis | rules | learned-context | learned-context+NER |
|---|---|---|---|
| domain | 0.833 | 0.866 | 0.986 |
| media_type | 0.629 | 0.778 | 0.964 |
| playback_type | 0.702 | 0.895 | 0.988 |
| structure | 0.708 | 0.907 | 0.990 |
| explicitness | 0.988 | 0.989 | 0.997 |

## Multi-label axes — macro-F1

| axis | rules | learned-context | learned-context+NER |
|---|---|---|---|
| content_form_genres | 0.706 | 0.738 | 0.975 |
| tags | 0.000 | 0.547 | 0.581 |
| qualifiers | 0.000 | 0.746 | 0.906 |

## Content filter (from the content_form_genres axis)

| rung | adult recall | hentai recall | false-block | median ms | p95 ms | size |
|---|---|---|---|---|---|---|
| rules | 0.481 (364/756) | 0.510 | 0.000 | 0.3191 | 0.4979 | — |
| learned-context | 0.481 (364/756) | 0.510 | 0.000 | 0.2109 | 0.2503 | 176 KiB |
| learned-context+NER | 0.922 (697/756) | 0.936 | 0.001 | 0.2144 | 0.2592 | 289 KiB |
