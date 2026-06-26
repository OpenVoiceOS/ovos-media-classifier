# ovos-media-classifier — the multi-task ladder

Held-out **test split**: 34,700 utterances. Per axis, the lift from **rules → learned (context-only) → learned (context+NER)** is the headline result.

## Single-label axes — accuracy

| axis | rules | learned-context | learned-context+NER |
|---|---|---|---|
| domain | 0.844 | 0.873 | 0.988 |
| media_type | 0.663 | 0.786 | 0.967 |
| playback_type | 0.717 | 0.895 | 0.990 |
| structure | 0.738 | 0.908 | 0.992 |
| explicitness | 0.989 | 0.989 | 0.998 |

## Multi-label axes — macro-F1

| axis | rules | learned-context | learned-context+NER |
|---|---|---|---|
| content_form_genres | 0.720 | 0.729 | 0.979 |
| tags | 0.000 | 0.561 | 0.596 |
| qualifiers | 0.000 | 0.780 | 0.945 |

## Content filter (from the content_form_genres axis)

| rung | adult recall | hentai recall | false-block | median ms | p95 ms | size |
|---|---|---|---|---|---|---|
| rules | 0.479 (350/731) | 0.453 | 0.001 | 0.5090 | 0.8222 | — |
| learned-context | 0.479 (350/731) | 0.453 | 0.000 | 0.2827 | 0.5186 | 380 KiB |
| learned-context+NER | 0.932 (681/731) | 0.937 | 0.001 | 0.2609 | 0.3817 | 289 KiB |
