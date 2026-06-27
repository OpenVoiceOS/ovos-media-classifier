# ovos-media-classifier — the multi-task ladder

Held-out **test split**: 34,700 utterances. The ladder runs **rules → sklearn (categorical) → neural × feature set** (categorical → +char-hash text → +domain word-vectors → all) on the SAME rows. Neural rungs build their text features from the raw utterance at inference (numpy only); latency is measured including that featurization.

## Single-label axes — accuracy

| axis | rules | sklearn context | sklearn context+NER | tfidf word(1,2)+linear | tfidf word(1,3)+linear | tfidf char(3,5)+linear | neural cat | neural cat+text(char-hash) | neural cat+wordvec(skip) | neural cat+wordvec(cbow) | neural cat+all | neural cat+all (deep) | neural cat+all (wide) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| domain | 0.844 | 0.873 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| media_type | 0.663 | 0.786 | 0.967 | 0.973 | 0.973 | 0.978 | 0.787 | 0.972 | 0.965 | 0.968 | 0.975 | 0.974 | 0.974 |
| playback_type | 0.717 | 0.895 | 0.990 | 0.987 | 0.987 | 0.989 | 0.887 | 0.987 | 0.984 | 0.986 | 0.988 | 0.988 | 0.988 |
| structure | 0.738 | 0.908 | 0.992 | 0.984 | 0.984 | 0.988 | 0.833 | 0.985 | 0.978 | 0.980 | 0.986 | 0.986 | 0.986 |
| explicitness | 0.989 | 0.989 | 0.998 | 0.997 | 0.997 | 0.997 | 0.921 | 0.992 | 0.985 | 0.983 | 0.995 | 0.995 | 0.993 |

## Multi-label axes — macro-F1

| axis | rules | sklearn context | sklearn context+NER | tfidf word(1,2)+linear | tfidf word(1,3)+linear | tfidf char(3,5)+linear | neural cat | neural cat+text(char-hash) | neural cat+wordvec(skip) | neural cat+wordvec(cbow) | neural cat+all | neural cat+all (deep) | neural cat+all (wide) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| content_form_genres | 0.720 | 0.729 | 0.979 | 0.863 | 0.872 | 0.799 | 0.510 | 0.858 | 0.717 | 0.741 | 0.875 | 0.895 | 0.876 |
| tags | 0.000 | 0.561 | 0.596 | 0.913 | 0.932 | 0.654 | 0.511 | 0.800 | 0.528 | 0.586 | 0.762 | 0.696 | 0.840 |
| qualifiers | 0.000 | 0.780 | 0.945 | 0.951 | 0.945 | 0.952 | 0.561 | 0.964 | 0.730 | 0.803 | 0.952 | 0.937 | 0.956 |

## Content filter (from the content_form_genres axis)

| rung | adult recall | hentai recall | false-block | median ms | p95 ms | size |
|---|---|---|---|---|---|---|
| rules | 0.479 (350/731) | 0.453 | 0.001 | 0.4750 | 0.8204 | — |
| sklearn context | 0.479 (350/731) | 0.453 | 0.000 | 0.2580 | 0.3708 | 380 KiB |
| sklearn context+NER | 0.932 (681/731) | 0.937 | 0.001 | 0.2494 | 0.2947 | 289 KiB |
| tfidf word(1,2)+linear | 0.826 (604/731) | 0.717 | 0.001 | 3.3401 | 3.8563 | 56.4 MiB |
| tfidf word(1,3)+linear | 0.841 (615/731) | 0.740 | 0.001 | 3.4380 | 3.9652 | 57.6 MiB |
| tfidf char(3,5)+linear | 0.845 (618/731) | 0.772 | 0.001 | 6.7258 | 11.1326 | 104.7 MiB |
| neural cat | 0.867 (634/731) | 0.894 | 0.077 | 0.1862 | 0.2533 | 1.3 MiB |
| neural cat+text(char-hash) | 0.921 (673/731) | 0.874 | 0.006 | 9.4803 | 28.1943 | 79.0 MiB |
| neural cat+wordvec(skip) | 0.977 (714/731) | 0.965 | 0.016 | 0.6803 | 1.0068 | 41.8 MiB |
| neural cat+wordvec(cbow) | 0.982 (718/731) | 0.980 | 0.017 | 0.6557 | 0.9881 | 41.8 MiB |
| neural cat+all | 0.919 (672/731) | 0.866 | 0.004 | 7.3285 | 23.3219 | 113.9 MiB |
| neural cat+all (deep) | 0.929 (679/731) | 0.862 | 0.004 | 12.5469 | 28.5395 | 165.5 MiB |
| neural cat+all (wide) | 0.943 (689/731) | 0.890 | 0.006 | 14.8122 | 32.4253 | 203.6 MiB |
