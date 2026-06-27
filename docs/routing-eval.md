# Harm-weighted, out-of-distribution routing eval

This is the **honest ground truth** for the classifier: does any backend route
*real* voice commands to the right place? It is deliberately separate from the
in-distribution benchmark in [`benchmarks/run.py`](../benchmarks/run.py), which
grades each backend on utterances **derived from the keyword backend's own
`.voc` templates** — so the keyword backend trivially scores ~99% accuracy
there (it is being tested on its own vocabulary). That number is a false green.

## The classifier is a router, not a resolver

`ovos-media-classifier` sits in the OCP pipeline as a **router**. It:

1. **gates** `is_ocp_query` — is this a media request at all;
2. **routes** by `media_type` / `playback_type` — which `MediaProvider`s to call
   and which to skip (e.g. skip video providers on an audio-only device);
3. applies **content policy** — `explicitness` / `adult` → drop adult providers;
4. emits `Signals` as **context** the providers then search.

It does **not** resolve the title to a stream — the providers do. That is what
makes the error cost **asymmetric**:

| route | effect | verdict |
|---|---|---|
| **confident-wrong** `media_type` | prunes the provider family that had the content; the user gets nothing | **harm** |
| **GENERIC / abstain** | every relevant provider still searches the query | **safe** |
| **false-hijack** (non-media → OCP) | steals the turn from the correct skill (weather, timer, smart-home) | **serious harm** |
| **false-miss** (media → not_ocp) | OCP never fires; user can rephrase | lesser harm |
| **adult-leak** (adult not flagged) | adult content reaches a clean provider / a child | **worst** |

So the headline metric is **mis-route rate** = the fraction of play-intent cases
that got a **confident, wrong** `media_type`. GENERIC/abstain is **never**
counted as wrong — it is the safe outcome, because the OCP search still happens.

## The eval set

[`benchmarks/routing_eval.jsonl`](../benchmarks/routing_eval.jsonl) — **186
hand-curated, out-of-distribution cases** across `en-us`, `es-es`, `de-de`,
`pt-pt`. The phrasings are intentionally **not** drawn from the
`locale/<lang>/dataset/*.intent` templates: they are how people actually talk —
elliptical ("i feel like listening to the beatles"), slang ("shove on some death
metal", "fire up call of duty"), typo'd ("wack on some lofi beats"), and
keyword-less (bare titles / artists with no media word at all).

Categories (see `category` field):

| category | n | what it probes |
|---|---|---|
| `media` | 59 | real media requests across all types, real titles/artists |
| `keywordless` | 21 | bare title/artist, **no** media keyword — the keyword backend's blind spot |
| `gate_negative` | 41 | non-media (weather, timer, smart-home, knowledge) — **false-hijack** is a serious harm here |
| `control` | 28 | transport control (pause / next / volume) → `ocp_control` |
| `content_policy` | 21 | adult requests — **must** be flagged; a leak is the worst error |
| `playback_divergent` | 13 | "read me X" (book) vs "play the audiobook X" (audiobook); "watch X" vs "listen to X" |
| `noise` | 3 | ASR garbage / social — must not hijack |

Each case carries explicit labels: `is_ocp_query`, `domain`, `media_type` (or
`generic`), `playback_type` (or `unknown`), `explicit` (adult), and an
**`abstain_ok`** flag marking cases where GENERIC is an acceptable route (a
genuinely ambiguous bare title — 145 of 186 cases). `abstain_ok` only excuses an
**abstain**; a confident *wrong* answer is still a mis-route.

Provenance: see [`benchmarks/README.md`](../benchmarks/README.md#routing-eval-set-provenance).

## The metrics

`benchmarks/routing_eval.py` computes, per backend (GENERIC == safe throughout):

- **`mis_route_rate`** — confident-wrong `media_type` over play-intent cases
  with a concrete expected type. **The number every future router must beat.**
- **gate `false_hijack_rate`** — non-media cases routed into OCP.
- **gate `false_miss_rate`** — media cases routed to `not_ocp`.
- **`adult_leak_rate`** (headline) — adult cases not flagged `adult` by
  `classify_content_form_genres`; plus an `overflag_rate` (clean wrongly
  flagged, a false block).
- per-axis **confident-wrong vs abstain** for `media_type` and `playback_type`.
- **control recall** — control cases routed to `ocp_control`.

```bash
python -m benchmarks.routing_eval                 # keyword + every data/models/* bundle
python -m benchmarks.routing_eval --only-keyword  # keyword only
python -m benchmarks.routing_eval --bundle <dir>  # add an explicit ONNX bundle
```

Outputs `benchmarks/routing_eval_results.{json,md}`.

## The real baseline

Numbers from the committed run (see
[`benchmarks/routing_eval_results.md`](../benchmarks/routing_eval_results.md)).
The two `data/models/context*` bundles are the current categorical-feature
trained output; the `models_text/*` and `models_torch/*` rows are
training-sweep artifacts kept for contrast.

| backend | mis-route | adult-leak | false-hijack | false-miss | control |
|---|---|---|---|---|---|
| **keyword** (default) | **0.070** (8/114) | 0.429 (9/21) | 0.227 (10/44) | 0.204 (29/142) | 0.500 (14/28) |
| onnx context (trained) | 0.097 (11/114) | 0.429 (9/21) | 0.182 (8/44) | **0.479** (68/142) | 0.000 (0/28) |
| onnx context_ner | 0.149 (17/114) | 0.429 (9/21) | 0.159 (7/44) | 0.528 (75/142) | 0.000 (0/28) |
| onnx torch/text sweep | 0.35–0.47 | 0.24–0.62 | **0.66–1.00** | 0.00–0.16 | 0.000 |

### Honest read

**The keyword backend is the strongest router**, and for the right reason: it is
**high-precision and abstains by default.** Its mis-route rate is the lowest
(0.070) and it abstains on 47% of play-intent cases (GENERIC = safe — providers
still search). It is the only backend that recognises control intents at all
(0.500 recall; the misses are volume/mute, which it does not model). This is the
number to beat — and beating it means **lowering mis-route or adult-leak without
trading them for false-hijack**, not raising raw accuracy.

**The trained bundles confirm the prediction: they look strong in-distribution
and mis-route out-of-distribution.** The `context` bundle reports
`val_macro_f1 ≈ 0.94` on its own held-out split, yet here it has a **0.479
false-miss rate** — it routes nearly half of real media requests (bare titles
like "play interstellar", "throw on breaking bad") to `not_ocp`, because its
synthetic training distribution under-covers keyword-less phrasings. It also has
**no control head**, so every control utterance is mis-gated.

**The torch/text training-sweep bundles are a cautionary tale**: most route
**100% of non-media into OCP** (false-hijack 1.000) — they learned "almost
everything is a play request" on a synthetic, play-heavy distribution. Their
in-distribution scores were fine; OOD they are unusable as routers. This is the
exact failure this eval exists to catch.

**Adult-leak is everyone's weak spot (≥ 0.238).** The shared miss is *coverage*,
not architecture: the keyword `AdultKeyword.voc` does not list "porno",
"pornhub", "striptease", "sex tape", "nudes", "erotica", "onlyfans", or German
plural "pornos" — and the trained heads inherit the gap from the same labelled
data. Because a leak is the worst error, **closing this is the highest-priority
follow-up** (expand the adult lexicon across locales; the harness re-measures it
directly).

### Concrete keyword mis-routes worth fixing

- `spiel jazzmusik` (de) → `game` (the play-verb "spiel" also matches the game
  voc; should be `music`).
- `spiel die simpsons` (de) → `game` (same "spiel" collision).
- `listen to harry potter` → `music` (should be `audiobook` — "listen" + book
  title).
- `play the latest episode of the daily` → `episodic_series` (should be
  `podcast`).
- `stream the lakers game` → `game` (should be `tv` — live sports).

These are the honest, actionable signal the in-distribution benchmark could
never surface.
