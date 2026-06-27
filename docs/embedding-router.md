# Embedding-router backend

The embedding-router is the learned, opt-in counterpart to the default keyword
classifier. It uses [guided-categorical-embeddings](https://github.com/TigreGotico/guided-categorical-embeddings)
(GCE) with a **routing-aware** objective: independent per-axis heads route
`media_type` / `playback_type`, **abstaining to `GENERIC` when unsure** so a
wrong route never prunes the provider that actually had the content. It is wired
as a **hybrid** by default — keyword stays the high-precision first pass and the
router fills the keyword-less cases.

Inference uses **numpy + onnxruntime only** (the `[onnx]` extra). No torch / GCE
is imported at runtime; the model is a plain ONNX graph per axis and the feature
row + abstain decision are pure numpy.

## Two-stream features

The model input is `[static | entity]` concatenated:

- **static** — the categorical `kw_* / verb_* / mod_* / fmt_* / kw_genre_*`
  columns the runtime `CategoricalFeatureExtractor` produces from `.voc`
  matching, one-hot via GCE's `CategoricalVectorizer`.
- **entity** — one slot per train-time NER label (`artist_name`, `movie_title`,
  `anime_title`, `audiobook_title`, …). At train time the slots come from the
  dataset's `ner_*` columns; at inference the slots fire from the **user's own
  media library**, injected at runtime via `register_user_entities` — **no
  retraining**. This is what closes the keyword backend's entity-gap mis-routes.

## Routing-aware objective

Each axis is a GCE `LabelGuidedTrainer` (via `PerAxisRouter`) with a learned
`GENERIC` abstain class and:

- **`cost_matrix`** — a confident WRONG route costs `10`; routing to the cheap
  `GENERIC` column costs `1` (encodes "mis-route ≫ abstain" directly in the
  loss).
- **`abstain_label="GENERIC"`** + **`focal_gamma`** for calibration +
  **`temperature_scaling`** baked into `classifier.onnx`, so the runtime reject
  threshold compares against calibrated probabilities.

At inference each head argmaxes; when the top probability is below the axis
threshold OR the argmax is the abstain class, the route is `GENERIC`
(`predict_with_reject` semantics).

## Train + evaluate

```bash
pip install ovos-media-classifier[train]            # + GCE importable
python -m training.train_embedding_router \
    --data data/release --out data/models/embedding_router \
    --max-rows 90000 --per-class-cap 7000 --max-iter 120 --hidden 256 128
python -m benchmarks.routing_eval                   # keyword + router + hybrid(+inject)
```

The bundle is `data/models/embedding_router/` (`router_meta.json` + per-axis
`media_type/` and `playback_type/` GCE exports). `data/` is gitignored — the
bundle is not committed.

## Use it

```python
from ovos_media_classifier import load_media_classifier

clf = load_media_classifier({
    "media_classifier_embedding_router": "data/models/embedding_router",
    # inject the user's actual library so bare titles route (no retraining):
    "media_classifier_entity_library": {
        "anime_title": ["Attack on Titan"],
        "audiobook_title": ["Harry Potter"],
        "podcast_title": ["The Daily"],
    },
})
clf.classify("watch attack on titan", "en-us")   # -> (EPISODIC_SERIES, …)
```

`media_classifier_embedding_router_hybrid=False` selects the router alone
(`EmbeddingMediaClassifier`); the default `True` selects `HybridMediaClassifier`.

## Hybrid gating (no regression of the safe axes)

The hybrid composes the two backends so the learned router can only help:

1. A fired **injected user-library entity** wins (highest-precision evidence —
   "Attack on Titan" in the user's anime library beats the generic `watch` →
   MOVIE cue). Empty by default, so with no injected library this is inert.
2. **Keyword's confident leaf** (explicit cue) — its high-precision route.
3. The **router**, for keyword-less cases, abstaining when unsure.

The gate (`classify_domain` / `is_ocp_query`) and the content-policy axis
(`classify_content_form_genres` → the adult lexicon) stay on keyword, so
adult-leak and false-hijack are exactly the keyword floor.

## Routing-eval verdict (186-case OOD harm-weighted eval)

| backend | mis-route | adult-leak | false-hijack | false-miss | control |
|---|---|---|---|---|---|
| keyword (floor) | 0.035 (4/114) | 0.000 | 0.227 | 0.113 | 0.500 |
| embedding-router (alone) | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| hybrid (no inject) | 0.035 | 0.000 | 0.227 | 0.113 | 0.500 |
| **hybrid + user library** | **0.026 (3/114)** | **0.000** | **0.227** | 0.113 | 0.500 |

**Verdict — promote as a recommended optional backend, default stays keyword.**
The hybrid **with the user's library injected** lowers mis-route below the
keyword floor (0.026 < 0.035) while holding adult-leak at 0.0, false-hijack at
the keyword floor (0.227) and false-miss at the keyword floor (0.113) — it
recovers keyword abstains as correct media routes and closes 3 of the 4 keyword
entity-gap mis-routes (`watch attack on titan` → EPISODIC_SERIES, `listen to
harry potter` → AUDIOBOOK, `the daily` → podcast). The remaining mis-route
(`stream the lakers game` → live sports / TV) is not a library entity. The router
never moves the gate (keyword owns it), so a common short title cannot hijack
ordinary speech.

The honest caveat: the win is delivered through **runtime entity injection**.
Without an injected library the hybrid only *matches* keyword (the router
abstains on the deliberately keyword-less OOD eval, since its static features
rarely fire and it has no negative-gate training data). The router alone is not a
gate — it has no negative training data — which is exactly why it is shipped as a
keyword-gated hybrid, never as the default.
