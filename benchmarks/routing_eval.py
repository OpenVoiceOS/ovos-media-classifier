"""Harm-weighted, out-of-distribution **routing** eval for ovos-media-classifier.

Why this exists
---------------
The classifier is a **router**, not an extractor/resolver.  It gates
``is_ocp_query``, routes by ``media_type`` / ``playback_type`` (which
MediaProviders to call or skip on an audio-only device), applies content-policy
(``adult`` → drop adult providers) and emits ``Signals`` as *context* for
providers to search.  The cost of an error is therefore **asymmetric**:

* a **confident-wrong** route filters out the provider that actually had the
  content — the user gets nothing.  This is the real harm.
* a **GENERIC / abstain** route is *harmless*: every relevant provider still
  searches the query, so a correct result is still reachable.

So the headline metric is **mis-route rate** (confident-wrong fraction), and
GENERIC/abstain is scored as **safe**, never as wrong.  This is the opposite of
``benchmarks/run.py``, which reports plain accuracy over the *in-distribution*
voc-derived eval set (where the keyword backend trivially scores ~99% because it
is graded on its own templates).

The eval set (``routing_eval.jsonl``) is **hand-curated and out-of-distribution**
— real, messy, elliptical, typo'd phrasings that deliberately do **not** reuse
the ``locale/<lang>/dataset/*.intent`` template structure — so this is the honest
ground truth: does a backend route *real* voice commands correctly?

Per-case schema (one JSON object per line)::

    utterance       the raw user phrasing
    lang            BCP-47 tag
    category        media / keywordless / gate_negative / control /
                    content_policy / playback_divergent / noise
    is_ocp_query    expected gate (bool)
    domain          ocp_play / ocp_control / not_ocp
    media_type      expected mediavocab MediaType value, or "generic"/"control"
    playback_type   expected mediavocab PlaybackType value, or "unknown"
    explicit        True when the request is adult (must be flagged)
    abstain_ok      True when GENERIC/abstain is an *acceptable* route for this
                    case (genuinely ambiguous, e.g. a bare title) — the metric
                    only penalises a CONFIDENT WRONG answer here, never abstain.
    note            provenance / rationale (free text)

Metrics (all "GENERIC == safe")
-------------------------------
* **mis_route_rate** — over the play-intent cases, the fraction that received a
  CONFIDENT WRONG media_type (predicted a non-GENERIC type that disagrees with
  the expected one).  GENERIC/abstain is excluded (safe).  This is THE number.
* **gate false_hijack** — non-media (not_ocp) cases the backend routed *into*
  OCP (play or control).  A serious harm: it steals the turn from the right
  skill.
* **gate false_miss** — media (ocp_play/ocp_control) cases the backend routed to
  not_ocp.  Less harmful than a hijack (the user can rephrase) but still a miss.
* **adult_leak_rate** (HEADLINE) — adult cases NOT flagged adult by
  ``classify_content_form_genres``.  The worst error: adult leaks to a clean
  provider.  Reported as its own headline, weighted highest.
* **media_type** confident-wrong vs abstain, and **playback_type** confident-wrong
  vs abstain — the routing axes broken out.
* **resolved_rate** — over the ``abstain_ok`` play-cases (open-vocab: a bare title
  where GENERIC/abstain is an *acceptable* answer, so mis-route can never credit a
  win there), the fraction the backend turned into the CORRECT confident route.
  This is the gazetteer / entity-injection VALUE metric: mis-route (GENERIC=safe)
  cannot reward turning an abstain into a correct answer, so a layer's open-vocab
  wins are invisible to it.  ``resolved_rate`` makes them visible — a clean win is
  mis-route NOT WORSE *and* resolved_rate HIGHER than the cheaper layer.

Usage::

    python -m benchmarks.routing_eval                 # keyword + any data/models/* bundles
    python -m benchmarks.routing_eval --bundle path   # add an explicit ONNX bundle
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
EVAL_JSONL = os.path.join(HERE, "routing_eval.jsonl")
RESULTS_JSON = os.path.join(HERE, "routing_eval_results.json")
RESULTS_MD = os.path.join(HERE, "routing_eval_results.md")

# Sentinel media-type values that mean "no confident route" (the safe outcome).
ABSTAIN_TYPES = {"generic", "not_media", "control"}
ABSTAIN_PLAYBACK = {"unknown"}


# ---------------------------------------------------------------------------
# Eval set
# ---------------------------------------------------------------------------

@dataclass
class Case:
    utterance: str
    lang: str
    category: str
    is_ocp_query: bool
    domain: str
    media_type: str
    playback_type: str
    explicit: bool
    abstain_ok: bool
    note: str = ""


def load_cases(path: str = EVAL_JSONL) -> List[Case]:
    cases: List[Case] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cases.append(Case(
                utterance=d["utterance"], lang=d["lang"], category=d["category"],
                is_ocp_query=bool(d["is_ocp_query"]), domain=d["domain"],
                media_type=d["media_type"], playback_type=d["playback_type"],
                explicit=bool(d["explicit"]), abstain_ok=bool(d["abstain_ok"]),
                note=d.get("note", ""),
            ))
    return cases


# ---------------------------------------------------------------------------
# Per-backend prediction
# ---------------------------------------------------------------------------

@dataclass
class Pred:
    domain: str          # ocp_play / ocp_control / not_ocp
    is_ocp: bool
    media_type: str      # mediavocab value or "generic"
    playback_type: str   # mediavocab value or "unknown"
    adult: bool          # 'adult' in content_form_genres


def predict(clf, case: Case) -> Pred:
    """Run every routing axis the backend exposes for one utterance.

    All calls are wrapped so a backend that raises on an axis degrades to the
    safe (abstain) value rather than crashing the run.
    """
    q, lang = case.utterance, case.lang

    def _safe(fn, default):
        try:
            return fn()
        except Exception:
            return default

    domain_obj, _ = _safe(lambda: clf.classify_domain(q, lang), (None, 0.0))
    domain = getattr(domain_obj, "value", None) or "not_ocp"
    is_ocp, _ = _safe(lambda: clf.is_ocp_query(q, lang), (domain != "not_ocp", 0.0))

    mt_obj, _ = _safe(lambda: clf.classify(q, lang), (None, 0.0))
    media_type = getattr(mt_obj, "value", None) or "generic"

    pb_obj = _safe(lambda: clf.classify_playback_type(q, lang), None)
    playback_type = getattr(pb_obj, "value", None) or "unknown"

    genres = _safe(lambda: clf.classify_content_form_genres(q, lang), []) or []
    adult = "adult" in {str(g).lower() for g in genres}

    return Pred(domain=domain, is_ocp=bool(is_ocp), media_type=media_type,
                playback_type=playback_type, adult=adult)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class AxisCount:
    """Confident-wrong / abstain / correct tally for one routing axis."""
    confident_correct: int = 0
    confident_wrong: int = 0   # the harm
    abstain: int = 0           # safe
    n: int = 0

    def add_correct(self):
        self.confident_correct += 1
        self.n += 1

    def add_wrong(self):
        self.confident_wrong += 1
        self.n += 1

    def add_abstain(self):
        self.abstain += 1
        self.n += 1

    @property
    def mis_route_rate(self) -> float:
        return self.confident_wrong / self.n if self.n else 0.0

    @property
    def abstain_rate(self) -> float:
        return self.abstain / self.n if self.n else 0.0


@dataclass
class BackendReport:
    name: str
    status: str = "available"
    reason: str = ""
    # gate
    gate_hijack_n: int = 0          # non-media routed into OCP
    gate_hijack_total: int = 0      # non-media cases
    gate_miss_n: int = 0            # media routed to not_ocp
    gate_miss_total: int = 0        # media cases
    # routing axes (over play-intent cases that have a concrete expected type)
    media_type: AxisCount = field(default_factory=AxisCount)
    playback_type: AxisCount = field(default_factory=AxisCount)
    # content policy (headline)
    adult_total: int = 0
    adult_leak_n: int = 0           # adult NOT flagged  ← worst
    adult_overflag_n: int = 0       # clean flagged adult (false block)
    clean_total: int = 0
    # control
    control_total: int = 0
    control_hit_n: int = 0          # control cases routed to ocp_control
    # open-vocab resolution (the gazetteer / entity-injection value metric):
    # abstain_ok play-cases the backend turned into a CORRECT confident route.
    resolved_total: int = 0         # abstain_ok play-cases with a concrete type
    resolved_n: int = 0             # of those, routed correctly + confidently
    n: int = 0
    # per-utterance predictions (for cross-layer fix attribution)
    preds: Dict[str, "Pred"] = field(default_factory=dict)
    # per-query latency (ms), populated only for the online layer
    latencies_ms: List[float] = field(default_factory=list)

    def latency_summary(self) -> Optional[dict]:
        if not self.latencies_ms:
            return None
        xs = sorted(self.latencies_ms)
        n = len(xs)
        median = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
        p95 = xs[min(n - 1, int(round(0.95 * (n - 1))))]
        return {"n": n, "median_ms": round(median, 1), "p95_ms": round(p95, 1),
                "max_ms": round(xs[-1], 1)}

    @property
    def mis_route_rate(self) -> float:
        """The headline number: confident-wrong media_type fraction (GENERIC safe)."""
        return self.media_type.mis_route_rate

    @property
    def false_hijack_rate(self) -> float:
        return self.gate_hijack_n / self.gate_hijack_total if self.gate_hijack_total else 0.0

    @property
    def false_miss_rate(self) -> float:
        return self.gate_miss_n / self.gate_miss_total if self.gate_miss_total else 0.0

    @property
    def adult_leak_rate(self) -> float:
        return self.adult_leak_n / self.adult_total if self.adult_total else 0.0

    @property
    def adult_overflag_rate(self) -> float:
        return self.adult_overflag_n / self.clean_total if self.clean_total else 0.0

    @property
    def control_recall(self) -> float:
        return self.control_hit_n / self.control_total if self.control_total else 0.0

    @property
    def resolved_rate(self) -> float:
        """Fraction of open-vocab (abstain_ok) play-cases routed correctly.

        The complement of the harm metric: where mis-route credits "abstain is
        safe", this credits "abstain turned into the right answer" — the
        open-vocab value a richer layer (gazetteer / injected library) adds.
        """
        return self.resolved_n / self.resolved_total if self.resolved_total else 0.0

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "n": self.n,
            "mis_route_rate": round(self.media_type.mis_route_rate, 4),
            "media_type": {
                "confident_correct": self.media_type.confident_correct,
                "confident_wrong": self.media_type.confident_wrong,
                "abstain": self.media_type.abstain,
                "n": self.media_type.n,
                "mis_route_rate": round(self.media_type.mis_route_rate, 4),
                "abstain_rate": round(self.media_type.abstain_rate, 4),
            },
            "playback_type": {
                "confident_correct": self.playback_type.confident_correct,
                "confident_wrong": self.playback_type.confident_wrong,
                "abstain": self.playback_type.abstain,
                "n": self.playback_type.n,
                "mis_route_rate": round(self.playback_type.mis_route_rate, 4),
                "abstain_rate": round(self.playback_type.abstain_rate, 4),
            },
            "gate": {
                "false_hijack_n": self.gate_hijack_n,
                "false_hijack_total": self.gate_hijack_total,
                "false_hijack_rate": round(self.false_hijack_rate, 4),
                "false_miss_n": self.gate_miss_n,
                "false_miss_total": self.gate_miss_total,
                "false_miss_rate": round(self.false_miss_rate, 4),
            },
            "adult_leak": {
                "adult_total": self.adult_total,
                "leak_n": self.adult_leak_n,
                "leak_rate": round(self.adult_leak_rate, 4),
                "clean_total": self.clean_total,
                "overflag_n": self.adult_overflag_n,
                "overflag_rate": round(self.adult_overflag_rate, 4),
            },
            "control": {
                "total": self.control_total,
                "hit_n": self.control_hit_n,
                "recall": round(self.control_recall, 4),
            },
            "resolved": {
                "total": self.resolved_total,
                "resolved_n": self.resolved_n,
                "rate": round(self.resolved_rate, 4),
            },
            "latency": self.latency_summary(),
        }


def evaluate(clf, cases: List[Case], name: str,
             measure_latency: bool = False) -> BackendReport:
    import time

    rep = BackendReport(name=name)
    rep.n = len(cases)

    for c in cases:
        if measure_latency:
            t0 = time.perf_counter()
            p = predict(clf, c)
            rep.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        else:
            p = predict(clf, c)
        rep.preds[c.utterance] = p

        # ---- gate (every case) -----------------------------------------
        if not c.is_ocp_query:
            # non-media: any OCP route is a hijack
            rep.gate_hijack_total += 1
            if p.is_ocp:
                rep.gate_hijack_n += 1
        else:
            # media (play or control): a not_ocp route is a miss
            rep.gate_miss_total += 1
            if not p.is_ocp:
                rep.gate_miss_n += 1

        # ---- content policy (headline) ---------------------------------
        if c.explicit:
            rep.adult_total += 1
            if not p.adult:
                rep.adult_leak_n += 1          # adult leaked to a clean provider
        else:
            rep.clean_total += 1
            if p.adult:
                rep.adult_overflag_n += 1      # clean wrongly flagged

        # ---- control recall --------------------------------------------
        if c.domain == "ocp_control":
            rep.control_total += 1
            if p.domain == "ocp_control":
                rep.control_hit_n += 1

        # ---- media_type routing axis (play-intent cases only) ----------
        # Only score the leaf where there is a concrete expected type to route
        # to (skip control / not_ocp / noise — those have no media leaf).
        if c.domain == "ocp_play" and c.media_type not in ABSTAIN_TYPES:
            # ---- open-vocab resolution (the gazetteer / injection value) ------
            # abstain_ok cases are the ones the harm metric can never credit a
            # win on (GENERIC is an accepted answer there).  Count how many a
            # backend turns into the CORRECT confident route.
            if c.abstain_ok:
                rep.resolved_total += 1
                if (p.media_type not in ABSTAIN_TYPES
                        and p.media_type == c.media_type):
                    rep.resolved_n += 1

            predicted_abstain = p.media_type in ABSTAIN_TYPES
            if predicted_abstain:
                rep.media_type.add_abstain()       # safe (providers still search)
            elif p.media_type == c.media_type:
                rep.media_type.add_correct()
            else:
                # a confident, WRONG leaf — unless this case is explicitly
                # marked abstain_ok AND the wrong leaf shares the playback
                # modality (then the right provider family still searches);
                # we still count it as a mis-route because a wrong concrete
                # type *does* prune providers.  abstain_ok only excuses ABSTAIN,
                # never a confident wrong answer.
                rep.media_type.add_wrong()

            # ---- playback_type routing axis ----------------------------
            if c.playback_type not in ABSTAIN_PLAYBACK:
                if p.playback_type in ABSTAIN_PLAYBACK:
                    rep.playback_type.add_abstain()
                elif p.playback_type == c.playback_type:
                    rep.playback_type.add_correct()
                else:
                    rep.playback_type.add_wrong()

    return rep


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

def _load_keyword():
    from ovos_media_classifier import KeywordMediaClassifier
    return KeywordMediaClassifier()


def _onnx_loader(bundle_dir: str) -> Callable:
    def _load():
        from ovos_media_classifier.onnx import OnnxMediaClassifier
        return OnnxMediaClassifier.from_path(bundle_dir)
    return _load


# A small representative user media library for the runtime-entity-injection
# demo: injecting it lets the embedding router close the keyword backend's
# entity-gap mis-routes WITHOUT retraining (e.g. "watch attack on titan" → the
# user's anime library → EPISODIC_SERIES, not the generic ``watch`` → MOVIE cue).
DEMO_LIBRARY: Dict[str, List[str]] = {
    "anime_title": ["Attack on Titan", "Naruto"],
    "audiobook_title": ["Harry Potter", "Moby Dick"],
    "artist_name": ["Radiohead", "Billie Eilish", "Kendrick", "The Beatles",
                    "Despacito", "Beethoven"],
    "tv_show_title": ["Seinfeld", "Sherlock", "Stranger Things", "The Witcher",
                      "Wednesday", "Spongebob"],
    "game_title": ["The Legend of Zelda"],
    "podcast_title": ["Serial", "The Daily"],
    "tv_channel": ["CNN"],
    "radio_station": ["Classic FM"],
    "movie_title": ["Interstellar", "The Godfather", "Hamilton"],
}


def _embedding_loader(bundle_dir: str) -> Callable:
    def _load():
        from ovos_media_classifier.embedding import EmbeddingMediaClassifier
        return EmbeddingMediaClassifier.from_path(bundle_dir)
    return _load


def _hybrid_loader(bundle_dir: str, inject: bool = False,
                   gazetteer: bool = False, online: bool = False) -> Callable:
    def _load():
        from ovos_media_classifier.embedding import HybridMediaClassifier
        online_clf = None
        if online:
            from ovos_media_classifier.metadatarr_backend import (
                MetadatarrMediaClassifier,
            )
            online_clf = MetadatarrMediaClassifier()
        clf = HybridMediaClassifier.from_path(bundle_dir, online=online_clf)
        if gazetteer:
            clf.register_default_gazetteer()
        if inject:
            clf.register_user_library(DEMO_LIBRARY)
        return clf
    return _load


def discover_router_bundles() -> Dict[str, str]:
    """Find embedding-router bundles under ``data/`` (a dir with router_meta.json)."""
    roots = [
        os.path.join(REPO_ROOT, "data", "models"),
        os.path.join(REPO_ROOT, "data", "release", "models"),
    ]
    found: Dict[str, str] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            d = os.path.join(root, entry)
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "router_meta.json")):
                found[entry] = d
    return found


def discover_bundles() -> Dict[str, str]:
    """Find trained ONNX bundles under ``data/`` (a dir with domain.onnx+meta.json)."""
    roots = [
        os.path.join(REPO_ROOT, "data", "models"),
        os.path.join(REPO_ROOT, "data", "models_text"),
        os.path.join(REPO_ROOT, "data", "models_torch"),
        os.path.join(REPO_ROOT, "data", "release", "models"),
    ]
    found: Dict[str, str] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            d = os.path.join(root, entry)
            if (os.path.isdir(d)
                    and os.path.isfile(os.path.join(d, "domain.onnx"))
                    and os.path.isfile(os.path.join(d, "meta.json"))):
                label = f"onnx:{os.path.basename(root)}/{entry}"
                found[label] = d
    return found


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(extra_bundles: Optional[List[str]] = None,
        only_keyword: bool = False,
        online: bool = False) -> Dict[str, object]:
    cases = load_cases()

    loaders: Dict[str, Callable] = {"keyword": _load_keyword}
    if not only_keyword:
        for label, path in discover_bundles().items():
            loaders[label] = _onnx_loader(path)
        for path in (extra_bundles or []):
            loaders[f"onnx:{os.path.basename(path.rstrip('/'))}"] = _onnx_loader(path)
        # embedding-router bundles: report the router alone, the keyword+router
        # hybrid, and the hybrid with a demo user library injected (the
        # runtime-injection result) so the promote/hold verdict is reproducible.
        for label, path in discover_router_bundles().items():
            loaders[f"embedding-router:{label}"] = _embedding_loader(path)
            loaders[f"hybrid:{label}"] = _hybrid_loader(path, inject=False)
            # Layer A — hybrid + offline gazetteer (no network, no user setup)
            loaders[f"hybrid+gazetteer:{label}"] = _hybrid_loader(
                path, gazetteer=True)
            loaders[f"hybrid+inject:{label}"] = _hybrid_loader(path, inject=True)
            if online:
                # Layer B — hybrid + gazetteer + online metadatarr (network)
                loaders[f"hybrid+gazetteer+online:{label}"] = _hybrid_loader(
                    path, gazetteer=True, online=True)

    reports: Dict[str, dict] = {}
    objs: Dict[str, BackendReport] = {}
    for name, loader in loaders.items():
        try:
            clf = loader()
        except Exception as e:  # noqa: BLE001 - record, never crash
            rep = BackendReport(name=name, status="unavailable",
                                reason=f"{type(e).__name__}: {e}")
            reports[name] = rep.as_dict()
            continue
        rep = evaluate(clf, cases, name, measure_latency="online" in name)
        objs[name] = rep
        reports[name] = rep.as_dict()

    composition = _composition(cases)
    fixes = _layer_fixes(objs, cases)
    slices = _category_slices(objs, cases)
    return {"composition": composition, "backends": reports, "fixes": fixes,
            "category_slices": slices}


# Categories worth reporting as their own slice (the ASR/realism slice is the
# one the conversational/spoken training is meant to move).
_SLICE_CATEGORIES = ["conversational"]


def _category_slices(objs: Dict[str, BackendReport],
                     cases: List[Case]) -> dict:
    """Re-score mis-route / resolved / adult-leak over a single category subset.

    Uses each backend's stored per-utterance predictions (so no re-inference),
    re-running the SAME metric accounting on the filtered case list — the honest
    way to see whether a slice (e.g. ``conversational``) moved without diluting
    it into the overall number.
    """
    out: Dict[str, dict] = {}
    for cat in _SLICE_CATEGORIES:
        sub = [c for c in cases if c.category == cat]
        if not sub:
            continue
        per_backend: Dict[str, dict] = {}
        for name, rep in objs.items():
            mt = AxisCount()
            resolved_total = resolved_n = 0
            adult_total = adult_leak = 0
            for c in sub:
                p = rep.preds.get(c.utterance)
                if p is None:
                    continue
                if c.explicit:
                    adult_total += 1
                    if not p.adult:
                        adult_leak += 1
                if c.domain == "ocp_play" and c.media_type not in ABSTAIN_TYPES:
                    if c.abstain_ok:
                        resolved_total += 1
                        if (p.media_type not in ABSTAIN_TYPES
                                and p.media_type == c.media_type):
                            resolved_n += 1
                    if p.media_type in ABSTAIN_TYPES:
                        mt.add_abstain()
                    elif p.media_type == c.media_type:
                        mt.add_correct()
                    else:
                        mt.add_wrong()
            per_backend[name] = {
                "mis_route_rate": round(mt.mis_route_rate, 4),
                "confident_wrong": mt.confident_wrong,
                "media_n": mt.n,
                "abstain": mt.abstain,
                "resolved_rate": round(
                    resolved_n / resolved_total if resolved_total else 0.0, 4),
                "resolved_n": resolved_n,
                "resolved_total": resolved_total,
                "adult_leak_rate": round(
                    adult_leak / adult_total if adult_total else 0.0, 4),
                "adult_leak_n": adult_leak,
                "adult_total": adult_total,
            }
        out[cat] = {"n_cases": len(sub), "backends": per_backend}
    return out


# Which open-vocab cases each layer CLOSES relative to the cheaper layer.  A case
# is "closed" when the cheaper layer abstained/mis-routed it and the richer layer
# routes it correctly (a non-GENERIC media_type matching the expected one).
_LAYER_CHAIN = ["keyword", "hybrid", "hybrid+gazetteer", "hybrid+gazetteer+online"]


def _correct(p: "Pred", c: Case) -> bool:
    return (c.domain == "ocp_play"
            and c.media_type not in ABSTAIN_TYPES
            and p.media_type == c.media_type)


def _layer_fixes(objs: Dict[str, BackendReport], cases: List[Case]) -> dict:
    """For each adjacent layer pair, list the play-cases the richer one fixes."""
    # resolve each layer label to the actual report (bundle-suffixed names)
    def _find(prefix: str) -> Optional[BackendReport]:
        if prefix == "keyword":
            return objs.get("keyword")
        for name, rep in objs.items():
            base = name.split(":", 1)[0]
            if base == prefix:
                return rep
        return None

    by_utt = {c.utterance: c for c in cases}
    out: Dict[str, List[dict]] = {}
    for prev, cur in zip(_LAYER_CHAIN, _LAYER_CHAIN[1:]):
        rp, rc = _find(prev), _find(cur)
        if rp is None or rc is None:
            continue
        fixed: List[dict] = []
        for utt, c in by_utt.items():
            pp, pc = rp.preds.get(utt), rc.preds.get(utt)
            if pp is None or pc is None:
                continue
            if not _correct(pp, c) and _correct(pc, c):
                fixed.append({"utterance": utt, "expected": c.media_type,
                              "was": pp.media_type, "now": pc.media_type,
                              "category": c.category, "note": c.note})
        out[f"{prev} -> {cur}"] = fixed
    return out


def _composition(cases: List[Case]) -> dict:
    import collections
    return {
        "total": len(cases),
        "by_lang": dict(collections.Counter(c.lang for c in cases)),
        "by_category": dict(collections.Counter(c.category for c in cases)),
        "by_domain": dict(collections.Counter(c.domain for c in cases)),
        "explicit_adult": sum(1 for c in cases if c.explicit),
        "abstain_ok": sum(1 for c in cases if c.abstain_ok),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_json(report: dict, path: str = RESULTS_JSON) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return path


def write_md(report: dict, path: str = RESULTS_MD) -> str:
    comp = report["composition"]
    lines: List[str] = []
    lines.append("# ovos-media-classifier — harm-weighted routing eval\n")
    lines.append(
        f"Out-of-distribution, hand-curated eval: **{comp['total']} cases** "
        f"across {len(comp['by_lang'])} languages "
        f"({', '.join(f'{k}={v}' for k, v in sorted(comp['by_lang'].items()))}); "
        f"**{comp['explicit_adult']} adult** cases, "
        f"**{comp['abstain_ok']} abstain-ok** cases.\n"
    )
    lines.append("Composition by category: "
                 + ", ".join(f"{k}={v}" for k, v in sorted(comp["by_category"].items()))
                 + ".\n")
    lines.append(
        "**The metric is mis-route rate (confident-wrong), and GENERIC/abstain "
        "is scored as SAFE.** A confident-wrong route prunes the correct "
        "provider (harm); an abstain still lets every provider search "
        "(harmless).\n")

    lines.append("## Headline\n")
    lines.append(
        "**mis-route** is the harm (GENERIC=safe); **resolved** is the open-vocab "
        "win (abstain_ok cases turned into the CORRECT confident route). A clean "
        "win over a cheaper layer is mis-route NOT WORSE *and* resolved HIGHER.\n")
    lines.append("| backend | mis-route | resolved | adult-leak | false-hijack | false-miss | control recall | latency med/p95 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, b in report["backends"].items():
        if b.get("status") != "available":
            lines.append(f"| {name} | unavailable | – | – | – | – | – | – |")
            continue
        g = b["gate"]; a = b["adult_leak"]; c = b["control"]
        rz = b.get("resolved", {"rate": 0.0, "resolved_n": 0, "total": 0})
        lat = b.get("latency")
        lat_s = (f"{lat['median_ms']:.0f}/{lat['p95_ms']:.0f} ms"
                 if lat else "–")
        lines.append(
            f"| {name} | **{b['mis_route_rate']:.3f}** "
            f"({b['media_type']['confident_wrong']}/{b['media_type']['n']}) | "
            f"**{rz['rate']:.3f}** ({rz['resolved_n']}/{rz['total']}) | "
            f"**{a['leak_rate']:.3f}** ({a['leak_n']}/{a['adult_total']}) | "
            f"{g['false_hijack_rate']:.3f} ({g['false_hijack_n']}/{g['false_hijack_total']}) | "
            f"{g['false_miss_rate']:.3f} ({g['false_miss_n']}/{g['false_miss_total']}) | "
            f"{c['recall']:.3f} ({c['hit_n']}/{c['total']}) | {lat_s} |"
        )
    lines.append("")

    # ---- per-category slices (the conversational / ASR-realism slice) -------
    slices = report.get("category_slices") or {}
    for cat, sl in slices.items():
        lines.append(f"## Slice: {cat} ({sl['n_cases']} cases)\n")
        lines.append(
            "Mis-route / resolved / adult-leak scored over ONLY this category "
            "(the realism slice the conversational/ASR training targets).\n")
        lines.append("| backend | mis-route | resolved | adult-leak |")
        lines.append("|---|---|---|---|")
        for name, b in sl["backends"].items():
            lines.append(
                f"| {name} | **{b['mis_route_rate']:.3f}** "
                f"({b['confident_wrong']}/{b['media_n']}) | "
                f"**{b['resolved_rate']:.3f}** "
                f"({b['resolved_n']}/{b['resolved_total']}) | "
                f"**{b['adult_leak_rate']:.3f}** "
                f"({b['adult_leak_n']}/{b['adult_total']}) |")
        lines.append("")

    # ---- per-layer open-vocab fixes -----------------------------------------
    fixes = report.get("fixes") or {}
    if any(fixes.values()):
        lines.append("## Open-vocab cases each layer closes\n")
        lines.append(
            "Each row is a play-case the cheaper layer abstained on or "
            "mis-routed that the richer layer routes correctly.\n")
        for pair, items in fixes.items():
            if not items:
                continue
            lines.append(f"### {pair}  ({len(items)} fixed)\n")
            lines.append("| utterance | expected | was | now |")
            lines.append("|---|---|---|---|")
            for it in items:
                lines.append(f"| {it['utterance']} | {it['expected']} | "
                             f"{it['was']} | **{it['now']}** |")
            lines.append("")

    lines.append("## Routing axes — confident-wrong vs abstain (safe)\n")
    lines.append("| backend | media_type wrong | media_type abstain | playback wrong | playback abstain | adult over-flag |")
    lines.append("|---|---|---|---|---|---|")
    for name, b in report["backends"].items():
        if b.get("status") != "available":
            lines.append(f"| {name} | – | – | – | – | – |")
            continue
        mt = b["media_type"]; pb = b["playback_type"]; a = b["adult_leak"]
        lines.append(
            f"| {name} | {mt['mis_route_rate']:.3f} ({mt['confident_wrong']}/{mt['n']}) | "
            f"{mt['abstain_rate']:.3f} ({mt['abstain']}/{mt['n']}) | "
            f"{pb['mis_route_rate']:.3f} ({pb['confident_wrong']}/{pb['n']}) | "
            f"{pb['abstain_rate']:.3f} ({pb['abstain']}/{pb['n']}) | "
            f"{a['overflag_rate']:.3f} ({a['overflag_n']}/{a['clean_total']}) |"
        )
    lines.append("")

    unavailable = [n for n, b in report["backends"].items()
                   if b.get("status") != "available"]
    if unavailable:
        lines.append("## Unavailable backends\n")
        for n in unavailable:
            lines.append(f"- **{n}**: {report['backends'][n]['reason']}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def gazetteer_latency_sweep(bundle_dir: str,
                            sizes: Optional[List[int]] = None) -> dict:
    """Per-query routing latency as a function of gazetteer size (entities/type).

    The entity matcher is word-boundary regex whose cost grows with the number
    of injected phrases, so a LIVE gazetteer must stay bounded.  This sweeps
    ``register_default_gazetteer(top_n=N)`` over *sizes*, routes every eval
    utterance, and reports median / p95 ms per query at each size — so the
    default cap can be picked at the latency knee.  ``0`` = no gazetteer baseline.
    """
    import time

    from ovos_media_classifier.embedding import HybridMediaClassifier

    sizes = sizes or [0, 100, 500, 1000, 5000, 10000, 50000, 100000]
    cases = load_cases()
    rows = {}
    for n in sizes:
        clf = HybridMediaClassifier.from_path(bundle_dir)
        injected = 0
        if n > 0:
            injected = clf.register_default_gazetteer(top_n=n)
        lat = []
        for c in cases:
            t0 = time.perf_counter()
            predict(clf, c)
            lat.append((time.perf_counter() - t0) * 1000.0)
        lat.sort()
        k = len(lat)
        med = lat[k // 2] if k % 2 else (lat[k // 2 - 1] + lat[k // 2]) / 2
        p95 = lat[min(k - 1, int(round(0.95 * (k - 1))))]
        rows[n] = {"cap_per_type": n, "injected_titles": injected,
                   "median_ms": round(med, 3), "p95_ms": round(p95, 3),
                   "max_ms": round(lat[-1], 3)}
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", action="append", default=[],
                    help="path to an extra ONNX bundle dir to evaluate")
    ap.add_argument("--only-keyword", action="store_true",
                    help="skip ONNX bundles (keyword backend only)")
    ap.add_argument("--online", action="store_true",
                    help="evaluate the online metadatarr layer (makes NETWORK "
                         "calls; reports per-query latency)")
    ap.add_argument("--latency-sweep", action="store_true",
                    help="report routing latency vs gazetteer size and exit")
    args = ap.parse_args(argv)

    if args.latency_sweep:
        bundles = discover_router_bundles()
        if not bundles:
            print("no router bundle found for the latency sweep")
            return 1
        bundle = sorted(bundles.values())[0]
        rows = gazetteer_latency_sweep(bundle)
        print(f"gazetteer latency sweep (bundle={bundle}, {len(load_cases())} queries):")
        print(f"  {'cap/type':>10} {'titles':>8} {'median ms':>10} {'p95 ms':>8} {'max ms':>8}")
        for n, r in rows.items():
            print(f"  {r['cap_per_type']:>10} {r['injected_titles']:>8} "
                  f"{r['median_ms']:>10.3f} {r['p95_ms']:>8.3f} {r['max_ms']:>8.3f}")
        return 0

    report = run(extra_bundles=args.bundle, only_keyword=args.only_keyword,
                 online=args.online)
    jp = write_json(report)
    mp = write_md(report)

    comp = report["composition"]
    print(f"cases: {comp['total']} ({comp['explicit_adult']} adult, "
          f"{comp['abstain_ok']} abstain-ok) langs={comp['by_lang']}")
    for name, b in report["backends"].items():
        if b.get("status") != "available":
            print(f"  {name:28s} UNAVAILABLE: {b['reason']}")
            continue
        rz = b.get("resolved", {"rate": 0.0})
        print(f"  {name:28s} mis-route={b['mis_route_rate']:.3f} "
              f"resolved={rz['rate']:.3f} "
              f"adult-leak={b['adult_leak']['leak_rate']:.3f} "
              f"hijack={b['gate']['false_hijack_rate']:.3f} "
              f"miss={b['gate']['false_miss_rate']:.3f} "
              f"ctrl={b['control']['recall']:.3f}")
    for cat, sl in (report.get("category_slices") or {}).items():
        print(f"\n[slice: {cat}] ({sl['n_cases']} cases)")
        for name, b in sl["backends"].items():
            print(f"  {name:28s} mis-route={b['mis_route_rate']:.3f} "
                  f"resolved={b['resolved_rate']:.3f} "
                  f"adult-leak={b['adult_leak_rate']:.3f}")
    print(f"wrote {jp}\nwrote {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
