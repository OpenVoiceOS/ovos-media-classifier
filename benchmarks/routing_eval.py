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
    n: int = 0

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
        }


def evaluate(clf, cases: List[Case], name: str) -> BackendReport:
    rep = BackendReport(name=name)
    rep.n = len(cases)

    for c in cases:
        p = predict(clf, c)

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
        only_keyword: bool = False) -> Dict[str, object]:
    cases = load_cases()

    loaders: Dict[str, Callable] = {"keyword": _load_keyword}
    if not only_keyword:
        for label, path in discover_bundles().items():
            loaders[label] = _onnx_loader(path)
        for path in (extra_bundles or []):
            loaders[f"onnx:{os.path.basename(path.rstrip('/'))}"] = _onnx_loader(path)

    reports: Dict[str, dict] = {}
    for name, loader in loaders.items():
        try:
            clf = loader()
        except Exception as e:  # noqa: BLE001 - record, never crash
            rep = BackendReport(name=name, status="unavailable",
                                reason=f"{type(e).__name__}: {e}")
            reports[name] = rep.as_dict()
            continue
        reports[name] = evaluate(clf, cases, name).as_dict()

    composition = _composition(cases)
    return {"composition": composition, "backends": reports}


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
    lines.append("| backend | mis-route | adult-leak | false-hijack | false-miss | control recall |")
    lines.append("|---|---|---|---|---|---|")
    for name, b in report["backends"].items():
        if b.get("status") != "available":
            lines.append(f"| {name} | unavailable | – | – | – | – |")
            continue
        g = b["gate"]; a = b["adult_leak"]; c = b["control"]
        lines.append(
            f"| {name} | **{b['mis_route_rate']:.3f}** "
            f"({b['media_type']['confident_wrong']}/{b['media_type']['n']}) | "
            f"**{a['leak_rate']:.3f}** ({a['leak_n']}/{a['adult_total']}) | "
            f"{g['false_hijack_rate']:.3f} ({g['false_hijack_n']}/{g['false_hijack_total']}) | "
            f"{g['false_miss_rate']:.3f} ({g['false_miss_n']}/{g['false_miss_total']}) | "
            f"{c['recall']:.3f} ({c['hit_n']}/{c['total']}) |"
        )
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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", action="append", default=[],
                    help="path to an extra ONNX bundle dir to evaluate")
    ap.add_argument("--only-keyword", action="store_true",
                    help="skip ONNX bundles (keyword backend only)")
    args = ap.parse_args(argv)

    report = run(extra_bundles=args.bundle, only_keyword=args.only_keyword)
    jp = write_json(report)
    mp = write_md(report)

    comp = report["composition"]
    print(f"cases: {comp['total']} ({comp['explicit_adult']} adult, "
          f"{comp['abstain_ok']} abstain-ok) langs={comp['by_lang']}")
    for name, b in report["backends"].items():
        if b.get("status") != "available":
            print(f"  {name:28s} UNAVAILABLE: {b['reason']}")
            continue
        print(f"  {name:28s} mis-route={b['mis_route_rate']:.3f} "
              f"adult-leak={b['adult_leak']['leak_rate']:.3f} "
              f"hijack={b['gate']['false_hijack_rate']:.3f} "
              f"miss={b['gate']['false_miss_rate']:.3f} "
              f"ctrl={b['control']['recall']:.3f}")
    print(f"wrote {jp}\nwrote {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
