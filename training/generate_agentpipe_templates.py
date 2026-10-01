"""Agentpipe-synthesized conversational ``.intent`` TEMPLATE GRAMMAR.

This is the *template* sibling of :mod:`training.generate_agentpipe`.  Where that
module asks a free agent for finished *utterances*, this one asks for new
**OVOS-INTENT-1 template lines** — grammar with live ``{slot}`` placeholders,
``(a|b|c)`` alternations, ``[optional]`` optionals and ``<voc>`` references — so
the surviving lines drop straight into
``ovos_media_classifier/locale/en-us/dataset/<intent>.intent`` and feed the same
``expand()`` + slot-fill pipeline as the hand-written templates.

The goal is **conversational register** in the templates themselves
(``hey``/``um``/``wanna``/``lemme``/``i'm in the mood for``/``how about`` /
self-repair), not bare ``<lead> {slot}`` skeletons.

Providers: **FREE AGENTS ONLY** — never claude or any paid model (hard rule).
Bulk synthesis uses ``opencode-free`` with ``kilo`` as the only fallback, mirroring
:mod:`training.generate_agentpipe` (provider rotation, concurrency, ``.done.jsonl``
checkpoint/resume).

RIGOROUS validation drops anything that fails (per-line, logged with the reason):

* ``(a)`` ``ovos_spec_tools.expand(line, vocs)`` must succeed (no error), where
  ``vocs`` is the same ``lead_*`` voc set ``build_dataset`` loads;
* ``(b)`` every ``{slot}`` must be an ALLOWED slot for that intent — the slots that
  already appear in the intent's existing template file (no invented slots);
* ``(c)`` the line must carry a routing cue: a ``<lead_*>`` reference OR at least
  one media keyword from the intent's keyword vocab(s);
* ``(d)`` it must be unique vs the existing *expanded* template set (dedup against
  what is already in the file, expanded).

Surviving lines are appended to the per-intent ``.intent`` file.  en-us only;
other languages come later via ovos-localize.

Usage::

    # smoke test (1-2 intents, small N)
    python -m training.generate_agentpipe_templates \
        --intents movie,music --n 12 --concurrency 2

    # full run (all 34 media intents)
    python -m training.generate_agentpipe_templates --n 40 --concurrency 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ovos_spec_tools import expand, find_lang_dir

# ---------------------------------------------------------------------------
# locale layout (mirror training.build_dataset)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(os.path.dirname(_HERE), "ovos_media_classifier")
LOCALE_DIR = os.path.join(_PKG, "locale")
DATASET_TEMPLATE_SUBDIR = "dataset"
LANG = "en-us"  # en-us only; other langs handled later by ovos-localize

# FREE AGENTS ONLY — never claude. (See training/generate_agentpipe.py.)
FREE_PROVIDERS = ["opencode-free", "kilo"]
FREE_FALLBACKS = ["opencode-free", "kilo"]
AGENT_TIMEOUT = 120

# ---------------------------------------------------------------------------
# Human-readable intent descriptions (extends generate_agentpipe.INTENT_DESC to
# cover every dataset/*.intent media label).
# ---------------------------------------------------------------------------
INTENT_DESC: Dict[str, str] = {
    "music": "play a song, album, artist or genre of music",
    "podcast": "play a podcast or podcast episode",
    "radio": "tune in to a live radio station",
    "audiobook": "play an audiobook",
    "news": "play the news",
    "radio_theatre": "play an audio drama / radio play",
    "asmr": "play ASMR audio",
    "audio_description": "watch the audio-described version of a film",
    "adult_audio": "play adult (NSFW) audio — used ONLY to train a content filter "
                   "that BLOCKS such requests",
    "audio": "play some unspecified audio",
    "ambient": "play ambient / white-noise / focus / sleep sounds",
    "playlist": "play a mood- or activity-based playlist",
    "sound_effect": "play a single sound effect",
    "movie": "watch a film / movie",
    "tv": "watch a live TV channel",
    "tv_show": "watch an episodic TV series",
    "anime": "watch anime",
    "cartoon": "watch a cartoon / animated show",
    "documentary": "watch a documentary",
    "short_film": "watch a short film",
    "silent_movie": "watch a silent movie",
    "bw_movie": "watch a black-and-white movie",
    "hentai": "watch adult anime (NSFW) — used ONLY to train a content filter that "
              "BLOCKS such requests",
    "adult": "watch adult (NSFW) video — used ONLY to train a content filter that "
             "BLOCKS such requests",
    "video": "watch a generic online video",
    "video_episodes": "watch an online video series / web series",
    "visual_story": "read / watch a motion comic or visual story",
    "game": "play a video game",
    "interactive_fiction": "play a text adventure / interactive fiction game",
    "music_video": "watch an official music video",
    "trailer": "watch a movie/show trailer",
    "behind_the_scenes": "watch behind-the-scenes / making-of footage",
    "book": "read a book / novel / ebook",
    "comic": "read a comic / manga / graphic novel",
}

# intent -> the media KEYWORD vocab file(s) carrying the routing cue.  Used both to
# tell the agent the cue words it MUST include and to validate that a line carries
# a cue (validation rule c).  Lead-in vocs (<lead_*>) also satisfy the cue.
INTENT_KEYWORD_VOCS: Dict[str, List[str]] = {
    "music": ["MusicKeyword"],
    "podcast": ["PodcastKeyword"],
    "radio": ["RadioKeyword"],
    "audiobook": ["AudioBookKeyword"],
    "news": ["NewsKeyword"],
    "radio_theatre": ["AudioDramaKeyword"],
    "asmr": ["ASMRKeyword"],
    "audio_description": ["ADKeyword"],
    "adult_audio": ["AdultKeyword", "AudioKeyword"],
    "audio": ["AudioKeyword"],
    "ambient": ["AmbientKeyword"],
    "playlist": ["PlaylistKeyword"],
    "sound_effect": ["SoundEffectKeyword"],
    "movie": ["MovieKeyword"],
    "tv": ["TVKeyword", "IPTVKeyword"],
    "tv_show": ["SeriesKeyword"],
    "anime": ["AnimeKeyword"],
    "cartoon": ["CartoonKeyword"],
    "documentary": ["DocumentaryKeyword"],
    "short_film": ["ShortKeyword"],
    "silent_movie": ["SilentKeyword"],
    "bw_movie": ["BWKeyword"],
    "hentai": ["HentaiKeyword"],
    "adult": ["AdultKeyword"],
    "video": ["VideoKeyword"],
    "video_episodes": ["SeriesKeyword", "VideoKeyword"],
    "visual_story": ["ComicBookKeyword"],
    "game": ["GameKeyword"],
    "interactive_fiction": ["InteractiveFictionKeyword"],
    "music_video": ["MusicVideoKeyword"],
    "trailer": ["TrailerKeyword"],
    "behind_the_scenes": ["BehindTheScenesKeyword"],
    "book": ["BookKeyword"],
    "comic": ["ComicBookKeyword"],
}

ADULT_INTENTS = {"adult", "adult_audio", "hentai"}

PROMPT_STYLES: List[str] = [
    "Lean into casual everyday speech: contractions, fillers ('um', 'hey', 'so'), "
    "'wanna'/'gonna'/'lemme'/'gimme'.",
    "Use indirect/contextual openings ('i'm in the mood for', 'how about', "
    "'let's', 'we should', 'feels like a ... night').",
    "Include some self-repair / false starts (e.g. 'play X, actually no Y', "
    "'put on X — wait, something with {slot}').",
    "Frame several as casual questions ('got any', 'do you have', 'is there', "
    "'what about', 'can you').",
    "Vary speaker register: a bored teenager, a tired parent, an excited fan — "
    "keep it spoken, never formal.",
]

_SLOT_RE = re.compile(r"\{([a-z0-9_]+\}?)")  # tolerant; cleaned below
_SLOT_OK_RE = re.compile(r"\{([a-z0-9_]+)\}")
_LEAD_RE = re.compile(r"<(lead_[a-z0-9_]+)>")
_VOC_RE = re.compile(r"<([a-z0-9_]+)>")
_JSON_ARRAY = re.compile(r"\[.*?\]", re.DOTALL)


# ---------------------------------------------------------------------------
# locale loading
# ---------------------------------------------------------------------------
def _intent_path(intent: str) -> str:
    return os.path.join(LOCALE_DIR, LANG, DATASET_TEMPLATE_SUBDIR, f"{intent}.intent")


def load_leadin_vocabs() -> Dict[str, Sequence[str]]:
    """The shared ``lead_*`` voc members, exactly as build_dataset loads them."""
    vocs: Dict[str, Sequence[str]] = {}
    lang_dir = find_lang_dir(LOCALE_DIR, LANG)
    if lang_dir is None:
        return vocs
    for path in sorted(lang_dir.glob("lead_*.voc")):
        members = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                   if ln.strip()]
        if members:
            vocs[path.stem] = members
    return vocs


def load_voc_members(voc_name: str) -> List[str]:
    """Members of a named ``.voc`` at the en-us locale root (lowercased)."""
    lang_dir = find_lang_dir(LOCALE_DIR, LANG)
    if lang_dir is None:
        return []
    path = lang_dir / f"{voc_name}.voc"
    if not path.is_file():
        return []
    return [ln.strip().lower() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def load_intent_lines(intent: str) -> List[str]:
    path = _intent_path(intent)
    if not os.path.isfile(path):
        return []
    return [ln.strip() for ln in open(path, encoding="utf-8").read().splitlines()
            if ln.strip()]


def list_media_intents() -> List[str]:
    """All ``dataset/*.intent`` media labels (the 34 intents)."""
    d = os.path.join(LOCALE_DIR, LANG, DATASET_TEMPLATE_SUBDIR)
    return sorted(fn[:-len(".intent")] for fn in os.listdir(d)
                  if fn.endswith(".intent"))


def slots_in(line: str) -> Set[str]:
    return set(_SLOT_OK_RE.findall(line))


def leads_in(line: str) -> Set[str]:
    return set(_LEAD_RE.findall(line))


def allowed_slots_for(intent: str) -> Set[str]:
    """Allowed slots = those already used in the intent's existing template file.

    This is the hard constraint that guarantees the agent cannot invent slots and
    that every fill resolves to a real entity pool.
    """
    allowed: Set[str] = set()
    for line in load_intent_lines(intent):
        allowed |= slots_in(line)
    return allowed


def leads_for(intent: str) -> List[str]:
    leads: Set[str] = set()
    for line in load_intent_lines(intent):
        leads |= leads_in(line)
    return sorted(leads)


def signal_slots_for(intent: str, keyword_set: Set[str]) -> Set[str]:
    """Slots that the EXISTING corpus treats as a standalone routing cue.

    Calibrated from the hand-written templates: any slot that appears in an
    existing line carrying *neither* a ``<lead_*>`` ref *nor* a literal keyword
    is, by the maintainers' own design, sufficient signal on its own (e.g.
    ``{movie_title}`` in ``"wanna watch {movie_title}"``).  Accepting these keeps
    rule (c) aligned with the accepted corpus instead of rejecting exactly the
    conversational register we want.
    """
    sig: Set[str] = set()
    for line in load_intent_lines(intent):
        if leads_in(line):
            continue
        if _line_has_keyword(line, keyword_set):
            continue
        sig |= slots_in(line)
    return sig


def _line_has_keyword(line: str, keyword_set: Set[str]) -> bool:
    """Whole-word, punctuation- and inflection-tolerant keyword presence test.

    The final word of a keyword may carry a light English inflection
    (``drama``→``dramas``, ``adventure``→``adventures``, ``described``) so a
    conversational surface still counts as carrying the cue.
    """
    if not keyword_set:
        return False
    low = line.lower()
    for kw in keyword_set:
        pat = (r"(?<![a-z0-9])" + re.escape(kw)
               + r"(?:s|es|d|ed|ing|ies)?(?![a-z0-9])")
        if re.search(pat, low):
            return True
    return False


def expanded_set(lines: List[str], vocs: Dict[str, Sequence[str]]) -> Set[str]:
    """Expanded, slot-bearing forms of ``lines`` (lowercased) for dedup."""
    out: Set[str] = set()
    for ln in lines:
        try:
            for e in expand(ln, vocs):
                out.add(" ".join(e.lower().split()))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------
def build_prompt(intent: str, n: int, allowed_slots: Sequence[str],
                 leads: Sequence[str], keyword_words: Sequence[str],
                 examples: Sequence[str], style: str) -> str:
    desc = INTENT_DESC.get(intent, intent.replace("_", " "))
    slot_list = ", ".join("{%s}" % s for s in allowed_slots) or "(none)"
    lead_list = ", ".join("<%s>" % l for l in leads) or "(none)"
    kw = ", ".join(keyword_words[:24]) or "(none)"
    ex = "\n".join(f"  {e}" for e in examples)
    nsfw = ""
    if intent in ADULT_INTENTS:
        nsfw = (
            " NOTE: these explicit-intent phrasings exist ONLY to train a safety "
            "classifier that BLOCKS such requests; keep them realistic but no "
            "graphic detail is required.")
    return (
        "You write GRAMMAR TEMPLATES (not finished sentences) for an OVOS "
        "voice-assistant intent dataset. Each template is ONE line of OVOS-INTENT-1 "
        "syntax that, when expanded and slot-filled, becomes a spoken command to: "
        f"{desc}.{nsfw}\n\n"
        "OVOS-INTENT-1 template syntax (use these, nothing else):\n"
        "  {slot}        a placeholder filled later with a real entity. You MUST "
        "use ONLY these exact slot names, verbatim, and you may use ZERO or MORE "
        f"of them per line: {slot_list}\n"
        "  (a|b|c)       a required choice between alternatives\n"
        "  [optional]    an optional chunk that may be dropped\n"
        f"  <voc>         a reference to a shared lead-in vocabulary. Allowed: "
        f"{lead_list}\n\n"
        "HARD REQUIREMENTS for every line:\n"
        "  1. It MUST carry the routing cue for this media type: either start with "
        f"one of the allowed <lead_*> references above, OR include one of these "
        f"keyword words literally: {kw}.\n"
        "  2. Use ONLY the slot names listed above. Never invent a slot. If a slot "
        "you want is not listed, do not use it.\n"
        "  3. Make it CONVERSATIONAL and spoken: contractions, fillers (hey, um, "
        "so), 'wanna'/'lemme'/'gimme', indirect openers ('i'm in the mood for', "
        "'how about', 'let's', 'got any'), and occasional self-repair. Vary the "
        "opening across lines. NOT stiff/formal, NOT a bare '<lead> {slot}' "
        "skeleton.\n"
        f"  4. {style}\n"
        "  5. Keep slot placeholders intact as literal {curly} text; do NOT fill "
        "them with example values.\n\n"
        "Existing templates for this intent (match this STYLE and slot usage, but "
        "produce NEW distinct lines):\n"
        f"{ex}\n\n"
        f"Return ONLY a JSON array of exactly {n} template strings. No prose, no "
        "markdown, no code fences."
    )


def parse_templates(raw: str) -> List[str]:
    """Extract a JSON string array (tolerates fences / prose), no newline fallback."""
    if not raw:
        return []
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    candidates = [text]
    m = _JSON_ARRAY.search(text)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            data = json.loads(cand)
        except Exception:
            continue
        if isinstance(data, list):
            out = [str(x).strip() for x in data if isinstance(x, str)]
            return [u for u in out if u]
    return []


async def _gen_one(base_prov: str, prompt: str, fallbacks: List[str]) -> List[str]:
    from agentpipe import Agent
    for prov in [base_prov] + fallbacks:
        try:
            ag = Agent(prov, timeout=AGENT_TIMEOUT)
            raw = await ag.generate(prompt)
            tpls = parse_templates(raw)
            if tpls:
                return tpls
        except Exception:
            continue
    return []


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
class DropReason:
    EXPAND = "expand_error"
    BAD_SLOT = "invalid_slot"
    NO_CUE = "no_media_cue"
    DUP = "duplicate"
    BAD_VOC = "unknown_voc_ref"
    EMPTY = "empty"


def validate_line(line: str, intent: str, allowed_slots: Set[str],
                  lead_vocs: Dict[str, Sequence[str]],
                  keyword_set: Set[str], signal_slots: Set[str],
                  existing_expanded: Set[str],
                  ) -> Tuple[bool, Optional[str], List[str]]:
    """Return ``(ok, drop_reason, new_expanded_forms)``."""
    line = line.strip()
    if not line:
        return False, DropReason.EMPTY, []

    line_slots = slots_in(line)
    # (b) slots must all be allowed (no invented slots)
    for s in line_slots:
        if s not in allowed_slots:
            return False, DropReason.BAD_SLOT, []

    # any <voc> reference must be a known lead voc (we only ship lead_* vocs to
    # expand(); an unknown <voc> would also make expand raise, but check eagerly
    # for a precise reason)
    for v in _VOC_RE.findall(line):
        if v not in lead_vocs:
            return False, DropReason.BAD_VOC, []

    # (a) expand must succeed
    try:
        forms = expand(line, lead_vocs)
    except Exception:
        return False, DropReason.EXPAND, []
    if not forms:
        return False, DropReason.EXPAND, []

    # (c) routing cue: a <lead_*> ref, OR a literal media keyword, OR a
    # type-identifying "signal" slot the existing corpus treats as cue enough.
    has_lead = bool(leads_in(line))
    has_kw = _line_has_keyword(line, keyword_set)
    has_signal = bool(line_slots & signal_slots)
    if not (has_lead or has_kw or has_signal):
        return False, DropReason.NO_CUE, []

    # (d) unique vs existing expanded template set
    norm = [" ".join(f.lower().split()) for f in forms]
    if any(n in existing_expanded for n in norm):
        return False, DropReason.DUP, []

    return True, None, norm


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _done_key(intent: str) -> str:
    return f"{LANG}|{intent}"


async def generate(intents: List[str], n_per: int, chunk: int, provider: str,
                   concurrency: int, seed: int, dry_run: bool,
                   checkpoint: str) -> None:
    import random
    rnd = random.Random(seed)

    lead_vocs = load_leadin_vocabs()

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint)), exist_ok=True)
    done: Set[str] = set()
    if os.path.isfile(checkpoint):
        with open(checkpoint, encoding="utf-8") as fh:
            for ln in fh:
                done.add(ln.strip())

    prov_pool = []
    seen_p: Set[str] = set()
    for p in [provider] + FREE_PROVIDERS:
        if p not in seen_p:
            seen_p.add(p)
            prov_pool.append(p)
    print(f"  rotation pool: {prov_pool}; fallbacks: {FREE_FALLBACKS}", flush=True)

    sem = asyncio.Semaphore(concurrency)
    totals = {"gen": 0, "kept": 0, "dropped": 0}

    async def run_intent(idx: int, intent: str) -> None:
        key = _done_key(intent)
        if key in done:
            print(f"[skip] {intent}: already done", flush=True)
            return
        allowed = allowed_slots_for(intent)
        leads = leads_for(intent)
        kw_words: List[str] = []
        for voc in INTENT_KEYWORD_VOCS.get(intent, []):
            kw_words.extend(load_voc_members(voc))
        keyword_set = set(kw_words)
        signal_slots = signal_slots_for(intent, keyword_set)
        existing_lines = load_intent_lines(intent)
        existing_expanded = expanded_set(existing_lines, lead_vocs)
        existing_raw = {l.strip() for l in existing_lines}

        # request in chunks for diversity/robustness
        n_chunks = max(1, (n_per + chunk - 1) // chunk)
        raw_lines: List[str] = []
        for ci in range(n_chunks):
            want = min(chunk, n_per - ci * chunk)
            if want <= 0:
                break
            ex = rnd.sample(existing_lines, min(8, len(existing_lines))) \
                if existing_lines else []
            style = PROMPT_STYLES[ci % len(PROMPT_STYLES)]
            base_prov = prov_pool[(idx + ci) % len(prov_pool)]
            fallbacks = [p for p in FREE_FALLBACKS if p != base_prov]
            prompt = build_prompt(intent, want, sorted(allowed), leads,
                                  kw_words, ex, style)
            async with sem:
                got = await _gen_one(base_prov, prompt, fallbacks)
            raw_lines.extend(got)

        # validate + dedup (against existing AND within this batch)
        kept: List[str] = []
        drops: Dict[str, int] = {}
        seen_batch = set(existing_expanded)
        seen_raw = set(existing_raw)
        for ln in raw_lines:
            ln = ln.strip()
            if ln in seen_raw:
                drops[DropReason.DUP] = drops.get(DropReason.DUP, 0) + 1
                continue
            ok, reason, forms = validate_line(
                ln, intent, allowed, lead_vocs, keyword_set, signal_slots,
                seen_batch)
            if not ok:
                drops[reason or "unknown"] = drops.get(reason or "unknown", 0) + 1
                continue
            kept.append(ln)
            seen_raw.add(ln)
            for f in forms:
                seen_batch.add(f)

        gen = len(raw_lines)
        dropped = sum(drops.values())
        totals["gen"] += gen
        totals["kept"] += len(kept)
        totals["dropped"] += dropped
        drop_str = ", ".join(f"{k}={v}" for k, v in sorted(drops.items())) or "-"
        print(f"[{intent}] generated={gen} kept={len(kept)} dropped={dropped} "
              f"({drop_str})", flush=True)

        if kept and not dry_run:
            path = _intent_path(intent)
            with open(path, "a", encoding="utf-8") as fh:
                for ln in kept:
                    fh.write(ln + "\n")
            with open(checkpoint, "a", encoding="utf-8") as fh:
                fh.write(key + "\n")
            done.add(key)
        elif not kept and not dry_run:
            # nothing survived — do NOT checkpoint so a resume retries the intent
            pass

    async def safe(i: int, it: str) -> None:
        try:
            await run_intent(i, it)
        except Exception as exc:  # never let one intent kill the batch
            print(f"[{it}] ERROR: {exc!r}", flush=True)

    await asyncio.gather(*(safe(i, it) for i, it in enumerate(intents)))
    print(f"\nTOTAL generated={totals['gen']} kept={totals['kept']} "
          f"dropped={totals['dropped']}"
          + (" (dry-run: nothing written)" if dry_run else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intents", default="",
                    help="comma-separated intents (default: all 34 media intents)")
    ap.add_argument("--n", type=int, default=40,
                    help="template lines requested per intent")
    ap.add_argument("--chunk", type=int, default=20,
                    help="lines requested per LLM call")
    ap.add_argument("--provider", default="opencode-free")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="generate + validate + log but do NOT write files")
    ap.add_argument("--checkpoint",
                    default=os.path.join(_HERE, ".agentpipe_templates.done.jsonl"),
                    help="sidecar of completed intents (resume skips them)")
    args = ap.parse_args()

    if args.intents:
        intents = [x.strip() for x in args.intents.split(",") if x.strip()]
    else:
        intents = list_media_intents()

    asyncio.run(generate(
        intents=intents, n_per=args.n, chunk=args.chunk, provider=args.provider,
        concurrency=args.concurrency, seed=args.seed, dry_run=args.dry_run,
        checkpoint=args.checkpoint,
    ))


if __name__ == "__main__":
    main()
