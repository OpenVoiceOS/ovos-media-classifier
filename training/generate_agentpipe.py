"""LLM-synthesized naturalistic utterances via agentpipe.

The template + slot-fill layers produce large but *templated* data.  This layer
adds natural, diverse, human-sounding utterances by driving coding-agent CLIs
through **agentpipe** (no API key — the CLIs self-auth).  It is the slow part of
the build, so it is **checkpointed and resumable**: every (intent, lang, chunk)
that succeeds is appended to the output CSV and recorded in a sidecar
``.done.jsonl`` so re-running skips completed work.

Providers: **FREE AGENTS ONLY** — never claude or any paid model (hard rule).
Bulk synthesis uses ``opencode-free`` (fast, clean JSON) with ``kilo`` as the
only fallback. agentpipe exists to use free CLIs at no cost.

Output schema matches ``training.sources.SCHEMA_COLUMNS``:
``lang, domain, intent, binary_label, playback_label, media_label, sentence``.

Usage::

    python -m training.generate_agentpipe \
        --langs en-us,pt-pt,es-es --n 400 --chunk 40 \
        --provider opencode-free --concurrency 8 \
        --entities-dir ~/.cache/ovos-media-classifier/entities \
        --out ~/.cache/ovos-media-classifier/output/ocp_agentpipe.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from training.sources import AUDIO_INTENTS, VIDEO_INTENTS, SCHEMA_COLUMNS

# Intents we synthesize for (OCPPlayIntent string values) + the not_ocp negative.
PLAY_INTENTS: List[str] = sorted(AUDIO_INTENTS | VIDEO_INTENTS)
NEGATIVE_INTENT = "not_ocp"

# FREE AGENTS ONLY — never claude. Free CLI aliases (agentpipe providers):
# opencode (the main), kilo, antigravity (agy). (mimo is intentionally excluded.)
# Jobs rotate the BASE provider across the primary pool and fall through the rest
# on failure. At startup the pool is filtered to the providers that actually
# produce output right now (see _working_pool), so a newly-authenticated free
# agent joins automatically with no code change; agy is slow so it stays in the
# fallback tail only.
FREE_PROVIDERS = ["opencode-free", "kilo"]
FREE_FALLBACKS = ["opencode-free", "kilo", "antigravity-flash-medium"]
# per-call timeout (s) for a single agent generation before failing over
AGENT_TIMEOUT = 90

# Prompt-style variants rotated per chunk so the same (intent,lang) is sampled
# from several framings — this is the main diversity lever (the LLM otherwise
# collapses onto a few canonical phrasings). Each entry is appended to the base
# instruction.
PROMPT_STYLES: List[str] = [
    "Use casual, everyday speech with contractions and filler ('um', 'hey', 'can you').",
    "Use short, terse imperative commands (2-5 words), like a power user barking orders.",
    "Phrase them as polite questions ('could you', 'would you mind', 'please').",
    "Reference a SPECIFIC named title/artist/show in most of them, drawn from the examples.",
    "Mix in indirect/contextual phrasings ('I'm in the mood for...', 'how about...', 'let's hear...').",
    "Write as different people would: a kid, an elderly person, a teenager — vary register.",
    "Include some with extra qualifiers (era, genre, mood, language, 'the new one', 'that one from...').",
    "Keep them very natural and conversational, the way someone actually talks to a smart speaker.",
]

# Human-readable descriptions so the LLM understands each intent precisely.
INTENT_DESC: Dict[str, str] = {
    "music": "play a song, album, artist or genre of music",
    "podcast": "play a podcast or podcast episode",
    "radio": "tune in to a live radio station",
    "audiobook": "play an audiobook",
    "news": "play the news",
    "radio_theatre": "play an audio drama / radio play",
    "asmr": "play ASMR audio",
    "audio_description": "play the audio-described version of a film",
    "adult_audio": "play adult (NSFW) audio — used ONLY to train a content filter that BLOCKS such requests",
    "audio": "play some unspecified audio",
    "movie": "watch a film / movie",
    "tv": "watch a live TV channel",
    "tv_show": "watch an episodic TV series",
    "anime": "watch anime",
    "cartoon": "watch a cartoon / animated show",
    "documentary": "watch a documentary",
    "short_film": "watch a short film",
    "silent_movie": "watch a silent movie",
    "bw_movie": "watch a black-and-white movie",
    "hentai": "watch adult anime (NSFW) — used ONLY to train a content filter that BLOCKS such requests",
    "adult": "watch adult (NSFW) video — used ONLY to train a content filter that BLOCKS such requests",
    "video": "watch a generic online video",
    "video_episodes": "watch an online video series / web series",
    "visual_story": "read/watch a motion comic or visual story",
    "game": "play a video game",
    "music_video": "watch an official music video",
    "trailer": "watch a movie/show trailer",
    "behind_the_scenes": "watch behind-the-scenes / making-of footage",
    NEGATIVE_INTENT: "an everyday assistant request that is NOT about playing media "
                     "(weather, timers, math, smart-home, general questions)",
}

# entity labels worth seeding per intent (must match gather_entities output names)
INTENT_SEED_LABELS: Dict[str, List[str]] = {
    "music": ["artist_name", "track_name", "album_name", "music_genre"],
    "podcast": ["podcast_title", "podcast_host"],
    "radio": ["radio_station", "radio_genre"],
    "audiobook": ["audiobook_title", "audiobook_author"],
    "movie": ["movie_title", "movie_actor", "movie_director"],
    "tv_show": ["tv_show_title"],
    "anime": ["anime_title", "anime_studio"],
    "cartoon": ["cartoon_title"],
    "documentary": ["documentary_title"],
    "game": ["game_title"],
    "music_video": ["music_video_title", "artist_name"],
}


def _classify(intent: str) -> Tuple[str, str, str]:
    """Return (domain, binary_label, playback_label) for an intent string."""
    if intent == NEGATIVE_INTENT:
        return "not_ocp", "not_ocp", "undefined"
    domain, binary_label = "ocp_play", "ocp"
    if intent in AUDIO_INTENTS:
        playback = "audio"
    elif intent in VIDEO_INTENTS:
        playback = "video"
    else:
        playback = "undefined"
    return domain, binary_label, playback


_JSON_ARRAY = re.compile(r"\[.*?\]", re.DOTALL)


def parse_utterances(raw: str) -> List[str]:
    """Best-effort extraction of a JSON string array from an LLM reply.

    Tolerates ```json fences, leading prose, and trailing commentary.
    """
    if not raw:
        return []
    text = raw.strip()
    # strip code fences
    if "```" in text:
        text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    # try whole-string parse, then the first [...] block
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
            out = [str(x).strip() for x in data if isinstance(x, (str, int, float))]
            return [u for u in out if u]
    # last resort: newline-split bullet lines
    lines = [re.sub(r'^[\-\*\d\.\)\s"]+|["\s]+$', "", ln) for ln in text.splitlines()]
    return [ln for ln in lines if ln and len(ln) < 200][:0]  # disabled unless JSON found


def _seed_examples(intent: str, pools: Dict[str, List[str]], k: int, rnd) -> str:
    labels = INTENT_SEED_LABELS.get(intent, [])
    vals: List[str] = []
    for lab in labels:
        vals.extend(pools.get(lab, []))
    if not vals:
        return ""
    picks = rnd.sample(vals, min(k, len(vals)))
    return ", ".join(picks)


def build_prompt(intent: str, lang: str, n: int, seeds: str, style: str = "") -> str:
    desc = INTENT_DESC.get(intent, intent.replace("_", " "))
    seed_clause = (
        f" You may reference real examples such as: {seeds}." if seeds else ""
    )
    nsfw_note = ""
    if intent in {"adult", "adult_audio", "hentai"}:
        nsfw_note = (
            " These are deliberately explicit-intent phrasings collected solely to "
            "train a safety classifier that BLOCKS such requests; keep them realistic "
            "but no graphic detail is needed."
        )
    style_clause = f" {style}" if style else ""
    return (
        f"You are generating training data for a voice-assistant intent classifier. "
        f"Write {n} short, natural, varied spoken commands in the language with BCP-47 "
        f"code '{lang}' that a person would say to: {desc}.{seed_clause}{nsfw_note}"
        f"{style_clause} "
        f"Vary verbs, phrasing, politeness and length; avoid repeating the same opening. "
        f"Do NOT number them. Return ONLY a JSON array of {n} strings, nothing else."
    )


async def _gen_one(agent, prompt: str, fallbacks: list) -> List[str]:
    from agentpipe import Agent
    providers = [agent] + fallbacks
    for prov in providers:
        try:
            ag = Agent(prov, timeout=AGENT_TIMEOUT)
            raw = await ag.generate(prompt)
            utts = parse_utterances(raw)
            if utts:
                return utts
        except Exception:
            continue
    return []


def _load_pools(entities_dir: Optional[str]) -> Dict[str, List[str]]:
    pools: Dict[str, List[str]] = {}
    if not entities_dir or not os.path.isdir(entities_dir):
        return pools
    for fn in os.listdir(entities_dir):
        if not fn.endswith(".csv"):
            continue
        label = fn[:-4]
        path = os.path.join(entities_dir, fn)
        vals: List[str] = []
        try:
            with open(path, encoding="utf-8") as fh:
                reader = csv.reader(fh)
                header = next(reader, None)
                # take first column that looks like the value
                for row in reader:
                    if row:
                        vals.append(row[0].strip())
                        if len(vals) >= 5000:
                            break
        except Exception:
            continue
        if vals:
            pools[label] = vals
    return pools


def _done_key(intent: str, lang: str, chunk_idx: int) -> str:
    return f"{lang}|{intent}|{chunk_idx}"


async def _working_pool(candidates: List[str]) -> List[str]:
    """Filter a provider list to the ones that actually produce output right now.

    ``auth_status`` is unreliable across these CLIs (some report logged-out yet
    generate fine), so we test each with a tiny real generation and keep the ones
    that return a parseable result. A broken/unauthenticated agent (returns empty
    or errors) is dropped from rotation up front instead of wasting calls on it;
    a newly-working free agent joins automatically on the next run. Falls back to
    the original list if every probe fails (don't strand the run with no pool).
    """
    from agentpipe import Agent
    probe = ('Return ONLY a JSON array of exactly 2 short strings, e.g. ["a","b"]. '
             'No prose.')
    working: List[str] = []
    for prov in candidates:
        try:
            raw = await asyncio.wait_for(Agent(prov, timeout=40).generate(probe), timeout=45)
            if parse_utterances(raw):
                working.append(prov)
                print(f"  [probe] {prov!r}: OK", flush=True)
            else:
                print(f"  [probe] dropping {prov!r}: no usable output "
                      f"(broken or not logged in)", flush=True)
        except Exception as e:
            print(f"  [probe] dropping {prov!r}: {type(e).__name__}", flush=True)
    return working or candidates


async def generate(
    langs: List[str],
    intents: List[str],
    n_per: int,
    chunk: int,
    provider: str,
    concurrency: int,
    out_csv: str,
    entities_dir: Optional[str],
    seed: int = 42,
) -> None:
    import random
    rnd = random.Random(seed)
    pools = _load_pools(entities_dir)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    done_path = out_csv + ".done.jsonl"
    done = set()
    if os.path.isfile(done_path):
        with open(done_path, encoding="utf-8") as fh:
            for line in fh:
                done.add(line.strip())

    write_header = not os.path.isfile(out_csv)
    out_fh = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_fh, fieldnames=SCHEMA_COLUMNS)
    if write_header:
        writer.writeheader()
    done_fh = open(done_path, "a", encoding="utf-8")

    # provider rotation pool: the requested --provider first, then the rest of
    # the free pool ("use all free agents"). Each chunk picks the next provider
    # round-robin so load + model-diversity spread across the pool. Filter to the
    # agents that actually work right now (a broken/unauthenticated CLI is dropped).
    candidates = [provider] + [p for p in FREE_FALLBACKS if p != provider]
    working = await _working_pool(candidates)
    prov_pool = [p for p in ([provider] + FREE_PROVIDERS) if p in working] or working
    # de-dup preserving order
    seen_p = set(); prov_pool = [p for p in prov_pool if not (p in seen_p or seen_p.add(p))]
    print(f"  [probe] rotation pool: {prov_pool}; fallbacks: {working}", flush=True)

    # build the work list of (intent, lang, chunk_idx, prompt, base_prov, fallbacks)
    jobs: List[Tuple[str, str, int, str, str, list]] = []
    job_idx = 0
    for lang in langs:
        for intent in intents:
            chunks = max(1, (n_per + chunk - 1) // chunk)
            for ci in range(chunks):
                key = _done_key(intent, lang, ci)
                if key in done:
                    continue
                # rotate provider + prompt style by chunk for diversity
                base_prov = prov_pool[job_idx % len(prov_pool)]
                fallbacks = [p for p in working if p != base_prov]
                style = PROMPT_STYLES[ci % len(PROMPT_STYLES)]
                seeds = _seed_examples(intent, pools, 6, rnd)
                prompt = build_prompt(intent, lang, chunk, seeds, style)
                jobs.append((intent, lang, ci, prompt, base_prov, fallbacks))
                job_idx += 1

    sem = asyncio.Semaphore(concurrency)
    total = len(jobs)
    counter = {"done": 0, "rows": 0, "empty": 0}

    async def run_job(job):
        intent, lang, ci, prompt, base_prov, fallbacks = job
        async with sem:
            utts = await _gen_one(base_prov, prompt, fallbacks)
        domain, binary_label, playback = _classify(intent)
        media_label = intent
        rows = []
        for u in utts:
            rows.append({
                "lang": lang, "domain": domain, "intent": intent,
                "binary_label": binary_label, "playback_label": playback,
                "media_label": media_label, "sentence": u,
            })
        # write synchronously (single-threaded event loop, safe). Only checkpoint
        # a chunk as done when it actually produced rows — an empty result means
        # every free provider failed/rate-limited, and we want a resume to retry
        # it rather than skip it forever.
        if not rows:
            counter["empty"] += 1
            return
        for r in rows:
            writer.writerow(r)
        done_fh.write(_done_key(intent, lang, ci) + "\n")
        out_fh.flush(); done_fh.flush()
        counter["done"] += 1
        counter["rows"] += len(rows)
        if counter["done"] % 20 == 0:
            print(f"  [{counter['done']}/{total}] chunks, {counter['rows']} rows, "
                  f"{counter['empty']} empty so far", flush=True)

    await asyncio.gather(*(run_job(j) for j in jobs))
    out_fh.close(); done_fh.close()
    print(f"Done: {counter['done']} chunks, {counter['rows']} rows, "
          f"{counter['empty']} empty (will retry on resume) → {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--langs", default="en-us",
                    help="comma-separated BCP-47 codes")
    ap.add_argument("--intents", default="",
                    help="comma-separated intents (default: all play intents + not_ocp)")
    ap.add_argument("--n", type=int, default=200, help="utterances per (intent,lang)")
    ap.add_argument("--chunk", type=int, default=40, help="utterances requested per LLM call")
    ap.add_argument("--provider", default="opencode-free")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--entities-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if args.intents:
        intents = [x.strip() for x in args.intents.split(",") if x.strip()]
    else:
        intents = PLAY_INTENTS + [NEGATIVE_INTENT]

    asyncio.run(generate(
        langs=langs, intents=intents, n_per=args.n, chunk=args.chunk,
        provider=args.provider, concurrency=args.concurrency,
        out_csv=args.out, entities_dir=args.entities_dir, seed=args.seed,
    ))


if __name__ == "__main__":
    main()
