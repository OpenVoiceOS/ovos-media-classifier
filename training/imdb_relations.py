#!/usr/bin/env python3
"""Build IMDb-derived **relational** artifacts + supplementary entity pools.

``ingest_entities`` produces flat per-label entity pools (one value column each).
That is enough for single-slot templates, but a template that fills *several*
slots from one group ("episode {n} of {tv_show}", "the silent film {title}")
should draw them from ONE real record so the surface text is coherent.  This
module joins the IMDb auxiliary datasets to ``imdb-titles`` and writes:

* ``data/relational/episodes.jsonl`` — ``{tv_show, season, episode, episode_title}``
  joined from ``imdb-episodes`` (series_id → series title, episode imdb_id →
  episode title).
* ``data/relational/bw_silent.jsonl`` — ``{title, qualifier, year}`` real films
  tagged ``black_and_white`` / ``silent`` from ``imdb-bw-silent`` /
  ``imdb-technical-specs`` joined to their real title.
* ``data/relational/movies.jsonl`` — ``{title, genre, year, num_votes,
  director, writer}`` coherent movie records.  ``num_votes`` (from
  ``imdb-ratings``) powers **popularity-weighted** entity sampling in
  ``build_dataset``.  Person slots are filled via the ``--credits`` hook (below).

It also refreshes a handful of entity pools that only a join can fill:
``season_number``, ``episode_number``, ``episode_title``, and the **real**
``bw_movie_title`` / ``silent_movie_title`` pools (replacing the
``movie_title`` aliases), plus ``data/entities/_imdb_votes.csv`` (the
``title<TAB>num_votes`` weight table the sampler reads).

The ``--credits`` hook (movie person slots)
------------------------------------------
``imdb-crew`` gives ``imdb_id → directors/writers`` as ``nconst`` person IDs
(``nm…``), NOT names.  To fill ``(movie, director, writer, actor)`` coherently we
need a ``nconst → primary_name`` table.  Two clean paths, auto-detected:

* **credits path** — if ``media-metadata-imdb-credits``
  (``imdb_id``/``primary_title``/``person_name``/``category``) exists, OR a
  ``name.basics``-style ``media-metadata-imdb-names`` (``nconst → primary_name``)
  exists to resolve ``imdb-crew``, the movie records are filled with the REAL
  director / writer / actor for each title.
* **fallback path** — otherwise the movie records carry no person fields and
  ``build_dataset`` fills the person slots **independently** from the flat
  ``movie_director`` / ``movie_writer`` / ``movie_actor`` pools (the long-standing
  behaviour).  Re-running this script once the credits dataset lands upgrades the
  movie records to coherent persons with no other change.

Which path was taken is logged at the end.  Run it::

    python -m training.imdb_relations                 # → data/relational + data/entities
    python -m training.imdb_relations --cap 300000
    python -m training.imdb_relations --no-local      # always read HuggingFace
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
DEFAULT_ENTITIES_DIR = os.path.join(REPO_ROOT, "data", "entities")
DEFAULT_RELATIONAL_DIR = os.path.join(REPO_ROOT, "data", "relational")

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

# IMDb title types that are real "movie" works for the movies relation.
_MOVIE_TYPES = {"movie", "tvmovie"}
_SERIES_TYPES = {"tvseries", "tvminiseries"}


# ---------------------------------------------------------------------------
# HF iteration (shared shape with ingest_entities, kept local to avoid a cycle)
# ---------------------------------------------------------------------------

def _iter_hf(repo: str, columns: Optional[List[str]] = None) -> Iterable[dict]:
    """Yield rows from a TigreGotico HF dataset, file by file (pandas)."""
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd

    full = f"TigreGotico/{repo}"
    files = sorted(f for f in HfApi().list_repo_files(full, repo_type="dataset")
                   if f.endswith((".parquet", ".csv")))
    for f in files:
        local = hf_hub_download(full, f, repo_type="dataset")
        if f.endswith(".parquet"):
            try:
                df = pd.read_parquet(local, columns=columns)
            except Exception:
                df = pd.read_parquet(local)
        else:
            df = pd.read_csv(local, dtype=str, keep_default_na=False)
        for rec in df.to_dict("records"):
            yield rec


def _clean(v) -> str:
    s = str(v if v is not None else "").strip()
    if s.lower() in ("", "nan", "none", "null", "\\n"):
        return ""
    return s


def _year(v) -> str:
    s = _clean(v)
    m = re.search(r"(\d{4})", s)
    return m.group(1) if m and m.group(1) != "0000" else ""


def _decade(year: str) -> str:
    return f"{(int(year) // 10) * 10}s" if year else ""


def _genres(v) -> List[str]:
    """IMDb ``genres`` is an ndarray / comma-joined string."""
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            v = v.tolist()
    except ImportError:
        pass
    if isinstance(v, (list, tuple)):
        out = []
        for g in v:
            out.extend(str(g).split(","))
        return [g.strip() for g in out if _clean(g) and g.strip() != "\\N"]
    s = _clean(v)
    return [g.strip() for g in s.split(",") if g.strip() and g.strip() != "\\N"]


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Title index — id → record (the join key for every aux dataset)
# ---------------------------------------------------------------------------

def load_title_index(prefer_local: bool) -> Dict[str, dict]:
    """``imdb_id → {title, type, is_adult, genres, year}`` from imdb-titles."""
    idx: Dict[str, dict] = {}
    cols = ["imdb_id", "title_type", "primary_title", "is_adult", "genres",
            "start_year"]
    n = 0
    for row in _iter_hf("media-metadata-imdb-titles", columns=cols):
        n += 1
        iid = _clean(row.get("imdb_id"))
        title = _clean(row.get("primary_title")) or _clean(row.get("original_title"))
        if not iid or not title:
            continue
        idx[iid] = {
            "title": title,
            "type": str(row.get("title_type") or "").strip().lower(),
            "is_adult": _truthy(row.get("is_adult")),
            "genres": _genres(row.get("genres")),
            "year": _year(row.get("start_year")),
        }
    print(f"  title index: {len(idx):,} titles ({n:,} rows)", flush=True)
    return idx


def load_votes() -> Dict[str, int]:
    """``imdb_id → num_votes`` from imdb-ratings (popularity weight)."""
    votes: Dict[str, int] = {}
    for row in _iter_hf("media-metadata-imdb-ratings",
                        columns=["imdb_id", "num_votes"]):
        iid = _clean(row.get("imdb_id"))
        try:
            nv = int(float(row.get("num_votes") or 0))
        except (TypeError, ValueError):
            nv = 0
        if iid and nv > 0:
            votes[iid] = nv
    print(f"  ratings: {len(votes):,} titles with votes", flush=True)
    return votes


# ---------------------------------------------------------------------------
# Credits hook — nconst → name (auto-detected; fallback = independent fill)
# ---------------------------------------------------------------------------

def detect_credits_source() -> Tuple[str, Optional[str]]:
    """Return ``(path_kind, repo)`` for the credits join.

    * ``("credits", repo)``  — a ready movie→person credits set exists.
    * ``("names", repo)``    — a ``nconst → primary_name`` set exists (resolve crew).
    * ``("fallback", None)`` — neither; movie person slots fill independently.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    have = {d.id for d in api.list_datasets(author="TigreGotico", limit=500)}
    if "TigreGotico/media-metadata-imdb-credits" in have:
        return "credits", "media-metadata-imdb-credits"
    if "TigreGotico/media-metadata-imdb-names" in have:
        return "names", "media-metadata-imdb-names"
    if "TigreGotico/media-metadata-imdb-name-basics" in have:
        return "names", "media-metadata-imdb-name-basics"
    return "fallback", None


def load_name_index(repo: str) -> Dict[str, str]:
    """``nconst → primary_name`` for resolving ``imdb-crew`` person IDs."""
    names: Dict[str, str] = {}
    for row in _iter_hf(repo):
        nid = _clean(row.get("nconst")) or _clean(row.get("person_id"))
        nm = _clean(row.get("primary_name")) or _clean(row.get("name"))
        if nid and nm:
            names[nid] = nm
    print(f"  name index: {len(names):,} persons", flush=True)
    return names


def _first_id(v) -> str:
    """First ``nm…`` id from a crew field (comma-joined or scalar)."""
    s = _clean(v)
    return s.split(",")[0].strip() if s else ""


def build_credits(title_idx: Dict[str, dict], prefer_local: bool
                  ) -> Tuple[str, Dict[str, dict]]:
    """Per-title ``{director, writer, actor}`` names, when a source exists.

    Returns ``(path_kind, {imdb_id: {director, writer, actor}})``.  ``path_kind``
    is ``"credits"`` / ``"names"`` / ``"fallback"``; the dict is empty for the
    fallback path (person slots then fill independently in build_dataset).
    """
    kind, repo = detect_credits_source()
    creds: Dict[str, dict] = {}
    if kind == "credits":
        for row in _iter_hf(repo):
            iid = _clean(row.get("imdb_id"))
            name = _clean(row.get("person_name")) or _clean(row.get("name"))
            cat = (_clean(row.get("category")) or "").lower()
            if not iid or not name:
                continue
            slot = ("director" if "direct" in cat else
                    "writer" if "writ" in cat else
                    "actor" if ("act" in cat or "self" in cat) else "")
            if slot:
                creds.setdefault(iid, {}).setdefault(slot, name)
    elif kind == "names":
        names = load_name_index(repo)
        for row in _iter_hf("media-metadata-imdb-crew"):
            iid = _clean(row.get("imdb_id"))
            if not iid or iid not in title_idx:
                continue
            d = names.get(_first_id(row.get("directors")))
            w = names.get(_first_id(row.get("writers")))
            rec = {}
            if d:
                rec["director"] = d
            if w:
                rec["writer"] = w
            if rec:
                creds[iid] = rec
    print(f"  credits path: {kind} → {len(creds):,} titles with persons",
          flush=True)
    return kind, creds


# ---------------------------------------------------------------------------
# Relational record builders
# ---------------------------------------------------------------------------

def build_movies(title_idx, votes, creds, cap) -> List[dict]:
    """Coherent ``(title, genre, year, num_votes [, director, writer])`` records.

    One per real, non-adult movie/tvMovie title.  Sorted by num_votes desc and
    capped so the popular head is always present; build_dataset re-weights.
    """
    recs: List[dict] = []
    for iid, t in title_idx.items():
        if t["is_adult"] or t["type"] not in _MOVIE_TYPES or not t["title"]:
            continue
        rec = {"title": t["title"], "year": t["year"],
               "genre": (t["genres"][0] if t["genres"] else ""),
               "genres": t["genres"], "num_votes": votes.get(iid, 0)}
        c = creds.get(iid)
        if c:
            rec.update({k: v for k, v in c.items() if v})
        recs.append(rec)
    recs.sort(key=lambda r: r["num_votes"], reverse=True)
    return recs[:cap]


def build_episodes(title_idx, votes, cap) -> List[dict]:
    """Coherent ``(tv_show, season, episode, episode_title)`` records.

    series_id → series title; episode imdb_id → episode title.  Only emitted
    when the series title resolves and a numeric season/episode is present.
    """
    recs: List[dict] = []
    seen = 0
    for row in _iter_hf("media-metadata-imdb-episodes"):
        seen += 1
        sid = _clean(row.get("series_id"))
        eid = _clean(row.get("imdb_id"))
        series = title_idx.get(sid)
        if not series or series["is_adult"] or not series["title"]:
            continue
        season = _clean(row.get("season_number"))
        episode = _clean(row.get("episode_number"))
        if not (season.isdigit() and episode.isdigit()):
            continue
        ep = title_idx.get(eid, {})
        recs.append({
            "tv_show": series["title"],
            "season": season,
            "episode": episode,
            "episode_title": ep.get("title", ""),
            "num_votes": votes.get(sid, 0),
        })
    # popular series first, capped
    recs.sort(key=lambda r: r["num_votes"], reverse=True)
    print(f"  episodes: {seen:,} rows → {len(recs):,} resolved", flush=True)
    return recs[:cap]


def build_bw_silent(title_idx, cap) -> List[dict]:
    """Real ``(title, qualifier, year)`` films tagged black_and_white / silent.

    Joins imdb-technical-specs (+ the bw-silent subset) to the real title so a
    qualifier template fills a REAL bw/silent film with the matching qualifier.
    """
    recs: List[dict] = []
    seen_ids: set = set()
    for repo in ("media-metadata-imdb-bw-silent",
                 "media-metadata-imdb-technical-specs"):
        for row in _iter_hf(repo, columns=["imdb_id", "coloration_concept_ids",
                                           "is_color", "is_silent",
                                           "sound_mix_ids"]):
            iid = _clean(row.get("imdb_id"))
            t = title_idx.get(iid)
            if not t or t["is_adult"] or not t["title"] or iid in seen_ids:
                continue
            quals: List[str] = []
            colids = " ".join(str(x) for x in _genres(row.get("coloration_concept_ids")))
            if "black_and_white" in colids or row.get("is_color") is False:
                quals.append("black_and_white")
            mixes = " ".join(str(x) for x in _genres(row.get("sound_mix_ids")))
            if _truthy(row.get("is_silent")) or "silent" in mixes:
                quals.append("silent")
            if not quals:
                continue
            seen_ids.add(iid)
            for q in quals:
                recs.append({"title": t["title"], "qualifier": q,
                             "year": t["year"]})
        if len(recs) >= cap:
            break
    print(f"  bw/silent: {len(recs):,} qualifier-tagged real titles", flush=True)
    return recs[:cap]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_jsonl(path: str, recs: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(recs):,} → {path}", flush=True)


def _merge_pool(entities_dir: str, label: str, values: Iterable[str],
                cap: int) -> int:
    """Add ``values`` to ``data/entities/<label>.csv`` (dedup, cap)."""
    path = os.path.join(entities_dir, f"{label}.csv")
    pool: Dict[str, str] = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                v = (rec.get("value") or "").strip()
                if v:
                    pool.setdefault(v.lower(), v)
    for v in values:
        v = (v or "").strip()
        if v and v.lower() not in pool and len(pool) < cap:
            pool[v.lower()] = v
    os.makedirs(entities_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["value"])
        for v in pool.values():
            w.writerow([v])
    return len(pool)


def build(entities_dir: str, relational_dir: str, cap: int,
          prefer_local: bool) -> Dict[str, object]:
    print("Loading IMDb title index + ratings …", flush=True)
    title_idx = load_title_index(prefer_local)
    votes = load_votes()

    print("Resolving credits (movie person slots) …", flush=True)
    cred_kind, creds = build_credits(title_idx, prefer_local)

    print("Building relational records …", flush=True)
    movies = build_movies(title_idx, votes, creds, cap)
    episodes = build_episodes(title_idx, votes, cap)
    bw_silent = build_bw_silent(title_idx, cap)

    _write_jsonl(os.path.join(relational_dir, "movies.jsonl"), movies)
    _write_jsonl(os.path.join(relational_dir, "episodes.jsonl"), episodes)
    _write_jsonl(os.path.join(relational_dir, "bw_silent.jsonl"), bw_silent)

    # supplementary entity pools only a join can fill
    counts: Dict[str, int] = {}
    counts["season_number"] = _merge_pool(
        entities_dir, "season_number",
        sorted({r["season"] for r in episodes}, key=lambda s: int(s)), cap)
    counts["episode_number"] = _merge_pool(
        entities_dir, "episode_number",
        sorted({r["episode"] for r in episodes}, key=lambda s: int(s)), cap)
    counts["episode_title"] = _merge_pool(
        entities_dir, "episode_title",
        (r["episode_title"] for r in episodes if r["episode_title"]), cap)
    counts["bw_movie_title"] = _merge_pool(
        entities_dir, "bw_movie_title",
        (r["title"] for r in bw_silent if r["qualifier"] == "black_and_white"),
        cap)
    counts["silent_movie_title"] = _merge_pool(
        entities_dir, "silent_movie_title",
        (r["title"] for r in bw_silent if r["qualifier"] == "silent"), cap)

    # popularity weight table: title<TAB>num_votes for the movie pool
    votes_path = os.path.join(entities_dir, "_imdb_votes.csv")
    with open(votes_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["value", "num_votes"])
        seen: set = set()
        for r in movies:
            k = r["title"].lower()
            if k not in seen:
                seen.add(k)
                w.writerow([r["title"], r["num_votes"]])
    print(f"  wrote vote weights ({len(movies):,}) → {votes_path}", flush=True)

    return {
        "credits_path": cred_kind,
        "movies": len(movies),
        "episodes": len(episodes),
        "bw_silent": len(bw_silent),
        "pools": counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build IMDb relational artifacts + supplementary pools",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--entities-dir", default=DEFAULT_ENTITIES_DIR)
    ap.add_argument("--relational-dir", default=DEFAULT_RELATIONAL_DIR)
    ap.add_argument("--cap", type=int, default=300_000)
    ap.add_argument("--no-local", action="store_true")
    args = ap.parse_args()

    summary = build(args.entities_dir, args.relational_dir, args.cap,
                    prefer_local=not args.no_local)
    print(f"\nDone. credits_path={summary['credits_path']!r}  "
          f"movies={summary['movies']:,}  episodes={summary['episodes']:,}  "
          f"bw_silent={summary['bw_silent']:,}")


if __name__ == "__main__":
    main()
