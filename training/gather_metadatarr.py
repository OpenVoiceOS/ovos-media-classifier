"""Supplementary entity pool gathering via **metadatarr** (keyless clients).

`gather_entities.py` builds the bulk pools from HuggingFace + public catalogue
APIs.  This adds a freshness layer from ``metadatarr``'s keyless cross-source
clients (OpenLibrary, TVmaze, TheAudioDB) — real, current titles/names — appended
to the same per-label ``value,source`` CSVs the main gather writes.

Usage::

    python -m training.gather_metadatarr --output /tmp/omc_ds/entities --per 60
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Set

# seed query terms used to fan out catalogue searches
_GENRE_SEEDS = [
    "rock", "pop", "jazz", "metal", "classical", "electronic", "hip hop", "soul",
    "folk", "blues", "country", "reggae", "punk", "indie", "ambient", "techno",
]
_TOPIC_SEEDS = [
    "fiction", "science fiction", "fantasy", "history", "mystery", "romance",
    "thriller", "biography", "horror", "adventure", "poetry", "philosophy",
]
_SHOW_SEEDS = [
    "the", "love", "world", "dark", "house", "game", "star", "city", "night",
    "lost", "doctor", "crime", "family", "law", "war",
]


def _append(pools: Dict[str, Set[str]], label: str, value: str) -> None:
    v = (value or "").strip()
    if v and len(v) <= 120:
        pools.setdefault(label, set()).add(v)


def gather(per: int = 60) -> Dict[str, Set[str]]:
    pools: Dict[str, Set[str]] = {}

    try:
        from metadatarr import OpenLibraryClient
        ol = OpenLibraryClient()
        for q in _TOPIC_SEEDS:
            try:
                for hit in ol.search(q, limit=per) or []:
                    _append(pools, "audiobook_title", getattr(hit, "title", ""))
                    authors = getattr(hit, "author_name", None) or []
                    if isinstance(authors, str):
                        authors = [authors]
                    for a in authors[:2]:
                        _append(pools, "audiobook_author", a)
            except Exception:
                continue
    except Exception:
        pass

    try:
        from metadatarr import TVmazeClient
        tv = TVmazeClient()
        for q in _SHOW_SEEDS:
            try:
                for hit in tv.search_shows(q) or []:
                    show = getattr(hit, "show", hit)
                    _append(pools, "tv_show_title", getattr(show, "name", ""))
            except Exception:
                continue
    except Exception:
        pass

    try:
        from metadatarr import AudioDBClient
        db = AudioDBClient()
        for q in _GENRE_SEEDS:
            try:
                for art in db.search_artist(q) or []:
                    _append(pools, "artist_name", getattr(art, "name", "") or
                            getattr(art, "strArtist", ""))
            except Exception:
                continue
    except Exception:
        pass

    return pools


def merge_into(entities_dir: str, pools: Dict[str, Set[str]]) -> None:
    os.makedirs(entities_dir, exist_ok=True)
    for label, values in pools.items():
        path = os.path.join(entities_dir, f"{label}.csv")
        existing: Set[str] = set()
        rows: List[List[str]] = []
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                reader = csv.reader(fh)
                header = next(reader, None)
                for r in reader:
                    if r:
                        rows.append(r)
                        existing.add(r[0].strip().lower())
        added = 0
        for v in sorted(values):
            if v.lower() not in existing:
                rows.append([v, "metadatarr"])
                existing.add(v.lower())
                added += 1
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["value", "source"])
            w.writerows(rows)
        print(f"    {label}: +{added} from metadatarr → {len(rows)} total")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, help="entities dir")
    ap.add_argument("--per", type=int, default=60)
    args = ap.parse_args()
    print("Gathering metadatarr entities …", flush=True)
    pools = gather(per=args.per)
    total = sum(len(v) for v in pools.values())
    print(f"  collected {total:,} unique values across {len(pools)} labels")
    merge_into(args.output, pools)
    print("Done.")


if __name__ == "__main__":
    main()
