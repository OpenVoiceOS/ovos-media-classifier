#!/usr/bin/env python3
"""Generate slot-filled training utterances from templates + entity pools.

Takes sentence templates (from ``generate_templates.py``) and entity pools
(from ``gather_entities.py``), then generates N filled utterances per template
by random sampling from per-slot entity lists.

Gracefully skips templates when a required slot has no entities.

Output schema: ``lang, domain, intent, binary_label, playback_label, media_label, sentence``
(same as ``ocp_final.csv``)

Usage::

    python -m training.generate_slot_filled_dataset
    python -m training.generate_slot_filled_dataset \\
        --entities-dir ~/.cache/ovos-media-classifier/entities \\
        --templates-dir ~/.cache/ovos-media-classifier/templates_new \\
        --n 10 \\
        --output /tmp/ocp_slot_filled.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import re
from typing import Iterator, Optional

import pandas as pd

from training import get_cache_dir, get_output_dir
from training.gather_entities import get_entities_dir
from training.generate_templates import get_templates_dir, load_templates
from training.generate_slot_literal_dataset import _classify
from training.sources import SCHEMA_COLUMNS

LOG = logging.getLogger(__name__)


def load_entity_pools(entities_dir: str) -> dict[str, list[str]]:
    """Load per-label entity CSVs into a ``{label: [values]}`` dict.

    Args:
        entities_dir: Directory containing ``<label>.csv`` files produced by
            ``gather_entities.py``.

    Returns:
        Mapping from OCPEntityLabel string to deduplicated list of entity values.
    """
    pools: dict[str, list[str]] = {}
    if not os.path.isdir(entities_dir):
        LOG.warning("entities_dir does not exist: %s", entities_dir)
        return pools
    for fname in os.listdir(entities_dir):
        if not fname.endswith(".csv") or fname == "entities_combined.csv":
            continue
        label = fname[:-4]  # strip .csv
        path = os.path.join(entities_dir, fname)
        try:
            df = pd.read_csv(path, usecols=["value"])
            values = [str(v).strip() for v in df["value"] if str(v).strip()]
            if values:
                pools[label] = values
        except Exception as exc:
            LOG.warning("Failed to load %s: %s", path, exc)
    return pools


def fill_template(
    template: str,
    pools: dict[str, list[str]],
    rng: random.Random,
    n: int = 5,
) -> Iterator[str]:
    """Generate up to *n* filled utterances from a single template.

    Yields filled strings only when ALL required slots have entities.
    Sampling is with replacement (same entity may appear across fills).

    Args:
        template: Template string with ``{slot_name}`` placeholders.
        pools: ``{label: [values]}`` entity pools.
        rng: Random generator for reproducible sampling.
        n: Maximum number of filled utterances to generate.

    Yields:
        Filled utterance strings.
    """
    slots = re.findall(r"\{(\w+)\}", template)
    # Verify all slots are covered
    for slot in slots:
        if slot not in pools or not pools[slot]:
            return  # skip template silently

    for _ in range(n):
        filled = template
        for slot in slots:
            filled = filled.replace(f"{{{slot}}}", rng.choice(pools[slot]), 1)
        yield filled


def generate_slot_filled(
    entities_dir: Optional[str] = None,
    templates_dir: Optional[str] = None,
    output: Optional[str] = None,
    n: int = 10,
    langs: Optional[list[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate slot-filled dataset from templates + entity pools.

    Args:
        entities_dir: Directory with per-label entity CSVs
            (default: ``~/.cache/ovos-media-classifier/entities/``).
        templates_dir: Root template directory
            (default: ``~/.cache/ovos-media-classifier/templates_new/``).
        output: Output CSV path
            (default: ``~/.cache/ovos-media-classifier/output/ocp_slot_filled.csv``).
        n: Filled utterances per template (default: 10).
        langs: Filter to specific language codes (default: all).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with ``SCHEMA_COLUMNS``.
    """
    ent_dir = entities_dir or get_entities_dir()
    tmpl_dir = templates_dir or get_templates_dir()
    out_path = output or os.path.join(get_output_dir(), "ocp_slot_filled.csv")

    print(f"  Loading entity pools from {ent_dir} …")
    pools = load_entity_pools(ent_dir)
    print(f"  {len(pools)} entity labels loaded")
    for label, vals in sorted(pools.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"    {label:<35}: {len(vals):,}")

    print(f"  Loading templates from {tmpl_dir} …")
    df_tmpl = load_templates(tmpl_dir, langs=langs)
    if df_tmpl.empty:
        print(f"  No templates found in {tmpl_dir}")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    print(f"  {len(df_tmpl):,} templates loaded")

    rng = random.Random(seed)
    rows = []
    skipped = 0

    for _, row in df_tmpl.iterrows():
        template = str(row["template"]).strip()
        intent = str(row["intent"]).strip()
        lang = str(row["lang"]).strip()
        if not template or not intent:
            continue

        domain, binary_label, playback_label = _classify(intent)
        count = 0
        for filled in fill_template(template, pools, rng, n=n):
            rows.append({
                "lang":          lang,
                "domain":        domain,
                "intent":        intent,
                "binary_label":  binary_label,
                "playback_label": playback_label,
                "media_label":   intent,
                "sentence":      filled,
            })
            count += 1
        if count == 0:
            skipped += 1

    print(f"  Skipped {skipped:,} templates (missing entity pools)")

    result = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    # Dedup on sentence
    before = len(result)
    result.drop_duplicates(subset=["sentence"], inplace=True)
    if before != len(result):
        print(f"  Deduped {before - len(result):,} duplicates")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"  {len(result):,} slot-filled rows → {out_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for generate_slot_filled_dataset pipeline step."""
    parser = argparse.ArgumentParser(
        description="Generate slot-filled training utterances from templates + entity pools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--entities-dir", default=None,
                        help="Entity CSV directory (default: ~/.cache/ovos-media-classifier/entities/)")
    parser.add_argument("--templates-dir", default=None,
                        help="Template CSV root directory (default: ~/.cache/ovos-media-classifier/templates_new/)")
    parser.add_argument("--n", type=int, default=10,
                        help="Filled utterances per template (default: 10)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: ~/.cache/ovos-media-classifier/output/ocp_slot_filled.csv)")
    parser.add_argument("--langs", nargs="*", default=None, metavar="LANG",
                        help="Filter to specific language codes (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    generate_slot_filled(
        entities_dir=args.entities_dir,
        templates_dir=args.templates_dir,
        output=args.output,
        n=args.n,
        langs=args.langs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
