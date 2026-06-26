#!/usr/bin/env python3
"""Generate a slot-literal training dataset from sentence templates.

In the output, slot placeholders are kept verbatim (e.g. ``{artist_name}``).
This allows deterministic categorical feature derivation at training time
without running NER: any ``{slot_name}`` in the sentence string directly
sets the corresponding NER feature to 1.

Output schema: ``lang, domain, intent, binary_label, playback_label, media_label, sentence``
(same as ``ocp_final.csv``)

Usage::

    python -m training.generate_slot_literal_dataset
    python -m training.generate_slot_literal_dataset \\
        --templates-dir ~/.cache/ovos-media-classifier/templates_new \\
        --output /tmp/ocp_slot_literal.csv
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd

from training import get_output_dir, get_cache_dir
from training.generate_templates import get_templates_dir, load_templates
from training.sources import AUDIO_INTENTS, VIDEO_INTENTS, SCHEMA_COLUMNS


def _classify(intent: str) -> tuple[str, str, str]:
    """Return ``(domain, binary_label, playback_label)`` for an intent string.

    Args:
        intent: raw media label string (a ``LABEL_TO_MEDIA_TYPE`` key).

    Returns:
        Tuple of (domain, binary_label, playback_label).
    """
    domain = "ocp_play"
    binary_label = "OCP"
    if intent in AUDIO_INTENTS:
        playback_label = "audio"
    elif intent in VIDEO_INTENTS:
        playback_label = "video"
    else:
        playback_label = "undefined"
    return domain, binary_label, playback_label


def generate_slot_literal(
    templates_dir: Optional[str] = None,
    output: Optional[str] = None,
    langs: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Generate a slot-literal dataset from template CSVs.

    Each template produces exactly one row with slot names kept as literal text.

    Args:
        templates_dir: Root directory of template CSVs (default: ``~/.cache/ovos-media-classifier/templates_new/``).
        output: Path to save output CSV (default: ``~/.cache/ovos-media-classifier/output/ocp_slot_literal.csv``).
        langs: Filter to specific language codes (default: all).

    Returns:
        DataFrame with ``SCHEMA_COLUMNS``.
    """
    tmpl_dir = templates_dir or get_templates_dir()
    out_path = output or os.path.join(get_output_dir(), "ocp_slot_literal.csv")

    df = load_templates(tmpl_dir, langs=langs)
    if df.empty:
        print(f"  No templates found in {tmpl_dir}")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    rows = []
    for _, row in df.iterrows():
        template = str(row["template"]).strip()
        intent = str(row["intent"]).strip()
        lang = str(row["lang"]).strip()
        if not template or not intent:
            continue
        domain, binary_label, playback_label = _classify(intent)
        rows.append({
            "lang":          lang,
            "domain":        domain,
            "intent":        intent,
            "binary_label":  binary_label,
            "playback_label": playback_label,
            "media_label":   intent,
            "sentence":      template,
        })

    result = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"  {len(result):,} slot-literal rows → {out_path}")
    return result


def extract_features_from_slot_literal(
    sentence: str,
    keyword_extractor: object,
    entity_label_values: list[str],
) -> dict[str, int]:
    """Extract categorical features from a slot-literal sentence.

    Slot names like ``{artist_name}`` in the sentence directly set the
    corresponding NER feature to 1 — no Aho-Corasick lookup needed.
    Keyword features are extracted normally via ``keyword_extractor``.

    This is a fast path used during training for slot-literal rows.

    Args:
        sentence: A slot-literal sentence string (e.g. ``"play {artist_name}"``).
        keyword_extractor: An object with a ``match(sentence, vocab_name, lang)``
            method (e.g. ``_VocMatcher``).  Pass ``None`` to skip keyword features.
        entity_label_values: List of valid ``OCPEntityLabel`` string values.

    Returns:
        ``{feature_name: 1}`` for every slot found in the sentence.
    """
    import re
    feat: dict[str, int] = {}

    # Slot names fire NER features directly
    for slot_name in re.findall(r"\{(\w+)\}", sentence):
        if slot_name in entity_label_values:
            feat[slot_name] = 1

    return feat


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for generate_slot_literal_dataset pipeline step."""
    parser = argparse.ArgumentParser(
        description="Generate slot-literal training dataset from templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--templates-dir", default=None,
                        help="Template CSV root directory (default: ~/.cache/ovos-media-classifier/templates_new/)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: ~/.cache/ovos-media-classifier/output/ocp_slot_literal.csv)")
    parser.add_argument("--langs", nargs="*", default=None, metavar="LANG",
                        help="Filter to specific language codes (default: all)")
    args = parser.parse_args()

    generate_slot_literal(
        templates_dir=args.templates_dir,
        output=args.output,
        langs=args.langs,
    )


if __name__ == "__main__":
    main()
