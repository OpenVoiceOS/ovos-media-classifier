"""Stage the committed English master templates into the ``templates_new`` layout.

`generate_slot_filled_dataset.load_templates` reads ``<dir>/<lang>/_all.csv`` with
columns ``lang,intent,template,slots``.  The repo ships flat
``training/templates/<intent>_templates.csv`` (``category,template``).  This
converts them so the slot-fill base can run offline without the online
(Wikidata) ``generate_templates`` step.

Usage::

    python -m training.stage_master_templates --lang en-us \
        --out /tmp/omc_ds/templates_new
"""
from __future__ import annotations

import argparse
import csv
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_MASTER = os.path.join(_HERE, "templates")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default="en-us")
    ap.add_argument("--out", required=True, help="templates_new root dir")
    args = ap.parse_args()

    lang_dir = os.path.join(args.out, args.lang)
    os.makedirs(lang_dir, exist_ok=True)
    rows = []
    n_files = 0
    for fn in sorted(os.listdir(_MASTER)):
        if not fn.endswith("_templates.csv"):
            continue
        intent = fn[: -len("_templates.csv")]
        n_files += 1
        with open(os.path.join(_MASTER, fn), encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                template = (r.get("template") or "").strip()
                if not template:
                    continue
                slots = ";".join(re.findall(r"\{(\w+)\}", template))
                rows.append({
                    "lang": args.lang, "intent": intent,
                    "template": template, "slots": slots,
                })
    out_csv = os.path.join(lang_dir, "_all.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["lang", "intent", "template", "slots"])
        w.writeheader()
        w.writerows(rows)
    print(f"Staged {len(rows):,} templates from {n_files} families → {out_csv}")


if __name__ == "__main__":
    main()
