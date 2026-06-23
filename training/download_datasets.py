"""Pre-download all datasets used by ovos-media-classifier to a local cache.

Downloads both the raw CSV intent datasets and every HuggingFace dataset
referenced by the NER and classifier training scripts.  Subsequent scripts
(gather_dataset.py, generate_synthetic.py, …) will read from the local cache
and never hit the network again.

Cache location (default ``~/.cache/ovos-media-classifier/``):
  csv/            Raw CSV downloads (keyed by URL hash)
  huggingface/    HuggingFace dataset cache (Arrow/Parquet shards)
  output/         Processed datasets written by gather_dataset.py

Override the root with ``OVOS_MEDIA_CLASSIFIER_CACHE`` env var.

Usage::

    python -m training.download_datasets
    python -m training.download_datasets --dry-run
    OVOS_MEDIA_CLASSIFIER_CACHE=/data/ocp python -m training.download_datasets
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Optional

import requests

from training import get_cache_dir, get_csv_cache_dir, get_hf_cache_dir
from training.sources import ALL_CSV_SOURCES as _CSV_SOURCES, HF_DATASETS as _HF_DATASETS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_local_path(url: str, cache_dir: str) -> str:
    stem = url.rstrip("/").split("/")[-1].split("?")[0]
    digest = hashlib.md5(url.encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"{stem}_{digest}")


def _download_csv(url: str, cache_dir: str, dry_run: bool = False) -> Optional[str]:
    local = _csv_local_path(url, cache_dir)
    if os.path.exists(local):
        size = os.path.getsize(local)
        print(f"  [cached] {url.split('/')[-1]}  ({size:,} bytes)")
        return local
    if dry_run:
        print(f"  [would download] {url}")
        return None
    print(f"  [downloading] {url}")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(local, "wb") as fh:
            fh.write(r.content)
        print(f"    → {local}  ({len(r.content):,} bytes)")
        return local
    except Exception as exc:
        print(f"    ! FAILED: {exc}")
        return None


def _download_hf(dataset_name: str, split: str, hf_cache_dir: str, dry_run: bool = False) -> bool:
    """Download a HuggingFace dataset to the local cache directory."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ! 'datasets' package not installed — skipping HF downloads.")
        print("    Install with: pip install datasets")
        return False

    # Check if already cached by probing the datasets library
    try:
        ds = load_dataset(dataset_name, split=split, cache_dir=hf_cache_dir)
        cached_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, filenames in os.walk(hf_cache_dir)
            for f in filenames
            if dataset_name.replace("/", "___") in dp
        )
        print(f"  [cached] {dataset_name} ({len(ds):,} rows, ~{cached_size // 1024:,} KB)")
        return True
    except Exception:
        pass

    if dry_run:
        print(f"  [would download] {dataset_name}")
        return False

    print(f"  [downloading] {dataset_name}  split={split}")
    try:
        ds = load_dataset(dataset_name, split=split, cache_dir=hf_cache_dir)
        print(f"    → {len(ds):,} rows")
        return True
    except Exception as exc:
        print(f"    ! FAILED: {exc}")
        return False


def download_csv_sources(
    urls: list[str],
    cache_dir: str,
    dry_run: bool = False,
) -> int:
    """Download a list of CSV URLs to *cache_dir*. Returns number of successes."""
    os.makedirs(cache_dir, exist_ok=True)
    ok = 0
    for url in urls:
        if _download_csv(url, cache_dir, dry_run=dry_run):
            ok += 1
        time.sleep(0.1)
    return ok


def download_hf_datasets(
    datasets: list[tuple[str, str]],
    hf_cache_dir: str,
    dry_run: bool = False,
) -> int:
    """Download HuggingFace datasets to *hf_cache_dir*. Returns number of successes."""
    os.makedirs(hf_cache_dir, exist_ok=True)
    ok = 0
    for dataset_name, split in datasets:
        if _download_hf(dataset_name, split, hf_cache_dir, dry_run=dry_run):
            ok += 1
    return ok


def _dir_size_mb(path: str) -> float:
    total = 0
    for dp, _, fnames in os.walk(path):
        for fn in fnames:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total / (1024 * 1024)


