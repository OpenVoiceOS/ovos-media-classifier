"""Training utilities for ovos-media-classifier.

The ``get_cache_dir()`` function is the single source of truth for where all
downloaded data (HuggingFace datasets and raw CSV files) are stored locally.
Override the location with the ``OVOS_MEDIA_CLASSIFIER_CACHE`` environment
variable before running any training script.
"""
import os


def get_cache_dir() -> str:
    """Root cache directory for all downloaded datasets.

    Defaults to ``~/.cache/ovos-media-classifier``.
    Override with ``OVOS_MEDIA_CLASSIFIER_CACHE`` env var.
    """
    default = os.path.join(os.path.expanduser("~"), ".cache", "ovos-media-classifier")
    return os.environ.get("OVOS_MEDIA_CLASSIFIER_CACHE", default)


def get_hf_cache_dir() -> str:
    """Sub-directory used as the HuggingFace datasets cache."""
    return os.path.join(get_cache_dir(), "huggingface")


def get_csv_cache_dir() -> str:
    """Sub-directory for raw downloaded CSV files."""
    return os.path.join(get_cache_dir(), "csv")


def get_output_dir() -> str:
    """Sub-directory for processed/generated output CSVs."""
    return os.path.join(get_cache_dir(), "output")
