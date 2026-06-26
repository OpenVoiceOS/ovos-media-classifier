"""Download, normalize, and build the OCP media-classifier training dataset.

This script fetches the same OVOS intent CSV files used by the general
intent classifier but re-labels them for the OCP two-level hierarchy:

  domain   → OCPDomain    ("ocp_play" | "ocp_control" | "not_ocp")
  label    → raw media label ("music" | "movie" | "podcast" | …)
             OCPControlIntent ("pause" | "next" | "resume" | …)
             "not_ocp"    (for all non-OCP utterances)

Output files (written to ``output/``)
--------------------------------------
``ocp_dataset.csv``
    Training CSV with columns:
    lang, domain, intent, binary_label, playback_label, media_label, sentence

``ocp_dataset_play_only.csv``
    Only ocp_play rows — used to train the play/intent head in isolation.

``by_lang/ocp_<lang>.csv``
    Per-language subsets of the full dataset.

``dataset_plots/``
    Exploratory plots (domain/intent distribution, language coverage).

Usage::

    python -m training.gather_dataset
    # or
    python gather_dataset.py
"""

import hashlib
import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

matplotlib.use("Agg")

from training import get_csv_cache_dir, get_output_dir
from training.sources import (
    CSV_SOURCES as _csv_sources_list,
    MUSIC_CSV_SOURCES as _music_sources,
    GITHUB_CSV_SOURCES as _github_sources,
    AUDIO_INTENTS as _AUDIO_INTENTS,
    VIDEO_INTENTS as _VIDEO_INTENTS,
)

OUTPUT_DIR: str = get_output_dir()
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_DIR = get_csv_cache_dir()
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# OCP intent mappings
# ---------------------------------------------------------------------------

from ovos_media_classifier.intents import (
    OCPDomain, OCPControlIntent,
    LABEL_TO_MEDIA_TYPE,
)

# Skill domain identifiers that are OCP play skills
OCP_PLAY_SKILL_DOMAINS = {
    "ocp",
    "ovos-skill-local-media.openvoiceos",
    "ovos-skill-audiobook.openvoiceos",
    "ovos-skill-news.openvoiceos",
    "ovos-skill-radio.openvoiceos",
}

# Intent-name substring → raw media label (a ``LABEL_TO_MEDIA_TYPE`` key) for
# ocp:play intents and music datasets.
_PLAY_INTENT_PATTERNS: list[tuple[str, str]] = [
    # Music query datasets (template → music label, ocp_play domain)
    ("play",          "generic"),
    ("music",         "music"),
    ("podcast",       "podcast"),
    ("radio",         "radio"),
    ("audiobook",     "audiobook"),
    ("news",          "news"),
    ("movie",         "movie"),
    ("film",          "movie"),
    ("tv_show",       "tv_show"),
    ("iptv",          "tv"),
    ("live_tv",       "tv"),
    ("tv_channel",    "tv"),
    ("series",        "video_episodes"),
    ("anime",         "anime"),
    ("cartoon",       "cartoon"),
    ("documentary",   "documentary"),
    ("short",         "short_film"),
    ("silent",        "silent_movie"),
    ("bw",            "bw_movie"),
    ("game",          "game"),
    ("asmr",          "asmr"),
    ("audio_descrip", "audio_description"),
    ("audio",         "audio"),
    ("video",         "video"),
    ("music_video",   "music_video"),
    ("trailer",       "trailer"),
    ("behind",        "behind_the_scenes"),
    ("bts",           "behind_the_scenes"),
]

# Intent name patterns → OCPControlIntent label
_CONTROL_INTENT_PATTERNS: list[tuple[str, str]] = [
    ("pause",          OCPControlIntent.PAUSE.value),
    ("resume",         OCPControlIntent.RESUME.value),
    ("next",           OCPControlIntent.NEXT.value),
    ("prev",           OCPControlIntent.PREVIOUS.value),
    ("stop",           OCPControlIntent.STOP.value),
    ("media_stop",     OCPControlIntent.STOP.value),
    ("open",           OCPControlIntent.OPEN.value),
    ("like_song",      OCPControlIntent.LIKE_SONG.value),
    ("play_favorites", OCPControlIntent.PLAY_FAVORITES.value),
    ("save_game",      OCPControlIntent.SAVE_GAME.value),
    ("load_game",      OCPControlIntent.LOAD_GAME.value),
    ("shuffle",        OCPControlIntent.SHUFFLE.value),
    ("repeat",         OCPControlIntent.REPEAT.value),
    ("seek_forward",   OCPControlIntent.SEEK_FORWARD.value),
    ("seek_backward",  OCPControlIntent.SEEK_BACKWARD.value),
]


def _intent_to_ocp_label(domain: str, intent: str) -> tuple[str, str]:
    """Return (ocp_domain, ocp_intent) for a raw (domain, intent) pair.

    Returns ("not_ocp", "not_ocp") for non-OCP utterances.
    """
    domain_lc = domain.lower()
    intent_lc = intent.lower()

    # Is this an explicit OCP play skill?
    is_ocp_domain = domain_lc in OCP_PLAY_SKILL_DOMAINS or domain_lc.startswith("ocp")

    if is_ocp_domain:
        # Check for control intents first (higher priority)
        for pattern, ctrl_label in _CONTROL_INTENT_PATTERNS:
            if pattern in intent_lc:
                return OCPDomain.OCP_CONTROL.value, ctrl_label
        # Then play intents
        for pattern, play_label in _PLAY_INTENT_PATTERNS:
            if pattern in intent_lc:
                return OCPDomain.OCP_PLAY.value, play_label
        # Fallback: generic play
        return OCPDomain.OCP_PLAY.value, "generic"

    # Check if a non-OCP skill happens to have a known control intent name
    for pattern, ctrl_label in _CONTROL_INTENT_PATTERNS:
        if pattern in intent_lc and domain_lc in ("ovos-skill-core.openvoiceos",
                                                    "mycroft-skill-core"):
            return OCPDomain.OCP_CONTROL.value, ctrl_label

    return OCPDomain.NOT_OCP.value, OCPDomain.NOT_OCP.value


# ---------------------------------------------------------------------------
# Extra label derivation — binary_label / playback_label / media_label
# ---------------------------------------------------------------------------


def _to_binary_label(domain: str) -> str:
    """ocp / not_ocp."""
    return "ocp" if domain in ("ocp_play", "ocp_control") else "not_ocp"


def _to_playback_label(domain: str, intent: str) -> str:
    """audio / video / undefined — meaningful only for ocp_play rows."""
    if domain != "ocp_play":
        return "undefined"
    if intent in _AUDIO_INTENTS:
        return "audio"
    if intent in _VIDEO_INTENTS:
        return "video"
    return "undefined"


def _to_media_label(domain: str, intent: str) -> str:
    """Fine-grained media type for ocp_play; 'not_ocp' for everything else."""
    return intent if domain == "ocp_play" else "not_ocp"


# ---------------------------------------------------------------------------
# Dataset sources (same as the general gather_dataset.py)
# ---------------------------------------------------------------------------

def _cached_path(url: str) -> str:
    stem = url.rstrip("/").split("/")[-1].split("?")[0]
    digest = hashlib.md5(url.encode()).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f"{stem}_{digest}")


def _fetch(url: str) -> str:
    local = _cached_path(url)
    if os.path.exists(local):
        return local
    print(f"  Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(local, "wb") as fh:
        fh.write(response.content)
    return local


# URL lists live in sources.py — imported at the top of this file.
# Local aliases for backward compatibility within this module.
_csv_sources = _csv_sources_list

_weather_sources = [s for s in _csv_sources_list if "weather" in s]
_pt_csv = [s for s in _csv_sources_list if "pt-PT" in s]
_ca_csv = [s for s in _csv_sources_list if "testset-ca" in s]
_es_csv = [s for s in _csv_sources_list if "testset-es" in s]
_nl_csv = [s for s in _csv_sources_list if "testset-nl" in s]


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return (
        str(text).lower().replace(",", "").split("/")[-1]
        .replace("  ", " ").strip().strip('"')
    )


def normalize_domain(text: str) -> str:
    n = str(text).strip().strip('"')
    return n.replace("skill-ovos", "ovos-skill").split(":", 1)[0]


def normalize_intent(text: str) -> str:
    return str(text).strip().strip('"').split(":", 1)[-1]


def extract_domain(text: str) -> str:
    return str(text).strip().strip('"').split(":")[0]


# ---------------------------------------------------------------------------
# Per-source loader
# ---------------------------------------------------------------------------

def load_source(url: str, forced_domain: Optional[str] = None,
                forced_intent: Optional[str] = None) -> pd.DataFrame:
    """Download and normalise one source CSV → OCP schema.

    Returns DataFrame with columns: lang, domain, intent, sentence.
    Empty on failure.
    """
    try:
        fpath = url if os.path.exists(url) else _fetch(url)
        df = pd.read_csv(fpath)

        lang = None

        if not os.path.exists(url):
            if "github" in url:
                lang = url.split("_")[-1].split(".csv")[0]
            elif url in _music_sources:
                lang = "en"
            elif url in _pt_csv:
                lang = "pt"
            elif url in _es_csv:
                lang = "es"
            elif url in _ca_csv:
                lang = "ca"
            elif url in _nl_csv:
                lang = "nl"
            elif url in _weather_sources:
                lang = "en"

            if url in _music_sources or "ocp_media" in url:
                df["domain"] = "ocp"
                df["intent"] = "music"
                for col in ("template", "sentence", "utterance", "example"):
                    if col in df.columns:
                        df = df.rename(columns={col: "sentence"})
                        break
            elif url in _weather_sources:
                df["domain"] = "ovos-skill-weather.openvoiceos"

            for old, new in (
                ("synthetic_query", "sentence"),
                ("example", "sentence"),
                ("utterance", "sentence"),
                ("label", "intent"),
                ("skill", "domain"),
            ):
                if old in df.columns and new not in df.columns:
                    df = df.rename(columns={old: new})

            if "domain" not in df.columns and "intent" in df.columns:
                df["domain"] = df["intent"].apply(extract_domain)

        existing_lang = df["lang"].copy() if "lang" in df.columns else None

        for col in ("domain", "intent", "sentence"):
            if col not in df.columns:
                return pd.DataFrame()

        df = df[["domain", "intent", "sentence"]].dropna(subset=["sentence"])
        df["domain"] = df["domain"].apply(normalize_domain)
        df["intent"] = df["intent"].apply(normalize_intent)
        df["sentence"] = df["sentence"].apply(normalize_text)
        df = df[df["sentence"].str.strip().astype(bool)]
        df = df[df["sentence"] != "nan"]

        if lang:
            df["lang"] = lang
        elif existing_lang is not None:
            df["lang"] = existing_lang
        else:
            df["lang"] = "en"

        # Apply forced domain/intent overrides (for music query datasets)
        if forced_domain:
            df["domain"] = forced_domain
        if forced_intent:
            df["intent"] = forced_intent

        # Map to OCP labels
        ocp_labels = df.apply(
            lambda r: _intent_to_ocp_label(r["domain"], r["intent"]), axis=1
        )
        df["domain"] = [lbl[0] for lbl in ocp_labels]
        df["intent"] = [lbl[1] for lbl in ocp_labels]

        # Derive extra label columns
        df["binary_label"] = df["domain"].apply(_to_binary_label)
        df["playback_label"] = df.apply(
            lambda r: _to_playback_label(r["domain"], r["intent"]), axis=1
        )
        df["media_label"] = df.apply(
            lambda r: _to_media_label(r["domain"], r["intent"]), axis=1
        )

        return df[["lang", "domain", "intent", "binary_label", "playback_label",
                    "media_label", "sentence"]]

    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return pd.DataFrame(columns=["lang", "domain", "intent", "binary_label",
                                      "playback_label", "media_label", "sentence"])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_ocp_dataset(df: pd.DataFrame, plots_dir: str) -> None:
    """Generate and save exploratory plots for the OCP intent dataset."""
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Domain distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    domain_counts = df["domain"].value_counts()
    ax.bar(domain_counts.index, domain_counts.values, color=["steelblue", "darkorange", "grey"])
    for i, (_, v) in enumerate(domain_counts.items()):
        ax.text(i, v + domain_counts.max() * 0.01, f"{v:,}", ha="center", fontsize=9)
    ax.set_ylabel("Examples")
    ax.set_title(f"OCP domain distribution (total {len(df):,})")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "domain_distribution.png"), dpi=120)
    plt.close(fig)

    # 2. Play intent distribution
    play_df = df[df["domain"] == "ocp_play"]
    intent_counts = play_df["intent"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(intent_counts) * 0.4)))
    bars = ax.barh(intent_counts.index, intent_counts.values, color="steelblue")
    for bar, val in zip(bars, intent_counts.values):
        ax.text(bar.get_width() + intent_counts.max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
    ax.set_xlabel("Examples")
    ax.set_title(f"ocp_play intent distribution ({len(play_df):,} examples)")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "play_intent_distribution.png"), dpi=120)
    plt.close(fig)

    # 3. Control intent distribution
    ctrl_df = df[df["domain"] == "ocp_control"]
    if len(ctrl_df):
        ctrl_counts = ctrl_df["intent"].value_counts().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(3, len(ctrl_counts) * 0.4)))
        bars = ax.barh(ctrl_counts.index, ctrl_counts.values, color="darkorange")
        for bar, val in zip(bars, ctrl_counts.values):
            ax.text(bar.get_width() + ctrl_counts.max() * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=8)
        ax.set_xlabel("Examples")
        ax.set_title(f"ocp_control intent distribution ({len(ctrl_df):,} examples)")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "control_intent_distribution.png"), dpi=120)
        plt.close(fig)

    # 4. Language distribution
    lang_counts = df["lang"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(lang_counts) * 0.35)))
    bars = ax.barh(lang_counts.index, lang_counts.values, color="steelblue")
    for bar, val in zip(bars, lang_counts.values):
        ax.text(bar.get_width() + lang_counts.max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
    ax.set_xlabel("Examples")
    ax.set_title(f"Language distribution (total {len(df):,})")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "lang_distribution.png"), dpi=120)
    plt.close(fig)

    # 5. Language × domain heatmap
    heat_df = (
        df.groupby(["lang", "domain"])
        .size()
        .unstack(fill_value=0)
    )
    heat_norm = heat_df.div(heat_df.sum(axis=1), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(heat_df) * 0.5)))
    sns.heatmap(heat_df, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                linewidths=0.3, annot_kws={"size": 7})
    axes[0].set_title("Examples per lang × domain (raw)")
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    sns.heatmap(heat_norm, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1],
                linewidths=0.3, annot_kws={"size": 7})
    axes[1].set_title("Examples per lang × domain (row-normalised)")
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "lang_domain_heatmap.png"), dpi=120)
    plt.close(fig)

    print(f"Dataset plots saved → {plots_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    """Build and return the normalised OCP dataset from all CSV sources."""
    all_sources = (
        [(url, None, None) for url in _csv_sources + _github_sources +
         _pt_csv + _ca_csv + _es_csv + _nl_csv + _weather_sources]
        + [(url, "ocp", "music") for url in _music_sources]
    )
    print(f"Fetching {len(all_sources)} sources …")
    frames = [load_source(url, fd, fi) for url, fd, fi in all_sources]
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Deduplicated {before - len(df)} rows  ({len(df)} remain)")
    return df


