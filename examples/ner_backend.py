"""Optional NER (entity-based) classifier backend — usage example.

The NER backend matches an utterance against the **real entities the user
has** (movie titles, artist names, streaming services, …) using an
Aho-Corasick automaton:

    "play Inception"  → MOVIE   (because *Inception* is a known movie_title)
    "put on Radiohead" → MUSIC  (because *Radiohead* is a known artist_name)

It is **opt-in** and requires the ``[ner]`` extra::

    pip install ovos-media-classifier[ner]

    # and, for the live media-server / HuggingFace loaders:
    pip install ovos-media-classifier[media_servers]   # Radarr/Sonarr/Lidarr/Jellyfin
    pip install ovos-media-classifier[huggingface]     # datasets

The lean ``.voc`` keyword classifier remains the default — if
``ahocorasick-ner`` is not installed the factory transparently falls back to
it, so this example always runs (just with reduced accuracy).

Run::

    python examples/ner_backend.py
"""

from ovos_media_classifier import load_media_classifier


def main() -> None:
    # Configure the NER backend with inline wordlists of "owned" entities.
    # In a real deployment these come from EntitiesContainer loaders pulling
    # live from Radarr / Sonarr / Lidarr / Jellyfin / Music Assistant / HF —
    # see EntitiesContainer.from_config() and media_classifier_entities below.
    config = {
        "media_classifier_wordlists": {
            "movie_title": ["Inception", "The Matrix", "Blade Runner"],
            "artist_name": ["Radiohead", "Miles Davis"],
            "music_streaming_service": ["Spotify", "Tidal"],
            "tv_show_title": ["Breaking Bad"],
        },
    }

    # Equivalent richer form, wiring live media servers (all keys optional):
    #
    # config = {
    #     "media_classifier_entities": {
    #         "wordlists": {"movie_title": ["Inception"]},
    #         "csv": ["/path/to/entities.csv"],
    #         "radarr":   {"url": "http://localhost:7878", "api_key": "…"},
    #         "lidarr":   {"url": "http://localhost:8686", "api_key": "…"},
    #         "jellyfin": {"url": "http://localhost:8096", "api_key": "…"},
    #         "huggingface": [{"dataset": "TigreGotico/ocp-entities"}],
    #     },
    # }

    clf = load_media_classifier(config)
    # When ahocorasick-ner is installed this is an AhocorasickMediaClassifier;
    # otherwise the factory logged a warning and returned the keyword backend.
    print(f"Loaded backend: {type(clf).__name__}\n")

    utterances = [
        "play Inception",
        "put on Radiohead",
        "play something on Spotify",
        "watch Breaking Bad",
        "what's the weather today",   # not a media query → GENERIC / NOT_OCP
    ]
    for utt in utterances:
        media_type, conf = clf.classify(utt, "en-us")
        is_ocp, _ = clf.is_ocp_query(utt, "en-us")
        genres = clf.classify_genres(utt, "en-us")
        print(f"{utt!r:40} -> {media_type} (conf={conf:.2f}, "
              f"ocp={is_ocp}, genres={genres})")


if __name__ == "__main__":
    main()
