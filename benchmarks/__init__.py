"""Benchmark + plotting suite for ovos-media-classifier.

Not packaged (lives at repo top-level, outside ``ovos_media_classifier/``).
Run from the repo root::

    python -m benchmarks.run --plots

Artifacts:
  * benchmarks/eval_set.csv   — generated labeled eval set (committed for repeatability)
  * benchmarks/results.json   — machine-readable per-backend metrics
  * benchmarks/results.md     — human-readable results table
  * docs/benchmarks/*.png     — matplotlib plots
"""
