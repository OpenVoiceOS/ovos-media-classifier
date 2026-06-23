# TODO — ovos-media-classifier

## Open issues

No remote yet.

## Gaps

- [ ] Not pushed to GitHub (local-only, no git remote).
- [ ] No CI. Missing standard OpenVoiceOS/gh-automations workflows: build-tests, coverage, license-check, release_workflow, publish_stable.
- [ ] `pyproject.toml` declares `readme = "README.md"` but no `README.md` exists at repo root (content lives in `docs/README.md`).
- [ ] No `.gitignore`.
- [ ] Committed scratch artifacts tracked in git: `.coverage`, `ovos_media_classifier.egg-info/`, `ovos_media_classifier/__pycache__/*.pyc`.
- [ ] `[tool.uv.sources]` pins `guided-categorical-embeddings` to a machine-specific local editable path (`../../Machine Learning Workspace/...`), not portable for other contributors.
- [ ] No lint/typecheck config committed despite a `.ruff_cache/` being present.
- [ ] Numerous root-level loose docs (AUDIT.md, COMPLETION_SUMMARY.md, SUGGESTIONS.md, MAINTENANCE_REPORT.md, QUICK_FACTS.md, FAQ.md, dataset.py) sit outside `docs/`; consolidate or move under `docs/`.

## Code TODOs

- [ ] `ovos_media_classifier/train/generate_from_ocp_templates.py:34` — `TODO - entity lists`
