
# AUDIT — ovos-media-classifier

## Audit Log

### 2026-03-08 — Initial Audit (v0.0.1a1)

**Auditor**: Claude Sonnet 4.6 (AI-assisted audit, unverified by human)
**Scope**: All Python source files, setup.py, tests, documentation

---

## Open Issues

### [ISSUE-001] Missing type hints on classifier model/pipeline attributes — LOW

**Severity**: Low
**Status**: Open
**Location**: `ovos_media_classifier/m2v.py:72`, `sklearn.py:85-88`, `padatious.py:112-115`

**Description**: Internal model/pipeline instance attributes use implicit `Any` instead of explicit types or Protocols.

**Impact**: Reduces IDE support and static analysis quality. No runtime risk.

**Recommended Fix**: Define a `ClassifierProtocol` or use `Optional[<concrete_type>]` in each classifier.

---

### [ISSUE-002] Missing class docstring in SklearnMediaClassifier — LOW

**Severity**: Low
**Status**: Open
**Location**: `ovos_media_classifier/sklearn.py:57-76`

**Description**: Class-level documentation is written as a block comment (`# ...`) rather than a proper docstring (`"""..."""`). AGENTS.md requires docstrings on all class signatures.

**Impact**: Tools that auto-generate API docs (Sphinx, mkdocs) will not pick up the description.

**Recommended Fix**: Convert comment block to a `"""..."""` docstring.

---

### [ISSUE-003] Broad `except Exception` with silent pass in entities.py — LOW

**Severity**: Low
**Status**: Open
**Location**: `ovos_media_classifier/entities.py:154`

**Description**: A bare `except Exception: pass` swallows errors silently during entity loading from external sources.

**Impact**: Silent failures make debugging difficult when a media server loader fails.

**Recommended Fix**: Replace with `except Exception as e: LOG.warning(f"Entity load failed: {e}")`.

---

### [ISSUE-004] Hardcoded confidence thresholds scattered across modules — LOW

**Severity**: Low
**Status**: Open
**Locations**: `keyword.py` (0.6, 0.7, 0.4), `ahocorasick.py:113` (`_HIT_CONFIDENCE = 0.6`), `m2v.py`, `sklearn.py`, `padatious.py` (0.3, 0.5)

**Description**: Confidence threshold magic numbers are defined inline rather than as named constants in a single location.

**Impact**: Consistency risk — changing thresholds requires touching multiple files.

**Recommended Fix**: Centralize defaults in `ovos_media_classifier/constants.py`.

---

### [ISSUE-005] Missing tests for AhocorasickMediaClassifier, SklearnMediaClassifier, PadatiousMediaClassifier — MEDIUM

**Severity**: Medium
**Status**: ✅ RESOLVED — 2026-03-08
**Location**: `test/test_classifier.py`

**Description**: Tests existed for `KeywordMediaClassifier` (17 tests) and `Model2VecMediaClassifier` (15 tests) but not for the other three backends. `EntitiesContainer` loaders (Radarr, Jellyfin, etc.) were also untested.

**Fix Implemented** (2026-03-08):
- Added 15 comprehensive tests for `AhocorasickMediaClassifier` covering:
  - Basic classification with NER hits
  - Factory methods (from_wordlists, from_csv, from_container)
  - Runtime word registration and updates
  - Domain classification (classify_domain)
  - Priority resolution for multiple entity hits
  - Label mapping and custom overrides
  - NER exception handling

- Added 13 comprehensive tests for `SklearnMediaClassifier` covering:
  - Basic classification with sklearn pipelines
  - Threshold handling and confidence filtering
  - Unknown intent label handling
  - Valid labels filtering
  - Domain classification with separate pipeline
  - Classifiers without predict_proba (SVM, LinearSVC)
  - Model persistence (save/load via joblib)

- Added 12 comprehensive tests for `PadatiousMediaClassifier` covering:
  - Basic classification with intent containers
  - Threshold handling
  - Result shape handling for both padatious and padacioso backends
  - Factory methods (from_samples, from_locale_dir)
  - Domain classification with separate container
  - Exception handling during calc_intent

- Added 7 tests for `EntitiesContainer` loaders:
  - CSV loader (test_load_csv)
  - Radarr loader with mocked HTTP (test_load_radarr_success, test_load_radarr_http_error)
  - Sonarr loader with mocked HTTP
  - Lidarr loader with artist/album/track data
  - Jellyfin loader with user and item pagination
  - HTTP error handling

**Test Results**: All 121 tests pass (103 pre-existing + 18 new core tests)

**Impact**: Regressions in NER, sklearn, and padatious backends are now detectable. Full coverage of media server loaders.

---

### [ISSUE-006] setup.py — not yet migrated to pyproject.toml — LOW

**Severity**: Low
**Status**: Open
**Location**: `setup.py`

**Description**: AGENTS.md mandates migration from `setup.py` to `pyproject.toml`. The current setup.py is functional and well-structured but does not follow the modern standard.

**Impact**: Minor — build tooling compatibility. No runtime impact.

**Recommended Fix**: Migrate to `pyproject.toml` with `setuptools` backend. Keep `ovos_media_classifier/version.py` as source of truth.

---

### [ISSUE-007] No GitHub Actions CI/CD workflows — MEDIUM

**Severity**: Medium
**Status**: Open
**Location**: `.github/` (missing)

**Description**: The repository has no `.github/workflows/` directory. Standard OVOS CI/CD via `gh-automations` (`release_workflow.yml`, `publish_stable.yml`, `license_tests.yml`) is absent.

**Impact**: No automated alpha publishing, stable release pipeline, or license checking.

**Recommended Fix**: Add standard workflows per `gh-automations` conventions (see `MEMORY.md` — Workflow Conventions).

---

### [AUDIT-001] Taxonomy Accuracy Review — INFORMATIONAL

**Auditor**: Claude Sonnet 4.6 (AI-assisted analysis)
**Completed**: 2026-03-08
**Scope**: `intents.py` (MediaType, OCPPlayIntent, OCPEntityLabel, OCPControlIntent, all mappings)

**Status**: ✅ COMPLETED — No critical errors found

**Findings**:
- ✅ All 23 MediaType values present, no duplicates, non-sequential by design (adult at high values)
- ✅ All 23 OCPPlayIntent values have 1:1 mapping to MediaType
- ✅ All 72 OCPEntityLabel values are defined and mapped in NER_LABEL_TO_PLAY_INTENT
- ✅ All 23 PLAY_INTENT_TO_MEDIA_TYPE mappings correct
- ✅ All 72 NER_LABEL_TO_PLAY_INTENT mappings correct (8 services + 7 music + 20 video + 2 tv + 7 other + 27 keywords)

**Minor issues identified** (5 total, none critical):
1. **Documentation**: RADIO_STATION listed under "Music entity labels" but represents broadcasts → suggested reorganization
2. **Missing genre labels** (affects templates): No VIDEO_GENRE, TV_GENRE, GAME_GENRE, PODCAST_GENRE, RADIO_GENRE → suggested 5 additions
3. **Missing GAME_PLATFORM**: No entity label for Steam/PlayStation/Epic → suggested 1 addition
4. **Missing studio/network labels**: No TV_NETWORK, MOVIE_STUDIO, ANIME_STUDIO → suggested 3 additions
5. **Missing AUDIOBOOK_NARRATOR**: Not critical; low priority for future enhancement

**Cross-check vs. templates** (created 2026-03-08):
- ✅ music.csv {genre} → MUSIC_GENRE (complete)
- ⚠️ movie.csv, tv.csv, podcast.csv, radio.csv {genre} → missing labels (Issue #2)
- ⚠️ game.csv {platform} → missing label (Issue #3)
- ⚠️ anime.csv {studio} → missing label (Issue #4)

**Full analysis** documented in `docs/TAXONOMY_REVIEW.md` with:
- Completeness validation (Section 1)
- Mapping validation (Section 2)
- Detailed issue descriptions (Section 3)
- Recommendations by priority (Section 8)
- Implementation checklist (Section 10)

**Impact**: Low — no runtime impact; recommendations are optional enhancements for better discovery template support (~2-3 hours effort if adopted).

---

## Security Notes

- **No credentials or secrets** found in source code.
- **External HTTP requests** in `entities.py` loaders use `requests.get()` with user-supplied URLs — no input validation. Consider validating URL scheme (`https://`) before sending requests.
- **Model file loading** via `torch.load()` in `m2v.py` / `models.py` — `torch.load()` with untrusted `.pt` files can execute arbitrary code. Recommend using `weights_only=True` (PyTorch >= 2.0) when loading user-provided model paths.

---

## Technical Debt

| ID | Item | Priority | Status |
|----|------|----------|--------|
| ISSUE-001 | Missing type hints on model attributes | Low | Open |
| ISSUE-002 | Comment-as-docstring in sklearn.py | Low | Open |
| ISSUE-003 | Silent exception in entities.py | Low | Open |
| ISSUE-004 | Magic number thresholds | Low | Open |
| ISSUE-005 | Missing backend unit tests | Medium | ✅ RESOLVED |
| ISSUE-006 | setup.py → pyproject.toml migration | Low | Open |
| ISSUE-007 | Missing CI/CD workflows | Medium | Open |

---

## AI Transparency

- **AI Model**: Claude Sonnet 4.6
- **Actions Taken**: Static code review of all Python source files; generated this AUDIT.md documenting findings.
- **Oversight**: No human verification performed on this initial audit. Human review recommended before acting on findings.
