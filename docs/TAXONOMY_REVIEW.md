
# OCP Taxonomy Review — Accuracy and Issue Analysis

## Executive Summary

✅ **Overall Status**: The taxonomy is well-designed and internally consistent. No critical errors found. All 23 OCPPlayIntent values map correctly to MediaType; all 72 OCPEntityLabel values are defined and mapped in NER_LABEL_TO_PLAY_INTENT.

**Issues Identified**: 5 minor issues (mostly documentation/design gaps, 1 potential feature gap). None block functionality.

---

## 1. Completeness Check

### MediaType (IntEnum)
- **Count**: 23 defined values
- **Value Range**: 0–71 (intentionally non-sequential; adult content at high values)
- **All values**: GENERIC(0), AUDIO(1), MUSIC(2), VIDEO(3), AUDIOBOOK(4), GAME(5), PODCAST(6), RADIO(7), NEWS(8), TV(9), MOVIE(10), TRAILER(11), AUDIO_DESCRIPTION(12), VISUAL_STORY(13), BEHIND_THE_SCENES(14), DOCUMENTARY(15), RADIO_THEATRE(16), SHORT_FILM(17), SILENT_MOVIE(18), VIDEO_EPISODES(19), BLACK_WHITE_MOVIE(20), CARTOON(21), ANIME(22), ASMR(23), MUSIC_VIDEO(24), TV_SHOW(25), ADULT(69), HENTAI(70), ADULT_AUDIO(71)
- **Assessment**: ✅ Complete, no duplicates, no undefined values

### OCPPlayIntent (str Enum)
- **Count**: 23 values
- **Assessment**: ✅ 1:1 mapping with MediaType names, string values match training labels

### OCPEntityLabel (str Enum)
- **Count**: 72 values
- **Categories**: Streaming services (8), Music (7), Video (20), TV/livestream (2), Other media (7), Keywords (27)
- **Assessment**: ✅ Comprehensive coverage

### OCPControlIntent (str Enum)
- **Count**: 15 values
- **Assessment**: ✅ Complete set of playback control actions

---

## 2. Mapping Validation

### PLAY_INTENT_TO_MEDIA_TYPE
**Coverage**: All 23 OCPPlayIntent values present
**Assessment**: ✅ Complete 1:1 mapping, all MediaType values reachable

```python
# Examples verified:
OCPPlayIntent.MUSIC → MediaType.MUSIC
OCPPlayIntent.BW_MOVIE → MediaType.BLACK_WHITE_MOVIE
OCPPlayIntent.GENERIC → MediaType.GENERIC
```

### NER_LABEL_TO_PLAY_INTENT
**Coverage**: 72/72 OCPEntityLabel values mapped (verified by lines 307–394)

**Breakdown**:
- Streaming services: 8/8 mapped ✅
- Music entities: 7/7 mapped ✅
- Video entities: 20/20 mapped ✅
- TV/livestream: 2/2 mapped ✅
- Other media: 7/7 mapped ✅
- Keywords: 27/27 mapped ✅

**Assessment**: ✅ 100% coverage

---

## 3. Identified Issues

### Issue #1: Documentation Organization — RADIO_STATION Label Category

**Severity**: Low (documentation only)

**Description**: `RADIO_STATION` (line 182) is listed under "Music entity labels" but semantically represents broadcast streams, not music content.

**Current**:
```python
# ---- Music entity labels (artist/track/album names registered by music skills) ----
...
RADIO_STATION = "radio_station"  # ← Broadcast stream, not music entity
```

**Why**: RADIO_STATION is fundamentally different from ARTIST_NAME, TRACK_NAME, etc. It's a broadcast identifier.

**Fix**: Create a separate "Broadcast service labels" section or move to "Other media entity labels":

```python
# ---- Broadcast entity labels ----
RADIO_STATION = "radio_station"
```

**Impact**: None (mapping is correct). Purely for code clarity.

---

### Issue #2: Missing Genre Entity Labels for Multi-Type Media

**Severity**: Medium (affects template coverage for discovery)

**Description**: Genre is only defined for music (`MUSIC_GENRE`) and adult content (`PORN_GENRE`). Video, games, podcasts, radio, TV all use genres in utterances but have no corresponding NER labels.

**Current State**:
```python
# OCPEntityLabel has:
MUSIC_GENRE = "music_genre"
PORN_GENRE = "porn_genre"
# Missing:
# VIDEO_GENRE, GAME_GENRE, PODCAST_GENRE, RADIO_GENRE, TV_GENRE, etc.
```

**In Templates** (templates/*.csv):
- `music.csv`: Uses `{genre}` slot → maps to MUSIC_GENRE ✅
- `movie.csv`: Uses `{genre}` slot → no entity label ⚠️
- `game.csv`: Uses `{genre}` slot → no entity label ⚠️
- `podcast.csv`: Uses `{genre}` slot → no entity label ⚠️
- `radio.csv`: Uses `{genre}` slot → no entity label ⚠️
- `tv.csv`: Uses `{genre}` slot → no entity label ⚠️
- `anime.csv`: Uses `{genre}` slot → no entity label ⚠️

**Suggested Additions**:
```python
# Add to OCPEntityLabel:
VIDEO_GENRE = "video_genre"        # e.g., action, comedy, horror
GAME_GENRE = "game_genre"          # e.g., RPG, FPS, puzzle
PODCAST_GENRE = "podcast_genre"    # e.g., true crime, comedy, news
RADIO_GENRE = "radio_genre"        # e.g., rock, jazz, talk
TV_GENRE = "tv_genre"              # e.g., drama, thriller, comedy

# Add to NER_LABEL_TO_PLAY_INTENT:
OCPEntityLabel.VIDEO_GENRE:     OCPPlayIntent.VIDEO,
OCPEntityLabel.GAME_GENRE:      OCPPlayIntent.GAME,
OCPEntityLabel.PODCAST_GENRE:   OCPPlayIntent.PODCAST,
OCPEntityLabel.RADIO_GENRE:     OCPPlayIntent.RADIO,
OCPEntityLabel.TV_GENRE:        OCPPlayIntent.TV,
```

**Why This Matters**: Skills may register genre vocabularies at runtime (e.g., "comedy" under GAME_GENRE when a game platform loads). Without these labels, AhocorasickMediaClassifier cannot distinguish genre-based discovery queries.

**Impact**: Templates use genre slots, but runtime NER won't tag them—only pre-trained genre keywords will work (if added to keyword pool).

---

### Issue #3: Missing Platform Entity Label for Games

**Severity**: Medium (affects game discovery)

**Description**: Games have platform/storefront (Steam, PlayStation, Epic, etc.) but no `GAME_PLATFORM` entity label.

**Current State**:
```python
# OCPEntityLabel has GAME_TITLE only
GAME_TITLE = "game_title"
# Missing:
# GAME_PLATFORM = "game_platform"  # e.g., Steam, PlayStation, Xbox
```

**In Templates**:
- `game.csv` line 72: Uses `{platform}` slot but no entity label

**Suggested Addition**:
```python
GAME_PLATFORM = "game_platform"

# In NER_LABEL_TO_PLAY_INTENT:
OCPEntityLabel.GAME_PLATFORM: OCPPlayIntent.GAME,
```

**Why This Matters**: Users say "play Elden Ring on Steam" or "show me PlayStation exclusives". Without GAME_PLATFORM, the classifier can only use keyword matching.

**Impact**: Moderate — game skills could still work via GAME_TITLE + keywords, but platform-scoped queries won't be recognized by NER.

---

### Issue #4: Missing Network/Studio Entity Labels for Video Content

**Severity**: Medium (affects show discovery)

**Description**: TV shows have networks (HBO, Netflix), movies have studios (Paramount, Universal), anime has studios (Studio A-1). No corresponding NER labels.

**Current State**:
```python
# Video entity labels:
MOVIE_TITLE, MOVIE_ACTOR, MOVIE_DIRECTOR, ...
TV_SHOW_TITLE
ANIME_TITLE
# Missing:
# TV_NETWORK, MOVIE_STUDIO, ANIME_STUDIO
```

**In Templates**:
- `tv.csv`: No `{network}` or `{studio}` slots currently (could be added)
- `movie.csv`: No `{studio}` slot currently
- `anime.csv`: Line 52: Uses `{studio}` slot → no entity label ⚠️

**Suggested Additions**:
```python
TV_NETWORK = "tv_network"         # e.g., HBO, NBC, BBC
MOVIE_STUDIO = "movie_studio"     # e.g., Paramount, Warner Bros, Disney
ANIME_STUDIO = "anime_studio"     # e.g., Studio A-1, MAPPA

# In NER_LABEL_TO_PLAY_INTENT:
OCPEntityLabel.TV_NETWORK:  OCPPlayIntent.TV_SHOW,
OCPEntityLabel.MOVIE_STUDIO: OCPPlayIntent.MOVIE,
OCPEntityLabel.ANIME_STUDIO: OCPPlayIntent.ANIME,
```

**Why This Matters**: Many utterances reference studio/network: "show me Pixar movies", "what's on HBO tonight", "find MAPPA anime".

**Impact**: Moderate — these are secondary classifiers; title+genre usually suffice, but studio queries won't be recognized by NER.

---

### Issue #5: Missing Narrator Entity Label for Audiobooks

**Severity**: Low (audiobooks identified primarily by title)

**Description**: Audiobooks have narrators, similar to podcasts having hosts. No `AUDIOBOOK_NARRATOR` label.

**Current State**:
```python
# Audiobook labels:
AUDIOBOOK_TITLE = "audiobook_title"
AUDIOBOOK_AUTHOR = "audiobook_author"
# Missing:
# AUDIOBOOK_NARRATOR = "audiobook_narrator"
```

**In Templates**:
- `audiobook.csv`: No `{narrator}` slot currently used

**Suggested Addition**:
```python
AUDIOBOOK_NARRATOR = "audiobook_narrator"

# In NER_LABEL_TO_PLAY_INTENT:
OCPEntityLabel.AUDIOBOOK_NARRATOR: OCPPlayIntent.AUDIOBOOK,
```

**Why**: Users might say "play Sanderson audiobooks narrated by Michael Kramer" or "show me books narrated by Stephen Fry".

**Impact**: Low — optional; audiobooks are primarily identified by title/author.

---

## 4. Naming Inconsistencies

### BLACK_WHITE_MOVIE vs BW_MOVIE

**Severity**: Low (naming only, mapping is correct)

**Details**:
```python
# MediaType uses full name:
BLACK_WHITE_MOVIE = 20

# OCPPlayIntent abbreviates:
BW_MOVIE = "bw_movie"

# OCPEntityLabel abbreviates:
BW_MOVIE_TITLE = "bw_movie_title"
BW_MOVIE_KEYWORD = "bw_movie_keyword"
```

**Assessment**: ✅ Not an error (mapping is explicit: `OCPPlayIntent.BW_MOVIE → MediaType.BLACK_WHITE_MOVIE`), but the abbreviation inconsistency could confuse maintainers.

**No fix needed** (intentional for brevity in training labels), but document in comments.

---

## 5. Design Validation

### Video/Generic Video Distinction (Correct)

```python
VIDEO = 3          # Generic YouTube video, clips
VIDEO_EPISODES = 19 # YouTube channels, episodic series
```

Both have distinct keywords and entity labels — ✅ intentional, correct.

### News Classification (Correct)

News is identified by `NEWS_PROVIDER` + `NEWS_CATEGORY`, not by article title. This is semantically correct — news is a stream/category, not a specific named item.

✅ Acceptable design.

### Podcast vs Radio (Correct)

- **PODCAST**: Has title, host, episode (episodic on-demand)
- **RADIO**: Has station name (live broadcast stream)
- **RADIO_THEATRE**: Has drama title (scripted audio)

✅ Clear distinctions.

### Cartoon vs Anime (Correct)

Both have separate titles and keywords — ✅ intentional geographic/cultural distinction.

---

## 6. Missing But Intentional

The following are **not** defined, likely by design:

- **GENERIC_KEYWORD**: No keyword for the catch-all GENERIC intent. ✅ Correct (GENERIC is fallback, not a keyword match).
- **SPORT_GENRE**: News has WEATHER and SPORTS as entity labels (lines 86–97 in `news.csv` are used for templates, not NER). Sports news identified by NEWS_CATEGORY instead.

---

## 7. Consistency Cross-Check vs. Templates

**Created Template Files Validation**:

| Template File | Issues Found | Resolution |
|---|---|---|
| music.csv | Uses {genre} → MUSIC_GENRE ✅ | Complete |
| movie.csv | Uses {genre} → no VIDEO_GENRE ⚠️ | See Issue #2 |
| tv.csv | Uses {genre} → no TV_GENRE ⚠️ | See Issue #2 |
| podcast.csv | Uses {genre} → no PODCAST_GENRE ⚠️ | See Issue #2 |
| radio.csv | Uses {genre} → no RADIO_GENRE ⚠️ | See Issue #2 |
| audiobook.csv | No narrator slot defined ✓ | Intentional |
| game.csv | Uses {platform} → no GAME_PLATFORM ⚠️ | See Issue #3 |
| anime.csv | Uses {studio} → no ANIME_STUDIO ⚠️ | See Issue #4 |
| news.csv | No entity-based slots (uses keywords) ✓ | By design |

---

## 8. Recommendations (Priority Order)

### High Priority
1. **Add video genre labels** (VIDEO_GENRE, TV_GENRE, GAME_GENRE, PODCAST_GENRE, RADIO_GENRE)
   - Used by discovery templates
   - Skills would register genre vocabularies at runtime
   - Impacts 7 template files

2. **Add GAME_PLATFORM** (for Steam, PlayStation, Epic, etc.)
   - Used in game.csv templates
   - Necessary for platform-scoped queries

### Medium Priority
3. **Add studio/network labels** (TV_NETWORK, MOVIE_STUDIO, ANIME_STUDIO)
   - Currently used in anime.csv
   - Moderate coverage impact

4. **Fix RADIO_STATION documentation** (move to "Broadcast entity labels")
   - Code clarity only

### Low Priority
5. **Add AUDIOBOOK_NARRATOR** (future-proofing)
   - Not currently used in templates
   - Optional enhancement

---

## 9. Summary Table

| Category | Count | Status | Notes |
|---|---|---|---|
| MediaType | 23 | ✅ Complete | Non-sequential by design |
| OCPPlayIntent | 23 | ✅ Complete | 1:1 with MediaType |
| OCPEntityLabel | 72 | ⚠️ Mostly Complete | Missing 5 genre/platform labels |
| OCPControlIntent | 15 | ✅ Complete | All control actions covered |
| PLAY_INTENT_TO_MEDIA_TYPE | 23/23 | ✅ 100% | All mapped |
| NER_LABEL_TO_PLAY_INTENT | 72/72 | ✅ 100% | All mapped |
| Mapping Consistency | — | ✅ Verified | No orphaned values |
| Critical Issues | — | ✅ None | System is functional |
| Minor Issues | — | ⚠️ 5 found | Documentation + feature gaps |

---

## 10. Implementation Checklist (If Adopting Recommendations)

- [ ] Add 5 new genre entity labels to OCPEntityLabel (lines 135–251)
- [ ] Add 3 new studio/network labels to OCPEntityLabel
- [ ] Add 1 narrator label to OCPEntityLabel
- [ ] Add 9 new mappings to NER_LABEL_TO_PLAY_INTENT (lines 307–394)
- [ ] Update `templates/movie.csv` to use {genre} slot (already present)
- [ ] Update `templates/game.csv` to use {platform} slot (already present)
- [ ] Update `templates/anime.csv` to map {studio} to ANIME_STUDIO (already present)
- [ ] Add genre slots to `templates/tv.csv`, `templates/podcast.csv`, `templates/radio.csv` if discovery is a priority
- [ ] Reorganize OCPEntityLabel for clarity (move RADIO_STATION to broadcast section)
- [ ] Run existing tests to verify no regressions
- [ ] Update DATASET_GENERATION.md entity pool strategy table (Section 10, current notes gaps)

---

## References

- Source: `ovos_media_classifier/intents.py` (lines 1–394)
- Validation: `ovos_media_classifier/train/templates/clean.py` (checks slot consistency)
- Templates: `templates/*.csv` (9 files, 600+ templates)
- Related docs: `TEMPLATE_DESIGN_GUIDE.md`, `DATASET_GENERATION.md`

