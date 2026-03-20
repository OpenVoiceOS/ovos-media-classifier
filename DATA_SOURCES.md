# Entity Data Sources for OCP Media Classifier

_Last updated: 2026-03-09 — reflects actual API testing and completed downloads_

---

## Current Entity Pool Status

**`scripts/ocp_entities.csv`**: 168,658 rows (up from 17,942 — **+9.4×**)

| Label | Count | Status |
|-------|-------|--------|
| `radio_station` | 43,433 | ✅ Target exceeded (radio-browser.info) |
| `audiobook_title` | 27,347 | ✅ Target exceeded (LibriVox + Gutendex) |
| `pornstar` | 21,174 | ✅ Target exceeded (PornHub Webmasters API) |
| `adult_title` | 12,280 | ✅ Target exceeded (PornHub video titles) |
| `movie_title` | 12,236 | ✅ Target exceeded (Wikidata/IMDB + Whisparr) |
| `audiobook_author` | 10,588 | ✅ Target exceeded (LibriVox + Gutendex) |
| `radio_genre` | 10,063 | ✅ Target exceeded (radio-browser.info) |
| `tv_show_title` | 5,240 | ✅ Good coverage (Wikidata) |
| `movie_actor` | 5,000 | ✅ Good coverage (Wikidata) |
| `movie_director` | 5,000 | ✅ Good coverage (Wikidata) |
| `cartoon_title` | 4,314 | ✅ Good coverage (Wikidata) |
| `album_name` | 3,685 | ✅ Good coverage |
| `anime_title` | 1,139 | 🟡 Partial (AniList) |
| `game_title` | 990 | 🟡 Partial (SteamSpy) |
| `hentai_title` | 900 | ✅ (AniList adult filter) |
| `adult_streaming_service` | 562 | ✅ (studios + curated) |
| `artist_name` | 383 | 🔴 Low — needs more data |
| `porn_genre` | 214 | ✅ (PornHub categories + curated) |
| `anime_studio` | 189 | 🟡 Partial (AniList) |
| `documentary_title` | 27 | 🔴 Low |
| `podcast_title` | 17 | 🔴 Low — needs API key |

### Gap Analysis (Updated)

| Entity | Unresolved synthetic slots | Priority | Status |
|--------|---------------------------|----------|--------|
| `audiobook_title` | 129 | 🔴 Critical | ✅ RESOLVED — 27k titles |
| `audiobook_author` | 25 | 🟠 High | ✅ RESOLVED — 10k authors |
| `audiobook_narrator` | — | 🟠 High | ✅ RESOLVED — LibriVox narrators |
| `radio_station` | 35 | 🔴 Critical | ✅ RESOLVED — 43k stations |
| `radio_genre` | 33 | 🔴 Critical | ✅ RESOLVED — 10k genre tags |
| `adult_title` | — | 🟠 High | ✅ RESOLVED — 3,877 titles |
| `pornstar` | — | 🟠 High | ✅ RESOLVED — 3,295 performers |
| `hentai_title` | — | 🟠 High | ✅ RESOLVED — 900 titles |
| `porn_genre` | — | 🟠 High | ✅ RESOLVED — 102 genre tags |
| `adult_streaming_service` | — | 🟡 Medium | ✅ RESOLVED — 562 services/studios |
| `podcast_show` | 25 | 🟠 High | 🔴 Needs Listen Notes API key |
| `news_provider` | 24 | 🟠 High | 🔴 Needs NewsAPI key |
| `anime_title` | 42 | 🟠 High | 🟡 Partial — 1,139 titles |
| `game_title` | 27 | 🟡 Medium | 🟡 Partial — 990 games |

---

## Script: `scripts/scrape_entity_sources.py`

Downloads entity pools from free public APIs. Run with:

```bash
# All sources (no API key required)
python scripts/scrape_entity_sources.py

# Specific sources
python scripts/scrape_entity_sources.py --sources gutendex,librivox,anilist

# Dry-run (count only)
python scripts/scrape_entity_sources.py --dry-run

# Write intermediate CSVs without modifying ocp_entities.csv
python scripts/scrape_entity_sources.py --no-merge
```

Intermediate CSVs are cached at `~/.cache/ovos-media-classifier/entities/`.

---

## Data Sources — Status After Testing

### 1. LibriVox ✅ WORKING
- **Source key**: `librivox`
- **URL**: `https://librivox.org/api/feed/audiobooks`
- **Result**: 21,556 audiobook_title + 7,051 audiobook_author + narrators (28,607 rows total)
- **Notes**: Fetches all ~21,900 books. Free, no auth. ~500ms delay between pages.

### 2. Gutendex (Project Gutenberg) ✅ WORKING
- **Source key**: `gutendex`
- **URL**: `https://gutendex.com/books/`
- **Result**: ~6,264 audiobook_title + 3,571 audiobook_author per 200-page run
- **Notes**: 78k books total, paginated 32 books/page. `--sources gutendex` runs
  200 pages by default (`max_pages=200`). Increase to cover all 78k.

### 3. Radio Garden (Search API) ⚠️ PARTIAL
- **Source key**: `radio_garden`
- **URL**: `https://radio.garden/api/search?q=<term>`
- **Result**: ~547 radio_station + 31 radio_genre
- **Notes**: The bulk channel list (`/api/ara/content/channels`) is no longer
  accessible. The script uses 30 genre search terms (max 20 results each).
  For broader coverage, use the TuneIn scraping approach below.

### 4. AniList GraphQL ✅ WORKING
- **Source key**: `anilist`
- **URL**: `https://graphql.anilist.co`
- **Result**: ~900 anime_title + 166 anime_studio per run (rate-limited at page 18)
- **Notes**: Rate limit ~90 req/min. Script uses 1.2s delays with 429 backoff.
  Run multiple times or wait 1 min between runs to get more pages.

### 5. SteamSpy ✅ WORKING (Steam API deprecated)
- **Source key**: `steam`
- **URL**: `https://steamspy.com/api.php?request=all`
- **Result**: ~990 game_title
- **Notes**: Steam's `ISteamApps/GetAppList/v2/` returns HTTP 404. SteamSpy
  provides top ~1k games sorted by player count. For comprehensive game lists,
  use IGDB API (free registration required).

### 6. Anime Offline Database ❌ URL BROKEN
- **Source key**: `anime_offline_db`
- **URL**: Various GitHub raw URLs (all returning 404)
- **Status**: Repository structure may have changed. Script tries 4 URL patterns
  and falls back gracefully.
- **Alternative**: Use `anilist` source instead.

### 7. Open Library ✅ WORKING
- **Source key**: `open_library`
- **URL**: `https://openlibrary.org/subjects/<subject>.json`
- **Result**: ~8k entries across 8 genre subjects
- **Notes**: Complements LibriVox/Gutendex with genre-filtered coverage.

---

## Remaining Gaps — Sources Needing API Keys

### Radio Stations ✅ RESOLVED
**Current**: 547 (Radio Garden) + 43,856 (radio-browser.info) = **44,403 total**
**Target**: 5,000+ → ✅ exceeded

#### Option A: TuneIn (Web Scraping)
```python
# Scrape category pages without official API
# https://tunein.com/radio/stations/ (format/genre browsing)
# Rotate user-agents, respect robots.txt
```

#### Option B: radio-browser.info (Free API, No Auth) ✅ IMPLEMENTED
- **Source key**: `radio_browser`
- **URL**: `https://de1.api.radio-browser.info/json/stations`
- **Auth**: None required
- **Result**: 43,856 station names + 10,063 genre tags in a single request
- **Usage**: `python scripts/scrape_entity_sources.py --sources radio_browser`

```python
def fetch_radio_browser() -> List[Dict[str, str]]:
    """https://www.radio-browser.info/ — community radio database, no auth"""
    data = _get("https://de1.api.radio-browser.info/json/stations?limit=50000&hidebroken=true")
    rows = []
    for station in data:
        name = station.get("name", "").strip()
        tags = station.get("tags", "").strip()  # comma-separated genres
        if name:
            rows.append(_row(title=name, ocp_label="radio_station",
                             media_type="RADIO", genre=tags, source="radio_browser"))
        for tag in tags.split(","):
            tag = tag.strip()
            if tag:
                rows.append(_row(title=tag, ocp_label="radio_genre", ...))
    return rows
```

### Podcast Shows
**Current**: 17 podcast_title entries
**Target**: 5,000+

#### Option A: Podcastindex.org (Free, No Auth)
- **URL**: `https://api.podcastindex.org/api/1.0/search/byterm?q=*`
- **Auth**: API key (free registration at podcastindex.org)
- **Size**: 4M+ podcasts

#### Option B: Listen Notes API
- **URL**: `https://www.listennotes.com/api/`
- **Auth**: Free tier (10 req/min, 600/month)

### News Providers
**Current**: 0 explicit `news_provider` entries (labels use title-based matching)
**Target**: 500+

#### Option A: NewsAPI
- **URL**: `https://newsapi.org/v2/top-headlines/sources`
- **Auth**: Free API key at newsapi.org
- **Action**: Add `fetch_newsapi(api_key=os.environ.get("NEWSAPI_KEY"))` function

#### Option B: Wikipedia (No Auth)
- Scrape `https://en.wikipedia.org/wiki/List_of_news_media_outlets`
- No API key needed, but requires HTML parsing

### Anime (Extended Coverage)
**Current**: 1,139 anime_title (AniList hit rate limit)
**Target**: 10,000+

#### Continue AniList (free, just slower)
```bash
# Run multiple times to collect more pages (rate-limited per run)
for i in {1..5}; do
    python scripts/scrape_entity_sources.py --sources anilist --no-merge
    sleep 60
done
# Then merge all intermediate CSVs
```

### Game Titles (Extended Coverage)
**Current**: 990 game_title (SteamSpy top games)
**Target**: 10,000+

#### IGDB API (Free Registration)
- **URL**: `https://api.igdb.com/v4/games`
- **Auth**: Free Twitch developer account + Client-ID/Secret
- **Size**: 200,000+ games
- **Action**: Add `fetch_igdb(client_id, access_token)` function

---

## Implementation Priority

### Phase 1 — No Auth (Run Now)
```bash
# ✅ Already complete
python scripts/scrape_entity_sources.py --sources librivox,anilist,radio_garden,steam

# Add radio-browser.info (action item: add to script)
python scripts/scrape_entity_sources.py --sources radio_browser

# Run gutendex more thoroughly (all 2400+ pages)
python scripts/scrape_entity_sources.py --sources gutendex  # increase max_pages
```

### Phase 2 — Free API Keys
1. **podcastindex.org** — register free → `PODCAST_INDEX_KEY` env var → `podcast_title`
2. **newsapi.org** — register free → `NEWSAPI_KEY` env var → `news_provider`
3. **IGDB** — register Twitch dev → `IGDB_CLIENT_ID` + `IGDB_SECRET` → `game_title`

### Phase 3 — Web Scraping (No Key)
1. **radio-browser.info** API (no auth, 30k+ stations) — just needs adding to script
2. **Wikipedia news list** — HTML scraping for `news_provider`
3. **MyAnimeList** — scraping for extended `anime_title`

---

## Action Items

- [x] Add `fetch_radio_browser()` to scrape_entity_sources.py (radio-browser.info, no auth) — 44k stations
- [ ] Add `fetch_podcastindex(api_key)` to scrape_entity_sources.py
- [ ] Add `fetch_newsapi(api_key)` to scrape_entity_sources.py
- [ ] Add `fetch_igdb(client_id, access_token)` to scrape_entity_sources.py
- [ ] Increase Gutendex `max_pages` to 2440 for full 78k coverage
- [ ] Re-run AniList 5× (with 60s wait between runs) for ~4,500 anime titles
- [ ] Update categorical features dataset after entity pool expansion

---

## HuggingFace Dataset Plan

### Dataset: `ovos-ocp-utterances`

**Structure**:
```
datasets/
├── utterances/
│   ├── train.parquet        (80% of 1.9M+ sentences)
│   ├── val.parquet          (10%)
│   ├── test.parquet         (10%)
│   └── README.md
├── categorical_features/
│   ├── train_features.parquet
│   ├── val_features.parquet
│   ├── test_features.parquet
│   └── feature_columns.json
└── entity_pools/
    ├── ocp_entities.csv     (48k+ merged entity pool)
    └── ...
```

**Upload**:
```bash
huggingface-cli login
huggingface-cli upload OpenVoiceOS/ovos-ocp-utterances-categorical splits/ \
    --repo-type dataset
```

See `docs/ENTITY_SOURCES.md` for full entity pool documentation.
