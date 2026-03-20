
# Template Design Guide — Best Practices for Dataset Generation

## Quick Reference

| Principle | Do | Don't |
|-----------|----|----|
| **Variety** | 10–15 templates per intent | 100 near-identical variations |
| **Slot pairing** | Independent entity pools | Forced semantic relationships |
| **Modality** | "watch {show}" (clear video) | "play {something}" (ambiguous) |
| **Negation** | Use at sentence edge only | Mid-sentence negation |
| **Realism** | Natural speech patterns | Stilted or formal language |
| **Specificity** | Most-specific templates first | No ordering strategy |
| **Coverage** | All slots in entity pool used | Missing entity types |

---

## 1. Template Structure

### Basic Format

```csv
category,template
music_artist,"play {artist}"
music_track,"play {track} by {artist}"
```

**Key Terms**:
- **category**: Semantic grouping (e.g., "music_artist", "by_actor"). Used for documentation; not enforced at generation time.
- **template**: String with `{slot_name}` placeholders. Slots are filled from entity pools at generation.

### Slot Naming

Follow a consistent pattern across all templates:

```
{noun}  (Preferred)         vs.   {verb_noun}  (Avoid redundancy)
{track}                          {play_track}
{artist}                         {music_artist}
{genre}                          {music_genre}
{provider}                       {streaming_provider}
```

**Exception**: Be specific only when disambiguating:
- `{movie_genre}` vs. `{music_genre}` (both needed in multi-intent pool)
- `{movie_title}` vs. `{show_name}` (clarity on media type)

---

## 2. Variety Without Proliferation

### ✅ Good: 10 templates with diverse phrasings

```
("play {artist}", ["artist"]),
("play some {artist}", ["artist"]),
("put on {artist}", ["artist"]),
("I want to listen to {artist}", ["artist"]),
("can you play {artist}", ["artist"]),
("I'm in the mood for {artist}", ["artist"]),
("stream {artist}", ["artist"]),
("queue up {artist}", ["artist"]),
("give me some {artist}", ["artist"]),
("find me {artist}", ["artist"]),
```

**Why?**: Each wording is natural and distinct. Random entity sampling creates 100+ unique utterances per template, totaling 1000+ per intent without duplication.

### ❌ Bad: 30 near-identical templates

```
("play {artist}", []),
("play {artist} music", []),
("play some {artist}", []),
("play a {artist} song", []),
("play music from {artist}", []),
("play songs by {artist}", []),
("play tracks by {artist}", []),
... (23 more variations)
```

**Why?**: Redundant. If 10 templates × 10 fills = 100 utterances, why write 30 near-duplicates that crowd the codebase?

---

## 3. Slot Independence: Avoiding Nonsense

### ❌ Bad: Forced semantic pairing

```
("play {album} from {artist}", ["album", "artist"])
```

**Problem**: Random entity sampling creates "The Beatles' _The Wall_" (real albums mixed arbitrarily).

**Options**:

1. **Drop the template** (if album-artist pairing is critical):
   ```python
   # Remove this template; albums are too tied to specific artists
   ```

2. **Treat each slot independently** (if generic enough):
   ```
   ("play {album}", ["album"])  # OK, album names are distinctive enough
   ("play an album by {artist}", ["artist"])  # OK, artist is independent
   ```

3. **Hardcode curated pairs** (if few and important):
   ```python
   ALBUM_ARTIST_PAIRS = [
       ("The Wall", "Pink Floyd"),
       ("Abbey Road", "The Beatles"),
   ]
   # Then: create a custom filler function for {album_artist_pair}
   ```

### ✅ Good: Independent slots

```
("play {track} by {artist}", ["track", "artist"])
```

**Why**: Any track + any artist = plausible (even if not a real pairing, the utterance is still natural).

---

## 4. Specificity Hierarchy

### Template Ordering

Order templates from most-specific (entity required) to most-generic (entity optional):

```python
_MUSIC_TEMPLATES = [
    # High specificity
    ("play {track} by {artist}", ["track", "artist"]),    ← both required
    ("play {track}", ["track"]),                           ← one required
    ("play {genre} music", ["genre"]),

    # Medium specificity
    ("play some music", []),                               ← zero required; no-op fill
    ("put on something", []),
]
```

**Rationale**:
1. When entity pool is exhausted, fallback to zero-slot templates to still generate utterances
2. Clustering similar specificity improves code readability
3. Easier to adjust ratios (e.g., "use 10 templates with slots, 2 without")

### Example: TV Shows

```python
_TV_SHOW_TEMPLATES = [
    # Show name required
    ("play {show}", ["show"]),
    ("put on {show}", ["show"]),
    ("I want to watch {show}", ["show"]),
    ("watch {show}", ["show"]),
    ("continue watching {show}", ["show"]),

    # No entity required
    ("find me something to watch", []),
    ("play a TV show", []),
    ("what should I watch", []),
]
```

---

## 5. Realistic Phrasings

### ✅ Good: Natural speech

```
play {artist}
put on some {artist}
I want to listen to {artist}
find me some {artist} music
I'm in the mood for {artist}
can you play {artist}
stream {artist}
give me {artist}
```

**Characteristics**:
- Conversational (contractions: "I'd", "I'm")
- Varied sentence structure (imperative, interrogative, declarative)
- Includes filler words ("some", "me", "for me")
- Natural verb choices (play, stream, put on, queue)

### ❌ Bad: Stilted or formal

```
play {artist}
play artist {artist}
play music by {artist} please
I wish to audition {artist}
I desire to listen to {artist}
afford me {artist} music
dispense {artist} tracks
```

**Issues**:
- "play artist {artist}" is unnatural phrasing
- "I wish to audition" is formal and uncommon
- "afford me" and "dispense" are absurd
- No variety in verb choice

---

## 6. Modality Clarity (Audio vs. Video)

### ✅ Good: Clear verb choice

```
# AUDIO
listen to {artist}
play {song}
put on {podcast}
tune into {station}

# VIDEO
watch {show}
see {movie}
stream {title}
find a film
```

### ❌ Bad: Ambiguous

```
play {something}        ← music OR video? Unclear
put on {content}        ← what modality?
find me {thing}         ← which?
```

**Issue**: The template alone cannot signal intent to the classifier. Classifier must rely on the entity to infer modality, which is unreliable.

**Fix**: Use modality-specific verbs:
- Audio: listen, hear, tune, play (music context), stream (audio)
- Video: watch, see, view, display

---

## 7. Negation: Use Sparingly

### ✅ OK: Peripheral negation

```
"I'm not in the mood for {genre}"
```

(Negation at sentence edge; entity still positive)

### ❌ Bad: Mid-sentence negation

```
"play anything except {genre}"
"don't play {artist}"
"I don't want to watch {show}"
"skip {{title}} and play something else"
```

**Why avoid**:
- Classifiers are worse at negation
- Creates negative examples that confuse "not OCP" boundary
- Hard to model truthfully (e.g., "don't play X" isn't really a request for X—it's a control command)

---

## 8. Provider-Agnostic Design

### ✅ Good: Provider as slot

```
("play {artist} on {provider}", ["artist", "provider"])
("find {track} on {{provider}}", ["track", "provider"])
```

**Benefit**: One template covers all providers (Netflix, Disney+, Hulu, etc.)

### ❌ Bad: Hardcoded providers

```
("play {artist} on Spotify", ["artist"])
("find {track} on Apple Music", ["track"])
("stream {title} on Netflix", ["title"])
```

**Issues**:
- N providers × M templates = N×M template bloat
- Hardcoded provider list becomes stale
- Doesn't scale to user's actual subscriptions

---

## 9. Multi-Slot Templates: Pairing Strategy

### Random Pairing (Default)

```python
("play {track} by {artist}", ["track", "artist"])
```

**Generation**:
```
Random sample from track_pool × artist_pool
→ "play Bohemian Rhapsody by Miles Davis"
  (Real track + real artist, but not a real pairing)
```

**When OK**:
- Utterance is still natural (any track + any artist works linguistically)
- Minor semantic incorrectness is tolerable
- Easy to implement

### Curated Pairing (Better)

For critical relationships (album-artist, show-season):

```python
ALBUM_ARTIST = [
    ("The Dark Side of the Moon", "Pink Floyd"),
    ("Abbey Road", "The Beatles"),
    ("Kind of Blue", "Miles Davis"),
]

# In template handler:
def fill_curated(template, pools):
    if "{album} by {artist}" in template:
        album, artist = random.choice(ALBUM_ARTIST)
        return template.replace("{album}", album).replace("{artist}", artist)
```

**When use**:
- Data correctness is critical (album-artist, actor-character)
- Small number of pairs (< 1000)
- Relationship is one-to-one (not many-to-many)

---

## 10. Category Grouping (Documentation)

Use the `category` column to organize templates semantically:

```csv
category,template
artist,"play {artist}"
artist,"put on some {artist}"
artist,"I want to listen to {artist}"
track,"play {track}"
track,"I want to hear {track}"
genre,"play some {genre} music"
generic,"play some music"
```

**Benefits**:
- Humans can skim and understand template organization
- Easier to add/remove categories
- CI/CD can validate (e.g., "does every category have ≥ N templates?")
- Debugging: know which category failed to generate

---

## 11. Slot Naming Conventions (Comprehensive Reference)

### Music
- `{artist}` — Name of a music artist/band
- `{track}`, `{song}` — Individual song title
- `{album}` — Album name
- `{genre}` — Music genre (jazz, rock, pop, etc.)
- `{playlist_name}` — User-visible playlist name
- `{featuring_artist}` — Collaborator/feature artist

### Video (Movies, TV)
- `{title}`, `{show}`, `{series}` — Media name
- `{season}` — Season number/name
- `{episode}` — Episode number/name
- `{actor}` — Actor name
- `{director}` — Director name
- `{writer}` — Writer/screenplay author
- `{producer}` — Producer name
- `{genre}` — Film/TV genre
- `{year}` — Release year

### Spoken Word
- `{show}` — Podcast/radio show name
- `{topic}` — Podcast topic/category
- `{host}` — Podcast host name
- `{station}` — Radio station name
- `{provider}` — Service provider (BBC, NPR, etc.)
- `{narrator}` — Audiobook narrator

### Gaming
- `{title}` — Game name
- `{genre}` — Game genre (action, RPG, etc.)
- `{platform}` — Gaming platform (Steam, PlayStation, etc.)

### General
- `{provider}` — Streaming service (Spotify, Netflix, etc.)
- `{genre}` — Any genre applicable to media type
- `{year}` — Release year

---

## 12. Validation Checklist

Before committing templates:

- [ ] **No slot name typos**: Grep for `{slot_name}` in pools; all slots are defined
- [ ] **Uniqueness**: No duplicate templates (exact or near-duplicate)
- [ ] **Specificity ordered**: High-specificity (entity required) templates listed first
- [ ] **Semantic plausibility**: Spot-check 5 random utterances; verify they're natural
- [ ] **Modality clarity**: Video templates use watch/see/view; audio use play/listen
- [ ] **No forced pairings**: No album-artist or show-season without curation
- [ ] **Coverage**: All major entity types used (artist, track, genre for music, etc.)
- [ ] **Category consistency**: Categories align with slot names where possible
- [ ] **Line endings**: Consistent (no trailing spaces or mixed line endings)

---

## 13. Common Pitfalls and Fixes

| Pitfall | Example | Fix |
|---------|---------|-----|
| Slot typo | `{artst}` (missing 'i') | Grep codebase for `{artst}`; fix |
| Unnamed slots | `"play some songs"` (no slots) | Add slot or accept it as generic |
| Forced pairing | `"play {album} by {artist}"` | Use curated pairs or drop template |
| Ambiguous modality | `"play {something}"` | Use "watch" or "listen" |
| Negation mid-sentence | `"don't play {artist}"` | Use peripheral negation or drop |
| Hardcoded providers | `"on Spotify"` (fixed in template) | Make provider a slot |
| Too many templates | 100 near-identical variations | Reduce to 10 diverse templates |
| Missing entity pool | Slot `{year}` but no `year_pool` | Define or remove slot |
| Category bloat | 50 categories, 2 templates each | Merge categories |

---

## 14. Entity Pool Mapping

For each template slot, define where entities come from:

**Example: Music**

```csv
Slot,Source (Priority),Fallback
{artist},"HF metal-archives-bands, jazz-archives","Wikidata artist_name, curated list"
{track},"HF metal-archives-tracks, trance_tracks","Wikidata song_name"
{album},"Wikidata album_name","Curated album list (100–500)"
{genre},"Wikidata music_genre","Curated genres (50–100)"
{provider},"Hardcoded list (Spotify, Apple Music)","Wikidata music_streaming_service"
{playlist_name},"Curated playlist names (My Workout, Chill Vibes)","Random names"
```

**How to use**:
1. When designing a template, check the mapping above
2. Verify entity pool exists and is non-empty
3. If missing, either:
   - Add a source (e.g., load HF dataset)
   - Remove the template
   - Use curated list

---

## 15. Testing Your Templates

### Minimal Test

```python
from scripts.generate_synthetic import generate_from_local_templates

df = generate_from_local_templates(
    templates_dir="templates/",
    media_csv=None,
    n_per_template=5,
    hf_cache="/tmp/hf_cache",
    dedup_against=None,
    seed=42
)

print(df.head(20))
# Spot-check: are sentences natural? Are all slots filled?
```

### Comprehensive Test

```python
# Check for missing slots
import re
df = generate_from_local_templates(...)
slots = re.findall(r'\{([^}]+)\}', df['sentence'].str.cat())
if slots:
    print(f"ERROR: Unfilled slots: {slots}")

# Check for duplicates
duplicates = df.duplicated(subset=['sentence']).sum()
print(f"Duplicates: {duplicates}")

# Check distribution
print(df['media_label'].value_counts())
```

---

## 16. Scaling: From 10 to 100K Utterances

**Workflow**:

1. **Start small**: 5 templates × 10 fills = 50 utterances → manually verify
2. **Expand entity pools**: Add Wikidata + HF datasets → 500 utterances
3. **Add more templates**: 20 templates × 50 fills = 1000 utterances
4. **Deduplicate**: Remove exact matches
5. **Balance**: Ensure each intent has ≥ 100 utterances
6. **Combine sources**: Merge synthetic + OCP template outputs
7. **Final deduplicate**: Against all sources
8. **Train model**: Use `build_dataset.py`

---

## References

- `docs/DATASET_GENERATION.md` — Full pipeline documentation
- `scripts/generate_synthetic.py` — Template definitions (music, movie, etc.)
- `scripts/generate_from_ocp_templates.py` — Large-scale Wikidata pipeline
- `templates/*.csv` — MediaType-specific templates (this guide)
