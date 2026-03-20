# OCP Taxonomy Specification

This document is the formal specification for the OCP media classification
taxonomy: `MediaType`, `OCPPlayIntent`, `OCPEntityLabel`, and `OCPControlIntent`.

All definitions are authoritative.  Code in `ovos_media_classifier/intents.py`
implements exactly this spec.

---

## 1. MediaType catalogue

`MediaType` is an `IntEnum` defined in `ovos_media_classifier.intents`.  Integer
values are identical to those in `ovos_utils.ocp.MediaType` for shared types,
enabling cross-package interoperability via integer comparison.

`MediaType.TV_SHOW = 25` is the only value not present in `ovos_utils`.

| MediaType | int | OCPPlayIntent | Description | Example utterances |
|---|---|---|---|---|
| `GENERIC` | 0 | `generic` | No media type detected | — |
| `AUDIO` | 1 | `audio` | Unclassified audio | "play some background noise" |
| `MUSIC` | 2 | `music` | Music tracks, albums, artists | "play jazz", "shuffle Metallica" |
| `VIDEO` | 3 | `video` | Generic video / YouTube | "show me that video", "play the clip" |
| `AUDIOBOOK` | 4 | `audiobook` | Audiobook narration | "read Dune", "audiobook by Tolkien" |
| `GAME` | 5 | `game` | Video game audio/video | "play Doom", "stream this game" |
| `PODCAST` | 6 | `podcast` | Podcast episodes / shows | "play the latest Lex Fridman", "subscribe to 99pi" |
| `RADIO` | 7 | `radio` | Live radio stream | "tune to KQED", "BBC radio 4" |
| `NEWS` | 8 | `news` | News broadcast / feed | "play the news", "headlines from CNN" |
| `TV` | 9 | `tv` | **Live** IPTV / cable stream | "put on CNN", "stream BBC One", "tune to live TV" |
| `MOVIE` | 10 | `movie` | Feature film | "watch Inception", "put on a movie" |
| `TRAILER` | 11 | `trailer` | Movie/show trailer or teaser | "show the trailer for Top Gun", "play the teaser" |
| `AUDIO_DESCRIPTION` | 12 | `audio_description` | Narrated film for the visually impaired | "audio description of Parasite" |
| `VISUAL_STORY` | 13 | `visual_story` | Motion comic / animated story | "play the Watchmen comic", "motion comic" |
| `BEHIND_THE_SCENES` | 14 | `behind_the_scenes` | Making-of / featurette content | "watch the making of Dune", "cast interview" |
| `DOCUMENTARY` | 15 | `documentary` | Documentary film | "watch Planet Earth", "documentary about space" |
| `RADIO_THEATRE` | 16 | `radio_theatre` | Audio drama / radio play | "play The Hitchhiker's Guide radio play" |
| `SHORT_FILM` | 17 | `short_film` | Short film | "play a short film", "short movie" |
| `SILENT_MOVIE` | 18 | `silent_movie` | Silent-era film | "show a silent film", "Charlie Chaplin silent movie" |
| `VIDEO_EPISODES` | 19 | `video_episodes` | YouTube channels / online video series | "play that YouTube channel", "Linus Tech Tips" |
| `BLACK_WHITE_MOVIE` | 20 | `bw_movie` | Black-and-white film | "classic black and white movie" |
| `CARTOON` | 21 | `cartoon` | Animated cartoon | "play SpongeBob", "cartoon for kids" |
| `ANIME` | 22 | `anime` | Japanese animation | "play Naruto", "stream One Piece" |
| `ASMR` | 23 | `asmr` | ASMR content | "play some ASMR", "relaxing sounds" |
| `MUSIC_VIDEO` | 24 | `music_video` | Official music video for a song | "play the music video for Thriller", "official video for Bohemian Rhapsody" |
| `TV_SHOW` | **25** | `tv_show` | Episodic TV series (broadcast / streaming) | "watch Breaking Bad", "play Game of Thrones S3" |
| `ADULT` | 69 | `adult` | Adult video content | — |
| `HENTAI` | 70 | `hentai` | Adult anime | — |
| `ADULT_AUDIO` | 71 | `adult_audio` | Adult audio content | — |

> **Note on `TV` vs `TV_SHOW`**: `TV` (int=9) means a live stream — a channel
> you "tune to" that is broadcasting right now.  `TV_SHOW` (int=25) means an
> episodic series like Breaking Bad that you can pause, seek, and select by
> season/episode.

> **Note on `VIDEO_EPISODES`**: YouTube channels and online video playlists that
> are not broadcast TV and not a traditional episodic series (e.g. a Let's Play
> series, a tech review channel).

---

## 2. OCPPlayIntent table

All 28 intent labels.  String values are the training labels used by ML backends.

| OCPPlayIntent | String | MediaType | Entity labels that resolve here |
|---|---|---|---|
| `DOCUMENTARY` | `"documentary"` | `DOCUMENTARY` | `documentary_title`, `documentary` keyword |
| `AUDIOBOOK` | `"audiobook"` | `AUDIOBOOK` | `audiobook_title`, `audiobook_author`, `audiobook` keyword |
| `NEWS` | `"news"` | `NEWS` | `news_provider`, `news_category`, `news` keyword |
| `ANIME` | `"anime"` | `ANIME` | `anime_title`, `anime` keyword |
| `CARTOON` | `"cartoon"` | `CARTOON` | `cartoon_title`, `cartoon` keyword |
| `PODCAST` | `"podcast"` | `PODCAST` | `podcast_title`, `podcast_episode`, `podcast_host`, `podcast_streaming_service`, `podcast` keyword |
| `RADIO_THEATRE` | `"radio_theatre"` | `RADIO_THEATRE` | `radio_drama_title`, `radio_theatre` keyword |
| `RADIO` | `"radio"` | `RADIO` | `radio_streaming_service`, `radio_station`, `radio` keyword |
| `TV` | `"tv"` | `TV` | `tv_streaming_service`, `tv_channel`, `tv` keyword, `IPTVKeyword.voc` |
| `MUSIC` | `"music"` | `MUSIC` | `artist_name`, `track_name`, `album_name`, `album_type`, `music_genre`, `record_label`, `music_streaming_service`, `music` keyword |
| `TV_SHOW` | `"tv_show"` | `TV_SHOW` | `tv_show_title`, `tv_show` keyword |
| `VIDEO_EPISODES` | `"video_episodes"` | `VIDEO_EPISODES` | `youtube_channel`, `video_episodes` keyword, `SeriesKeyword.voc` |
| `SHORT_FILM` | `"short_film"` | `SHORT_FILM` | `shorts_streaming_service`, `short_film` keyword |
| `SILENT_MOVIE` | `"silent_movie"` | `SILENT_MOVIE` | `silent_movie_title`, `silent_movie` keyword |
| `BW_MOVIE` | `"bw_movie"` | `BLACK_WHITE_MOVIE` | `bw_movie_title`, `bw_movie` keyword |
| `MUSIC_VIDEO` | `"music_video"` | `MUSIC_VIDEO` | `music_video_title`, `music_video` keyword, `MusicVideoKeyword.voc` |
| `TRAILER` | `"trailer"` | `TRAILER` | `trailer_title`, `trailer` keyword, `TrailerKeyword.voc` |
| `BEHIND_THE_SCENES` | `"behind_the_scenes"` | `BEHIND_THE_SCENES` | `bts_title`, `behind_the_scenes` keyword, `BehindTheScenesKeyword.voc` |
| `MOVIE` | `"movie"` | `MOVIE` | `movie_title`, `movie_actor`, `movie_director`, `movie_producer`, `movie_writer`, `movie_composer`, `movie_streaming_service`, `movie` keyword |
| `VISUAL_STORY` | `"visual_story"` | `VISUAL_STORY` | `visual_story_title`, `visual_story` keyword |
| `GAME` | `"game"` | `GAME` | `game_title`, `game` keyword |
| `AUDIO_DESCRIPTION` | `"audio_description"` | `AUDIO_DESCRIPTION` | `audio_description` keyword |
| `ASMR` | `"asmr"` | `ASMR` | `asmr_artist`, `asmr` keyword |
| `HENTAI` | `"hentai"` | `HENTAI` | `hentai_title`, `hentai` keyword |
| `ADULT_AUDIO` | `"adult_audio"` | `ADULT_AUDIO` | `adult_audio` keyword |
| `ADULT` | `"adult"` | `ADULT` | `adult_title`, `adult_streaming_service`, `adult` keyword |
| `VIDEO` | `"video"` | `VIDEO` | `video` keyword |
| `AUDIO` | `"audio"` | `AUDIO` | `audio` keyword |
| `GENERIC` | `"generic"` | `GENERIC` | (fallback) |

---

## 3. OCPEntityLabel taxonomy

All entity labels with their string value, the OCPPlayIntent they resolve to,
and example entity values.

### Streaming service labels

| Constant | String value | Resolves to | Example values |
|---|---|---|---|
| `MUSIC_STREAMING_SERVICE` | `music_streaming_service` | `music` | Spotify, Tidal, Deezer |
| `MOVIE_STREAMING_SERVICE` | `movie_streaming_service` | `movie` | Netflix, Disney+, Prime Video |
| `SHORTS_STREAMING_SERVICE` | `shorts_streaming_service` | `short_film` | YouTube Shorts, TikTok |
| `PODCAST_STREAMING_SERVICE` | `podcast_streaming_service` | `podcast` | Spotify Podcasts, Pocket Casts |
| `AUDIOBOOK_STREAMING_SERVICE` | `audiobook_streaming_service` | `audiobook` | Audible, Libro.fm |
| `NEWS_PROVIDER` | `news_provider` | `news` | BBC News, Reuters, AP |
| `TV_STREAMING_SERVICE` | `tv_streaming_service` | `tv` | Pluto TV, Plex Live TV, Locast |
| `RADIO_STREAMING_SERVICE` | `radio_streaming_service` | `radio` | iHeartRadio, TuneIn |
| `ADULT_STREAMING_SERVICE` | `adult_streaming_service` | `adult` | (adult services) |

### Music entity labels

| Constant | String value | Resolves to | Example values |
|---|---|---|---|
| `ARTIST_NAME` | `artist_name` | `music` | Metallica, Pink Floyd, Billie Eilish |
| `TRACK_NAME` | `track_name` | `music` | Bohemian Rhapsody, Thriller |
| `ALBUM_NAME` | `album_name` | `music` | Dark Side of the Moon |
| `ALBUM_TYPE` | `album_type` | `music` | EP, LP, mixtape |
| `MUSIC_GENRE` | `music_genre` | `music` | jazz, blues, metal |
| `RECORD_LABEL` | `record_label` | `music` | Capitol, Sony Music |
| `RADIO_STATION` | `radio_station` | `radio` | KQED, BBC Radio 4 |

### Video entity labels

| Constant | String value | Resolves to | Example values |
|---|---|---|---|
| `MOVIE_TITLE` | `movie_title` | `movie` | Inception, The Matrix |
| `MOVIE_ACTOR` | `movie_actor` | `movie` | Leonardo DiCaprio |
| `MOVIE_DIRECTOR` | `movie_director` | `movie` | Christopher Nolan |
| `MOVIE_PRODUCER` | `movie_producer` | `movie` | Emma Thomas |
| `MOVIE_WRITER` | `movie_writer` | `movie` | Jonathan Nolan |
| `MOVIE_COMPOSER` | `movie_composer` | `movie` | Hans Zimmer |
| `TV_SHOW_TITLE` | `tv_show_title` | `tv_show` | Breaking Bad, Game of Thrones |
| `ANIME_TITLE` | `anime_title` | `anime` | Naruto, One Piece |
| `CARTOON_TITLE` | `cartoon_title` | `cartoon` | SpongeBob, Looney Tunes |
| `DOCUMENTARY_TITLE` | `documentary_title` | `documentary` | Planet Earth, Free Solo |
| `TRAILER_TITLE` | `trailer_title` | `trailer` | (movie name in trailer context) |
| `BTS_TITLE` | `bts_title` | `behind_the_scenes` | (title in making-of context) |
| `MUSIC_VIDEO_TITLE` | `music_video_title` | `music_video` | (song title in music video context) |
| `VISUAL_STORY_TITLE` | `visual_story_title` | `visual_story` | (motion comic title) |
| `SILENT_MOVIE_TITLE` | `silent_movie_title` | `silent_movie` | (silent movie title) |
| `BW_MOVIE_TITLE` | `bw_movie_title` | `bw_movie` | (black and white movie title) |
| `HENTAI_TITLE` | `hentai_title` | `hentai` | (hentai title) |
| `RADIO_DRAMA_TITLE` | `radio_drama_title` | `radio_theatre` | (radio drama title) |
| `ADULT_TITLE` | `adult_title` | `adult` | (adult title) |
| `PORNSTAR` | `pornstar` | `adult` | (pornstar name) |
| `PORN_GENRE` | `porn_genre` | `adult` | (porn genre) |

### TV / live stream entity labels

| Constant | String value | Resolves to | Example values |
|---|---|---|---|
| `TV_CHANNEL` | `tv_channel` | `tv` | CNN, BBC One, Eurosport |

### Other media entity labels

| Constant | String value | Resolves to | Example values |
|---|---|---|---|
| `PODCAST_TITLE` | `podcast_title` | `podcast` | Radiolab, Serial |
| `PODCAST_EPISODE` | `podcast_episode` | `podcast` | (episode titles) |
| `AUDIOBOOK_TITLE` | `audiobook_title` | `audiobook` | Dune, The Hobbit |
| `AUDIOBOOK_AUTHOR` | `audiobook_author` | `audiobook` | Frank Herbert |
| `NEWS_CATEGORY` | `news_category` | `news` | sports, weather, politics, tech |
| `GAME_TITLE` | `game_title` | `game` | Doom, Minecraft |
| `ASMR_ARTIST` | `asmr_artist` | `asmr` | (ASMR creator names) |

> `NEWS_TOPIC` (removed) was untrainable — any word qualifies as a topic.
> `NEWS_CATEGORY` is its replacement: a coarse set of categories a model can learn.

### Media-type keyword labels

Used when no specific named entity is found but a keyword strongly signals
the media type.  Lower priority than named entity labels.

| Constant | String value | Resolves to |
|---|---|---|
| `MUSIC_KEYWORD` | `music` | `music` |
| `PODCAST_KEYWORD` | `podcast` | `podcast` |
| `RADIO_KEYWORD` | `radio` | `radio` |
| `AUDIOBOOK_KEYWORD` | `audiobook` | `audiobook` |
| `NEWS_KEYWORD` | `news` | `news` |
| `MOVIE_KEYWORD` | `movie` | `movie` |
| `TV_KEYWORD` | `tv` | `tv` (live) |
| `TV_SHOW_KEYWORD` | `tv_show` | `tv_show` (episodic) |
| `VIDEO_KEYWORD` | `video` | `video` |
| `VIDEO_EPISODES_KEYWORD` | `video_episodes` | `video_episodes` |
| `AUDIO_KEYWORD` | `audio` | `audio` |
| `GAME_KEYWORD` | `game` | `game` |
| `ANIME_KEYWORD` | `anime` | `anime` |
| `CARTOON_KEYWORD` | `cartoon` | `cartoon` |
| `DOCUMENTARY_KEYWORD` | `documentary` | `documentary` |
| `SHORT_FILM_KEYWORD` | `short_film` | `short_film` |
| `SILENT_MOVIE_KEYWORD` | `silent_movie` | `silent_movie` |
| `BW_MOVIE_KEYWORD` | `bw_movie` | `bw_movie` |
| `RADIO_THEATRE_KEYWORD` | `radio_theatre` | `radio_theatre` |
| `VISUAL_STORY_KEYWORD` | `visual_story` | `visual_story` |
| `ASMR_KEYWORD` | `asmr` | `asmr` |
| `AUDIO_DESCRIPTION_KEYWORD` | `audio_description` | `audio_description` |
| `MUSIC_VIDEO_KEYWORD` | `music_video` | `music_video` |
| `TRAILER_KEYWORD` | `trailer` | `trailer` |
| `BEHIND_THE_SCENES_KEYWORD` | `behind_the_scenes` | `behind_the_scenes` |
| `ADULT_KEYWORD` | `adult` | `adult` |
| `ADULT_AUDIO_KEYWORD` | `adult_audio` | `adult_audio` |
| `HENTAI_KEYWORD` | `hentai` | `hentai` |

---

## 4. OCPControlIntent reference

All 15 control intent labels.

| OCPControlIntent | String | Example utterances |
|---|---|---|
| `PLAY` | `"play"` | "play", "start", "go" |
| `NEXT` | `"next"` | "next song", "skip this", "next track" |
| `PREVIOUS` | `"prev"` | "go back", "previous track" |
| `PAUSE` | `"pause"` | "pause", "hold on" |
| `RESUME` | `"resume"` | "resume", "continue", "unpause" |
| `STOP` | `"stop"` | "stop", "turn it off", "quit" |
| `OPEN` | `"open"` | "open the player" |
| `LIKE_SONG` | `"like_song"` | "like this", "thumbs up", "add to favorites" |
| `PLAY_FAVORITES` | `"play_favorites"` | "play my favorites", "play liked songs" |
| `SAVE_GAME` | `"save_game"` | "save my game", "save progress" |
| `LOAD_GAME` | `"load_game"` | "load my game", "restore save" |
| `SHUFFLE` | `"shuffle"` | "shuffle my playlist", "random order", "mix it up" |
| `REPEAT` | `"repeat"` | "repeat this", "loop", "play again" |
| `SEEK_FORWARD` | `"seek_forward"` | "skip 30 seconds", "fast forward", "skip ahead" |
| `SEEK_BACKWARD` | `"seek_backward"` | "go back a minute", "rewind", "skip back" |

---

## 5. Invariants

The following rules must hold after any change to the taxonomy:

1. **No orphaned MediaType** — every `MediaType` value must appear in
   `PLAY_INTENT_TO_MEDIA_TYPE` as a value (or be `GENERIC`).

2. **Every `OCPPlayIntent` in priority list** — `set(OCPPlayIntent) == set(_INTENT_PRIORITY)`
   with no duplicates.

3. **No string value collisions** — no two `OCPEntityLabel` constants share a
   string value.

4. **TV ≠ TV_SHOW** — `MediaType.TV` (live stream, int=9) must always be distinct
   from `MediaType.TV_SHOW` (episodic, int=25).

5. **No `"porn_streaming_service"` string** — the value is `"adult_streaming_service"`.

6. **No `"news_topic"` label** — replaced by `"news_category"`.

7. **MediaType integers match ovos-utils** — all shared values must use the same
   integer as `ovos_utils.ocp.MediaType`.  New values (currently only `TV_SHOW=25`)
   must use integers not present in ovos-utils.

---

## 6. Keyword voc files

New voc files added in this release (all 13 locales):

- `IPTVKeyword.voc` — triggers `MediaType.TV` (live TV keywords: channel, iptv, live tv, …)
- `BehindTheScenesKeyword.voc` — triggers `MediaType.BEHIND_THE_SCENES`
- `TrailerKeyword.voc` — pre-existing; now wired into `keyword.py`

Priority order in `keyword.py` (condensed):
1. DocumentaryKeyword → DOCUMENTARY
2. AudioBookKeyword → AUDIOBOOK
3. NewsKeyword → NEWS
4. AnimeKeyword → ANIME
5. CartoonKeyword → CARTOON
6. PodcastKeyword → PODCAST
7. AudioDramaKeyword → RADIO_THEATRE (must beat RadioKeyword)
8. RadioKeyword → RADIO
9. MusicKeyword → MUSIC (must beat MovieKeyword)
10. **IPTVKeyword → TV** (more specific than TVKeyword)
11. TVKeyword → TV
12. SeriesKeyword → VIDEO_EPISODES
13. MovieKeyword family → SHORT_FILM / SILENT_MOVIE / BLACK_WHITE_MOVIE / MOVIE
14. **TrailerKeyword → TRAILER**
15. **BehindTheScenesKeyword → BEHIND_THE_SCENES**
> Note: `MusicVideoKeyword` is checked before `MusicKeyword` (more specific)
16. ComicBookKeyword → VISUAL_STORY
17. GameKeyword → GAME
18. ADKeyword → AUDIO_DESCRIPTION
19. ASMRKeyword → ASMR
20. AdultKeyword family → HENTAI / ADULT_AUDIO / ADULT
21. HentaiKeyword → HENTAI
22. VideoKeyword → VIDEO
23. AudioKeyword → AUDIO
24. (fallback) → GENERIC
