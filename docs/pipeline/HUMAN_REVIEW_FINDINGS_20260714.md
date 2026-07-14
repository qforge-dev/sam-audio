# Human review findings: dialogue/background sample

The completed review contains 216 decisions: 12 Perfect, 98 Good, and 106 Not
OK. The review JSON from the shared server is the authority for these counts.

## Repeated failure patterns

| Human reason | Clips | Share of rejected clips |
|---|---:|---:|
| Lacking background audio / SFX | 53 | 50.0% |
| Lacking music | 39 | 36.8% |
| Too low quality | 27 | 25.5% |
| Wrong voice/background balance | 19 | 17.9% |
| Distorted or clipped | 15 | 14.2% |
| Lacking voice/dialogue | 14 | 13.2% |
| Too quiet | 5 | 4.7% |
| Speech is not dialogue | 1 | 0.9% |
| Other: AI voice | 1 | 0.9% |

Reasons are multi-select, so percentages do not sum to 100%.

The strongest negative source patterns were talking-head interviews, lectures,
motivational speeches, product tutorials, reviews/reactions, walking tours,
crowd-only recordings, synthetic narration, and low-quality user recordings.
These often satisfy a generic AudioSet speech/music label while still lacking
foreground dialogue, useful effects, or a film-like mix.

The strongest positive patterns were raw movie/TV/animated scenes, game
cutscenes, sports or news packages with production sound, comedy scenes, and
short-form narrative footage. Perfect examples included *Captain America*
elevator footage, *Cobra Kai*, *Invincible*, a MOBA narrative scene, and
produced sports/news segments.

## Generalized acquisition policy

1. Search for raw English movie scenes, TV scenes, animated scenes, game
   cutscenes, cinematic short films, and produced field-news packages.
2. Bias queries toward scenes that naturally contain dialogue, score, and
   effects: action, chase, street, restaurant, car, police, hospital, battle,
   suspense, comedy, and crowd scenes.
3. Reject metadata identifying reactions, reviews, analysis, recaps, podcasts,
   vlogs, walking tours, tutorials, speeches, audiobooks, AI/text-to-speech,
   fan edits, AMVs, or full movies.
4. Reject explicit India/Indian and Indian-language metadata (Bollywood, Hindi,
   Tamil, Telugu, Malayalam, Kannada, Bengali, and Punjabi). Do not infer a
   person's ethnicity or nationality from their voice.
5. Require real stereo, adequate source bitrate/sample rate, limited silence,
   healthy level, and no clipping before semantic validation.
6. Require sustained strong voice evidence. For the cinematic output, also
   require independent music and non-music effects evidence rather than the old
   union-style “background” label.
7. Permit multiple non-overlapping excerpts from one promising scene; retain
   the source URL, exact timestamps, query, and segment index for provenance.

## Foreground-voice calibration

The original strong-speech gate still admitted 14 clips that reviewers marked
`lacking_voice`. Their generic M2D Speech scores were often caused by crowd
chatter or speech babble rather than a usable foreground voice. M2D foreground
speech subclasses were tested but proved too sparse on the cinematic pilot to
be a mandatory gate, so they remain diagnostic metadata.

The final voice confirmation uses faster-whisper with VAD and confidence
checks: at least 1.5 seconds of voice activity, two decoded words, best segment
average log probability of `-0.75` or better, and no-speech probability no
higher than `0.40`. Applied retrospectively, it rejects 12 of 14
`lacking_voice` examples and retains 25 of a deterministic 30-clip Good/Perfect
calibration sample. It is combined with cinematic-source metadata and the
existing strong-speech gate; it is not treated as a standalone guarantee.

## Model limitation discovered

The old weak M2D speech gate was too permissive. More importantly, aggregate
M2D coverage was almost identical for Good and Not OK clips: median strong
speech and background coverage were both 1.0 in each group. Human feedback is
therefore used primarily to change acquisition sources and reject source-title
patterns; M2D remains a high-recall content gate, not the sole quality judge.
