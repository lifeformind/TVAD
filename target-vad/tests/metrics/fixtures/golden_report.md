# Session Metrics — fixture.wav

**Duration:** 90.0 s (1 min 30 s) · **Speech:** 78.4 s · **Silence:** 11.6 s
**Speakers:** 2 (2 enrolled, 0 recurring unknown, 0 catchall) · **Words:** 312 · **Segments:** 9
**Analyzed:** 2026-05-16T18:42:01Z

## Notable moments
- **Longest contribution** — Speaker A, 42.4 s at 00:00: *"The whole idea is that both of them should..."*
- **Most positive** — Speaker B at 01:05 (score 0.94): *"Yes, that's a great point about the radar pairing."*

## Participation

| Speaker   | Talk  | %     | Segs | Words | WPM   | Mean seg | Max seg |
|-----------|------:|------:|-----:|------:|------:|---------:|--------:|
| Speaker A | 52.1s | 66.5% | 5 | 198 | 228.0 | 10.4 s | 42.6 s |
| Speaker B | 26.3s | 33.5% | 4 | 114 | 260.0 | 6.6 s | 9.1 s |

## Sentiment — polarity (per speaker)

| Speaker   | Positive | Neutral | Negative | Mean conf. |
|-----------|---------:|--------:|---------:|-----------:|
| Speaker A | 20% | 80% | 0% | 0.81 |
| Speaker B | 25% | 75% | 0% | 0.79 |

## Sentiment — emotion (per speaker)

| Speaker   | Joy | Neutral | Surprise | Disgust* | Anger | Fear | Sadness | Mean conf. |
|-----------|----:|--------:|---------:|---------:|------:|-----:|--------:|-----------:|
| Speaker A | 0% | 60% | 20% | 20% | 0% | 0% | 0% | 0.62 |
| Speaker B | 25% | 50% | 0% | 25% | 0% | 0% | 0% | 0.58 |

\* "Disgust" from the emotion model tends to fire on polite-disagreement phrasing — read as "registered disagreement" rather than visceral disgust.

## Turn-taking

| Speaker   | Turns | Mean gap before | Interruptions |
|-----------|------:|----------------:|--------------:|
| Speaker A | 4 | 1.42 s | 1 |
| Speaker B | 4 | 0.85 s | 0 |

## Who follows whom

Rows = previous speaker, columns = next speaker. Cell = transition count.

|              | → Speaker A | → Speaker B |
|--------------| ----: | ----: |
| Speaker A → | — | 0 |
| Speaker B → | 0 | — |

---
_Caveat: 'unknown' segments may represent multiple physical speakers; the diarization layer collapses all unenrolled clusters into one bucket._