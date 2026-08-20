# Pitt corpus (DementiaBank), local preparation summary

Data stays on the external drive under the TalkBank agreement. Only scripts and
aggregate numbers live here.

## Corpus
- 2,392 media files, 1,290 transcripts, 294 speakers (99 control, 195 dementia)
- Controls only recorded the cookie task, so cookie-only is the valid comparison:
  243 control vs 309 dementia recordings, 293 speakers
- Age gap: control mean 64.1, dementia 71.2. Needs matching, same as neurovoz.
- Dementia arm is mixed: ProbableAD 774, MCI 156, PossibleAD 69, Vascular 20, other 20.
- MMSE available per speaker: control mean 29.1, dementia 19.9 (severity target).

## Recording floor, cookie task (552 clips)
| property alone | AUC |
|---|---|
| duration | 0.613 |
| silence fraction | 0.586 |
| spectral centroid | 0.565 |
| rolloff | 0.549 |
| rms level | 0.519 |
| noise floor | 0.508 |
| all six combined | 0.651 |

Verdict: usable. Level and noise measures sit at chance, so the channel is clean,
unlike italian. What floor exists is duration and pauses, which are plausibly the
disease rather than the equipment, and both readings must be stated.
Sample rate uniform 44.1 kHz in both groups; formats parallel (mp3+wav each).

## Conflict set (built from transcripts, audio cut)
- 708 segments of ~30 s from participant speech only, cookie task
- Text-only out-of-fold words-vs-diagnosis: AUC 0.753, so for alzheimer's the
  words genuinely carry the diagnosis, the opposite of pc-gita read where text
  is near chance. This is the design point: reading is partially correct here.
- Conflict: 120 segments (86 dementia speakers whose words look fluent,
  34 controls whose words look impaired), 82 speakers
- Agreement: 348 segments
- All 468 segments cut to 16 kHz mono wav, ready for the model scorers
