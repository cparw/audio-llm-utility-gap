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
- The fluency score uses CONTENT markers only (retracing, fillers, errors,
  vocabulary). Rate and word count are excluded on purpose: the first build
  included them and the validity battery caught a speaking-rate shortcut inside
  the conflict arm at AUC 0.92. Rebuilt, that falls to 0.57.
- Text-only out-of-fold words-vs-diagnosis: AUC 0.699, so for alzheimer's the
  words genuinely carry the diagnosis, the opposite of pc-gita read where text
  is near chance. Reading is partially correct behaviour here.
- Conflict: 146 segments (101 dementia speakers whose words look fluent,
  45 controls whose words look impaired), 100 speakers. Agreement: 322.
- Validity battery: investigator speech inside the windows is ~0. The text
  score tracks MMSE at spearman -0.35 within dementia speakers.
- Age and sex matched subset built ADReSS-style: 81 pairs, gap 7.1y -> 1.1y.
- All 468 segments cut to 16 kHz mono wav, ready for the model scorers

## First model result on the conflict set (run locally on Apple silicon)
qwen2-audio, all 468 segments, both prompt arms, answer mass 0.996:
- agreement arm AUC 0.699 [0.636, 0.759], essentially the text baseline (0.699 OOF)
- conflict arm AUC 0.413 [0.305, 0.533], a 0.29 drop, to chance
So on alzheimer's the model's accuracy is the accuracy of the words: it matches the
text baseline when the words agree with the diagnosis and falls to chance when they
point away. The conflict interval includes 0.5, so the claim is "falls to chance",
not "inverts", and this set is tertile-selected rather than matched pairs; both
caveats travel with the number.

## Second model on the conflict set (also run locally)
qwen2.5-omni, all 468 segments, answer mass 0.96:
- agreement arm AUC 0.766 [0.706, 0.820]
- conflict arm AUC 0.444 [0.330, 0.581]
Same shape as qwen2-audio: strong where the words agree, chance where they point
away. One flag that must travel with it: omni answers No on 100 percent of these
segments, so its AUC ranks confidences without ever crossing the threshold, the
same degeneracy it shows on parkinson's. qwen2-audio decides properly here
(80 percent yes) and is the cleaner citation of the two.
