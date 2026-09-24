# Verified numbers for Paper 1
Every figure below was recomputed and confirmed. Anything not on this list does not go in the paper.

## The conflict test (195 matched pairs, 74 speakers, no transcript given)
| model | conflict AUC | agreement AUC |
|---|---|---|
| salmonn | 0.291 | 0.850 |
| diva | 0.299 | 0.699 |
| qwen2.5-omni | 0.307 | 0.848 |
| qwen3-omni | 0.349 | 0.867 |
| qwen2-audio | 0.351 | 0.923 |
| audio flamingo 3 | 0.362 | 0.926 |
| mimo-audio | 0.466 | 0.819 |
| kimi-audio | 0.521 | 0.865 |
| audio flamingo 2 | 0.549 | 0.574 |

Six of nine below guessing. Flamingo 2 is flat on BOTH arms, so it has no signal to invert.

## Transcript equivalence  ** CORRECTED **
- Pearson r, omni audio vs its own transcript: **0.827** (297 unique clips), **0.851** (390-row manifest)
- Pearson on logits: 0.824
- Same yes/no at threshold 0.5: **0.960**
- Same yes/no at tuned threshold 0.115: 0.906
- Same yes/no on conflict segments only: 0.887
- ** THE 0.917 PREVIOUSLY REPORTED IS WRONG. ** It traces to an unrelated kcl layer-probe AUC in RESULTS.md.

## The gap
- Encoder probe, pc-gita read, NESTED layer selection: **0.9167**
- Model's own answer, same clips: 0.54 (pcg_pd_read_summary.json behavioral_diag_AUC 0.539; the earlier 0.63 here was a transcription error)
- Peeking inflation across 18 encoder cells: +0.031 mean (NOT 0.06)
- Nested numbers move up to 0.049 across seeds -> quote 2 decimals, never 3

## Nested vs published encoder numbers
| cell | published | nested | delta |
|---|---|---|---|
| kcl spontaneous | 0.9167 | 0.7822 | -0.135 |
| kcl read | 0.9667 | 0.8881 | -0.079 |
| daic interview | 0.6989 | 0.6550 | -0.044 |
| pc-gita read | 0.9318 | 0.9167 | -0.015 |
| neurovoz read | 0.9484 | 0.9591 | +0.011 |

## The window artifact
- qwen2-audio caps at 30 s, omni at 300 s, daic interviews run 12-25 min
- At matched 300 s: flamingo 3 0.858, omni 0.835, kimi 0.831 (indistinguishable)
- omni on daic: 0.669 at 30 s vs 0.835 at 300 s

## Text-only grid (30 of 30 cells)
- Depression: **0 of 6 models beat their own transcript.** median text 0.813 vs audio 0.833
- Overall: audio beats text in 11 of 30 cells, all on PD read tasks or italian where text is 0.47-0.67 already
- Paradox split: text 0.20-0.29 and audio 0.29-0.52 on conflict; both 0.82-0.94 on agreement

## Failed interventions (all six)
| attempt | result |
|---|---|
| 12 prompt wordings | never above guessing on conflict |
| layer mixer | 0.909 vs 0.916 for one honest layer |
| prompt-conditioned mixer | +0.0004 |
| contrastive tuning | no gain |
| threshold calibration | moves the line only |
| logit-space text subtraction | see sweep below |

### Text-subtraction sweep (omni)
| alpha | conflict | agreement | conflict vs words |
|---|---|---|---|
| 0.0 | 0.297 | 0.868 | 0.703 |
| 0.9 | 0.502 | 0.718 | 0.498 |
| 1.0 | 0.528 | 0.619 | 0.472 |
| 2.0 | 0.673 | 0.178 | 0.327 |
Nested on held-out speakers: alpha=0 chosen in 4 of 5 folds, delta -0.026.

### Activation steering (qwen2-audio, difference-in-means direction, projector frozen)
| c | conflict | agreement |
|---|---|---|
| 0 | 0.353 | 0.922 |
| 0.25 | 0.364 | 0.879 |
| 1.0 | 0.379 | 0.695 |
| 2.0 | 0.443 | 0.554 |
| 4.0 | 0.536 | 0.497 |
- Guard rail fires at the SMALLEST coefficient (agreement significantly damaged at c=0.25)
- **A random direction of the same norm moves the answer just as much.** v sits inside the random spread at every c.
- corr(conflict AUC, lexical AUC) across 20 steered conditions = **-0.916**

### PCLM + DPO checkpoint (the advisor's own ICML fix)
- conflict, free-form: 0.365 vs base 0.350
- conflict, MCQ: 0.410 vs base 0.421
- every clinical delta on pc-gita read and daic sits inside the base model's bootstrap CI
- **the gate is close to a no-op on clinical prompts: 66% of its mass on the final encoder layer vs 37% on VoxParadox-style prompts**

## The encoder association  ** PREDICATE CORRECTED **
- WHISPER PROVENANCE PREDICTS NOTHING: 5 of 6 invert vs 2 of 3, Fisher exact p = 1.00
- Qwen3-Omni uses AuT, trained from scratch on 20M hours (NOT Whisper) - its own tech report says "replace Whisper"
- WHAT HOLDS: encoder trained with a transcription objective. AuT is 90% ASR data.
  **6 of 7 ASR-trained encoders invert vs 0 of 2 non-ASR. Fisher p = 0.083.**
- Caveats that must travel with it: predicate swap is post hoc; flamingo 2 has no signal to invert so
  the non-ASR group has effectively one informative member; kimi is Whisper-large-v3 and does NOT invert.

## Italian is channel-confounded
- Recording measurements alone (no speech content): 0.995
- Silence alone: 1
- File container alone, zero audio: 0.776
- 112 patient files contain runs of pure digital silence; 0 control files do
- All 394 control files 16 kHz; 160 patient files 44.1 kHz
- Ten recording days, not one contains both groups
- **THE CONTROL THAT SURVIVES THE AGE OBJECTION:**
  | comparison | both groups | speaker AUC | p |
  |---|---|---|---|
  | 2017-03-17 vs 2017-03-21 | both ELDERLY controls | 1.000 | 0.039 |
  | 2017-03-17 vs 2017-03-23 | both ELDERLY controls | 1.000 | 0.039 |
  | 2017-03-21 vs 2017-03-23 | both ELDERLY controls | 0.815 | 0.137 n.s. |
  | 2017-03-23 vs 2017-03-30 | both ELDERLY controls | 0.111 | 0.882 n.s. |
  Caveat: the separating session has only 2 speakers.
- Nested encoder probe on italian read picks LAYER 0 in every fold and every seed.

## Severity
- updrs ccc: neurovoz 0.61, pc-gita 0.45; hoehn-yahr nothing
- **omni tracks ptsd severity 0.602 vs depression severity 0.585** -> reads distress, not depression
- neurovoz age matching: encoder survives 0.949 -> 0.906; silence result dies 0.724 -> 0.580
- age vs updrs total on neurovoz, 47 patients: pearson 0.427, spearman 0.433
- our stored age_only 0.182 is an OUT-OF-FOLD CCC, not a correlation (0.315 in-sample)

## Prior work
- SpeechDx (arXiv 2606.17339): 12 clinical datasets, 12 encoders. NO audio LLMs, no voice-vs-words.
- FEND (arXiv 2512.20948): 13 datasets, splits speech vs text. NO parkinson's, LLMs are text-only on transcripts.
- LISTEN (EACL 2026): six audio LLMs, same transcribe-not-listen conclusion. Emotion only.
- VoxParadox (ICML 2026, Pang/Chaubey/Soleymani): the methodological parent. Synthetic, non-clinical.
- Only 8 papers cite both PC-GITA and DAIC-WOZ, 3 of those are noise.
- No health track in AIR-Bench, Dynamic-SUPERB, MMAU or AudioBench.

## Degeneracy flags (must appear next to any AUC)
- qwen2-audio says yes to 100% of 1117 pc-gita clips
- omni says no to 0% of them
- 15 of 118 audio-LLM cells degenerate by yes-fraction
- answer mass never drops below 0.839 in the 61-file grid
