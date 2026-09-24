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

## The gap, measured on alzheimer's (same clips as the answers)
Encoder probe on qwen2-audio's own audio encoder, nested layer selection inside
speaker-disjoint folds (layers 18-27 chosen):
- all 468 segments: probe 0.78
- conflict arm:  probe 0.61 vs the model's answer 0.41
- agreement arm: probe 0.86 vs the model's answer 0.70
The marginal intervals are wide (conflict probe [0.49, 0.72]), so the claim is
made paired, on identical clips: probe minus answer is +0.195 [+0.025, +0.356]
on the conflict arm (p=0.011) and +0.162 [+0.084, +0.240] on agreement
(p=0.0005). The encoder knows significantly more than the answer uses, measured
on the same segments, for the third condition.

## Steering at the probe's own layer (the final local experiment)
Layer 22, the median nested choice. Direction = difference in class means, fit on
held-out speakers per fold, added to the hidden states with per-frame
renormalisation, projector and llm frozen. Matched-norm random control.
- baseline: conflict 0.41, agreement 0.70
- diagnosis direction, c=0.5/1/2: conflict 0.37/0.56/0.46, agreement 0.52/0.42/0.56
- random direction,   c=0.5/1/2: conflict 0.43/0.58/0.52, agreement 0.65/0.54/0.49
The random direction moves the answer as much as the diagnosis direction does,
agreement degrades immediately either way, and nothing is monotonic. Steering at
the layer the probe itself selected, on the clips where the paired gap is
significant (+0.195), still cannot push the answer toward the diagnosis. The
deficit is in the readout, now shown on all three conditions.

## Per-layer probing, encoder and language model (the meeting plot)
Requested in the Aug 21 meeting: probing across every stage, not just the chosen layer.
Same recipe everywhere: StandardScaler + logistic regression (balanced), 5-fold GroupKFold
by speaker, all 468 segments, 228 speakers.
- Encoder (33 states): 0.655 at the conv front end, climbs through the transformer,
  peak 0.824 at layer 24, 0.758 at the last layer. Full curve in ad_encoder_perlayer.csv.
- Language model at the audio token positions (projector then 32 layers, diagnosis
  prompt, no transcript given): 0.754 at the projector, rises steadily to 0.862 around
  layer 29, 0.851 at the final layer. Full curve in ad_llm_perlayer.csv.
- The model's own answer on the same clips: 0.618 (recording prompt), 0.633 (voice prompt).
So the information is present in the encoder, survives the projector, becomes MORE
linearly readable through the language model, and the answer still does not use it.
One caveat that travels with the LLM half: on this task the words carry signal too
(text baseline 0.699), so part of the late rise can be internal transcription; the
matching Parkinson's read-speech result (July run: KCL stays 0.82-0.88 through the LLM,
words carry nothing there) closes that hole.

## Plugging the probe's best layer into the projector (no training)
Encoder layer 24 (the per layer probe's peak, 0.824) fed to the projector in place of
the final encoder layer, everything frozen, all 468 segments rescored with the same
diagnosis prompt. It does not work: overall answer AUC 0.488 vs baseline 0.618,
agreement 0.469 vs 0.699, conflict 0.554 vs 0.413, and the yes rate collapses from
80 percent to 9 percent (answer mass stays 0.995). The projector expects the final
layer's distribution, so the training free plug is out. The trainable versions
(retrain the projector on layer 24, or a small trained readout on the frozen encoder)
are the remaining candidates for a fix section.

## The last three fixes, all run locally (26 Aug)
- Few shot, one dementia and one healthy example in the prompt: kcl read falls 0.65 to 0.46,
  alzheimers falls 0.62 to 0.45, and the model answers no to nearly everything both times.
  Multi audio prompting collapses it, as predicted in the meeting.
- Projector retrained on encoder layer 24, everything else frozen, 2 of 5 folds, 3 epochs:
  0.596 out of fold. It recovers from the broken untrained plug (0.488) but does not beat
  the original answer (0.618). Changing the layer is not the bottleneck.
- Text only through the model itself: qwen2 audio reading just the transcript scores 0.613
  overall vs 0.618 with audio, agreement 0.671 vs 0.699, conflict 0.472 vs 0.413. The two
  routes land in the same place. Per clip correlation is modest though, r 0.24, same yes/no
  55 percent, so on alzheimers the aggregate match is the claim, not clip level copying.
- Text subtraction on alzheimers, from the same files: pushing the words out lifts conflict
  only to 0.476 while agreement falls to 0.454. Same trade as depression, no free lunch.
Scores stay local per the data agreement (ad_fewshot_scores.csv, ad_textonly_scores.csv).

## CORRECTION and the real projector result (28 Aug, all five folds)
The earlier "0.596, only gets back to even" line was wrong: that number scored a
half empty prediction file while three folds were still queued. With all five folds
done, retraining ONLY the projector on encoder layer 24 (everything else frozen,
3 epochs, speaker disjoint) moves the model's own answers to 0.798 overall
(baseline 0.618), agreement 0.887 (was 0.699), conflict 0.573 (was 0.413).
Per fold 0.795 to 0.888, no weak fold. Same supervision as the readout, so the
fair comparison is readout 0.824 vs retrained projector answers 0.798: the model
can be made to use what it hears by retraining its smallest part.

## Words check on Pitt (pitt_words_check.csv, from ad_arm_transfer_curves.npz)
Probe trained only on the agreement segments, tested within agreement (5 fold) and on the
conflict segments, every layer. Encoder: within agreement peaks at 0.955 (layer 24), on
conflict at most 0.582 (layer 21). Language-model stages: 0.953 within, at most 0.547 on
conflict. The arms are tertile-selected and share 47 of 228 speakers, so this is the same
caveat as the conflict number itself. Same picture as ADReSSo (one clip per speaker there).

## Bootstrap intervals regenerated (scripts/paper1/bootstrap_cis.py, 11 Sep)
2000 clip-level resamples, seed 0, one script for every row of results/health/bootstrap_cis.csv.
Pitt transcript only, clean text: 0.676 [0.626, 0.723]. Audio answer minus clean transcript,
paired on the same 468 clips: -0.059 [-0.124, +0.009]. Conflict arm, paired: readout minus
answer +0.195 [+0.043, +0.344], projector minus answer +0.160 [+0.025, +0.298].

## No-plug control: projector retrained on its default input (11 Sep, train_projector_noplug.log, projector_noplug_oof.npy)
Same recipe as the layer-24 run (AdamW 1e-4, 3 epochs, fp32 projector, everything else frozen,
speaker-disjoint 5 fold) but the projector keeps its default input, the final encoder layer (32).
Overall 0.763 [0.720, 0.806]; agreement 0.865; conflict 0.484 [0.384, 0.582].
Paired against the answer: +0.145 [+0.082, +0.209]. Layer 24 minus no plug, paired: +0.035
[-0.004, +0.073] overall, +0.089 [+0.004, +0.177] on the conflict arm.
Reading: the training carries most of the gain; the earlier layer adds a little overall and
matters on the conflict arm. This separates "training the projector" from "changing its input".

## Age and sex matched evaluation (11 Sep, scripts/dementiabank/age_matched_eval.py, age_matched_eval.csv)
Controls are younger (speaker mean 64.3 vs 72.0 years over the 228 speakers) and age alone reads 0.70
on the 468 segments. Matching within the 228 speakers with the age_match_pitt.py rule (same sex, nearest
age within 3 years, Probable/Possible AD only) gives 64 pairs, 268 segments, mean age AD 65.8 vs control
67.0, age alone 0.38. On those segments the existing out-of-fold encoder probe reads 0.79 and the model's
answer 0.66 (agreement n=195: 0.84 / 0.72; conflict n=73: 0.65 / 0.47). Evaluation only is restricted;
folds and layer were fit on all speakers. The gap is not an age effect.

## Recording floor on Pitt (DementiaBank/floor.log)
Six recording measurements with no speech content: duration alone 0.61, level 0.52, silence fraction
0.59, noise floor 0.51, centroid 0.57, rolloff 0.55; all six together 0.65. Below the text baseline
(0.70) and the probe (0.78).

## Synthetic voice provenance (Minoo Ahmadi's run on CARC)
The Pitt synthetic-voice and words-only numbers in the paper (qwen2-audio 0.60 / 0.72, qwen2.5-omni
0.67 / 0.72, diva 0.69 / 0.71, mimo-audio 0.58 / 0.57) come from Minoo's Piper TTS run on the cluster
(pitt468_tts_minoo, daic275_tts_minoo) and were read from the shared Google Doc TTS tab. Per-clip csvs
are not on this disk yet; requested. The qwen2-audio real-voice 0.62 and qwen2.5-omni real-voice 0.68
are p_main from ad_scores_qwen2audio.csv and ad_scores_omni.csv (recording wording).
