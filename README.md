# The Representation Utility Gap in Audio LLMs for Clinical Speech

Chaitanya Parwatkar, Nima Kelidari, Minoo Ahmadi, Ashutosh Chaubey, Mohammad Soleymani
University of Southern California

Companion page for the ICASSP 2027 submission. It holds the numbers that did not fit in the four-page paper. Everything below uses the same probes, prompts, folds and scoring as the paper. All values are AUC against the clinical label unless stated otherwise. Segment identifiers and the scripts will be added here after the review period.

## 1. Interventions on the conflict sets

Conflict and agreement AUC for every intervention we tried. The paper reports the two that retrain a component and summarises the rest in one sentence. Depression rows use the E-DAIC conflict set (195 matched pairs, 74 speakers). Alzheimer's rows use the Pitt conflict set (146 conflict, 322 agreement segments).

| Intervention | Model | Conflict | Agreement |
|---|---|---|---|
| **Depression, weights frozen** | | | |
| the model's own answer | Qwen2.5-Omni | 0.30 | 0.87 |
| best of 12 prompt wordings | Qwen2.5-Omni | 0.34 | 0.80 |
| text subtraction, alpha = 0.9 | Qwen2.5-Omni | 0.50 | 0.72 |
| text subtraction, alpha = 2.0 | Qwen2.5-Omni | 0.67 | 0.18 |
| the model's own answer | Qwen2-Audio | 0.35 | 0.92 |
| best of 12 prompt wordings | Qwen2-Audio | 0.48 | 0.85 |
| activation steering, best strength | Qwen2-Audio | 0.54 | 0.50 |
| **Alzheimer's, weights frozen** | | | |
| the model's own answer | Qwen2-Audio | 0.41 | 0.70 |
| few-shot audio examples (answers Yes on 1% of clips) | Qwen2-Audio | 0.46 | 0.45 |
| probe's best encoder layer fed to the untrained projector | Qwen2-Audio | 0.55 | 0.47 |
| steering at the probe's layer, c = 1 | Qwen2-Audio | 0.56 | 0.42 |
| **Alzheimer's, one component trained** | | | |
| linear readout on the frozen hidden states | Qwen2-Audio | 0.61 | 0.86 |
| projector retrained on the encoder's final output | Qwen2-Audio | 0.48 | 0.86 |

Notes on the frozen interventions.

- Text subtraction removes the transcript-only logits from the audio logits. At alpha = 0.9 the answer stops following the words (agreement with the lexical direction falls from 0.70 to 0.50) and conflict reaches 0.50. Larger alpha reaches 0.67 on conflict only by collapsing the agreement arm to 0.18. Held-out speakers choose alpha = 0 in four of five folds.
- Activation steering adds a multiple of the difference-in-means diagnosis direction to the encoder output. A random direction of the same norm moves the answer as much as the diagnosis direction. Across 20 steered conditions the apparent conflict gain correlates at -0.92 with the lexical signal being destroyed. On Pitt at the probe's own layer, the diagnosis direction gives 0.56 conflict and 0.42 agreement against 0.58 and 0.54 for a random direction.
- Learned layer mixing over the encoder reaches 0.909 overall AUC on PC-GITA against 0.916 for one layer chosen inside the training folds. Conditioning the mixture on the prompt adds 0.0004.
- Feeding the probe's best encoder layer into the projector without retraining drops the overall Pitt answer from 0.62 to 0.49. The projector was trained on the final encoder output and does not accept another layer's output without retraining.
- A checkpoint trained with prompt-conditioned layer mixing and preference tuning (the VoxParadox recipe) scores 0.37 on the depression conflict pairs against 0.35 for its base model.

Overall Pitt numbers for the trained interventions: the linear readout reaches 0.78 (95% interval 0.74 to 0.83) where the model's own answer reads 0.62, a paired gain of +0.17 (0.10 to 0.23). The retrained projector reaches 0.76 (0.72 to 0.81).

## 2. Longer audio windows on depression

Qwen2-Audio accepts 30 s of audio. Three models that accept longer input were scored at a matched 300 s window on E-DAIC.

| Model | 300 s window | 30 s window |
|---|---|---|
| Audio Flamingo 3 | 0.86 | |
| Qwen2.5-Omni | 0.84 | 0.67 |
| Kimi-Audio | 0.83 | |

## 3. Transcript-only control, full detail

| Setting | Value |
|---|---|
| Pitt, transcript only, official transcript with annotation codes stripped | 0.68 (0.63 to 0.72) |
| Pitt, transcript only, raw transcript | 0.61 (0.56 to 0.66) |
| Pitt, audio, same clips | 0.62 (0.56 to 0.67) |
| Pitt, audio minus transcript, paired | -0.06 (-0.12 to +0.01) |
| ADReSSo, transcript only, whisper-large-v3 text | 0.68 (0.61 to 0.75) |
| ADReSSo, audio, same clips | 0.66 (0.60 to 0.74) |
| E-DAIC, Qwen2.5-Omni transcript-only answer against its own audio answer | r = 0.83, identical Yes/No on 96% of segments |

Words check. A probe trained only on Pitt segments where the words agree with the diagnosis reaches 0.95 within that set and at most 0.58 on the segments where the words contradict it. On ADReSSo, one clip per speaker, the same test gives 0.99 within and 0.39 to 0.61 on a conflict group of 19.

## 4. Recording and demographic controls

Six measurements that carry no speech content (clip duration, RMS level, spectral centroid, spectral rolloff, zero-crossing rate, estimated SNR) fed to the same logistic classifier as the probe.

| Dataset | Recording-only floor | Encoder probe |
|---|---|---|
| PC-GITA | 0.68 | 0.92 |
| NeuroVoz | 0.80 | 0.96 |
| MDVR-KCL | 0.74 | 0.89 |
| E-DAIC | 0.45 | 0.66 |
| Pitt | 0.65 | 0.78 |

NeuroVoz: patients' recordings are shorter than controls' (31 s against 47 s on average); duration alone gives 0.71. On age-matched speakers the encoder probe moves from 0.95 to 0.91 while a silence-only probe falls from 0.72 to 0.58.

Pitt: duration alone gives 0.61. Controls are younger (64 against 72 years) and age alone gives 0.70. On 64 age and sex matched speaker pairs (268 segments) the probe reads 0.79, the model's answer 0.66 and age alone 0.38.

Classical acoustic baseline, eGeMAPS features into the same logistic classifier: Pitt 0.62, ADReSSo 0.73, ADReSS-2020 0.70, MDVR-KCL 0.80.

Degenerate answering. Qwen2.5-Omni answers No on every Pitt clip (p_yes never exceeds 0.30 across 468 clips), so its Pitt AUC ranks confidences that never cross the threshold. Qwen2-Audio answers Yes on every PC-GITA clip.

## 5. Alzheimer's datasets beyond Pitt

| Dataset | Encoder probe | Model's answer | Transcript only |
|---|---|---|---|
| ADReSSo, 237 speakers, none in Pitt | 0.87 | 0.66 (0.60 to 0.74) | 0.68 |
| ADReSS-2020, official split, 48 test recordings | 0.80 AUC, 73% accuracy | 0.78 (0.63 to 0.90) | |
| ADReSS-2020, 108 training recordings | | 0.62 (0.51 to 0.73) | |
| ADReSS-2020, all 156 pooled | 0.82 | 0.68 (0.60 to 0.76) | |

The ADReSS-2020 challenge baselines on the same test split are 62.5% accuracy for acoustic features and 75% for manual transcripts.

## 6. Depth of the signal

Probes on the language model's stages at the audio-token positions: Pitt rises to 0.86 by the deep stages; MDVR-KCL holds 0.82 to 0.88 at every stage. A probe on the hidden state at the answer position reads 0.80 at the final layer on Pitt and 0.69 on the Pitt conflict segments, while the emitted answer on the same state reads 0.62.

## 7. Setup

- Probe: scikit-learn logistic regression, L2 penalty, C = 1, balanced class weights, 2000 iterations, features standardized inside the training fold. GroupKFold with 5 speaker-disjoint folds. The layer is chosen inside the training folds.
- Intervals: 2000 bootstrap resamples of clips, seed 0, percentile method. Paired rows resample the same clips for both arms.
- Answers: p_yes = P(Yes) / (P(Yes) + P(No)) at the first answer position, one forward pass, no generation.
- Projector retraining: AdamW, learning rate 1e-4, 3 epochs, one clip per update, cross entropy over the Yes and No logits only, encoder and language model frozen, 5 speaker-disjoint folds.
- Depression conflict set: participant turns from the E-DAIC transcripts after interviewer turns are removed by a question-stem rule, scored with a valence lexicon checked against an out-of-fold TF-IDF logistic classifier. 138 positive turns from depressed participants and 57 negative turns from controls, paired with matched agreement segments from the same speakers: 195 pairs over 74 speakers.
- Alzheimer's conflict set: Pitt segments scored by a text-only logistic model over CHAT fluency markers fitted out of fold, speaking rate excluded. 146 conflict and 322 agreement segments.
- Transcripts: Pitt official with annotation codes stripped; ADReSSo transcribed with whisper-large-v3; E-DAIC ships its own.
- Hardware: Alzheimer's runs on Apple Silicon (MPS); the nine-model sweep on one V100 or A40 per job.

## Contact

Chaitanya Parwatkar, parwatka@usc.edu. Corresponding author: Mohammad Soleymani, soleymani@ict.usc.edu.
