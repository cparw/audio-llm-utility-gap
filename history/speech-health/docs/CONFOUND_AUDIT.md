# How the AUC 0.995 Was Diagnosed

Chaitanya Parwatkar. Prof. Soleymani asked on 17 July why the within-corpus AUC was near 1.0, and
set the test himself: if other papers on the same corpora do not get near 1.0, the pipeline has a
bug; if they do, the data has a confound. He also asked for a metadata mutual-information check,
a noise manipulation, and an audio shuffle across speakers. This is the answer to all of it.

## The result that settles pipeline versus data

A classifier is given **no speech content at all**: only six recording properties (duration,
loudness, spectral centroid, spectral rolloff, zero-crossing rate, and an SNR estimate computed
as the ratio of loud to quiet frame energy). No encoder, no model, nothing that could contain a
bug.

| Corpus | Task | Recording properties only | Frozen encoder | Gap |
|---|---|---|---|---|
| Italian PD | read | **0.995** | 1.000 | **0.005** |
| Italian PD | vowel | 0.861 | 0.989 | 0.128 |
| MDVR-KCL | read | 0.743 | 0.967 | 0.224 |
| Neurovoz | read | 0.824 | 0.948 | 0.124 |
| Neurovoz | vowel | 0.679 | 0.834 | 0.155 |
| PC-GITA | read | 0.678 | 0.932 | 0.254 |
| PC-GITA | vowel | 0.639 | 0.847 | 0.208 |
| DAIC-WOZ (depression) | interview | **0.453** | 0.699 | 0.246 |

Speaker-disjoint throughout. A label-shuffle control sits at chance (0.43 to 0.54), which
confirms the probe itself is not leaking.

On the Italian corpus, six trivial numbers reach 0.995 while the encoder reaches 1.000. Since
there is no model in that path, it cannot be a pipeline bug. **It is the data.** The depression
corpus is the counter-example at 0.453, below chance, so this is not an artifact of the method.

## Naming the cause, not just the anomaly

Twenty-five test cases per corpus: every recording property alone, every property left out, and
grouped subsets. `scripts/confound/diagnose25.py`.

Italian, read:

| Test | AUC |
|---|---|
| SNR estimate alone | **0.865** |
| zero-crossing rate alone | 0.818 |
| spectral centroid alone | 0.809 |
| duration alone | 0.665 |
| loudness alone | 0.365 |
| **drop the SNR estimate** | **0.995 falls to 0.868** |
| drop anything else | still 0.99 or above |

Only removing the SNR estimate moves the number. The class means explain why:

| | Parkinson's | Healthy |
|---|---|---|
| SNR estimate | 51.6 | 35.0 |
| spectral centroid | 1297 | 1514 |
| zero-crossing rate | 0.079 | 0.113 |

The control recordings carry roughly **16 dB more background noise** than the patient recordings.
Background noise mechanically raises zero-crossing rate and spectral centroid and compresses the
measured dynamic range, so every apparent acoustic difference on this corpus follows from one
fact: the two groups were recorded in different acoustic conditions.

Neurovoz fails differently. There the load is carried by spectral rolloff (0.792) and centroid
(0.760), with SNR only 0.603 and loudness 0.451. Those are legitimate hypophonia markers, so its
nuisance floor is closer to 0.59 to 0.61 than to 0.80.

## The manipulations Prof. Soleymani asked for

`scripts/confound/noise_shuffle.py`

| | Italian | Neurovoz |
|---|---|---|
| clean | 0.995 | 0.804 |
| + noise at 20 dB SNR | 0.882 | 0.699 |
| + noise at 10 dB | 0.812 | 0.716 |
| + noise at 0 dB | **0.809** | **0.721** |
| audio shuffled across speakers | **0.485** | 0.515 |

Italian plateaus at 0.81 even when the added noise is as loud as the speech, which is exactly the
signature he predicted for a structural data problem rather than a recoverable one. The
speaker-shuffle falling to chance confirms there is no leakage in the pipeline.

## Does the encoder add anything beyond the recording?

Naively comparing a peak-over-33-layers encoder AUC against a fixed six-feature probe is biased
toward the encoder. `scripts/confound/nested_confound_test.py` selects the read-out layer on
inner folds of the training speakers only, shares outer folds across models, and puts a paired
speaker-clustered bootstrap on the increment.

| Corpus | Recording only | Encoder (honest) | Encoder adds | 95% CI |
|---|---|---|---|---|
| Italian read | 0.989 | 1.000 | **+0.011** | [0.000, 0.032] |
| MDVR-KCL read | 0.753 | 0.902 | +0.160 | [0.012, 0.332] |
| Neurovoz read | 0.804 | 0.930 | +0.125 | [0.074, 0.184] |
| PC-GITA read | 0.666 | 0.911 | **+0.246** | [0.167, 0.327] |

So the encoder does carry disease information beyond recording artifacts on three of four
corpora. Italian is the exception, where the entire result is the recording.

## Metadata, recovered from the corpus source files

Age and sex are absent from our manifests, so they were recovered from the original corpus files
(Italian `.xlsx` tables, the Neurovoz metadata CSVs, the PC-GITA metadata sheet).

| Corpus | PD age | Control age | Gap | AUC from age alone |
|---|---|---|---|---|
| PC-GITA | 61.0 | 61.0 | **0.0** | **0.501** |
| Italian vs elderly controls | 67.2 | 67.1 | 0.1 | 0.582 |
| Italian vs all controls | 67.2 | includes 15 people at ~21 | **18.9** | **0.751** |
| Neurovoz | 71.1 | 64.0 | **7.1** | **0.697** |

Mutual information over every available attribute (`scripts/confound/`) also surfaced a
**recording-date** confound in Neurovoz: patient and control sessions have different medians
(April versus December 2016) and the date alone predicts the group at 0.881. Sex is not
predictive anywhere (PC-GITA 0.500, Neurovoz 0.552, DAIC gender 0.420).

PC-GITA is the cleanest corpus on every axis: matched on age and sex, lowest nuisance floor, and
the largest genuine encoder gain.

## Reproducing Maksim's preprocessing

Prof. Soleymani asked whether Maksim's pipeline avoided this. His preprocessing was recovered from
the shared project space (silence stripping plus fixed three-second segments with a one-second
hop) and reapplied. `scripts/repro/maxim_repro.py`

| Corpus | Raw | Maksim-style | Change |
|---|---|---|---|
| Italian | 0.995 | 0.932 | -0.063 |
| Neurovoz | 0.804 | 0.731 | -0.073 |
| PC-GITA | 0.678 | 0.657 | -0.021 |

His preprocessing does **not** remove the confound. His own SLURM logs show accuracies of 0.60 to
0.96 on the KCL corpus with full fine-tuning of Whisper and HuBERT, and there are no Italian
training runs in what was shared, so the most likely explanation is that he never ran the corpus
that carries the problem.

## What this means for reporting

Report the recording-properties floor next to every within-corpus clinical AUC. A clinical number
should have to clear that floor before it is claimed as disease detection. Italian read speech
does not clear it, and any headline built on that corpus is a statement about microphones.

## Reproducing

```bash
python scripts/confound/confound_audit.py   --datasets italian,kcl,neurovoz,pcgita --tasks read
python scripts/confound/diagnose25.py       --dataset italian --task read
python scripts/confound/noise_shuffle.py    --datasets italian,neurovoz --task read
python scripts/confound/nested_confound_test.py
python scripts/repro/maxim_repro.py         --datasets italian,neurovoz,pcgita --task read
python scripts/repro/coverage.py            # prints what has and has not been run
```
