# DISCREPANCIES.md: E-DAIC rerun and data audit

Items 3, 4, 5, 6, 7, 8, 12, 13 and 14 describe the 14 September draft and are resolved in the final tex.

This file carries every item of the working audit log as it stood on 23 September 2026 at 13:26 UTC: 176 numbered items (128 headed items and 48 standalone bullets). 4 headings only group the bullets under them and carry no finding of their own, so they are not numbered.

Every numbered item ends with a line that starts "Status now:". It says one of three things. Fixed in the paper: the line quotes, verbatim, the sentence of the final tex that fixes or avoids the finding. Released here: the line names the files that document or release it. Most are in this repository. A path followed by (authors' machine path) stays on the authors' machine, where release/ stands for the authors' release folder. Open: the line says what decides it. Counts: 43 fixed in the paper, 128 released here, 5 open (items 2, 69, 84, 125 and 149).

The status lines were checked again on 24 September 2026 against the final tex, the Overleaf copy of 23 September 2026, 23:23 UTC (md5 2923ced559329a4413b75efbaf477168). Every quoted sentence is in that tex, outside its comments.

Paths inside the findings that start with release/, gdrive/ or DementiaBank/ stay on the authors' machine. release/ is the authors' release folder, gdrive/ is the folder of local runs on the authors' external drive, and DementiaBank/ is the authors' copy of that corpus. Paths that start with /workspace/, /root/ or /scratch1/ were on the GPU pods (rented cloud GPU machines). None of these resolve in this repository. Paths in the status lines are relative to the repository root, except the ones marked (authors' machine path).

Participant utterance text, per-speaker PHQ-8 values with their ids, participant, speaker and clip ids, and temporary folder paths are withheld in the findings and marked where they were. The findings are otherwise kept as logged, with formatting fixes only.

The log as it was kept:

Run started 2026-09-22. HIGH items were logged as soon as they were found.

---
## 1. D1. Qwen2.5-Omni cannot ingest "full" E-DAIC files. SEVERITY: HIGH
What the data shows. dcaps_proc file lengths (n=275, from `audio_alignment_audit.csv`):
p0 65.9 s, p25 334.2 s, p50 591.8 s, p75 1088.2 s, p95 1910.6 s, p100 2847.4 s. Total 58.8 hours.
Qwen2.5-Omni emits ~25 audio tokens per second of audio and has a 32,768-token context.
That gives median 14,795 tokens, p75 27,205, p95 47,766, max 71,185.
52 of 275 files (19%) exceed the context and cannot be encoded whole at any setting.
A 900 s ("central 15 minutes") cap is 22,500 tokens and still truncates 89 of 275 files.
Paper sentence it contradicts. Any claim that the E-DAIC condition uses the full interview.
What would fix it. Report the E-DAIC audio condition as an explicitly capped window with the cap
stated and the number of truncated files printed, or move to a chunk-and-pool design that encodes
the file in segments and pools the states. Do not call it "full".
Status. Proceeding with an empirically determined cap. Every capped file is listed in the sidecar.

Status now: fixed in the paper. Quote: "The model hears the participant's whole interview, interviewer turns removed and capped at 15 minutes for the 89 longest, with the processor's default input limit lifted so that every window is encoded whole." The paper's E-DAIC audio numbers now come from the run with the processor's 300 s cut lifted (PART 23): zero-shot answer 0.8366 [0.7786, 0.8866], against 0.8278 with the cut, paired +0.0088 [-0.0320, 0.0510] (part23/verify/PART23_VERIFIED.tsv, authors' machine path). A second run of the same setting gives 0.8379 [0.7805, 0.8876]. Files: scores/part22/P22_step1_processor_config.md, scores/part22/p22_step2_truncation_proof.json, scores/part22/edaic_lifted_perclip.csv, scores/part22/edaic_lifted_auc.json, manifests/edaic_windows_full.csv. 89 of the 275 windows still reach the 900 s cap, as the quoted sentence says.

---
## 2. D2. dcaps_proc is a stitched concatenation, not the raw interview. SEVERITY: HIGH
What the data shows. `dcaps_prep.py` concatenates every transcript row with End > Start.
Ratio dcaps_proc / raw tarball audio over 275 speakers: median 0.5416, p25 0.4020, p75 1.2883,
max 2.4595, and 0 of 275 within 5% of 1.0. Ratios above 1.0 mean overlapping ASR rows are
duplicated into the stitched file.
Consequence for this run. "First 30 s", "middle 30 s" and "middle 300 s" of dcaps_proc are
positions in a stitched artifact, not timestamps in a real interview. A mid-file window is not
"the middle of the conversation". It is the middle of a concatenation whose ordering follows the
raw ASR rows including the interviewer.
Paper sentence it contradicts. Section 3's description of the E-DAIC audio input.
What would fix it. Cut windows from the raw `<pid>_AUDIO.wav` using the transcript timings
(now downloaded for all 275 speakers), not from dcaps_proc.
Status. Logged. Windows for this run are cut from dcaps_proc as planned. Results are marked
AFFECTED in every sidecar.

Status now: open. The final tex reads "The model hears the participant's whole interview, interviewer turns removed and capped at 15 minutes for the 89 longest, with the processor's default input limit lifted so that every window is encoded whole." The file is the participant's transcript rows with End > Start, joined in file order (the rule in scripts/edaic_full/build_windows.py), so rows that overlap are heard twice. The tex does not say this. What decides it: a recut from the raw recording. Related files: scripts/edaic_full/build_windows.py, manifests/edaic_windows_full.csv.

---
## 3. D3. The E-DAIC first 30 s is setup chatter, not interview speech. SEVERITY: HIGH
What the data shows. Over the 275 raw transcripts, the FIRST 30 s window contains a median of
11 words total per speaker and a median of 7 participant words.
188 of 275 speakers (68.3636%) have fewer than 15 participant words. 47 have zero. 44 have no
transcript row in the window at all.
The ten most common opening stems in that window are the technician setting up the session, not Ellie:
[four opening stems withheld here, transcript text is not released].
First 20 words for random speakers read: [four utterances withheld here],
and two EMPTY windows.
By contrast the FULL file shows a real interview script (what 617, how 556, tell me 276).
Paper sentence it contradicts. The Table 1 E-DAIC cell and Section 3's description of the E-DAIC
audio condition. The model is being asked to detect depression from the sound of a microphone check.
What would fix it. Score E-DAIC on a window that contains actual interview speech. The mid-30 s and
mid-300 s variants in this run are exactly that test.
Status. This is the strongest single result of the audit and it is why the rerun matters.

Status now: fixed in the paper. This item describes the 14 September draft. Quote: "On E-DAIC the audio answer on the whole interview matches the answer from the whole transcript." The first 30 s window of the 14 September draft is gone from Table 1. The final tex reports it only as one step of the window series: "The E-DAIC answer rises with the audio it hears: 0.66 on the first 30\,s, which is mostly session setup, 0.72 on a middle 30\,s, 0.81 on a middle five minutes and 0.84 on the whole interview, while the encoder probe never exceeds 0.63 on any window."

---
## 4. D4. Scored windows contain interviewer speech on the AD sets. SEVERITY: HIGH
What the data shows.
- Pitt: 213 of 468 windows (45.5128%) contain interviewer words. Timed INV overlap > 0 s in 168/468,
  > 2 s in 67/468, longest exactly-timed stretch 12.8170 s.
- ADReSS-2020: 114 of 156 windows (73.0769%) contain interviewer speech. INV audio > 2 s in 60/156,
  max 17.7990 s. One sample window (S143) is interviewer speech for all of its first 20 words.
- ADReSSo: no timed transcript exists at all, so this cannot be measured directly. ASR of the cut clips
  shows 10 of 237 still opening with an interviewer prompt, which is a lower bound.
Paper sentence it contradicts. "the 30 s window starting at the patient's first words".
Windows starting at 0.0000 s with no shift at all: ADReSSo 90/237, ADReSS-2020 59/156.
Also. `pitt_conflict_manifest.csv` column `inv_ms_in_window` is 0.0000 on all 468 rows. That is an
artefact of 795 Pitt `*INV:` utterances carrying no time bullet, not evidence the interviewer is absent.
Do not cite that column.
What would fix it. Recut from the first timed participant utterance and exclude INV spans. PART 5.

Status now: fixed in the paper. This item describes the 14 September draft and is resolved in the final tex. Quote: "Each Pitt segment starts at the participant's first timed utterance, and the ADReSSo and ADReSS-2020 windows start at the beginning of the recording, so interviewer prompts inside the window are kept."

---
## 5. D5. "468 thirty-second segments" is false. SEVERITY: HIGH
What the data shows. Pitt clip length: min 30.0100, p25 31.3175, median 33.2850, p75 35.9375,
max 55.5600, mean 34.5104, sd 4.3122. Exactly 30 s: 0 of 468. Longer than 30 s: 468 of 468.
Paper sentence it contradicts. Section 3's "468 thirty-second segments".
What would fix it. State the real distribution, or recut to a true 30 s.

Status now: fixed in the paper. This item describes the 14 September draft. Quote: "Segments run from 30 to 56\,s (median 33\,s) and the models hear the first 30\,s of each." The dataset sentence gives the counts without a thirty-second claim: "468 segments from 227 participants".

---
## 6. D6. Heavy speaker overlap across Pitt, ADReSSo and ADReSS-2020. SEVERITY: HIGH
What the data shows. Recovered two independent ways (transcript identity and acoustic identity),
agreeing 156/156 and validated 48/48 against the released crosswalk.
- ADReSS-2020 ∩ ADReSSo = 144 speakers, 92.3077% of ADReSS-2020
- Pitt probe set (228 speakers) ∩ ADReSS-2020 = 123, ∩ ADReSSo = 162, ∩ either = 170 of 228 = 74.5614%
- three-way overlap = 115. Same Pitt recording present in both = 110
Clean remainders: ADReSS-2020 only 12, ADReSSo only 62, Pitt only 58.
Paper sentence it contradicts. Any treatment of Pitt, ADReSSo and ADReSS-2020 as three datasets, and
any cross-dataset transfer claim among them.
What would fix it. Restrict transfer claims to the clean remainders, or state the overlap.

Status now: fixed in the paper. This item describes the 14 September draft and is resolved in the final tex. Quote: "The three sets share speakers, so every result is within one set on speaker-disjoint folds and we make no transfer claim among them."

---
## 7. D7. The set called ADReSSo is not ADReSSo. SEVERITY: HIGH
What the data shows. `challenges/ADReSSo/` is empty. The 237 clips come from
`challenges/ADReSS-M/ADReSS-M-train_x/train` (ADReSS-M English training split, ids adrso002...).
Paper sentence it contradicts. Every mention of ADReSSo as the dataset.
What would fix it. Rename to ADReSS-M, or obtain real ADReSSo.

Status now: fixed in the paper. This item describes the 14 September draft. Quote: "We use the DementiaBank Pitt cookie-theft descriptions \cite{pitt}, with 468 segments from 227 participants (one has sessions in both groups and is grouped under two labels), the top and bottom third of the fluency score defined under Conflict set construction below, the ADReSSo recordings \cite{adresso} from 237 speakers, as released with the ADReSS-M challenge \cite{adressm}, and the ADReSS-2020 benchmark split \cite{adress2020} including 108 training and 48 test recordings."

---
## 8. D8. NeuroVoz is channel confounded. SEVERITY: HIGH
What the data shows. One scalar from the silence between words, the quiet-frame spectral centroid,
gives a speaker-level AUC of 0.8650 (perm p 0.0000, n=107). PD median 3958.8774 Hz vs control
4717.5089 Hz. Roll-off 0.7986 and tilt 0.8168 repeat it. PD noise floor sits 2.7689 dB higher.
That beats several zero-shot audio-LLM rows in Table 1.
PC-GITA is milder and floor-only (0.6872). MDVR-KCL flags loudness 0.7054, duration 0.6994, SNR 0.6905
but with 16 PD / 21 HC the p values are 0.0333 / 0.0426 / 0.0505, so suspicious rather than proven.
Clean: Pitt (max 0.5943), ADReSSo (0.5650), ADReSS-2020 (0.6358), E-DAIC (0.5737).
What would fix it. Report the confound floor per dataset, as Paper 2 does.

Status now: fixed in the paper. This item describes the 14 September draft. Quote: "Five recording features from the quietest frames (noise floor, SNR, spectral centroid, roll-off and tilt) separate the NeuroVoz groups at 0.80 and the PC-GITA groups at 0.67, so part of the Parkinson's probe may come from the recording channel." and "Normalising loudness and noise floor brings them to 0.56 on PC-GITA, where the encoder probe falls to 0.83 ($-$0.07 [$-$0.12, $-$0.02]) and the answer stays at 0.57, and only to 0.60 on NeuroVoz, where the probe falls to 0.89 ($-$0.03 [$-$0.061, $-$0.001])."

---
## 9. D9. PHQ-8 binary threshold does not reproduce the label. SEVERITY: MEDIUM
What the data shows. Every y=1 speaker has sev >= 10, but 20 of 209 y=0 speakers also have
sev >= 10 (up to 16). [Participant ids withheld here, clinical scores are not released.]
Class balance y=1 66 (24.0000%), y=0 209 (76.0000%).
What would fix it. State the actual label source rather than "PHQ-8 binary at threshold 10".

Status now: fixed in the paper. Quote: "We use E-DAIC \cite{daic}, the extended dataset released for AVEC 2019, with 275 participants, where the binary label is the dataset's released label based on the eight-item Patient Health Questionnaire (PHQ-8) \cite{Kroenke2009}, stricter than PHQ-8 $\geq 10$ for 20 participants, rather than a clinical diagnosis."

---
## 10. D10. PC-GITA and NeuroVoz have no 30 s window. SEVERITY: MEDIUM
What the data shows. PC-GITA 1100 clips median 3.4240 s. NeuroVoz 1270 clips median 2.9722 s.
Each scored file is one whole short utterance. A 30 s window does not exist for either set.
What would fix it. Do not describe a uniform 30 s window across all seven datasets.

Status now: fixed in the paper. Quote: "From the Parkinson's disease datasets, we only use material whose words are fixed by the protocol, so the words carry little of the diagnosis, and most of the signal has to come from the voice: PC-GITA \cite{pcgita} (Spanish, ten sentences and one read text per speaker, 1100 clips), NeuroVoz \cite{neurovoz} (Spanish, twelve isolated words, 1270 clips from 107 speakers) and MDVR-KCL \cite{kcl} (English, 37 speakers, most reading one of two passages)." The 30 s window is stated for the Alzheimer's sets only.

---
## 11. D11. NeuroVoz is stored twice. SEVERITY: LOW
2903 shared basenames between `zenodo_upload/audios` and `zenodo_upload_silencemode_1/audios`,
paired cosine 1.000000, durations identical for 2102 of 2903, md5 identical 0 of 2903 (the second tree
is silence-trimmed). No byte-identical duplicates anywhere (8414 files, md5). No genuine near-duplicate
audio within any set once the log-energy coefficient is dropped from the fingerprint.

Status now: released here. Files: manifests/part1e/fp.csv, scripts/part1e/fingerprint.py, scripts/part1e/dupes2.py. fp.csv holds the md5 and fingerprint of both NeuroVoz trees (2903 files each).

---
## 12. D12. The Pitt 468 is a tertile-selected subset, not a sample of Pitt. SEVERITY: HIGH
What the data shows. `af2_run/speech-health/scripts/dementiabank/build_ad_conflict.py` lines 98-106
keeps ONLY the extreme tertiles of an out-of-fold text-fluency score:
```
lo, hi = np.percentile(oof, [33, 67])
label==1 and p <= lo -> conflict
label==0 and p >= hi -> conflict
label==1 and p >= hi -> agreement
label==0 and p <= lo -> agreement
```
The middle tertile is discarded. Verified against the shipped manifest:
conflict 146 (101 dementia-fluent + 45 control-impaired, 100 speakers),
agreement 322 (189 + 133, 175 speakers). 146 + 322 = 468.
So the "468 thirty-second segments" that the Table 1 Pitt row is built on IS the conflict+agreement
union. The Pitt probe is trained and evaluated on a deliberately polarised subset chosen by a text
model, with the middle third of segments removed.
Paper sentence it touches. Section 3's "468 thirty-second segments" and every Pitt cell in Table 1,
plus the Section 4.2 Pitt probe figure.
What would fix it. Rebuild the segment list keeping every tertile and report the probe on the full
set, and separately on the middle tertile, so the selection effect is visible. That is PART 13.
Status. PART 13 launched.

Status now: fixed in the paper. This item describes the 14 September draft and is resolved in the final tex. Quote: "On all 708 Pitt segments, middle third included, the encoder probe reaches 0.80 and the zero-shot answer 0.67." The dataset sentence also names "the top and bottom third of the fluency score".

---
## 13. D13. The E-DAIC valence lexicon (dq_lex) is not on this machine. SEVERITY: HIGH
What the data shows. `px_build.py` line 12 does `import dq_lex as LX` from
`/scratch1/parwatka/pd_probing/code_clean`, a cluster path. The module is not on this Mac or the
G-Drive. The archived working log records that dq_lex is not on disk, so the lexicon it wraps
is not known.
The manifest carries its OUTPUTS (`val`, `npos`, `nneg`) per segment, but not the lexicon itself, so
truncated text cannot be rescored under the identical rule.
Consequence for PART 7. The lexicon half of the conflict rule cannot be reproduced bit-identically.
The TF-IDF half IS fully specified in px_build.py lines 140-151 and is reproducible.
What would fix it. Obtain dq_lex.py, or restate the paper's conflict rule in terms of a named,
public lexicon so it is reproducible by a reader.
Status. PART 7 proceeds on the reproducible TF-IDF rule plus a named public sentiment model, with
the lexicon half explicitly marked NOT REPRODUCED rather than silently substituted.

Status now: fixed in the paper. This item describes the 14 September draft. Quote: "From the raw E-DAIC transcripts, with the interviewer removed, we take every participant segment of at least 25 words and 5\,s of speech and score it with the Twitter-RoBERTa sentiment classifier trained on TweetEval \cite{tweeteval}." The paper's conflict rule names a public sentiment model. The valence lexicon is not used.

---
## 14. D14. The E-DAIC answer is not weak. The window was. SEVERITY: HIGH
What the data shows. Qwen2.5-Omni zero shot, same prompt, same scoring, only the window changes:

| window | n | AUC | vs paper |
|---|---|---|---|
| first 30 s (the paper's cell) | 275 | 0.6620 | n/a |
| middle 30 s | 275 | 0.7205 | +0.0585 |
| middle 300 s | 275 | 0.8122 | +0.1502 |
| full file, 900 s cap, 89/275 capped | 275 | 0.8285 | +0.1665 |

The E-DAIC encoder probe on the paper's window is 0.5966. On the full file the model's OWN ANSWER
reaches 0.8285, which BEATS that probe by +0.2319.
Paper sentence it contradicts. The central claim, for E-DAIC: that a probe recovers information the
model's own answer cannot use. On E-DAIC that ordering reverses once the model is given interview
speech instead of the first 30 s, which D3 showed is a microphone check (median 7 participant words,
68.36% of speakers under 15 words, 47 with zero).
What would fix it. Either report E-DAIC on a window containing actual speech and drop it from the
utility-gap claim, or state explicitly that the E-DAIC gap is a windowing artefact.
Status. Probes, text and fine-tune on all three windows still running. This is the zero-shot row.

Status now: fixed in the paper. This item describes the 14 September draft. Quote: "The E-DAIC answer rises with the audio it hears: 0.66 on the first 30\,s, which is mostly session setup, 0.72 on a middle 30\,s, 0.81 on a middle five minutes and 0.84 on the whole interview, while the encoder probe never exceeds 0.63 on any window." and "On E-DAIC the language model probe reaches 0.74 and the encoder probe 0.60, both below the answer of 0.84 ($-$0.09 [$-$0.16, $-$0.03] and $-$0.24 [$-$0.32, $-$0.15]), while a probe on the answer state matches the answer (0.84)." The earlier sentence on the order of the gap across conditions is gone from the final tex.

- 15. [HIGH] M10 (part16) KCL transcript AUC recomputed 0.7634 from release/overnight2/text_new/o25_kcl_text.csv does not match published 0.7619 in master_lookup.csv / release/overnight2/text_new/o25_kcl_text.json

  Status now: fixed in the paper. Table 1 row, verbatim: "MDVR-KCL    &   37 (37)  & 0.71 & 0.76 & 0.70 \\". Table 1 prints 0.76, the value under both tie rules (item 25). Evidence in this repository: scores/part16/M/M10_kcl_transcript.csv.

- 16. [MEDIUM] M10 (part16) KCL Omni encoder probe: recomputed 0.7768 from the single saved OOF file release/omni_final/omni_kcl_enc_nested_oof.csv. master_lookup publishes 0.7506 which is the mean of 5 nested repeats in release/omni_final/omni_kcl_nested_repeats.json (per_repeat [0.744, 0.8304, 0.7768, 0.625, 0.7768]). The OOF csv holds ONE repeat, so the two are not the same estimator. Not an error, but the published encoder number cannot be reproduced from the per-clip file alone.

  Status now: released here. Files: lookup/master_lookup.csv, folds/kcl_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/kcl_groupkfold5_pod_seed4_UNVERIFIED.csv, folds/README.md. The lookup gives the MDVR-KCL encoder probe 0.7506 as the mean of five repeats. No per-clip file of the five repeats was saved, so the fold files reproduce the saved per-repeat AUCs only.

- 17. [LOW] M10 (part16) No saved fold assignment file was found anywhere under release or gdrive (searched *fold*.csv/json/npz), and the KCL *_nested_oof.csv files carry no 'fold' column. M10 did NOT need folds: it recomputes AUC from already-out-of-fold saved scores. No folds were recomputed.

  Status now: released here. Files: folds/kcl_groupkfold5_mac.csv, folds/kcl_groupkfold5_pod.csv, folds/README.md.

## PART16 M7: paired encoder probe vs zero-shot answer (Qwen2.5-Omni, 7 datasets)
run: /usr/local/bin/python3 release/edaic_rerun/part16/M7_gap_per_dataset.py

(Group heading with no finding of its own. Items 18 to 24 below belong to it.)

- 18. HIGH [PART16/M7] pitt encoder probe: recomputed from omni_pitt_enc_nested_oof.csv = 0.7969, master_lookup.csv = 0.7706 (estimator there: five split nested, mean of 5 repeats (part10 re-run)). The OOF csv is the SINGLE-SPLIT nested run. master_lookup carries the mean of 5 repeats, so these are different estimators of the same quantity, not a data error.

  Status now: released here. Files: lookup/master_lookup.csv, lookup/SUPERSEDED.csv, scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/README.md. The authors' rule: the Pitt encoder probe 0.7706 is the mean of five per-repeat AUCs, and the released fold files reproduce it exactly. 0.7917, the AUC of the averaged probabilities, is not used.

- 19. HIGH [PART16/M7] adresso encoder probe: recomputed from omni_adresso_enc_nested_oof.csv = 0.8899, master_lookup.csv = 0.8752 (estimator there: five split nested, mean of 5 repeats). The OOF csv is the SINGLE-SPLIT nested run. master_lookup carries the mean of 5 repeats, so these are different estimators of the same quantity, not a data error.

  Status now: released here. Files: lookup/master_lookup.csv, folds/adresso_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/adresso_groupkfold5_pod_seed4_UNVERIFIED.csv, folds/README.md. The paper value is the mean of five per-repeat AUCs. The single-split file is a different estimator and is not used.

- 20. HIGH [PART16/M7] adress2020 encoder probe: recomputed from omni_adress2020_enc_nested_oof.csv = 0.7878, master_lookup.csv = 0.8043 (estimator there: five split nested, mean of 5 repeats). The OOF csv is the SINGLE-SPLIT nested run. master_lookup carries the mean of 5 repeats, so these are different estimators of the same quantity, not a data error.

  Status now: released here. Files: lookup/master_lookup.csv, folds/adress2020_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/adress2020_groupkfold5_pod_seed4_UNVERIFIED.csv, folds/README.md. The paper value is the mean of five per-repeat AUCs. The single-split file is a different estimator and is not used.

- 21. HIGH [PART16/M7] pcgita encoder probe: recomputed from omni_pcgita_enc_nested_oof.csv = 0.9089, master_lookup.csv = 0.8941 (estimator there: five split nested, mean of 5 repeats). The OOF csv is the SINGLE-SPLIT nested run. master_lookup carries the mean of 5 repeats, so these are different estimators of the same quantity, not a data error.

  Status now: released here. Files: lookup/master_lookup.csv, folds/pcgita_groupkfold5_pod_seed0.csv to folds/pcgita_groupkfold5_pod_seed4.csv, folds/README.md. The paper value is the mean of five per-repeat AUCs. The single-split file is a different estimator and is not used.

- 22. HIGH [PART16/M7] neurovoz encoder probe: recomputed from omni_neurovoz_enc_nested_oof.csv = 0.9160, master_lookup.csv = 0.9213 (estimator there: five split nested, mean of 5 repeats). The OOF csv is the SINGLE-SPLIT nested run. master_lookup carries the mean of 5 repeats, so these are different estimators of the same quantity, not a data error.

  Status now: released here. Files: lookup/master_lookup.csv, folds/neurovoz_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/neurovoz_groupkfold5_pod_seed4_UNVERIFIED.csv, folds/README.md. The paper value is the mean of five per-repeat AUCs. The single-split file is a different estimator and is not used.

- 23. HIGH [PART16/M7] kcl encoder probe: recomputed from omni_kcl_enc_nested_oof.csv = 0.7768, master_lookup.csv = 0.7506 (estimator there: five split nested, mean of 5 repeats). The OOF csv is the SINGLE-SPLIT nested run. master_lookup carries the mean of 5 repeats, so these are different estimators of the same quantity, not a data error.

  Status now: released here. Files: lookup/master_lookup.csv, folds/kcl_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/kcl_groupkfold5_pod_seed4_UNVERIFIED.csv, folds/README.md. The paper value is the mean of five per-repeat AUCs. The single-split file is a different estimator and is not used.

- 24. HIGH [PART16/M7] edaic encoder probe: recomputed from omni_edaic_enc_nested_oof.csv = 0.6324, master_lookup.csv = 0.5966 (estimator there: five split nested, mean of 5 repeats). The OOF csv is the SINGLE-SPLIT nested run. master_lookup carries the mean of 5 repeats, so these are different estimators of the same quantity, not a data error.

  Status now: released here. Files: lookup/master_lookup.csv, folds/edaic_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/edaic_groupkfold5_pod_seed4_UNVERIFIED.csv, folds/README.md. The paper value is the mean of five per-repeat AUCs. The single-split file is a different estimator and is not used. The first 30 s E-DAIC probe is no longer in the paper (item 14).

- 25. [HIGH -> resolved to LOW] M10 (part16) ADDENDUM, cause found. The KCL transcript gap 0.7634 vs 0.7619 is exactly one tied score pair. In o25_kcl_text.csv the value p_yes = 0.02596096 occurs twice, once on a label=1 clip and once on a label=0 clip, giving 16*21 = 336 pairs with 256 strict wins and 1 tie. 256/336 = 0.761905 (published: tie counted as a LOSS, strict > comparison). (256 + 0.5*1)/336 = 0.763393 (rank formula, tie gets half credit). sklearn roc_auc_score returns 0.763393, agreeing with the rank formula to 6 decimals. Per PART16 rule 5 the rank-formula value 0.7634 is the one to carry. The published 0.7619 is not wrong data, it is a no-tie-correction AUC. Magnitude 0.0015, far inside the interval.

  Status now: fixed in the paper. Table 1 row, verbatim: "MDVR-KCL    &   37 (37)  & 0.71 & 0.76 & 0.70 \\". Table 1 prints 0.76, the value under both tie rules. Evidence in this repository: scores/part16/M/M10_kcl_transcript.csv.

- 26. [LOW] PART16 M4: agreement arm has 311 unique seg_uids across 483 pair rows (agreement segments are reused across pairs of the same speaker). All 483 rows kept, so a reused clip contributes its score once per pair. Conflict arm has 483 unique seg_uids.

  Status now: fixed in the paper. The final tex prints the 311 distinct segment values 0.96, 0.96, 0.97, 0.63, 0.84 and 0.89 in Table 2. Quote (caption): "Depression, E-DAIC \cite{daic}: 483 conflict segments paired with agreement segments of the same speakers (311 distinct), 138 speakers." Files: scores/part17/T1_table2_distinct.csv, scores/part17/T1_perclip_distinct_agreement.csv, scores/part17/T1_perclip_all966.csv.

- 27. [MEDIUM] PART16 M4: AudioFlamingo2 / conflict: Yes rate 0.9710 is near-constant (p_yes sd=0.0532). The model answers the same word on almost every clip, so its arm contrast carries little information.

  Status now: released here. Files: reports/PART14_yes_rates.csv. The file gives the Audio Flamingo 2 Yes rate on the conflict arm.

- 28. [MEDIUM] PART16 M4: AudioFlamingo2 / agreement: Yes rate 0.9834 is near-constant (p_yes sd=0.0437). The model answers the same word on almost every clip, so its arm contrast carries little information.

  Status now: released here. Files: reports/PART14_yes_rates.csv. The file gives the Audio Flamingo 2 Yes rate on the agreement arm.

## 29. [HIGH] Qwen3-Omni Pitt zero-shot: 0.7673 (2026-09-22 rerun) vs 0.7619 (shipped)
- Shipped: `release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv`, quoted as 0.7619 in `master_lookup.csv:179` and `table1_all_models.csv:16`.
- 2026-09-22 rerun on a fresh H200 with transformers 5.17.0, same manifest `mf_pitt468.csv`, WINDOW_S=30, same `extract_probe_layers.py`: n=468, AUC 0.7673, median answer mass 1.000.
- Same n, same clips. Cause not yet established. POD3 is diffing the two per-clip files clip by clip.
- Affects a published Table 1 cell. Not resolved.

Status now: fixed in the paper. Quote: "On Qwen3-Omni the encoder probe reaches 0.90 on PC-GITA and 0.81 on Pitt against zero-shot answers of 0.60 and 0.76, so its Alzheimer's gap is only 0.05." The printed 0.76 is the shipped file behind Table 2, 0.7619. The rerun 0.7673 is listed as not used. Evidence: scores/part20/POD4b/P20_4b_q3o_orig_samepod_overall.perclip.csv in this repository.

## 30. [MEDIUM] Qwen3-Omni Pitt projector probe is a single repeat
- `q3o_pitt_nested_repeats.json` gives proj mean 0.7820 over 1 repeat / 1 candidate point, while enc, llm and ans are 5 repeats over 32/49/49 points.
- The projector cell is therefore not comparable to the other three in the same table. Same defect as the earlier `overnight2/final/q3o_pitt_nested_repeats.json`, where proj=0.7619 coincided exactly with the zero-shot AUC.

Status now: released here. Files: lookup/master_lookup.csv. The lookup labels the Qwen3-Omni Pitt projector probe as a single split.

- 31. LOW [PART16/M7] pitt supplement: recomputing the pitt encoder probe from the 5-repeat OOF npz release/overnight2/part10/pitt_enc_nested5_oof.npz gives per-repeat AUCs [0.7598, 0.7725, 0.7459, 0.7903, 0.7843], mean 0.7706, which reproduces master_lookup.csv 0.7706 EXACTLY. This confirms the earlier HIGH pitt entry is an estimator difference (single-split OOF csv 0.7969 vs 5-repeat mean 0.7706), not a data error. Only pitt has a saved per-repeat per-clip OOF. The other six datasets have only the single-split csv.

  Status now: released here. Files: lookup/master_lookup.csv, lookup/SUPERSEDED.csv, scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/README.md. The authors' rule: the Pitt encoder probe 0.7706 is the mean of five per-repeat AUCs, and the released fold files reproduce it exactly. 0.7917, the AUC of the averaged probabilities, is not used.

- 32. [HIGH] PART16 M2 (2026-09-22 23:5x): the Part 14 conflict/agreement selection is NOT stable under a
  second sentiment scorer. Swapping cardiffnlp/twitter-roberta-base-sentiment-latest for
  distilbert-base-uncased-finetuned-sst-2-english on the identical 6,214 segment texts, with every other
  Part 14 rule unchanged (speech_s>=5, nw>=25, span>=8, conflict = (y==1 & sev>=15 & clearly positive) |
  (y==0 & sev<=4 & clearly negative), same 0.70 cut, same closest-span same-speaker matching), gives
  1,334 pairs / 2,668 clips / 162 speakers instead of 483 pairs / 966 clips / 138 speakers.
  Cohen kappa on the per-segment rule arm over the 5,523 eligible segments is 0.3438 [0.3224, 0.3665],
  raw agreement 0.5524 [0.5286, 0.5754]. On the selected arm, kappa 0.3362 [0.3008, 0.3717].
  Cause: SST-2 has no neutral head and is far more confident, so only 231 of 6,214 segments fall in its
  induced 0.30-0.70 neutral band versus 3,461 for cardiffnlp. This is a confidence-scale effect as much
  as a content disagreement: at a count-matched DistilBERT threshold of 0.995 the rule-arm kappa rises to
  0.4978 [0.4728, 0.5234], still only moderate.
  MITIGATING: the specific 483 conflict segments the paper uses largely survive the scorer swap. 415 of
  483 (85.9%) are also flagged conflict by DistilBERT at 0.70, 29 land in its agreement pool and 39 in
  neither. So the clips being scored are reproducible. The *size* of the eligible conflict population is
  not. Any sentence claiming the conflict set is scorer-independent must be softened to this.
  Files: release/edaic_rerun/part16/M2_REPORT.txt,
  release/edaic_rerun/part16/M2_distilbert_sentiment.csv

  Status now: released here. Files: scores/part16/M/M2_distilbert_sentiment.csv, scores/part16/M/M2_distilbert_sentiment.json, scores/part20/POD3b/sst2_pairs_o25_audio.csv, scores/part20/POD3b/sst2_pairs_o25_audio.json, scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv, release/scores/part20/rows/POD3b.tsv (authors' machine path). The paper names its scorer (a RoBERTa classifier trained on TweetEval, threshold 0.70) and reports the second scorer: "A second sentiment scorer (DistilBERT fine-tuned on SST-2) selects 1,334 pairs, on which the conflict arm scores 0.46 [0.42, 0.51], at chance rather than below it, and stays 0.39 below the agreement arm." "The inversion holds on the 420 segments both scorers call conflicting (0.20) and not on the 855 the first scorer calls neutral (0.62)." The DistilBERT comparison is item 146.

- 33. [LOW] PART16 M2: the Part 14 builder scripts were not under
  release/edaic_rerun/scripts_part14/ (that folder holds only
  part14_boot.py). The actual selection and sentiment builders were recovered from an earlier
  temporary working folder on the authors' machine:
  {part14_select.py,part14_sentiment.py}. Such folders are
  temporary and will be cleared. These two files should be copied into scripts_part14/
  before they are lost. Re-running part14_select.py's rule on the saved RoBERTa columns reproduces
  part14_manifest_new.csv EXACTLY (966 rows, 483 pairs, 138 speakers, all 794 (arm, seg_uid) keys), so
  the rule restated in the M2 sidecar is verified, not assumed.

  Status now: released here. Files: scripts/part14/part14_select.py, scripts/part14/part14_sentiment.py. These are the two recovered builders.

## 34. POD1 / Part16 1a: Part 14 clip reuse across pairs (MEDIUM)
2026-09-23. part14_manifest_new.csv has 966 rows but only 794 UNIQUE (set, speaker_id, seg_uid)
keys: 172 rows are repeat appearances of a clip that participates in more than one matched pair.
Consequence: (seg_uid) and even (set,speaker,seg_uid) are NOT valid join keys for these files.
Verified that part14_manifest_new.csv, part14_mf_local.csv, part14/p14_o25_zeroshot_scores.csv,
part14/p14_q2a_zeroshot_scores.csv, part16/pod_sync/p14_o25_text.csv and p14_q2a_text.csv are all
966 rows in IDENTICAL row order with matching label and speaker columns, so POD1 joins them
POSITIONALLY. All 1a/1f numbers below use that positional join.
Also note per-clip scores are duplicated for reused clips, so pooled AUCs weight those clips twice.
This is a property of the existing Part 14 design, not something introduced here.

Status now: fixed in the paper. The manifest has 966 rows and 794 distinct segments. The final tex prints the 311 distinct segment values 0.96, 0.96, 0.97, 0.63, 0.84 and 0.89 in Table 2. Quote (caption): "Depression, E-DAIC \cite{daic}: 483 conflict segments paired with agreement segments of the same speakers (311 distinct), 138 speakers." Quote (method): "This yields 488 conflict segments, of which 483 have an agreement segment from the same speaker, giving 483 pairs from 138 speakers." Files: manifests/part14_manifest_new.csv, scores/part17/T1_table2_distinct.csv, scores/part17/T1_perclip_distinct_agreement.csv, scores/part17/T1_perclip_all966.csv.

## 35. [MEDIUM] PART16 M1: two different Qwen2.5-Omni transcript-only runs exist on the same Pitt 468 segments
- Run A (used): `gdrive/omni_pitt_text.csv`, mtime 2026-09-15 15:14, 468 rows, columns `id,speaker,label,set,p_yes,mass`. Recomputed rank-formula AUC all=0.6975, conflict=0.4550, agreement=0.7987, paired conflict-minus-agreement=-0.3437.
- Run B: `release/overnight2/text_new/o25_pitt_text.csv` (and a byte-identical copy at `podD2_final/out/o25_pitt_text.csv`, md5 c5b37ecc5c7de852caa87ec7a5b16e21), mtime 2026-09-20 02:38, 468 rows, columns `speaker_id,id,label,p_yes,answer_mass,prompt`. Recomputed AUC all=0.6629, conflict=0.5560, agreement=0.7078.
- Both carry the same model id `Qwen/Qwen2.5-Omni-7B` and the same prompt string ("Based only on this transcript of what the person said, does this speaker show signs of dementia? Answer with one word, Yes or No."). Same 468 clips, same labels, same speakers.
- The arm contrast is what moves: conflict 0.4550 vs 0.5560 (0.1010 apart) and agreement 0.7987 vs 0.7078 (0.0909 apart). The paper's Pitt transcript arm gap is built on Run A.
- Run A is the one the Sep 18 bootstrap tables point at. `overnight/bootstrap_intervals.csv` rows `transcript_auc`, `pitt_conflict_auc`, `pitt_agreement_auc`, `pitt_conflict_minus_agreement` for Qwen2.5-Omni / pitt all name Run A's path, as does `overnight/bootstrap_conflict_detail.csv` row `Qwen2.5-Omni,transcript`, and `overnight/grid_transcript.csv` row `Qwen2.5-Omni,Pitt,AD,full` (0.6975). Run A's own log `gdrive/logs/omni_pitt_text.log` records "RESULT omni text pitt_text.csv n=468 AUC=0.697 agreement=0.799 conflict=0.455".
- M1 therefore reports Run A as the headline and carries Run B as explicit `M1_ALT_text_*` rows. Cause of the Run A / Run B gap is not established. Run A was a local MPS run, Run B a pod run through `/workspace/text_score.py`. Decoding precision and the transcript source file (`/workspace/text/pitt_text.csv` for B) were not verified to be identical.
- Also on disk and NOT used: `gdrive/rebuild/omni_pitt_text_v{1s,3s,7s}_strip.csv` (2026-09-19, 468 rows, prompt-variant rebuilds, AUC 0.6377 / 0.6442 per `rebuild/summary_v1s-v3s-v7s_strip.json`, all marked `pass: false` against Run A), and `release/edaic_rerun/pitt_all/o25_pitt_all_text.csv` (708 rows, a different clip set).

Status now: released here. Files: scores/part16/M/M1_pitt_text_arms.csv, scores/part17/T6_pitt_text_sep20_arms.csv, scores/part17/T6_pitt_text_sep20_arms.json, release/edaic_rerun/part17/rows/T6.tsv (authors' machine path), lookup/master_lookup.csv. Table 1 prints the Sep 20 run (lookup value 0.6629). The per-clip file carries both runs. The cause of the gap between runs cannot be settled from disk (item 132).

## 36. [LOW] PART16 M1: all four Sep 18 published Pitt transcript values reproduce exactly
- Recomputed from Run A per-clip scores with the rank formula against `overnight/bootstrap_intervals.csv`: transcript_auc 0.6975 = 0.6975, pitt_conflict_auc 0.4550 = 0.4550, pitt_agreement_auc 0.7987 = 0.7987, pitt_conflict_minus_agreement -0.3437 = -0.3437. Audio arm AUCs also reproduce `overnight/grid_pitt_arms.csv` exactly (conflict 0.5322, agreement 0.7054). No action needed, logged so the check is on the record.

Status now: released here. Files: scores/part17/T6_pitt_text_sep20_arms.csv, release/edaic_rerun/part17/rows/T6.tsv (authors' machine path). The Sep 15 per-clip scores sit beside the Sep 20 ones in the released file.

- 37. [MEDIUM] PART16 M5 (2026-09-23 00:00) No saved fold assignment file exists for Pitt. no saved fold file found under release or gdrive (searched *fold*.csv/json/npz and the fold column of every *_oof.csv in probe2/ and part15/, none carry one). Folds recomputed deterministically with GroupKFold(n_splits=5) grouped by speaker on the clips in npz order, which is the same scheme p15/direction_q3o.py and part15/H_readout_arms.py used, so the three backbones stay comparable. Affects M5_readout_direction_q2a.csv (auc_probe, auc_probe_without_d and both arm variants).

  Status now: released here. Files: folds/pitt_groupkfold5_mac.csv, folds/README.md. This split is confirmed clip by clip against the Qwen2-Audio Pitt probes.

## M8 conflict-cell sources (part16), 2026-09-23

(Group heading with no finding of its own. Items 38 to 42 below belong to it.)

- 38. [HIGH] M8 (part16) Audio Flamingo 2 has two Pitt-468 score files that disagree badly: `gdrive/af2_results/af2_pitt468.csv` (conflict AUC 0.6310 [0.5336,0.7260], all-468 0.5724) vs `gdrive/af2_pitt_audio.csv` (conflict AUC 0.6691 [0.5650,0.7654], all-468 0.5273). Per-clip p_yes correlation is only 0.6137, max abs diff 0.4324 on the same 468 clips. master_lookup.csv cites af2_pitt468.csv. BOTH rows are reported in M8_conflict_cells.csv. Neither was silently dropped.

  Status now: released here. Files: lookup/SUPERSEDED.csv, lookup/master_lookup.csv, release/scores/part20/POD4c/work/af2_val10.csv (authors' machine path). The authors' rule: af2_results/af2_pitt468.csv is the canonical AF2 Pitt file, and a pod rescore of 10 windows matches it bit for bit. That file heard the full 30 to 56 s windows. The final Table 2 prints the values of the same windows cut to 30 s, 0.5835 and 0.5337, row verbatim: "Audio Flamingo 2 \cite{af2}     & 0.53 & 0.63 & 0.58 & 0.53 \\" (release/scores/part20/POD4c/af2_orig30_conflict.csv and af2_orig30_agreement.csv, authors' machine paths).

- 39. [MEDIUM] M8 (part16) Kimi-Audio Pitt-468 exists in four copies with two different score sets. `overnight2/part3` = `overnight2/kimi` (conflict 0.7197) but `overnight/kimi/kimi_pitt_zeroshot_scores.csv` differs (conflict 0.7078, per-clip corr 0.9299, max abs diff 0.3630), and `gdrive/kimi_pitt_audio.csv` gives conflict 0.7166. master_lookup.csv cites the part3 copy, which M8 used as primary. The overnight copy is reported as an `alt_overnight_copy` row. The verdict (above 0.5) is the same under every copy.

  Status now: released here. Files: lookup/SUPERSEDED.csv, lookup/master_lookup.csv. The authors' rule: the overnight2/part3 file is the canonical Kimi-Audio Pitt file. Table 2 prints 0.72 and 0.72 from it.

- 40. [LOW] M8 (part16) Audio Flamingo 3 Pitt-468 exists in three copies that differ slightly (`overnight2/part3/af3_pitt.csv` conflict 0.6169, `overnight2/part10/af3_pitt.csv`, `gdrive/af3_pitt_audio.csv` conflict 0.6031, max abs per-clip diff 0.0624). master_lookup.csv cites the part3 copy, used here. Verdict (above 0.5) is unchanged across copies.

  Status now: fixed in the paper. The final Table 2 prints the part10 copy, row verbatim: "Audio Flamingo 3 \cite{af3}     & 0.42 & 0.84 & 0.61 & 0.59 \\" The part3 copy (0.6169 and 0.5822) is not used. A pod rescore reproduces part10 exactly (release/scores/part20/POD4c/work/af3_val10.csv (authors' machine path)), and an input-length check shows that the part10 run cut every Pitt window to its first 30 s before the processor, 480,000 samples and 750 audio tokens per clip, as the text says (paper1_submission_23sep/final_checks_2325Z/af3/AF3_INPUT_LENGTH.md, authors' machine path). lookup/master_lookup.csv still cites part3 for these two cells.

- 41. [LOW] M8 (part16) Two older E-DAIC conflict score sets exist under `release/edaic_conflict_new/` with 390 rows per model (the pre-Part-14 build). M8 Set A uses the Part 14 966-clip files under `edaic_rerun/part14/` exclusively, as instructed. The 390-row files are a different clip set, not an alternative scoring of the same clips.

  Status now: released here. Files: manifests/edaic_paradox_manifest_390.csv, reports/PART14_table2_old_vs_new.csv. The paper's conflict set is the 483-pair Part 14 set.

- 42. [HIGH] M8 (part16) SUBSTANTIVE: only 4 of the 12 primary conflict cells have a 95% speaker-bootstrap interval entirely below 0.5 (Qwen2.5-Omni, Qwen2-Audio, Qwen3-Omni, Audio Flamingo 3 on E-DAIC Part 14). All six Pitt conflict cells fail the below-chance bar. Three of them (AF2, AF3, Kimi) are entirely ABOVE 0.5. Any paper sentence saying models are below chance on the conflict arm has to be scoped to E-DAIC and to four of the six models.

  Status now: fixed in the paper. The abstract now names depression: "Where the words contradict a depression diagnosis, four of six models fall below chance." The body sentence is "On the depression segments four of the six models fall below chance when the words point away from the label, while every model except Audio Flamingo 2 scores above 0.80 when the words and the label agree." The Pitt sentence is "On the Pitt dataset the three Qwen models drop by 0.17 to 0.29 from the agreement to the conflict arm but none falls significantly below chance, Kimi-Audio shows no drop, and the two Audio Flamingo models score 0.53 to 0.61 on both arms (Table~\ref{tab:conflict})." Related files: scores/part16/M/M8_conflict_cells.csv.

- 43. [LOW] PART16 M5 (2026-09-23) Mass weighting is effectively degenerate for Qwen2-Audio: 0.999916 of the mean Yes softmax mass sits on token id 9454 ('Yes') and 0.999970 of the No mass on id 2753 ('No'). cos(d_mass_weighted, normalise(W[9454]-W[2753])) = 1.000000 to 6 dp, and the conflict-arm, agreement-arm and all-data directions are identical to 10 dp (cos = 1.0000000000). So the "mass-weighted" refinement of the readout direction is a no-op on this backbone, and an arm-specific readout direction cannot exist. Reported as-is. The direction is still the correct one, it is just not distinguishable from the plain top-token difference. File: part16/M5_readout_direction_q2a.json.

  Status now: released here. Files: scores/part16/M/M5_readout_direction_q2a.csv, scores/part17/T3_q2a_direction.csv, scores/part17/T3_q2a_direction_summary.csv, release/edaic_rerun/part17/rows/T3.tsv (authors' machine path). The final tex reports the Qwen2-Audio direction test without a cosine: "On Qwen2-Audio and Qwen3-Omni the readout direction gives 0.62 and 0.77 and the probe 0.82 and 0.83, and 0.81 and 0.83 with the direction removed."

- 44. [LOW] PART16 M5 (2026-09-23) auc along the readout direction (0.617358) and auc of the scorer's own saved p_yes column in pitt_states.npz (0.617338) differ in the 5th decimal. Spearman between the two scores is 0.9999575, Kendall tau 0.9961, and 243 of 468 clips swap rank by a small amount. Cause: the saved p_yes was produced by probe2/extract2.py under fp16 inference with a wider Yes/No id list (["Yes"," Yes","yes"," yes","YES"], taking t[0] of any encoding) and renormalised over the Yes/No logits only, whereas M5 uses float64, the direction_q3o.py id rule (single-token encodings only) and the full-vocab softmax. Not a contradiction, and both round to 0.6174 / 0.6173. The 0.6174 in README_probe2.txt Table 1 (zero-shot audio, Pitt) matches the direction value. File: part16/M5_readout_direction_q2a.csv.

  Status now: released here. Files: scores/part16/M/M5_readout_direction_q2a.csv, scores/part17/T3_q2a_direction.csv, release/edaic_rerun/part17/rows/T3.tsv (authors' machine path).

- 45. [LOW] PART16 M5 (2026-09-23) Apple Accelerate BLAS raises spurious divide-by-zero / overflow / invalid FPU status flags on the 468x4096 @ 4096xV matmul. Cross-checked against np.einsum: max abs difference 8.88e-16, every entry finite, max |logit| 16.42. The identical flags fire on pure random data of the same shape. Suppressed with np.seterr, no effect on any number. Same verdict the original scripts/part10_readout_direction_v2.py reached.

  Status now: released here. Files: scripts/core/part10_readout_direction_v2.py. The script's header note records the same Accelerate flags and the np.seterr call.

## 46. [HIGH] M7 first pass used a SUPERSEDED Pitt probe file
- `edaic_rerun/part16/M7_gap_per_dataset.py` read `omni_final/omni_pitt_enc_nested_oof.csv` (single split) -> 0.7969.
- `master/master_lookup.csv` marks that value superseded: "Qwen2.5-Omni pitt encoder probe 0.7706 ... [SUPERSEDES omni_final/omni_pitt_nested_repeats.json 0.7761 and single-split 0.7969]", source `overnight2/part10/pitt_enc_nested5_oof.npz`.
- Corrected in `part16/M7_pitt_fix.py`: per-repeat AUCs [0.7598, 0.7725, 0.7459, 0.7903, 0.7843], mean 0.7706 [0.7157, 0.8220], reproducing the published value exactly.
- Corrected paired gap: +0.1128 [0.0455, 0.1768] (was reported 0.1391). Still excludes zero.
- NOTE two aggregations coexist and differ: mean of the 5 per-repeat AUCs = 0.7706 (what the paper cites). AUC of the averaged OOF probabilities = 0.7917. The paper must state which it uses.

Status now: released here. Files: lookup/master_lookup.csv, lookup/SUPERSEDED.csv, scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/README.md. The authors' rule: the Pitt encoder probe 0.7706 is the mean of five per-repeat AUCs, and the released fold files reproduce it exactly. 0.7917, the AUC of the averaged probabilities, is not used.

## 47. [HIGH] Kimi-Audio has two disagreeing Pitt files, like AF2
- `overnight2/part3/kimi_pitt_zeroshot_scores.csv`: conflict 0.7197, agreement 0.7227.
- `overnight/kimi/kimi_pitt*`: conflict 0.7078, agreement 0.7014.
- Same 468 clips. A canonical file must be chosen before Kimi's Pitt row is cited.

Status now: released here. Files: lookup/SUPERSEDED.csv, lookup/master_lookup.csv. The authors' rule is the same as item 39.

## 48. [HIGH] On Pitt, no conflict cell is entirely below chance
- M8, six models, Pitt conflict arm (n=146, 100 speakers), 2000-draw speaker intervals: every one of the six intervals straddles or sits above 0.5.
  Qwen2.5-Omni 0.5322 [0.4216, 0.6447]. Qwen2-Audio 0.4132 [0.3005, 0.5423]. Qwen3-Omni 0.6066 [0.4956, 0.7123].
  AF2 0.6310 [0.5336, 0.7260]. AF3 0.6169 [0.5188, 0.7209]. Kimi 0.7197 [0.6201, 0.8147].
- On the E-DAIC Part 14 set (n=483, 138 speakers) four of six ARE entirely below 0.5: Omni 0.2218 [0.1829, 0.2632], Qwen2-Audio 0.2280 [0.1816, 0.2782], Qwen3-Omni 0.3008 [0.2454, 0.3641], AF3 0.4235 [0.3606, 0.4880]. AF2 0.5286 [0.4512, 0.6015] and Kimi 0.5258 [0.4501, 0.5977] straddle.
- Any sentence claiming below-chance behaviour on Pitt at n=146 is not supported by the interval.

Status now: fixed in the paper. Quote: "On the Pitt dataset the three Qwen models drop by 0.17 to 0.29 from the agreement to the conflict arm but none falls significantly below chance, Kimi-Audio shows no drop, and the two Audio Flamingo models score 0.53 to 0.61 on both arms (Table~\ref{tab:conflict})." The Pitt sentence makes no below-chance claim.

- 49. [HIGH] PART16 M11: two DIFFERENT ASR transcript sets exist on disk for the same 37 MDVR-KCL scored clips, and they agree on zero rows. `release/part4_text/asr/asr_kcl.csv` (the pod copy, clip_path `/workspace/data/kcl_read30b/`, the one the Omni text condition consumed as `/workspace/text/kcl_text.csv`) averages 56.1 words per clip. `gdrive/kcl_read30b_text.csv` averages 84.9. Worst cases, six of 37 speakers: 1 word vs 39, 16 vs 164, 24 vs 315, 8 vs 88, 8 vs 85, 9 vs 78 [speaker ids withheld here]. One local-copy clip holds 315 words in 30 s (~630 wpm), which is a Whisper repetition loop, so neither copy is clean. Whichever is used changes the transcript condition's input. Not resolved.

  Status now: fixed in the paper. The final tex names the transcript set: "Pitt and ADReSS-2020 \cite{adress2020} come with manual CHAT transcripts and E-DAIC \cite{daic} with the challenge's automatic transcripts, used with the annotation codes stripped, and ADReSSo \cite{adresso} and the three Parkinson's sets are transcribed with whisper-large-v3 \cite{whisper}." The MDVR-KCL transcript cell in Table 1 comes from the pod whisper-large-v3 copy, and transcripts cannot be released. Related files: scores/part16/M/M11_kcl_wer.csv.

- 50. [LOW] PART16 M11: MDVR-KCL text-condition AUC recomputed from the per-clip file with the tie-averaged rank formula is 0.7634, against 0.7619 stored in `release/podE_final/out/o25_kcl_text.json`. Difference is exactly half a rank unit out of n1*n0=336, i.e. one tied pair handled differently. Same clips, same n=37.

  Status now: fixed in the paper. Table 1 row, verbatim: "MDVR-KCL    &   37 (37)  & 0.71 & 0.76 & 0.70 \\". Table 1 prints 0.76, the value under both tie rules. Evidence in this repository: scores/part16/M/M10_kcl_transcript.csv.

## PART 16 / M9 answer mass (2026-09-23T00:02:38)

(Group heading with no finding of its own. Items 51 to 56 below belong to it.)

- 51. LOW [M9 answer mass] Qwen2.5-Omni / edaic_full / projector fine tune: answer-probability file `release/edaic_rerun/variants/sft_full_oof.csv` carries no mass column. Cell reported as 'no mass recorded'.

  Status now: released here. Files: scores/edaic_windows/sft_full_oof.csv. The file is released as run, with no mass column.

- 52. LOW [M9 answer mass] Qwen2.5-Omni / edaic_mid30 / projector fine tune: answer-probability file `release/edaic_rerun/variants/sft_mid30_oof.csv` carries no mass column. Cell reported as 'no mass recorded'.

  Status now: released here. Files: scores/edaic_windows/sft_mid30_oof.csv. The file is released as run, with no mass column.

- 53. LOW [M9 answer mass] Qwen2.5-Omni / edaic_mid300 / projector fine tune: answer-probability file `release/edaic_rerun/variants/sft_mid300_oof.csv` carries no mass column. Cell reported as 'no mass recorded'.

  Status now: released here. Files: scores/edaic_windows/sft_mid300_oof.csv. The file is released as run, with no mass column.

- 54. LOW [M9 answer mass] Qwen2.5-Omni / pitt_all / projector fine tune: answer-probability file `release/edaic_rerun/pitt_all/sft_pitt_all_oof.csv` carries no mass column. Cell reported as 'no mass recorded'.

  Status now: released here. Files: scores/pitt_all/sft_pitt_all_oof.csv. The file is released as run, with no mass column.

- 55. MEDIUM [M9 answer mass] sweep: `gdrive/af3_adress2020_rerun.csv` has a mass column (`answer_mass`) but no finite values.

  Status now: released here. Files: lookup/master_lookup.csv. The AF3 ADReSS-2020 row cites overnight2/af3_new/af3_adress2020.json, so the rerun file with empty mass is not a paper source.

- 56. MEDIUM [M9 answer mass] sweep: `gdrive/af3_pitt468_rerun.csv` has a mass column (`answer_mass`) but no finite values.

  Status now: released here. Files: lookup/master_lookup.csv. The AF3 Pitt row cites overnight2/part3/af3_pitt.json, so the rerun file with empty mass is not a paper source. The AF3 Pitt choice itself is item 40.

## PART16 / POD2 (Pitt LoRA baseline, Qwen2.5-Omni), 2026-09-23

(Group heading with no finding of its own. Items 57 to 63 below belong to it.)

- 57. MEDIUM: no saved fold file found. Searched all of `release` for a fold assignment: no file named `*fold*`, and no `*_oof.csv` anywhere carries a `fold` column (checked headers of every oof csv, the Pitt ones are `path,speaker,label,p_yes,set,prompt` and the nested ones are `clip,speaker,label,p_probe`). Folds were therefore recomputed with `sklearn GroupKFold(n_splits=5)` grouped by speaker, over the source csv's own row order, identical to the call in `sft_projector.py`, which is the script that produced the paper's projector fine-tune. Recorded in the sidecar too.

  Status now: released here. Files: folds/pitt_groupkfold5_pod.csv, folds/README.md. This split is confirmed and agrees with the pod fold file on 468 of 468 clips.

- 58. MEDIUM: two different speaker-string conventions for the same 468 Pitt clips. `p15/mf_pitt468.csv` writes speakers as unpadded ids (the group name plus the bare number). The paper's own output `omni_final/omnisft_pitt_oof.csv` writes zero-padded ids (the group name plus three digits). Same 228 speakers, same partition of clips, but `GroupKFold` breaks ties using the sorted order of the unique group strings, so the two conventions can yield *different* fold assignments. POD2 rebuilt the manifest from `omnisft_pitt_oof.csv` (the paper's file) so the folds match the paper's projector run. Row order, labels and set labels are byte-identical across `mf_pitt468.csv`, `omnisft_pitt_oof.csv` and `abl_pitt_oof.csv` (verified), so only the speaker string differs.

  Status now: released here. Files: folds/README.md, folds/pitt_groupkfold5_pod_unpaddedids_seed0.csv to folds/pitt_groupkfold5_pod_unpaddedids_seed4.csv.

- 59. MEDIUM: the LoRA-on-Pitt baseline POD2 was set to produce already exists. `omni_final/abl_pitt_lora.json` + `abl_pitt_oof.csv`, dated 2026-09-18, is Qwen2.5-Omni, Pitt 468, LoRA rank 8, 3 epochs, lr 1e-4, 5 folds, `trainable_params 5046272`, which is exactly r=8 on q/k/v/o of all 28 thinker layers, i.e. the same config POD2 was told to run fresh. Its recomputed OOF AUC is 0.8202. POD2 ran its own fresh replication anyway and reports both side by side. Neither file was overwritten.

  Status now: released here. Files: lookup/master_lookup.csv, scores/part16/POD2/lora_pitt_oof.csv, scores/part20/POD2/balanced_minus_earlier_lora_pitt_perclip.csv. The lookup records the POD2 LoRA run 0.8250, and the final tex reports it: "Low-rank adaptation (LoRA) \cite{lora} of the language model (rank 8, all attention projections of the 28 layers, 5.0\,M trainable parameters against 4.6\,M for the projector) reaches 0.825 on Pitt, 0.06 [0.02, 0.10] above the projector run, and 0.55 on its conflict segments, and training both together gives 0.77."

- 60. LOW: the prior LoRA/projector per-clip csvs carry no `answer_mass` column, so answer-mass cannot be audited for those runs. POD2's own csv records it per clip.

  Status now: released here. Files: scores/part20/POD2/balanced_lora_pitt_oof.csv, scores/part20/POD1/standard_rerun_ft_pitt_oof.csv. These later runs record answer mass per clip. The older files stay without it.

- 61. NOTE (not a discrepancy): independent recomputation agrees with the paper. POD2 recomputed, with its own rank-formula AUC and its own 2000-draw speaker bootstrap seeded `default_rng(0)`: projector-FT 0.7673 [0.7086, 0.8231] vs the established 0.7673 [0.7086, 0.8231]. Zero-shot 0.6578 [0.6033, 0.7125] vs the established 0.6578 [0.6033, 0.7125]. Match to 4 decimals on point estimate and on both interval ends.

  Status now: released here. Files: release/scores/part20/rows/POD5.tsv (authors' machine path), release/edaic_rerun/part17/rows/T5.tsv (authors' machine path). The rows give the projector fine tune 0.7673 [0.7086, 0.8231] and the zero shot 0.6578 [0.6033, 0.7125] on the paper files.

- 62. LOW [M9 answer mass] Clean result, recorded so it is not re-checked: no cell in the published grid and no file on disk has a median answer mass below 0.15. Across 305 mass-bearing per-clip files and 119,775 clips, zero clips have mass below 0.15 and zero below 0.01. The lowest published cell is Audio Flamingo 3 / pcgita / zero shot answer at 0.8968. The lowest file anywhere is `gdrive/af2_kcl30b_audio.csv` at 0.8662. The lowest single clip anywhere is 0.4425 in `gdrive/qwen2audio_pitt_text.csv`. No published AUC is built on a degenerate answer distribution.

  Status now: released here. Files: scores/part16/M/M9_answer_mass.csv.

- 63. LOW [M9 answer mass] The mass column is not in a fixed position. Three layouts exist: `clip,speaker,label,p_yes,mass,prompt` (Qwen/Kimi score files), `speaker_id,clip_path,label,p_yes,answer_mass[,prompt]` (Audio Flamingo and text files), and `path,speaker,label,set,p_yes,mass` (the af2_*_audio.csv and qwen2audio_*_text.csv family, where column 5 is p_yes, not mass). Anything parsing these by column index will read the wrong column on the third layout.

  Status now: released here. Files: scores/edaic_windows/o25_full_zeroshot_scores.csv, scores/edaic_windows/af3_full.csv. These two files show two of the layouts (mass and answer_mass in different positions). Read columns by name.

## 64. POD4 / Part 16 4b: Pitt manifest `inv_ms_in_window` column is wrong (HIGH)
`DementiaBank/pitt_conflict_manifest.csv` carries
`inv_ms_in_window` = 0 and `inv_share_window` = 0.0 for all 468 rows.
Recomputed independently from the CHAT (.cha) tiers
(DementiaBank/transcripts/{grp}/cookie/{session}.cha):
158 of 468 windows (33.8%) contain timed *INV speech inside the first 30 s the
model actually hears (mean INV share 2.00%, max 28.2%, max 8.45 s).
Cause: build_ad_conflict.py merges consecutive *PAR utterances and cuts audio
from first-utterance start to last-utterance end, so the inter-utterance gaps
(which carry the interviewer) are inside the cut. The zero column was not
written by build_ad_conflict.py and does not reflect the audio.
Consequence: the "no interviewer in the window" claim cannot be sourced to that
column. Severity HIGH because it would otherwise be quoted as a rebuttal.
Additional: only 58.7% of the 30 s window is timed *PAR speech. 39.3% is in
neither tier, and 788 *INV utterances in these transcripts carry no timestamp,
so 33.8% is a LOWER BOUND on interviewer presence.
Evidence: .../part16/POD4/cha_overlap_468.csv

Status now: released here. Files: scores/part16/POD4/cha_overlap_468.csv, manifests/pitt_conflict_manifest_468.csv. cha_overlap_468.csv gives the interviewer time in every window, recomputed from the CHAT tiers. The zero columns inv_ms_in_window and inv_share_window stay in the shipped manifest and must not be cited.

## 65. POD1 / Part16 1d: E-DAIC released binary label != PHQ-8 >= 10 (MEDIUM)
2026-09-23. In part14_segments_sentiment.csv the per-speaker label column `y` does NOT equal the
standard PHQ-8 cutoff. Over the 275 speakers: 189 have y=0 & PHQ8<=9, 66 have y=1 & PHQ8>=10, and
20 speakers have y=0 while PHQ8>=10 ([per-speaker PHQ-8 values and ids withheld here,
clinical scores are not released]).
No speaker has y=1 with PHQ8<=9. So the released binary is STRICTER than PHQ8>=10.
Consequence: part 1d ("rebuild with the dataset cutoff") is not a no-op relabel. It moves those 20
speakers into the depressed arm. 1d therefore defines label := (sev >= 10) and drops the paper's
sev>=15 / sev<=4 severity gates, keeping the eligibility gate, the 0.70 sentiment rule and the
closest-span same-speaker matching unchanged. The paper's Table 2 set is unaffected by this note.

Status now: fixed in the paper. Quote: "We use E-DAIC \cite{daic}, the extended dataset released for AVEC 2019, with 275 participants, where the binary label is the dataset's released label based on the eight-item Patient Health Questionnaire (PHQ-8) \cite{Kroenke2009}, stricter than PHQ-8 $\geq 10$ for 20 participants, rather than a clinical diagnosis."

## 66. [RETRACTED, SEE CORRECTION BELOW] POD4 / Part 16 4b: quoted Pitt encoder-probe figure 0.7706 not found in any file (MEDIUM)
THIS ENTRY IS WRONG. 0.7706 reproduces exactly from `overnight2/part10/pitt_enc_nested5_oof.npz`. Retained only for audit. See the CORRECTION entry at the end of this file.
Brief states the published encoder probe is 0.7706. Recomputed with the rank
formula from the per-clip file
release/omni_final/omni_pitt_enc_nested_oof.csv
gives 0.7969 (n=468, 228 speakers), which matches
omni_pitt_nested.json "auc_nested_oof": 0.7968810538550949.
omni_pitt_nested_repeats.json gives enc mean 0.7761 over 5 repeats
(per_repeat 0.7921/0.7937/0.7652/0.7778/0.7518).
0.7706 appears in neither. Zero-shot 0.6578 DOES reproduce exactly from
omni_pitt_zeroshot_scores.csv. POD4 therefore compares the interviewer-free cut
against both 0.7969 (per-clip OOF, the only arm that can be paired clip-by-clip)
and 0.7761 (5-repeat mean), and states which is which.

Status now: released here. Files: scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv. Retracted: these files reproduce 0.7706 exactly (item 104).

## 67. POD4 / Part 16 4b: no saved fold file for the Pitt 468 (MEDIUM)
Per the instruction to load saved folds rather than recompute: no fold column
exists in any Pitt *_oof.csv (all are clip,speaker,label,p_probe) and no fold
file was found under omni_final/ or edaic_rerun/. POD4 therefore uses
sklearn GroupKFold(n_splits=5) grouped by speaker over the source csv's own
order, identical folds across all arms so the paired contrast is exact.

Status now: released here. Files: folds/pitt_groupkfold5_pod.csv, folds/pitt_groupkfold5_mac.csv, folds/README.md.

- 68. [MEDIUM] PART16 M11: the planned fallback for MDVR-KCL WER is degenerate. Whisper large-v3, greedy, language=en, reproduces `release/part4_text/asr/asr_kcl.csv` EXACTLY on 37/37 clips after normalisation (WER 0.0000, S=D=I=0 on every file). That stored transcript is itself Whisper large-v3 greedy output, so scoring large-v3 against it measures nothing about ASR quality. M11 therefore reports reference-free substitutes (cross-system large-v3 vs medium disagreement, Whisper's own mean token logprob and compression ratio, words recovered per 30 s) and labels them as such.

  Status now: fixed in the paper. Quote (second clause of the transcripts sentence, verbatim): "ADReSSo \cite{adresso} and the three Parkinson's sets are transcribed with whisper-large-v3 \cite{whisper}." The paper reports no word error rate. Evidence in this repository: scores/part16/M/M11_kcl_wer.csv, scores/part16/M/M11_kcl_wer.json.

- 69. [HIGH] PART16 M11: the 30 s window used for MDVR-KCL catches setup chatter instead of the read passage on 6 of 37 speakers. Words recovered in the window on those six speakers: 1 ([one word withheld here]), 8, 9, 10, 10 and 16 (vs a median of 60 across the set) [speaker ids withheld here]. The transcript condition scored those clips anyway, so for those speakers the text condition is classifying an utterance that contains no passage at all. Window starts are measured in `edaic_rerun/scripts_1a1c/kcl_offsets.json`.

  Status now: open. The six windows are inside every MDVR-KCL cell of Table 1, and the final tex has no sentence on them. What decides it: a recut of those six windows. Related files: scripts/audit_1a1c/kcl_offsets.json, scripts/audit_1a1c/kcloff.py, scores/part16/M/M11_kcl_wer.csv.

- 70. [MEDIUM] PART16 M11: MDVR-KCL read speakers were not given one common passage. Classifying each large-v3 transcript by keyword gives north_wind PD=13 control=10, light_scattering PD=2 control=7, neither PD=1 control=4. Passage identity alone (north_wind=1) recomputes to AUC 0.6682 against the diagnosis on the 37 scored clips, against 0.7634 for the full transcript condition. Reading material is therefore a live confound for the KCL text result, and a true WER against a single reference passage is not definable for this set.

  Status now: fixed in the paper. Quotes: "From the Parkinson's disease datasets, we only use material whose words are fixed by the protocol, so the words carry little of the diagnosis, and most of the signal has to come from the voice: PC-GITA \cite{pcgita} (Spanish, ten sentences and one read text per speaker, 1100 clips), NeuroVoz \cite{neurovoz} (Spanish, twelve isolated words, 1270 clips from 107 speakers) and MDVR-KCL \cite{kcl} (English, 37 speakers, most reading one of two passages)." and "On PC-GITA and NeuroVoz the transcript sits near chance, since every speaker produces the same words, while on MDVR-KCL, where speakers read one of two passages, it is above chance." Related files: scores/part16/M/M11_kcl_wer.csv.

- 71. [LOW] PART16 M11: the two group-mean bootstrap implementations differ in the last digits of the interval only. The main script reseeds per cell and resamples that group's speakers alone. The verifier draws the full 37-speaker list once per replicate and subsets. Point estimates, paired differences and Spearman intervals agree exactly. Example: B_xsys_disagree PD 0.3049 [0.1301, 0.4944] vs [0.1157, 0.5115].

  Status now: released here. Files: scores/part16/M/M11_kcl_wer.json, scores/part16/M/M11_kcl_wer.csv. The sidecar records the bootstrap used. The M11 intervals are not in the paper.

## 72. POD4 / Part 16 4b: arm A re-run does not reproduce published Pitt zero-shot bit-for-bit (LOW)
Re-scoring the IDENTICAL original segment wavs with the identical prompt
("Based only on this recording, does this speaker show signs of dementia?
Answer with one word, Yes or No."), WINDOW_S=30, bf16, Qwen2.5-Omni-7B gives
zero-shot AUC 0.6497 vs the published 0.6578 (delta -0.0081), and encoder
nested probe 0.7949 vs the published per-clip 0.7969 (delta -0.0020).
Per-clip p_yes drift is small but nonzero (clip 1: 0.4230 published, 0.408 now).
Cause is almost certainly library drift (this pod runs transformers 5.17.0) and
bf16 kernel nondeterminism, not a data change, the audio files are byte-identical
to the ones that produced the published numbers.
Consequence: the PAIRED interviewer contrast is computed against POD4's OWN
arm A (same code, same session), not against the published number, so the
comparison is internally exact. Both are reported.

Status now: released here. Files: scores/part16/POD4/p16_pitt_orig_zeroshot_scores.csv, scores/part20/POD4/o25_orig_recording_rerun.csv, scores/part20/POD4/o25_orig_recording_rerun.sidecar.json, release/scores/part20/POD4/check/o25_check10_vs_omni_final.csv (authors' machine path), release/scores/part20/rows/POD4.tsv (authors' machine path).

## 73. POD3 / Part 16 3a: Qwen3-Omni Pitt zero-shot 0.7673 (this pod) vs 0.7619 (shipped) (HIGH)
Shipped: `release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv`, AUC 0.7619, run 2026-09-19/20.
This pod: `/workspace/out/q3o_pitt_zeroshot_scores.csv`, AUC 0.7673, run 2026-09-23.
Both recomputed here with the rank formula: shipped 0.761933, pod 0.767319, delta +0.005386.

WHAT IS IDENTICAL between the two runs (checked, not assumed):
- Clip list, clip ORDER and labels: identical, all 468 rows.
- Prompt string: byte-identical.
- Window: 30 s in both sidecars. Every one of the 468 clips is longer than 30 s,
  so both runs truncate every clip. Window handling cannot be the differentiator.
- Yes/No token ids: [7414,9454,9693,9834,14004] / [902,2152,2308,2753,8996] in the
  2026-09-19 extract log AND in today's rerun. Audio token 151675 in both.
  The first-token and single-token id sets are also identical for this tokenizer,
  so the extract_probe_layers.py vs direction_q3o.py id-selection difference is moot here.
- Encoder layers 32, thinker stages 49 in both.
- AUDIO: the 468 wavs on the pod are md5-identical to the Mac masters in
  DementiaBank/segments/ (spot-checked on the first clip and on
  the largest-divergence clip [clip name withheld here]).

WHAT THE DIFFERENCE IS NOT, nondeterminism is RULED OUT:
Scoring all 468 clips TWICE in one process on this pod gives 468/468 bit-identical
p_yes (max abs diff 0.000e+00) and AUC 0.767319 both times
(`/workspace/out/q3o_pitt_rescore.json`, runA/runB csvs). The forward pass is fully
deterministic in a fixed environment, so this is NOT bf16 kernel nondeterminism and
NOT MoE routing jitter within a run. It is a deterministic environment difference.
NOTE: this contradicts the "bf16 kernel nondeterminism" attribution in the POD4 entry
at the end of this file, same-environment reruns here are bit-exact.

SIZE AND SHAPE OF THE DIFFERENCE:
- 412/468 clips (88.0%) differ by more than 1e-3. 87 clips (18.6%) by more than 0.1.
  max |delta p_yes| 0.3808 (one clip: 0.499992 shipped, 0.119204 pod).
- The shift is SYSTEMATIC, not random scatter: pod p_yes is lower on 64.1% of clips,
  mean signed delta -0.0230, mean logit -0.2898 shipped vs -0.4285 pod (mean |d logit| 0.3389),
  while the logit SPREAD is unchanged (sd 1.4111 vs 1.4126).
- p_yes lands on a COARSE DISCRETE LADDER: only 340 (shipped) and 344 (pod) distinct
  values across 468 clips, with exact repeats (0.085102 x5, 0.201815 x4, ...). This is the
  bf16 logit quantisation signature. Only 127 ladder levels are shared between the files.
- Spearman rank correlation of p_yes is 0.9529, which is why a mean |d logit| of 0.34
  moves the AUC by only 0.0054.

CAUSE, stated at the confidence the evidence supports:
A deterministic difference in the numeric environment between the two runs, amplified by
Qwen3-Omni being a sparse MoE (top-k expert routing flips discretely under a tiny numeric
perturbation, so outputs jump between ladder rungs rather than drifting by 1e-6) and by the
coarse bf16 ladder p_yes sits on. Candidate environment variables: GPU model / kernel
selection, attention backend, and library version. `p15/setup.sh` installs
`transformers>=5.0` UNPINNED, so the library version is not controlled across runs.
NOT ESTABLISHED: the 2026-09-19 run left no record on disk of its transformers
version or its GPU model (DRIVER.log and pip.log contain neither), so the specific variable
cannot be named from files that exist. Today's environment is transformers 5.17.0,
torch 2.8.0+cu128, librosa 1.0.0, NVIDIA H200.
Corroboration that it is environmental and not data: the two runs also used DIFFERENT
manifest files (`/workspace/mf_pitt.csv` then, `/workspace/mf_pitt468.csv` now) whose
speaker ids are formatted differently (zero-padded vs unpadded ids), same 228 speakers and
same grouping either way, but proof the two sessions were built by different code.

CONSEQUENCE FOR THE PAPER: the published 0.7619 is not reproducible bit-for-bit outside its
original session. The encoder-over-answer gap survives either number: encoder nested 0.8122
and LM nested 0.8445 against a zero-shot answer of 0.7619-0.7673 on the same 468 clips.
Everything POD3 reports for Pitt is computed from THIS pod's states, so it is internally
consistent. Both zero-shot values are printed side by side.

Status now: fixed in the paper. Quote: "On Qwen3-Omni the encoder probe reaches 0.90 on PC-GITA and 0.81 on Pitt against zero-shot answers of 0.60 and 0.76, so its Alzheimer's gap is only 0.05." The printed 0.76 is the shipped 0.7619, the value of the file behind Table 2. This pod's 0.7673 and the same-pod rescore 0.7638 are not used. Evidence in this repository: scores/part20/POD4b/P20_4b_q3o_orig_samepod_overall.perclip.csv.

## 74. POD3 / Part 16 3a: no saved fold file for the Pitt 468 or PC-GITA 1100 (MEDIUM)
Per the instruction to load saved folds rather than recompute: no fold column exists in any
Pitt or PC-GITA *_oof.csv (all are clip,speaker,label,p_probe) and no fold file was found
under omni_final/, overnight2/ or edaic_rerun/. POD3 therefore uses sklearn
GroupKFold(n_splits=5) grouped by speaker, with the speaker order permuted per repeat by
numpy.random.default_rng(seed) exactly as nested_repeats_all.py does, so the folds match the
shipped nested runs' construction. Recorded in every POD3 sidecar json.

Status now: released here. Files: folds/pitt_groupkfold5_pod_unpaddedids_seed0.csv to folds/pitt_groupkfold5_pod_unpaddedids_seed4.csv, folds/pcgita_groupkfold5_pod_seed0.csv to folds/pcgita_groupkfold5_pod_seed4.csv, folds/README.md. Both sets of splits are confirmed clip by clip.

## 75. POD3 / Part 16 3b: a second job is writing to the same pod, and disputes the readout-direction standardisation (MEDIUM)
While POD3 was running, another session created `/workspace/direction_q3o_fix.py` (07:31 UTC)
and wrote `/workspace/out/readout_direction_q3o_FIXED.json` + `q3o_perp_cols.npz` (07:33 UTC)
into the SAME /workspace/out that POD3 writes to. POD3 did not create those files and did not
act on them. They are recorded here as another job's output, not as an instruction.
Their claim, quoted from that file: direction_q3o.py "standardised the held-out projection with
the TRAINING fold mean/std", and their canonical alternative standardises within the held-out
fold, giving auc_without_d 0.8276 rather than the shipped 0.8305. Their file also records
"arms": {"error": "list index out of range"}, i.e. their by-arm attempt failed.
Why the choice changes the number at all: within one fold the two standardisations are affine
transforms of the same scores, so the within-fold ranking is identical. But the pooled OOF
vector concatenates folds, so a per-fold offset/scale difference changes the CROSS-fold ranking
and therefore the pooled AUC.
POD3's position: the Part 16 instruction for 3b was to use "the same method as direction_q3o.py",
so every POD3 by-arm number (all / conflict / agreement) uses the shipped train-fold
standardisation and reproduces direction_q3o.py's published row exactly (cos 0.0051,
auc_probe 0.8307, auc_d 0.7656, auc_without_d 0.8305, all four match the shipped
readout_direction_q3o.csv). If the canonical definition is adopted, the by-arm test must be
re-run under it. POD3 did not do that, and the conclusion would not change, because in both
arms auc_without_d is within noise of auc_probe (conflict 0.6196 vs 0.6145, agreement
0.9341 vs 0.9349) and the cosine sits inside the random floor.

Status now: released here. Files: lookup/readout_direction.csv, release/edaic_rerun/part16/Q3O_direction_FIXED.json (authors' machine path), lookup/SUPERSEDED.csv. The direction test in the paper uses the held-out-fold standardisation (lookup/readout_direction.csv). SUPERSEDED.csv records the Qwen3-Omni by-arm rule. The final tex adds the Qwen2-Audio and Qwen3-Omni values: "On Qwen2-Audio and Qwen3-Omni the readout direction gives 0.62 and 0.77 and the probe 0.82 and 0.83, and 0.81 and 0.83 with the direction removed." The Qwen3-Omni 0.83 comes from one pod run. A refit of the same probe from the saved states gives 0.7864, not 0.8307 (release/edaic_rerun/part16/verify/POD3_partF_out.json, authors' machine path).

## 76. [HIGH] readout-direction `auc_without_d` was standardised with the TRAINING fold, not the held-out fold
- Canonical definition, `release/omni/readout_direction.json` and `scripts/part10_readout_direction_v2.py:274`:
  `perp_z[te] = (b - b.mean()) / b.std()` where `b = X[te] @ w_perp`, i.e. standardise WITHIN the held-out fold, then pool.
- `direction_q3o.py` in a temporary working folder instead used `(Xe@wp - (Xt@wp).mean())/((Xt@wp).std()+1e-12)`, the TRAINING fold's mean and std.
- Found by the independent verifier on M5 (Qwen2-Audio), where it matters a lot: `auc_without_d` 0.8327 as computed vs 0.8114 canonical. The conflict_restricted cell moves 0.6326 -> 0.6013, whose interval then crosses 0.5.
- Rerun for Qwen3-Omni (`part16/Q3O_direction_FIXED.json`): it matters very little there.
  CANONICAL fold-z 0.8276 | raw pooled 0.8298 | mean of per-fold AUCs 0.8318 | train-z as shipped 0.8305. Per-fold AUCs with d removed [0.8512, 0.7835, 0.7899, 0.8674, 0.8669].
- Conclusion unchanged for Q3O: probe 0.8307 -> 0.8276 after removing the readout direction, a drop of 0.0031.
- ACTION: Qwen2-Audio's direction numbers must be taken from the verifier's canonical recomputation, not the M5 first pass.

Status now: released here. Files: scores/part17/T3_q2a_direction.csv, release/edaic_rerun/part16/Q3O_direction_FIXED.json (authors' machine path), scripts/core/part10_readout_direction_v2.py. T3 gives the Qwen2-Audio value with the held-out-fold standardisation (0.8114), and Q3O_direction_FIXED.json the Qwen3-Omni one.

## 77. [MEDIUM] M5 random-direction floor used an advancing, not reseeded, generator
- M5 reported three different random floors (0.0343 / 0.0350 / 0.0359) for three directions that are identical to 10 decimals. Under a fixed seed-0, 2000-draw protocol the floor must be bit-identical across the three cells. Verifier's reseeded floor: 0.0321 in all three.
- The Qwen3-Omni floor is NOT affected: one cell only, `default_rng(0)` used once. mean|cos| 0.0177 matches theory sqrt(2/(pi*2048)) = 0.01763.

Status now: released here. Files: scores/part17/T3_q2a_direction.csv, release/edaic_rerun/part17/rows/T3.tsv (authors' machine path). The reseeded floors are 0.0123 against the readout direction and 0.0124 against the probe direction (item 120).

## 78. [MEDIUM] Two Qwen2.5-Omni Pitt transcript runs disagree by 0.10 on the arm contrast
- Sep 15 `gdrive/omni_pitt_text.csv`: conflict 0.4550, agreement 0.7987.
- Sep 20 `release/overnight2/text_new/o25_pitt_text.csv`: conflict 0.5560, agreement 0.7078.
- Same 468 clips, same model id, same prompt string. Cause not established. The Sep 18 bootstrap tables point at the first file.

Status now: released here. Files: scores/part16/M/M1_pitt_text_arms.csv, scores/part17/T6_pitt_text_sep20_arms.csv, scores/part17/T6_pitt_text_sep20_arms.json, release/edaic_rerun/part17/rows/T6.tsv (authors' machine path). Table 1 prints the Sep 20 run.

## 79. [MEDIUM] Six master_lookup AUCs do not reproduce from their per-clip files, all transcript-only
- Qwen3-Omni/pcgita 0.5264 stored vs 0.5325 recomputed. Qwen2.5-Omni/pcgita 0.5246 vs 0.5218. Qwen3-Omni/kcl 0.4673 vs 0.4688.
  Qwen2.5-Omni/kcl 0.7619 vs 0.7634. Qwen3-Omni/neurovoz 0.5656 vs 0.5664. Qwen2.5-Omni/neurovoz 0.5371 vs 0.5372.
- 92 of 98 reproduce exactly. The KCL one is explained: exactly one tied pair, counted as a loss by the stored value and as half a win by the rank formula.

Status now: fixed in the paper. Table 1 rows, verbatim: "PC-GITA     & 1100 (100) & 0.56 & 0.52 & 0.89 \\" / "NeuroVoz    & 1270 (107) & 0.70 & 0.54 & 0.90 \\" / "MDVR-KCL    &   37 (37)  & 0.71 & 0.76 & 0.70 \\". These rows print 0.52, 0.54 and 0.76, the same under the stored and the recomputed values. The Qwen3-Omni transcript values are not printed.

## 80. [HIGH] Prose in the M9 draft that must NOT reach the paper
- "Qwen3-Omni sits at 0.9995+ everywhere" is false: its lowest cell is edaic_full at 0.9992.
- "the Qwen2.x models and Kimi at 0.988-0.999" is false: Qwen2-Audio's lowest is adresso/transcript at 0.9110.
- "both Audio Flamingo models an order of magnitude looser" is false twice: Qwen2-Audio at 0.9110 is looser than AF2's worst (0.9304), and 0.897 vs 0.999 is not an order of magnitude.

Status now: released here. Files: scores/part16/M/M9_answer_mass.csv. The final tex has one answer-mass sentence, "the two sets hold a median of at least 0.89 of the probability mass in every cell." The smallest median is 0.8968. None of the three claims above is in the tex.

- 81. LOW: POD2's fresh LoRA does not reproduce the pre-existing LoRA run clip for clip. Same config, same 468 clips, same 5 folds: OOF AUC 0.8250 (POD2, fresh) vs 0.8202 (`abl_pitt_oof.csv`, 2026-09-18). Only 6 of 468 per-clip p_yes values are bit-identical, but Pearson r = 0.9402 and rank correlation = 0.9507 between the two score vectors, and the 0.0048 gap sits far inside either interval. Cause is ordinary run-to-run nondeterminism: the LoRA A-matrix init and the dropout masks draw from torch's global RNG, which neither script seeds, and the CUDA kernels are not in deterministic mode. Treat this as an independent replication of the baseline, not a mismatch. Nothing was overwritten.

  Status now: released here. Files: lookup/master_lookup.csv, scores/part20/POD2/balanced_minus_earlier_lora_pitt_perclip.csv, scores/part20/POD2/control/uniform_lora_pitt_control_oof.csv. The lookup records 0.8250, the value the final tex prints as 0.825. The control file is a seeded rerun of the same recipe (item 169).

- 82. LOW: two defensible bootstrap protocols for a single-arm interval. Resampling speakers *within that arm's* speaker universe, versus resampling all 228 speakers and then subsetting to the arm, give slightly different ends (e.g. POD2 conflict arm: [0.4348, 0.6833] vs [0.4376, 0.6789]). POD2 reports the arm-restricted version as primary, consistent with the paired-difference rule that draws one speaker list per replicate. Both are recorded in the sidecar.

  Status now: released here. Files: scores/part17/T6_pitt_text_sep20_arms.json, release/scores/part20/rows/POD2.tsv (authors' machine path). The released arm intervals resample the speakers present in that arm, as the sidecar records.

## 83. POD4 / Part 16 4a: NeuroVoz IS channel-confounded, and the normalisation FAILS the acceptance test (HIGH)
Five quiet-frame channel features (noise floor, SNR, spectral centroid, roll-off,
tilt), measured on the 16 kHz mono signal the encoder actually receives, separate
PD from HC in NeuroVoz at AUC 0.7977 [0.7248, 0.8611] with a
GroupKFold(5)-by-speaker logistic probe, BEFORE any normalisation.
Roll-off alone 0.8005, centroid alone 0.7782, tilt alone 0.7020.
So the Italian-corpus objection does apply to NeuroVoz: a probe on this data
could be reading the recording setup.
After -23 LUFS normalisation plus shaped noise fitted to the dataset's 90th
percentile noise floor and tilt, the five-feature AUC is 0.6025 [0.5479, 0.6580].
The authors' stated acceptance test is that this must fall BELOW 0.60.
It did not. 0.6025 > 0.60, so the normalisation FAILED, by 0.0025.
Per the authors' own rule, the NeuroVoz after-normalisation encoder probe number
therefore CANNOT be presented as a clean rebuttal of the channel objection, and
POD4 does not present it as one. It is reported as a number obtained under a
normalisation that did not fully close the channel.
Shaping was verified per file: residual noise-floor error mean +1.216 dB (sd 2.109,
79.7% within 1 dB), residual tilt error mean -0.0213 dB/kHz (sd 0.1916, 78.3%
within 0.1 dB/kHz). Noise can only be added, never subtracted, which is why the
residual is biased positive.

Status now: fixed in the paper. Quotes: "Five recording features from the quietest frames (noise floor, SNR, spectral centroid, roll-off and tilt) separate the NeuroVoz groups at 0.80 and the PC-GITA groups at 0.67, so part of the Parkinson's probe may come from the recording channel." and "Normalising loudness and noise floor brings them to 0.56 on PC-GITA, where the encoder probe falls to 0.83 ($-$0.07 [$-$0.12, $-$0.02]) and the answer stays at 0.57, and only to 0.60 on NeuroVoz, where the probe falls to 0.89 ($-$0.03 [$-$0.061, $-$0.001])." The tex reports the NeuroVoz check as it came out, not as a clean rebuttal.

## 84. POD4 / Part 16 4b: the interviewer's PRESENCE is itself label-correlated in the Pitt windows (HIGH, and it is now controlled)
Measured from the CHAT tiers over the 30 s the model hears:
mean timed *INV speech is 852.5 ms in dementia windows vs 186.4 ms in control
windows. 127/290 dementia windows contain interviewer speech vs 31/178 control
windows. Interviewer-milliseconds ALONE predicts the diagnosis at AUC 0.6427.
So the reviewer objection was well founded in its premise: the original Pitt cut
does carry a label-correlated interviewer cue.
What POD4 found when that cue is removed (arm B, participant intervals only):
  zero-shot  0.6497 -> 0.7230, PAIRED delta +0.0732 [+0.0312, +0.1165]
  enc probe  0.7949 -> 0.7497, PAIRED delta -0.0452 [-0.0857, -0.0083]
Both CIs exclude zero. The zero-shot model was NOT exploiting the interviewer -
removing it makes the model better. The encoder PROBE was partly exploiting it,
losing 0.045 AUC.
Duration control (arm C, same 17.6 s mean but interviewer and gaps retained):
  enc probe 0.7980, PAIRED vs arm A +0.0032 [-0.0367, +0.0394], duration alone
  does nothing to the probe, so the 0.045 probe drop is attributable to the
  interviewer/gap removal and not to the shorter audio.
REMAINING CAVEAT, stated plainly: inside arm B the participant-only duration is
itself label-correlated (mean 19.85 s control vs 16.24 s dementia, duration alone
gives a direction-free AUC of 0.6648). That is "how much the person spoke", a
plausible clinical signal rather than a channel artefact, but arm B's 0.7497 is
not free of it.
CONSEQUENCE FOR THE PAPER: on interviewer-free audio the Pitt representation
utility gap shrinks from 0.1452 (0.7949 probe vs 0.6497 zero-shot) to 0.0267
(0.7497 vs 0.7230). The headline gap on this dataset is substantially an
interviewer effect. This must not be reported as "the Pitt result survives
unchanged".

Status now: open. Under the paper's estimators the interviewer-free cut gives zero shot 0.7231 [0.6663, 0.7750] and encoder probe 0.7643 [0.7135, 0.8116] on the Mac folds, against 0.6578 and 0.7706 on the original windows. The final tex says "Each Pitt segment starts at the participant's first timed utterance, and the ADReSSo and ADReSS-2020 windows start at the beginning of the recording, so interviewer prompts inside the window are kept." but does not report this check. What decides it: whether a later version reports it. Related files: scores/part16/POD4/pitt_no_interviewer.csv, scores/part20/POD5/POD5_zs_o25_noinv.csv, scores/part20/POD5/POD5_nested_enc_mac_seed.csv, release/scores/part20/rows/POD5.tsv (authors' machine path).

## 85. POD4 / Part 16 4a: PC-GITA is only mildly channel-confounded and the normalisation WORKS there (MEDIUM, informational)
PC-GITA, clean manifest (100 speakers x 11 clips, 550/550, all recordings taken
from the "sin normalizar"/"non-normalized" folders, the known-leaky
pcgita_states_leakyspk.npz was NOT used).
Five quiet-frame channel features together, GroupKFold(5)-by-speaker probe:
  BEFORE 0.6657 [0.5851, 0.7452]   AFTER 0.5590 [0.4922, 0.6213]
Acceptance test (must fall below 0.60): PASSED.
Only the noise floor carried much on its own before (0.6627). Centroid, roll-off
and tilt were all near 0.55.
This is a sharp contrast with NeuroVoz (before 0.7977, after 0.6025, FAILED).
So the channel objection is dataset-specific: it bites hard on NeuroVoz, only
mildly on PC-GITA, and only for PC-GITA can the after-normalisation probe be
presented as a clean rebuttal.
Shaping verified per file for PC-GITA: residual noise-floor error mean +1.152 dB
(sd 1.148, 63.6% within 1 dB). Residual tilt error mean -0.0581 dB/kHz
(sd 0.1244, 69.1% within 0.1 dB/kHz).

Status now: fixed in the paper. Quote: "Normalising loudness and noise floor brings them to 0.56 on PC-GITA, where the encoder probe falls to 0.83 ($-$0.07 [$-$0.12, $-$0.02]) and the answer stays at 0.57, and only to 0.60 on NeuroVoz, where the probe falls to 0.89 ($-$0.03 [$-$0.061, $-$0.001])."

## 86. POD3 / Part 16 3a: the same drift reproduces on PC-GITA, and it refines the diagnosis (HIGH, same root cause as the Pitt entry above)
POD3 re-extracted PC-GITA 1100 on Qwen3-Omni from the original audio. Zero-shot answer AUC
0.6015 here vs 0.5967 in `release/overnight2/q3o_new/q3o_pcgita_zeroshot.json`,
delta +0.0048, the same magnitude and the same sign as the Pitt delta (+0.0054).
Clip order, labels and prompt are byte-identical between the two PC-GITA runs.
Same bf16 ladder signature: only 501 (shipped) and 504 (pod) distinct p_yes values in 1100 clips.
Spearman 0.9658, so the ordering is again nearly preserved.

TWO NEW FACTS THAT SHARPEN THE CAUSE:
1. The Pitt shift is NOT a universal bias. On Pitt the pod was systematically less likely to
   say Yes (mean logit -0.2898 -> -0.4285, a -0.139 shift). On PC-GITA there is no such shift
   (mean logit 0.3330 -> 0.3407, +0.008, pod higher on 53.4% of clips, i.e. a coin flip).
   A changed chat template, a changed prompt or a changed token set would bias BOTH datasets in
   the same direction. They do not. This rules those out and leaves numeric perturbation.
2. The size of the divergence tracks AUDIO LENGTH. Pitt clips are all longer than 30 s and are
   truncated to the full 30 s window: mean |d logit| 0.339. PC-GITA is mostly single sentences
   of a few seconds: mean |d logit| 0.146, and only 2.8% of clips move more than 0.1 in p_yes
   against 18.6% on Pitt. More encoder frames means more accumulated numeric divergence before
   the answer position, which is what a low-level numeric difference predicts and what a
   configuration difference does not.
Taken with the bit-exact double rerun on this pod, the cause is a deterministic low-level
numeric difference between the two sessions (GPU/kernel/attention backend/library build),
amplified by MoE top-k routing and by the coarse bf16 ladder p_yes sits on. The effect on the
reported AUC is about +0.005 on both datasets, well inside the speaker bootstrap interval
(Pitt pod 0.7673 [0.7101, 0.8198] vs shipped 0.7619 [0.7054, 0.8130]).

Status now: fixed in the paper. Quote: "On Qwen3-Omni the encoder probe reaches 0.90 on PC-GITA and 0.81 on Pitt against zero-shot answers of 0.60 and 0.76, so its Alzheimer's gap is only 0.05." PC-GITA prints as 0.60 under the shipped value 0.5967 and under the rerun 0.6015. The Pitt 0.76 is the shipped 0.7619 (items 29 and 73).

## 87. POD4 / Part 16 4a: which PC-GITA manifest was used (LOW, informational)
The known-leaky states file
gdrive/probe2/pcgita_states_leakyspk.npz
was NOT used and was never read.
POD4 built its PC-GITA clip list from
release/omni_final/omni_pcgita_zeroshot_scores.csv
and re-extracted states from the source audio. Verified clean: 100 speakers,
exactly 11 clips each, 550 PD / 550 HC, and ZERO speakers carrying both labels
(controls are AVPEPUDEAC*, patients AVPEPUDEA0*). All folds are
GroupKFold(5) grouped by that speaker id.

Status now: released here. Files: folds/pcgita_groupkfold5_pod.csv, folds/README.md. The fold file lists the 1100 clips and 100 speaker labels.

## 88. POD3 / Part 16 3a: DEMONSTRATION that a kernel swap alone reproduces the drift signature (HIGH, closes the Pitt/PC-GITA entries above)
To turn the diagnosis from inference into a measurement, POD3 scored the SAME 468 Pitt clips
twice with only the attention backend changed (attn_implementation "sdpa" vs "eager"), holding
the checkpoint, weights, prompt, audio, 30 s window, bf16 dtype, GPU (H200), transformers
5.17.0 and torch 2.8.0+cu128 all constant.
Files: `part16/POD3/q3o_pitt_attn_backend.csv` / `.json`.

  sdpa AUC 0.767319, eager AUC 0.767919, delta +0.0006
  0 of 468 clips identical. 80.8% differ by more than 0.001. 4.7% by more than 0.1
  max |delta p_yes| 0.1826. Mean |delta logit| 0.2019. Mean SIGNED delta logit -0.0048
  Spearman 0.9830

Side by side with the shipped-vs-pod gap being explained:
  quantity                  kernel swap only      shipped vs this pod
  frac clips |dp| > 0.001         0.808                 0.880
  frac clips |dp| > 0.1           0.047                 0.186
  mean |delta logit|              0.2019                0.3389
  Spearman                        0.9830                0.9529
  AUC delta                      +0.0006               +0.0054

Changing nothing but a low-level kernel therefore produces the SAME KIND of per-clip
divergence at roughly 60% of the magnitude. The shipped-vs-pod gap is somewhat larger,
which is what you expect when more than one numeric variable differs between two sessions
(kernel plus GPU model plus library build) rather than one. Combined with the bit-exact
double rerun, the near-zero signed shift, and the divergence scaling with audio length,
this closes the diagnosis: the 0.7619 vs 0.7673 gap is low-level numeric environment, not
data, not configuration, and not within-run nondeterminism.
PRACTICAL NOTE for the paper: a Qwen3-Omni zero-shot AUC on these sets is reproducible only
to about +/-0.005 across environments unless the GPU, kernel and library build are pinned.
That is far inside the speaker bootstrap interval (Pitt +/-0.055) and does not touch any
conclusion, but the exact digits should not be presented as bit-reproducible.

Status now: released here. Files: release/edaic_rerun/part16/POD3/q3o_pitt_attn_backend.csv (authors' machine path), release/edaic_rerun/part16/POD3/q3o_pitt_attn_backend.json (authors' machine path).

## 89. POD1 / Part16 1c: sft_full per-fold checkpoints never existed, and its TRAINING AUDIO is unrecoverable (HIGH)
2026-09-23.
(a) NO CHECKPOINTS. sft_projector.py (read at release/scripts/sft_projector.py and the p15 copy)
    writes only <OUTPREFIX>_state.json (fold list + oof score list) and <OUTPREFIX>_oof.csv. It never
    calls torch.save and never writes a .pt/.safetensors. Searched release/ and
    gdrive/ for *sft*, *.pt, *.safetensors, *.bin: the only .pt files
    anywhere are overnight2/part10/omni25_readout.pt, overnight2/part6/transfer_neurovoz_projector.pt
    and transfer_pcgita_projector.pt, none is an E-DAIC per-fold sft checkpoint. So 1c cannot reload.
    it can only retrain.
(b) TRAINING AUDIO NOT RECOVERABLE. The sft_full run trained on /workspace/data/full/<pid>.wav
    (275 speakers, one file each, recorded in release_public/scores/edaic_windows/sft_full_sft.json
    and sft_full_oof.csv). Those files are not on the Mac or the G-Drive. They are NOT the raw E-DAIC
    tarball <pid>_AUDIO.wav: rebuilding from the tarballs and scoring zero-shot against the published
    release_public/scores/edaic_windows/o25_full_zeroshot_scores.csv gives
      first 30 s of raw session, WINDOW_S=30 : max|dp_yes| 0.9105, mean 0.2755, r 0.0795
      complete raw session, WINDOW_S=0 (n=20): max|dp_yes| 0.8271, mean 0.2852, r 0.5150
    Identical audio reproduces to ~1e-10 on this pipeline (demonstrated: the rebuilt Part 14 clips are
    bit-identical to part14_clips and their scores match the published p14_o25 scores to 5.0e-11).
    So data/full/*.wav is a different stream, most plausibly the participant-only "dcaps_proc" stitched
    audio referenced in edaic_conflict/edaic_fetch_transcripts.sh. The exact stitching rule is not
    recorded anywhere on disk, so it was NOT guessed.
(c) WHAT WAS DELIVERED. A labelled RETRAIN on the raw session audio, first 30 s, same hyperparameters
    (projector only, AdamW lr 1e-4, 3 epochs, 5 speaker-disjoint GroupKFold folds, bf16) and strict
    out-of-fold-by-speaker scoring of the 966 Part 14 clips. Its E-DAIC OOF AUC is 0.5315 against the
    original sft_full 0.6421, which confirms the input mismatch.
    CONSEQUENCE: the 1c conflict/agreement AUCs are internally valid but are NOT comparable to the
    paper's sft_full row and must not be reported as reproducing it.

Status now: released here. Files: scores/edaic_windows/sft_full_oof.csv, scores/edaic_windows/sft_full_sft.json, scripts/edaic_full/build_windows.py. build_windows.py rebuilds the stitched training windows (item 102). The 1c retrain on raw audio is not used in the paper.

## 90. POD4 / Part 16 4a: final before/after encoder probe numbers (informational)
NeuroVoz (normalisation FAILED the 0.60 bar, so the after number is NOT a clean rebuttal):
  zero-shot  0.7108 [0.6523, 0.7671] -> 0.6991 [0.6348, 0.7598], PAIRED -0.0117 [-0.0332, +0.0098] (CI includes 0)
  enc probe  0.9233 [0.8800, 0.9573] -> 0.8924 [0.8481, 0.9323], PAIRED -0.0308 [-0.0608, -0.0012]
PC-GITA (normalisation PASSED, so the after number IS usable):
  zero-shot  0.5728 [0.5002, 0.6440] -> 0.5674 [0.5054, 0.6304], PAIRED -0.0054 [-0.0437, +0.0357] (CI includes 0)
  enc probe  0.8959 [0.8339, 0.9448] -> 0.8298 [0.7546, 0.8968], PAIRED -0.0661 [-0.1181, -0.0180]
Reading: on PC-GITA, with the five-feature channel probe driven to 0.5590, the
encoder probe still reaches 0.8298 against a zero-shot of 0.5674. The utility gap
on PC-GITA is not a channel artefact. On NeuroVoz the same claim cannot be made,
because the channel was not closed to the agreed bar.

Status now: fixed in the paper. Quote: "Normalising loudness and noise floor brings them to 0.56 on PC-GITA, where the encoder probe falls to 0.83 ($-$0.07 [$-$0.12, $-$0.02]) and the answer stays at 0.57, and only to 0.60 on NeuroVoz, where the probe falls to 0.89 ($-$0.03 [$-$0.061, $-$0.001])."

## 91. POD3 / Part 16 3a: PC-GITA projector AUC differs in the 4th decimal between memory and the per-clip csv (LOW)
Verification flagged one value out of 17: PC-GITA projector nested AUC, run-side json 0.8872
vs 0.8873 recomputed from `part16/POD3/q3o_pcgita_proj_nested_oof.csv`.
Cause, confirmed: the nested_oof csvs store p_probe at 8 decimals. The probe saturates on this
stream, so 99 clips write as exactly 1.00000000 and 29 as exactly 0.00000000, producing 329
cross-class exact ties that do not exist in the full-precision vector held in memory. A
tie-averaged AUC therefore differs slightly depending on which vector it is computed from:
in memory 0.887175, recomputed from the csv 0.887284.
Both AUC implementations agree with each other exactly on the SAME input (rank formula and
brute-force pairwise both return 0.887284297521 from the csv), so this is a write-precision
effect, not an AUC-code disagreement.
POD3 reports 0.8873, the value recomputable from the per-clip file, because the Part 16 rule is
to recompute the AUC from the per-clip scores rather than copy it out of a json. The in-memory
value is recorded here and in the sidecar. All 16 other POD3 values agree to 4 dp between the
two independent implementations, so the effect is below 4 dp everywhere else.

Status now: released here. Files: release/edaic_rerun/part16/POD3/q3o_pcgita_proj_nested_oof.csv (authors' machine path).

## 92. MEDAIC / Part 16: the full-window E-DAIC probe has NO per-clip file, so the requested paired interval cannot be computed (HIGH)
TASK ASKED FOR: the paired "probe minus answer" interval for E-DAIC on the FULL window
(LM probe ~0.69 against answer ~0.83).
WHAT EXISTS ON DISK:
  - answer side, per clip, 275 rows: edaic_rerun/variants/o25_full_zeroshot_scores.csv
    recomputed with the rank formula = 0.8285 [0.7708, 0.8827]. The ~0.83 REPRODUCES exactly.
  - probe side, AGGREGATE ONLY: edaic_rerun/variants/o25_full_nested_repeats.json
    llm mean 0.6853 over 5 repeats, enc 0.5930, ans 0.7542, proj 0.5293. The ~0.69 matches
    the llm mean. But this json holds only per-repeat scalars. There is NO *_nested_oof.csv
    and NO o25_full_states.npz anywhere on the Mac.
EVIDENCE OF ABSENCE: an exhaustive walk of release for csv files
whose name or path contains edaic/conflict AND that carry a per-row score column found 151 per-clip
files. For the FULL window the only per-clip files are the five zero-shot answer files
(o25/q2a/q3o/kimi/af3), sft_full_oof.csv (projector fine tune) and o25_edaicfull_text.csv
(transcript only). No nested-probe OOF for the full window exists. The same walk finds the
30 s window DOES have omni_edaic_{enc,llm,ans,proj}_nested_oof.csv, which is why M7 could compute
its paired interval and this cell cannot. The existing PART10 bootstrap table likewise carries
full-window rows only for zero shot and projector SFT, never for a full-window nested probe.
WHY IT CANNOT BE REGENERATED ON THE MAC: the full-window run was a pod run
(o25_full_zeroshot.json records device "cuda", clip_list /workspace/mf_full.csv). Re-extracting
states needs Qwen2.5-Omni-7B over 275 clips whose median window is 591.8 s and whose cap is 900 s
(~22k audio tokens per clip at the 25 tok/s audio rate). That is not runnable on MPS, and this cell
is Mac only, no pods, no GPU. The stitched audio does exist at
gdrive/drive_data/dcaps_proc, so the run is reproducible on a GPU
pod, just not here.
DELIVERED INSTEAD, all per clip and all with paired speaker intervals: projector fine tune minus
answer and transcript only minus answer on the full window, plus the three 30 s probe stages minus
answer. The full-window nested probe point estimates are reported as json aggregates, explicitly
labelled not-recomputed, with no interval attached. No interval was invented for them.

Status now: released here. Files: scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv, scores/part17/T7b/T7b_edaic300_meanof5.csv, release/edaic_rerun/part17/rows/T7b.tsv (authors' machine path), lookup/master_lookup.csv, scores/part16/M/MEDAIC_full_window.csv. The authors' rule for E-DAIC is the mean of five per-repeat AUCs, as for Pitt: LM probe 0.7001 [0.6231, 0.7727], encoder probe 0.5884 [0.5268, 0.6523], and paired minus the zero-shot answer 0.8278, LM -0.1276 [-0.1993, -0.0629] and encoder -0.2394 [-0.3188, -0.1572]. The averaged-probability values 0.7088 and 0.5916 are listed as not used.

## 93. POD4B-D1  HIGH  ADReSSo released segmentation DOES exist on disk, inside the un-extracted tgz
PART5_interviewer_measure.txt section 5.2 states: "There is no timed transcript on disk for it, and
there is no speaker-labelled transcript either. The ADReSSo folder ... holds only four .tgz archives."
That conclusion was drawn from the folder listing only. The archives were never opened.
`tar -tzf "DementiaBank/challenges/ADReSSo/ADReSSo21-diagnosis-train.tgz?f=open"`
lists ADReSSo21/diagnosis/train/segmentation/{cn,ad}/adrsoNNN.csv, the released speaker-diarised
segment lists. So the ADReSSo interviewer share IS measurable and PART 5's "NOT MEASURABLE" for
ADReSSo is wrong as written. POD4B uses the released segmentation from inside the archive.
Severity HIGH because PART 5's ADReSSo verdict ("rule cannot be applied") rests on it.

Status now: released here. Files: scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json. These are built from the released ADReSSo segmentation. reports/PART5_interviewer_measure.txt section 5.2 is the superseded statement.

## 94. POD3 / Part 16: the Mac boot disk is at 1.5 GiB free, so the extracted states could not go to part16/ (MEDIUM, operational)
The boot disk of the authors' machine was full, with 1.5 GiB
free. POD3's two state tensors are 206 MB (Pitt 468) and 484 MB (PC-GITA 1100). Writing them
into part16/POD3/ would have consumed most of the remaining headroom on the boot volume.
POD3 therefore synced every RESULT file (per-clip csvs, per-layer csvs, nested OOF csvs,
sidecar jsons, logs, scripts, 1.7 MB total) to part16/POD3/ as planned, and put the large
state tensors on the external drive instead, at
  gdrive/part16_POD3_states/
The PC-GITA states are the expensive artefact: re-making them needs an H200 and a fresh upload
of the 1100 wavs. They are worth keeping before the pod is terminated.
Flagging the disk itself because 1.5 GiB free will start causing unrelated failures.

Status now: released here. Files: release/edaic_rerun/part16/POD3/q3o_pitt_attn_backend.csv (authors' machine path), release/edaic_rerun/part16/POD3/q3o_pcgita_proj_nested_oof.csv (authors' machine path). The POD3 result files are carried here. The state tensors are not released and stay on the authors' drive.

## 95. POD4B-D2  MEDIUM  ADReSSo released segmentation covers 166 of the 237 scored clips, not all 237
The 237 ADReSSo clips are the ADReSS-M English train split (ids adrsoNNN), already logged as D7.
The ADReSSo21 released segmentation files carry ids as follows, checked by tar TOC on all four archives:
  diagnosis-train  166 ids, all adrsoNNN  -> 166/166 match the manifest, 0 unmatched
  diagnosis-test    71 ids, all adrsdtN   -> 0 match (test set was re-anonymised for release)
  progression-train 48 ids, all adrspNNN  -> 0 match
  progression-test  16 ids, all adrspt N  -> 0 match
So exactly 166 of the 237 can be resolved to the RELEASED segmentation and 71 cannot, because no
public adrsdt -> adrso id map ships with the challenge data. POD4B runs ADReSSo on those 166 only
and does NOT substitute a diariser for the other 71. n=166 is stated on every ADReSSo row.

Status now: released here. Files: scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json. n is stated on every ADReSSo row.

## 96. POD4B-D3  MEDIUM  Five ADReSSo released segmentation files have no PAR tier at all
cn/adrso018, adrso019, adrso021, adrso022, adrso023 contain only INV rows (3, 3, 4, 8 and 11 rows
respectively) and zero PAR rows in the released ADReSSo21 diagnosis-train segmentation. Checked by
counting the tier column in the released csv, so it is a property of the distribution and not a
parse failure in the run. These five cannot form the participant-only arm, so ADReSSo runs on
161 clips, not 166. Every ADReSSo row reports n=161.

Status now: released here. Files: scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json. The ADReSSo rows run on 161 clips.

## 97. POD4B-D4  LOW  ADReSS-2020 CHAT bullets run slightly past the source wav on some files
Last CHAT time bullet minus source recording length, over all 156 files: median -377 ms,
min -6819 ms, max +3981 ms, 11 files off by more than 2000 ms in absolute value. Intervals are
clipped to the scored window and to the clip length before use, so an overhang cannot create audio.
it can only shorten the last participant interval. Recorded because the alignment is not exact.

Status now: released here. Files: scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json. The window spans in this file are clipped to the recording as the finding describes.

## 98. POD4B-D5  MEDIUM  No saved fold assignment exists for the ADReSS-2020 or ADReSSo clips
Checked the published per-clip files that rule 7 names as the best source:
  release/omni_final/omni_adress2020_enc_nested_oof.csv
  release/omni_final/omni_adresso_enc_nested_oof.csv
Both carry only clip,speaker,label,p_probe, there is no fold column, and no separate fold file
exists for either corpus. POD4B therefore uses sklearn GroupKFold(n_splits=5) grouped by speaker
over the clips in the source csv's own order, per rule 7. On both corpora each speaker contributes
exactly one clip, so this reduces to a deterministic 5-fold split that is IDENTICAL across arms A,
B and C, which is what makes the paired arm contrast exact. Inner layer choice is GroupKFold(4) on
the training speakers. Recorded in every POD4B sidecar as well.

Status now: released here. Files: scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json. The sidecars record the fold rule used for arms A, B and C.

## 99. POD4B-D6  MEDIUM  ADReSSo published encoder probe scores higher than the POD4B arm-A rerun on the same clips
On the same 161 clips: published encoder nested-probe AUC 0.9046, POD4B arm A (the identical audio,
same prompt, same model) 0.8624. The audio and the scores are not the cause, the zero-shot stream
agrees closely on the same clips (published 0.6320, arm A 0.6291). The difference is the probe's
training set: the published probe was fitted with folds over all 237 clips and is merely EVALUATED
on 161 here, while the arm-A probe was fitted with folds over the 161 only, so it trains on ~68% as
much data. The arm A/B/C comparison is unaffected because all three arms share the identical fold
assignment and the identical 161 clips. The published number is therefore NOT the right baseline for
the POD4B deltas, and only arm A is used as the within-experiment anchor.
For ADReSS-2020 no such issue arises: n=156 in both, published 0.7878 vs arm A 0.7959.

Status now: released here. Files: scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json. Arm A is the comparator for these deltas.

---
## 100. POD2 / Part16 EDAICFULL, D1 is wrong about WHY the full window is truncated. SEVERITY: HIGH
Date 2026-09-23. Who POD2 (dtz7yg0r57zikq), rebuild of the E-DAIC full window from raw tarballs.

What D1 says. That Qwen2.5-Omni emits ~25 audio tokens/s against a 32,768-token context, that
"52 of 275 files (19%) exceed the context", and that the 900 s cap "is 22,500 tokens and still
truncates 89 of 275 files". The `windows_full.json` sidecar carries the same caveat as
"AFFECTED: D1 context limit".

What the model actually does. `Qwen2_5OmniProcessor.feature_extractor` is a
`WhisperFeatureExtractor` with `chunk_length = 300`, `n_samples = 4,800,000` (= 300 s x 16 kHz),
`truncation = True` and `max_length` defaulting to `self.n_samples`. Any waveform longer than
300 seconds is silently truncated to its first 300 seconds before the audio tower ever runs.

Measured over the 275 rebuilt full windows (`out/full_zeroshot_scores.csv`, columns
`n_audio_tok`, `seq_len`, `dur_s`):
- audio tokens: min 1648, median 7500, max 7500. 7500 = 300 s x 25 tok/s, an exact ceiling.
- sequence length: min 1696, max 7548.
- `seq_len <= 32768` for 275 of 275 clips. The 32k context is never the binding constraint and
  is never even approached: the longest sequence uses 23% of it.
- 217 of 275 windows (78.9%) are longer than 300 s and are therefore truncated, not 89, and
  not for the reason D1 gives.
- Audio discarded by the 300 s cap: 84,144 s total (23.4 hours). Median 438 s over the
  truncated clips, max 600 s.

Consequence for the published number. `edaic_full` zero-shot 0.8285 is computed on the
first 300 seconds of each full window, not on 900 s and not on the full interview. The 900 s
cap in `windows_full.csv` is inert: for every clip longer than 300 s the extra audio is discarded
inside the processor. The `full` and `mid300` variants differ only because their windows *start*
at different offsets (full at `start_s`, mid300 at the window midpoint), not because `full`
carries three times the audio.

Evidence this is not a rebuild artefact. The rebuild reproduces the published run clip by clip:
recomputed AUC 0.8278 vs published 0.8285, Pearson r = 0.99988 on per-clip p_yes, mean |diff|
0.0019, median answer_mass 0.9962 in both. The same 300 s truncation was therefore in force for the
original run too.

What would fix it. Either state that the E-DAIC audio condition is a 300 s window (and retire
the 900 s cap language), or move to a chunk-and-pool design that encodes the window in 300 s
segments and pools across them. Do not describe this condition as the full interview.
Status. Logged. The probe and the paired interval below are computed on these same windows, so
they are directly comparable to the published 0.8285. Every one of them inherits the 300 s cap.

Status now: fixed in the paper. Quote: "The model hears the participant's whole interview, interviewer turns removed and capped at 15 minutes for the 89 longest, with the processor's default input limit lifted so that every window is encoded whole." The paper's E-DAIC audio numbers now come from the run with the processor's 300 s cut lifted (PART 23): zero-shot answer 0.8366 [0.7786, 0.8866], against 0.8278 with the cut, paired +0.0088 [-0.0320, 0.0510] (part23/verify/PART23_VERIFIED.tsv, authors' machine path). A second run of the same setting gives 0.8379 [0.7805, 0.8876]. Files: scores/part22/P22_step1_processor_config.md, scores/part22/p22_step2_truncation_proof.json, scores/part22/edaic_lifted_perclip.csv, scores/part22/edaic_lifted_auc.json, manifests/edaic_windows_full.csv. The authors' rule: Qwen2_5OmniProcessor (transformers 5.17.0) cuts audio at 300 s by default, and 217 of 275 windows are longer than 300 s. The final tex lifts that cut. 89 of the 275 windows still reach the 900 s cap, as the quoted sentence says.

## 101. POD2 / Part16 EDAICFULL, o25_full_proj_perlayer.csv is a copy of the LM embedding stage. SEVERITY: MEDIUM
`variants/o25_full_proj_perlayer.csv` has one row and it is byte-identical to row `stage 0` of
`variants/o25_full_llm_perlayer.csv` (auc_mean 0.5950620155038759, auc_std 0.11072781384588515,
auc_oof 0.5895316804407713). The projector per-layer file therefore records the language model's
embedding stage, not the audio projector output. Note this value (0.5950) also disagrees with the
projector entry in `o25_full_nested_repeats.json` (0.5293, 1 repeat), so the two published projector
numbers were not produced by the same code path. POD2's own rebuild probes the actual
`thinker.audio_tower.proj` output and gets 0.5610 (mean of 5 repeats) / 0.5631 (AUC of mean OOF).
Nothing was overwritten.

Status now: released here. Files: scores/edaic_windows/o25_full_proj_perlayer.csv, scores/edaic_windows/o25_full_llm_perlayer.csv, release/edaic_rerun/part17/rows/T7b.tsv (authors' machine path). The identical row can be checked in the two released files. T7b gives the rebuild's projector probe, 0.5610 as the mean of five.

## 102. POD2 / Part16 EDAICFULL, NOTE (not a discrepancy): the rebuild reproduces the published run.
Independent rebuild from the raw E-DAIC tarballs, with POD2's own stitching, own extraction, own
rank-formula AUC and own bootstrap:
- zero shot recomputed 0.8278 vs published 0.8285. Per-clip p_yes Pearson r 0.999882,
  mean |diff| 0.0019, max |diff| 0.0303. Median answer_mass 0.9962 in both.
- per-layer curves vs the original `*_perlayer.csv`: ans r 0.9992 (peak 0.7903 vs 0.7929),
  llm r 0.9997 (peak 0.7636 vs 0.7630), enc r 0.9995 (peak 0.6922 vs 0.6896).
- nested probe, mean of 5 repeats, POD2 vs published: enc 0.5884 vs 0.5930, llm 0.7001 vs 0.6853,
  ans 0.7435 vs 0.7542, proj 0.5610 vs 0.5293 (published proj had 1 repeat).
Residual differences are ordinary run-to-run variation plus unknown fold seeds in the original
nested run: sklearn `GroupKFold` takes `shuffle`/`random_state`, and no fold or seed file for the
original Omni nested run exists on disk, so POD2 used `random_state = repeat` for repeats 0..4 and
records that in the sidecar. E-DAIC full is one clip per speaker (275/275), so speaker folds and
clip folds coincide.

Status now: released here. Files: scores/part16/EDAICFULL/full_zeroshot_scores.csv, scores/edaic_windows/o25_full_zeroshot_scores.csv, scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv, scores/part22/edaic_lifted_perclip.csv. The published run and the rebuild are both released per clip. The rebuild gives the zero shot 0.8278.

## 103. POD2 / Part16 EDAICFULL, LOW: a label-indexing bug in POD2's first probe pass, caught and fixed.
POD2's first nested run passed inner-loop indices to a helper that closed over the global label
vector, so the inner layer-selection CV fitted against labels belonging to other clips. It selected
near-random layers (e.g. stage 0 of `ans`, which is constant and scores exactly 0.5) and depressed
the nested values: enc 0.5974, llm 0.6609, ans 0.6633. The outer fit was never affected. Caught by
comparing POD2's per-layer curves against the original `*_perlayer.csv`, the curves matched at
r > 0.999 while the nested numbers did not, which localised the fault to the nesting rather than the
states. Fixed by passing the label vector explicitly. Reported values are from the fixed run
(`probe_boot.py`, log `probe2.log`). Both logs are kept in `part16/EDAICFULL/`.

Status now: released here. Files: scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv, release/edaic_rerun/part17/rows/T7b.tsv (authors' machine path). The fixed nested probe is reproduced on Linux x86 with sklearn 1.9.1.

## 104. [CORRECTION] POD4's claim that "0.7706 exists in no file" is WRONG, retracted
- POD4 appended a MEDIUM entry asserting the Pitt encoder probe value 0.7706 could not be found in any file. That is false.
- 0.7706 reproduces exactly from `release/overnight2/part10/pitt_enc_nested5_oof.npz`:
  oof shape (5, 468), per-repeat AUCs [0.7598, 0.7725, 0.7459, 0.7903, 0.7843], mean 0.7706, n=468, 228 speakers.
  Independently confirmed twice: by `part16/M7_pitt_fix.py` and by POD4's own verifier with brute-force pairwise AUC.
  It is also written in plain text in `release_public/README.md`, `master/master_lookup.csv` and `bootstrap/bootstrap_cis.csv`.
- The earlier entry is retracted. This is the second time a run used `omni_final/omni_pitt_enc_nested_oof.csv` (single split, 0.7969) instead of the authoritative 5-repeat source.

Status now: released here. Files: lookup/master_lookup.csv, lookup/SUPERSEDED.csv, scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/README.md. 0.7706 reproduces exactly from the released npz with the released Mac fold files.

## 105. [HIGH] POD4's 4b used the SUPERSEDED Pitt encoder baseline for arm A
- POD4 reports the Pitt gap collapsing 0.1452 -> 0.0267 when the interviewer is removed. Its arm A encoder value (0.7949) is a single-split nested OOF, the estimator `release_public/README.md` marks superseded in favour of the 5-repeat mean 0.7706.
- The PAIRED contrasts are NOT affected, because POD4 used its own arm A consistently on its own re-cut audio: zero-shot A->B +0.0732 [+0.0312, +0.1165], encoder probe A->B -0.0452 [-0.0857, -0.0083], duration control zero-shot -0.0352, probe +0.0032 [-0.0367, +0.0394]. Those stand.
- What must NOT be quoted is the absolute "gap 0.1452" beside the paper's 0.1128, which uses a different estimator. No interviewer-free gap exists yet under the authoritative 5-repeat estimator.

Status now: released here. Files: scores/part20/POD5/POD5_nested_enc_mac_seed.csv, scores/part20/POD5/POD5_zs_o25_noinv.csv, release/scores/part20/rows/POD5.tsv (authors' machine path). These give the interviewer-free check under the paper's estimator (item 84).

## 106. [MEDIUM] POD4's two gap figures were computed from ROUNDED inputs
- gap_orig: 0.794866 - 0.649729 = 0.145138 -> 0.1451, not 0.1452 (POD4 did 0.7949 - 0.6497).
- gap_free: 0.749709 - 0.722956 = 0.026753 -> 0.0268, not 0.0267.

Status now: released here. Files: scores/part20/POD4/o25_noinv_recording.csv, scores/part20/POD4/o25_orig_recording_rerun.csv. These hold the full-precision per-clip scores.

## 107. [HIGH] Three different values are in circulation for the Q3O 3b by-arm probe
- POD3 (refit the probe INSIDE each arm): conflict 0.6145 [0.5021, 0.7246], agreement 0.9349 [0.9034, 0.9614].
- part16 3b_fix (full 468-clip fit, RESTRICTED to each arm): conflict 0.6394 [0.5323, 0.7418], agreement 0.9032 [0.8626, 0.9374].
- POD3's independent verifier, its own within-arm refit from the states: conflict 0.5520.
- These are different estimators, not a numeric error, but `PART16_RESULTS.md` currently ships the first two side by side for the same quantity. One must be chosen and labelled before any of them is cited.

Status now: released here. Files: lookup/SUPERSEDED.csv, lookup/master_lookup.csv, release/edaic_rerun/part17/rows/T4.tsv (authors' machine path). The authors' rule: the full-fit probe restricted to the arm, conflict 0.6394 and agreement 0.9032. The refit within the arm (0.6145 and 0.9349) is not used.

## 108. [HIGH] POD3's own internal verifier was a no-op on 9 of 16 cells
- Its verify script passed the same csv cell in as both `run_value` and `verify_value`, so those 9 "agreements" compared a number to itself. Only 7 of 16 were genuine recomputations. Caught by the independent verifier. The 3a block was separately and genuinely reverified and holds.

Status now: released here. Files: release/edaic_rerun/part17/rows/T4.tsv (authors' machine path). T4 recomputes the Qwen3-Omni Pitt (0.8122) and PC-GITA (0.8969) encoder probes from the POD3 per-clip files.

## 109. [HIGH] part15 B_text_conflict.json claims a de-duplication that did not happen
- Its note says "AUC computed on unique segments (483 conflict + 483 agreement). Agreement seg_uids repeat across pairs so scores are deduped before joining". The same file also records `unique_segments: 794`, contradicting itself.
- The agreement arm holds only 311 distinct segments, not 483. POD1's exact reproduction of 0.9596 proves all 483 rows were used, repeats included.
- De-duplicated values: transcript agreement 0.9596 -> 0.9620. Table 2 audio agreement 0.9482 -> 0.9647.
- Every gap WIDENS on de-duplication, so no conclusion is at risk, but the agreement-arm bootstrap intervals are narrower than they should be, because 172 of 483 rows are repeated segments.

Status now: fixed in the paper. The final tex prints the 311 distinct segment values 0.96, 0.96, 0.97, 0.63, 0.84 and 0.89 in Table 2. Quote (caption): "Depression, E-DAIC \cite{daic}: 483 conflict segments paired with agreement segments of the same speakers (311 distinct), 138 speakers." Files: scores/part17/T1_table2_distinct.csv, scores/part17/T1_perclip_distinct_agreement.csv, scores/part17/T1_perclip_all966.csv.

## 110. [MEDIUM] A third adaptation arm exists and was never used
- `omni_final/abl2_pitt_both_oof.csv` + `abl_pitt_both.json`: LoRA PLUS projector on the same Pitt 468, 9,637,376 trainable params.
- Recomputed: OOF 0.7696 [0.7135, 0.8253], conflict 0.5710, agreement 0.8443. The LARGER budget is WORSE than LoRA alone (0.8250). This strengthens the paper's argument and should be in the ablation table.
- Separately, that file pair disagrees with itself: `abl_pitt_both.json` claims auc_oof 0.7872 while its csv recomputes to 0.7696.

Status now: released here. Files: lookup/master_lookup.csv, lookup/SUPERSEDED.csv. The lookup records LoRA plus projector 0.7696, and SUPERSEDED.csv lists the 0.7872 claim as not used. The paper has no ablation table. Its LoRA sentence ends "and training both together gives 0.77", which is 0.7696.

## 111. [MEDIUM] 1d changes two things at once, not one
- Besides swapping the label to PHQ-8 >= 10, the rebuild drops the original conflict severity gate (sev>=15 for positives, sev<=4 for negatives) entirely. That is why n nearly doubles from 483 to 924 pairs.
- 1d is therefore NOT a like-for-like severity-matched comparison against Table 2.

Status now: fixed in the paper. Quote: "Replacing the severity gates of 15 and 4 with a PHQ-8 cut at 10 gives 924 pairs from 237 speakers, on which the conflict arm scores 0.18." The sentence now names the PHQ-8 cut at 10, says the gates were replaced, and counts pairs. Evidence: release/edaic_rerun/part16/POD1/p14_phq10_omni.csv (authors' machine path).

## 112. [MEDIUM] E-DAIC's released binary label is not the standard PHQ-8 cutoff
- 20 of 275 speakers carry y=0 despite PHQ-8 >= 10.

Status now: fixed in the paper. Quote: "We use E-DAIC \cite{daic}, the extended dataset released for AVEC 2019, with 275 participants, where the binary label is the dataset's released label based on the eight-item Patient Health Questionnaire (PHQ-8) \cite{Kroenke2009}, stricter than PHQ-8 $\geq 10$ for 20 participants, rather than a clinical diagnosis."

## 113. [HIGH, CORRECTS AN EARLIER CONCLUSION] The ADReSSo released segmentation DOES exist
- `PART5_interviewer_measure.txt` section 5.2 declares ADReSSo "NOT MEASURABLE, no timed transcript exists". That was concluded from a folder listing. The four .tgz archives were never opened.
- `tar -tzf ADReSSo21-diagnosis-train.tgz` lists `ADReSSo21/diagnosis/train/segmentation/{ad,cn}/adrsoNNN.csv`, the genuine released speaker-diarised segment lists, columns `"",speaker,begin,end`, speaker in {PAR,INV}, milliseconds. 166 files, 2268 PAR rows, 880 INV rows, 154/166 carry INV.
- Independently re-derived by the verifier from the raw files: 0/166 mismatches. No diariser was substituted anywhere.
- Coverage: 166 of 237 scored clips resolve. The other 71 are the diagnosis TEST set, re-anonymised as `adrsdt*` with no public id map. 5 more carry an INV tier only. ADReSSo analysis therefore runs on n=161.

Status now: released here. Files: scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json. reports/PART5_interviewer_measure.txt section 5.2 is the superseded statement.

## 114. [MEDIUM] POD4B quoted the wrong Pitt comparator in its matched-duration sentence
- It wrote "at matched duration ... against Pitt -0.0452". 0.0452 is Pitt's A->B (unmatched) delta. Pitt's matched-duration C->B is -0.0483 [-0.0907, -0.0062].
- The correct like-for-like line is: ADReSS-2020 -0.0434, ADReSSo -0.0485, Pitt -0.0483. The conclusion is unchanged and in fact tighter.

Status now: released here. Files: scores/part16/POD4B/boot_adresso.json, scores/part16/POD4B/boot_adress2020.json, scores/part16/POD4/boot4b.json.

## 115. [MEDIUM] `window_seconds` in the sidecars is hardcoded, not derived, do not cite it
- `sft_mid300_sft.json` says `window_seconds 30` despite `clip_list mf_mid300.csv`. All three Kimi sidecars (mid30, mid300, full) say 30. `sft_full_sft.json` says 30 while its OOF paths are `/workspace/data/full/*.wav`.
- Any argument that leans on `window_seconds 0.0` meaning "no truncation" is unsound. A reviewer checking `sft_full_sft.json` would see `window_seconds 30` on the projector arm of the headline row.

Status now: released here. Files: lookup/master_lookup.csv, scores/edaic_windows/sft_full_sft.json, scores/edaic_windows/sft_mid300_sft.json. The lookup notes the stale window_seconds field on the projector fine tune rows. The sidecars are released as written.

Update, 23 Sep 2026. For `sft_full_sft.json` the field is right, not stale. `scripts/core/sft_projector.py` cuts every clip to its first 30 s before the processor (line 59, `x = x[:16000 * 30]`) and writes 30 into the sidecar (line 102). The sft_full run log ends with that script's result line, AUC 0.6421. So the Table 1 E-DAIC fine-tune heard only the first 30 s of each window. PART 23 removed the cut: 0.8150 on the first 300 s and 0.7931 on the whole window (README, section d). The mid30 and mid300 fine-tune sidecars come with the same field. Their logs were not checked, so what they heard is not confirmed. The note on the projector fine tune rows of `lookup/master_lookup.csv` in this repository predates this update.

## 116. [MEDIUM] ADReSSo's published encoder probe is the wrong baseline for the 4b deltas
- Published 0.9046 was fitted over all 237 clips and merely EVALUATED on the 161 resolvable ones. POD4B's arm A on those 161 is 0.8624. The zero-shot streams agree (0.6320 vs 0.6291), so this is training-set size, not audio.
- Arms A/B/C are internally consistent (one fold assignment, one clip set), but the published number must not be used as the comparator for these deltas.

Status now: released here. Files: scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json. Arm A is the comparator for these deltas.

## 117. [LOW] POD4B arm A is not bit-identical to the paper's cut
- Per-clip p_yes vs published: Pearson r 0.9888 (ADReSS-2020), 0.9939 (ADReSSo). 83/156 and 74/161 clips differ by more than 1e-2, bimodally (about half agree to <1e-4, half differ by ~1e-2). Not bf16 jitter. Source wavs were not synced so the cause is unsettled. The A/B/C deltas are unaffected.

Status now: released here. Files: scores/part16/POD4B/p16_adresso_orig_zeroshot_scores.csv, scores/part16/POD4B/p16_adress2020_orig_zeroshot_scores.csv. The cause stays unsettled because the source wavs were not synced.

## 118. [HIGH] [PART17] T1: every E-DAIC Table 2 agreement cell changes at 2 dp on the 311 distinct segments
- The 483 agreement rows hold only 311 distinct seg_uids (repeat distribution 1x:208, 2x:63, 3x:21, 4x:12, 5x:4, 6x:3, 172 extra rows). Repeats carry identical p_yes in all 8 streams (max abs diff 0.0). Conflict arm: 483 rows = 483 distinct, unaffected.
- Live tex (pasted copy, 2026-09-23T05:58Z) prints agreement 0.95 / 0.95 / 0.96 / 0.61 / 0.81 / 0.87 (Qwen2-Audio / Qwen2.5-Omni / Qwen3-Omni / AF2 / AF3 / Kimi). On 311 distinct segments: 0.9608 / 0.9647 / 0.9727 / 0.6297 / 0.8425 / 0.8936, i.e. 0.96 / 0.96 / 0.97 / 0.63 / 0.84 / 0.89. All six change.
- Transcript cells (not printed in live Table 2): Qwen2.5-Omni 0.9596 -> 0.9620 (0.96 both), Qwen2-Audio 0.9081 -> 0.9199 (0.91 -> 0.92).
- Text claims still hold on distinct segments: every model except AF2 above 0.80 on agreement. Paired conflict-minus-agreement intervals exclude zero for all six (AF2 -0.1011 [-0.1647, -0.0384]).
- Pitt agreement arm has 322 rows = 322 distinct segment_path, so the Pitt columns of Table 2 are not affected by this issue.
- Files: release/edaic_rerun/part17/T1_table2_distinct.csv, T1_perclip_distinct_agreement.csv, T1_table2_distinct.json.

Status now: fixed in the paper. The final tex prints the 311 distinct segment values 0.96, 0.96, 0.97, 0.63, 0.84 and 0.89 in Table 2. Quote (caption): "Depression, E-DAIC \cite{daic}: 483 conflict segments paired with agreement segments of the same speakers (311 distinct), 138 speakers." Files: scores/part17/T1_table2_distinct.csv, scores/part17/T1_perclip_distinct_agreement.csv, scores/part17/T1_perclip_all966.csv.

## 119. [LOW] [PART17] T3 states path moved (2026-09-23 03:57)
- The old drive path of `probe2/pitt_states.npz` no longer resolves, because the `paper1_local_runs` folder moved into another folder on the same drive (gdrive/ in this file). Same file: sha256 of first 1 MB `5d8272dc673cdbded7c0a84207d5306ffdd30b6d1fee6a99b549b05d7087494c` equals the Part 16 M5 sidecar value. Every Part 16 sidecar that cites the old drive path now points at a dead path.

Status now: released here. Files: scores/part17/T3_q2a_direction.json. The sidecar records a hash of the first 1 MB of the states file. Paths in sidecars are authors' machine paths.

## 120. [LOW] [PART17] T3 first-pass random floor was NOT an advancing generator (2026-09-23 03:57)
- The Part 17 plan says the first pass drew its floor from an advancing generator. Its `rand_floor` creates `default_rng(0)` fresh on every call, and the rng.normal-in-a-loop stream equals the standard_normal block stream bit for bit (checked here: True). Its three floors differ because it measured |cos| against each cell's own PROBE direction (three different probes), while the verifier measured against the READOUT direction d (three near-identical vectors). Both are valid floors: vs d 0.0123 [0.0005, 0.0321], vs probe direction 0.0124 [0.0005, 0.0343]. Analytic E|cos| in 4096 dims = 0.0125. The part10 canonical floor (Omni) is vs d, 1000 draws. The q3o floor is vs the probe direction, 2000 draws. The three backbones' floors are therefore not computed against the same reference vector. The difference is immaterial (all sit at sqrt(2/(pi*dim))).

Status now: released here. Files: scores/part17/T3_q2a_direction.csv, release/edaic_rerun/part17/rows/T3.tsv (authors' machine path).

## 121. [MEDIUM] [PART17] T3 'nested probe direction' vs the verifier's run (2026-09-23 03:57)
- The authors' wording asks for the cosine with the NESTED probe direction 'from the verifier's corrected run'. The verifier's run (and Omni part10, and q3o) uses the FINAL-STAGE probe on ans[:, -1, :], not the nested probe. The nested answer probe picks layers [23, 22, 23, 16, 23] (none is the final stage 32). Both are reported in part17/T3_q2a_direction_summary.csv: final-stage cos 0.0145 (headline, comparable across the three backbones) and nested cos 0.0256 (supplement, d is the final-stage readout direction applied at mid layers, a logit-lens reading). The paper text must say which one it quotes.

Status now: released here. Files: scores/part17/T3_q2a_direction_summary.csv, release/edaic_rerun/part17/rows/T3.tsv (authors' machine path). The rows give the final-stage cosine 0.0145 and the nested cosine 0.0256. The final tex quotes the direction test for Qwen2-Audio and Qwen3-Omni without a cosine: "On Qwen2-Audio and Qwen3-Omni the readout direction gives 0.62 and 0.77 and the probe 0.82 and 0.83, and 0.81 and 0.83 with the direction removed."

## 122. [LOW] [PART17] T3 direction weighting differs between first pass and verifier (2026-09-23 03:57)
- First pass and q3o weight the Yes/No rows by FULL-vocabulary softmax mass. The verifier and part10 (Omni) use the softmax over the Yes/No logits only. For Qwen2-Audio Pitt the two unit directions agree to cos 1.0000000000. The reported 4-decimal values are identical.

Status now: released here. Files: scores/part17/T3_q2a_direction.csv.

## 123. [MEDIUM] [PART17] T3 One Pitt participant carries two speaker ids in every Pitt states file grouping (2026-09-23)
- `pitt_states.npz` (and every Pitt probe grouped on its `spk` column, i.e. all 228-speaker Pitt probes: Qwen2-Audio, Qwen2.5-Omni part10, Qwen3-Omni) splits one manifest participant into two speaker labels, one in each diagnosis group (3 clips) [participant id withheld here]. The true participant count is 227, not 228. In the GroupKFold(5) used for T3 the two ids land in folds 1 and 3, so one person (3 clips) is on both sides of a train/test split.
- Sensitivity, Qwen2-Audio readout test, grouped by the manifest participant id (227 groups, changes the whole GroupKFold partition, not only this person): all probe 0.8273, all probe minus d 0.8200, conflict minus d 0.6607, agreement minus d 0.8838, paired -0.2230 [-0.3304, -0.1192]. File: part17/T3_selfcheck_participant_grouped.csv.
- Most of that movement is the partition, not the leak: over 50 random participant-grouped partitions (GroupKFold(5, shuffle=True, random_state=0..49)) conflict minus d has median 0.6095, 2.5-97.5% [0.5749, 0.6593]. All probe median 0.8112. Paired median -0.2724 [-0.3020, -0.2247]. The canonical values (0.8223, 0.6013, -0.2884) sit inside those ranges. File: part17/T3_selfcheck_partitions.csv. No conclusion changes. The paper should say 227 participants for Pitt, or state that that participant is counted once per diagnosis group.

Status now: fixed in the paper. Quote: "We use the DementiaBank Pitt cookie-theft descriptions \cite{pitt}, with 468 segments from 227 participants (one has sessions in both groups and is grouped under two labels), the top and bottom third of the fluency score defined under Conflict set construction below, the ADReSSo recordings \cite{adresso} from 237 speakers, as released with the ADReSS-M challenge \cite{adressm}, and the ADReSS-2020 benchmark split \cite{adress2020} including 108 training and 48 test recordings." The files count 228 speaker labels, because one participant appears under two labels. Files: folds/README.md, release/edaic_rerun/part17/T3_selfcheck_participant_grouped.csv (authors' machine path), release/edaic_rerun/part17/T3_selfcheck_partitions.csv (authors' machine path).

- 124. [PART17] [MEDIUM] T5 (2026-09-23 03:59): make_fig1_v5.py does not exist on disk (Desktop, release, paper1_final, public repo clones searched). The newest figure script is edaic_rerun/make_fig1_v4.py. It wrote paper1_final/figures/fig1_gap.pdf (E-DAIC variant full, answer 0.8285) at 22:25 on 22 Sep, and fig1_gray_v5-1.png is the grayscale render of that output. Which E-DAIC variant sits in Overleaf cannot be checked from disk. T5 used make_fig1_v4.py.

  Status now: released here. Files: figures/fig1_gap.pdf, scripts/make_fig1_v4.py. Per the authors, figures/fig1_gap.pdf is byte identical to the Overleaf figure and comes from make_fig1_v4.py. "v5" was the fifth render of that script.

## 125. [HIGH][PART17] G-Drive paths relocated on 2026-09-23, every `gdrive/...` path is now stale
- The folder moved to `gdrive/` (the authors moved it for Mac disk space). exFAT has no symlinks, so the old path cannot be aliased.
- Affects source paths recorded in `master/master_lookup.csv`, `overnight/perclip/INDEX.csv`, and many sidecars written before today. The data is intact at the new location. Only the recorded path is out of date.
- `DementiaBank/` did NOT move.
- A batch folder on the authors' drive (`ashutosh_batch2/`) holds runs made by the TA on a different computer. Not used in the paper, never to be mixed with local results.

Status now: open. 75 rows of lookup/master_lookup.csv still cite the old gdrive/ prefix (authors' machine path). What decides it: rewriting those source paths to the new prefix. Related files: lookup/master_lookup.csv.

## 126. [HIGH] [PART17] T4 E-DAIC 300 s window probe values use a different estimator from the decided Pitt convention (2026-09-23)
- master_lookup now carries, as the authors listed, Qwen2.5-Omni edaic_full LM probe 0.7088 and encoder probe 0.5916. Both are AUCs of the per-clip OOF probability AVERAGED over the five repeats (recomputed from part16/EDAICFULL/edaic_full_perclip.csv: 0.708786, 0.591634).
- For Pitt the authors decided the paper uses the MEAN OF FIVE PER-REPEAT AUCs (0.7706, not the averaged-probability 0.7917). Under that convention the same E-DAIC run gives LM 0.7001 and encoder 0.5884 (per_repeat lists in edaic_full_nested_repeats.json). Both are now in the lookup as separate rows: streams "LM probe (mean of 5 per-repeat AUCs)" and "encoder probe (mean of 5 per-repeat AUCs)".
- The superseded 0.6853 was itself a mean of five (original run). So 0.6853 -> 0.7088 changes the run AND the estimator. Like for like (mean of five): LM 0.6853 -> 0.7001, encoder 0.5930 -> 0.5884.
- The mean-of-five values cannot be recomputed from per-clip scores. edaic_full_perclip.csv holds only the five-repeat averaged oof_* columns. probe_boot.py saved the per-repeat OOF as edaic_full_oof_repeats.npz on the pod. That file is not on the authors' machine, the authors' drive or any temporary working folder. So no mean-of-five paired probe-minus-answer interval can be computed. The averaged-probability intervals reproduce exactly: LM minus answer -0.1190 [-0.1995, -0.0478], encoder minus answer -0.2361 [-0.3272, -0.1394].
- The edaic_full encoder row was updated in place (0.5930 -> 0.5916) together with the LM row, because the list gives both from the same run. Revert that one row from master_lookup.csv.bak_pre_part17 if only the LM row was meant.
- ACTION: the authors pick one convention for E-DAIC and Pitt before either number is cited. If the pod still has edaic_full_oof_repeats.npz, copying it back allows the mean-of-five interval.
- Files: master/master_lookup.csv, edaic_rerun/part17/T4_lookup.json, edaic_rerun/part17/rows/T4.tsv.

Status now: fixed in the paper. Quote: "On E-DAIC the language model probe reaches 0.74 and the encoder probe 0.60, both below the answer of 0.84 ($-$0.09 [$-$0.16, $-$0.03] and $-$0.24 [$-$0.32, $-$0.15]), while a probe on the answer state matches the answer (0.84)." The final tex uses the whole-interview run with the processor cut lifted (PART 23), under the decided mean-of-five estimator: language model probe 0.7437 [0.6752, 0.8097], encoder probe 0.5981 [0.5267, 0.6672], paired minus the answer 0.8366, -0.0929 [-0.1609, -0.0257] and -0.2385 [-0.3244, -0.1505], answer-state probe 0.8359 [0.7819, 0.8832] (part23/verify/PART23_VERIFIED.tsv, authors' machine path). The first-300 s values under the same estimator, 0.7001 and 0.5884 (scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv), are the earlier window. The superseded 0.6853 is gone from the tex.

## 127. [MEDIUM] [PART17] T4 master_lookup builder cannot reproduce the lookup, left untouched (2026-09-23)
- No build_master*.py exists under release/. The only copy found is build_master.py in a temporary folder on the authors' machine. It opens release/master/master_lookup.csv for writing directly.
- Run only as a COPY writing to a temp path. With its own (now stale) G-Drive prefix it makes 101 rows: drops 120 of the 221 pre-Part-17 rows. With the current prefix it makes 178 rows: drops the 43 appended rows (edaic_mid30, edaic_mid300, edaic_full, pitt_all, edaic_conflict_v2) and rewrites 76 source prefixes. Against the edited 231-row file it drops 53 rows.
- Its overrides are one hard-coded block (Omni Pitt encoder 0.7706, two edaic_conflict rows), then a sort and first-wins de-duplication. Making it reproduce the file would mean pasting the whole file into it. Left alone. Part 17 changes live only in the csv (backup: master/master_lookup.csv.bak_pre_part17). Do not run that script.

Status now: released here. Files: lookup/master_lookup.csv. The lookup is the maintained table. The builder script is not released and must not be run.

## 128. [MEDIUM] [PART17] T4 Qwen3-Omni Pitt rows now come from two runs (2026-09-23)
- Encoder probe is now 0.8122 (POD3 re-extraction, per-clip file). LM 0.8477, answer-state 0.8268 and projector 0.7619 still come from overnight2/final/q3o_pitt_nested_repeats.json (older run, no per-clip file). The POD3 run gives 0.8445, 0.8237 and 0.7820 for those (part16/pod_sync/q3o_pitt_nested_repeats.json). Not changed, no decision covers them.

Status now: released here. Files: lookup/master_lookup.csv, lookup/SUPERSEDED.csv. Each Qwen3-Omni Pitt row names the run it comes from.

## 129. [MEDIUM] [PART17] T4 original-run edaic_full rows still say "capped at 900 s" (2026-09-23)
- Qwen2.5-Omni edaic_full zero shot 0.8285, answer-state 0.7542, projector 0.5293 and projector fine tune 0.6421 (and the other models' edaic_full zero-shot rows) describe the window as full audio capped at 900 s. For Qwen2.5-Omni the processor only sees the first 300 s (see the POD2 EDAICFULL HIGH entry). Only the two rows T4 changed now say "300 s window". The rest were not rewritten.

Status now: released here. Rows 200, 202, 204 and 205 of lookup/master_lookup.csv (zero shot 0.8285, projector 0.5293, answer-state 0.7542 and projector fine tune 0.6421) carry the authors' notes: the first 300 s is what the model heard in those runs, and the fine-tune heard only the first 30 s (item 100). The final tex uses the runs with the processor cut lifted instead. Rows 236 to 244 hold them: the PART 23 zero-shot answer 0.8366 and the four mean-of-five probes, the PART 23 fine-tunes 0.7931 (whole window) and 0.8150 (first 300 s), and the PART 24 fine-tunes 0.8589 (seed 0, the Table 1 cell) and 0.8184 (seed 1). The other models' edaic_full rows are unchanged. Related files: lookup/master_lookup.csv.

## 130. [LOW] [PART17] T4 notes (2026-09-23)
- AF2 Pitt agreement: the decision text typed 0.5536. af2_pitt468.csv gives 0.554442 with zero tied scores, so no tie rule gives 0.5536. 0.5544 recorded.
- abl_pitt_both.json (0.7872) has no per-clip file. abl2_pitt_both_oof.csv pairs with abl2_pitt_both.json (0.7696) and recomputes to 0.769585. 0.7696 recorded.
- Qwen3-Omni PC-GITA encoder: POD3 per-repeat AUCs [0.9223, 0.8697, 0.8742, 0.8988, 0.9196] differ from the old json [0.9235, 0.8703, 0.8745, 0.8976, 0.9187], but both means are 0.8969. Same convention as Pitt 0.8122 (mean of five). Averaged-probability AUCs, not used: PC-GITA 0.9109, Pitt 0.8354.
- Qwen3-Omni arm intervals depend on the speaker label strings. Manifest keys (zero-padded ids) give conflict [0.5326, 0.7451]. The npz keys (unpadded ids) give [0.5323, 0.7418], as in Q3Ofix. Same partition, different sort order, same point values.
- The one older [SUPERSEDES] note (Omni Pitt encoder) sits in the source column. T4 notes are in the estimator column.
- 75 lookup rows still carry gdrive/ (76 before, T4 rewrote only the AF2 Pitt row it had to touch).
- Two Pitt LoRA runs exist: POD2 lora_pitt_oof.csv 0.8250 (recorded) and omni_final/abl_pitt_oof.csv 0.8202 (not recorded, not marked superseded, no decision).
- release_public/lookup/master_lookup.csv (178 rows) was not touched.

Status now: released here. Files: lookup/master_lookup.csv, lookup/SUPERSEDED.csv. The lookup records AF2 Pitt agreement 0.5544, LoRA plus projector 0.7696 and LoRA 0.8250.

## 131. [HIGH] [PART17] T6: the Pitt transcript by-arm sentence used the Sep 15 run, Table 1 prints the Sep 20 run (2026-09-23)
- The Table 1 Pitt transcript cell (0.66) is the Sep 20 run, overnight2/text_new/o25_pitt_text.csv, recomputed at 0.6629 [0.5983, 0.7245]. The by-arm numbers used so far (conflict 0.4550, agreement 0.7987, gap -0.3437) are from the Sep 15 run, paper work/paper1_local_runs/omni_pitt_text.csv (0.6975).
- Sep 20 run by arm (speaker bootstrap, 2000 draws, rng(0) per cell): conflict 0.5560 [0.4516, 0.6680] (146 clips, 100 spk), agreement 0.7078 [0.6381, 0.7696] (322, 175), paired conflict minus agreement -0.1518 [-0.2665, -0.0271] (468, 228).
- What changes in the sentence: (a) the conflict arm is no longer below chance at the point estimate (0.4550 -> 0.5560). The interval spans 0.5 in both runs, so "below chance" was never supported by the interval. (b) The transcript no longer beats audio on agreement: paired transcript minus audio +0.0024 [-0.0824, 0.0779] (Sep 15 was +0.0933 [0.0157, 0.1637]). (c) The transcript arm gap is no longer about twice the audio's: -0.1518 vs audio -0.1731, ratio 0.88 (Sep 15 ratio 1.98). Paired gap difference +0.0214 [-0.1413, 0.1806]. What survives: the transcript arm gap is below zero with an interval that excludes zero, as the audio's does.
- Paired Sep 20 minus Sep 15: conflict +0.1010 [0.0270, 0.1766], agreement -0.0909 [-0.1345, -0.0518], arm gap +0.1919 [0.1104, 0.2800]. The two runs disagree beyond resampling noise.
- Files: part17/T6_pitt_text_sep20_arms.csv, T6_pitt_text_sep20_arms.json, rows/T6.tsv.

Status now: fixed in the paper. The final tex prints the Sep 20 run in Table 1, row verbatim: "Pitt        &  468 (227) & 0.66 & 0.66 & 0.77 \\" and a by-arm sentence: "On the Pitt dataset the transcript alone shows the same arm gap as the audio (0.56 on conflict, 0.71 on agreement)." Files: scores/part17/T6_pitt_text_sep20_arms.csv, scores/part17/T6_pitt_text_sep20_arms.json.

## 132. [MEDIUM] [PART17] T6: Sep 15 and Sep 20 Pitt transcript runs are not the same scorer (2026-09-23)
- Per-clip p_yes, Sep 20 vs Sep 15: Pearson r 0.7640, Spearman 0.7876, max abs diff 0.6147 (one clip: 0.7433 vs 0.1285), mean abs diff 0.3019, 460/468 clips differ by more than 0.1. Mean p_yes 0.4010 vs 0.0991. Median answer mass 0.9966 vs 0.9519.
- The Sep 20 prompt column is one constant string (the question clause only, the transcript wrapper is not recorded). The Sep 15 csv has no prompt column and its script is lost (per the header of paper1_local_runs/omni_text_score.py), so the Sep 15 wording cannot be checked. The cause of the difference cannot be settled from disk.

Status now: released here. Files: scores/part17/T6_pitt_text_sep20_arms.csv, scores/part17/T6_pitt_text_sep20_arms.json. The cause cannot be settled from disk.

## 133. [LOW] [PART17] T6: Pitt speaker id forms (2026-09-23)
- Sep 20 speaker_id (zero-padded) equals manifest grp + spk on 468/468 rows, and equals the Sep 15 and audio speaker columns on 468/468. Join was on clip basename. Labels agree on 468/468. Manifest arm equals the Sep 15 'set' column on 468/468. The id form does not affect arm assignment.
- The participant listed twice is the one bare spk number in both groups (see the T3 entry). Clustering on the 227 bare manifest ids instead of 228 speaker ids: conflict [0.4485, 0.6723], agreement [0.6373, 0.7673], paired gap [-0.2697, -0.0275]. Points unchanged, no conclusion changes.

Status now: released here. Files: scores/part17/T6_pitt_text_sep20_arms.json, folds/README.md. The sidecar holds the 227-participant clustering check.

## 134. [HIGH] [PART17] T7: the E-DAIC 300 s window mean-of-five probe values (LM 0.7001, encoder 0.5884) cannot get their own intervals on this Mac (2026-09-23)
- Reproduction gate FAILED. The nested probe from part16/EDAICFULL/probe_boot.py was rerun on the preserved states (`gdrive/part16_EDAICFULL_states/edaicfull_shards.tgz`, 275 shards, same clip order, same folds, same probe) on three Mac stacks: sklearn 1.7.2 + numpy 2.2.6 Accelerate, sklearn 1.8.0 + numpy 2.4.4 Accelerate, sklearn 1.6.1 + numpy 1.26.4 OpenBLAS. None matches the pod's per-repeat AUCs at 4 dp. The pod ran on x86_64 with sklearn 1.9.1.
- LM per repeat, Mac (sklearn 1.7.2) vs pod: 0.6915/0.6918, 0.6977/0.7062, 0.6887/0.6893, 0.7057/0.7055, 0.7072/0.7078. Mean 0.6982 vs 0.7001. Layer picks match 24 of 25. The one that differs is repeat 1 outer fold 1: Mac picks layer 14 (inner AUC 0.80183), the pod picked layer 18 (0.80137), a gap of 0.00046.
- Encoder per repeat: 0.5963/0.5964, 0.5826/0.5826, 0.5966/0.5962, 0.5610/0.5608, 0.6061/0.6060. Mean 0.5885 vs 0.5884. Layer picks match 25 of 25.
- Answer state: repeat 0 outer fold 1 flips too (Mac layer 19, 0.82740, pod layer 28, 0.82631). All three Mac stacks flip the same way.
- The folds are not the cause. GroupKFold(shuffle=True) takes the rng.permutation branch and never reaches the argsort of group sizes (checked in local sklearn 1.7.2 and 1.8.0, and in the 1.9.1 tag on GitHub), so the stable tie-break emulation does not apply. The folds are identical across the three Mac stacks (5 x 55 per repeat). The cause is the lbfgs stopping point (tol 1e-4 on 220 x 3584 data). Even with the pod's own layer picks forced, per-repeat AUCs still differ by up to 0.0007 (LM) and 0.0004 (encoder). The per-clip mean OOF correlates r 0.99993 (LM) and 0.99999 (encoder) with the published file.
- On the pod, probe_boot.py DID save the per-repeat vectors to `/workspace/edaicfull/out/edaic_full_oof_repeats.npz`. That file is not in the tgz. It was not found on the authors' machine or in the searched folders of the authors' drive. Some other drive folders were too slow to search fully. If nobody pulled it, it was lost with the pod.
- Closest proxy (Mac, pod layer picks, NOT the paper's vectors): LM mean of five 0.6997 [0.6229, 0.7722], paired vs answer -0.1280 [-0.1999, -0.0632]. Encoder 0.5885 [0.5268, 0.6525], paired -0.2392 [-0.3186, -0.1572]. Both exclude zero. Mac-selected layers give LM 0.6982, paired -0.1296 [-0.2019, -0.0630], and the same encoder values.
- The not-used rows reproduce exactly from edaic_full_perclip.csv with the same bootstrap: 0.7088 [0.6232, 0.7874], paired -0.1190 [-0.1995, -0.0478]. 0.5916 [0.5110, 0.6706], paired -0.2361 [-0.3272, -0.1394].
- Every value is the 300 s window. The processor cuts audio at 300 s and 217 of 275 windows are truncated.
- Files: part17/T7_edaic300_meanof5.csv, T7_edaic300_meanof5.json, T7_edaic300_meanof5_perclip.csv (pod picks), T7_edaic300_meanof5_perclip_macselected.csv, rows/T7.tsv, T7_scratch/T7_gate_*.json.

Status now: released here. Files: scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv, scores/part17/T7b/T7b_edaic300_meanof5.csv, release/edaic_rerun/part17/rows/T7b.tsv (authors' machine path), lookup/master_lookup.csv. The authors' rule for E-DAIC is the mean of five per-repeat AUCs, as for Pitt: LM probe 0.7001 [0.6231, 0.7727], encoder probe 0.5884 [0.5268, 0.6523], and paired minus the zero-shot answer 0.8278, LM -0.1276 [-0.1993, -0.0629] and encoder -0.2394 [-0.3188, -0.1572]. The averaged-probability values 0.7088 and 0.5916 are listed as not used. The per-repeat per-clip scores were reproduced on Linux x86 (T7b).

## 135. [HIGH] [PART17] T2: every dataset has two GroupKFold(5) splits, a Mac one and a pod one, and the Table 1 Qwen2.5-Omni Pitt encoder cell sits on the Mac split (2026-09-23)
- sklearn GroupKFold without shuffle orders speakers by clip count with np.argsort. Speakers with equal counts stay in a platform-dependent order, and that order decides the split. On ADReSSo (237/237), ADReSS-2020 (156/156), MDVR-KCL (37/37) and E-DAIC (275/275) every speaker ties. On Pitt, PC-GITA and NeuroVoz most do.
- Mac-run outputs reproduce only with this Mac's order (numpy 2.2.6, arm64, default argsort): Qwen2-Audio `probe2/<ds>_{enc,llm,ans}_nested_oof.csv` for all seven datasets (max abs diff 1.1e-16, saved layers re-derived identically in 21 of 21 streams), the eGeMAPS baseline `overnight/perclip/egemaps_<ds>_oof.csv` (7 of 7, 1.1e-16), and `overnight2/part10/pitt_enc_nested5_oof.npz` (5 of 5 seeds, max abs diff 0.0). The pod order misses them by 0.55 to 1.00.
- Pod-run outputs reproduce only with a stable tie-break (np.argsort(kind='stable'), reversed as GroupKFold does): Qwen2.5-Omni `omni_final` Pitt and E-DAIC, Qwen3-Omni `overnight2/q3o_new` ADReSS-2020, ADReSSo, MDVR-KCL, NeuroVoz, PC-GITA, Qwen2-Audio `probe2/kcl_local` (30 run-streams, max abs diff 1.4e-06 to 0.061, saved layers re-derived in 29 of 30), the part16 POD3 Qwen3-Omni repeats, and the two fold files written on pods by GroupKFold itself (`part16/POD2/pitt468_folds.csv` 468/468 rows, `part16/POD1/sft1c_folds.json` 275/275). The Mac order misses them by 0.67 to 1.00. Where only AUCs were saved, they point the same way: 35 pod per-layer AUC curves (Qwen2-Audio, Qwen2.5-Omni, Kimi-Audio) agree with the stable split within 0.006 at every layer, and 28 five-repeat run-streams (every Table 1 probe stream on disk: Qwen2-Audio, Qwen2.5-Omni Pitt and E-DAIC, Qwen3-Omni five datasets, Kimi-Audio seven) reproduce their saved per-repeat AUCs with median |diff| 0.0001, 94% within 0.002, max 0.008. In every curve and every seed-0 repeat the stable split is closer than the Mac one.
- Consequence: Table 1 Qwen2.5-Omni Pitt encoder 0.7706 (part10 nested5, computed on this Mac) is on Mac folds. The superseded 0.7761 (`omni_final/omni_pitt_nested_repeats.json`) is the same five-repeat nested probe on the same states (part10 o25 states, zero-shot p_yes equal to omni_final to 1e-16) computed on a pod. Refit with the pod order: [0.7920, 0.7936, 0.7653, 0.7779, 0.7515] vs saved [0.7921, 0.7937, 0.7652, 0.7778, 0.7518]. With the Mac order seed 0 gives 0.7598, which is nested5 seed 0 exactly. So 0.7706 versus 0.7761 is a fold-assignment difference, not the estimator difference recorded in PART16 M7 and release_public/README.md. Every other Table 1 probe cell is a pod five-repeat mean on pod folds, so this is the one Table 1 probe cell on different folds from the rest.
- Also on Mac folds: the eGeMAPS baseline, so the deltas in `overnight/egemaps_baseline.csv` (eGeMAPS minus Omni encoder probe) compare different splits. And the Qwen2-Audio single-split `probe2/*_nested.json` values.
- The earlier verifier's failure to reproduce the Omni Pitt and E-DAIC per-layer curves here (0 of 32 layers at 4 dp) is explained: with the stable order the encoder curves agree to max |dAUC| 5.6e-04 (Pitt) and 8.7e-04 (E-DAIC). With the Mac order they are off by 0.053 and 0.088.
- Files: release/folds/ (64 csv + README.md), release/edaic_rerun/part17/T2_folds.json, part17/T2/.

Status now: released here. Files: folds/README.md, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/pitt_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/pitt_groupkfold5_pod_seed4_UNVERIFIED.csv, scores/part10/pitt_enc_nested5_oof.npz, release/edaic_rerun/part17/verify/T2_verify_omni_pitt_repeats.json (authors' machine path). The authors' rule: 0.7706 is on the Mac folds, which reproduce it exactly. The older 0.7761 is the same probe on the pod split, and the difference 0.0055 [-0.0083, 0.0210] is within resampling noise.

## 136. [MEDIUM] [PART17] T2: probe2/README_probe2.txt misstates where three probe2 outputs were made (2026-09-23)
- It says E-DAIC and NeuroVoz ran on the pod. The `probe2/edaic_*` and `probe2/neurovoz_*` nested files reproduce exactly on this Mac with the Mac order and the on-disk states (1.1e-16, layers identical): they are Mac re-runs. The pod versions are in `probe2/pod_runs/`, fitted on states the pod extracted itself. Those states are not on disk (pod_runs zero-shot p_yes differs from the on-disk states by up to 0.0147 E-DAIC, 0.0163 NeuroVoz), so pod_runs cannot be refitted exactly. Nearest split is the pod one (max abs diff 0.065 to 0.25, layer choices differ in some folds).
- `probe2/pod_runs/pitt_enc_nested_oof.csv` is byte-identical to `probe2/pitt_enc_nested_oof.csv` (sha256 8ca5068f249e3a274edd11fc9f389285722c6dad135cb75a992235a7df3acc33) and reproduces only with the Mac order, so it is a copy of the Mac output, not a pod result. README_probe2's Table 1 Pitt probe 0.7742 is in no saved file. Both Pitt enc files give 0.8040.
- README_probe2's E-DAIC 0.6241 and NeuroVoz 0.9296 are the pod_runs values. The probe2/ root files now give 0.5937 and 0.9383 (different states and different folds). All of these single-split values are superseded by the five-repeat means.

Status now: released here. Files: folds/README.md. The fold README records which probe2 files are Mac reruns and which are pod runs.

## 137. [MEDIUM] [PART17] T2: the part16 POD3 Qwen3-Omni Pitt states write unpadded speaker ids, which changes every GroupKFold split (2026-09-23)
- POD3 (and `part16/pod_sync/q3o_pitt_states.npz`, same sha256) use unpadded ids, while every other Pitt run uses zero-padded ids. np.unique sorts these differently, so the seeded renumbering and the tie order give different splits: POD3's Pitt repeats (`pitt_groupkfold5_pod_unpaddedids_seed0..4.csv`, confirmed per clip) are not the Table 1 Pitt splits (`pitt_groupkfold5_pod_seed0..4_UNVERIFIED.csv`).
- The POD3 Pitt states are also a different extraction from the Qwen3-Omni Pitt runs behind Table 1 (`overnight2/final`, part10) and `q3o_new` (zero-shot p_yes differs by up to 0.381). Those Qwen3-Omni Pitt states, and all Qwen3-Omni E-DAIC states, are not on disk, so the Qwen3-Omni Pitt and E-DAIC Table 1 splits are inferred, not refitted.

Status now: released here. Files: folds/pitt_groupkfold5_pod_unpaddedids_seed0.csv to folds/pitt_groupkfold5_pod_unpaddedids_seed4.csv, folds/README.md.

## 138. [LOW] [PART17] T2: pod-run refits on this Mac cannot reach 1e-6 (2026-09-23)
- On identical states and folds, pod predictions differ from Mac refits by up to 0.061 (single split) and 0.083 (POD3 repeats). LogisticRegression (lbfgs, tol 1e-4) converges in 15 to 163 iterations here, never at the 2000 cap. The gap is where each platform's lbfgs stops inside tolerance. Casting the same features to float64 on the Mac alone moves predictions by up to 0.038. Near-tied inner layer scores flip in a few folds (Qwen3-Omni PC-GITA ans 4 vs 3, POD3 PC-GITA llm and ans seeds 0 and 4, E-DAIC full llm and ans), moving a per-repeat AUC by up to 0.008 (0.026 on the near-chance E-DAIC 390-row conflict set, not released).

Status now: released here. Files: folds/README.md.

## 139. [LOW] [PART17] T2: splits inferred, not refitted, where states are missing (2026-09-23)
- Qwen2.5-Omni states exist only for Pitt and E-DAIC (part10 o25). Its ADReSSo, ADReSS-2020, PC-GITA, NeuroVoz and MDVR-KCL probes ran on the same pod with the same speaker labels, so they are taken to use the pod split, but this is not refitted.
- Five-repeat probes have per-clip OOF only for Qwen3-Omni POD3 (Pitt with unpadded ids, PC-GITA) and the Mac-run Omni Pitt encoder. All other five-repeat splits are checked on saved per-repeat AUCs only and are released as `*_UNVERIFIED.csv`.
- Projector fine-tunes (`sft_projector.py`, omnisft, q3osft, Qwen2-Audio sft) saved no fold record and cannot be refitted. Their splits are not released.

Status now: released here. Files: folds/README.md. The _UNVERIFIED fold files carry that status in their names.

## 140. [LOW] [PART20] POD3c: the Qwen2.5-Omni transcript check matches p14_o25_text.csv on 4 of the first 5 clips, not 5 of 5 (2026-09-23)
- Check: `score_text_p20.py` with Qwen/Qwen2.5-Omni-7B on the first 25 rows of `p14_text_input.csv` (same file, sha256 281805c5...83c9), compared with `part16/pod_sync/p14_o25_text.csv` using the same token set as that run.
- 13 of 25 clips match to 1e-12 (bit-identical) and 4 more within 5e-6. 8 of 25 differ by 7.8e-4 to 0.0266. In the first 5, one clip [id withheld here] differs by 0.0061.
- Every clip that differs has a short transcript (25 to 43 words). Every transcript of 46 words or more matches to 1e-12. Bit-identical rows mean the prompt, the chat template and the token set are the same. The leftover gap looks like a bf16 kernel difference at short sequence lengths between this pod (H100 NVL, transformers 5.17.0, torch 2.8.0+cu128) and whatever machine made p14_o25_text.csv. That machine is not recorded in p14_o25_text.json.
- Files: release/scores/part20/POD3c/val_o25_first25.csv, val_o25_first25_run.json.

Status now: released here. Files: release/scores/part20/POD3c/val_o25_first25.csv (authors' machine path), release/scores/part20/POD3c/val_o25_first25_run.json (authors' machine path).

## 141. [LOW] [PART20] POD3c: p15/text_score.py counts Yeah/yeah and Nope as Yes/No, which rule 3 does not (2026-09-23)
- text_score.py is the script behind p14_o25_text.csv and p14_q2a_text.csv. Its Yes set adds Yeah/ Yeah/yeah/ yeah and its No set adds " Nope". Rule 3 counts only Yes/yes/YES and No/no/NO. The audio script extract_probe_layers.py uses the rule 3 words.
- The Qwen3-Omni transcript run saved both. p14_q3o_text.csv reports rule 3. The biggest per-clip difference is 8.9e-06, but AUC moves by up to 0.0005 (conflict 0.2243 with rule 3, 0.2248 with the text_score set, agreement 483 0.9470 vs 0.9471, agreement 311 0.9523 vs 0.9522). The cause is 4 conflict p_yes values above 0.9999 that swap order.
- On the Qwen2.5-Omni check clips the two sets differ by at most 1.2e-05. The Qwen2.5-Omni and Qwen2-Audio transcript AUCs already on disk use the wider set. They were not rescored here.

Status now: released here. Files: scores/part20/POD3c/p14_q3o_text.csv, release/scores/part20/rows/POD3c.tsv (authors' machine path). Both token sets are saved per clip.

## 142. [MEDIUM] [PART20] POD4b: the 10-window Qwen3-Omni pipeline check misses the 1e-3 bar on 6 of 10 windows (2026-09-23)
- Check: `score_q3o_noinv.py` (scoring path copied from part16/POD3/rescore_probe.py) on 10 original Pitt windows from DementiaBank/segments/, compared with `part16/pod_sync/q3o_pitt_zeroshot_scores.csv` (the 0.7673 rerun).
- 4 of 10 match within 7e-6 (three clips, and one more below 1e-5). 6 of 10 differ by 0.024 to 0.059 (worst clip: 0.6513 now, 0.5927 in the rerun) [clip names withheld here].
- Every value on both sides sits on the grid sigmoid(k/8): 0.4378 = sigmoid(-0.25), 0.4688 = sigmoid(-0.125), 0.2689 = sigmoid(-1.0), 0.3208 = sigmoid(-0.75), 0.5927 = sigmoid(0.375), 0.6513 = sigmoid(0.625). So each miss is 1 or 2 bf16 steps (0.125) in the Yes minus No logit, not a different prompt, token set or input.
- Same on both machines: transformers 5.17.0, torch 2.8.0+cu128, librosa 1.0.0, numpy 2.1.2, NVIDIA H200, sdpa attention, token ids 7414/9454/9693/9834/14004 vs 902/2152/2308/2753/8996, the same audio bytes (POD3 sidecar says md5-identical to the G-Drive segments). Feature extraction on this pod gives identical features with 192 threads and 1 thread. POD3's own runs in separate processes agreed exactly, so the gap is between machines, not between runs. Pod CPU here: Intel Xeon Platinum 8568Y+. POD3's CPU is not recorded.
- The main result does not depend on this: the paired original-minus-interviewer-free differences are reported against the shipped file, the rerun file, and (being run now) the 468 original windows re-scored on this same pod with the same code, which removes the machine term.
- Files: release/scores/part20/POD4b/q3o_val10_original_windows.csv, q3o_val10_original_windows.check.json.

Status now: released here. Files: release/scores/part20/POD4b/q3o_val10_original_windows.csv (authors' machine path), release/scores/part20/POD4b/q3o_val10_original_windows.check.json (authors' machine path). The counts are corrected in item 144.

## 143. [LOW] [PART20] POD4b: rule 3 adds " YES" (14080) and " NO" (5664), which the part16 Qwen3-Omni scripts left out (2026-09-23)
- rescore_probe.py and extract_probe_layers.py list Yes/ Yes/yes/ yes/YES (5 ids each side). Rule 3 also asks for the leading-space capitals. Both are single tokens in this tokenizer. POD4b reports the 6+6 set as p_yes and keeps the 5+5 value as p_yes_ref5.
- On the 468 interviewer-free clips the two differ by at most 4.2e-08 per clip. AUC 0.7802 either way (5+5: 0.780163).

Status now: released here. Files: release/scores/part20/POD4b/q3o_pitt_noinv_zeroshot_scores.csv (authors' machine path). p_yes and p_yes_ref5 are both saved per clip.

## 144. [LOW] [PART20] POD4b: correction to the POD4b 10-window entry above (2026-09-23)
- The counts in that entry are wrong. From q3o_val10_original_windows.csv: 3 of 10 match within 7e-6 (three clips, at 6.9e-06, 2.7e-06 and 1.4e-06) [clip names withheld here], and 7 of 10 differ by 0.024 to 0.059. Everything else in that entry stands.

Status now: released here. Files: release/scores/part20/POD4b/q3o_val10_original_windows.csv (authors' machine path).

## 145. [LOW] [PART20] POD3b: Qwen2.5-Omni audio re-score matches p14_o25_zeroshot_scores.csv to 1e-3 on 10 of 20 validation clips, not 20 of 20 (2026-09-23)
- Check: 20 clips in both M2 and part14_clips, re-cut on POD3b with part16/POD1/cut_segs.py (paths and worker count changed only). Cut: 20 of 20 byte-identical. Extended to all 794 part14_clips by full-file sha256, 794 of 794 identical.
- Score (score_audio.py core, Section 3 depression voice prompt verbatim, 30 s, bf16): 10 of 20 within 1e-3 (those within 3.2e-5), the other 10 differ by 2.2e-3 to 0.0107. Over all 794 Part 14 clips: 427 within 1e-3, median 5.0e-5, max 0.0403, Pearson 0.9995.
- The same pod is deterministic (rerun max diff 0.0, the recut clip and the Mac part14 clip give identical p_yes). The gaps are about one bf16 logit step, which points to a kernel difference between this pod (H100 NVL, torch 2.8.0+cu128, transformers 5.17.0, librosa 1.0.0) and the machine that made Part 14, which is not recorded. A first attempt with transformers 4.57.1 was further off (max 0.030) and was discarded.
- AUC impact is small: Part 14 pairs re-scored here give conflict 0.2210 (paper 0.2218), agreement 0.9467 (0.9482), agreement 311 distinct 0.9635 (0.9647). Same at two decimals except agreement distinct (0.96 both).
- Files: release/scores/part20/POD3b/val20_cut_and_score_check.csv, p14_rescore_o25_audio_envcheck.csv (+ .json).

Status now: released here. Files: scores/part20/POD3b/val20_cut_and_score_check.csv, scores/part20/POD3b/p14_rescore_o25_audio_envcheck.csv, scores/part20/POD3b/p14_rescore_o25_audio_envcheck.json.

## 146. [MEDIUM] [PART20] POD3b: with DistilBERT SST-2 arms the Omni audio conflict AUC is 0.46 [0.42, 0.51], not about 0.22 (2026-09-23)
- This is a result, not an error. M2 pairs (1334 pairs, 2668 rows, 162 speakers), same prompt, 30 s window and cut as Part 14: conflict 0.4619 [0.4165, 0.5102], agreement 0.8316 [0.7766, 0.8795] on all rows and 0.8554 [0.8199, 0.8885] on 799 distinct segments, paired conflict minus agreement -0.3697 [-0.4269, -0.3073]. RoBERTa Part 14: 0.2218 / 0.9482 / 0.9647.
- Diagnostic split by the RoBERTa rule arm of the same segment: the 420 M2 conflict rows that RoBERTa also calls conflict give 0.1951 [0.1545, 0.2413]. The 855 that RoBERTa calls neutral give 0.6150 [0.5593, 0.6755]. The below-chance conflict AUC holds where both classifiers call the words polarized against the label. It does not hold on the whole DistilBERT arm. So any sentence saying the inversion does not depend on the sentiment model would not be supported.
- Files: release/scores/part20/POD3b/sst2_pairs_o25_audio.csv, sst2_pairs_o25_audio_diag_by_roberta_arm.csv (+ .json).

Status now: released here. Files: scores/part20/POD3b/sst2_pairs_o25_audio.csv, scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv, release/scores/part20/rows/POD3b.tsv (authors' machine path). The paper makes no claim that the inversion is independent of the sentiment model.

## 147. [MEDIUM] [PART20] POD4c: AF3 on this pod reproduces the part10 Pitt copy exactly, not the part3 copy (2026-09-23)
- Check: 10 original Pitt windows (rows 0, 47, ..., 423 of overnight2/part3/af3_pitt.csv, 4 conflict, 6 agreement), scored on POD4c (H200, torch 2.8.0+cu128, transformers 5.17.0, AudioFlamingo3ForConditionalGeneration, bf16, 30 s cut, p15/af3_score.py pipeline, Pitt recording prompt verbatim).
- Against release/overnight2/part10/af3_pitt.csv: 10 of 10 within 1e-3 (max |dp_yes| < 1e-5, max |d answer_mass| 1e-5).
- Against release/overnight2/part3/af3_pitt.csv: 5 of 10 within 1e-3. The other 5 differ by 0.0278 to 0.0297 (one bf16 logit step). So the planned 1e-3 gate against part3 fails. It passes against part10.
- Both copies are reported side by side with the new cut as asked. part3 gives 0.5885 / 0.6169 / 0.5822 (all / conflict / agreement), part10 gives 0.5894 / 0.6075 / 0.5855. The part10 json also says 0.5894. The paired differences use each copy separately.
- File: release/scores/part20/POD4c/work/af3_val10.csv

Status now: fixed in the paper. The final Table 2 prints the part10 copy, row verbatim: "Audio Flamingo 3 \cite{af3}     & 0.42 & 0.84 & 0.61 & 0.59 \\" The part3 copy (0.6169 and 0.5822) is not used. A pod rescore reproduces part10 exactly (release/scores/part20/POD4c/work/af3_val10.csv (authors' machine path)), and an input-length check shows that the part10 run cut every Pitt window to its first 30 s before the processor, 480,000 samples and 750 audio tokens per clip, as the text says (paper1_submission_23sep/final_checks_2325Z/af3/AF3_INPUT_LENGTH.md, authors' machine path). lookup/master_lookup.csv still cites part3 for these two cells.

## 148. [MEDIUM] [PART20] POD6 A: the Parkinson's < depression < Alzheimer's language ordering holds only with the full E-DAIC interview transcript (2026-09-23)
- TF-IDF word logistic regression, vectorizer fit inside each training fold, single-split _mac speaker folds, 2000-draw speaker intervals.
- With the first-30 s transcript (the Table 1 transcript condition, paper1_local_runs/edaic30_text.csv) E-DAIC is 0.4844 [0.4028, 0.5624]. That is below PC-GITA 0.6196 [0.5700, 0.6681] with the intervals separated, and inside the NeuroVoz 0.5814 [0.5422, 0.6206] and MDVR-KCL 0.4315 [0.2560, 0.6287] intervals. So on the 30 s transcript the ordering fails at Parkinson's < depression.
- With the full-interview transcript (edaicfull_text.csv) E-DAIC is 0.6778 [0.5973, 0.7549]. The point estimates then follow Parkinson's < depression < Alzheimer's, but the depression interval overlaps all three Parkinson's intervals. Only depression < Alzheimer's (Pitt lower bound 0.7865 > 0.7549) and Parkinson's < Alzheimer's are separated.
- Files: release/scores/part20/POD6/A_language_saliency/A_summary.json and A_tfidf_<ds>_oof.csv (+ .sidecar.json).

Status now: released here. Files: release/scores/part20/POD6/A_language_saliency/A_summary.json (authors' machine path), scores/part20/POD6/A_language_saliency/A_tfidf_<dataset>_oof.csv, release/scores/part20/rows/POD6.tsv (authors' machine path).

## 149. [MEDIUM] [PART20] POD6 A: on read-speech Parkinson's sets the TF-IDF signal comes from ASR output, not word choice (2026-09-23)
- PC-GITA 0.6196 [0.5700, 0.6681] and NeuroVoz 0.5814 [0.5422, 0.6206] are above 0.5. But both are read speech. The Table 1 caption says every PC-GITA speaker reads the same passage, and NeuroVoz speakers read the same fixed sentences. The only text source is release/part4_text/asr/asr_<ds>.csv (ASR). So what the probe reads there is ASR errors caused by the voice, not the words the speaker chose. Any "language saliency" sentence about the Parkinson's sets should say this.

Status now: open. The final tex reads "As a measure of how much the words alone carry, a TF-IDF logistic regression on the transcripts, speaker-disjoint, reaches 0.43 to 0.62 on the Parkinson's sets, 0.68 on the E-DAIC transcripts and 0.84 to 0.93 on the Alzheimer's sets." It does not say that on the read Parkinson's sets the text is whisper-large-v3 output of fixed words, so the signal there is ASR error. What decides it: that sentence saying so. Evidence: scores/part20/POD6/A_language_saliency/A_tfidf_<dataset>_oof.csv.

## 150. [LOW] [PART20] POD6 A: Pitt transcript text source (2026-09-23)
- The paper transcript condition read /workspace/text/pitt_text.csv (o25_pitt_text.json). The local copy paper1_local_runs/pitt_text.csv has cleaned text. The 'text' column of pitt_conflict_manifest.csv is raw CHAT with annotation codes, and it differs from the cleaned text on 468 of 468 segments.
- TF-IDF AUC: cleaned 0.8353 [0.7865, 0.8862] (reported), raw CHAT 0.8452 [0.7975, 0.8918] (sensitivity). The two-decimal values differ (0.84 vs 0.85), and the intervals overlap almost completely.

Status now: fixed in the paper. Quote: "Pitt and ADReSS-2020 \cite{adress2020} come with manual CHAT transcripts and E-DAIC \cite{daic} with the challenge's automatic transcripts, used with the annotation codes stripped, and ADReSSo \cite{adresso} and the three Parkinson's sets are transcribed with whisper-large-v3 \cite{whisper}."

## 151. [LOW] [PART20] POD6 B: NeuroVoz metadata gaps that limit the matched pairs (2026-09-23)
- zenodo_upload/metadata/data_hc.csv: Two healthy controls lack sex or age, and one PD speaker is in data_pd.csv but outside the 1270-clip paper set [ids withheld here]. All three are excluded from pairing. That leaves 52 PD and 53 HC eligible. Greedy matching gives 38 pairs, which equals the maximum bipartite matching (scipy), so no larger set exists under same sex and |age diff| <= 5.

Status now: released here. Files: scores/part20/POD6/B_neurovoz_matched/B_encprobe_nested5_nvmatched_oof.csv, scores/part20/POD6/B_neurovoz_matched/B_encprobe_nested5_nvmatched_oof.sidecar.json.

## 152. [MEDIUM] [PART20] POD4d: Kimi-Audio validation gate failed at 1e-3, pod rerun of the 468 original Pitt windows drifts per clip but not in AUC (2026-09-23)
- Gate: first 10 original windows (canonical file order) vs `overnight2/part3/kimi_pitt_zeroshot_scores.csv`, tolerance 1e-3. FAIL: max |diff| 0.0768 (one clip, 0.7982 canonical vs 0.7214 pod), 9 of 10 over 1e-3.
- Environment was the working one: separate venv, torch 2.8.0+cu128, torchaudio 2.8.0, transformers 4.51.3, prebuilt flash_attn 2.8.3.post1 wheel, Kimi-Audio repo + GLM-4-Voice submodules (same submodule commits eb00ce9 / dd9105b), lazy detokenizer patch, same model snapshot 9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b, bf16, same prompt, same yes/no ids, same 30 s cut. GPU NVIDIA H200. requirements.txt not installed, as in the working run (its pip -r failed there on a flash_attn source build, so nothing from it was installed).
- Pod is deterministic: each clip scored twice in one process and again in a separate process gives identical p_yes. So the gap is between machines, not run to run.
- Whole 468 rerun on the pod vs canonical: per-clip max |diff| 0.1997, median 0.0103, 413 of 468 over 1e-3, Pearson 0.989. AUC: overall 0.7151 vs 0.7180, conflict 0.7215 vs 0.7197, agreement 0.7192 vs 0.7227. Paired canonical minus pod rerun (2000 speaker draws, rng 0): overall 0.0029 [-0.0049, 0.0108], conflict -0.0018 [-0.0215, 0.0160], agreement 0.0035 [-0.0053, 0.0126]. All three intervals include zero.
- Hidden states vs `overnight2/kimi/kimi_pitt_states.npz` for the 10 windows: relative error 0.3-0.5% at LM layer 0, rising to 1.6-3.0% at layer 27. Most consistent with bf16 numeric drift across hardware that grows with depth. The canonical run saved no discrete audio token ids, so input identity at token level cannot be checked.
- Decision: the cut was scored anyway (the gate cannot pass at 1e-3 on this hardware). Every original-minus-interviewer-free difference is reported twice, once against the canonical file and once against this pod's own rerun of the originals (same environment as the cut). They agree to within 0.004 on every arm. Same class as the existing M8 note that Kimi Pitt already has two disagreeing score files.
- Files: release/scores/part20/POD4d/kimi_val10_vs_canonical.{csv,json}, kimi_pitt_orig468_pod_scores.csv, kimi_pitt_canon_minus_podorig_*.{csv,json}.

Status now: released here. Files: scores/part20/POD4d/kimi_val10_vs_canonical.csv, scores/part20/POD4d/kimi_val10_vs_canonical.json, scores/part20/POD4d/kimi_pitt_orig_podrerun_overall.csv, release/scores/part20/rows/POD4d.tsv (authors' machine path).

## 153. [MEDIUM] [PART20] POD4d: removing the interviewer changes Kimi's Pitt agreement arm, not its conflict arm, and the cut also shortens the audio (2026-09-23)
- Kimi-Audio zero-shot on the 468 interviewer-free (PAR-only) clips: overall 0.5981 [0.5345, 0.6617], conflict 0.6972 [0.5905, 0.7946], agreement 0.5612 [0.4860, 0.6364] (includes 0.5).
- Paired canonical original minus interviewer-free: overall 0.1199 [0.0718, 0.1665], conflict 0.0224 [-0.0495, 0.0996] (includes zero), agreement 0.1615 [0.1068, 0.2168].
- Caveat for any sentence built on this: arm B shortens the clip as well as removing the interviewer (heard duration 0.68 s to 29.2 s, 68 of 468 windows have under 10 s of participant speech per windows_under10s.csv). Original minus interviewer-free therefore mixes interviewer removal with the duration drop. The duration-matched arm C of part16 cut_arms.py was not part of this job, so the two cannot be separated from these files.

Status now: released here. Files: scores/part20/POD4d/kimi_pitt_noinv_overall.csv, release/scores/part20/rows/POD4d.tsv (authors' machine path). The paper reports no interviewer-free Kimi-Audio result.

## 154. [MEDIUM] [PART20] POD4c: AF2 ran in the environment that made af2_pitt468.csv, not the python3.10 / torch 2.0.1 env in the plan (2026-09-23)
- The plan called for a separate python3.10, torch 2.0.1, torchaudio 2.0.2, transformers 4.46.3, laion-clap 1.1.3 env. The transcript that made af2_results/af2_pitt468.csv (release/c_conversation.txt, the 20 Sep af2 pod, /workspace/af2_run.py) shows that file came from image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 on an H200: system python 3.12, torch 2.8.0+cu128, transformers 4.46.3, the lab AF2Adapter (speech-health/src/models/af2_adapter.py) and the official NVIDIA af2_code bundled from ~/Desktop/af2_run. The python3.10 / torch 2.0.1 env is the Mac ports/af2env, which the 15 Sep audit says ranks Pitt clips differently from af2_pitt468.csv.
- So POD4c used a separate venv (/workspace/af2venv, --system-site-packages over the same image's torch 2.8.0) with transformers 4.46.3 and tokenizers 0.20.3. The AF3 system python was not touched.
- Validation: 10 original windows gave p_yes and answer_mass bit-identical to af2_pitt468.csv (max abs diff 0.0).
- Scoring rule: af2_pitt468.csv used the adapter's rule (fp32, Yes/yes against No/no, bare and leading space, first token id), not rule 3 (bf16, YES/NO added, single-token only). The paired comparison has to use the same rule as the paper file, so p_yes uses the adapter rule. The same forward pass also records p_yes_rule3 (YES/NO added, single-token only). On the validation clips it differs by at most 7.3e-8. The fp32 dtype is kept because bf16 would break the bit match.
- File: release/scores/part20/POD4c/work/af2_val10.csv

Status now: released here. Files: release/scores/part20/POD4c/work/af2_val10.csv (authors' machine path).

## 155. [MEDIUM] [PART20] POD4: the 10-window Qwen2.5-Omni pipeline check misses the 1e-3 bar on 4 of 10 windows (2026-09-23)
- Check: score_zs.py (scoring path copied from p15 extract_probe_layers.py: same processor and chat template calls, float32 to bf16 cast, first-token Yes/No ids 7414/9454/9693/9834/14004 vs 902/2152/2308/2753/8996, 30 s cap) on 10 original windows from DementiaBank/segments/ (sha256 identical to the Mac copies), recording wording, against omni_final/omni_pitt_zeroshot_scores.csv. Prompt string byte-identical to that file's prompt column.
- 6 of 10 within 7e-5. 4 of 10 off by 0.0128 to 0.0153 (one clip 0.4078 vs 0.4230, and three more clips) [clip names withheld here].
- Full 468 rerun on this pod: AUC 0.6497 vs 0.6578 in the file. 229 of 468 clips within 1e-3, median gap 0.010, max 0.062, Pearson 0.9905. Every miss checked is exactly one bf16 step (1/16) in the Yes minus No logit. Same pod rerun is bit-identical (check10 = full rerun on those clips, output_hidden_states on or off gives 0.0 difference). The Part 16 POD4 entry above also got 0.6497 on its pod.
- Environment here: transformers 5.17.0, torch 2.8.0+cu128, librosa 1.0.0, numpy 2.1.2, NVIDIA H200 (driver 575.57.08), Intel Xeon Gold 6548Y+, sdpa attention. transformers 5.17.0 was on PyPI from 2026-09-09, before the 2026-09-18 omni_final run, so an unpinned install then would most likely have been the same version. The machine behind omni_final is not recorded.
- The machine term alone is not zero at the arm level: paired this-pod-rerun minus paper file, agreement arm, -0.0102 [-0.0196, -0.0011] (n=322, 175 speakers). Overall -0.0080 [-0.0160, +0.0002], conflict -0.0024 [-0.0218, +0.0150].
- Consequence: every POD4 paired contrast is given twice, against the paper file and against this pod's own rerun of the original windows (same code, same session). They agree in sign and in whether the interval excludes zero: original minus interviewer-free overall -0.0652 [-0.1081, -0.0228] (paper file) and -0.0732 [-0.1165, -0.0312] (pod rerun). Agreement -0.0954 [-0.1512, -0.0401] and -0.1056 [-0.1613, -0.0489]. Conflict -0.0086 [-0.0890, +0.0679] and -0.0110 [-0.0937, +0.0671].
- Files: release/scores/part20/POD4/check/o25_check10_vs_omni_final.csv (+ .json), o25_orig_recording_rerun.csv (+ .sidecar.json).

Status now: released here. Files: release/scores/part20/POD4/check/o25_check10_vs_omni_final.csv (authors' machine path), release/scores/part20/POD4/check/o25_check10_vs_omni_final.json (authors' machine path), scores/part20/POD4/o25_orig_recording_rerun.csv.

## 156. [LOW] [PART20] POD4: Qwen2-Audio dtype. Rule 3 says bf16, the paper's Qwen2-Audio pipeline is fp16 (2026-09-23)
- probe2/extract2.py and p15 extract_probe_layers.py both load Qwen2-Audio in float16. POD4 ran both. Primary rows use fp16 so the pipeline matches the paper file.
- fp16 rerun of the original 468 windows: 0.6174 vs the file's 0.6173 (258 of 468 clips within 1e-3, max 0.0104). Paired rerun minus file overall +0.0001 [-0.0009, +0.0011].
- bf16: original windows 0.6206, interviewer-free 0.6236 (fp16: 0.6174 and 0.6211). Arms and paired differences agree with fp16 to 0.01 or better, and whether an interval excludes zero is the same.
- Files: release/scores/part20/POD4/q2a_*_bf16.csv (+ sidecars).

Status now: released here. Files: scores/part20/POD4/q2a_orig_recording_rerun.csv, scores/part20/POD4/q2a_orig_recording_rerun_bf16.csv, scores/part20/POD4/q2a_noinv_recording_bf16.csv.

## 157. [LOW] [PART20] POD4: rule 3 id set adds " YES" (14080) and " NO" (5664), which extract_probe_layers.py leaves out (2026-09-23)
- Same tokenizer ids in Qwen2.5-Omni and Qwen2-Audio. POD4 keeps the paper's 5+5 set as p_yes (needed to match the paper scorer) and writes the 6+6 single-token set as p_yes_r3 in every per-clip csv.
- Largest per-clip difference 1.5e-7 (Omni) and 2.5e-6 (Qwen2-Audio). Largest AUC change 0.0003 (Qwen2-Audio bf16 interviewer-free 0.6236 vs 0.6233). Omni interviewer-free 0.722956 both ways.

Status now: released here. Files: scores/part20/POD4/o25_noinv_recording.csv. p_yes and p_yes_r3 are both saved per clip.

## 158. [LOW] [PART20] POD4: the interviewer-free cut manifest's `arm` column is not the conflict/agreement arm (2026-09-23)
- pitt_noinv_cut/manifest.csv has arm = "B_paronly_noinv" on all 468 rows (the Part 16 cut variant name). POD4 took conflict/agreement from DementiaBank/pitt_conflict_manifest.csv column `set`, keyed by basename(segment_path) = orig_clip_id. It agrees with the orig_clip_id name prefix on 468 of 468 (146 conflict, 322 agreement). Speaker and label in the cut manifest match omni_final and probe2 on 468 of 468.

Status now: released here. Files: manifests/pitt_conflict_manifest_468.csv, scores/part20/POD4/o25_noinv_recording.csv. The arm comes from the set column of the manifest.

## 159. [LOW] [PART20] POD4: another job's setup ran on the POD4 pod (2026-09-23)
- At about 12:10Z /workspace/setup.sh on blkq19zhy0aliz was replaced by a script that makes /workspace/scores/part20/POD5 and downloads only Qwen2.5-Omni. The setup that actually ran was that script (transformers 5.17.0). POD4 downloaded Qwen2-Audio itself. POD4 wrote only under /workspace/pod4 and /workspace/scores/part20/POD4, so no outputs were mixed. Whichever job owns POD5 may be pointed at this pod's address.

Status now: released here. Files: scores/part20/POD4/. Every POD4 output sits in its own folder with a sidecar that gives its command and sources.

## 160. [LOW] [PART20] POD4b: follow-up, all 468 original Pitt windows re-scored on the POD4b pod with the same code (2026-09-23)
- Same-pod AUC on the original windows: 0.7638 overall, 0.6062 conflict, 0.8221 agreement (shipped file 0.7619 / 0.6066 / 0.8231, part16 rerun 0.7673 / 0.6079 / 0.8265). All three runs agree within 0.0054 AUC on every cell.
- Per clip (5+5 ids, as the reference files): within 1e-3 of the part16 rerun on 108 of 468, more than 0.1 away on 18 (max 0.200). Within 1e-3 of the shipped file on 50 of 468, more than 0.1 away on 95 (max 0.304). The shipped and rerun files agree within 1e-3 on only 56 of 468 (max 0.381). So per-clip Qwen3-Omni p_yes is not reproducible across machines at 1e-3 in any pair of runs. The AUC is.
- On this pod the 10 validation windows came out identical (max |d| 0.0) in two separate processes, so the run here is deterministic.
- Paired original-minus-interviewer-free, overall / conflict / agreement: shipped -0.0182 / -0.0121 / -0.0207. Rerun -0.0128 / -0.0108 / -0.0174. Same pod -0.0164 / -0.0125 / -0.0217. Every interval includes zero whichever original file is used.
- Files: release/scores/part20/POD4b/q3o_pitt_orig468_samepod_zeroshot_scores.csv and the P20_4b_q3o_*samepod* per-clip csv + sidecar pairs.

Status now: released here. Files: scores/part20/POD4b/P20_4b_q3o_orig_samepod_overall.perclip.csv, release/scores/part20/rows/POD4b.tsv (authors' machine path).

## 161. [LOW] [PART20] POD6 B: this pod's Qwen2.5-Omni NeuroVoz zero-shot matches omni_final to 1e-3 on 478 of 907 subset clips (2026-09-23)
- Check: extract_probe_layers.py (the copy in a temporary working folder, sha256 46e98591...), pd voice prompt verbatim (same string as the prompt column of omni_final/omni_neurovoz_zeroshot_scores.csv), 30 s, bf16, on an H100 NVL with torch 2.8.0+cu128 and transformers >=5. The input is the original 44.1 kHz WAVs from zenodo_upload/audios. PCM was checked identical on all 907.
- Per clip vs omni_final: median |diff| 3.4e-05, max 0.0459, Pearson 0.9925, 478 of 907 within 1e-3. Every gap is about one bf16 logit step (first clip: 0.5313 = sigmoid(0.125) in the paper file, 0.5467 here). This is the same kernel term POD3b logged. Subset zero-shot AUC is 0.6312 here vs 0.6282 from the paper file.
- The job B zero-shot row uses the paper file, as planned. The encoder probe uses this pod's states, so the subset probe carries this machine term. The full-1270 rerun on the same pod (B_neurovoz_full_check) is there to measure it.

Status now: released here. Files: release/scores/part20/POD6/B_neurovoz_full_check/omni_nvfull_zeroshot_scores_samepod.csv (authors' machine path), release/scores/part20/POD6/B_neurovoz_full_check/B_full_check_summary.json (authors' machine path).

## 162. [LOW] [PART20] POD2: the earlier Pitt LoRA values 0.8250 / 0.5520 / 0.9199 use the two-logit p_yes, not the rule 3 p_yes (2026-09-23)
- In part16/POD2/lora_pitt_oof.csv the column `p_yes` is softmax over [' Yes',' No'] only (lora_pitt.py, same as sft_projector.py line 89, which gives the projector 0.7673). The rule 3 score is the column `p_yes_multivariant`: 0.8220 [0.7702, 0.8711] overall, 0.5516 conflict, 0.9161 agreement (recomputed today from that file).
- The PART20 POD2 balanced LoRA file (scores/part20/POD2/balanced_lora_pitt_oof.csv) saves both: `p_yes` = rule 3, `p_yes_2logit` = the two-logit score. Every balanced-minus-earlier difference is computed like for like (two-logit vs two-logit, rule 3 vs rule 3). The two conclusions agree.

Status now: released here. Files: scores/part20/POD2/balanced_lora_pitt_oof.csv, release/scores/part20/rows/POD2.tsv (authors' machine path). Both p_yes definitions are saved per clip.

## 163. [LOW] [PART20] POD3: rule 3 wording also covers " YES" and " NO", which the paper scripts leave out (2026-09-23)
- In the Qwen2.5-Omni tokenizer " YES" (14080) and " NO" (5664) are single tokens. Rule 3 says leading-space variants of YES and NO count. extract_probe_layers.py and part16 score_audio.py count only Yes, " Yes", yes, " yes", YES and No, " No", no, " no", NO.
- On the Part 20 POD3 fine-tune (p14_ft966_oof.csv) both lists were scored from the saved final fold checkpoints. The largest per-clip p_yes difference is 8.9e-07. Every AUC agrees at 4 dp (conflict 0.5932, agreement 483 0.6228, agreement 311 0.6686). The primary p_yes column uses the paper list, so it compares directly with the zero-shot and 1c files. The literal rule 3 value is in the p_yes_rule3_literal column.

Status now: released here. Files: scores/part20/POD3/p14_ft966_oof.csv. Both id lists are saved per clip.

## 164. [LOW] [PART20] POD5: projector fine-tune restarted twice, nested original-states probe restarted once (2026-09-23)
- SFT attempt 1 died at start (FileNotFoundError on /workspace/p20/work/pitt_groupkfold5_pod.csv): the fold file was copied after the launch. Log kept as sft_attempt1_racefail.log.
- SFT attempt 2 ran fold 0 epoch 0 in 2.1 min, then stalled for 8+ min with the GPU at 0 to 4 %: the pod's cgroup CPU quota is 15.3 CPUs (cpu.cfs_quota_us 1530000) although nproc reports 144, and the CPU probes ran 100 to 120 joblib workers, which starved the SFT's 144 torch threads. No fold had finished, so nothing to resume. Restarted from scratch with OMP_NUM_THREADS=4 (torch.set_num_threads(4)) and a step log line. Recipe unchanged. Log kept as sft_attempt2_cpu_throttled.log.
- The first nested probe on the original states (110 workers) was killed for the same reason and rerun with 8 workers. The rerun is the one reported. All later CPU jobs used 2 to 8 workers.

Status now: released here. Files: release/scores/part20/POD5/sft_attempt1_racefail.log (authors' machine path), release/scores/part20/POD5/sft_attempt2_cpu_throttled.log (authors' machine path), release/scores/part20/POD5/nested_orig_attempt1_killed_cpu_quota.log (authors' machine path).

## 165. [LOW] [PART20] POD5: part20 p_yes rule adds " YES" and " NO" to the paper's token set (2026-09-23)
- Single-token bare and leading-space Yes/yes/YES, No/no/NO on the Qwen2.5-Omni tokenizer: yes [7414, 9454, 9693, 9834, 14004, 14080], no [902, 2152, 2308, 2753, 5664, 8996]. The paper's extract_probe_layers.py set (and omni25_readout.pt) is 5+5 without 14080 (" YES") and 5664 (" NO"). On the interviewer-free zero-shot both sets give AUC 0.7231. Both columns are in POD5_zs_o25_noinv.csv. The direction test uses the 6+6 rule set. The 5+5 legacy direction is in the sidecar (legacy_ids_check).
- The projector fine-tune keeps the sft_projector.py recipe p_yes (softmax over the ' Yes' 7414 and ' No' 2308 logits) as the headline, so it is comparable with omnisft_pitt_oof.csv. The rule p_yes is reported beside it (POD5_sft_rulepyes_overall).

Status now: released here. Files: scores/part20/POD5/POD5_zs_o25_noinv.csv. Both id sets are saved per clip.

## 166. [INFO] [PART20] POD5: pipeline checks on this pod (transformers 5.17.0, torch 2.8.0+cu128, sklearn 1.9.1, numpy 2.1.2) (2026-09-23)
- Extraction: 25 ORIGINAL Pitt windows (first 25 of pitt_groupkfold5_pod.csv) re-extracted here: p_yes vs omni_final/omni_pitt_zeroshot_scores.csv max |d| 0.0152, mean 0.0061, r 0.9941. States vs overnight2/part10/o25_pitt_states.npz cosine min 0.99999 enc, 0.99997 proj, 0.99980 llm, 0.99985 ans.
- Nested probes: the same nested_p20.py on o25_pitt_states.npz reproduces the paper per-repeat AUCs exactly at 4 dp: LM 0.8032 [0.8135, 0.8046, 0.7985, 0.7961, 0.8031], answer 0.7636, encoder pod seeds 0.7761, projector seed-0 0.7507. Encoder Mac seeds give 0.7705 on this pod vs 0.7706 on the Mac (pod-vs-Mac logistic numerics). The paired encoder comparison uses the paper's own per-clip file pitt_enc_nested5_oof.npz.
- Direction test on o25 states here: cos -0.0109, probe 0.7709, after removal 0.7661 (paper 0.7662). Random floor 0.0133 from 2000 directions (task spec) vs the paper's 0.0135 from 1000 directions (part10_readout_direction_v2.py). Cosine interval here uses one fresh default_rng(0) stream per cell. v2 seeded each draw with default_rng(1000+b), so the paper's interval [-0.0235, 0.0097] differs slightly from [-0.0234, 0.0099].
- The paper's projector value 0.7507 (omni_pitt_nested_repeats.json) is on the seed-0 renumbered split (nested_repeats_all.py runs one repeat for a 1-point stream), not on pitt_groupkfold5_pod.csv. On the latter the o25 states give 0.7669.

Status now: released here. Files: scores/part20/POD5/POD5_direction_origrerun_check.csv, scores/part20/POD5/POD5_nested_enc_pod_seed.csv, release/scores/part20/rows/POD5.tsv (authors' machine path), lookup/readout_direction.csv. The authors' rule for the answer state: 0.7709 is the final LM layer at the first answer position, and removing the readout direction leaves 0.7662 (lookup/readout_direction.csv).

## 167. [LOW] [PART20] POD5: shared temporary working folder (2026-09-23)
- A temporary working folder is written by several part20 jobs at once (their scripts and upload chunks appeared there). POD5's scripts were moved into a POD5 subfolder of it and every POD5 run used the copies uploaded to /root/p20/scripts on the pod.

Status now: released here. Files: release/scores/part20/POD5/scripts/ (authors' machine path). These are the scripts POD5 ran.

## 168. [HIGH] [PART20] POD4c: for AF2, the paper file and the interviewer-free cut do not cover the same audio span (2026-09-23)
- af2_results/af2_pitt468.csv scored every original window at full length. The 20 Sep af2_run.py does librosa.load with no cut, and AF2's own grid takes 10 s windows up to 120 s. All 468 windows are longer than 30 s: median 33.3 s, max 55.6 s, 54 longer than 40 s. The interviewer-free cut (pitt_noinv_cut build_report) takes speech only from the first min(window, 30 s).
- So the planned AF2 paired "original minus interviewer-free" mixes two changes: removing the interviewer, and dropping the audio after 30 s. For AF3 the two sides match, because the AF3 paper files were also cut at 30 s.
- Added on POD4c as a control, not in the plan: AF2 on the same 468 original windows cut to their first 30 s (af2_zs_win.py, otherwise identical). Rows af2_orig30_*, af2_diff_orig30_minus_noinv_* (removing the interviewer only, same span) and af2_diff_origfull_minus_orig30_* (the span change only). The originals were sent as FLAC and checked clip by clip on the pod against the int16 sample sha256 of the `clips_for_af3/pitt468/*.wav` files on the authors' machine. Those wavs are byte-identical to the cut builder's source segments (468 of 468 file sha256 match source_segments.sha256).

Status now: released here. Files: release/scores/part20/POD4c/af2_orig30_all.csv (authors' machine path), release/scores/part20/POD4c/af2_diff_orig30_minus_noinv_all.csv (authors' machine path), release/scores/part20/POD4c/af2_diff_origfull_minus_orig30_all.csv (authors' machine path), release/scores/part20/rows/POD4c.tsv (authors' machine path). The same-span control is released for all three arms.

## 169. [MEDIUM] [PART20] POD2: balanced LoRA on Pitt drops pooled out-of-fold AUC, part of the drop is fold-to-fold calibration, not ranking (2026-09-23)
- Pooled OOF AUC (two-logit, the column behind 0.8250): balanced LoRA 0.7200 [0.6494, 0.7866]. Same-pod uniform-weight control, same code and seeds, 0.8346 [0.7856, 0.8825]. Earlier LoRA 0.8250. Control minus earlier +0.0096 [-0.0204, +0.0437], so the recipe reproduces and the drop comes from the weights (balanced minus control -0.1146 [-0.1751, -0.0564]).
- Per-fold AUCs are closer: balanced 0.724 / 0.889 / 0.803 / 0.908 / 0.889 (mean 0.8425), control 0.831 / 0.914 / 0.821 / 0.930 / 0.912 (mean 0.8818). But the mean two-logit p_yes per test fold swings 0.475 to 0.899 for the balanced run, against 0.559 to 0.773 for the control. Pooling folds with shifted offsets costs rank AUC.
- The paper's protocol is pooled OOF AUC, so 0.7200 is the reported value. The per-fold numbers are a diagnostic only. Files: scores/part20/POD2/balanced_lora_pitt_oof.csv, scores/part20/POD2/control/uniform_lora_pitt_control_oof.sidecar.json (per_fold_diagnostics).
- Also: the earlier LoRA (part16 POD2) left the LoRA init and dropout unseeded. This run and its control seed both (torch seed 0 for init, 1000+k per fold) and ran folds in 3 parallel processes. Same data bytes (sha256 of all 468 wavs checked against the G-Drive originals).

Status now: released here. Files: scores/part20/POD2/balanced_lora_pitt_oof.csv, scores/part20/POD2/control/uniform_lora_pitt_control_oof.csv, scores/part20/POD2/control/uniform_lora_pitt_control_oof.sidecar.json, release/scores/part20/rows/POD2.tsv (authors' machine path). The paper reports the unbalanced LoRA run only: "Low-rank adaptation (LoRA) \cite{lora} of the language model (rank 8, all attention projections of the 28 layers, 5.0\,M trainable parameters against 4.6\,M for the projector) reaches 0.825 on Pitt, 0.06 [0.02, 0.10] above the projector run, and 0.55 on its conflict segments, and training both together gives 0.77."

## 170. [MEDIUM] [PART20] POD1: balanced projector fine-tune, fold 3 collapsed in the first run and not in two repeats (2026-09-23)
- Run: `release/scores/part20/POD1/balanced_ft_pitt.py` (sft_projector.py recipe, only the per-clip loss weight changed), Qwen2.5-Omni, Pitt 468, folds `release/folds/pitt_groupkfold5_pod.csv`, pod H100 NVL, torch 2.8.0+cu128, transformers 5.17.0.
- Repeat 1, fold 3: training CE by epoch 0.6658, 0.5949, 0.7737 (other folds end near 0.18 to 0.26). On all 93 test clips the Yes+No answer mass is about 1e-4 (median 0.0001), so the tuned projector stopped the model from answering Yes or No at all. Fold AUC 0.463 (the paper standard run on the same 93 clips: 0.742).
- The same fold run alone again (`balanced_rerun_fold3/`): CE 0.6317, 0.4412, 0.2588, answer mass normal. Repeat 2 of the whole run (`balanced_r2_folds/`): fold 3 CE 0.6241, 0.4184, 0.2046, normal. Same script, folds, clip order and weights every time. Epoch-0 CE already differs between runs, so training on this pod is not bit-reproducible (bf16 GPU kernels), and the collapse is one bad draw, not a fixed effect of the weights.
- Effect: balanced repeat 1 as run = 0.7301 / 0.4693 / 0.8279 (all / conflict / agreement). With fold 3 swapped for the lone rerun = 0.7603 / 0.5003 / 0.8573. Repeat 2 = 0.7360 / 0.4810 / 0.8379. All three files are kept. None is dropped.
- Files: release/scores/part20/POD1/balanced_ft_pitt_oof.csv, balanced_ft_pitt_fold3rerun_oof.csv, balanced_ft_pitt_r2_oof.csv, logs/balanced_0_3.log, logs/balanced_rerun_fold3.log.

Status now: released here. Files: scores/part20/POD1/balanced_ft_pitt_oof.csv, scores/part20/POD1/balanced_ft_pitt_fold3rerun_oof.csv, release/scores/part20/POD1/balanced_ft_pitt_r2_oof.csv (authors' machine path), release/scores/part20/rows/POD1.tsv (authors' machine path). The paper mentions the balanced fine-tune in one sentence: "A class-balanced loss does not change this." Its conflict arm is 0.4693.

## 171. [MEDIUM] [PART20] POD1: the standard projector fine-tune reproduces overall but not by arm (2026-09-23)
- Two reruns of the unchanged recipe (weight 1 for every clip) on this pod, same script, folds and clip order as the balanced run: overall 0.7720 and 0.7734 (paper file omnisft_pitt_oof.csv 0.7673), conflict 0.4697 and 0.4490 (paper 0.5189), agreement 0.8895 and 0.8967 (paper 0.8593).
- Paired against the paper file, repeat 2 agreement is +0.0374 [+0.0025, +0.0786] (excludes zero). Conflict -0.0700 [-0.1678, +0.0142]. Overall +0.0061 [-0.0337, +0.0493].
- So the paper's arm split (0.52 conflict, 0.86 agreement) is one draw. Run to run, the conflict arm moves by about 0.05 to 0.07 and agreement by about 0.03 to 0.04. The conflict arm stays at chance in every run (every interval includes 0.5). The machine that made omnisft_pitt_oof.csv is not recorded in omnisft_pitt_sft.json.
- Files: release/scores/part20/POD1/standard_rerun_ft_pitt_oof.csv, standard_rerun_ft_pitt_r2_oof.csv (+ .sidecar.json).

Status now: fixed in the paper. Quote: "Its conflict arm stays at chance in every run (0.45 to 0.57, every interval including 0.5) while its agreement arm rises to 0.82 to 0.90, so the projector recovers the overall AUC through the arm the words already help." Files: scores/part20/POD1/standard_rerun_ft_pitt_oof.csv, release/scores/part20/POD1/standard_rerun_ft_pitt_r2_oof.csv (authors' machine path), release/scores/part20/rows/POD1.tsv (authors' machine path).

## 172. [LOW] [PART20] POD1: p_yes definition for the fine-tune comparison (2026-09-23)
- Rule 3 asks for p_yes over all single-token Yes/yes/YES vs No/no/NO. The recipe (sft_projector.py) and the paper file omnisft_pitt_oof.csv use softmax over just [' Yes',' No']. To keep the paired balanced-minus-standard difference like for like, the headline column `p_yes` uses the recipe definition. The rule 3 value is saved per clip as `p_yes_multi`, with `answer_mass`. It moves AUC by at most 0.016 (repeat 1 agreement 0.8279 vs 0.8433, repeat 2 all 0.7360 vs 0.7381). Answer mass is at least 0.995 on every clip in every run except the collapsed repeat 1 fold 3.

Status now: released here. Files: scores/part20/POD1/standard_rerun_ft_pitt_oof.csv, scores/part20/POD1/balanced_ft_pitt_oof.csv. p_yes and p_yes_multi are saved per clip with answer_mass.

## 173. [LOW] [PART20] POD1: correction to the p_yes entry just above (2026-09-23)
- Answer mass is at least 0.9948 on every clip (not 0.995) in every POD1 run except the collapsed repeat 1 fold 3. Lowest: repeat 1 fold 1, 0.9948.

Status now: released here. Files: scores/part20/POD1/standard_rerun_ft_pitt_oof.csv, scores/part20/POD1/balanced_ft_pitt_oof.csv.

## 174. [MEDIUM] [PART20] POD4c: AF2 zero-shot falls below chance on the interviewer-free Pitt cut, agreement arm 0.40 [0.33, 0.47] (2026-09-23)
- This is a result, not an error. AF2 interviewer-free: all 0.4655 [0.4038, 0.5265], conflict 0.6477 [0.5357, 0.7444], agreement 0.3966 [0.3305, 0.4657]. Paper file af2_pitt468.csv: 0.5724 / 0.6310 / 0.5544.
- Paired original minus interviewer-free: all +0.1068 [+0.0595, +0.1529], agreement +0.1579 [+0.1052, +0.2093] (both exclude zero), conflict -0.0167 [-0.1068, +0.0759].
- Same-span control (original first 30 s): all 0.5456, conflict 0.5835, agreement 0.5337. Of the agreement drop, +0.1371 [+0.0781, +0.1927] comes from removing the interviewer on the same span. Only +0.0208 [-0.0179, +0.0613] comes from the span change (full window minus first 30 s).
- Diagnostic, not a claim: on the cut, AF2 p_yes tracks how much participant speech is left (Spearman 0.43 with dur_par_s). dur_par_s alone gives AUC 0.335 for the dementia label. AF3 p_yes does not track it (Spearman -0.01). A length artifact is the likely driver of the below-chance AF2 agreement value. This was not tested further.
- AF3 on the same cut changes nothing beyond noise: all 0.5920, conflict 0.5417, agreement 0.6140. All paired differences against both paper copies include zero.
- Files: release/scores/part20/POD4c/ (af2_*, af3_* .csv/.json/.RESULT, raw scores in work/)

Status now: released here. Files: scores/part20/POD4c/af2_noinv_all.csv, scores/part20/POD4c/af2_diff_af2pitt468_minus_noinv_all.csv, release/scores/part20/rows/POD4c.tsv (authors' machine path). The paper reports no interviewer-free AF2 result.

## 175. [MEDIUM] [PART20] POD1b: the standard Pitt projector fine-tune collapses on some folds, and the answer mass goes to zero (2026-09-23)
- Recipe unchanged from p15/sft_projector.py (projector only, AdamW lr 1e-4, 3 epochs, batch 1, loss on the ' Yes' and ' No' logits only), folds from release/folds/pitt_groupkfold5_pod.csv. Seeds 1, 2, 3 change only the clip order.
- 2 of the 15 new seed-folds collapsed. Seed 1 fold 4: training loss by epoch 0.607, 0.382, 0.694. Seed 2 fold 0: 0.605, 0.637, 0.600. On every test clip of those two folds, P(Yes variants)+P(No variants) from the full-vocabulary softmax is under 0.5 (median 0.0000, 93 and 94 clips). So the model stops answering Yes or No at all. The recipe's two-logit p_yes still ranks those clips, but worse: within-fold AUC 0.58 and 0.69, against 0.72 to 0.90 on the folds that did not collapse.
- The original run (omni_final/omnisft_pitt_oof.csv) did not record answer mass. Its fold 3 shows the same squeezed p_yes as the collapsed folds here (range 0.35 to 0.93, sd 0.16, against sd 0.33 to 0.37 on its other folds). So the paper's 0.7673 may include one collapsed fold. This is not confirmed.
- Files: release/scores/part20/POD1b/std_ft_seed{1,2}_pitt_oof.csv (answer_mass column), per_fold_diagnostics in the matching .sidecar.json

Status now: fixed in the paper. The final tex reports the seed range: "On the Pitt dataset the retrained projector reaches 0.77, the level of the answer-state probe of Section~\ref{sec:gap}, and 0.74 to 0.81 across three further seeds." Whether the original run's fold 3 collapsed still cannot be settled from its scores, since it saved no answer mass. Related files: release/scores/part20/POD1b/std_ft_seed1_pitt_oof.csv (authors' machine path), release/scores/part20/POD1b/std_ft_seed2_pitt_oof.csv (authors' machine path), release/scores/part20/rows/POD1b.tsv (authors' machine path).

## 176. [MEDIUM] [PART20] POD1b: the Pitt projector fine-tune moves 0.07 overall across seeds, and the same seed does not reproduce 0.7673 on this pod (2026-09-23)
- sft_projector.py has no seed argument. Its only randomness is the clip order, np.random.RandomState(fold*10+epoch). Torch is unseeded, and every dropout in the Omni thinker config is 0.0 (checked on the pod). The omni_final sidecar has no seed field. Here the order is RandomState(seed*1000+fold*10+epoch) for seeds 1, 2, 3.
- Overall out-of-fold AUC: seed 1 0.7396, seed 2 0.8085, seed 3 0.7975, original 0.7673. Range 0.74 to 0.81.
- Conflict: 0.5364, 0.5636, 0.5666, original 0.5189. Range 0.52 to 0.57. Every conflict interval includes 0.5.
- Agreement: 0.8213, 0.9022, 0.8844, original 0.8593. Range 0.82 to 0.90.
- Extra check, not requested: seed 0 is the original clip order exactly. On this pod (transformers 5.17.0, H100 NVL) it gives overall 0.7955, conflict 0.5282, agreement 0.8983, and no fold collapsed. Against the original clip by clip, Spearman is 0.77 overall and 0.67 on fold 3. So a same-seed rerun on a different machine moves the value by about as much as a new seed does. The original's library versions are not recorded, so this cannot be split further.
- For the paper: the single-run 0.77 is inside the seed range. The earlier LoRA 0.8250 is above every projector seed, but only by 0.017 over the best one (0.8085). Any sentence ranking projector against LoRA on Pitt overall should carry this spread.
- Files: release/scores/part20/POD1b/std_ft_seed{0,1,2,3}_pitt_oof.csv and .sidecar.json, std_ft_seed_range_pitt.json

Status now: released here. Files: release/scores/part20/POD1b/std_ft_seed0_pitt_oof.csv (authors' machine path) to release/scores/part20/POD1b/std_ft_seed3_pitt_oof.csv (authors' machine path), release/scores/part20/POD1b/std_ft_seed_range_pitt.json (authors' machine path), release/scores/part20/rows/POD1b.tsv (authors' machine path). The final tex compares LoRA with the paper's projector run only, not with the seed range: "Low-rank adaptation (LoRA) \cite{lora} of the language model (rank 8, all attention projections of the 28 layers, 5.0\,M trainable parameters against 4.6\,M for the projector) reaches 0.825 on Pitt, 0.06 [0.02, 0.10] above the projector run, and 0.55 on its conflict segments, and training both together gives 0.77."
