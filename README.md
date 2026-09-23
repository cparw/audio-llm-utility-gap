# The Representation Utility Gap in Audio LLMs for Clinical Speech

Chaitanya Parwatkar, Nima Kelidari, Minoo Ahmadi, Ashutosh Chaubey, Mohammad Soleymani.
University of Southern California, Los Angeles, CA, USA. Submitted to ICASSP 2027.
Corresponding author: soleymani@ict.usc.edu.

The paper asks whether an audio LLM's answer to a clinical question uses the information that its own
hidden states hold. We train linear probes on the frozen hidden states of Qwen2.5-Omni and ask the same
model the same Yes or No question on the same clips, on seven datasets in three conditions that differ in
how much the spoken words carry: Parkinson's read speech, depression interviews and Alzheimer's picture
descriptions. The encoder probes reach 0.75 to 0.92 AUC on the Parkinson's sets and 0.77 on Pitt, while the
zero-shot answer stays between 0.56 and 0.71 on the Parkinson's and Alzheimer's sets. On depression, where the
model hears the interview, the answer is above the best probe. On segments where the words contradict the
diagnosis, four of six audio LLMs fall below chance on depression. Retraining only the projector raises the
answer on the Parkinson's and Alzheimer's sets and lowers it on depression.

This repository will be released when the paper is accepted. Until then it is private.

This README is a lookup. Every cell of Table 1, every cell of Table 2 and every number in the text of the
paper is listed below with the file it was read from, its n and its estimator. Values are AUC against the
clinical label, to 4 decimals where the source file carries them, beside the 2 decimal value the paper prints.
The paper text used here is the copy pasted on 23 September 2026 at 05:58 UTC. Sentences the authors added
later that day are listed by topic in section f, and their wording will be matched to the final tex.

Paths in backticks with no note are files in this repository. A path followed by (authors' machine path) is a
file that stays on the authors' machine, usually because it holds data we may not redistribute. Three prefixes
are used for those: `release/` is /Users/chaitanyaparwatkar/Desktop/release/, `gdrive/` is
/Volumes/G-Drive Pro/paper work/paper1_local_runs/ and `DementiaBank/` is /Volumes/G-Drive Pro/DementiaBank/.

---

## b. What is and is not in this repository

In this repository:

- the scripts that produced the released numbers (`scripts/`)
- segment lists and manifests, with speech and clinical columns removed (`manifests/`)
- per-clip score files with their sidecar JSON (`scores/`)
- the GroupKFold(5) speaker fold assignments behind the probes (`folds/`, see section i)
- lookup tables of every cell with its estimator and source (`lookup/`)
- every prompt string (`prompts/prompts.json`)
- the figures (`figures/`), the audit log (`DISCREPANCIES.md`) and a file list with sizes and origins (`MANIFEST.txt`)

Not in this repository, on purpose:

- No audio. No `.wav`, `.flac`, `.mp3`, `.cha` or any other recording.
- No transcripts and no participant speech. No file holds utterance text or ASR output. Where a source table had
  a text column it was dropped. Counts derived from the text (`nw`, `ttr`, `retrace`, `fillers`, `unintel`,
  `errors`, `pauses`, `rate`, `p_text`, `val`, `npos`, `nneg`) are kept, because the conflict rules are defined on them.
- No clinical metadata beyond the binary label. Columns such as `age`, `sex`, `mmse`, `dx`, `sev` (PHQ-8 total)
  and PHQ items were dropped from every file. DementiaBank and E-DAIC distribute those under a data-use
  agreement and they are not ours to republish.
- No participant identifier beyond each dataset's own released ids.
- No hidden states and no model weights.
- The four long-form audit files are not here, because they quote participant and interviewer speech. They stay in
  `release/edaic_rerun/` (authors' machine path) as `AUDIT_1a1c_ad.txt`, `AUDIT_1b1c_edaic.txt`, `AUDIT_1d_confounds.txt`
  and `AUDIT_1e_dupes.txt` (authors' machine paths). `DISCREPANCIES.md` summarises them.

Reproducing a number needs the original corpora, each from its own custodian: DementiaBank (Pitt, ADReSS-2020,
ADReSS-M), E-DAIC, PC-GITA, NeuroVoz and MDVR-KCL.

---

## c. Estimator rules

Several cells exist in more than one version, because a probe was rerun on other folds or summarised in another
way. One version per cell is used. These are the rules the authors decided on 23 September 2026.

| cell | value used | estimator | file | values not used |
|---|---|---|---|---|
| Qwen2.5-Omni, Pitt, encoder probe | 0.7706 [0.7157, 0.8220] | mean of five per-repeat AUCs, nested GroupKFold(5) with the layer chosen by an inner GroupKFold(4), speakers renumbered with seeds 0 to 4 | `scores/part10/pitt_enc_nested5_oof.npz`, folds `folds/pitt_groupkfold5_mac_seed0.csv` to `folds/pitt_groupkfold5_mac_seed4.csv` | 0.7917 [0.7324, 0.8480] (AUC of the per-clip probability averaged over the five repeats), 0.7761 (the same probe on the pod split), 0.7969 (single split) |
| Qwen2.5-Omni, Pitt, answer-state probe | 0.7709 [0.7210, 0.8224] | logistic probe on the final language model stage at the first answer position, GroupKFold(5) by speaker, scaler fit inside the training fold | `lookup/readout_direction.csv` (row pitt, column auc_probe), interval from `release/edaic_rerun/part15/H_readout_arms.csv` (authors' machine path) | 0.7636 (nested answer-state probe, mean of five, a different estimator over all stages) |
| Qwen2.5-Omni, E-DAIC, LM probe | 0.7001 [0.6231, 0.7727] | mean of five per-repeat AUCs, nested, GroupKFold(5, shuffle, random_state 0 to 4) | `scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv`, folds `folds/edaic_full_groupkfold5_shuffle_rs0.csv` to `folds/edaic_full_groupkfold5_shuffle_rs4.csv` | 0.7088 [0.6232, 0.7874] (AUC of averaged probabilities), 0.6853 (original run) |
| Qwen2.5-Omni, E-DAIC, encoder probe | 0.5884 [0.5268, 0.6523] | as above | `scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv` | 0.5916 [0.5110, 0.6706] (AUC of averaged probabilities), 0.5930 (original run) |
| E-DAIC, LM probe minus zero-shot answer 0.8278, paired | -0.1276 [-0.1993, -0.0629] | one speaker draw per replicate, all five AUCs and the answer AUC recomputed inside the draw | `scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv` | -0.1190 [-0.1995, -0.0478] (averaged probabilities) |
| E-DAIC, encoder probe minus zero-shot answer 0.8278, paired | -0.2394 [-0.3188, -0.1572] | as above | `scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv` | -0.2361 [-0.3272, -0.1394] (averaged probabilities) |

The Pitt encoder value is the mean of the five per-repeat AUCs 0.7598, 0.7725, 0.7459, 0.7903 and 0.7843. The five
released fold files reproduce the per-clip scores of the npz exactly (largest difference 0.0). The interval in row 1
of `lookup/bootstrap_cis.csv`, [0.7324, 0.8480], belongs to the averaged-probability score 0.7917. The interval of 0.7706 is
[0.7157, 0.8220]. The older 0.7761 (`release/omni_final/omni_pitt_nested_repeats.json` (authors' machine path)) is the same five-repeat probe on the same
states, run on a pod whose GroupKFold tie order gave a different split (section i). The difference 0.7761 minus 0.7706
is 0.0055 [-0.0083, 0.0210], inside resampling noise (`release/edaic_rerun/part17/verify/T2_verify_omni_pitt_repeats.json` (authors' machine path)).

The E-DAIC per-repeat AUCs are LM 0.6918, 0.7062, 0.6893, 0.7055, 0.7078 and encoder 0.5964, 0.5826, 0.5962, 0.5608,
0.6060. The same rebuild gives the answer-state probe 0.7435 [0.6793, 0.8065] and the projector probe 0.5610
[0.4871, 0.6339], both mean of five. The unmodified probe script, rerun on Linux x86_64 with scikit-learn 1.9.1,
reproduces every published per-repeat AUC and all 25 layer choices per stream, and its outputs are byte-identical to
the published files (`scores/part17/T7b/T7b_edaic300_meanof5.json`).

### The E-DAIC window is the first 300 s

Every Qwen2.5-Omni audio number on the E-DAIC interview window is computed on the first 300 s of that window.
`Qwen2_5OmniProcessor` in transformers 5.17.0 uses a Whisper feature extractor with `chunk_length` 300 in the model's
own `preprocessor_config.json` (Hugging Face snapshot, quoted in `scores/part22/P22_step1_processor_config.md`), its
`truncation` argument defaults to True, and the waveform is cut at
`feature_extraction_sequence_utils.py:333` before the audio encoder runs. 217 of the 275 windows are longer than
300 s and are cut. The window itself is the participant's stitched speech capped at 900 s (89 windows reach the cap).
On the three longest windows the processor inputs for the whole window and for its first 300 s are byte-identical
(7500 audio tokens each). The zero-shot answer, the probes and the projector fine-tune on this window all inherit the cut.

With truncation turned off the model hears the whole window (up to 22,500 audio tokens). The zero-shot answer is then
0.8379 [0.7805, 0.8876] against 0.8278 [0.7703, 0.8822] with the cut, paired difference +0.0101 [-0.0309, 0.0518],
which is not significant. On the 217 cut windows alone it is 0.8571 against 0.8380, +0.0192 [-0.0309, 0.0703].
Files: `scores/part22/edaic_lifted_perclip.csv`, `scores/part22/edaic_lifted_auc.json`,
`scores/part22/p22_step2_truncation_proof.json`, `scores/part22/P22_step1_processor_config.md`.

Two zero-shot files exist for this window. The original run gives 0.8285 (`scores/edaic_windows/o25_full_zeroshot_scores.csv`)
and the rebuild that also produced the probes gives 0.8278 (`scores/part16/EDAICFULL/edaic_full_perclip.csv`).
Their per-clip p_yes correlate at r 0.99988 (`DISCREPANCIES.md`, item 102). The paired intervals use the rebuild, so that answer and probes come from
one run. Both print as 0.83.

### Other conventions

- AUC is the rank statistic with ties averaged, computed from the per-clip file:

  ```python
  r = pd.Series(score).rank().values
  n1 = (y == 1).sum()
  n0 = (y == 0).sum()
  auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
  ```

- Intervals are 2000 speaker-level bootstrap draws with `numpy.random.default_rng(0)`, percentiles 2.5 and 97.5.
  A paired interval draws the speaker list once per replicate and recomputes both quantities inside that draw.
- Probes: StandardScaler fit inside the training fold, `LogisticRegression(max_iter=2000, class_weight="balanced")`,
  outer GroupKFold(5) by speaker, inner GroupKFold(4) on the training speakers picks the layer (`scripts/core/nested_repeats_all.py`).
- `p_yes` is the probability of Yes given the answer is Yes or No, from the first-token Yes and No logits at the first
  answer position. `mass` or `answer_mass` is the share of the full next-token distribution those tokens hold.
- Probe values by conflict arm use the probe fitted on all 468 Pitt clips, restricted to the arm. A probe refit inside
  one arm is not used (Qwen3-Omni answer state: 0.6394 and 0.9032 used, 0.6145 and 0.9349 not used).
- Table 2 E-DAIC agreement counts each agreement segment once. The 483 agreement rows hold 311 distinct segments
  (repeats carry identical scores), so the agreement AUC is on 311 segments. The 483-row values are superseded (section e).
- Pitt has 468 segments from 227 participants. Participant 172 appears in the conflict manifest under both diagnosis
  groups, as `Control172` and `Dementia172`, so the files count 228 speaker labels and in some splits this one person
  (3 clips) sits in two folds. Grouping on the 227 participants instead changes no conclusion (`DISCREPANCIES.md`, item 123,
  `release/edaic_rerun/part17/T3_selfcheck_participant_grouped.csv` (authors' machine path), `release/edaic_rerun/part17/T3_selfcheck_partitions.csv` (authors' machine path)).

---

## d. Table 1, cell by cell

Table 1 of the paper is Qwen2.5-Omni on seven datasets: the zero-shot answer from the audio, the zero-shot answer from
the transcript alone, and the answer after retraining only the projector. The pasted tex prints 2 decimals. Every
printed cell agrees at 2 decimals with the file below. The one disagreement is the Pitt speaker count (see the note
under the table).

| dataset | clips (speakers) | column | paper | this release | n | estimator | file |
|---|---|---|---|---|---|---|---|
| PC-GITA | 1100 (100) | audio | 0.56 | 0.5635 | 1100 | zero shot, voice wording | `scores/part17/T5_zeroshot_perclip.csv` (dataset pcgita) |
| PC-GITA | 1100 (100) | text | 0.52 | 0.5246 (json), 0.5218 recomputed from the per-clip file | 1100 | transcript only, whisper-large-v3 ASR | `release/overnight2/text_new/o25_pcgita_text.csv` (authors' machine path) |
| PC-GITA | 1100 (100) | fine-tuned | 0.89 | 0.8862 | 1100 | projector only, 5 folds, 3 epochs, lr 1e-4, pooled out-of-fold | `release/omni_final/omnisft_pcgita_oof.csv` (authors' machine path) |
| NeuroVoz | 1270 (107) | audio | 0.70 | 0.6951 | 1270 | zero shot, voice wording | `scores/part17/T5_zeroshot_perclip.csv` (dataset neurovoz) |
| NeuroVoz | 1270 (107) | text | 0.54 | 0.5371 (json), 0.5372 recomputed | 1270 | transcript only, whisper-large-v3 ASR | `release/overnight2/text_new/o25_neurovoz_text.csv` (authors' machine path) |
| NeuroVoz | 1270 (107) | fine-tuned | 0.90 | 0.9028 | 1270 | projector only, as above | `release/omni_final/omnisft_neurovoz_oof.csv` (authors' machine path) |
| MDVR-KCL | 37 (37) | audio | 0.71 | 0.7143 | 37 | zero shot, voice wording | `scores/part17/T5_zeroshot_perclip.csv` (dataset kcl) |
| MDVR-KCL | 37 (37) | text | 0.76 | 0.7634 (ties averaged), 0.7619 in the lookup counts one tied pair as a loss | 37 | transcript only, whisper-large-v3 ASR | `release/overnight2/text_new/o25_kcl_text.csv` (authors' machine path) |
| MDVR-KCL | 37 (37) | fine-tuned | 0.70 | 0.6994 | 37 | projector only, as above | `release/omni_final/omnisft_kcl_oof.csv` (authors' machine path) |
| E-DAIC | 275 (275) | audio | 0.83 | 0.8285 (original run), 0.8278 (rebuild) | 275 | zero shot, voice wording, first 300 s of the interview window | `scores/edaic_windows/o25_full_zeroshot_scores.csv`, `scores/part16/EDAICFULL/edaic_full_perclip.csv` |
| E-DAIC | 275 (275) | text | 0.82 | 0.8230 | 275 | transcript only, full-window transcript | `release/overnight2/text_new/o25_edaicfull_text.csv` (authors' machine path) |
| E-DAIC | 275 (275) | fine-tuned | 0.64 | 0.6421 | 275 | projector only, as above, first 300 s of the window | `scores/edaic_windows/sft_full_oof.csv` |
| Pitt | 468 (227) | audio | 0.66 | 0.6578 | 468 | zero shot, recording wording, 30 s window | `scores/part17/T5_zeroshot_perclip.csv` (dataset pitt) |
| Pitt | 468 (227) | text | 0.66 | 0.6629 | 468 | transcript only, dataset transcripts, run of 20 Sep | `release/overnight2/text_new/o25_pitt_text.csv` (authors' machine path) |
| Pitt | 468 (227) | fine-tuned | 0.77 | 0.7673 | 468 | projector only, as above | `release/omni_final/omnisft_pitt_oof.csv` (authors' machine path) |
| ADReSSo | 237 (237) | audio | 0.62 | 0.6150 | 237 | zero shot, recording wording, 30 s window | `scores/part17/T5_zeroshot_perclip.csv` (dataset adresso) |
| ADReSSo | 237 (237) | text | 0.55 | 0.5542 | 237 | transcript only, whisper-large-v3 ASR | `release/overnight2/text_new/o25_adresso_text.csv` (authors' machine path) |
| ADReSSo | 237 (237) | fine-tuned | 0.82 | 0.8246 | 237 | projector only, as above | `release/omni_final/omnisft_adresso_oof.csv` (authors' machine path) |
| ADReSS-2020 | 156 (156) | audio | 0.62 | 0.6241 | 156 | zero shot, recording wording, 30 s window | `scores/part17/T5_zeroshot_perclip.csv` (dataset adress2020) |
| ADReSS-2020 | 156 (156) | text | 0.75 | 0.7538 | 156 | transcript only, dataset transcripts, run of 20 Sep | `release/overnight2/text_new/o25_adress2020_text.csv` (authors' machine path) |
| ADReSS-2020 | 156 (156) | fine-tuned | 0.77 | 0.7684 | 156 | projector only, as above | `release/omni_final/omnisft_adress2020_oof.csv` (authors' machine path) |

Notes on Table 1.

- The audio column's per-clip scores for the six datasets other than E-DAIC are in `scores/part17/T5_zeroshot_perclip.csv`, one
  block per dataset, copied from the run files `release/omni_final/omni_pcgita_zeroshot_scores.csv` (authors' machine path) and its siblings
  `omni_<dataset>_zeroshot_scores.csv` (authors' machine paths). The text and fine-tuned per-clip files stay on the authors' machine as listed.
- Pitt clips (speakers). The pasted tex prints 468 (228). The authors decided 468 (227): participant 172 is one person
  listed under two labels (section c). All Pitt score files carry 228 speaker labels.
- The PC-GITA, NeuroVoz and MDVR-KCL text cells differ in the 4th decimal between the stored json value and the per-clip
  recomputation. The printed 2 decimal values are the same either way (`DISCREPANCIES.md`, item 79).
- Two Pitt transcript runs exist. The Table 1 cell is the run of 20 September, 0.6629 [0.5983, 0.7245]. The run of
  15 September gives 0.6975 (`gdrive/omni_pitt_text.csv` (authors' machine path)). Their per-clip scores correlate at r 0.7640.
- E-DAIC. The audio, the probes and the fine-tune see the first 300 s of the window (section c). The transcript cell
  uses the transcript of the full window.
- The per-cell lookup with all models and streams is `lookup/master_lookup.csv`. Its `source` column gives the path of
  every value. Superseded sources are listed in `lookup/SUPERSEDED.csv`. The audio and text cells above are rows of
  the lookup and agree with the pasted tex at 2 decimals (MDVR-KCL text is stored there as 0.7619, see the table). The
  fine-tuned cells of the six datasets other than E-DAIC are not rows of the lookup. They come from the fine-tune sidecars
  (`auc_oof` in `release/omni_final/omnisft_pitt_sft.json` (authors' machine path) and the matching `omnisft_<dataset>_sft.json` (authors' machine paths)) and agree with the
  per-clip recomputation.

---

## e. Table 2, cell by cell

Table 2 is the conflict test: the zero-shot answer on segments where the words point away from the diagnosis, and on
matched agreement segments from the same speakers. E-DAIC has 483 conflict segments and 311 distinct agreement
segments from 138 speakers. Pitt has 146 conflict segments (100 speakers) and 322 agreement segments (175 speakers).
Intervals are speaker bootstrap. The paired column is conflict minus agreement with one speaker draw per replicate.

### Depression, E-DAIC

| model | conflict (483) | agreement (311 distinct) | pasted tex prints | paired conflict minus agreement | per-clip file |
|---|---|---|---|---|---|
| Qwen2-Audio | 0.2280 [0.1816, 0.2782] | 0.9608 [0.9351, 0.9814] | 0.23, 0.95 | -0.7328 [-0.7780, -0.6837] | `scores/edaic_conflict_v2/p14_q2a_zeroshot_scores.csv` |
| Qwen2.5-Omni | 0.2218 [0.1829, 0.2632] | 0.9647 [0.9376, 0.9857] | 0.22, 0.95 | -0.7429 [-0.7869, -0.6944] | `scores/edaic_conflict_v2/p14_o25_zeroshot_scores.csv` |
| Qwen3-Omni | 0.3008 [0.2454, 0.3641] | 0.9727 [0.9507, 0.9904] | 0.30, 0.96 | -0.6719 [-0.7266, -0.6105] | `scores/edaic_conflict_v2/p14_q3o_zeroshot_scores.csv` |
| Audio Flamingo 2 | 0.5286 [0.4512, 0.6015] | 0.6297 [0.5451, 0.7087] | 0.53, 0.61 | -0.1011 [-0.1647, -0.0384] | `scores/edaic_conflict_v2/p14_af2.csv` |
| Audio Flamingo 3 | 0.4235 [0.3606, 0.4880] | 0.8425 [0.7963, 0.8854] | 0.42, 0.81 | -0.4190 [-0.4878, -0.3461] | `scores/edaic_conflict_v2/p14_af3.csv` |
| Kimi-Audio | 0.5258 [0.4501, 0.5977] | 0.8936 [0.8479, 0.9329] | 0.53, 0.87 | -0.3678 [-0.4394, -0.3020] | `scores/edaic_conflict_v2/p14_kimi_zeroshot_scores.csv` |

All E-DAIC cells are in `scores/part17/T1_table2_distinct.csv`, and the per-clip rows of the 311 distinct
agreement segments are in `scores/part17/T1_perclip_distinct_agreement.csv`. The conflict cells printed in
the pasted tex match the files. The agreement cells printed there are the old 483-row values. On the 311 distinct
segments every agreement cell changes at 2 decimals, to 0.96, 0.96, 0.97, 0.63, 0.84 and 0.89.

Superseded and not used: agreement on all 483 rows (repeated segments counted each time), Qwen2-Audio 0.9478,
Qwen2.5-Omni 0.9482, Qwen3-Omni 0.9638, Audio Flamingo 2 0.6083, Audio Flamingo 3 0.8095, Kimi-Audio 0.8744
(`scores/part17/T1_table2_distinct.csv`, rows marked OLD_483rows).

The E-DAIC pairs come from the Part 14 build: RoBERTa sentiment (`cardiffnlp/twitter-roberta-base-sentiment-latest`) at
threshold 0.70 on participant segments of at least 25 words and 5 s of speech, conflict when a participant with PHQ-8
of 15 or more speaks positively or a control with PHQ-8 of 4 or less speaks negatively (`manifests/part14_manifest_new.json`,
`manifests/part14_sentiment.json`, segment list `manifests/part14_manifest_new.csv`).

### Alzheimer's, Pitt

| model | conflict (146) | agreement (322) | pasted tex prints | paired conflict minus agreement | per-clip file |
|---|---|---|---|---|---|
| Qwen2-Audio | 0.4132 [0.3005, 0.5423] | 0.6985 [0.6372, 0.7597] | 0.41, 0.70 | -0.2853 [-0.3950, -0.1485] | `scores/part16/M/M8_perclip.csv`, tag primary |
| Qwen2.5-Omni | 0.5322 [0.4216, 0.6447] | 0.7054 [0.6431, 0.7658] | 0.53, 0.71 | -0.1731 [-0.2958, -0.0392] | `scores/part16/M/M8_perclip.csv`, tag primary |
| Qwen3-Omni | 0.6066 [0.4956, 0.7123] | 0.8231 [0.7664, 0.8720] | 0.61, 0.82 | -0.2165 [-0.3242, -0.1052] | `scores/part16/M/M8_perclip.csv`, tag primary |
| Audio Flamingo 2 | 0.6310 [0.5336, 0.7260] | 0.5544 [0.4760, 0.6296] | 0.63, 0.55 | +0.0766 [-0.0307, 0.1869] | `scores/part16/M/M8_perclip.csv`, tag af2_pitt468 (canonical) |
| Audio Flamingo 3, copy part3 | 0.6169 [0.5188, 0.7209] | 0.5822 [0.5080, 0.6528] | 0.61, 0.59 | +0.0347 [-0.0698, 0.1474] | `scores/part16/M/M8_perclip.csv`, tag primary |
| Audio Flamingo 3, copy part10 | 0.6075 [0.5063, 0.7129] | 0.5855 [0.5144, 0.6554] | 0.61, 0.59 | +0.0220 [-0.0846, 0.1357] | `scores/part20/POD6/C_landed/C_af3_pitt_part10_arms.csv` |
| Kimi-Audio | 0.7197 [0.6201, 0.8147] | 0.7227 [0.6560, 0.7896] | 0.72, 0.72 | -0.0030 [-0.1134, 0.1065] | `scores/part16/M/M8_perclip.csv`, tag primary (canonical) |

All Pitt cells are in `scores/part16/M/M8_conflict_cells.csv`. The per-clip rows in `scores/part16/M/M8_perclip.csv`
carry the model, the tag, the arm and the run file each score was copied from: Qwen2-Audio `gdrive/probe2/pitt_zeroshot_scores.csv` (authors' machine path),
Qwen2.5-Omni `release/omni_final/omni_pitt_zeroshot_scores.csv` (authors' machine path), Qwen3-Omni `release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv` (authors' machine path),
Audio Flamingo 2 `gdrive/af2_results/af2_pitt468.csv` (authors' machine path), Audio Flamingo 3 `release/overnight2/part3/af3_pitt.csv` (authors' machine path) and
`release/overnight2/part10/af3_pitt.csv` (authors' machine path), Kimi-Audio `release/overnight2/part3/kimi_pitt_zeroshot_scores.csv` (authors' machine path). The arm of each clip
comes from the Pitt conflict manifest, released without its text and clinical columns as
`manifests/pitt_conflict_manifest_468.csv`.

Audio Flamingo 3 on Pitt is OPEN. Two score files exist and they disagree: part3 gives 0.6169 and 0.5822, part10 gives
0.6075 and 0.5855. The paper prints 0.61 and 0.59, which matches part10 and does not match part3 at 2 decimals. A rerun
on a fresh pod reproduces the part10 copy on 10 of 10 check windows within 1e-3 and the part3 copy on 5 of 10
(`DISCREPANCIES.md`, item 147). The authors have not decided which copy is canonical.

The Kimi-Audio and Audio Flamingo 2 cells use the files the authors named canonical. The other copies are superseded:
Kimi-Audio `release/overnight/kimi/kimi_pitt_zeroshot_scores.csv` (authors' machine path) gives 0.7078 and 0.7014, and Audio Flamingo 2
`gdrive/af2_pitt_audio.csv` (authors' machine path) gives 0.6691 and 0.4895 (`lookup/SUPERSEDED.csv`).

The Pitt agreement arm has 322 rows and 322 distinct segments, so the distinct-segment rule changes nothing on Pitt.

Lines 73 to 75 of `lookup/bootstrap_cis.csv` give Qwen2.5-Omni Pitt arm values 0.4444 and 0.7664 from an older score
file (`DementiaBank/ad_scores_omni.csv` (authors' machine path)). They do not match the Table 2 cells 0.5322 and 0.7054 and are not used.

---

## f. Every number in the text

### Numbers in the pasted tex

One row per number in the text of the pasted tex (tables are in sections d and e, commented-out text and grant
numbers are left out). "line" is the line of the pasted copy. Status MATCH means the file value rounds to the printed
value. MISMATCH means it does not, and the row says what the file gives.

| line | number in the text | value in the file | file | estimator | status |
|---|---|---|---|---|---|
| 34, 154 | probes reach 0.75 to 0.92 | Qwen2.5-Omni encoder probe: MDVR-KCL 0.7506 lowest, NeuroVoz 0.9213 highest, PC-GITA 0.8941 (Pitt 0.7706, ADReSSo 0.8752 and ADReSS-2020 0.8043 fall inside) | `release/omni_final/omni_kcl_nested_repeats.json` (authors' machine path), `release/omni_final/omni_neurovoz_nested_repeats.json` (authors' machine path), `release/omni_final/omni_pcgita_nested_repeats.json` (authors' machine path), `scores/part10/pitt_enc_nested5_oof.npz` | mean of five per-repeat AUCs, nested | MATCH. On NeuroVoz the LM probe (0.9264) is higher than the encoder probe, so "best probe" there is the LM probe |
| 34, 184 | four of six models fall below chance | E-DAIC conflict arm below 0.5: Qwen2-Audio 0.2280, Qwen2.5-Omni 0.2218, Qwen3-Omni 0.3008, Audio Flamingo 3 0.4235 | `scores/part17/T1_table2_distinct.csv` | zero shot, 483 conflict segments | MATCH on E-DAIC. On Pitt one of six is below 0.5 (Qwen2-Audio 0.4132) |
| 48, 97 | 32 encoder stages, 29 language model stages | 32 encoder layers and 29 language model stages | `scripts/make_fig1_v4.py` (docstring) | design | MATCH |
| 60, 81 | seven datasets, six audio LLMs | seven datasets and six models in Tables 1 and 2 | `lookup/master_lookup.csv` | design | MATCH |
| 88 | PC-GITA 1100 clips, NeuroVoz 1270 clips, MDVR-KCL 37 speakers | n 1100 (100 speakers), 1270 (107), 37 (37) | `scores/part17/T5_zeroshot_perclip.csv` | clip counts | MATCH |
| 90 | E-DAIC 275 participants | 275 clips, 275 speakers | `scores/part16/EDAICFULL/edaic_full_perclip.csv` | clip count | MATCH |
| 90 | capped at 15 minutes for the 89 longest | windows capped at 900 s: 89 of 275 | `scores/part22/P22_step1_processor_config.md`, `manifests/edaic_windows_full.json` | window cap | MATCH for the window. For Qwen2.5-Omni the processor keeps only the first 300 s, so the model does not hear the whole interview on 217 of 275 windows (section c) |
| 92 | Pitt 468 segments from 228 speakers | 468 segments, 228 speaker labels, 227 participants | `manifests/pitt_conflict_manifest_468.csv` | counts | MISMATCH on speakers: the authors decided 227 (participant 172 carries two labels) |
| 92 | ADReSSo 237 speakers, ADReSS-2020 108 training and 48 test | 237, 108 and 48 | `release/edaic_rerun/part17/part12_prep/handcheck_values.csv` (authors' machine path) | counts | MATCH |
| 92 | models hear the first 30 s | 30 s cap in the zero-shot scorers | `scripts/core/extract_probe_layers.py` | design | MATCH |
| 96 | C=1, 2000 iterations, 5 folds | `LogisticRegression(max_iter=2000, class_weight="balanced")` (C left at its default 1.0), GroupKFold(n_splits=5) | `scripts/core/nested_repeats_all.py` | design | MATCH |
| 99 | learning rate 1e-4, three epochs, 5 folds | lr 0.0001, epochs 3, folds 5 in every fine-tune sidecar | `release/omni_final/omnisft_pitt_sft.json` (authors' machine path), `scripts/core/sft_projector.py` | design | MATCH |
| 101 | two other wordings move the answers by 0.02 on average | no file holds this average. The per-wording runs are `release/omni_final/omni_pitt_p2.json` (authors' machine path) and the other `omni_<dataset>_p2.json` and `omni_<dataset>_p3.json` files in `release/omni_final/` (authors' machine path) | none | pending verification | UNMATCHED |
| 104 | segments of at least 25 words and 5 s of speech, probability of at least 0.70, PHQ-8 of 15 or more, 4 or less | nw at least 25, speech_s at least 5 (and span at least 8), threshold 0.7, sev at least 15 or at most 4 | `manifests/part14_manifest_new.json`, `manifests/part14_sentiment.json` | design | MATCH |
| 104 | 208 of the first kind and 280 of the second | 208 and 280 conflict segments before pairing. After pairing 205 and 278 are kept, because 5 had no agreement partner | `scores/part16/M/M2_distilbert_sentiment.csv`, `manifests/part14_manifest_new.json` | counts | MATCH (the counts are before pairing) |
| 104 | 483 pairs over 138 speakers | 483 pairs, 138 speakers | `scores/part17/T1_table2_distinct.csv` | counts | MATCH |
| 104 | six fluency markers | ttr, retraces, fillers, unintelligible, errors, pauses | `scripts/upstream/build_ad_conflict.py` | design | MATCH |
| 104 | speaking rate alone reaches 0.92 | no result file holds 0.92. It appears only in a code comment (`scripts/upstream/build_ad_conflict.py`, line 73). Speaking rate alone on the 468 kept segments gives 0.7082 | `release/edaic_rerun/part17/part12_prep/handcheck_values.csv` (authors' machine path) (computed from `manifests/pitt_all_segments.csv`) | clip-level rank AUC, direction free | MISMATCH |
| 104 | five speaker-disjoint folds, top or bottom third | 5 folds, 33rd and 67th percentiles of the out-of-fold score | `scripts/upstream/build_ad_conflict.py`, `reports/PART13_tertiles.txt` | design | MATCH |
| 104 | 146 in conflict and 322 in agreement | 146 and 322 | `manifests/pitt_conflict_manifest_468.csv` | counts | MATCH |
| 104 | the median segment fits inside the 30 s window, and 91 percent have at least half their words inside it | no verified file holds these two numbers for Pitt. The status audit links them to the older 390-row E-DAIC set in `manifests/part7_heard_window.csv` | none | pending verification | UNMATCHED |
| 108 | patients are seven years older | 6.58 years at clip level (71.74 against 65.17), 7.54 at speaker level | `DementiaBank/pitt_conflict_manifest.csv` (authors' machine path, not released), age column, computed in `release/edaic_rerun/part17/part12_prep/red_spans_dryrun.csv` (authors' machine path) | mean age difference | MATCH at clip level. The speaker-level gap rounds to 8 |
| 108 | 275 clips from 129 age and sex matched speakers | 275 clips, 129 speakers | `release/omni/pitt_probe_estimators.csv` (authors' machine path) | counts | MATCH |
| 143 | zero-shot between 0.56 and 0.71 on Parkinson's and Alzheimer's | PC-GITA 0.5635 lowest, MDVR-KCL 0.7143 highest | `scores/part17/T5_zeroshot_perclip.csv` | zero shot | MATCH |
| 145 | for AD the transcript alone matches the audio on Pitt, beats it on ADReSS-2020 and trails it on ADReSSo | transcript against audio: Pitt 0.6629 and 0.6578, ADReSS-2020 0.7538 and 0.6241, ADReSSo 0.5542 and 0.6150 | `lookup/master_lookup.csv` | zero shot | MATCH |
| 145 | on the Parkinson's sets the transcript sits near chance, since every speaker reads the same sentences | transcript PC-GITA 0.5246, NeuroVoz 0.5371, MDVR-KCL 0.7634 [0.5936, 0.9118] | `lookup/master_lookup.csv`, `scores/part16/M/M10_kcl_transcript.csv` | zero shot on the transcript | MISMATCH for MDVR-KCL, whose speakers read two different passages (`DISCREPANCIES.md`, item 70, open) |
| 145 | on E-DAIC the audio answer matches the transcript alone | audio 0.8285, transcript 0.8230, paired transcript minus audio -0.0055 [-0.0566, 0.0431] | `scores/part16/M/MEDAIC_full_window.csv` | paired speaker bootstrap | MATCH. The audio is the first 300 s of the window, the transcript covers the whole window |
| 145 | Qwen3-Omni and Audio Flamingo 3 reach 0.87 and 0.86 | 0.8741 and 0.8565 | `scores/edaic_windows/q3o_full_zeroshot_scores.csv`, `scores/edaic_windows/af3_full.csv` | zero shot, interview window | MATCH. Whether these two processors also cut long audio was not tested (PART 22 covers Qwen2.5-Omni) |
| 145 | Qwen2-Audio, which hears 30 s, stays at 0.63 | 0.6330 | `scores/edaic_windows/q2a_full_zeroshot_scores.csv` | zero shot | MATCH |
| 154 | 0.77 on the Pitt dataset | 0.7706 | `scores/part10/pitt_enc_nested5_oof.npz` | mean of five per-repeat AUCs | MATCH |
| 154 | all 708 Pitt segments: encoder probe 0.80, zero-shot answer 0.67 | 708 segments, 0.7968 and 0.6695 | `scores/pitt_all/o25_pitt_all_nested_repeats.json`, `scores/pitt_all/o25_pitt_all_zeroshot_scores.csv` | mean of five, zero shot | MATCH |
| 154 | Qwen2-Audio probes 0.77 to 0.93, answers 0.54 to 0.68 | encoder MDVR-KCL 0.7714 to NeuroVoz 0.9324, zero shot PC-GITA 0.5369 to ADReSS-2020 0.6824 | `release/omni_final/q2a_kcl_nested_repeats.json` (authors' machine path), `release/omni_final/q2a_neurovoz_nested_repeats.json` (authors' machine path), `gdrive/probe2/pcgita_zeroshot_scores.csv` (authors' machine path), `gdrive/probe2/adress2020_zeroshot_scores.csv` (authors' machine path) | mean of five, zero shot | MATCH |
| 154 | Qwen3-Omni 0.77 to 0.94 against 0.60 to 0.78 | encoder MDVR-KCL 0.7661 to NeuroVoz 0.9372, zero shot PC-GITA 0.5967 to ADReSSo 0.7774 | `release/overnight2/q3o_new/q3o_kcl_nested_repeats.json` (authors' machine path), `release/overnight2/q3o_new/q3o_neurovoz_nested_repeats.json` (authors' machine path), `release/overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv` (authors' machine path), `release/overnight2/q3o_new/q3o_adresso_zeroshot_scores.csv` (authors' machine path) | mean of five, zero shot | MATCH |
| 154 | encoder probe beats eGeMAPS on all seven datasets, by 0.06 on PC-GITA to 0.18 on ADReSSo | MDVR-KCL +0.0095, PC-GITA +0.0609, E-DAIC first 30 s +0.0657, NeuroVoz +0.0774, ADReSS-2020 +0.0921, Pitt +0.1159 (0.7706 minus 0.6547), ADReSSo +0.1810 | `release/overnight/egemaps_baseline.csv` (authors' machine path), `release/edaic_rerun/part17/part12_prep/red_spans_dryrun.csv` (authors' machine path) | encoder mean of five minus eGeMAPS out-of-fold AUC | MISMATCH at the low end: the smallest margin is MDVR-KCL, 0.0095. The range holds without MDVR-KCL. The Pitt delta stored in the file uses the superseded 0.7761 |
| 154 | NeuroVoz 0.87 by spectral centroid, PC-GITA 0.69 | quiet-frame spectral centroid 0.8650 (NeuroVoz), noise floor 0.6872 (PC-GITA) | `manifests/confounds/confound_auc.csv` | speaker-level AUC, direction free | MATCH |
| 154 | Alzheimer's and E-DAIC sets stay below 0.64 | largest such feature 0.6358 (ADReSS-2020 noise floor) | `manifests/confounds/confound_auc.csv` | as above | MATCH |
| 154 | age alone 0.71, on age matched clips 0.39 | 0.7111 and 0.3866 | `release/omni/pitt_probe_estimators.csv` (authors' machine path) | clip-level AUC | MATCH |
| 154 | age matched: encoder probe 0.80, zero-shot 0.65 | 0.7953 and 0.6540 | `release/omni/pitt_probe_estimators.csv` (authors' machine path) | mean of five, zero shot, clip level | MATCH |
| 154 | LM probe on Alzheimer's rises to 0.80 | Pitt LM probe 0.8032 | `release/omni_final/omni_pitt_nested_repeats.json` (authors' machine path) | mean of five | MATCH |
| 154 | the answer state itself achieves 0.77 | 0.7709 | `lookup/readout_direction.csv` | final stage, first answer position | MATCH |
| 154 | cosine 0.01 in magnitude, no larger than between random directions | cosine -0.0109, random mean absolute cosine 0.0135 | `lookup/readout_direction.csv` | Yes minus No readout direction | MATCH |
| 154 | removing the readout direction leaves the probe at 0.77 | 0.7662, and 0.7661 when each held-out fold is standardised on its own | `lookup/readout_direction.csv`, `scores/part20/POD5/POD5_direction_origrerun_check.csv` | probe with the direction projected out | MATCH |
| 154 | on E-DAIC the best probe is 0.69, at the language model | LM probe 0.7001 (decided). 0.69 is the superseded original run 0.6853. The answer-state probe of the same run is 0.7435 | `scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv` | mean of five | MISMATCH, wording to be matched to the final tex |
| 154 | below the answer of 0.83 | 0.8285 (original run), 0.8278 (rebuild) | `scores/edaic_windows/o25_full_zeroshot_scores.csv`, `scores/part16/EDAICFULL/edaic_full_perclip.csv` | zero shot | MATCH |
| 184 | every model except Audio Flamingo 2 above 0.80 on agreement | lowest of the other five is Audio Flamingo 3, 0.8425 on the 311 distinct segments (0.8095 on the superseded 483 rows) | `scores/part17/T1_table2_distinct.csv` | zero shot | MATCH |
| 184 | intervals for the difference between the arms exclude zero for all six models | all six paired intervals exclude zero, the closest is Audio Flamingo 2, -0.1011 [-0.1647, -0.0384] | `scores/part17/T1_table2_distinct.csv` | paired speaker bootstrap, 311 distinct agreement segments | MATCH |
| 184 | the three Qwen models drop by 0.18 to 0.29 on Pitt | agreement minus conflict at 4 decimals: Qwen2.5-Omni 0.1731, Qwen3-Omni 0.2165, Qwen2-Audio 0.2853 | `scores/part16/M/M8_conflict_cells.csv` | paired speaker bootstrap | MISMATCH at the low end by rounding: 0.17 to 0.29 from the files. 0.18 is the difference of the printed cells 0.71 and 0.53 |
| 184 | Kimi-Audio holds | conflict minus agreement -0.0030 [-0.1134, 0.1065] | `scores/part16/M/M8_conflict_cells.csv` | paired | MATCH |
| 184 | both Audio Flamingo models sit near chance on both sets | Audio Flamingo 2 conflict 0.6310 [0.5336, 0.7260], agreement 0.5544 [0.4760, 0.6296]. Audio Flamingo 3 conflict 0.6169 [0.5188, 0.7209] (part3) or 0.6075 [0.5063, 0.7129] (part10), agreement 0.5822 [0.5080, 0.6528] or 0.5855 [0.5144, 0.6554] | `scores/part16/M/M8_conflict_cells.csv` | zero shot by arm | Both conflict intervals lie above 0.5, so "near chance" does not hold for the conflict arm (`DISCREPANCIES.md`, item 42, open) |
| 230 | MDVR-KCL has 37 speakers | 37 | `release/omni_final/omnisft_kcl_oof.csv` (authors' machine path) | count | MATCH |
| 230 | fine-tuning improves every dataset except MDVR-KCL and E-DAIC | paired fine-tune minus zero shot: PC-GITA +0.3227 [0.2391, 0.4040], NeuroVoz +0.2078 [0.1588, 0.2578], MDVR-KCL -0.0149 [-0.2709, 0.2364], Pitt +0.1095 [0.0454, 0.1728], ADReSSo +0.2096 [0.1294, 0.2867], ADReSS-2020 +0.1443 [0.0490, 0.2370]. E-DAIC zero shot minus fine-tune 0.1863 [0.1098, 0.2696] | `release/edaic_rerun/part15/C_intervals.csv` (authors' machine path) | paired speaker bootstrap | MATCH |
| 230 | Pitt conflict segments stay at 0.52, agreement rises to 0.86 | 0.5189 [0.4151, 0.6300] and 0.8593 [0.8063, 0.9069] | `release/omni_final/omnisft_pitt_oof.csv` (authors' machine path) | projector fine-tune by arm | MATCH. Reruns of the same recipe give conflict 0.4697 to 0.5666 (section g) |
| 230 | PC-GITA projector lifts NeuroVoz from 0.70 to 0.74 | 0.6951 to 0.7375 | `release/overnight2/part6/transfer_pcgita_transfer.json` (authors' machine path) | projector trained on all PC-GITA clips, scored on NeuroVoz | MATCH |
| 230 | NeuroVoz projector lifts PC-GITA from 0.56 to 0.68 | 0.5635 to 0.6761 | `release/overnight2/part6/transfer_neurovoz_transfer.json` (authors' machine path) | as above, reversed | MATCH |
| 248 | (red in the paste) fine-tuning gives a significant gain in AD and MDD detection | AD gains as in line 230. On MDD the fine-tuned E-DAIC answer drops from 0.8285 to 0.6421 | `release/edaic_rerun/part15/C_intervals.csv` (authors' machine path) | paired speaker bootstrap | MISMATCH for MDD, holds for AD |

### Sentences added on 23 September (wording to be matched to the final tex)

These sentences are not in the pasted copy. Each row gives the topic, the verified values and the file. The wording
will be matched to the final tex.

| topic | verified values | file | estimator |
|---|---|---|---|
| paired gap, probe minus zero-shot answer, per dataset | PC-GITA 0.3454 [0.2636, 0.4275], NeuroVoz 0.2209 [0.1660, 0.2783], MDVR-KCL 0.0625 [-0.1813, 0.2900], ADReSSo 0.2748 [0.1920, 0.3526], ADReSS-2020 0.1637 [0.0557, 0.2693], E-DAIC first 30 s -0.0297 [-0.1291, 0.0752] | `scores/part16/M/M7_gap_per_dataset.csv`, per clip `scores/part16/M/M7_perclip/pcgita.csv` and the other files in that folder | single-split nested encoder probe minus zero shot, paired speaker bootstrap (the single-split probes are PC-GITA 0.9089, NeuroVoz 0.9160, MDVR-KCL 0.7768, ADReSSo 0.8899, ADReSS-2020 0.7878, E-DAIC 0.6324) |
| paired gap on Pitt, corrected source | probe 0.7706 [0.7157, 0.8220], probe minus zero shot +0.1128 [0.0455, 0.1768] | `scores/part16/M/M7_pitt_FIXED.json`, `scores/part16/M/M7_perclip/pitt_FIXED.csv` | mean of five per-repeat AUCs, paired |
| paired gap on E-DAIC, interview window | LM probe minus answer -0.1276 [-0.1993, -0.0629], encoder probe minus answer -0.2394 [-0.3188, -0.1572] | `scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv` | mean of five, paired, first 300 s |
| Qwen3-Omni probe cells | Pitt encoder 0.8122 [0.7613, 0.8625], PC-GITA encoder 0.8969 [0.8322, 0.9454] (AUC of averaged probabilities 0.8354 and 0.9109, not used). Pitt LM 0.8477, answer state 0.8268 (older run) against 0.8445 and 0.8237 in the rerun | `scores/part16/POD3/q3o_pitt_encoder_nested_oof.csv`, `scores/part16/POD3/q3o_pcgita_encoder_nested_oof.csv`, `lookup/master_lookup.csv` | mean of five per-repeat AUCs |
| Qwen3-Omni zero-shot, two runs | Pitt 0.7619 (file behind Tables 1 and 2) and 0.7673 (rerun). The rerun gives 0.7673 with sdpa attention and 0.7679 with eager attention. PC-GITA 0.5967 and 0.6015 | `release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv` (authors' machine path), `release/edaic_rerun/part16/POD3/q3o_pitt_zeroshot_scores.csv` (authors' machine path) | zero shot |
| direction test, Qwen2.5-Omni | cosine -0.0109 (random 0.0135), probe 0.7709, AUC along the readout direction 0.6608, probe with the direction removed 0.7662 | `lookup/readout_direction.csv` | final stage, first answer position |
| direction test, Qwen3-Omni | cosine 0.0051 (random 0.0177 [0.0008, 0.0498]), probe 0.8307 [0.7833, 0.8741], along the direction 0.7656 [0.7088, 0.8179], probe with the direction removed 0.8276 [0.7795, 0.8696] | `scores/part16/pod_sync/q3o_perp_cols.npz` | as above |
| direction test, Qwen2-Audio | cosine 0.0145 (random 0.0123 [0.0005, 0.0321]), probe 0.8223 [0.7742, 0.8671], along the direction 0.6174 [0.5539, 0.6800], probe with the direction removed 0.8114 [0.7646, 0.8577] | `scores/part17/T3_q2a_direction.csv`, `scores/part17/T3_q2a_direction_summary.csv` | as above, held-out fold standardised |
| readout equality | the AUC along the Yes minus No readout direction equals the zero-shot answer AUC: Qwen2-Audio 0.6174 against 0.6173, conflict 0.4136 against 0.4132, agreement 0.6984 against 0.6985 (Spearman 0.99996 between the projection and p_yes). Qwen2.5-Omni answer-state probe by arm: all 0.7709 [0.7210, 0.8224], conflict 0.5272 [0.4229, 0.6354], agreement 0.8672 [0.8261, 0.9064] | `scores/part17/T3_q2a_direction.json`, `release/edaic_rerun/part15/H_readout_arms.csv` (authors' machine path) | full-fit probe restricted to the arm |
| channel normalisation (loudness to -23 LUFS plus shaped noise) | NeuroVoz: five quiet-frame features 0.7977 to 0.6025 (the acceptance bar of 0.60 is missed), spectral centroid alone 0.7782 to 0.5603, zero shot 0.7108 to 0.6991 (paired -0.0117 [-0.0332, 0.0098]), encoder probe 0.9233 to 0.8924 (paired -0.0308 [-0.0608, -0.0012]). PC-GITA: five features 0.6657 to 0.5590 (passes), zero shot 0.5728 to 0.5674 (-0.0054 [-0.0437, 0.0357]), encoder probe 0.8959 to 0.8298 (-0.0661 [-0.1181, -0.0180]) | `release/edaic_rerun/part16/POD4/channel_norm_neurovoz.csv` (authors' machine path), `release/edaic_rerun/part16/POD4/channel_norm_pcgita.csv` (authors' machine path), `release/edaic_rerun/part16/POD4/boot4a_nv.json` (authors' machine path), `release/edaic_rerun/part16/POD4/boot4a_pg.json` (authors' machine path) | GroupKFold(5) by speaker, paired speaker bootstrap |
| transcript on the E-DAIC pairs | Qwen2.5-Omni transcript: conflict 0.1498 [0.1131, 0.1887], agreement 0.9620 [0.9345, 0.9822] (311 distinct), Pearson r with its audio answer 0.8442, same Yes or No on 0.8634 of clips. Qwen2-Audio transcript: 0.2359 [0.1851, 0.2886] and 0.9199 [0.8791, 0.9560], r 0.5155, same decision 0.7122. Qwen3-Omni transcript: 0.2243 [0.1757, 0.2742] and 0.9523 [0.9249, 0.9750], r 0.7634, same decision 0.8313 | `scores/part17/T1_table2_distinct.csv`, `release/edaic_rerun/part16/POD1/p14_o25_1a_joined.csv` (authors' machine path), `release/edaic_rerun/part16/POD1/p14_q2a_1a_joined.csv` (authors' machine path), `scores/part20/POD3c/p14_q3o_text.csv` | zero shot on the transcript, 483 conflict segments |
| prompt wording on the E-DAIC pairs | Qwen2.5-Omni, original wording conflict 0.2218 and agreement 0.9482 (483 rows). Wording 2: 0.3439 [0.2900, 0.3979] and 0.9119 [0.8505, 0.9594] (change +0.1221 and -0.0363). Wording 3: 0.2392 [0.1940, 0.2865] and 0.9442 [0.8926, 0.9821] (change +0.0174 and -0.0040) | `release/edaic_rerun/part16/POD1/p14_o25_audio_p2.csv` (authors' machine path), `release/edaic_rerun/part16/POD1/p14_o25_audio_p3.csv` (authors' machine path) | zero shot |
| sentiment threshold on the E-DAIC pairs | threshold 0.60: conflict 0.2619 [0.2188, 0.3075], agreement 0.9406 [0.9026, 0.9711] (646 pairs, 147 speakers). Threshold 0.80: 0.1936 [0.1501, 0.2438] and 0.9594 [0.9190, 0.9895] (281 pairs, 104 speakers). Conflict range over 0.60, 0.70 and 0.80: 0.0683 | `release/edaic_rerun/part16/POD1/p14_thr060_omni.csv` (authors' machine path), `release/edaic_rerun/part16/POD1/p14_thr080_omni.csv` (authors' machine path) | Qwen2.5-Omni zero shot, pairs rebuilt at each threshold |
| label cutoff | pairs rebuilt with PHQ-8 of 10 or more as the positive class: conflict 0.1759 [0.1448, 0.2095], agreement 0.9311 [0.9028, 0.9568] (924 pairs, 237 speakers) | `release/edaic_rerun/part16/POD1/p14_phq10_omni.csv` (authors' machine path) | Qwen2.5-Omni zero shot |
| Yes rates on the E-DAIC pairs (conflict, agreement, paired conflict minus agreement) | Qwen2.5-Omni 0.2236, 0.2919, -0.0683 [-0.1788, 0.0474]. Qwen2-Audio 0.9400, 0.8675, +0.0725 [0.0200, 0.1330]. Qwen3-Omni 0.3892, 0.4182, -0.0290 [-0.1540, 0.1078]. Audio Flamingo 2 0.9710, 0.9834, -0.0124 [-0.0280, 0.0020]. Audio Flamingo 3 0.7039, 0.6687, +0.0352 [-0.0491, 0.1281]. Kimi-Audio 0.8364, 0.7578, +0.0787 [0.0160, 0.1461]. Median answer mass 0.9249 to 0.9998 | `scores/part16/M/M4_yes_rates.csv`, `scores/part16/M/M4_yes_rate_diffs.csv` | share of clips with p_yes above 0.5, 483 rows per arm |
| LoRA baseline on Pitt | LoRA rank 8 on q, k, v, o of all 28 language model layers: 0.8250 [0.7735, 0.8729], conflict 0.5520 [0.4348, 0.6833], agreement 0.9199 [0.8846, 0.9513], conflict minus agreement -0.3679 [-0.4861, -0.2433]. LoRA minus projector fine-tune +0.0577 [0.0154, 0.1044], LoRA minus zero shot +0.1672 [0.1071, 0.2264]. Trainable parameters 5,046,272 (0.04702 percent of the model) against 4,591,104 for the projector (0.042779 percent). LoRA plus projector 0.7696. An earlier LoRA run gives 0.8202 | `scores/part16/POD2/lora_pitt_oof.csv`, `scores/part16/POD2/lora_pitt_params.json`, `release/omni_final/abl2_pitt_both_oof.csv` (authors' machine path), `release/omni_final/abl_pitt_oof.csv` (authors' machine path) | out-of-fold AUC, 5 folds, paired speaker bootstrap |
| greedy decoding check | the first generated word is Yes or No on 966 of 966 clips (share 1.0000). It agrees with the logit decision on 0.9865 of clips (conflict 0.9917, agreement 0.9814) | `release/edaic_rerun/part16/POD1/p14_o25_greedy.csv` (authors' machine path) | Qwen2.5-Omni, E-DAIC pairs |
| MDVR-KCL intervals | transcript 0.7634 [0.5936, 0.9118], audio 0.7143 [0.5210, 0.8810], encoder probe (one saved split) 0.7768 [0.5994, 0.9265], transcript minus audio 0.0491 [-0.1827, 0.2818], transcript minus probe -0.0134 [-0.2383, 0.2381] | `scores/part16/M/M10_kcl_transcript.csv` | 37 speakers, paired speaker bootstrap |

---

## g. Released analyses that are not in the paper

Each analysis below was run after the 14 September draft. None of it is in the paper. Intervals are speaker
bootstrap as in section c. "Paired" means one speaker draw per replicate for both quantities.

### Interviewer removal on Pitt, ADReSSo and ADReSS-2020, with a duration control

Qwen2.5-Omni on three cuts of the same windows: arm A the original cut, arm B the participant's speech only
(interviewer removed), arm C the original cut with the interviewer kept but trimmed to arm B's duration. Arm C
separates the effect of removing the interviewer from the effect of hearing less audio.

| dataset | zero shot A, B, C | paired zero shot, A to B and C to B | encoder probe A, B, C | paired probe, A to B and C to B |
|---|---|---|---|---|
| Pitt (468) | 0.6497, 0.7230, 0.6145 | +0.0732 [0.0312, 0.1165], +0.1084 [0.0638, 0.1550] | 0.7949, 0.7497, 0.7980 | -0.0452 [-0.0857, -0.0083], -0.0483 [-0.0907, -0.0062] |
| ADReSS-2020 (156) | 0.6272, 0.6524, 0.6313 | +0.0251 [-0.0254, 0.0769], +0.0210 [-0.0325, 0.0756] | 0.7959, 0.7641, 0.8075 | -0.0317 [-0.0761, 0.0102], -0.0434 [-0.0888, -0.0033] |
| ADReSSo (161 clips whose released segmentation marks the participant) | 0.6291, 0.6302, 0.6083 | +0.0011 [-0.0441, 0.0462], +0.0219 [-0.0253, 0.0683] | 0.8624, 0.8242, 0.8726 | -0.0382 [-0.0693, -0.0094], -0.0485 [-0.0832, -0.0161] |

On Pitt, removing the interviewer raises the zero-shot answer and the rise is not explained by the shorter audio
(C to B +0.1084). On ADReSSo and ADReSS-2020 the answer does not move beyond noise. On all three sets the encoder
probe drops a little without the interviewer. On Pitt, 158 of 468 windows contain timed interviewer speech, the
interviewer's duration alone predicts dementia at 0.6427 and the participant-only duration left in arm B predicts it
at 0.6648 (direction free). The encoder values here are single-split re-runs on the re-cut audio. The Table 1 estimator is the mean of five.

Files: `scores/part16/POD4/p16_pitt_orig_zeroshot_scores.csv`, `scores/part16/POD4/p16_pitt_paronly_zeroshot_scores.csv`,
`scores/part16/POD4/p16_pitt_durmatch_zeroshot_scores.csv`, the matching `_enc_nested_oof.csv` files,
`scores/part16/POD4/boot4b.json`, `scores/part16/POD4/pitt_no_interviewer.csv`,
`scores/part16/POD4B/adress2020_no_interviewer.csv`, `scores/part16/POD4B/adresso_no_interviewer.csv`.

### Interviewer-free Pitt window, all six models (PART 20)

The same interviewer-free Pitt cut (arm B above) scored by every model with its paper pipeline. Paired values are
original minus interviewer-free, so a negative value means the answer rises without the interviewer.

| model | original, interviewer-free (overall) | paired, overall | paired, conflict | paired, agreement | direction |
|---|---|---|---|---|---|
| Qwen2.5-Omni | 0.6578, 0.7230 | -0.0652 [-0.1081, -0.0228] | -0.0086 [-0.0890, 0.0679] | -0.0954 [-0.1512, -0.0401] | up |
| Qwen2-Audio | 0.6173, 0.6211 | -0.0037 [-0.0438, 0.0357] | -0.0121 [-0.0924, 0.0619] | -0.0144 [-0.0619, 0.0331] | unchanged |
| Qwen3-Omni | 0.7619, 0.7802 | -0.0182 [-0.0455, 0.0083] | -0.0121 [-0.0664, 0.0422] | -0.0207 [-0.0522, 0.0097] | unchanged |
| Kimi-Audio | 0.7180, 0.5981 | +0.1199 [0.0718, 0.1665] | +0.0224 [-0.0495, 0.0996] | +0.1615 [0.1068, 0.2168] | down |
| Audio Flamingo 2 | 0.5724, 0.4655 | +0.1068 [0.0595, 0.1529] | -0.0167 [-0.1068, 0.0759] | +0.1579 [0.1052, 0.2093] | down |
| Audio Flamingo 3 (part10 copy) | 0.5894, 0.5920 | -0.0026 [-0.0600, 0.0548] | +0.0658 [-0.0359, 0.1653] | -0.0285 [-0.0959, 0.0404] | unchanged |

Removing the interviewer moves Qwen2.5-Omni up and Kimi-Audio and Audio Flamingo 2 down, and leaves Qwen2-Audio,
Qwen3-Omni and Audio Flamingo 3 unchanged. For Kimi-Audio and Audio Flamingo 2 the change sits in the agreement arm.
The Audio Flamingo 2 paper file scored each window at full length, so a same-span control was added: on the first
30 s of the original windows it gives 0.5456, and original first 30 s minus interviewer-free is +0.0800
[0.0341, 0.1301], so the drop holds on the same span. The part3 Audio Flamingo 3 copy gives the same direction
(-0.0034 [-0.0611, 0.0542]).

Qwen2.5-Omni probes and fine-tune on the interviewer-free cut (paper value first): encoder probe 0.7706 and 0.7643,
paired 0.0063 [-0.0232, 0.0375]. LM probe 0.8032 and 0.8020. Answer-state probe 0.7636 and 0.7777. Projector
fine-tune 0.7673 and 0.7912, paired -0.0239 [-0.0744, 0.0270]. Direction test: probe 0.7958, with the readout
direction removed 0.7769, cosine 0.0067. The probes do not move beyond noise.

Files: `scores/part20/POD4/`, `scores/part20/POD4b/`, `scores/part20/POD4c/`, `scores/part20/POD4d/` and
`scores/part20/POD5/`, each with per-clip csv and sidecar json.

### A second sentiment scorer for the E-DAIC pairs

The E-DAIC pairs were rebuilt with DistilBERT SST-2 in place of RoBERTa. The two scorers agree on the sentiment label
of a segment with Cohen kappa 0.2481 (6214 segments) and on the rule arm with kappa 0.3438 (5523 eligible segments).
The rebuilt set has 1334 pairs from 162 speakers. Qwen2.5-Omni on it: conflict 0.4619 [0.4165, 0.5102], agreement
0.8554 [0.8199, 0.8885] on 799 distinct segments, paired -0.3934 [-0.4420, -0.3410]. Split by the RoBERTa arm of the
same segment, the conflict rows that RoBERTa also calls conflict give 0.1951 [0.1545, 0.2413] and those RoBERTa calls
neutral give 0.6150 [0.5593, 0.6755].

With the second scorer the conflict arm sits near chance and the agreement arm stays high, so the gap between the arms
remains large. The below-chance conflict value holds where both scorers call the words polarised against the label.
Re-scoring the Part 14 pairs on the same pod gives conflict 0.2210 against the published 0.2218.
Files: `scores/part16/M/M2_distilbert_sentiment.csv`, `scores/part20/POD3b/sst2_pairs_o25_audio.csv`,
`scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv`, `scores/part20/POD3b/p14_rescore_o25_audio_envcheck.csv`.

### Class-balanced fine-tuning and seed repeats on Pitt

The projector and LoRA recipes were rerun with per-clip loss weights that balance the conflict and agreement arms, each
beside a same-pod control with uniform weights.

| run | overall | conflict | agreement | file |
|---|---|---|---|---|
| projector, paper file | 0.7673 [0.7086, 0.8231] | 0.5189 [0.4151, 0.6300] | 0.8593 [0.8063, 0.9069] | `release/omni_final/omnisft_pitt_oof.csv` (authors' machine path) |
| projector, balanced | 0.7301 [0.6656, 0.7905] | 0.4693 [0.3686, 0.5854] | 0.8279 [0.7732, 0.8792] | `scores/part20/POD1/balanced_ft_pitt_oof.csv` |
| projector, uniform control | 0.7720 [0.7166, 0.8280] | 0.4697 [0.3613, 0.5906] | 0.8895 [0.8456, 0.9286] | `scores/part20/POD1/standard_rerun_ft_pitt_oof.csv` |
| projector, standard recipe seed 1 | 0.7396 [0.6891, 0.7921] | 0.5364 [0.4306, 0.6426] | 0.8213 [0.7716, 0.8689] | `release/scores/part20/POD1b/std_ft_seed1_pitt_oof.csv` (authors' machine path) |
| projector, standard recipe seed 2 | 0.8085 [0.7560, 0.8572] | 0.5636 [0.4593, 0.6779] | 0.9022 [0.8648, 0.9361] | `release/scores/part20/POD1b/std_ft_seed2_pitt_oof.csv` (authors' machine path) |
| projector, standard recipe seed 3 | 0.7975 [0.7456, 0.8479] | 0.5666 [0.4660, 0.6746] | 0.8844 [0.8424, 0.9225] | `release/scores/part20/POD1b/std_ft_seed3_pitt_oof.csv` (authors' machine path) |
| LoRA, earlier run | 0.8250 [0.7735, 0.8729] | 0.5520 [0.4348, 0.6833] | 0.9199 [0.8846, 0.9513] | `scores/part16/POD2/lora_pitt_oof.csv` |
| LoRA, balanced | 0.7200 [0.6494, 0.7866] | 0.5770 [0.4586, 0.7006] | 0.7843 [0.7074, 0.8510] | `scores/part20/POD2/balanced_lora_pitt_oof.csv` |
| LoRA, uniform control | 0.8346 [0.7856, 0.8825] | 0.6198 [0.4963, 0.7470] | 0.9112 [0.8710, 0.9471] | `scores/part20/POD2/control/uniform_lora_pitt_control_oof.csv` |

Balancing the arms does not lift the conflict arm: balanced minus control on conflict is -0.0004 [-0.0742, 0.0757]
for the projector and -0.0428 [-0.1707, 0.0837] for LoRA, and for LoRA it costs overall AUC (-0.1146
[-0.1751, -0.0564]). Three new seeds of the standard projector recipe span 0.0688 overall (0.7396 to 0.8085) and
0.0476 on conflict once the paper run is included (0.5189 to 0.5666), about as large as the differences between
recipes. The conflict arm stays near chance in every run. In the first balanced projector run one fold collapsed
(answer mass near zero on its test clips). With that fold replaced by a lone rerun the balanced run gives 0.7603,
0.5003 and 0.8573 (`scores/part20/POD1/balanced_ft_pitt_fold3rerun_oof.csv`).

### NeuroVoz, age and sex matched subset

NeuroVoz speakers were paired PD to control on sex and age within 5 years (38 pairs, 76 speakers, 907 clips, which
is the largest such matching). On this subset the Qwen2.5-Omni zero-shot answer is 0.6282 [0.5584, 0.6985] against
0.6951 on all 1270 clips, and the encoder probe (mean of five) is 0.8986 [0.8430, 0.9428]. The five quiet-frame
channel features together still separate the groups at 0.7112 [0.6079, 0.7961], and the spectral centroid alone at
0.7261 (direction free). The 363 clips left out of the subset give a zero-shot answer of 0.8462 [0.7645, 0.9144].
The encoder probe on NeuroVoz survives matching, the channel cue does not go away, and the zero-shot answer is higher
on the unmatched clips.
Files: `scores/part20/POD6/B_neurovoz_matched/B_zeroshot_nvmatched.csv`,
`scores/part20/POD6/B_neurovoz_matched/B_encprobe_nested5_nvmatched_oof.csv`,
`scores/part20/POD6/B_neurovoz_matched/B_channel5_nvmatched.csv`, `scores/part20/POD6/B_neurovoz_matched/nv_matched_pairs.csv`
(released without age and sex).

### How much the words carry, per dataset

A TF-IDF logistic regression on the transcript alone, speaker-disjoint folds: PC-GITA 0.6196 [0.5700, 0.6681],
NeuroVoz 0.5814 [0.5422, 0.6206], MDVR-KCL 0.4315 [0.2560, 0.6287], E-DAIC first 30 s 0.4844 [0.4028, 0.5624],
E-DAIC full interview 0.6778 [0.5973, 0.7549], Pitt 0.8353 [0.7865, 0.8862], ADReSSo 0.8471 [0.7961, 0.8922],
ADReSS-2020 0.9254 [0.8866, 0.9587]. The ordering Parkinson's below depression below Alzheimer's holds only with
the full E-DAIC transcript, and on the read-speech Parkinson's sets the only text is ASR output, so what the
classifier reads there is ASR errors caused by the voice.
Files: `scores/part20/POD6/A_language_saliency/A_tfidf_<dataset>_oof.csv` with its sidecar, and the summary
`release/scores/part20/POD6/A_language_saliency/A_summary.json` (authors' machine path).

### Projector fine-tune on the E-DAIC conflict pairs

Qwen2.5-Omni's projector trained on the 966 clips of the E-DAIC pairs, out of fold by speaker: conflict 0.5932
[0.5087, 0.6831], agreement 0.6686 [0.5727, 0.7586] on the 311 distinct segments, paired -0.0754 [-0.1513, 0.0010].
Training on the pairs removes most of the gap between the arms, and both arms end near 0.6.
File: `scores/part20/POD3/p14_ft966_oof.csv`.

### Pitt, all 708 segments (Part 13)

The paper's Pitt set keeps the top and bottom thirds of a text fluency score and drops the middle third
(`scripts/upstream/build_ad_conflict.py`, lines 98 to 106). Rebuilt with every third kept: 708 segments from
266 speaker labels (265 participants, because participant 172 again carries two labels).

| stream | paper set, 468 | all 708 | estimator |
|---|---|---|---|
| zero shot answer | 0.6578 | 0.6695 | zero shot |
| encoder probe | 0.7706 | 0.7968 | mean of five |
| projector probe | 0.7507 | 0.7632 | one repeat |
| LM probe | 0.8032 | 0.8125 | mean of five |
| answer-state probe | 0.7636 | 0.7744 | mean of five |
| projector fine-tune | 0.7673 | 0.7904 | out of fold |
| transcript only | 0.6629 | 0.6215 | zero shot on the transcript |

By third, the zero-shot answer gives 0.6641 (low, 234 clips), 0.6665 (middle, the 240 the paper drops) and 0.6437
(high, 234 clips). The same 468 rows scored in this run give 0.6711 against 0.6578 in the paper file, so differences
of about 0.013 between two runs of one protocol are run-to-run noise. The dropped middle third does not inflate the
zero-shot result.
Files: `scores/pitt_all/o25_pitt_all_zeroshot_scores.csv`, `scores/pitt_all/o25_pitt_all_nested_repeats.json`,
`scores/pitt_all/sft_pitt_all_oof.csv`, `manifests/pitt_all_segments.csv`, `manifests/mf_pitt_all.csv`,
`reports/PART13_tertiles.txt`.

### E-DAIC window rerun

The 14 September draft scored E-DAIC on the first 30 s of each file, which is mostly session setup. Every model was
rerun with the same prompt and scoring on three other windows cut from the participant's stitched speech: the middle
30 s, the middle 300 s, and the interview window capped at 900 s (for Qwen2.5-Omni the processor keeps its first 300 s,
section c). All n = 275.

Zero-shot answer by window:

| model | first 30 s | middle 30 s | middle 300 s | interview window |
|---|---|---|---|---|
| Qwen2.5-Omni | 0.6620 | 0.7205 | 0.8122 | 0.8285 (rebuild 0.8278, truncation off 0.8379) |
| Qwen3-Omni | 0.6290 | 0.7153 | 0.8217 | 0.8741 |
| Audio Flamingo 3 | 0.5975 | 0.6949 | 0.8374 | 0.8565 |
| Qwen2-Audio | 0.6464 | 0.6891 | 0.6332 | 0.6330 |
| Kimi-Audio | 0.6293 | 0.6664 | 0.6581 | 0.5929 |

Qwen2.5-Omni, every stream by window:

| stream | first 30 s | middle 30 s | middle 300 s | interview window |
|---|---|---|---|---|
| zero shot answer | 0.6620 | 0.7205 | 0.8122 | 0.8285 |
| encoder probe | 0.5966 | 0.5770 | 0.6168 | 0.5884 |
| projector probe | 0.5041 | 0.4380 | 0.5703 | 0.5610 |
| LM probe | 0.4939 | 0.6249 | 0.6351 | 0.7001 |
| answer-state probe | 0.6146 | 0.6593 | 0.7065 | 0.7435 |
| projector fine-tune | 0.6083 | 0.6669 | 0.6595 | 0.6421 |
| transcript only | 0.6613 | no value | no value | 0.8230 |

The interview-window probe values are the mean-of-five values of the rebuild (section c). The original run on that
window gave encoder 0.5930, projector 0.5293, LM 0.6853 and answer state 0.7542. The projector probe values for the
first 30 s, middle 30 s and middle 300 s are one repeat. The first-30 s fine-tune value is `release/omni_final/omnisft_edaic_oof.csv` (authors' machine path).
On the interview window the answer is above every probe (answer minus encoder probe 0.2394, section c), the
transcript alone matches the audio answer (-0.0055 [-0.0566, 0.0431]) and the projector fine-tune falls below the
zero-shot answer by 0.1863 [0.1098, 0.2696].

Files: per-clip scores and sidecars for the three windows in `scores/edaic_windows` (window cut points in
`manifests/edaic_windows_full.csv`, `manifests/edaic_windows_mid30.csv` and `manifests/edaic_windows_mid300.csv`),
the rebuild in `scores/part16/EDAICFULL/edaic_full_perclip.csv` and `scores/part16/EDAICFULL/full_zeroshot_scores.csv`,
the mean-of-five per-repeat scores in `scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv`, the truncation-off
run in `scores/part22/edaic_lifted_perclip.csv`, and the table as CSV in `lookup/new_results_edaic_windows.csv`
and as text in `reports/EDAIC_REPORT.txt`.

Caveats carried by every window: each window is cut from a stitched file of the participant's ASR segments, so
"middle 30 s" is the middle of that stitched file. The raw recording was not used for the cut. The E-DAIC binary label is the dataset's
released label, and 20 of the 209 speakers labelled negative have a PHQ-8 of 10 or more (`DISCREPANCIES.md`,
items 2 and 9).

---

## h. Prompts

`prompts/prompts.json` lists five prompt strings with the runs that used them:

- `Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.`
- `Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No.`
- `Based only on this transcript of what the person said, does this speaker show signs of Parkinson's disease? Answer with one word, Yes or No.`
- `Based only on this transcript of what the person said, does this speaker show signs of dementia? Answer with one word, Yes or No.`
- `Based only on this transcript of what the person said, does this speaker show signs of depression? Answer with one word, Yes or No.`

Two more prompt strings are used by released runs and are not in `prompts/prompts.json`. Their text is in the prompt
column or the sidecar of each run:

- `Based only on how this person's voice sounds, does this speaker show signs of Parkinson's disease? Answer with one word, Yes or No.`
  This is the zero-shot audio prompt of PC-GITA, NeuroVoz and MDVR-KCL in Table 1 (`scripts/core/extract_probe_layers.py`,
  `scores/part20/POD6/B_neurovoz_matched/B_zeroshot_nvmatched.csv`).
- `Based only on how this person's voice sounds, does this speaker show signs of dementia? Answer with one word, Yes or No.`
  This is the voice-wording check on Pitt in `scores/part20/POD4/o25_orig_voice.csv`.

The alternative wordings of section f are recorded with their runs: for the 0.02 sentence in
`release/omni_final/omni_pitt_p2.json` (authors' machine path) and its siblings, and for the E-DAIC pairs in the prompt
column of `release/edaic_rerun/part16/POD1/p14_o25_audio_p2.csv` (authors' machine path) and
`release/edaic_rerun/part16/POD1/p14_o25_audio_p3.csv` (authors' machine path).

No prompt contains participant speech. The transcript conditions place the transcript after the prompt at run time.
Those transcripts are not released.

---

## i. Folds

The GroupKFold(5) speaker fold assignment of every probe split that was run is in `folds/`, one csv per dataset and
split, with columns `clip_id`, `speaker_id` and `fold`. `folds/README.md` lists every file, which run it reproduces
and how it was checked.

Why a dataset has more than one split: without shuffling, scikit-learn's GroupKFold orders speakers by clip count
with `numpy.argsort`, and speakers with equal counts stay in an order that differed between the Mac used for some runs
(numpy 2.2.6, arm64) and the Linux GPU pods used for the others. On ADReSSo, ADReSS-2020, MDVR-KCL and E-DAIC every
speaker has one clip, so that order decides the whole split. The same code and labels therefore gave two splits, and
both are released: `_mac` (the Mac order) and `_pod` (a stable argsort, which reproduces every pod run that could be
checked).

Which split each run used:

| run | where it ran | split |
|---|---|---|
| Qwen2-Audio single-split nested probes, all seven datasets | Mac | `folds/<dataset>_groupkfold5_mac.csv` |
| eGeMAPS baseline, all seven datasets | Mac | `folds/<dataset>_groupkfold5_mac.csv` |
| Qwen2.5-Omni Pitt encoder probe, 0.7706 | Mac | `folds/pitt_groupkfold5_mac_seed0.csv` to `folds/pitt_groupkfold5_mac_seed4.csv` |
| Qwen2.5-Omni single-split nested probes | pod | `folds/<dataset>_groupkfold5_pod.csv` |
| Qwen3-Omni single-split nested probes | pod | `folds/<dataset>_groupkfold5_pod.csv` |
| five-repeat probe means (Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, Kimi-Audio), all cells except the one above | pod | `folds/<dataset>_groupkfold5_pod_seed0_UNVERIFIED.csv` to `_seed4_UNVERIFIED.csv`, and `folds/pcgita_groupkfold5_pod_seed0.csv` to `_seed4.csv` |
| Qwen3-Omni Pitt and PC-GITA re-extraction (part16 POD3) | pod | `folds/pitt_groupkfold5_pod_Control15ids_seed0.csv` to `folds/pitt_groupkfold5_pod_Control15ids_seed4.csv`, `folds/pcgita_groupkfold5_pod_seed0.csv` to `folds/pcgita_groupkfold5_pod_seed4.csv` |
| E-DAIC interview-window probes (part16 EDAICFULL) | pod | `folds/edaic_full_groupkfold5_shuffle_rs0.csv` to `folds/edaic_full_groupkfold5_shuffle_rs4.csv` |

What could and could not be reproduced clip for clip on the Mac:

- Runs made on the Mac reproduce exactly with the `_mac` files: the Qwen2-Audio single-split probes, the eGeMAPS
  baseline (largest difference 1.1e-16) and the Qwen2.5-Omni Pitt encoder probe 0.7706 (largest difference 0.0).
- No pod run reproduces clip for clip on the Mac. With the `_pod` split, refitting a pod probe on the Mac re-derives
  the saved layer choices and the AUC, but the per-clip probabilities differ from the pod's by up to 0.083, because
  the logistic regression stops at a gradient tolerance and the numeric libraries stop at slightly different points.
  These files are marked CONFIRMED cross-platform in `folds/README.md`.
- Two pod runs are not confirmed at all, because the pod extracted its own states and they are not on disk: the
  Qwen2-Audio NeuroVoz and E-DAIC probes in `gdrive/probe2/pod_runs` (authors' machine path).
- The five-repeat pod splits for Pitt, ADReSSo, ADReSS-2020, NeuroVoz, MDVR-KCL and E-DAIC carry `_UNVERIFIED` in the
  name. No per-clip out-of-fold file was saved for them, so they reproduce the saved per-repeat AUCs (within 0.008)
  and nothing finer. Among the five-repeat splits only PC-GITA, and the separate Pitt split of the Qwen3-Omni
  re-extraction, are confirmed clip by clip.
- The E-DAIC interview-window mean-of-five probe did not reproduce on three Mac stacks (scikit-learn 1.6.1, 1.7.2 and
  1.8.0). The cause is the float32 fit path: two inner layer choices sit near a tie (LM repeat 1 fold 1, answer state
  repeat 0 fold 1), and casting the input to float64 on the pod flips exactly those two. The unmodified script on
  Linux x86_64 with scikit-learn 1.9.1 and numpy 2.1.2 reproduces every per-repeat AUC and all layer choices exactly,
  so the E-DAIC mean-of-five values 0.7001 and 0.5884 are reproduced (section c, and `DISCREPANCIES.md` items 134, 135
  and 138).
- Projector fine-tunes saved no fold record. The fold file written on a pod by the fine-tune's own GroupKFold call
  for Pitt equals `folds/pitt_groupkfold5_pod.csv` on 468 of 468 clips.

Pitt participant 172 is listed as `Control172` and `Dementia172`. The labels are kept exactly as the runs used them,
so in some splits this person sits in two folds. `folds/README.md` says where in every Pitt split.

---

## j. Repository layout

```
README.md                     this file
DISCREPANCIES.md              the audit log
MANIFEST.txt                  every file with its size, where it came from and what was dropped, plus the exclusion check
docs/README_v1_companion.md   companion README of the first release, kept for reference
figures/
  fig1_gap.pdf                Figure 1 of the paper, the same bytes as the submitted figure, made by scripts/make_fig1_v4.py
  fig1_gap_grayscale_check.png  grayscale render of Figure 1
  fig1_gap_300.pdf, fig1_gap_mid.pdf  two other renders of the same figure, unused in the paper
  fig1_gap.png, probe_vs_answer.png, conflict_dumbbell.png  figures of the first release
folds/                        GroupKFold(5) speaker fold assignments, 64 csv files and README.md (section i)
lookup/
  master_lookup.csv           234 rows: model, dataset, stream, AUC, n, estimator and source path
  SUPERSEDED.csv              source files replaced by the decisions of 23 September 2026
  bootstrap_cis.csv           speaker-bootstrap intervals, 2000 draws
  readout_direction.csv       Qwen2.5-Omni direction test on Pitt and E-DAIC, with readout_direction.json
  new_results_edaic_windows.csv, new_results_pitt_all708.csv  the E-DAIC window and Pitt all-708 reruns
  table1_all_models.csv, table1_omni25.csv  earlier Table 1 extracts, kept for reference
manifests/                    segment lists and window cut points, speech and clinical columns removed
  confounds/                  recording features per clip and the confound AUC table
  part1e/                     speaker crosswalks between Pitt, ADReSSo and ADReSS-2020
prompts/prompts.json          every prompt string
reports/                      text reports of the audit parts (E-DAIC windows, interviewer measure, Part 13, Part 14)
results/                      the tables of the 14 September draft as CSV, kept for the record and superseded by sections d and e
scores/
  edaic_conflict_v2/          Table 2 E-DAIC per-clip scores on the Part 14 pairs, six models
  edaic_windows/              E-DAIC window rerun: per-clip scores, per-layer curves, nested repeats, fine-tunes
  part10/                     pitt_enc_nested5_oof.npz, behind the Pitt encoder probe 0.7706
  pitt_all/                   Pitt all-708 rerun (the *_full_* files in this folder are E-DAIC interview-window copies)
  part16/                     EDAICFULL rebuild, POD2 LoRA, POD3 Qwen3-Omni, POD4 and POD4B interviewer removal,
                              pod_sync (Qwen3-Omni direction test), M (per-clip files of the M checks)
  part17/                     T1 distinct agreement, T3 Qwen2-Audio direction test, T5 zero-shot per clip, T6, T7b
  part20/                     PART 20, POD1 to POD6, per-clip csv and sidecar json
  part22/                     PART 22, processor configuration, truncation proof, truncation-off E-DAIC run
scripts/
  make_fig1_v4.py             Figure 1 (make_fig1_v3.py is the version before it)
  core/                       probing, per-layer curves, nested repeats, projector fine-tune, zero-shot and transcript scoring
  upstream/                   segment builders and the two conflict rules
  audit_1a1c/, part1e/, confounds/  scripts of the audit
  part14/, edaic_full/        the Part 14 pair builders and the E-DAIC interview-window builder
```

Most per-clip files carry `clip` or `path`, `speaker`, `label`, `p_yes` and `mass` or `answer_mass`, and each run has a
sidecar json with the checkpoint, prompt, n, AUC, dtype and date. File paths inside the lookup tables and sidecars that
start with /Users/chaitanyaparwatkar/ or /Volumes/G-Drive Pro/ are authors' machine paths. The build logs of this
release are not part of the repository.

---

## k. Status of every audit item

DISCREPANCIES.md numbers 176 items. The table gives one row per item. Fixed in the paper: 29. Released here: 136. Open: 11. Items 3, 4, 5, 6, 7, 8, 12, 13 and 14 describe the 14 September draft and are resolved in the submitted version. Quotes are verbatim from the submitted text as pasted on 23 September 2026 at 05:58 UTC. Sentences added later that day will be matched to the final tex.

| # | Finding | Status | Quoted sentence or file |
|---|---|---|---|
| 1 | Qwen2.5-Omni cannot take the full E-DAIC files, so the E-DAIC window is capped | released here | scores/part22/P22_step1_processor_config.md, scores/part22/p22_step2_truncation_proof.json, scores/part22/edaic_lifted_perclip.csv, scores/part22/edaic_lifted_auc.json, release/scores/part22/rows/P22_verified.tsv (authors' machine path), manifests/edaic_windows_full.csv |
| 2 | The E-DAIC audio is a stitched concatenation of transcript rows | open | Decided by: how the final tex describes the stitched E-DAIC input (scripts/edaic_full/build_windows.py), or a recut from the raw recording |
| 3 | The first 30 s of E-DAIC is setup chatter (14 September draft, resolved) | fixed in the paper | "On E-DAIC the audio answer on the whole interview matches the transcript alone." |
| 4 | Scored AD windows contain interviewer speech (14 September draft, resolved) | fixed in the paper | "Each Pitt segment starts at the participant's first timed utterance, and the ADReSSo and ADReSS-2020 windows start at the beginning of the recording, so interviewer prompts inside the window are kept." |
| 5 | The Pitt segments are not thirty seconds long (14 September draft, resolved) | fixed in the paper | "Models hear the first 30\,s, the length Qwen2-Audio accepts, so the conflict test in Table~\ref{tab:conflict} runs on the same clips for all six models." |
| 6 | Pitt, ADReSSo and ADReSS-2020 share most speakers (14 September draft, resolved) | fixed in the paper | "The three sets share speakers, so every result is within one set on speaker-disjoint folds and we make no transfer claim among them." |
| 7 | The set called ADReSSo is the ADReSS-M English training split (14 September draft, resolved) | fixed in the paper | "We use the DementiaBank Pitt cookie-theft descriptions \cite{pitt}, with 468 segments from 228 speakers, the top and bottom third of the fluency score described below, the ADReSSo recordings \cite{adresso} from 237 speakers, as released with the ADReSS-M challenge \cite{adressm}, and the ADReSS-2020 benchmark split \cite{adress2020} including 108 training and 48 test recordings." |
| 8 | NeuroVoz is channel confounded (14 September draft, resolved) | fixed in the paper | "Recording features from the quietest frames separate the NeuroVoz groups at 0.87 by spectral centroid alone and PC-GITA at 0.69, so part of the Parkinson's probe may come from the recording channel." |
| 9 | The E-DAIC binary label is not PHQ-8 at threshold 10 | fixed in the paper | "We use E-DAIC \cite{daic}, the extended corpus released for AVEC 2019, with 275 participants, where the binary label is the dataset's PHQ-8 \cite{Kroenke2009} based label rather than a clinical diagnosis." |
| 10 | PC-GITA and NeuroVoz have no 30 s window | fixed in the paper | "We use the three following corpora: PC-GITA (Spanish, 1100 clips) \cite{pcgita}, NeuroVoz (Spanish, 1270 clips) \cite{neurovoz} and MDVR-KCL (English, 37 speakers) \cite{kcl}." |
| 11 | NeuroVoz is stored twice on disk | released here | manifests/part1e/fp.csv, scripts/part1e/fingerprint.py, scripts/part1e/dupes2.py |
| 12 | The Pitt 468 is the top and bottom thirds of a text fluency score (14 September draft, resolved) | fixed in the paper | "On all 708 Pitt segments, middle third included, the encoder probe reaches 0.80 and the zero-shot answer 0.67." |
| 13 | The E-DAIC valence lexicon dq_lex is not on disk (14 September draft, resolved) | fixed in the paper | "From the raw E-DAIC transcripts, with the interviewer removed, we take every participant segment of at least 25 words and 5\,s of speech and score it with a RoBERTa sentiment classifier trained on TweetEval \cite{tweeteval}." |
| 14 | On E-DAIC the answer beats the probe once the model hears interview speech (14 September draft, resolved) | fixed in the paper | "The gap is widest on Parkinson's read-speech, narrower on Alzheimer's picture descriptions and reversed on depression." |
| 15 | KCL transcript AUC 0.7634 recomputed against 0.7619 published | fixed in the paper | "MDVR-KCL    &   37 (37)  & 0.71 & 0.76 & 0.70 \\" |
| 16 | KCL encoder probe: single saved OOF file against the mean of five repeats | released here | lookup/master_lookup.csv, folds/kcl_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/kcl_groupkfold5_pod_seed4_UNVERIFIED.csv, folds/README.md |
| 17 | No saved fold assignment file for KCL | released here | folds/kcl_groupkfold5_mac.csv, folds/kcl_groupkfold5_pod.csv, folds/README.md |
| 18 | Pitt encoder probe: single-split OOF against the mean of five repeats | released here | lookup/master_lookup.csv, lookup/SUPERSEDED.csv, scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/README.md |
| 19 | ADReSSo encoder probe: single-split OOF against the mean of five repeats | released here | lookup/master_lookup.csv, folds/adresso_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/adresso_groupkfold5_pod_seed4_UNVERIFIED.csv, folds/README.md |
| 20 | ADReSS-2020 encoder probe: single-split OOF against the mean of five repeats | released here | lookup/master_lookup.csv, folds/adress2020_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/adress2020_groupkfold5_pod_seed4_UNVERIFIED.csv, folds/README.md |
| 21 | PC-GITA encoder probe: single-split OOF against the mean of five repeats | released here | lookup/master_lookup.csv, folds/pcgita_groupkfold5_pod_seed0.csv to folds/pcgita_groupkfold5_pod_seed4.csv, folds/README.md |
| 22 | NeuroVoz encoder probe: single-split OOF against the mean of five repeats | released here | lookup/master_lookup.csv, folds/neurovoz_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/neurovoz_groupkfold5_pod_seed4_UNVERIFIED.csv, folds/README.md |
| 23 | KCL encoder probe: single-split OOF against the mean of five repeats | released here | lookup/master_lookup.csv, folds/kcl_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/kcl_groupkfold5_pod_seed4_UNVERIFIED.csv, folds/README.md |
| 24 | E-DAIC encoder probe: single-split OOF against the mean of five repeats | released here | lookup/master_lookup.csv, folds/edaic_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/edaic_groupkfold5_pod_seed4_UNVERIFIED.csv, folds/README.md |
| 25 | The KCL transcript gap is one tied score pair | fixed in the paper | "MDVR-KCL    &   37 (37)  & 0.71 & 0.76 & 0.70 \\" |
| 26 | The E-DAIC agreement arm has 311 distinct segments in 483 rows | released here | scores/part17/T1_table2_distinct.csv, scores/part17/T1_perclip_distinct_agreement.csv, scores/part17/T1_perclip_all966.csv, release/edaic_rerun/part17/rows/T1.tsv (authors' machine path) |
| 27 | Audio Flamingo 2 answers Yes on almost every E-DAIC conflict clip | released here | reports/PART14_yes_rates.csv |
| 28 | Audio Flamingo 2 answers Yes on almost every E-DAIC agreement clip | released here | reports/PART14_yes_rates.csv |
| 29 | Qwen3-Omni Pitt zero shot 0.7673 on rerun against 0.7619 shipped | fixed in the paper | "On Qwen3-Omni they obtain AUC scores of 0.77 to 0.94 against 0.60 to 0.78." |
| 30 | The Qwen3-Omni Pitt projector probe is a single repeat | released here | lookup/master_lookup.csv |
| 31 | The five-repeat file reproduces the Pitt encoder probe 0.7706 exactly | released here | lookup/master_lookup.csv, lookup/SUPERSEDED.csv, scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/README.md |
| 32 | The E-DAIC conflict selection changes under a second sentiment scorer | released here | scores/part16/M/M2_distilbert_sentiment.csv, scores/part16/M/M2_distilbert_sentiment.json, scores/part20/POD3b/sst2_pairs_o25_audio.csv, scores/part20/POD3b/sst2_pairs_o25_audio.json, scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv, release/scores/part20/rows/POD3b.tsv (authors' machine path) |
| 33 | The Part 14 builder scripts were only in a temporary folder | released here | scripts/part14/part14_select.py, scripts/part14/part14_sentiment.py |
| 34 | Part 14 clips are reused across pairs, so joins must be positional | released here | manifests/part14_manifest_new.csv, scores/part17/T1_table2_distinct.csv, scores/part17/T1_perclip_distinct_agreement.csv, scores/part17/T1_perclip_all966.csv, release/edaic_rerun/part17/rows/T1.tsv (authors' machine path) |
| 35 | Two Qwen2.5-Omni Pitt transcript runs exist | released here | scores/part16/M/M1_pitt_text_arms.csv, scores/part17/T6_pitt_text_sep20_arms.csv, scores/part17/T6_pitt_text_sep20_arms.json, release/edaic_rerun/part17/rows/T6.tsv (authors' machine path), lookup/master_lookup.csv |
| 36 | The Sep 18 Pitt transcript values reproduce exactly | released here | scores/part17/T6_pitt_text_sep20_arms.csv, release/edaic_rerun/part17/rows/T6.tsv (authors' machine path) |
| 37 | No saved fold file for Pitt in M5 | released here | folds/pitt_groupkfold5_mac.csv, folds/README.md |
| 38 | Audio Flamingo 2 has two Pitt score files that disagree | released here | lookup/SUPERSEDED.csv, lookup/master_lookup.csv, release/scores/part20/POD4c/work/af2_val10.csv (authors' machine path) |
| 39 | Kimi-Audio has four Pitt copies with two score sets | released here | lookup/SUPERSEDED.csv, lookup/master_lookup.csv |
| 40 | Audio Flamingo 3 has three Pitt copies that differ slightly | open | Decided by: the authors choosing the canonical AF3 Pitt file, part3 (cited by the lookup) or part10 (matches the printed 0.61 and 0.59) |
| 41 | Older 390-row E-DAIC conflict files are a different clip set | released here | manifests/edaic_paradox_manifest_390.csv, reports/PART14_table2_old_vs_new.csv |
| 42 | Only 4 of 12 conflict cells are below chance, all on E-DAIC | open | Decided by: the final wording of the abstract sentence, which does not name depression, and of the Pitt sentence that calls both Audio Flamingo models near chance while their Pitt conflict intervals lie above 0.5 |
| 43 | Mass weighting of the readout direction is a no-op on Qwen2-Audio | released here | scores/part16/M/M5_readout_direction_q2a.csv, scores/part17/T3_q2a_direction.csv, scores/part17/T3_q2a_direction_summary.csv, release/edaic_rerun/part17/rows/T3.tsv (authors' machine path) |
| 44 | AUC along the readout direction and the saved p_yes differ in the 5th decimal | released here | scores/part16/M/M5_readout_direction_q2a.csv, scores/part17/T3_q2a_direction.csv, release/edaic_rerun/part17/rows/T3.tsv (authors' machine path) |
| 45 | Apple Accelerate raises spurious floating point flags | released here | scripts/core/part10_readout_direction_v2.py |
| 46 | The first M7 pass used a superseded Pitt probe file | released here | lookup/master_lookup.csv, lookup/SUPERSEDED.csv, scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/README.md |
| 47 | Kimi-Audio has two disagreeing Pitt files | released here | lookup/SUPERSEDED.csv, lookup/master_lookup.csv |
| 48 | On Pitt no conflict cell is entirely below chance | fixed in the paper | "On the Pitt conflict segments the three Qwen models drop by 0.18 to 0.29 against their agreement segments, Kimi-Audio holds, and both Audio Flamingo models sit near chance on both sets (Table~\ref{tab:conflict})." |
| 49 | Two different KCL ASR transcript sets exist | open | Decided by: the final tex naming the pod whisper-large-v3 transcript set behind the MDVR-KCL text cell, with items 69 and 70 settled |
| 50 | KCL text AUC differs by one tied pair | fixed in the paper | "MDVR-KCL    &   37 (37)  & 0.71 & 0.76 & 0.70 \\" |
| 51 | sft_full has no answer mass column | released here | scores/edaic_windows/sft_full_oof.csv |
| 52 | sft_mid30 has no answer mass column | released here | scores/edaic_windows/sft_mid30_oof.csv |
| 53 | sft_mid300 has no answer mass column | released here | scores/edaic_windows/sft_mid300_oof.csv |
| 54 | sft_pitt_all has no answer mass column | released here | scores/pitt_all/sft_pitt_all_oof.csv |
| 55 | af3_adress2020_rerun.csv has an empty mass column | released here | lookup/master_lookup.csv |
| 56 | af3_pitt468_rerun.csv has an empty mass column | released here | lookup/master_lookup.csv |
| 57 | No saved fold file for the Pitt LoRA baseline | released here | folds/pitt_groupkfold5_pod.csv, folds/README.md |
| 58 | Two speaker-string forms for the Pitt clips can change GroupKFold | released here | folds/README.md, folds/pitt_groupkfold5_pod_Control15ids_seed0.csv to folds/pitt_groupkfold5_pod_Control15ids_seed4.csv |
| 59 | The Pitt LoRA baseline already existed | released here | lookup/master_lookup.csv, scores/part16/POD2/lora_pitt_oof.csv, scores/part20/POD2/balanced_minus_earlier_lora_pitt_perclip.csv |
| 60 | Prior LoRA and projector per-clip files carry no answer mass | released here | scores/part20/POD2/balanced_lora_pitt_oof.csv, scores/part20/POD1/standard_rerun_ft_pitt_oof.csv |
| 61 | Independent recomputation agrees with the paper's Pitt values | released here | release/scores/part20/rows/POD5.tsv (authors' machine path), release/edaic_rerun/part17/rows/T5.tsv (authors' machine path) |
| 62 | No published cell has a median answer mass below 0.15 | released here | scores/part16/M/M9_answer_mass.csv |
| 63 | The answer mass column sits in different positions across files | released here | scores/edaic_windows/o25_full_zeroshot_scores.csv, scores/edaic_windows/af3_full.csv |
| 64 | The Pitt manifest's interviewer column is zero on every row | released here | scores/part16/POD4/cha_overlap_468.csv, manifests/pitt_conflict_manifest_468.csv |
| 65 | The E-DAIC released label is stricter than PHQ-8 >= 10 | fixed in the paper | "We use E-DAIC \cite{daic}, the extended corpus released for AVEC 2019, with 275 participants, where the binary label is the dataset's PHQ-8 \cite{Kroenke2009} based label rather than a clinical diagnosis." |
| 66 | Retracted claim that 0.7706 is in no file | released here | scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv |
| 67 | No saved fold file for the Pitt 468 in POD4 | released here | folds/pitt_groupkfold5_pod.csv, folds/pitt_groupkfold5_mac.csv, folds/README.md |
| 68 | The KCL WER fallback measures Whisper against itself | fixed in the paper | "ADReSSo \cite{adresso} and the three Parkinson's sets have no transcripts, so we transcribe them with whisper-large-v3 \cite{whisper}." |
| 69 | The KCL 30 s window misses the read passage for some speakers | open | Decided by: a recut of the six MDVR-KCL windows, or a sentence in the final tex (window starts in scripts/audit_1a1c/kcl_offsets.json) |
| 70 | KCL speakers read two different passages | open | Decided by: the final tex wording for MDVR-KCL: the pasted copy says every Parkinson's speaker reads the same passage and the transcript sits near chance, while MDVR-KCL has two passages and a 0.76 transcript cell |
| 71 | Two bootstrap implementations differ in the last digits | released here | scores/part16/M/M11_kcl_wer.json, scores/part16/M/M11_kcl_wer.csv |
| 72 | A rerun does not reproduce the Pitt zero shot bit for bit | released here | scores/part16/POD4/p16_pitt_orig_zeroshot_scores.csv, scores/part20/POD4/o25_orig_recording_rerun.csv, scores/part20/POD4/o25_orig_recording_rerun.sidecar.json, release/scores/part20/POD4/check/o25_check10_vs_omni_final.csv (authors' machine path), release/scores/part20/rows/POD4.tsv (authors' machine path) |
| 73 | Qwen3-Omni Pitt zero-shot drift is a numeric environment effect | fixed in the paper | "On Qwen3-Omni they obtain AUC scores of 0.77 to 0.94 against 0.60 to 0.78." |
| 74 | No saved fold file for POD3 Pitt and PC-GITA | released here | folds/pitt_groupkfold5_pod_Control15ids_seed0.csv to folds/pitt_groupkfold5_pod_Control15ids_seed4.csv, folds/pcgita_groupkfold5_pod_seed0.csv to folds/pcgita_groupkfold5_pod_seed4.csv, folds/README.md |
| 75 | A second job disputed the readout-direction standardisation | released here | lookup/readout_direction.csv, release/edaic_rerun/part16/Q3O_direction_FIXED.json (authors' machine path), lookup/SUPERSEDED.csv |
| 76 | auc_without_d was standardised with the training fold | released here | scores/part17/T3_q2a_direction.csv, release/edaic_rerun/part16/Q3O_direction_FIXED.json (authors' machine path), scripts/core/part10_readout_direction_v2.py |
| 77 | The M5 random floor generator | released here | scores/part17/T3_q2a_direction.csv, release/edaic_rerun/part17/rows/T3.tsv (authors' machine path) |
| 78 | Two Pitt transcript runs disagree on the arm contrast | released here | scores/part16/M/M1_pitt_text_arms.csv, scores/part17/T6_pitt_text_sep20_arms.csv, scores/part17/T6_pitt_text_sep20_arms.json, release/edaic_rerun/part17/rows/T6.tsv (authors' machine path) |
| 79 | Some transcript-only lookup AUCs do not reproduce at 4 dp | fixed in the paper | Table 1 rows: "PC-GITA     & 1100 (100) & 0.56 & 0.52 & 0.89 \\" / "NeuroVoz    & 1270 (107) & 0.70 & 0.54 & 0.90 \\" / "MDVR-KCL    &   37 (37)  & 0.71 & 0.76 & 0.70 \\" |
| 80 | Draft prose on answer mass was wrong | released here | scores/part16/M/M9_answer_mass.csv |
| 81 | A fresh LoRA run does not reproduce the older one clip for clip | released here | lookup/master_lookup.csv, scores/part20/POD2/balanced_minus_earlier_lora_pitt_perclip.csv, scores/part20/POD2/control/uniform_lora_pitt_control_oof.csv |
| 82 | Two bootstrap protocols for a single-arm interval | released here | scores/part17/T6_pitt_text_sep20_arms.json, release/scores/part20/rows/POD2.tsv (authors' machine path) |
| 83 | NeuroVoz channel normalisation fails the 0.60 bar | fixed in the paper | "Recording features from the quietest frames separate the NeuroVoz groups at 0.87 by spectral centroid alone and PC-GITA at 0.69, so part of the Parkinson's probe may come from the recording channel." |
| 84 | Interviewer presence in the Pitt windows is label correlated | open | Decided by: whether the final tex reports the interviewer-free Pitt check (scores/part20/POD5/) |
| 85 | PC-GITA is only mildly channel confounded | fixed in the paper | "Recording features from the quietest frames separate the NeuroVoz groups at 0.87 by spectral centroid alone and PC-GITA at 0.69, so part of the Parkinson's probe may come from the recording channel." |
| 86 | The same zero-shot drift appears on PC-GITA | fixed in the paper | "On Qwen3-Omni they obtain AUC scores of 0.77 to 0.94 against 0.60 to 0.78." |
| 87 | Which PC-GITA manifest POD4 used | released here | folds/pcgita_groupkfold5_pod.csv, folds/README.md |
| 88 | A kernel swap alone reproduces the drift signature | released here | release/edaic_rerun/part16/POD3/q3o_pitt_attn_backend.csv (authors' machine path), release/edaic_rerun/part16/POD3/q3o_pitt_attn_backend.json (authors' machine path) |
| 89 | sft_full saved no checkpoints and its training audio was not on disk | released here | scores/edaic_windows/sft_full_oof.csv, scores/edaic_windows/sft_full_sft.json, scripts/edaic_full/build_windows.py |
| 90 | Before and after channel normalisation numbers | fixed in the paper | "Recording features from the quietest frames separate the NeuroVoz groups at 0.87 by spectral centroid alone and PC-GITA at 0.69, so part of the Parkinson's probe may come from the recording channel." |
| 91 | PC-GITA projector AUC differs in the 4th decimal | released here | release/edaic_rerun/part16/POD3/q3o_pcgita_proj_nested_oof.csv (authors' machine path) |
| 92 | The full-window E-DAIC probe had no per-clip file | released here | scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv, scores/part17/T7b/T7b_edaic300_meanof5.csv, release/edaic_rerun/part17/rows/T7b.tsv (authors' machine path), lookup/master_lookup.csv, scores/part16/M/MEDAIC_full_window.csv |
| 93 | ADReSSo released segmentation exists inside the archive | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json |
| 94 | The Mac disk was full, so POD3 states went to the external drive | released here | release/edaic_rerun/part16/POD3/q3o_pitt_attn_backend.csv (authors' machine path), release/edaic_rerun/part16/POD3/q3o_pcgita_proj_nested_oof.csv (authors' machine path) |
| 95 | ADReSSo segmentation covers only part of the scored clips | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json |
| 96 | Five ADReSSo segmentation files have no PAR tier | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json |
| 97 | ADReSS-2020 CHAT bullets run past the wav on some files | released here | scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json |
| 98 | No saved folds for ADReSS-2020 or ADReSSo | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json |
| 99 | The published ADReSSo probe is trained on more clips than arm A | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json |
| 100 | The E-DAIC window is cut at 300 s by the processor | released here | scores/part22/P22_step1_processor_config.md, scores/part22/p22_step2_truncation_proof.json, scores/part22/edaic_lifted_perclip.csv, scores/part22/edaic_lifted_auc.json, release/scores/part22/rows/P22_verified.tsv (authors' machine path), manifests/edaic_windows_full.csv |
| 101 | The full-window projector per-layer file copies the LM embedding stage | released here | scores/edaic_windows/o25_full_proj_perlayer.csv, scores/edaic_windows/o25_full_llm_perlayer.csv, release/edaic_rerun/part17/rows/T7b.tsv (authors' machine path) |
| 102 | The E-DAIC rebuild reproduces the published run | released here | scores/part16/EDAICFULL/full_zeroshot_scores.csv, scores/edaic_windows/o25_full_zeroshot_scores.csv, scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv, scores/part22/edaic_lifted_perclip.csv |
| 103 | A label-indexing bug in the first probe pass was fixed | released here | scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv, release/edaic_rerun/part17/rows/T7b.tsv (authors' machine path) |
| 104 | Correction: 0.7706 does reproduce | released here | lookup/master_lookup.csv, lookup/SUPERSEDED.csv, scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/README.md |
| 105 | POD4 used the superseded Pitt encoder baseline | released here | scores/part20/POD5/POD5_nested_enc_mac_seed.csv, scores/part20/POD5/POD5_zs_o25_noinv.csv, release/scores/part20/rows/POD5.tsv (authors' machine path) |
| 106 | Two POD4 gap figures used rounded inputs | released here | scores/part20/POD4/o25_noinv_recording.csv, scores/part20/POD4/o25_orig_recording_rerun.csv |
| 107 | Three values circulate for the Qwen3-Omni by-arm probe | released here | lookup/SUPERSEDED.csv, lookup/master_lookup.csv, release/edaic_rerun/part17/rows/T4.tsv (authors' machine path) |
| 108 | POD3's internal verifier compared cells with themselves | released here | release/edaic_rerun/part17/rows/T4.tsv (authors' machine path) |
| 109 | B_text_conflict.json claims a de-duplication that did not happen | released here | scores/part17/T1_table2_distinct.csv, scores/part17/T1_perclip_distinct_agreement.csv, scores/part17/T1_perclip_all966.csv, release/edaic_rerun/part17/rows/T1.tsv (authors' machine path) |
| 110 | A LoRA plus projector arm exists | released here | lookup/master_lookup.csv, lookup/SUPERSEDED.csv |
| 111 | The 1d rebuild changes the label and drops the severity gates | fixed in the paper | "A participant with PHQ-8 of 15 or more speaking positively, or a healthy control with PHQ-8 of 4 or less speaking negatively, is a conflict segment." |
| 112 | E-DAIC's released label is not the standard PHQ-8 cutoff | fixed in the paper | "We use E-DAIC \cite{daic}, the extended corpus released for AVEC 2019, with 275 participants, where the binary label is the dataset's PHQ-8 \cite{Kroenke2009} based label rather than a clinical diagnosis." |
| 113 | The ADReSSo segmentation exists (correction) | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json |
| 114 | POD4B quoted the wrong Pitt comparator | released here | scores/part16/POD4B/boot_adresso.json, scores/part16/POD4B/boot_adress2020.json, scores/part16/POD4/boot4b.json |
| 115 | window_seconds in the sidecars is hardcoded | released here | lookup/master_lookup.csv, scores/edaic_windows/sft_full_sft.json, scores/edaic_windows/sft_mid300_sft.json |
| 116 | The published ADReSSo probe is the wrong baseline for the 4b deltas | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json |
| 117 | POD4B arm A is not bit identical to the paper's cut | released here | scores/part16/POD4B/p16_adresso_orig_zeroshot_scores.csv, scores/part16/POD4B/p16_adress2020_orig_zeroshot_scores.csv |
| 118 | Every E-DAIC Table 2 agreement cell changes on 311 distinct segments | released here | scores/part17/T1_table2_distinct.csv, scores/part17/T1_perclip_distinct_agreement.csv, scores/part17/T1_perclip_all966.csv, release/edaic_rerun/part17/rows/T1.tsv (authors' machine path) |
| 119 | The Pitt states file moved on the G-Drive | released here | scores/part17/T3_q2a_direction.json |
| 120 | The first-pass random floor was reseeded after all | released here | scores/part17/T3_q2a_direction.csv, release/edaic_rerun/part17/rows/T3.tsv (authors' machine path) |
| 121 | Final-stage and nested probe directions differ | released here | scores/part17/T3_q2a_direction_summary.csv, release/edaic_rerun/part17/rows/T3.tsv (authors' machine path) |
| 122 | Direction weighting differs between the first pass and the verifier | released here | scores/part17/T3_q2a_direction.csv |
| 123 | Pitt participant 172 carries two speaker ids | released here | folds/README.md, release/edaic_rerun/part17/T3_selfcheck_participant_grouped.csv (authors' machine path), release/edaic_rerun/part17/T3_selfcheck_partitions.csv (authors' machine path) |
| 124 | make_fig1_v5.py does not exist | released here | figures/fig1_gap.pdf, scripts/make_fig1_v4.py |
| 125 | G-Drive paths moved, so recorded paths are stale | open | Decided by: rewriting the 75 stale /Volumes/G-Drive Pro/paper1_local_runs/ source paths in lookup/master_lookup.csv |
| 126 | E-DAIC 300 s probe values used a different estimator from Pitt | released here | scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv, scores/part17/T7b/T7b_edaic300_meanof5.csv, release/edaic_rerun/part17/rows/T7b.tsv (authors' machine path), lookup/master_lookup.csv |
| 127 | The lookup builder cannot reproduce the lookup | released here | lookup/master_lookup.csv |
| 128 | Qwen3-Omni Pitt rows come from two runs | released here | lookup/master_lookup.csv, lookup/SUPERSEDED.csv |
| 129 | Original-run edaic_full lookup rows still say capped at 900 s | open | Decided by: rewriting the Qwen2.5-Omni edaic_full lookup rows (0.8285, 0.7542, 0.5293 and 0.6421) to the 300 s cut |
| 130 | T4 bookkeeping notes | released here | lookup/master_lookup.csv, lookup/SUPERSEDED.csv |
| 131 | The Pitt transcript by-arm sentence used the Sep 15 run | fixed in the paper | "For AD the transcript alone matches the audio on the Pitt dataset, beats it on ADReSS-2020 and trails it on ADReSSo (Table~\ref{tab:main})." |
| 132 | The Sep 15 and Sep 20 transcript runs are different scorers | released here | scores/part17/T6_pitt_text_sep20_arms.csv, scores/part17/T6_pitt_text_sep20_arms.json |
| 133 | Pitt speaker id forms | released here | scores/part17/T6_pitt_text_sep20_arms.json, folds/README.md |
| 134 | The E-DAIC mean-of-five probe values could not get intervals on the Mac | released here | scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv, scores/part17/T7b/T7b_edaic300_meanof5.csv, release/edaic_rerun/part17/rows/T7b.tsv (authors' machine path), lookup/master_lookup.csv |
| 135 | Each dataset has a Mac split and a pod split | released here | folds/README.md, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/pitt_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/pitt_groupkfold5_pod_seed4_UNVERIFIED.csv, scores/part10/pitt_enc_nested5_oof.npz, release/edaic_rerun/part17/verify/T2_verify_omni_pitt_repeats.json (authors' machine path) |
| 136 | README_probe2 misstates where some probe2 outputs were made | released here | folds/README.md |
| 137 | POD3 Pitt states use unpadded speaker ids | released here | folds/pitt_groupkfold5_pod_Control15ids_seed0.csv to folds/pitt_groupkfold5_pod_Control15ids_seed4.csv, folds/README.md |
| 138 | Pod refits on the Mac cannot reach 1e-6 | released here | folds/README.md |
| 139 | Some splits are inferred | released here | folds/README.md |
| 140 | The Qwen2.5-Omni transcript check does not match every check clip | released here | release/scores/part20/POD3c/val_o25_first25.csv (authors' machine path), release/scores/part20/POD3c/val_o25_first25_run.json (authors' machine path) |
| 141 | text_score.py counts Yeah and Nope | released here | scores/part20/POD3c/p14_q3o_text.csv, release/scores/part20/rows/POD3c.tsv (authors' machine path) |
| 142 | The 10-window Qwen3-Omni check misses 1e-3 | released here | release/scores/part20/POD4b/q3o_val10_original_windows.csv (authors' machine path), release/scores/part20/POD4b/q3o_val10_original_windows.check.json (authors' machine path) |
| 143 | Rule 3 adds YES and NO ids for Qwen3-Omni | released here | release/scores/part20/POD4b/q3o_pitt_noinv_zeroshot_scores.csv (authors' machine path) |
| 144 | Correction to the POD4b 10-window counts | released here | release/scores/part20/POD4b/q3o_val10_original_windows.csv (authors' machine path) |
| 145 | The Qwen2.5-Omni audio rescore matches 10 of 20 clips at 1e-3 | released here | scores/part20/POD3b/val20_cut_and_score_check.csv, scores/part20/POD3b/p14_rescore_o25_audio_envcheck.csv, scores/part20/POD3b/p14_rescore_o25_audio_envcheck.json |
| 146 | With DistilBERT arms the Omni conflict AUC is near chance | released here | scores/part20/POD3b/sst2_pairs_o25_audio.csv, scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv, release/scores/part20/rows/POD3b.tsv (authors' machine path) |
| 147 | AF3 on a pod reproduces the part10 Pitt copy | open | Decided by: the authors choosing the canonical AF3 Pitt file, part3 (cited by the lookup) or part10 (matches the printed 0.61 and 0.59) |
| 148 | The language ordering holds only with the full E-DAIC transcript | released here | release/scores/part20/POD6/A_language_saliency/A_summary.json (authors' machine path), scores/part20/POD6/A_language_saliency/A_tfidf_<dataset>_oof.csv, release/scores/part20/rows/POD6.tsv (authors' machine path) |
| 149 | On read speech the TF-IDF signal comes from ASR output | fixed in the paper | "ADReSSo \cite{adresso} and the three Parkinson's sets have no transcripts, so we transcribe them with whisper-large-v3 \cite{whisper}." |
| 150 | Pitt transcript text source | fixed in the paper | "For the Pitt database, ADReSS-2020 \cite{adress2020} and E-DAIC \cite{daic}, we use the transcripts provided with the dataset after stripping the annotation codes" |
| 151 | NeuroVoz metadata gaps limit the matched pairs | released here | scores/part20/POD6/B_neurovoz_matched/B_encprobe_nested5_nvmatched_oof.csv, scores/part20/POD6/B_neurovoz_matched/B_encprobe_nested5_nvmatched_oof.sidecar.json |
| 152 | The Kimi-Audio gate fails at 1e-3 but the AUC holds | released here | scores/part20/POD4d/kimi_val10_vs_canonical.csv, scores/part20/POD4d/kimi_val10_vs_canonical.json, scores/part20/POD4d/kimi_pitt_orig_podrerun_overall.csv, release/scores/part20/rows/POD4d.tsv (authors' machine path) |
| 153 | Removing the interviewer changes Kimi's agreement arm | released here | scores/part20/POD4d/kimi_pitt_noinv_overall.csv, release/scores/part20/rows/POD4d.tsv (authors' machine path) |
| 154 | AF2 ran in the environment that made the paper file | released here | release/scores/part20/POD4c/work/af2_val10.csv (authors' machine path) |
| 155 | The 10-window Qwen2.5-Omni check misses 1e-3 on some windows | released here | release/scores/part20/POD4/check/o25_check10_vs_omni_final.csv (authors' machine path), release/scores/part20/POD4/check/o25_check10_vs_omni_final.json (authors' machine path), scores/part20/POD4/o25_orig_recording_rerun.csv |
| 156 | Qwen2-Audio dtype is fp16 in the paper pipeline | released here | scores/part20/POD4/q2a_orig_recording_rerun.csv, scores/part20/POD4/q2a_orig_recording_rerun_bf16.csv, scores/part20/POD4/q2a_noinv_recording_bf16.csv |
| 157 | Rule 3 adds YES and NO ids | released here | scores/part20/POD4/o25_noinv_recording.csv |
| 158 | The cut manifest's arm column is the cut variant | released here | manifests/pitt_conflict_manifest_468.csv, scores/part20/POD4/o25_noinv_recording.csv |
| 159 | Another job's setup ran on the POD4 pod | released here | scores/part20/POD4/ |
| 160 | All 468 original Pitt windows rescored on the same pod | released here | scores/part20/POD4b/P20_4b_q3o_orig_samepod_overall.perclip.csv, release/scores/part20/rows/POD4b.tsv (authors' machine path) |
| 161 | NeuroVoz zero-shot rescore against the paper file | released here | release/scores/part20/POD6/B_neurovoz_full_check/omni_nvfull_zeroshot_scores_samepod.csv (authors' machine path), release/scores/part20/POD6/B_neurovoz_full_check/B_full_check_summary.json (authors' machine path) |
| 162 | The earlier LoRA values use the two-logit p_yes | released here | scores/part20/POD2/balanced_lora_pitt_oof.csv, release/scores/part20/rows/POD2.tsv (authors' machine path) |
| 163 | Rule 3 also covers YES and NO for the POD3 fine-tune | released here | scores/part20/POD3/p14_ft966_oof.csv |
| 164 | POD5 runs restarted | released here | release/scores/part20/POD5/sft_attempt1_racefail.log (authors' machine path), release/scores/part20/POD5/sft_attempt2_cpu_throttled.log (authors' machine path), release/scores/part20/POD5/nested_orig_attempt1_killed_cpu_quota.log (authors' machine path) |
| 165 | The part20 p_yes rule adds YES and NO | released here | scores/part20/POD5/POD5_zs_o25_noinv.csv |
| 166 | Pipeline checks on the POD5 pod | released here | scores/part20/POD5/POD5_direction_origrerun_check.csv, scores/part20/POD5/POD5_nested_enc_pod_seed.csv, release/scores/part20/rows/POD5.tsv (authors' machine path), lookup/readout_direction.csv |
| 167 | A shared scratch folder | released here | release/scores/part20/POD5/scripts/ (authors' machine path) |
| 168 | For AF2 the paper file and the cut cover different audio spans | released here | release/scores/part20/POD4c/af2_orig30_all.csv (authors' machine path), release/scores/part20/POD4c/af2_diff_orig30_minus_noinv_all.csv (authors' machine path), release/scores/part20/POD4c/af2_diff_origfull_minus_orig30_all.csv (authors' machine path), release/scores/part20/rows/POD4c.tsv (authors' machine path) |
| 169 | Balanced LoRA drops pooled AUC on Pitt | released here | scores/part20/POD2/balanced_lora_pitt_oof.csv, scores/part20/POD2/control/uniform_lora_pitt_control_oof.csv, scores/part20/POD2/control/uniform_lora_pitt_control_oof.sidecar.json, release/scores/part20/rows/POD2.tsv (authors' machine path) |
| 170 | Balanced projector fine-tune fold 3 collapsed once | released here | scores/part20/POD1/balanced_ft_pitt_oof.csv, scores/part20/POD1/balanced_ft_pitt_fold3rerun_oof.csv, release/scores/part20/POD1/balanced_ft_pitt_r2_oof.csv (authors' machine path), release/scores/part20/rows/POD1.tsv (authors' machine path) |
| 171 | The standard projector fine-tune reproduces overall but not by arm | released here | scores/part20/POD1/standard_rerun_ft_pitt_oof.csv, release/scores/part20/POD1/standard_rerun_ft_pitt_r2_oof.csv (authors' machine path), release/scores/part20/rows/POD1.tsv (authors' machine path) |
| 172 | p_yes definition for the fine-tune comparison | released here | scores/part20/POD1/standard_rerun_ft_pitt_oof.csv, scores/part20/POD1/balanced_ft_pitt_oof.csv |
| 173 | Correction to the p_yes entry | released here | scores/part20/POD1/standard_rerun_ft_pitt_oof.csv, scores/part20/POD1/balanced_ft_pitt_oof.csv |
| 174 | AF2 falls below chance on the interviewer-free Pitt cut | released here | scores/part20/POD4c/af2_noinv_all.csv, scores/part20/POD4c/af2_diff_af2pitt468_minus_noinv_all.csv, release/scores/part20/rows/POD4c.tsv (authors' machine path) |
| 175 | The standard projector fine-tune collapses on some folds | open | Decided by: the authors reporting the projector seed range (item 176), or a rerun on the original machine that records answer mass (release/scores/part20/POD1b/ (authors' machine path)) |
| 176 | The Pitt projector fine-tune moves across seeds | released here | release/scores/part20/POD1b/std_ft_seed0_pitt_oof.csv (authors' machine path) to release/scores/part20/POD1b/std_ft_seed3_pitt_oof.csv (authors' machine path), release/scores/part20/POD1b/std_ft_seed_range_pitt.json (authors' machine path), release/scores/part20/rows/POD1b.tsv (authors' machine path) |
