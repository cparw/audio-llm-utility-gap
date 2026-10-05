# The Representation Utility Gap in Audio LLMs for Clinical Speech

<p align="center"><b><a href="https://cparw.github.io/audio-llm-utility-gap/">Project page</a> · <a href="https://cparw.github.io/audio-llm-utility-gap/explorer.html">Interactive explorer</a></b></p>

Chaitanya Parwatkar, Nima Kelidari, Minoo Ahmadi, Ashutosh Chaubey, Mohammad Soleymani.
University of Southern California, Los Angeles, CA, USA. Submitted to ICASSP 2027.
Corresponding author: soleymani@ict.usc.edu.

## a. Summary

**What the paper asks.** When an audio LLM answers a clinical question about a recording, does the answer use what
the model's own hidden states hold? We compare linear probes on the frozen hidden states of Qwen2.5-Omni with the
same model's zero-shot Yes or No answer, on the same clips, in seven datasets from three conditions: Parkinson's read
speech, depression interviews and Alzheimer's picture descriptions.

**Main result.** On the Parkinson's and Alzheimer's sets the encoder probe reaches 0.75 to 0.92 AUC, while the
zero-shot answer stays between 0.56 and 0.71. The paired gap between them is significant on every one of these sets
except MDVR-KCL, from 0.11 on Pitt to 0.33 on PC-GITA. On depression (E-DAIC) the order reverses: the answer reaches
0.84, above the encoder probe (0.60) and the language model probe (0.74). On depression segments whose words point
away from the diagnosis, four of six audio LLMs fall below chance. Retraining only the projector lifts the answer
significantly on every dataset except MDVR-KCL and E-DAIC, but on Pitt the segments whose words point away from the
diagnosis stay at chance.

<p align="center"><img src="docs/assets/fig1_gap.png" width="560" alt="Figure 1 of the paper: probe AUC across the audio encoder and language model of Qwen2.5-Omni for PC-GITA, Pitt and E-DAIC, with each zero-shot answer as a dotted line and the gap printed in each panel: +0.33, +0.11 and -0.24."></p>

Figure 1 of the paper, as submitted (`figures/fig1_gap.pdf`, shown here from `docs/assets/fig1_gap.png`). Each
panel prints Δ, the nested encoder probe minus the zero-shot answer.

**Folder map.**

```
scores/         per-clip score files with sidecar json, one folder per run, clinical label dropped
lookup/         master_lookup.csv: model, dataset, stream, AUC, n, estimator and source of each cell
figures/        Figure 1 as submitted (fig1_gap.pdf), earlier drafts in figures/drafts/
scripts/        the scripts behind the released numbers
folds/          GroupKFold(5) speaker fold assignments (section i)
manifests/      segment lists and window cut points, speech and clinical columns removed
prompts/        prompts.json with part of the prompt strings; section h lists more
reports/        run reports, logs and checks
docs/           the project page and the interactive explorer
checks/, history/, results/, summary/, archive/   earlier checks and drafts, kept for the record
DISCREPANCIES.md, MANIFEST.txt                    the audit log, and every file with its size and origin
```

**How to check one number.**

1. Find the sentence in section f, under "Numbers in the submitted tex" (table cells are in sections d and e). Its
   row gives the tex line, the value in the file, the file, the estimator and a status.
2. Open that file. Many cells are also a row of `lookup/master_lookup.csv`, whose `source` column names the file the
   value was read from.
3. To recompute an AUC from a per-clip file, join the clinical label from your own licensed copy of the corpus on the
   id column (section b), then use the rank formula in section c.

Reproducing the probe refits needs the hidden states, and redrawing Figure 1 needs the PC-GITA and Pitt per-layer
curve files. Neither is released. The curve values that `scripts/make_fig1_whole.py` plots are in
`scores/part24/fig/fig1_gap_whole_plotted.csv`.

### In more detail

When an audio LLM answers a clinical question about a recording, does the answer use
what the model's own hidden states already hold? We train linear probes on the frozen hidden states of Qwen2.5-Omni
and ask the same model the same Yes or No question on the same clips. Seven datasets cover three conditions that
differ in how much the spoken words carry: Parkinson's read speech, depression interviews and Alzheimer's picture
descriptions.

On Parkinson's and Alzheimer's speech the hidden states hold more than the answer uses. The encoder probes reach
0.75 to 0.92 AUC on the Parkinson's sets and 0.77 to 0.88 on the Alzheimer's sets, while the zero-shot answer stays
between 0.56 and 0.71. On depression the order reverses. The model hears the participant's whole interview, up to 15 minutes,
and its answer (0.84) beats the encoder probe (0.60) and the language model probe (0.74). A probe at the answer
position matches it (0.84). Where the words point away from a depression diagnosis, four of six audio LLMs fall below
chance. Retraining only the projector lifts the answer significantly on five of the six Parkinson's and Alzheimer's
sets, all but MDVR-KCL. On E-DAIC it stays at the zero-shot level (0.86, paired +0.02 [-0.01, 0.05]).

The repository and its project page are public. The paper is an ICASSP 2027 submission.

This README is a lookup. Every cell of Table 1, every cell of Table 2 and every number in the text of the
paper is listed below with the file it was read from, its n and its estimator, except a few that have no file in
this repository; section f names them after its table. Values are AUC against the
clinical label, to 4 decimals where the source file carries them, beside the 2 decimal value the paper prints.
The paper text used here is the tex submitted to ICASSP 2027 (md5 eacf6faf73dec5540d3c088e4aa6f542, PDF md5
708010aebf305980813c67f8e4f8e33e). The previous version of this README followed an earlier copy, the Overleaf tex of
23 September 2026, 23:23 UTC (md5 2923ced559329a4413b75efbaf477168). Sentences of that copy that the submitted tex
dropped are kept at the end of section f. Where the submitted text and the files disagree, or a printed number has no
independent check, this README keeps the printed number beside the file value and says so in the audit note at the
top of section f.

Paths in backticks with no note are files in this repository. A path followed by (authors' machine path) is a
file that stays on the authors' machine, usually because it holds data we may not redistribute. These prefixes
are used for those: `release/` is the authors' release folder, `gdrive/` is the folder of local runs on the
authors' external drive, `part23/`, `part24/` and `part27/` are the PART 23, PART 24 and PART 27 run folders on the same drive,
`paper1_submission_23sep/` is the submission folder on the same drive, and `DementiaBank/` is the authors' copy of
the DementiaBank corpus.

### Words and path prefixes

- PART n: a numbered round of checks and reruns. Its files sit in folders named partN, for example `scores/part23/`.
- POD n: one rented cloud GPU machine within a round, for example `scores/part20/POD4/`.
- M n and T n: the numbered checks of PART 16 and PART 17 (`scores/part16/M/`, `scores/part17/`).
- Rule 3: the scoring rule of the PART 20 checks. For the answer it counts every single-token spelling of Yes, yes,
  YES and No, no, NO, with and without a leading space (`DISCREPANCIES.md`, item 143, and section h).
- The Mac: the authors' laptop.

The folders that stay on the authors' machines are named one way in this README and in `DISCREPANCIES.md`, and
another way in the origin column of `MANIFEST.txt`:

| README.md and DISCREPANCIES.md | MANIFEST.txt |
|---|---|
| `release/` | `release/` |
| `part23/`, `part24/`, `part27/` | `authors-drive/release_from_mac/scores/part23/`, `part24/`, `part27/` |
| `paper1_submission_23sep/` | `authors-drive/paper1_submission_23sep/` |
| `gdrive/` | `authors-drive/paper1_local_runs/` |
| `DementiaBank/` | `authors-drive/DementiaBank/` |
| not used | `authors-machine/`, a folder on the authors' laptop |

---

## b. What is and is not in this repository

In this repository:

- the scripts that produced the released numbers (`scripts/`)
- segment lists and manifests, with speech columns and clinical columns removed (`manifests/`). Some clinical
  information remains in ids and file membership, see "What still carries clinical information" below
- per-clip score files with their sidecar JSON (`scores/`), without the clinical label (see "Clinical labels" below)
- the GroupKFold(5) speaker fold assignments behind the probes (`folds/`, see section i)
- lookup tables of every cell with its estimator and source (`lookup/`)
- every prompt string (`prompts/prompts.json`)
- the figures (`figures/`), the audit log (`DISCREPANCIES.md`) and a file list with sizes and origins (`MANIFEST.txt`)

Not in this repository:

- No audio. No `.wav`, `.flac`, `.mp3`, `.cha` or any other recording.
- No transcripts and no participant speech. No file holds utterance text or ASR output. Where a source table had
  a text column it was dropped. Counts derived from the text (`nw`, `ttr`, `retrace`, `fillers`, `unintel`,
  `errors`, `pauses`, `rate`, `p_text`, `val`, `npos`, `nneg`) are kept, because the conflict rules are defined on them.
- No clinical label and no clinical metadata column. Columns such as `age`, `sex`, `mmse`, `dx`, `sev` (PHQ-8
  total) and PHQ items were dropped from every file, and the binary label was dropped on 24 September 2026 (see
  "Clinical labels" below). DementiaBank and E-DAIC distribute those under a data-use agreement and they are not
  ours to republish. What remains in ids and file membership is listed under "What still carries clinical
  information" below.
- No participant identifier beyond each dataset's own released ids.
- No hidden states and no model weights.
- The four long-form audit files are not here, because they quote participant and interviewer speech. They stay in
  `release/edaic_rerun/` (authors' machine path) as `AUDIT_1a1c_ad.txt`, `AUDIT_1b1c_edaic.txt`, `AUDIT_1d_confounds.txt`
  and `AUDIT_1e_dupes.txt` (authors' machine paths). `DISCREPANCIES.md` summarises them.

Reproducing a number needs the original corpora, each from its own custodian: DementiaBank (Pitt, ADReSS-2020,
ADReSS-M), E-DAIC, PC-GITA, NeuroVoz and MDVR-KCL.

### Clinical labels

On 24 September 2026 the clinical label was dropped from every file that held a participant, speaker or clip id in
the same row as a label: 237 files. `lookup/LABELS_DROPPED.tsv` lists each file with the columns dropped, and
`reports/part25/LABELS_DROPPED.tsv` lists the 111 PART 25 files the same way. They are
the `label` column of the per-clip score files and manifests, the columns whose values name the diagnosis group
(`grp`, `group`, `label_y`, and the ADReSS-2020 `split`, whose values train_cc and train_cd name the group), the
`label` array of `scores/part10/pitt_enc_nested5_oof.npz` and `scores/part16/pod_sync/q3o_perp_cols.npz`, the
`label` field of `scores/part22/p22_step2_truncation_proof.json`, and the label field of ten lines of
`reports/PART5_interviewer_measure.txt`. Every id column and every score column is kept.

To recompute an AUC, read the label from your own licensed copy of the corpus and join it on the id column:

- E-DAIC: `pid`, `speaker` or `speaker_id` is the participant number. Use the dataset's released binary label
  (1 = depressed).
- DementiaBank Pitt: `spk` or `speaker` holds the participant number, and the speaker label or the clip name
  carries the group (Control or Dementia, Con or Dem). 1 = Dementia.
- ADReSSo and ADReSS-2020: the recording id (`adrso...`, `S...`) joins the challenge metadata. 1 = AD.
- PC-GITA, NeuroVoz and MDVR-KCL: the speaker id joins the corpus metadata. 1 = Parkinson's disease.

Sidecar JSON files and scripts written before this change may still name a `label` column.

### What still carries clinical information

Dropping the label columns does not remove every trace of the diagnosis. These remain in the tree:

- Ids with the group in the name. The files keep each corpus's own ids, and several corpora put the diagnosis group
  in them: Pitt speaker labels and clip names (Control or Dementia, Con or Dem), PC-GITA speaker ids (the control ids
  carry a C), NeuroVoz clip ids (PD or HC) and MDVR-KCL clip names (hc or pd). The MDVR-KCL names also carry the
  dataset's own Hoehn and Yahr stage and two UPDRS item scores. `scores/part20/POD6/B_neurovoz_matched/nv_matched_pairs.csv`
  lists NeuroVoz speaker numbers under `pd_id` and `hc_id`.
- The E-DAIC pair set. The conflict rule keeps a depressed speaker only with PHQ-8 of 15 or more and a control only
  with PHQ-8 of 4 or less (`scripts/part14/part14_select.py`, line 15). So every speaker id in the pair-set files
  implies a PHQ-8 band, and the arm of a segment together with its sentiment direction gives that speaker's label.
  These are `manifests/part14_manifest_new.*`, `manifests/part14_sentiment.json`,
  `manifests/edaic_paradox_manifest_390.csv`, `manifests/part7_heard_window.csv`, `scores/edaic_conflict_v2/`,
  `scores/part16/M/M2_distilbert_sentiment.*`, the `T1_perclip_*` files in `scores/part17/` and the POD3, POD3b and
  POD3c files in `scores/part20/`.
- The Pitt arms. The arm of a Pitt segment together with its fluency score gives the group.
- ADReSS crosswalks. `manifests/part1e/` maps ADReSSo and ADReSS-2020 ids to Pitt ids, which carry the group.
- Machine paths. Many files in `scores/`, `scripts/`, `reports/`, `manifests/` and `lookup/` still hold paths from the
  authors' machines, for example the score sidecars and the `source` column of `lookup/master_lookup.csv`.

---

## c. Estimator rules

Several cells exist in more than one version, because a probe was rerun on other folds or summarised in another
way. One version per cell is used. These are the rules the authors decided on 23 September 2026.

Many runs were made on pods. A pod is a rented cloud GPU machine running Linux. Folder names such as POD1 to POD6
name the pod a run was made on. "The Mac" is the authors' laptop.

| cell | value used | estimator | file | values not used |
|---|---|---|---|---|
| Qwen2.5-Omni, Pitt, encoder probe | 0.7706 [0.7157, 0.8220] | mean of five per-repeat AUCs, nested GroupKFold(5) with the layer chosen by an inner GroupKFold(4), speakers renumbered with seeds 0 to 4 | `scores/part10/pitt_enc_nested5_oof.npz`, folds `folds/pitt_groupkfold5_mac_seed0.csv` to `folds/pitt_groupkfold5_mac_seed4.csv` | 0.7917 [0.7324, 0.8480] (AUC of the per-clip probability averaged over the five repeats), 0.7761 (the same probe on the pod split), 0.7969 (single split) |
| Qwen2.5-Omni, Pitt, answer-state probe | 0.7709 [0.7210, 0.8224] | logistic probe on the final language model stage at the first answer position, GroupKFold(5) by speaker, scaler fit inside the training fold | `lookup/readout_direction.csv` (row pitt, column auc_probe), interval from `release/edaic_rerun/part15/H_readout_arms.csv` (authors' machine path) | 0.7636 (nested answer-state probe, mean of five, a different estimator over all stages) |
| Qwen2.5-Omni, E-DAIC, zero-shot answer | 0.8366 [0.7786, 0.8866] | whole interview with the processor cut lifted (PART 23) | `part23/A/A2_edaic_whole_zeroshot_perclip.csv` (authors' machine path; label-free copy: `scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv`), row 236 of `lookup/master_lookup.csv` | 0.8278 and 0.8285 (first 300 s), 0.8379 (PART 22, same setting, other pod) |
| Qwen2.5-Omni, E-DAIC, LM probe | 0.7437 [0.6752, 0.8097] | mean of five per-repeat AUCs, nested, GroupKFold(5, shuffle, random_state 0 to 4), whole interview (PART 23) | `part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv` (authors' machine path; label-free copy: `scores/part23/A/A3_edaic_whole_meanof5_perclip.csv`), row 239 of `lookup/master_lookup.csv`, folds `folds/edaic_full_groupkfold5_shuffle_rs0.csv` to `folds/edaic_full_groupkfold5_shuffle_rs4.csv` | 0.7001 [0.6231, 0.7727] (first 300 s, `scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv`), 0.7088 (AUC of averaged probabilities, first 300 s), 0.6853 (original run) |
| Qwen2.5-Omni, E-DAIC, encoder probe | 0.5981 [0.5267, 0.6672] | as above | as above, row 237 | 0.5884 [0.5268, 0.6523] (first 300 s), 0.5916 (averaged probabilities, first 300 s), 0.5930 (original run) |
| E-DAIC, LM probe minus zero-shot answer 0.8366, paired | -0.0929 [-0.1609, -0.0257] | one speaker draw per replicate, all five AUCs and the answer AUC recomputed inside the draw | as above | -0.1276 [-0.1993, -0.0629] (first 300 s, against 0.8278) |
| E-DAIC, encoder probe minus zero-shot answer 0.8366, paired | -0.2385 [-0.3244, -0.1505] | as above | as above | -0.2394 [-0.3188, -0.1572] (first 300 s, against 0.8278) |
| Qwen2.5-Omni, E-DAIC, projector fine-tune | 0.8589 [0.8050, 0.9029], paired minus the answer +0.0223 [-0.0064, 0.0547] | projector only, pooled out of fold, whole interview, PART 23 folds, seed 0 (PART 24) | `part24/FT/FT24_whole_seed0_perclip.csv` (authors' machine path; label-free copy: `scores/part24/FT/FT24_whole_seed0_perclip.csv`), row 243 of `lookup/master_lookup.csv` | 0.8184 (PART 24 seed 1), 0.7931 (PART 23, one fold stopped answering), 0.8150 (PART 23, first 300 s), 0.6421 (first 30 s only) |

The Pitt encoder value is the mean of the five per-repeat AUCs 0.7598, 0.7725, 0.7459, 0.7903 and 0.7843. The five
released fold files reproduce the per-clip scores of the npz exactly (largest difference 0.0). The interval in row 1
of `lookup/bootstrap_cis.csv`, [0.7324, 0.8480], belongs to the averaged-probability score 0.7917. The interval of 0.7706 is
[0.7157, 0.8220]. The older 0.7761 (`release/omni_final/omni_pitt_nested_repeats.json` (authors' machine path)) is the same five-repeat probe on the same
states, run on a pod whose GroupKFold tie order gave a different split (section i). The difference 0.7761 minus 0.7706
is 0.0055 [-0.0083, 0.0210], inside resampling noise (`release/edaic_rerun/part17/verify/T2_verify_omni_pitt_repeats.json` (authors' machine path)).

On the first 300 s the E-DAIC per-repeat AUCs are LM 0.6918, 0.7062, 0.6893, 0.7055, 0.7078 and encoder 0.5964, 0.5826,
0.5962, 0.5608, 0.6060. The same rebuild gives the answer-state probe 0.7435 [0.6793, 0.8065] and the projector probe
0.5610 [0.4871, 0.6339], both mean of five. These first-300 s values reproduce exactly. We reran the unmodified probe script on Linux
x86_64 with scikit-learn 1.9.1. It gives every published per-repeat AUC and all 25 layer choices per stream. Its
outputs are byte-identical to the published files (`scores/part17/T7b/T7b_edaic300_meanof5.json`).

### The E-DAIC window: the whole interview, with the processor cut lifted

Every Qwen2.5-Omni E-DAIC audio number in the submitted tex comes from the PART 23 run, except the 400 Hz low-pass result (line 144). The model hears the participant's
turns joined in order and capped at 900 s (89 of the 275 windows reach the cap, up to 22,500 audio tokens), with the
audio processor's default input cut lifted. The submitted tex says "The model is fed by the whole interview, with interviewer turns removed and capped at 15 minutes" (line 75).

Why the cut matters. `Qwen2_5OmniProcessor` in transformers 5.17.0 uses a Whisper feature extractor. The model's own
`preprocessor_config.json` sets its `chunk_length` to 300 (Hugging Face snapshot, quoted in
`scores/part22/P22_step1_processor_config.md`). Its `truncation` argument defaults to True. So by default the waveform
is cut at `feature_extraction_sequence_utils.py:333` before the audio encoder runs, and 217 of the 275 windows are
longer than 300 s. On the three longest windows the default processor inputs for the whole window and for its first
300 s are byte-identical (7500 audio tokens each). An earlier version of the paper used that cut window. Its values
are listed below as the first 300 s.

| PART 23, cut lifted | AUC [95%] | minus the zero-shot answer, paired [95%] | submitted tex |
|---|---|---|---|
| zero-shot answer | 0.8366 [0.7786, 0.8866] | change from the first 300 s: +0.0088 [-0.0320, 0.0510] | 0.84 |
| encoder probe, mean of five | 0.5981 [0.5267, 0.6672] | -0.2385 [-0.3244, -0.1505] | 0.60, -0.24 [-0.32, -0.15] |
| projector probe, mean of five | 0.5866 [0.5166, 0.6585] | -0.2500 [-0.3339, -0.1623] | not printed |
| language model probe, mean of five | 0.7437 [0.6752, 0.8097] | -0.0929 [-0.1609, -0.0257] | 0.74, -0.09 [-0.16, -0.03] |
| answer-state probe, mean of five | 0.8359 [0.7819, 0.8832] | -0.0007 [-0.0455, 0.0436] | 0.84 |

The paired differences are against the zero-shot answer of the same run, 0.8366. The change of the answer from the
first 300 s, +0.0088, is not significant. Files: `A2_edaic_whole_zeroshot_perclip.csv` in `part23/A/` and
`A3_edaic_whole_meanof5_perclip.csv` in `part23/pull/p23-a-probes/p23a/out/`, checked in
`part23/verify/PART23_VERIFIED.tsv` (authors' machine paths; label-free copies of the two per-clip files:
`scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv`, `scores/part23/A/A3_edaic_whole_meanof5_perclip.csv`),
and recomputed again for this version (rows 236 to 240
of `lookup/master_lookup.csv`). The per-layer curves of this run are in `scores/part23/`.

The first 300 s, the earlier window, under the same estimators: zero-shot answer 0.8278 [0.7703, 0.8822], encoder
probe 0.5884 [0.5268, 0.6523] (paired -0.2394 [-0.3188, -0.1572]), language model probe 0.7001 [0.6231, 0.7727]
(paired -0.1276 [-0.1993, -0.0629]), answer-state probe 0.7435 [0.6793, 0.8065] (paired -0.0843 [-0.1339, -0.0370])
and projector probe 0.5610 [0.4871, 0.6339], all in `scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv`.

PART 22 is a second run of the cut-lifted setting on a different pod. Its zero-shot answer is 0.8379 [0.7805, 0.8876]
against 0.8278 [0.7703, 0.8822] with the cut, paired difference +0.0101 [-0.0309, 0.0518], also not significant. On
the 217 cut windows alone it is 0.8571 against 0.8380, +0.0192 [-0.0309, 0.0703]. Both cut-lifted runs print as 0.84.
Files: `scores/part22/edaic_lifted_perclip.csv`, `scores/part22/edaic_lifted_auc.json`,
`scores/part22/p22_step2_truncation_proof.json`, `scores/part22/P22_step1_processor_config.md`.

Two zero-shot files exist for the first 300 s. The original run gives 0.8285 (`scores/edaic_windows/o25_full_zeroshot_scores.csv`)
and the rebuild that also produced the first-300 s probes gives 0.8278 (`scores/part16/EDAICFULL/edaic_full_perclip.csv`).
Their per-clip p_yes correlate at r 0.99988 (`DISCREPANCIES.md`, item 102).

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
- The Holm correction of line 95 is in `scores/part25/B/B_holm.tsv`. The tex_line column of B_holm.tsv uses the 23:23
  copy: 149 is now 141 to 144, 180 is 175 and 177, 226 is 223.
- Probes: StandardScaler fit inside the training fold, `LogisticRegression(max_iter=2000, class_weight="balanced")`,
  outer GroupKFold(5) by speaker, inner GroupKFold(4) on the training speakers picks the layer (`scripts/core/nested_repeats_all.py`).
- `p_yes` is the probability of Yes given the answer is Yes or No, from the first-token Yes and No logits at the first
  answer position. `mass` or `answer_mass` is the share of the full next-token distribution those tokens hold.
- Some columns of `scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv`, `A3_edaic_whole_meanof5_perclip.csv` and
  `A4_ctl300_samegpu_perclip.csv` store numbers as text, for example `np.float64(0.0637583763794425)` in `p_yes_whole`.
  The `auc_mean` and `auc_std` columns in `scores/leftovers_23sep/t5_verify/` do the same. To read such a column `c`
  with pandas: `df[c] = df[c].str.removeprefix("np.float64(").str.removesuffix(")").astype(float)`.
- Probe values by conflict arm use the probe fitted on all 468 Pitt clips, restricted to the arm. A probe refit inside
  one arm is not used (Qwen3-Omni answer state: 0.6394 and 0.9032 used, 0.6145 and 0.9349 not used).
- Table 2 E-DAIC agreement counts each agreement segment once. The 483 agreement rows hold 311 distinct segments
  (repeats carry identical scores), so the agreement AUC is on 311 segments. The 483-row values are superseded (section e).
- Pitt has 468 segments from 227 participants. One participant appears in the conflict manifest under both diagnosis
  groups, with a different speaker label in each, so the files count 228 speaker labels and in some splits this one person
  (3 clips) sits in two folds. Grouping on the 227 participants instead changes no conclusion (`DISCREPANCIES.md`, item 123,
  `release/edaic_rerun/part17/T3_selfcheck_participant_grouped.csv` (authors' machine path), `release/edaic_rerun/part17/T3_selfcheck_partitions.csv` (authors' machine path)).

---

## d. Table 1, cell by cell

Table 1 of the paper is Qwen2.5-Omni on seven datasets: the zero-shot answer from the audio, the zero-shot answer from
the transcript alone, and the answer after retraining only the projector. The submitted tex prints 2 decimals. Every
printed cell agrees at 2 decimals with the file below. The E-DAIC row is the whole interview with the processor cut
lifted (section c).

| dataset | clips (speakers) | column | submitted tex | this release | n | estimator | file |
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
| E-DAIC | 275 (275) | audio | 0.84 | 0.8366 | 275 | zero shot, voice wording, whole interview with the processor cut lifted (PART 23) | `part23/A/A2_edaic_whole_zeroshot_perclip.csv` (authors' machine path; label-free copy: `scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv`), row 236 of `lookup/master_lookup.csv` |
| E-DAIC | 275 (275) | text | 0.82 | 0.8230 | 275 | transcript only, full-window transcript | `release/overnight2/text_new/o25_edaicfull_text.csv` (authors' machine path) |
| E-DAIC | 275 (275) | fine-tuned | 0.86 | 0.8589 | 275 | projector only, whole interview, PART 23 folds, seed 0, pooled out of fold (PART 24) | `part24/FT/FT24_whole_seed0_perclip.csv` (authors' machine path; label-free copy: `scores/part24/FT/FT24_whole_seed0_perclip.csv`), row 243 of `lookup/master_lookup.csv` |
| Pitt | 468 (227) | audio | 0.66 | 0.6578 | 468 | zero shot, recording wording, 30 s window | `scores/part17/T5_zeroshot_perclip.csv` (dataset pitt) |
| Pitt | 468 (227) | text | 0.66 | 0.6629 | 468 | transcript only, dataset transcripts, run of 20 Sep | `release/overnight2/text_new/o25_pitt_text.csv` (authors' machine path) |
| Pitt | 468 (227) | fine-tuned | 0.77 | 0.7673 | 468 | projector only, as above | `release/omni_final/omnisft_pitt_oof.csv` (authors' machine path) |
| ADReSSo | 237 (237) | audio | 0.62 | 0.6150 | 237 | zero shot, recording wording, 30 s window | `scores/part17/T5_zeroshot_perclip.csv` (dataset adresso) |
| ADReSSo | 237 (237) | text | 0.55 | 0.5542 | 237 | transcript only, whisper-large-v3 ASR | `release/overnight2/text_new/o25_adresso_text.csv` (authors' machine path) |
| ADReSSo | 237 (237) | fine-tuned | 0.82 | 0.8246 | 237 | projector only, as above | `release/omni_final/omnisft_adresso_oof.csv` (authors' machine path) |
| ADReSS-2020 | 156 (156) | audio | 0.62 | 0.6241 | 156 | zero shot, recording wording, 30 s window | `scores/part17/T5_zeroshot_perclip.csv` (dataset adress2020) |
| ADReSS-2020 | 156 (156) | text | 0.75 | 0.7538 | 156 | transcript only, dataset transcripts, run of 20 Sep | `release/overnight2/text_new/o25_adress2020_text.csv` (authors' machine path) |
| ADReSS-2020 | 156 (156) | fine-tuned | 0.77 | 0.7684 | 156 | projector only, as above | `release/omni_final/omnisft_adress2020_oof.csv` (authors' machine path) |

Not released: the transcript-only per-clip files of all seven rows and the fine-tuned per-clip files of the six rows other than E-DAIC stay on the authors' machine, so those cells cannot be recomputed from this repository.

Notes on Table 1.

- The audio column's per-clip scores for the six datasets other than E-DAIC are in `scores/part17/T5_zeroshot_perclip.csv`, one
  block per dataset, copied from the run files `release/omni_final/omni_pcgita_zeroshot_scores.csv` (authors' machine path) and its siblings
  `omni_<dataset>_zeroshot_scores.csv` (authors' machine paths). The text and fine-tuned per-clip files stay on the authors' machine as listed.
- Pitt clips (speakers). The submitted tex prints 468 (227), the number of participants. One participant is listed
  under two labels (section c), so all Pitt score files carry 228 speaker labels.
- The PC-GITA, NeuroVoz and MDVR-KCL text cells differ in the 4th decimal between the stored json value and the per-clip
  recomputation. The printed 2 decimal values are the same either way (`DISCREPANCIES.md`, item 79).
- Two Pitt transcript runs exist. The Table 1 cell is the run of 20 September, 0.6629 [0.5983, 0.7245]. The run of
  15 September gives 0.6975 (`gdrive/omni_pitt_text.csv` (authors' machine path)). Their per-clip scores correlate at r 0.7640.
- E-DAIC. The audio answer, the probes and the fine-tune hear the whole interview with the processor cut lifted
  (section c). The transcript cell uses the transcript of the same window.
- E-DAIC fine-tuned, 0.86. The file value is 0.8589 [0.8050, 0.9029], PART 24 seed 0 on the PART 23 folds. Against
  the zero-shot answer 0.8366 the paired difference is +0.0223 [-0.0064, 0.0547], so the fine-tune does not move the
  answer. No fold stopped answering (median answer mass 0.9934 or more in every fold). A second seed on all five
  folds gives 0.8184 [0.7496, 0.8758], paired -0.0182 [-0.0519, 0.0139] (row 244 of `lookup/master_lookup.csv`).
  Both runs are checked in `part24/verify/FT24_VERIFIED.tsv` (authors' machine path) and were recomputed again for
  this version.
- Earlier E-DAIC fine-tunes, not used. The run in earlier copies of the paper, 0.6421, heard only the first 30 s of
  each window: the recipe cut every clip at 30 s (`scripts/core/sft_projector.py`, line 59, `x = x[:16000 * 30]`,
  and `window_seconds` 30 in `scores/edaic_windows/sft_full_sft.json`). It saved no checkpoint and cannot be rerun
  as it was made (`DISCREPANCIES.md`, item 89). PART 23 ran the recipe without the 30 s cut: 0.7931 [0.7263, 0.8536]
  on the whole window, where fold 3 stopped answering (all 55 of its test clips put under 0.01 of the probability on
  Yes or No), and 0.8150 [0.7465, 0.8776] on the first 300 s (rows 241 and 242 of `lookup/master_lookup.csv`).
- The per-cell lookup with all models and streams is `lookup/master_lookup.csv`. Its `source` column gives the path of
  every value. Superseded sources are listed in `lookup/SUPERSEDED.csv`. The audio and text cells above are rows of
  the lookup and agree with the submitted tex at 2 decimals (MDVR-KCL text is stored there as 0.7619, see the table). The
  fine-tuned cells of the six datasets other than E-DAIC are not rows of the lookup. They come from the fine-tune sidecars
  (`auc_oof` in `release/omni_final/omnisft_pitt_sft.json` (authors' machine path) and the matching `omnisft_<dataset>_sft.json` (authors' machine paths)) and agree with the
  per-clip recomputation.

---

## e. Table 2, cell by cell

Table 2 is the conflict test: the zero-shot answer on segments where the words point away from the diagnosis, and on
segments where they point toward it. E-DAIC has 483 conflict segments, each paired with an agreement segment of the
same speaker, and those pairs hold 311 distinct agreement segments from 138 speakers. Pitt has 146 conflict segments
(100 speakers) and 322 agreement segments (175 speakers). The Pitt arms are not paired. 48 participants appear in
both arms. By speaker label it is 47, because one participant carries two labels.
Intervals are speaker bootstrap. The paired column is conflict minus agreement with one speaker draw per replicate.

### Depression, E-DAIC

| model | conflict (483) | agreement (311 distinct) | submitted tex prints | paired conflict minus agreement | per-clip file |
|---|---|---|---|---|---|
| Qwen2-Audio | 0.2280 [0.1816, 0.2782] | 0.9608 [0.9351, 0.9814] | 0.23, 0.96 | -0.7328 [-0.7780, -0.6837] | `scores/edaic_conflict_v2/p14_q2a_zeroshot_scores.csv` |
| Qwen2.5-Omni | 0.2218 [0.1829, 0.2632] | 0.9647 [0.9376, 0.9857] | 0.22, 0.96 | -0.7429 [-0.7869, -0.6944] | `scores/edaic_conflict_v2/p14_o25_zeroshot_scores.csv` |
| Qwen3-Omni | 0.3008 [0.2454, 0.3641] | 0.9727 [0.9507, 0.9904] | 0.30, 0.97 | -0.6719 [-0.7266, -0.6105] | `scores/edaic_conflict_v2/p14_q3o_zeroshot_scores.csv` |
| Audio Flamingo 2 | 0.5286 [0.4512, 0.6015] | 0.6297 [0.5451, 0.7087] | 0.53, 0.63 | -0.1011 [-0.1647, -0.0384] | `scores/edaic_conflict_v2/p14_af2.csv` |
| Audio Flamingo 3 | 0.4235 [0.3606, 0.4880] | 0.8425 [0.7963, 0.8854] | 0.42, 0.84 | -0.4190 [-0.4878, -0.3461] | `scores/edaic_conflict_v2/p14_af3.csv` |
| Kimi-Audio | 0.5258 [0.4501, 0.5977] | 0.8936 [0.8479, 0.9329] | 0.53, 0.89 | -0.3678 [-0.4394, -0.3020] | `scores/edaic_conflict_v2/p14_kimi_zeroshot_scores.csv` |

All E-DAIC cells are in `scores/part17/T1_table2_distinct.csv`, and the per-clip rows of the 311 distinct
agreement segments are in `scores/part17/T1_perclip_distinct_agreement.csv`. Every E-DAIC cell printed in the final
tex matches the files, and all six agreement cells are printed on the 311 distinct segments.

Superseded and not used: agreement on all 483 rows (repeated segments counted each time), Qwen2-Audio 0.9478,
Qwen2.5-Omni 0.9482, Qwen3-Omni 0.9638, Audio Flamingo 2 0.6083, Audio Flamingo 3 0.8095, Kimi-Audio 0.8744
(`scores/part17/T1_table2_distinct.csv`, rows marked OLD_483rows).

The E-DAIC pairs come from the Part 14 build. It scores participant segments with RoBERTa sentiment
(`cardiffnlp/twitter-roberta-base-sentiment-latest`) at threshold 0.70. A segment needs at least 25 words and 5 s of
speech. A segment is a conflict when a participant with PHQ-8 of 15 or more speaks positively, or a control with
PHQ-8 of 4 or less speaks negatively. Files: `manifests/part14_manifest_new.json`, `manifests/part14_sentiment.json`
and the segment list `manifests/part14_manifest_new.csv`.

### Alzheimer's, Pitt

| model | conflict (146) | agreement (322) | submitted tex prints | paired conflict minus agreement | per-clip file |
|---|---|---|---|---|---|
| Qwen2-Audio | 0.4132 [0.3005, 0.5423] | 0.6985 [0.6372, 0.7597] | 0.41, 0.70 | -0.2853 [-0.3950, -0.1485] | `scores/part16/M/M8_perclip.csv`, tag primary |
| Qwen2.5-Omni | 0.5322 [0.4216, 0.6447] | 0.7054 [0.6431, 0.7658] | 0.53, 0.71 | -0.1731 [-0.2958, -0.0392] | `scores/part16/M/M8_perclip.csv`, tag primary |
| Qwen3-Omni | 0.6066 [0.4956, 0.7123] | 0.8231 [0.7664, 0.8720] | 0.61, 0.82 | -0.2165 [-0.3242, -0.1052] | `scores/part16/M/M8_perclip.csv`, tag primary |
| Audio Flamingo 2, first 30 s | 0.5835 [0.4730, 0.6913] | 0.5337 [0.4574, 0.6116] | 0.58, 0.53 | not computed on the 30 s clips | `release/scores/part20/POD4c/af2_orig30_conflict.csv` and `af2_orig30_agreement.csv` (authors' machine paths) |
| Audio Flamingo 2, full 30 to 56 s window (not used) | 0.6310 [0.5336, 0.7260] | 0.5544 [0.4760, 0.6296] | earlier copies: 0.63, 0.55 | +0.0766 [-0.0307, 0.1869] | `scores/part16/M/M8_perclip.csv`, tag af2_pitt468 (canonical file) |
| Audio Flamingo 3, copy part3 (not used) | 0.6169 [0.5188, 0.7209] | 0.5822 [0.5080, 0.6528] | no | +0.0347 [-0.0698, 0.1474] | `scores/part16/M/M8_perclip.csv`, tag primary |
| Audio Flamingo 3, copy part10 (first 30 s, the Table 2 source) | 0.6075 [0.5063, 0.7129] | 0.5855 [0.5144, 0.6554] | 0.61, 0.59 | +0.0220 [-0.0846, 0.1357] | `scores/part20/POD6/C_landed/C_af3_pitt_part10_arms.csv` |
| Kimi-Audio | 0.7197 [0.6201, 0.8147] | 0.7227 [0.6560, 0.7896] | 0.72, 0.72 | -0.0030 [-0.1134, 0.1065] | `scores/part16/M/M8_perclip.csv`, tag primary (canonical) |

All Pitt cells are in `scores/part16/M/M8_conflict_cells.csv`. The per-clip rows in `scores/part16/M/M8_perclip.csv`
carry the model, the tag, the arm and the run file each score was copied from: Qwen2-Audio `gdrive/probe2/pitt_zeroshot_scores.csv` (authors' machine path),
Qwen2.5-Omni `release/omni_final/omni_pitt_zeroshot_scores.csv` (authors' machine path), Qwen3-Omni `release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv` (authors' machine path),
Audio Flamingo 2 `gdrive/af2_results/af2_pitt468.csv` (authors' machine path), Audio Flamingo 3 `release/overnight2/part3/af3_pitt.csv` (authors' machine path) and
`release/overnight2/part10/af3_pitt.csv` (authors' machine path), Kimi-Audio `release/overnight2/part3/kimi_pitt_zeroshot_scores.csv` (authors' machine path). The arm of each clip
comes from the Pitt conflict manifest, released without its text and clinical columns as
`manifests/pitt_conflict_manifest_468.csv`.

Audio Flamingo 3 on Pitt. Two score files exist: part3 gives 0.6169 and 0.5822, part10 gives 0.6075 and 0.5855. The submitted tex prints 0.61 and 0.59, the part10 copy. A rerun on a fresh pod reproduces the part10 copy on 10 of 10 check windows within 3e-06 and the part3 copy on 5 of 10 (`DISCREPANCIES.md`, item 147). The input-length check shows that the part10 run cut every window to its first 30 s before the processor, as the text says, so the part10 cells are the paper's condition (`paper1_submission_23sep/final_checks_2325Z/af3/AF3_INPUT_LENGTH.md` (authors' machine path)). The part3 copy is not used, although `lookup/master_lookup.csv` still cites it for these two cells.

The Kimi-Audio and Audio Flamingo 2 cells use the files the authors named canonical. The other copies are superseded:
Kimi-Audio `release/overnight/kimi/kimi_pitt_zeroshot_scores.csv` (authors' machine path) gives 0.7078 and 0.7014, and Audio Flamingo 2
`gdrive/af2_pitt_audio.csv` (authors' machine path) gives 0.6691 and 0.4895 (`lookup/SUPERSEDED.csv`).

The canonical Audio Flamingo 2 file heard each Pitt window at full length, 30 to 56 s. The submitted tex prints the same
windows cut to their first 30 s, as for the other five models: conflict 0.5835 [0.4730, 0.6913], agreement 0.5337
[0.4574, 0.6116] and 0.5456 overall (`release/scores/part20/POD4c/af2_orig30_conflict.csv`, `af2_orig30_agreement.csv`
and `af2_orig30_all.csv`, authors' machine paths). No arm-difference interval was computed on the 30 s clips.

The Pitt agreement arm has 322 rows and 322 distinct segments, so the distinct-segment rule changes nothing on Pitt.

Lines 73 to 75 of `lookup/bootstrap_cis.csv` give Qwen2.5-Omni Pitt arm values 0.4444 and 0.7664 from an older score
file (`DementiaBank/ad_scores_omni.csv` (authors' machine path)). They do not match the Table 2 cells 0.5322 and 0.7054 and are not used.

---

## f. Every number in the text

### Audit note: where the submitted tex and the files disagree

This is the short list. Each item names the line of the submitted tex source, what it prints, and what the files show.
The printed numbers are kept in the table that follows, with the file value beside them. Every other number in the
text matches a file that was checked independently, except the few named after the table, which have
no file in this repository.

1. Figure 1, lines 39 and 40. Fixed in the submitted paper. `figures/fig1_gap.pdf` (md5
   a102c1a8872ba59fe73dd58700aae142) draws the E-DAIC panel on the whole interview, answer line 0.84, as in Table 1.
   The note on the earlier figure is kept under "Values of earlier copies".
2. Qwen3-Omni direction test, line 143. VERIFIED. The probe reproduces exactly on Linux with the pod's own folds
   (scikit-learn 1.9.1): 0.8307 [0.7833, 0.8741], and 0.8305 [0.7826, 0.8737] with the readout direction removed.
   The refit of 0.7864 used the folds that scikit-learn 1.7.2 made on the Mac, which order speakers with the same
   number of clips differently. Over 200 speaker splits that differ only in that order, the probe averages 0.8033
   (middle 95 percent 0.7786 to 0.8244) and only 2 of 200 reach 0.8307, so the printed 0.83 is near the top of that
   range. The probe minus the probe without the direction averages -0.0029 with training-fold scaling, as in the
   run, or +0.0034 with held-out-fold scaling, and the cosine stays between 0.0023 and 0.0090
   (`scores/leftovers_23sep/verify/pod3b_VERIFIED.tsv`, `scores/leftovers_23sep/pod3b/pod3b_values.tsv`). The
   earlier note is kept under "Values of earlier copies".
3. Significance of the gap, lines 49 and 141. Fixed in the submitted paper. All five printed gaps are paired and use the
   mean-of-five probe (`scores/part25/A/A_gaps.tsv`): PC-GITA +0.3306 [0.2498, 0.4088], NeuroVoz +0.2262
   [0.1735, 0.2851], ADReSSo +0.2602 [0.1755, 0.3396], ADReSS-2020 +0.1802 [0.0766, 0.2805], Pitt +0.1128
   [0.0455, 0.1768]. MDVR-KCL has only the single-split stand-in, +0.0625 [-0.1813, +0.2900]
   (`scores/part25/B/B_holm.tsv`). The earlier note is kept under "Values of earlier copies".
4. The E-DAIC whole interview, line 75. The window is the participant's transcript rows joined in file order, so
   rows that overlap are heard twice (`DISCREPANCIES.md`, item 2, open).
5. eGeMAPS range, line 142. The low end, 0.53, is E-DAIC on the first 30 s window, while Table 1's E-DAIC is the
   whole interview.
6. Pitt age-matched clips, lines 95 and 142. The printed values reproduce on the 275-clip join of the matched list:
   probe 0.7953, answer 0.6540, age alone 0.3866. A second join of the same list gives 273 clips from 128 speakers,
   probe 0.8191 (single split), answer 0.6534 and age alone 0.3720 (`release/omni/pitt_agematched.csv`, authors'
   machine path).
7. Prompt wording, line 86. The shifts 0.02 on average and 0.07 at most were rechecked (0.024 and 0.066 over 14
   runs). The E-DAIC pair of that check is on the first 30 s window, not the Table 1 window.
8. The cosine, line 143. The numbers match (-0.0109, random directions 0.0135). In a hidden space this large, two
   unrelated directions always give a cosine near zero, so this check cannot fail. The plainer result is that the
   readout direction separates the Pitt groups at only 0.6608, the level of the answer.
9. Release sentence, line 223. The zero-shot per-clip answer scores are in this repository, without the clinical
   label (section b). The PART 23 and PART 24 E-DAIC per-clip files are here too, without the label, in
   `scores/part23/A/` and `scores/part24/FT/`. The transcript-only and fine-tuned per-clip files of the other Table 1
   cells stay on the authors' machine.
10. Audio Flamingo 3 on Pitt, Table 2, line 167. The printed 0.61 and 0.59 are the part10 copy of the scores
    (0.6075 and 0.5855). An input-length check reran that script's loading lines with the same processor: it cut
    every Pitt window to its first 30 s before the processor (480,000 samples, 750 audio tokens), so these cells are
    the first 30 s condition the text describes (`paper1_submission_23sep/final_checks_2325Z/af3/AF3_INPUT_LENGTH.md` (authors' machine path)). `lookup/master_lookup.csv` still cites the part3
    copy (0.6169 and 0.5822) for these two cells.

### Numbers in the submitted tex

One row per number in the text of the submitted tex, except the few named after the table, which have no file in
this repository. Tables are in sections d and e. Commented-out text and grant numbers
are left out. "line" is the line of the submitted tex source. MATCH means the file value rounds to the printed value and
the sentence says what the file shows. UNVERIFIED means the value comes from one run with no independent check. The
printed number is kept. NOTE means the number matches, with a caveat given in the audit note above or in the status cell.

| line | number in the text | value in the file | file | estimator | status |
|---|---|---|---|---|---|
| 25 | on Parkinson's and Alzheimer's, probes detect the disorder with an AUC of 0.75 to 0.92 | Qwen2.5-Omni encoder probe: MDVR-KCL 0.7506 lowest, NeuroVoz 0.9213 highest (PC-GITA 0.8941, Pitt 0.7706, ADReSSo 0.8752 and ADReSS-2020 0.8043 fall inside) | `release/omni_final/omni_<dataset>_nested_repeats.json` (authors' machine paths), `scores/part10/pitt_enc_nested5_oof.npz` | mean of five per-repeat AUCs, nested | MATCH |
| 130 | zero-shot answers sit at 0.56 to 0.71 on Parkinson's and Alzheimer's, and only E-DAIC is higher | PC-GITA 0.5635 lowest, MDVR-KCL 0.7143 highest (NeuroVoz 0.6951, Pitt 0.6578, ADReSSo 0.6150, ADReSS-2020 0.6241). E-DAIC 0.8366 | `scores/part17/T5_zeroshot_perclip.csv`, row 236 of `lookup/master_lookup.csv` | zero shot | MATCH. The submitted abstract (line 25) does not print this |
| 144 | on depression the answer beats every probe | answer 0.8366. Encoder probe 0.5981, projector probe 0.5866, language model probe 0.7437, each with a paired interval below zero. Answer-state probe 0.8359, paired -0.0007 [-0.0455, 0.0436] | `part23/A/A2_edaic_whole_zeroshot_perclip.csv`, `part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv` (authors' machine paths; label-free copies: `scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv`, `scores/part23/A/A3_edaic_whole_meanof5_perclip.csv`), rows 236 to 240 of `lookup/master_lookup.csv` | whole interview, mean of five per-repeat AUCs, paired speaker bootstrap | MATCH for the encoder, projector and language model probes. The answer-state probe is level with the answer, as line 144 says. The submitted abstract (line 25) does not print this |
| 25, 175 | where the words contradict the correct diagnosis, four of six models fall below chance | E-DAIC conflict arm below 0.5: Qwen2-Audio 0.2280, Qwen2.5-Omni 0.2218, Qwen3-Omni 0.3008, Audio Flamingo 3 0.4235, all four intervals below 0.5 | `scores/part17/T1_table2_distinct.csv` | zero shot, 483 conflict segments | MATCH |
| 49, 223 | retraining only the projector narrows the gap on five of the six Parkinson's and Alzheimer's sets, and recovers the overall AUC on five of seven datasets | paired fine-tune minus zero shot: PC-GITA +0.3227 [0.2391, 0.4040], NeuroVoz +0.2078 [0.1588, 0.2578], MDVR-KCL -0.0149 [-0.2709, 0.2364], Pitt +0.1095 [0.0454, 0.1728], ADReSSo +0.2096 [0.1294, 0.2867], ADReSS-2020 +0.1443 [0.0490, 0.2370]. E-DAIC +0.0223 [-0.0064, 0.0547] (PART 24) | `release/edaic_rerun/part15/C_intervals.csv` (authors' machine path), row 243 of `lookup/master_lookup.csv` | paired speaker bootstrap | MATCH: five of six, not MDVR-KCL, and not E-DAIC. The submitted abstract (line 25) does not print this |
| 39, 40 | Figure 1, each point is that layer's out-of-fold AUC | single-split out-of-fold AUC per stage, no layer selection. The Pitt encoder curve peaks at 0.8083 (layer 27), above the nested 0.7706 the text quotes. Answer lines 0.5635, 0.6578 and 0.8366 (labelled 0.84). Panel gaps +0.3306, +0.1128, -0.2385 | `figures/fig1_gap.pdf`, `scores/part23/o25_whole_encoder_perlayer.csv`, `scores/part23/o25_whole_llm_perlayer.csv`, `scores/part23/o25_whole_proj_perlayer.csv`, `scores/part25/A/A_gaps.tsv`, row 237 of `lookup/master_lookup.csv` (E-DAIC gap), PC-GITA and Pitt curves in `release/overnight/curves/omni/` (authors' machine path) | per-stage out-of-fold AUC | MATCH on the printed labels |
| 49 | seven clinical datasets, six Audio LLMs | seven datasets and six models in Tables 1 and 2 | `lookup/master_lookup.csv` | design | MATCH. The three Alzheimer's sets share most speakers (`DISCREPANCIES.md`, item 6) |
| 49, 141 | the gap is significant on every Parkinson's and Alzheimer's set except MDVR-KCL and reverses on depression | paired mean-of-five probe minus answer: PC-GITA +0.3306 [0.2498, 0.4088], NeuroVoz +0.2262 [0.1735, 0.2851], ADReSSo +0.2602 [0.1755, 0.3396], ADReSS-2020 +0.1802 [0.0766, 0.2805], Pitt +0.1128 [0.0455, 0.1768]. MDVR-KCL has only the single-split stand-in, +0.0625 [-0.1813, +0.2900]. E-DAIC encoder probe minus answer -0.2385 [-0.3244, -0.1505]. The single-split values of the other sets are in the first row of the next table | `scores/part25/A/A_gaps.tsv`, `scores/part25/verify/A_VERIFIED.tsv`, `scores/part25/B/B_holm.tsv`, `scores/part16/M/M7_gap_per_dataset.csv`, `scores/part16/M/M7_pitt_FIXED.json`, `part23/A/A2_edaic_whole_zeroshot_perclip.csv`, `part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv` (authors' machine paths; label-free copies: `scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv`, `scores/part23/A/A3_edaic_whole_meanof5_perclip.csv`), rows 236 to 240 of `lookup/master_lookup.csv` | paired speaker bootstrap | MATCH |
| 49 | the gap holds on three further models | Qwen2-Audio, Qwen3-Omni and Kimi-Audio: encoder probe minus zero-shot answer has an interval above zero on 14 of the 18 Parkinson's and Alzheimer's cells (values in the line 141 row of these three models) | `scores/part26/PART26_TABLE.tsv`, `scores/part25/C/C_other_models.tsv` | mean of five per-repeat AUCs minus zero shot, paired speaker bootstrap | NOTE: it holds on 14 of the 18 cells, not on all of them. Line 141 names the four exceptions |
| 49 | the conflict segments stay at chance after retraining the projector | Pitt conflict arm 0.4490 to 0.5666 over the seven runs of the standard recipe, every interval including 0.5 | section g, class-balanced fine-tuning and seed repeats | projector fine-tune by arm | MATCH |
| 61 | the Yes and No token sets hold a median of at least 0.89 of the probability mass in every cell | smallest median answer mass 0.8968 | `scores/part16/M/M9_answer_mass.csv` | median share of the full next-token distribution on the Yes and No tokens, per zero-shot cell | MATCH |
| 66 | six models named | Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, Audio Flamingo 2 and 3, Kimi-Audio | `lookup/master_lookup.csv` | design | MATCH |
| 68 | on E-DAIC each conflict segment is paired with an agreement segment of the same speaker; on Pitt the arms are not paired (47 speakers appear in both) | E-DAIC: 483 pairs, each within one speaker. Pitt: 146 conflict segments (100 speakers) and 322 agreement segments (175 speakers), 47 speaker labels in both (48 participants, since one participant carries two labels) | `scores/part17/T1_table2_distinct.csv`, `manifests/pitt_conflict_manifest_468.csv` | counts | MATCH |
| 71 | a TF-IDF logistic regression on the transcripts reaches 0.43 to 0.62 on the Parkinson's sets, 0.68 on the E-DAIC transcripts and 0.84 to 0.93 on the Alzheimer's sets | TF-IDF logistic regression: MDVR-KCL 0.4315, NeuroVoz 0.5814, PC-GITA 0.6196, E-DAIC whole transcript 0.6778 (first 30 s 0.4844), Pitt 0.8353, ADReSSo 0.8471, ADReSS-2020 0.9254 | `scores/part20/POD6/A_language_saliency/A_tfidf_<dataset>_oof.csv` | speaker-disjoint folds | MATCH. On the read Parkinson's sets the text is ASR output of fixed words, so the signal there is ASR error (`DISCREPANCIES.md`, item 149) |
| 73 | read speech tasks: PC-GITA ten sentences and one read text, 1100 clips; NeuroVoz twelve isolated words, 1270 clips from 107 speakers; MDVR-KCL 37 speakers, most reading one of two passages | PC-GITA 1100 clips from 100 speakers (11 each). NeuroVoz 1270 clips from 107 speakers. MDVR-KCL 37 speakers, 32 of them reading one of the two passages | `scores/part17/T5_zeroshot_perclip.csv`, `scores/part16/M/M11_kcl_wer.csv` | clip counts, passage classification | MATCH |
| 75 | E-DAIC, 275 participants | 275 clips, 275 speakers | `scores/part16/EDAICFULL/edaic_full_perclip.csv` | counts | MATCH |
| 75 | the model is fed by the whole interview, with interviewer turns removed and capped at 15 minutes | participant turns joined in order, capped at 900 s (89 windows capped, up to 22,500 audio tokens), processor cut lifted. With the default cut, 217 of 275 windows would be cut at 300 s | `manifests/edaic_windows_full.csv`, `scores/part22/P22_step1_processor_config.md`, `scores/part22/p22_step2_truncation_proof.json`, `part23/A/A1_edaic_whole_audiotok.csv` (authors' machine path) | window | NOTE: the turns come from transcript rows joined in file order (audit note 4) |
| 75 | the transcript condition uses the whole transcript | transcript of the same window, 0.8230 | `release/overnight2/text_new/o25_edaicfull_text.csv` (authors' machine path) | zero shot on the transcript | MATCH |
| 77 | Pitt 468 segments from 227 participants; ADReSSo 237 speakers; ADReSS-2020 108 training and 48 test recordings | 468 segments, 227 participants, 228 speaker labels; 237; 108 and 48 | `manifests/pitt_conflict_manifest_468.csv`, `release/edaic_rerun/part17/part12_prep/handcheck_values.csv` (authors' machine path) | counts | MATCH |
| 77 | models hear the first 30 s, so the conflict test runs on the same clips for all six models | 30 s cap in the zero-shot scorers. Table 2 prints Audio Flamingo 2 on the first 30 s of each window. Audio Flamingo 3, the part10 copy behind Table 2, cut every window to its first 30 s before the processor (750 audio tokens) | `scripts/core/extract_probe_layers.py`, section e | design | MATCH |
| 81 | C=1, 2000 iterations, 5 folds, layer chosen inside the training folds, mean over five repeats | `LogisticRegression(max_iter=2000, class_weight="balanced")` (C left at its default 1.0), GroupKFold(n_splits=5), inner GroupKFold(4) | `scripts/core/nested_repeats_all.py` | design | MATCH. The Pitt encoder probe uses the Mac splits, the others the pod splits (section i) |
| 82 | 32 hidden states in the audio encoder and 29 in the language model, the projected input and 28 layers | 32 encoder layers, and 29 language model stages (the projected input and 28 layers) | `scripts/fig_options.py` (`N_ENC, N_LM = 32, 29`) | design | MATCH |
| 84 | learning rate 1e-4, three epochs, one clip per update, 5 folds, no seed fixed | lr 0.0001, epochs 3, folds 5 in every fine-tune sidecar | `release/omni_final/omnisft_pitt_sft.json` (authors' machine path), `scripts/core/sft_projector.py` | design | NOTE: the first run fixes only the clip order (`scripts/core/sft_projector.py` line 76, `RandomState(fold * 10 + ep)`); the reruns and the E-DAIC fine-tune fix their seeds (`scripts/part20/POD1b/std_ft_seed.py` line 24, `scripts/part24/pod/sft_whole24.py` line 23). See the seed runs in section g |
| 86 | two other wordings move the Table 1 audio answers by 0.02 on average and 0.07 at most | mean absolute change 0.024, largest 0.066, over 14 runs (two wordings on seven datasets) | `release/omni_final/omni_<dataset>_p2.json` and `omni_<dataset>_p3.json` (authors' machine paths), rechecked in `paper1_submission_23sep/final_checks_2325Z/part12/verify/values_reverified_mac.csv` (authors' machine path) | Qwen2.5-Omni zero shot | NOTE: the E-DAIC pair is on the first 30 s window (audit note 7) |
| 86 | under greedy decoding the first word is Yes or No on all 966 depression pair clips and matches the logit decision on 99 percent of the conflict clips | Yes or No on 966 of 966 clips. Agreement with the logit decision 0.9917 on conflict clips, 0.9865 overall | `release/edaic_rerun/part16/POD1/p14_o25_greedy.csv` (authors' machine path) | Qwen2.5-Omni, E-DAIC pairs | MATCH for Qwen2.5-Omni on the E-DAIC pairs |
| 89 | at least 25 words and 5 s of speech, probability of at least 0.70, PHQ-8 of 15 or more, 4 or less | nw at least 25, speech_s at least 5 (and span at least 8), threshold 0.7, sev at least 15 or at most 4 | `manifests/part14_manifest_new.json`, `manifests/part14_sentiment.json` | design | MATCH. The span rule is not in the tex |
| 89 | 488 conflict segments, 483 with an agreement segment from the same speaker, 483 pairs from 138 speakers, 311 distinct agreement segments | 488 before pairing (208 + 280). 483 pairs (205 + 278) from 138 speakers, 311 distinct agreement segments | `scores/part16/M/M2_distilbert_sentiment.csv`, `scores/part17/T1_table2_distinct.csv` | counts | MATCH |
| 91 | six fluency markers | ttr, retraces, fillers, unintelligible, errors, pauses | `scripts/upstream/build_ad_conflict.py` | design | MATCH |
| 91 | speaking rate alone separates the groups at 0.71 | 0.7082 on the 468 segments | computed from `manifests/pitt_all_segments.csv`, checked in `release/edaic_rerun/part17/part12_prep/handcheck_values.csv` (authors' machine path) | clip-level rank AUC, direction free | MATCH |
| 91 | five speaker-disjoint folds, top or bottom third | 5 folds, 33rd and 67th percentiles of the out-of-fold score | `scripts/upstream/build_ad_conflict.py`, `reports/PART13_tertiles.txt` | design | MATCH |
| 91 | 146 in conflict and 322 in agreement | 146 and 322 | `manifests/pitt_conflict_manifest_468.csv` | counts | MATCH |
| 91 | segments run from 30 to 56 s (median 33 s) and the models hear the first 30 s of each | min 30.0 s, median 33.3 s, max 55.6 s | `manifests/pitt_conflict_manifest_468.csv` (end_ms minus start_ms) | durations | MATCH |
| 95 | participants with dementia are 7.5 years older than controls per speaker | 7.54 years per speaker (6.58 per clip) | `DementiaBank/pitt_conflict_manifest.csv` (authors' machine path, not released) | mean age difference | MATCH |
| 95 | 275 clips of the 129 speakers kept by an age and sex matched selection (81 pairs over the whole dataset) | 275 clips and 129 speakers in the 20 Sep join, from 81 matched pairs. A second join gives 273 clips and 128 speakers | `release/omni/pitt_probe_estimators.csv`, `release/omni/pitt_agematched.json` (authors' machine paths) | counts | NOTE: two joins of the same list (audit note 6) |
| 95 | 95 percent intervals from 2000 speaker draws, one draw per replicate for paired differences; all 24 differences called significant in this paper survive Holm correction within their family (gaps, arms, fine-tuning, controls) | 2000 speaker draws, `numpy.random.default_rng(0)`, percentiles 2.5 and 97.5. `B_holm.tsv` has 41 rows. The 28 with role claimed all survive: 24 in the four families of the sentence (GAPS 7, ARMS 9, FINE TUNE 6, CONTROLS 2) and 4 E-DAIC conflict arms tested against chance (EXTRA CHANCE) | every sidecar json, `scores/part25/B/B_holm.tsv` | design | MATCH for the 24 claimed rows of the four named families in `scores/part25/B/B_holm.tsv` |
| 132 | on PC-GITA and NeuroVoz the transcript sits near chance; on MDVR-KCL it is above chance | transcript PC-GITA 0.5246, NeuroVoz 0.5371, MDVR-KCL 0.7634 [0.5936, 0.9118] | `lookup/master_lookup.csv`, `scores/part16/M/M10_kcl_transcript.csv` | zero shot on the transcript | MATCH |
| 132 | on E-DAIC the audio answer on the whole interview is within 0.02 of the answer from the whole transcript | audio 0.8366, transcript 0.8230 | row 236 of `lookup/master_lookup.csv`, `release/overnight2/text_new/o25_edaicfull_text.csv` (authors' machine path) | zero shot | MATCH on the values. No paired interval on the whole window is in this release. On the first 300 s, transcript minus audio is -0.0055 [-0.0566, 0.0431] |
| 132 | Qwen3-Omni, which also hears the interviews whole, 0.87; Audio Flamingo 3, which hears the first ten minutes, 0.86; Qwen2-Audio, which hears 30 s, 0.63 | 0.8741 (no window cut), 0.8565 (136 of 275 windows cut at 600 s), 0.6330 | `scores/edaic_windows/q3o_full_zeroshot_scores.csv`, `scores/edaic_windows/af3_full.csv`, `scores/edaic_windows/q2a_full_zeroshot_scores.csv`, `part23/C/C_trunc_275_derived.csv` (authors' machine path) | zero shot | MATCH |
| 132 | the E-DAIC answer rises with the audio it hears: 0.66 on the first 30 s, 0.72 on a middle 30 s, 0.81 on a middle five minutes and 0.84 on the whole interview, while the encoder probe never exceeds 0.63 on any window | answer 0.6620, 0.7205, 0.8122, 0.8366. Encoder probe, mean of five: 0.5966, 0.5770, 0.6168, 0.5884 (first 300 s) and 0.5981 (whole) | `lookup/new_results_edaic_windows.csv`, `scores/edaic_windows/`, `part23/A/A2_edaic_whole_zeroshot_perclip.csv`, `part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv` (authors' machine paths; label-free copies: `scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv`, `scores/part23/A/A3_edaic_whole_meanof5_perclip.csv`), rows 236 to 240 of `lookup/master_lookup.csv` | zero shot, mean of five per-repeat AUCs | MATCH. The single-split per-layer curves peak higher (0.6875 on the whole interview) |
| 132 | the encoder probe, its layer chosen inside the folds, never exceeds 0.63 on any window | nested encoder probe, mean of five: first 30 s 0.5966, middle 30 s 0.5770, middle 300 s 0.6168, first 300 s 0.5884, whole interview 0.5981. The single-split per-layer curves, with no layer chosen inside the folds, peak higher (0.6875 on the whole interview) | `lookup/new_results_edaic_windows.csv`, `lookup/master_lookup.csv`, `scores/part23/o25_whole_encoder_perlayer.csv` | mean of five per-repeat AUCs, layer chosen by an inner GroupKFold(4) | MATCH |
| 141 | encoder probe 0.75 to 0.92 on the Parkinson's sets and 0.77 to 0.88 on the Alzheimer's sets | MDVR-KCL 0.7506, PC-GITA 0.8941, NeuroVoz 0.9213; Pitt 0.7706, ADReSS-2020 0.8043, ADReSSo 0.8752 | as in the line 25 row | mean of five per-repeat AUCs, nested | MATCH |
| 141 | the gap ranges from 0.11 [0.05, 0.18] on Pitt to 0.33 [0.25, 0.41] on PC-GITA, with NeuroVoz at 0.23 [0.17, 0.29], ADReSSo at 0.26 [0.18, 0.34] and ADReSS-2020 at 0.18 [0.08, 0.28] | Pitt +0.1128 [0.0455, 0.1768]. PC-GITA +0.3306 [0.2498, 0.4088] (0.8941 minus 0.5635). NeuroVoz +0.2262 [0.1735, 0.2851], ADReSSo +0.2602 [0.1755, 0.3396], ADReSS-2020 +0.1802 [0.0766, 0.2805] | `scores/part25/A/A_gaps.tsv`, `scores/part16/M/M7_pitt_FIXED.json`, `lookup/master_lookup.csv` | mean-of-five probe minus zero shot | MATCH |
| 141 | Qwen3-Omni: encoder probe 0.90 on PC-GITA and 0.81 on Pitt, zero-shot answers 0.60 and 0.76, Alzheimer's gap 0.05 | 0.8969 [0.8322, 0.9454] and 0.8122 [0.7613, 0.8625]. Answers 0.5967 and 0.7619 (the file behind Table 2). Gap 0.0503 | `scores/part16/POD3/q3o_pcgita_encoder_nested_oof.csv`, `scores/part16/POD3/q3o_pitt_encoder_nested_oof.csv`, `release/overnight2/q3o_new/` (authors' machine path) | mean of five per-repeat AUCs, zero shot | MATCH |
| 141 | the encoder probes of Qwen2-Audio, Qwen3-Omni and Kimi-Audio exceed their zero-shot answers with an interval above zero on 14 of the 18 Parkinson's and Alzheimer's cells (the exceptions are MDVR-KCL for all three and Qwen3-Omni on Pitt) and sit below them on E-DAIC for all three models | 14 of 18 intervals above zero. Exceptions: MDVR-KCL Qwen2-Audio +0.1792 [-0.0503, +0.4305], Qwen3-Omni -0.0077 [-0.1618, +0.1385], Kimi-Audio +0.0518 [-0.1637, +0.2901]; Qwen3-Omni on Pitt +0.0503 [-0.0087, +0.1095]. E-DAIC: Qwen2-Audio -0.0791 [-0.1664, +0.0120], Qwen3-Omni -0.0462 [-0.1412, +0.0527], Kimi-Audio -0.0469 [-0.1393, +0.0533] | `scores/part26/PART26_TABLE.tsv` (all three models), `scores/part25/C/C_other_models.tsv` (Qwen2-Audio and Qwen3-Omni) | mean of five per-repeat AUCs minus zero shot, paired speaker bootstrap. Kimi-Audio is probed on its Whisper encoder (PART 26) | MATCH. On E-DAIC the three intervals include zero; the sentence says only that the probes sit below the answers |
| 142 | the eGeMAPS baseline reads 0.53 to 0.84; the encoder probe is above it on every set, by 0.01 on MDVR-KCL to 0.18 on ADReSSo; on PC-GITA and NeuroVoz the baseline beats the zero-shot answer by 0.27 and 0.15 | eGeMAPS E-DAIC 0.5309 (first 30 s) to NeuroVoz 0.8439. Probe minus eGeMAPS: MDVR-KCL +0.0095 smallest, ADReSSo +0.1810 largest. eGeMAPS minus answer: PC-GITA 0.2697, NeuroVoz 0.1488 | `release/overnight/egemaps_baseline.csv` (authors' machine path) | eGeMAPS logistic regression, out of fold | NOTE: the E-DAIC eGeMAPS value is on the first 30 s window (audit note 5) |
| 142 | five recording features from the quietest frames separate NeuroVoz at 0.80 and PC-GITA at 0.67 | 0.7977 and 0.6657 | `release/edaic_rerun/part16/POD4/channel_norm_neurovoz.csv`, `release/edaic_rerun/part16/POD4/channel_norm_pcgita.csv` (authors' machine paths) | five features together, GroupKFold(5) by speaker | MATCH |
| 142 | normalising brings them to 0.56 on PC-GITA (probe 0.83, -0.07 [-0.12, -0.02], answer 0.57) and 0.60 on NeuroVoz (probe 0.89, -0.03 [-0.061, -0.001]) | PC-GITA 0.5590, probe 0.8298, paired -0.0661 [-0.1181, -0.0180], answer 0.5674. NeuroVoz 0.6025, probe 0.8924, paired -0.0308 [-0.0608, -0.0012] | `release/edaic_rerun/part16/POD4/boot4a_pg.json`, `release/edaic_rerun/part16/POD4/boot4a_nv.json` (authors' machine paths) | paired speaker bootstrap | MATCH |
| 142 | 38 age and sex matched NeuroVoz pairs (907 clips): encoder probe 0.90, answer 0.63, recording features 0.71 | 0.8986 [0.8430, 0.9428], 0.6282 [0.5584, 0.6985], 0.7112 | `scores/part20/POD6/B_neurovoz_matched/` | mean of five, zero shot, five features | MATCH |
| 142 | age alone separates the Pitt groups at 0.71; on the age-matched clips age alone 0.39, encoder probe 0.80, zero-shot answer 0.65 | 0.7111; 0.3866, 0.7953, 0.6540 on the 275-clip join | `release/omni/pitt_probe_estimators.csv` (authors' machine path) | clip-level AUC; full-set probe restricted to the matched clips | NOTE: a second join gives other values (audit note 6) |
| 142 | interviewer speech sits in at least 158 of the 468 Pitt windows, more of it for dementia participants | 158 of 468 windows hold timed interviewer speech, a lower bound because untimed interviewer turns cannot be placed. Dementia 127 of 290 windows, mean interviewer time 852.5 ms per window; control 31 of 178, 186.4 ms | `scores/part16/POD4/pitt_no_interviewer.sidecar.json`, `scores/part16/POD4/cha_overlap_468.csv`, rechecked in `checks/final_2325Z/part12/verify/values_reverified_mac.csv` | counts on the heard 30 s | MATCH |
| 142 | removing it raises the answer from 0.66 to 0.72 (+0.07 [0.02, 0.11]) and leaves the encoder probe at 0.76 (-0.01 [-0.04, 0.02]) | answer 0.6578 on the paper windows and 0.7230 [0.6664, 0.7753] without the interviewer, interviewer-free minus paper +0.0652 [0.0228, 0.1081]. Encoder probe 0.7706 and 0.7643 [0.7135, 0.8116], interviewer-free minus paper -0.0063 [-0.0375, 0.0232] | `scores/part20/POD4/o25_noinv_recording.csv`, `scores/part20/POD5/POD5_nested_enc_mac_seed.csv` and their sidecars (which give paper minus interviewer-free), rechecked in `reports/part20/rows/POD4.tsv`, `reports/part20/verify/chk_nested_enc_mac_seed.log` and `checks/final_2325Z/part12/verify/values_reverified_mac.csv` | Qwen2.5-Omni zero shot; encoder probe mean of five per-repeat AUCs on the Mac splits; paired speaker bootstrap | MATCH |
| 142 | on participant-only audio the Pitt gap is 0.04 | 0.0413, the encoder probe 0.7643 minus the answer 0.7230 | `checks/final_2325Z/part12/verify/values_reverified_mac.csv` (row pitt_noinv_gap) | difference of the two values in the row above | MATCH. The paper prints no interval for it |
| 143 | language model probe at the audio-token positions 0.80, probe on the final-layer answer state 0.77 | Pitt language model probe 0.8032, answer-state probe 0.7709 | `release/omni_final/omni_pitt_nested_repeats.json` (authors' machine path), `lookup/readout_direction.csv` | mean of five, final stage at the first answer position | MATCH |
| 143 | along the readout direction the groups separate at 0.66; a probe on the same vector reads 0.77, and 0.77 with that direction removed; cosine 0.01, at the level of two random directions | AUC along the direction 0.6608, probe 0.7709, without the direction 0.7662, cosine -0.0109 (random 0.0135) | `lookup/readout_direction.csv`, `scores/part20/POD5/POD5_direction_origrerun_check.csv` | final stage at the first answer position | NOTE: the numbers match; the cosine check cannot fail (audit note 8) |
| 143 | Qwen2-Audio: readout direction 0.62, probe 0.82, 0.81 with the direction removed | 0.6174, 0.8223, 0.8114 | `scores/part17/T3_q2a_direction.csv`, `scores/part17/T3_q2a_direction_summary.csv` | final stage, held-out fold standardised | MATCH |
| 143 | Qwen3-Omni: readout direction 0.77, probe 0.83, 0.83 with the direction removed | 0.7656, 0.8307, 0.8305 (0.8276 in the arm file), from one pod run. A refit from the saved states gives 0.7864 for the probe. That refit used the Mac's folds; a Linux refit with the pod's folds gives 0.8307 again | `scores/part16/pod_sync/q3o_perp_cols.npz`, `release/edaic_rerun/part16/verify/POD3_partF_out.json` (authors' machine path), `scores/leftovers_23sep/verify/pod3b_VERIFIED.tsv`, `scores/leftovers_23sep/pod3b/pod3b_values.tsv` | final stage | NOTE: reproduces exactly, near the top of the fold-order spread (audit note 2) |
| 144 | on E-DAIC the language model probe 0.74 and the encoder probe 0.60, both below the answer of 0.84 (-0.09 [-0.16, -0.03] and -0.24 [-0.32, -0.15]); a probe on the answer state matches the answer (0.84) | 0.7437, 0.5981, answer 0.8366. Paired -0.0929 [-0.1609, -0.0257] and -0.2385 [-0.3244, -0.1505]. Answer state 0.8359, paired -0.0007 [-0.0455, 0.0436] | `part23/A/A2_edaic_whole_zeroshot_perclip.csv`, `part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv` (authors' machine paths; label-free copies: `scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv`, `scores/part23/A/A3_edaic_whole_meanof5_perclip.csv`), rows 236 to 240 of `lookup/master_lookup.csv` | whole interview, mean of five, paired | MATCH |
| 175 | four of six models below chance on depression, every model except Audio Flamingo 2 above 0.80 on agreement | conflict as in the line 25, 175 row. Lowest agreement of the other five is Audio Flamingo 3, 0.8425 on the 311 distinct segments | `scores/part17/T1_table2_distinct.csv` | zero shot | MATCH |
| 175 | intervals for the difference between the arms exclude zero for all six models on depression and for the three Qwen models on Pitt | depression: closest to zero Audio Flamingo 2, -0.1011 [-0.1647, -0.0384]. Pitt: Qwen2-Audio -0.2853 [-0.3950, -0.1485], Qwen2.5-Omni -0.1731 [-0.2958, -0.0392], Qwen3-Omni -0.2165 [-0.3242, -0.1052]; Audio Flamingo 3 +0.0220 [-0.0846, 0.1357] and Kimi-Audio -0.0030 [-0.1134, 0.1065] include zero. No interval was computed for Audio Flamingo 2 on the 30 s clips | `scores/part17/T1_table2_distinct.csv`, `scores/part16/M/M8_conflict_cells.csv`, `scores/part20/POD6/C_landed/C_af3_pitt_part10_arms.csv` | paired speaker bootstrap | MATCH |
| 175 | transcript alone 0.15 on conflict and 0.96 on agreement for Qwen2.5-Omni, 0.22 and 0.95 for Qwen3-Omni | 0.1498 and 0.9620, 0.2243 and 0.9523 | `scores/part17/T1_table2_distinct.csv`, `scores/part20/POD3c/p14_q3o_text.csv` | zero shot on the transcript | MATCH |
| 175 | on Pitt the transcript alone shows the same arm gap: 0.56 on conflict, 0.71 on agreement | 0.5560 and 0.7078 | `scores/part17/T6_pitt_text_sep20_arms.csv` | Qwen2.5-Omni zero shot on the transcript, run of 20 Sep | MATCH |
| 175 | two other prompt wordings: conflict 0.34 and 0.24 | 0.3439 [0.2900, 0.3979] and 0.2392 [0.1940, 0.2865] | `release/edaic_rerun/part16/POD1/p14_o25_audio_p2.csv`, `release/edaic_rerun/part16/POD1/p14_o25_audio_p3.csv` (authors' machine paths) | Qwen2.5-Omni zero shot | MATCH |
| 175 | sentiment thresholds 0.60 and 0.80 (646 and 281 pairs): conflict 0.26 and 0.19 | 0.2619 (646 pairs) and 0.1936 (281 pairs) | `release/edaic_rerun/part16/POD1/p14_thr060_omni.csv`, `release/edaic_rerun/part16/POD1/p14_thr080_omni.csv` (authors' machine paths) | Qwen2.5-Omni zero shot, pairs rebuilt | MATCH |
| 175 | a PHQ-8 cut at 10 in place of the gates of 15 and 4 gives 924 pairs from 237 speakers, conflict 0.18 | 924 pairs, 237 speakers, conflict 0.1759 [0.1448, 0.2095] | `release/edaic_rerun/part16/POD1/p14_phq10_omni.csv` (authors' machine path) | Qwen2.5-Omni zero shot | MATCH |
| 177 | DistilBERT SST-2 selects 1,334 pairs, conflict 0.46 [0.42, 0.51], 0.39 below the agreement arm; 420 segments both scorers call conflicting give 0.20, the 855 the first scorer calls neutral give 0.62 | 1334 pairs, 0.4619 [0.4165, 0.5102], agreement 0.8554 on distinct segments, paired -0.3934 [-0.4420, -0.3410]; 0.1951 on 420; 0.6150 on 855 | `scores/part20/POD3b/sst2_pairs_o25_audio.csv`, `scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv` | Qwen2.5-Omni zero shot | MATCH |
| 177 | Audio Flamingo 2 answers Yes on 97 percent or more of the clips, and Qwen2-Audio on 86 percent or more | Audio Flamingo 2 0.9710 (conflict) and 0.9834 (agreement). Qwen2-Audio 0.9400 and 0.8675 on the 483 rows. On the 311 distinct agreement segments of Table 2, Qwen2-Audio is 0.8424 (262 of 311, `scores/part17/T1_perclip_distinct_agreement.csv`) | `scores/part16/M/M4_yes_rates.csv` | share of clips with p_yes above 0.5 | NOTE: MATCH on the 483 rows. On the 311 distinct agreement segments, the Qwen2-Audio rate, 0.8424, is below the printed 86 percent |
| 177 | on Pitt the three Qwen models drop by 0.17 to 0.29 from agreement to conflict, none falls significantly below chance | drops 0.1731, 0.2165, 0.2853. Lowest Qwen conflict arm Qwen2-Audio 0.4132 [0.3005, 0.5423] | `scores/part16/M/M8_conflict_cells.csv` | zero shot by arm, paired speaker bootstrap | MATCH |
| 177 | "Two raters blind to the labels and to the models' answers agree with the scorer's direction on 89 and 91 percent of 80 excerpts (83 and 88 percent of the 40 conflict excerpts) and with each other on 90 percent (Cohen's $\kappa$ = 0.80)" | 0.8875, 0.9125; conflict 0.8250, 0.8750; raters 0.9000, kappa 0.7995. The scorer values are the rows with scorer roberta_heard, and the rater values are the interrater rows (scorer none), all from the second rating round | `scores/rater/rater_stats_paste_v2.txt`, `scores/rater/rater_stats_v2.csv` | item agreement, 80 E-DAIC excerpts | MATCH. Line 177 is the line of the submitted tex. The 23:23 UTC copy does not print this sentence. The first rating round is in `scores/rater/rater_stats_final.csv` |
| 223 | AUC improves significantly on every dataset except MDVR-KCL (37 speakers) and E-DAIC, which does not move | as in the line 49, 223 row. E-DAIC +0.0223 [-0.0064, 0.0547] | `release/edaic_rerun/part15/C_intervals.csv` (authors' machine path), row 243 of `lookup/master_lookup.csv` | paired speaker bootstrap | MATCH |
| 223 | on Pitt the retrained projector reaches 0.77, the level of the answer-state probe, and 0.74 to 0.81 across three further seeds | 0.7673 and 0.7709. Seeds 1 to 3: 0.7396, 0.8085, 0.7975 | `release/omni_final/omnisft_pitt_oof.csv`, `release/scores/part20/POD1b/` (authors' machine paths), `lookup/readout_direction.csv` | out-of-fold AUC | MATCH |
| 223 | its conflict arm stays at chance in every run (0.45 to 0.57, every interval including 0.5) while its agreement arm rises to 0.82 to 0.90; a class-balanced loss does not change this | seven runs of the standard recipe: conflict 0.4490 to 0.5666, agreement 0.8213 to 0.9022. Balanced loss: conflict 0.4693 [0.3686, 0.5854] | section g, class-balanced fine-tuning and seed repeats | projector fine-tune by arm | MATCH |
| 223 | LoRA, rank 8, 5.0 M trainable parameters against 4.6 M, reaches 0.825 on Pitt, 0.06 [0.02, 0.10] above the projector run, 0.55 on its conflict segments; both together 0.77 | 5,046,272 and 4,591,104 parameters. 0.8250 [0.7735, 0.8729], LoRA minus projector +0.0577 [0.0154, 0.1044], conflict 0.5520 [0.4348, 0.6833]. LoRA plus projector 0.7696 | `scores/part16/POD2/lora_pitt_oof.csv`, `scores/part16/POD2/lora_pitt_params.json`, `release/omni_final/abl2_pitt_both_oof.csv` (authors' machine path) | out-of-fold AUC, paired speaker bootstrap | MATCH |
| 223 | on E-DAIC the retrained projector stays at the zero-shot level (+0.02 [-0.01, 0.05], and 0.82 with a second seed) | seed 0: 0.8589, paired +0.0223 [-0.0064, 0.0547]. Seed 1: 0.8184 | `part24/FT/FT24_whole_seed0_perclip.csv`, `part24/FT/FT24_whole_seed1all_perclip.csv` (authors' machine paths; label-free copies: `scores/part24/FT/FT24_whole_seed0_perclip.csv`, `scores/part24/FT/FT24_whole_seed1all_perclip.csv`), rows 243 and 244 of `lookup/master_lookup.csv` | pooled out of fold, paired speaker bootstrap | MATCH |
| 223 | retraining it on the E-DAIC pairs lifts the conflict arm from 0.22 to 0.59 and lowers agreement from 0.96 to 0.67 | conflict 0.2218 to 0.5932 [0.5087, 0.6831], agreement 0.9647 to 0.6686 [0.5727, 0.7586] (311 distinct) | `scores/part20/POD3/p14_ft966_oof.csv`, `scores/part17/T1_table2_distinct.csv` | out of fold by speaker | MATCH |
| 223 | we release the segment lists, fold assignments, scripts and per-clip answer scores | segment lists, folds, scripts and the zero-shot per-clip scores are here, without the clinical label. The PART 23 and PART 24 E-DAIC per-clip files are here too, without the label (`scores/part23/A/`, `scores/part24/FT/`). The transcript-only and fine-tuned per-clip files of the other Table 1 cells are not | section b | release | NOTE (audit note 9) |
| 241 | a significant performance gain in AD and PD detection, but not in MDD | as in the line 49, 223 row. E-DAIC +0.0223 [-0.0064, 0.0547] | `release/edaic_rerun/part15/C_intervals.csv` (authors' machine path) | paired speaker bootstrap | MATCH |

Not in the table: numbers of the submitted tex that have no file in this repository.

- Line 142: "A Qwen2-Audio encoder probe trained on PC-GITA and applied unchanged to MDVR-KCL reaches 0.88 [0.75, 0.97], 0.29 [0.09, 0.50] above that model's own KCL answer of 0.59". Only the answer is here: Qwen2-Audio's zero-shot answer on MDVR-KCL is 0.5923 (`scores/part25/C/C_other_models.tsv`).
- Line 144: "Low-passing the whole interviews at 400\,Hz leaves Whisper under a third of the words, yet the answer still reaches 0.76 ($-$0.08 [$-$0.14, $-$0.01]) and the language model probe 0.66 while the encoder probe stays at 0.60".
- Line 177: "on 80 Pitt transcripts two raters agree with the fluency score's direction on 90 and 93 percent".

### More values behind the submitted tex sentences

Each row gives the topic, the line of the submitted tex it supports, the verified values and the file.

| line | topic | verified values | file | estimator |
|---|---|---|---|---|
| 141 | paired gap from a single-split probe, per dataset (not printed) | PC-GITA 0.3454 [0.2636, 0.4275], NeuroVoz 0.2209 [0.1660, 0.2783], MDVR-KCL 0.0625 [-0.1813, 0.2900], ADReSSo 0.2748 [0.1920, 0.3526], ADReSS-2020 0.1637 [0.0557, 0.2693], E-DAIC first 30 s -0.0297 [-0.1291, 0.0752] | `scores/part16/M/M7_gap_per_dataset.csv`, per clip `scores/part16/M/M7_perclip/pcgita.csv` and the other files in that folder | single-split nested encoder probe minus zero shot, paired speaker bootstrap (the single-split probes are PC-GITA 0.9089, NeuroVoz 0.9160, MDVR-KCL 0.7768, ADReSSo 0.8899, ADReSS-2020 0.7878, E-DAIC 0.6324) |
| 141 | paired gap on Pitt, corrected source | probe 0.7706 [0.7157, 0.8220], probe minus zero shot +0.1128 [0.0455, 0.1768] | `scores/part16/M/M7_pitt_FIXED.json`, `scores/part16/M/M7_perclip/pitt_FIXED.csv` | mean of five per-repeat AUCs, paired |
| 144 | paired gap on E-DAIC, whole interview | LM probe minus answer -0.0929 [-0.1609, -0.0257], encoder probe minus answer -0.2385 [-0.3244, -0.1505], projector probe minus answer -0.2500 [-0.3339, -0.1623], answer-state probe minus answer -0.0007 [-0.0455, 0.0436]. First 300 s: -0.1276 [-0.1993, -0.0629] and -0.2394 [-0.3188, -0.1572] | `part23/A/A2_edaic_whole_zeroshot_perclip.csv`, `part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv` (authors' machine paths; label-free copies: `scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv`, `scores/part23/A/A3_edaic_whole_meanof5_perclip.csv`), rows 236 to 240 of `lookup/master_lookup.csv`, `scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv` | mean of five, paired |
| 141 | Qwen3-Omni probe cells | Pitt encoder 0.8122 [0.7613, 0.8625], PC-GITA encoder 0.8969 [0.8322, 0.9454] (AUC of averaged probabilities 0.8354 and 0.9109, not used). Pitt LM 0.8477, answer state 0.8268 (older run) against 0.8445 and 0.8237 in the rerun | `scores/part16/POD3/q3o_pitt_encoder_nested_oof.csv`, `scores/part16/POD3/q3o_pcgita_encoder_nested_oof.csv`, `lookup/master_lookup.csv` | mean of five per-repeat AUCs |
| 141 | Qwen3-Omni zero-shot, two runs | Pitt 0.7619 (file behind Table 2) and 0.7673 (rerun). The rerun gives 0.7673 with sdpa attention and 0.7679 with eager attention. PC-GITA 0.5967 and 0.6015 | `release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv` (authors' machine path), `release/edaic_rerun/part16/POD3/q3o_pitt_zeroshot_scores.csv` (authors' machine path) | zero shot |
| 143 | direction test, Qwen2.5-Omni | cosine -0.0109 (random 0.0135), probe 0.7709, AUC along the readout direction 0.6608, probe with the direction removed 0.7662 | `lookup/readout_direction.csv` | final stage, first answer position |
| 143 | direction test, Qwen3-Omni | cosine 0.0051 (random 0.0177 [0.0008, 0.0498]), probe 0.8307 [0.7833, 0.8741], along the direction 0.7656 [0.7088, 0.8179], probe with the direction removed 0.8276 [0.7795, 0.8696] | `scores/part16/pod_sync/q3o_perp_cols.npz` (one pod run, reproduced since on Linux, audit note 2), `scores/leftovers_23sep/verify/pod3b_VERIFIED.tsv` | as above |
| 143 | direction test, Qwen2-Audio | cosine 0.0145 (random 0.0123 [0.0005, 0.0321]), probe 0.8223 [0.7742, 0.8671], along the direction 0.6174 [0.5539, 0.6800], probe with the direction removed 0.8114 [0.7646, 0.8577] | `scores/part17/T3_q2a_direction.csv`, `scores/part17/T3_q2a_direction_summary.csv` | as above, held-out fold standardised |
| 143 | readout equality | the AUC along the Yes minus No readout direction equals the zero-shot answer AUC: Qwen2-Audio 0.6174 against 0.6173, conflict 0.4136 against 0.4132, agreement 0.6984 against 0.6985 (Spearman 0.99996 between the projection and p_yes). Qwen2.5-Omni answer-state probe by arm: all 0.7709 [0.7210, 0.8224], conflict 0.5272 [0.4229, 0.6354], agreement 0.8672 [0.8261, 0.9064] | `scores/part17/T3_q2a_direction.json`, `release/edaic_rerun/part15/H_readout_arms.csv` (authors' machine path) | full-fit probe restricted to the arm |
| 142 | channel normalisation (loudness to -23 LUFS plus shaped noise) | NeuroVoz: five quiet-frame features 0.7977 to 0.6025 (the acceptance bar of 0.60 is missed), spectral centroid alone 0.7782 to 0.5603, zero shot 0.7108 to 0.6991 (paired -0.0117 [-0.0332, 0.0098]), encoder probe 0.9233 to 0.8924 (paired -0.0308 [-0.0608, -0.0012]). PC-GITA: five features 0.6657 to 0.5590 (passes), zero shot 0.5728 to 0.5674 (-0.0054 [-0.0437, 0.0357]), encoder probe 0.8959 to 0.8298 (-0.0661 [-0.1181, -0.0180]) | `release/edaic_rerun/part16/POD4/channel_norm_neurovoz.csv` (authors' machine path), `release/edaic_rerun/part16/POD4/channel_norm_pcgita.csv` (authors' machine path), `release/edaic_rerun/part16/POD4/boot4a_nv.json` (authors' machine path), `release/edaic_rerun/part16/POD4/boot4a_pg.json` (authors' machine path) | GroupKFold(5) by speaker, paired speaker bootstrap |
| 175 | transcript on the E-DAIC pairs | Qwen2.5-Omni transcript: conflict 0.1498 [0.1131, 0.1887], agreement 0.9620 [0.9345, 0.9822] (311 distinct), Pearson r with its audio answer 0.8442, same Yes or No on 0.8634 of clips. Qwen2-Audio transcript: 0.2359 [0.1851, 0.2886] and 0.9199 [0.8791, 0.9560], r 0.5155, same decision 0.7122. Qwen3-Omni transcript: 0.2243 [0.1757, 0.2742] and 0.9523 [0.9249, 0.9750], r 0.7634, same decision 0.8313 | `scores/part17/T1_table2_distinct.csv`, `release/edaic_rerun/part16/POD1/p14_o25_1a_joined.csv` (authors' machine path), `release/edaic_rerun/part16/POD1/p14_q2a_1a_joined.csv` (authors' machine path), `scores/part20/POD3c/p14_q3o_text.csv` | zero shot on the transcript, 483 conflict segments |
| 175 | prompt wording on the E-DAIC pairs | Qwen2.5-Omni, original wording conflict 0.2218 and agreement 0.9482 (483 rows). Wording 2: 0.3439 [0.2900, 0.3979] and 0.9119 [0.8505, 0.9594] (change +0.1221 and -0.0363). Wording 3: 0.2392 [0.1940, 0.2865] and 0.9442 [0.8926, 0.9821] (change +0.0174 and -0.0040) | `release/edaic_rerun/part16/POD1/p14_o25_audio_p2.csv` (authors' machine path), `release/edaic_rerun/part16/POD1/p14_o25_audio_p3.csv` (authors' machine path) | zero shot |
| 175 | sentiment threshold on the E-DAIC pairs | threshold 0.60: conflict 0.2619 [0.2188, 0.3075], agreement 0.9406 [0.9026, 0.9711] (646 pairs, 147 speakers). Threshold 0.80: 0.1936 [0.1501, 0.2438] and 0.9594 [0.9190, 0.9895] (281 pairs, 104 speakers). Conflict range over 0.60, 0.70 and 0.80: 0.0683 | `release/edaic_rerun/part16/POD1/p14_thr060_omni.csv` (authors' machine path), `release/edaic_rerun/part16/POD1/p14_thr080_omni.csv` (authors' machine path) | Qwen2.5-Omni zero shot, pairs rebuilt at each threshold |
| 175 | label cutoff | pairs rebuilt with PHQ-8 of 10 or more as the positive class and no severity gates: conflict 0.1759 [0.1448, 0.2095], agreement 0.9311 [0.9028, 0.9568] (924 pairs, 237 speakers) | `release/edaic_rerun/part16/POD1/p14_phq10_omni.csv` (authors' machine path) | Qwen2.5-Omni zero shot |
| 177 | Yes rates on the E-DAIC pairs (conflict, agreement, paired conflict minus agreement) | Qwen2.5-Omni 0.2236, 0.2919, -0.0683 [-0.1788, 0.0474]. Qwen2-Audio 0.9400, 0.8675, +0.0725 [0.0200, 0.1330]. Qwen3-Omni 0.3892, 0.4182, -0.0290 [-0.1540, 0.1078]. Audio Flamingo 2 0.9710, 0.9834, -0.0124 [-0.0280, 0.0020]. Audio Flamingo 3 0.7039, 0.6687, +0.0352 [-0.0491, 0.1281]. Kimi-Audio 0.8364, 0.7578, +0.0787 [0.0160, 0.1461]. Median answer mass 0.9249 to 0.9998 | `scores/part16/M/M4_yes_rates.csv`, `scores/part16/M/M4_yes_rate_diffs.csv` | share of clips with p_yes above 0.5, 483 rows per arm |
| 223 | LoRA baseline on Pitt | LoRA rank 8 on q, k, v, o of all 28 language model layers: 0.8250 [0.7735, 0.8729], conflict 0.5520 [0.4348, 0.6833], agreement 0.9199 [0.8846, 0.9513], conflict minus agreement -0.3679 [-0.4861, -0.2433]. LoRA minus projector fine-tune +0.0577 [0.0154, 0.1044], LoRA minus zero shot +0.1672 [0.1071, 0.2264]. Trainable parameters 5,046,272 (0.04702 percent of the model) against 4,591,104 for the projector (0.042779 percent). LoRA plus projector 0.7696. An earlier LoRA run gives 0.8202 | `scores/part16/POD2/lora_pitt_oof.csv`, `scores/part16/POD2/lora_pitt_params.json`, `release/omni_final/abl2_pitt_both_oof.csv` (authors' machine path), `release/omni_final/abl_pitt_oof.csv` (authors' machine path) | out-of-fold AUC, 5 folds, paired speaker bootstrap |
| 86 | greedy decoding check | the first generated word is Yes or No on 966 of 966 clips (share 1.0000). It agrees with the logit decision on 0.9865 of clips (conflict 0.9917, agreement 0.9814) | `release/edaic_rerun/part16/POD1/p14_o25_greedy.csv` (authors' machine path) | Qwen2.5-Omni, E-DAIC pairs |
| 132 | MDVR-KCL intervals | transcript 0.7634 [0.5936, 0.9118], audio 0.7143 [0.5210, 0.8810], encoder probe (one saved split) 0.7768 [0.5994, 0.9265], transcript minus audio 0.0491 [-0.1827, 0.2818], transcript minus probe -0.0134 [-0.2383, 0.2381] | `scores/part16/M/M10_kcl_transcript.csv` | 37 speakers, paired speaker bootstrap |

### Values of earlier copies that the submitted tex no longer prints

Earlier copies of the text had these sentences: the copy pasted on the morning of 23 September and the copy of
23 September, 13:34 UTC, that the first version of this README followed. The submitted tex dropped, corrected or
commented them out. Their values stay here. Rows marked 23:23 copy come from the Overleaf copy of 23 September,
23:23 UTC, which the previous version of this README followed, with the line of that copy.

| sentence in the earlier copy | value in the file | file |
|---|---|---|
| E-DAIC answer 0.83, language model probe 0.70 and encoder probe 0.59 (13:34 copy) | first 300 s: 0.8278, 0.7001, 0.5884. The submitted tex uses the whole interview (section c) | `scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv` |
| E-DAIC fine-tuned answer 0.64, "drops" (13:34 copy) | 0.6421, a run that heard only the first 30 s of each window | `scores/edaic_windows/sft_full_oof.csv` |
| the gap is widest on Parkinson's, narrower on Alzheimer's and shrinks as the words carry more (13:34 copy) | gaps PC-GITA 0.3306, ADReSSo 0.2602, NeuroVoz 0.2262, ADReSS-2020 0.1802, Pitt 0.1128, MDVR-KCL 0.0363: the order does not follow how much the words carry | `lookup/master_lookup.csv`, `scores/part20/POD6/A_language_saliency/` |
| PC-GITA gap 0.35, Qwen3-Omni Pitt answer 0.77, eGeMAPS margin from 0.06, Qwen Pitt drop from 0.18 (13:34 copy) | 0.3306, 0.7619, 0.0095 (MDVR-KCL), 0.1731 | as in section f |
| speaking rate alone 0.92, and 91 percent of Pitt segments with half their words inside 30 s (13:34 copy) | 0.7082. No file holds the 91 percent | `manifests/pitt_all_segments.csv` |
| Table 2: Audio Flamingo 2 Pitt 0.63 and 0.55, Audio Flamingo 3 E-DAIC agreement 0.81 (13:34 copy) | full-length Audio Flamingo 2 windows, and the 483-row agreement value 0.8095 | section e |
| Qwen2-Audio probes 0.77 to 0.93 against answers of 0.54 to 0.68 | encoder MDVR-KCL 0.7714 to NeuroVoz 0.9324, zero shot PC-GITA 0.5369 to ADReSS-2020 0.6824 | `release/omni_final/q2a_kcl_nested_repeats.json`, `release/omni_final/q2a_neurovoz_nested_repeats.json` (authors' machine paths), `gdrive/probe2/pcgita_zeroshot_scores.csv`, `gdrive/probe2/adress2020_zeroshot_scores.csv` (authors' machine paths) |
| Qwen3-Omni probes 0.77 to 0.94 against answers of 0.60 to 0.78 | encoder MDVR-KCL 0.7661 to NeuroVoz 0.9372, zero shot PC-GITA 0.5967 to ADReSSo 0.7774 | `release/overnight2/q3o_new/` (authors' machine path) |
| the Alzheimer's and E-DAIC sets stay below 0.64 on every recording feature | largest such feature 0.6358 (ADReSS-2020 noise floor) | `manifests/confounds/confound_auc.csv` |
| for AD the transcript alone matches the audio on Pitt, beats it on ADReSS-2020 and trails it on ADReSSo (commented out in the submitted tex) | transcript against audio: Pitt 0.6629 and 0.6578, ADReSS-2020 0.7538 and 0.6241, ADReSSo 0.5542 and 0.6150 | `lookup/master_lookup.csv` |
| a projector trained on PC-GITA lifts NeuroVoz from 0.70 to 0.74, and one trained on NeuroVoz lifts PC-GITA from 0.56 to 0.68 | 0.6951 to 0.7375, and 0.5635 to 0.6761 | `release/overnight2/part6/transfer_pcgita_transfer.json`, `release/overnight2/part6/transfer_neurovoz_transfer.json` (authors' machine paths) |
| Figure 1 answer lines 0.5635, 0.6578 and 0.8285 (labelled 0.83), E-DAIC panel on the first 300 s run (23:23 copy) | 0.8285 | `figures/drafts/fig1_gap_overleaf_2325Z.pdf`, `scores/edaic_windows/o25_full_encoder_perlayer.csv`, `scores/edaic_windows/o25_full_llm_perlayer.csv` |
| E-DAIC, 275 participants, the released label is stricter than PHQ-8 of 10 or more for 20 participants (23:23 copy, line 81) | 275 clips, 275 speakers, 20 participants whose released label differs from PHQ-8 of 10 or more (counts, MATCH) | `scores/part16/EDAICFULL/edaic_full_perclip.csv`, `DISCREPANCIES.md` item 112 |
| the model hears the participant's whole interview, interviewer turns removed and capped at 15 minutes for the 89 longest, with the processor's default input limit lifted (23:23 copy, line 81) | participant turns joined in order, capped at 900 s (89 windows capped, up to 22,500 audio tokens), processor cut lifted. With the default cut, 217 of 275 windows would be cut at 300 s (window, NOTE: the turns come from transcript rows joined in file order (audit note 4)) | `manifests/edaic_windows_full.csv`, `scores/part22/P22_step1_processor_config.md`, `scores/part22/p22_step2_truncation_proof.json`, `part23/A/A1_edaic_whole_audiotok.csv` (authors' machine path) |
| Pitt 468 segments from 227 participants (one grouped under two labels); ADReSSo 237 speakers; ADReSS-2020 108 training and 48 test recordings (23:23 copy, line 83) | 468 segments, 227 participants, 228 speaker labels; 237; 108 and 48 (counts, MATCH) | `manifests/pitt_conflict_manifest_468.csv`, `release/edaic_rerun/part17/part12_prep/handcheck_values.csv` (authors' machine path) |
| 95 percent intervals from 2000 speaker draws, one draw per replicate for paired differences, no correction for multiple comparisons (23:23 copy, line 101) | 2000 speaker draws, `numpy.random.default_rng(0)`, percentiles 2.5 and 97.5 (design, MATCH) | every sidecar json |
| all 708 Pitt segments: encoder probe 0.80, zero-shot answer 0.67 (23:23 copy, line 149) | 0.7968 and 0.6695 (mean of five, zero shot, MATCH) | `scores/pitt_all/o25_pitt_all_nested_repeats.json`, `scores/pitt_all/o25_pitt_all_zeroshot_scores.csv` |
| the transcript decision matches the audio decision on 86 percent of clips (Qwen2.5-Omni) (23:23 copy, line 180) | Qwen2.5-Omni 0.8634 (Qwen3-Omni 0.8313) (same Yes or No decision, MATCH) | `release/edaic_rerun/part16/POD1/p14_o25_1a_joined.csv` (authors' machine path) |
| Audio Flamingo 2 answers Yes on 97 percent or more of the clips, Qwen2-Audio on 87 percent or more (23:23 copy, line 180) | Audio Flamingo 2 0.9710 (conflict) and 0.9834 (agreement). Qwen2-Audio 0.9400 and 0.8675 (share of clips with p_yes above 0.5, MATCH) | `scores/part16/M/M4_yes_rates.csv` |
| on Pitt the three Qwen models drop by 0.17 to 0.29 from agreement to conflict, none falls significantly below chance, Kimi-Audio shows no drop, the two Audio Flamingo models score 0.53 to 0.61 on both arms (23:23 copy, line 180) | drops 0.1731, 0.2165, 0.2853. Lowest Qwen conflict arm Qwen2-Audio 0.4132 [0.3005, 0.5423]. Kimi-Audio -0.0030 [-0.1134, 0.1065]. First 30 s: Audio Flamingo 2 0.5835 and 0.5337, Audio Flamingo 3 0.6075 and 0.5855 (zero shot by arm, paired speaker bootstrap, MATCH. The Audio Flamingo 3 values are the part10 copy, which heard the first 30 s (audit note 10)) | `scores/part16/M/M8_conflict_cells.csv`, `scores/part20/POD6/C_landed/C_af3_pitt_part10_arms.csv`, section e |
| asking Qwen2.5-Omni to judge the voice alone moves its Pitt answer by -0.01 [-0.04, 0.02] (23:23 copy, line 180) | 0.6473 [0.5931, 0.7039] against 0.6578, paired -0.0104 [-0.0356, 0.0155] (Qwen2.5-Omni zero shot, paired, MATCH) | `scores/part20/POD4/o25_orig_voice.csv` |

Audit notes of the 23:23 copy that the submitted paper fixed, word for word:

1. Figure 1, lines 39 and 40. The figure the final tex includes, `figures/drafts/fig1_gap_overleaf_2325Z.pdf` (the same bytes as the
   Overleaf copy, md5 bbefb559a5768dcb07d11ed5d440da18), draws the E-DAIC panel from the first 300 s run: its answer
   line is labelled 0.83. Table 1 and the text use the whole interview, where the answer is 0.84. A whole-interview
   version of the figure exists (`scripts/make_fig1_whole.py`, curves in `scores/part23/`) but the final tex does
   not include it.
3. Significance of the gap, lines 52 and 149. Only Pitt has a paired interval on the paper's estimator (mean of five
   per-repeat AUCs): +0.1128 [0.0455, 0.1768]. For PC-GITA, NeuroVoz, ADReSSo and ADReSS-2020 the interval behind
   "excludes zero" uses a single-split probe (`scores/part16/M/M7_gap_per_dataset.csv`). All four exclude zero there.

Audit notes of the 23:23 copy that a later check settled, word for word:

2. Qwen3-Omni direction test, line 149. The probe 0.83 and the probe with the readout direction removed, 0.83, come
   from one pod run with no independent check. A refit of the same probe from the saved states gives 0.7864, not
   0.8307 (`release/edaic_rerun/part16/verify/POD3_partF_out.json`, authors' machine path). The printed numbers are
   kept and marked UNVERIFIED.

---

## g. Released analyses beyond the paper

Each analysis below was run after the 14 September draft. Some of it is in the paper; each subsection says where. Where the submitted tex uses one
in a sentence, the subsection says which line. Intervals are speaker bootstrap as in section c. "Paired" means one
speaker draw per replicate for both quantities.

### Interviewer removal on Pitt, ADReSSo and ADReSS-2020, with a duration control

In the submitted paper: line 142 (at least 158 of 468 windows; no move on ADReSS-2020 or ADReSSo; duration-matched cut:
answer -0.0352 [-0.0729, -0.0001] in `scores/part16/POD4/boot4b.json`).

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
probe drops a little without the interviewer. On Pitt, 158 of 468 windows contain timed interviewer speech. The
interviewer's duration alone predicts dementia at 0.6427 and the participant-only duration left in arm B predicts it
at 0.6648 (direction free). The encoder values here are single-split re-runs on the re-cut audio. The Table 1 estimator is the mean of five.

Files: `scores/part16/POD4/p16_pitt_orig_zeroshot_scores.csv`, `scores/part16/POD4/p16_pitt_paronly_zeroshot_scores.csv`,
`scores/part16/POD4/p16_pitt_durmatch_zeroshot_scores.csv`, the matching `_enc_nested_oof.csv` files,
`scores/part16/POD4/boot4b.json`, `scores/part16/POD4/pitt_no_interviewer.csv`,
`scores/part16/POD4B/adress2020_no_interviewer.csv`, `scores/part16/POD4B/adresso_no_interviewer.csv`.

### Interviewer-free Pitt window, all six models (PART 20)

Partly in the submitted paper: line 142 reports the Qwen2.5-Omni row (0.66 to 0.72, +0.07 [0.02, 0.11]; probe 0.76,
-0.01 [-0.04, 0.02]); the values are in the Pitt sentence of the subsection above (158 of 468 windows), the
Qwen2.5-Omni row of the table below and the paragraph that starts "Qwen2.5-Omni probes and fine-tune". The other five
models are not in the paper.

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
The Audio Flamingo 2 paper file scored each window at full length. So we added a same-span control. On the first
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

In the submitted tex: line 177 (1,334 pairs, conflict 0.46 [0.42, 0.51], 0.39 below agreement, 0.20 on the 420 segments
both scorers call conflicting and 0.62 on the 855 the first scorer calls neutral).

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

In the submitted tex: line 223 (0.74 to 0.81 across three further seeds, conflict 0.45 to 0.57, agreement 0.82 to 0.90,
one sentence on the class-balanced loss, and LoRA 0.825 with +0.06 [0.02, 0.10] over the projector run).

The projector and LoRA recipes were rerun with per-clip loss weights that balance the conflict and agreement arms, each
beside a same-pod control with uniform weights.

| run | overall | conflict | agreement | file |
|---|---|---|---|---|
| projector, paper file | 0.7673 [0.7086, 0.8231] | 0.5189 [0.4151, 0.6300] | 0.8593 [0.8063, 0.9069] | `release/omni_final/omnisft_pitt_oof.csv` (authors' machine path) |
| projector, balanced | 0.7301 [0.6656, 0.7905] | 0.4693 [0.3686, 0.5854] | 0.8279 [0.7732, 0.8792] | `scores/part20/POD1/balanced_ft_pitt_oof.csv` |
| projector, uniform control (rerun A) | 0.7720 [0.7166, 0.8280] | 0.4697 [0.3613, 0.5906] | 0.8895 [0.8456, 0.9286] | `scores/part20/POD1/standard_rerun_ft_pitt_oof.csv` |
| projector, uniform control, repeat 2 (rerun B) | 0.7734 [0.7172, 0.8286] | 0.4490 [0.3385, 0.5682] | 0.8967 [0.8592, 0.9322] | `release/scores/part20/POD1/standard_rerun_ft_pitt_r2_oof.csv` (authors' machine path) |
| projector, standard recipe seed 0 | 0.7955 [0.7393, 0.8478] | 0.5282 [0.4200, 0.6421] | 0.8983 [0.8575, 0.9360] | `release/scores/part20/POD1b/std_ft_seed0_pitt_oof.csv` (authors' machine path) |
| projector, standard recipe seed 1 | 0.7396 [0.6891, 0.7921] | 0.5364 [0.4306, 0.6426] | 0.8213 [0.7716, 0.8689] | `release/scores/part20/POD1b/std_ft_seed1_pitt_oof.csv` (authors' machine path) |
| projector, standard recipe seed 2 | 0.8085 [0.7560, 0.8572] | 0.5636 [0.4593, 0.6779] | 0.9022 [0.8648, 0.9361] | `release/scores/part20/POD1b/std_ft_seed2_pitt_oof.csv` (authors' machine path) |
| projector, standard recipe seed 3 | 0.7975 [0.7456, 0.8479] | 0.5666 [0.4660, 0.6746] | 0.8844 [0.8424, 0.9225] | `release/scores/part20/POD1b/std_ft_seed3_pitt_oof.csv` (authors' machine path) |
| LoRA, earlier run | 0.8250 [0.7735, 0.8729] | 0.5520 [0.4348, 0.6833] | 0.9199 [0.8846, 0.9513] | `scores/part16/POD2/lora_pitt_oof.csv` |
| LoRA, balanced | 0.7200 [0.6494, 0.7866] | 0.5770 [0.4586, 0.7006] | 0.7843 [0.7074, 0.8510] | `scores/part20/POD2/balanced_lora_pitt_oof.csv` |
| LoRA, uniform control | 0.8346 [0.7856, 0.8825] | 0.6198 [0.4963, 0.7470] | 0.9112 [0.8710, 0.9471] | `scores/part20/POD2/control/uniform_lora_pitt_control_oof.csv` |

Balancing the arms does not lift the conflict arm. Balanced minus control on conflict is -0.0004 [-0.0742, 0.0757]
for the projector and -0.0428 [-0.1707, 0.0837] for LoRA. For LoRA it also costs overall AUC (-0.1146
[-0.1751, -0.0564]). Three new seeds of the standard projector recipe span 0.0688 overall (0.7396 to 0.8085) and
0.0476 on conflict once the paper run is included (0.5189 to 0.5666), about as large as the differences between
recipes. The conflict arm stays near chance in every run. In the first balanced projector run one fold collapsed
(answer mass near zero on its test clips). With that fold replaced by a lone rerun the balanced run gives 0.7603,
0.5003 and 0.8573 (`scores/part20/POD1/balanced_ft_pitt_fold3rerun_oof.csv`).

### NeuroVoz, age and sex matched subset

In the submitted tex: line 142 (38 pairs, encoder probe 0.90, zero-shot answer 0.63).

NeuroVoz speakers were paired PD to control on sex and age within 5 years (38 pairs, 76 speakers, 907 clips, which
is the largest such matching). On this subset the Qwen2.5-Omni zero-shot answer is 0.6282 [0.5584, 0.6985] against
0.6951 on all 1270 clips, and the encoder probe (mean of five) is 0.8986 [0.8430, 0.9428]. The five quiet-frame
channel features together still separate the groups at 0.7112 [0.6079, 0.7961], and the spectral centroid alone at
0.7261 (direction free). The 363 clips left out of the subset give a zero-shot answer of 0.8462 [0.7645, 0.9144].
The encoder probe on NeuroVoz stays high after matching. The channel cue remains, and the zero-shot answer is higher
on the unmatched clips.
Files: `scores/part20/POD6/B_neurovoz_matched/B_zeroshot_nvmatched.csv`,
`scores/part20/POD6/B_neurovoz_matched/B_encprobe_nested5_nvmatched_oof.csv`,
`scores/part20/POD6/B_neurovoz_matched/B_channel5_nvmatched.csv`, `scores/part20/POD6/B_neurovoz_matched/nv_matched_pairs.csv`
(released without age and sex).

### How much the words carry, per dataset

In the submitted tex: line 71 (a TF-IDF logistic regression, 0.43 to 0.62, 0.68 and 0.84 to 0.93).

A TF-IDF logistic regression on the transcript alone, speaker-disjoint folds: PC-GITA 0.6196 [0.5700, 0.6681],
NeuroVoz 0.5814 [0.5422, 0.6206], MDVR-KCL 0.4315 [0.2560, 0.6287], E-DAIC first 30 s 0.4844 [0.4028, 0.5624],
E-DAIC full interview 0.6778 [0.5973, 0.7549], Pitt 0.8353 [0.7865, 0.8862], ADReSSo 0.8471 [0.7961, 0.8922],
ADReSS-2020 0.9254 [0.8866, 0.9587]. The ordering Parkinson's below depression below Alzheimer's holds only with
the full E-DAIC transcript. On the read-speech Parkinson's sets the only text is ASR output, so what the
classifier reads there is ASR errors caused by the voice.
Files: `scores/part20/POD6/A_language_saliency/A_tfidf_<dataset>_oof.csv` with its sidecar, and the summary
`release/scores/part20/POD6/A_language_saliency/A_summary.json` (authors' machine path).

### Projector fine-tune on the E-DAIC conflict pairs

In the submitted tex: line 223 (conflict 0.22 to 0.59, agreement 0.96 to 0.67).

Qwen2.5-Omni's projector trained on the 966 clips of the E-DAIC pairs, out of fold by speaker: conflict 0.5932
[0.5087, 0.6831], agreement 0.6686 [0.5727, 0.7586] on the 311 distinct segments, paired -0.0754 [-0.1513, 0.0010].
Training on the pairs removes most of the gap between the arms, and both arms end near 0.6.
File: `scores/part20/POD3/p14_ft966_oof.csv`.

### Pitt, all 708 segments (Part 13)

Not in the submitted paper. The 23:23 copy printed it at line 149 (encoder probe 0.80, zero-shot answer 0.67).

The paper's Pitt set keeps the top and bottom thirds of a text fluency score and drops the middle third
(`scripts/upstream/build_ad_conflict.py`, lines 98 to 106). Rebuilt with every third kept: 708 segments from
266 speaker labels (265 participants, because the same participant again carries two labels).

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

In the submitted tex: line 132 (the answer by window, 0.66, 0.72, 0.81 and 0.84). Table 1 and line 144 use the whole
interview with the processor cut lifted (PART 23, section c), the last column below.

The 14 September draft scored E-DAIC on the first 30 s of each file, which is mostly session setup. Every model was
rerun with the same prompt and scoring on three other windows cut from the participant's stitched speech: the middle
30 s, the middle 300 s, and the interview window capped at 900 s. For Qwen2.5-Omni the default processor keeps only
the first 300 s of that window, and PART 23 reran it with the cut lifted (section c). All n = 275.

Zero-shot answer by window:

| model | first 30 s | middle 30 s | middle 300 s | interview window |
|---|---|---|---|---|
| Qwen2.5-Omni | 0.6620 | 0.7205 | 0.8122 | 0.8366 with the cut lifted (PART 23; PART 22 0.8379). First 300 s: 0.8285, rebuild 0.8278 |
| Qwen3-Omni | 0.6290 | 0.7153 | 0.8217 | 0.8741 |
| Audio Flamingo 3 | 0.5975 | 0.6949 | 0.8374 | 0.8565 |
| Qwen2-Audio | 0.6464 | 0.6891 | 0.6332 | 0.6330 |
| Kimi-Audio | 0.6293 | 0.6664 | 0.6581 | 0.5929 |

Qwen2.5-Omni, every stream by window:

| stream | first 30 s | middle 30 s | middle 300 s | first 300 s of the interview window | whole interview, cut lifted (submitted tex) |
|---|---|---|---|---|---|
| zero shot answer | 0.6620 | 0.7205 | 0.8122 | 0.8285 | 0.8366 |
| encoder probe | 0.5966 | 0.5770 | 0.6168 | 0.5884 | 0.5981 |
| projector probe | 0.5041 | 0.4380 | 0.5703 | 0.5610 | 0.5866 |
| LM probe | 0.4939 | 0.6249 | 0.6351 | 0.7001 | 0.7437 |
| answer-state probe | 0.6146 | 0.6593 | 0.7065 | 0.7435 | 0.8359 |
| projector fine-tune | 0.6083 | 0.6669 | 0.6595 | 0.6421 (first 30 s of the window only). Without the 30 s cut: 0.8150 | 0.8589 (PART 24 seed 0; seed 1 0.8184) |
| transcript only | 0.6613 | no value | no value | 0.8230 | 0.8230 (the same whole-window transcript) |

The first-300 s probe values are the mean-of-five values of the rebuild, and the last column is PART 23 and PART 24
(section c). The original run on that
window gave encoder 0.5930, projector 0.5293, LM 0.6853 and answer state 0.7542. The projector probe values for the
first 30 s, middle 30 s and middle 300 s are one repeat. The first-30 s fine-tune value is `release/omni_final/omnisft_edaic_oof.csv` (authors' machine path).
On both the first 300 s and the whole interview the answer is above the encoder, projector and language model
probes (section c). On the first 300 s the transcript alone matches the audio answer (-0.0055 [-0.0566, 0.0431]).

The fine-tune row needs care. The original recipe cuts every clip to its first 30 s (`scripts/core/sft_projector.py`,
line 59). So the first-300 s cell, 0.6421, heard only the first 30 s of each window, and its gap of 0.1863
[0.1098, 0.2696] below the zero-shot answer compares a 30 s fine-tune with a 300 s answer. Without the 30 s cut the
recipe gives 0.8150 on the first 300 s, paired -0.0128 [-0.0483, 0.0214], and 0.8589 on the whole interview, paired
+0.0223 [-0.0064, 0.0547] (section d). The middle 30 s and middle 300 s fine-tune sidecars also read
`window_seconds` 30, so read those two cells with the same care.

Files: per-clip scores and sidecars for the three windows in `scores/edaic_windows` (window cut points in
`manifests/edaic_windows_full.csv`, `manifests/edaic_windows_mid30.csv` and `manifests/edaic_windows_mid300.csv`),
the rebuild in `scores/part16/EDAICFULL/edaic_full_perclip.csv` and `scores/part16/EDAICFULL/full_zeroshot_scores.csv`,
the mean-of-five per-repeat scores in `scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv`, the truncation-off
runs in `scores/part22/edaic_lifted_perclip.csv` and rows 236 to 244 of `lookup/master_lookup.csv`, the whole-interview
per-layer curves in `scores/part23/`, and the table as CSV in `lookup/new_results_edaic_windows.csv`
and as text in `reports/EDAIC_REPORT.txt`.

Caveats for every window. Each window is cut from a stitched file of the participant's ASR segments, so
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

Line 86 of the submitted tex says two other wordings are listed with the release. They are the two templates in
`scripts/core/omni_prompt_variants.py` (lines 9 and 10), where `{CONDNAME}` is Parkinson's disease, depression or
dementia:

- p2: `Listen to this recording. Does the speaker have {CONDNAME}? Answer Yes or No.`
- p3: `From the audio alone, is this speaker likely to have {CONDNAME}? Reply with one word, Yes or No.`

Their E-DAIC instances, with the filled prompt in the prompt column, are `scores/part16/POD1/p14_o25_audio_p2.csv` and
`scores/part16/POD1/p14_o25_audio_p3.csv`, the released copies of the two E-DAIC files above.

Yes and No token spellings. Line 61 of the submitted tex says the single-token spellings of Yes and No are listed in
the release. They are:

- Audio runs (`scripts/core/extract_probe_layers.py`, line 44): Yes, " Yes", yes, " yes", YES against No, " No", no,
  " no", NO. The script keeps the first token id of each spelling.
- Transcript runs (`scripts/core/text_score.py`, lines 35 to 42): Yes, yes, YES, Yeah, yeah against No, no, NO, Nope,
  nope, each with and without a leading space. Only spellings that are a single token count.
- Rule 3 of the PART 20 checks adds " YES" and " NO" to the audio lists (`DISCREPANCIES.md`, items 143, 157 and 163,
  which also name the column of each file that uses it).

No prompt contains participant speech. The transcript conditions place the transcript before the prompt at run time
(`scripts/core/text_score.py`, line 47).
Those transcripts are not released.

---

## i. Folds

The GroupKFold(5) speaker fold assignment of every probe split that was run is in `folds/`, one csv per dataset and
split, with columns `clip_id`, `speaker_id` and `fold`. `folds/README.md` lists every file, which run it reproduces
and how it was checked.

Why a dataset has more than one split. Without shuffling, scikit-learn's GroupKFold orders speakers by clip count
with `numpy.argsort`. Speakers with equal counts stay in an order that differed between the Mac used for some runs
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
| Qwen3-Omni Pitt and PC-GITA re-extraction (part16 POD3) | pod | `folds/pitt_groupkfold5_pod_unpaddedids_seed0.csv` to `folds/pitt_groupkfold5_pod_unpaddedids_seed4.csv`, `folds/pcgita_groupkfold5_pod_seed0.csv` to `folds/pcgita_groupkfold5_pod_seed4.csv` |
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
  1.8.0). The cause is the float32 fit path. Two inner layer choices sit near a tie (LM repeat 1 fold 1, answer state
  repeat 0 fold 1). Casting the input to float64 on the pod flips exactly those two. The unmodified script on
  Linux x86_64 with scikit-learn 1.9.1 and numpy 2.1.2 reproduces every per-repeat AUC and all layer choices exactly,
  so the E-DAIC mean-of-five values 0.7001 and 0.5884 are reproduced (section c, and `DISCREPANCIES.md` items 134, 135
  and 138).
- Projector fine-tunes saved no fold record. The fold file written on a pod by the fine-tune's own GroupKFold call
  for Pitt equals `folds/pitt_groupkfold5_pod.csv` on 468 of 468 clips.

One Pitt participant is listed under two speaker labels, one in each diagnosis group. The labels are kept exactly as
the runs used them, so in some splits this person sits in two folds. `folds/README.md` says where in every Pitt split.

---

## j. Repository layout

```
README.md                     this file
DISCREPANCIES.md              the audit log
MANIFEST.txt                  every file with its size, where it came from and what was dropped, plus the exclusion check
CITATION.cff                  citation metadata of the paper
docs/
  index.html                  project page, served by GitHub Pages from this folder
  explorer.html               interactive explorer over the released tables
  .nojekyll                   empty, tells GitHub Pages to serve the folder as it is
  assets/fig1_gap.png         Figure 1 at 400 dpi, rendered from figures/fig1_gap.pdf
  data/explorer_data.json     the aggregate numbers the explorer draws
archive/
  README_v1_companion.md      companion README of the first release, kept for the record. Its claims are superseded
                              and it sits outside docs/ so the project page does not serve it.
checks/                       checks made on 23 and 24 September 2026, before submission. Where a check reads the
                              paper, it reads the 23 Sep 23:23 UTC copy, not the submitted tex
  accel_24sep/                checks of 24 September: Audio Flamingo 3 on Pitt, the E-DAIC cells, Figure 1, font sizes
                              and page layout of the compiled paper, Holm tests, the three other models, Part 12 rows
                              and T5
  final_2325Z/                checks of the 23:23 copy: Audio Flamingo 3 input length, Figure 1 and the Part 12 value
                              check with its triage, each with an independent rerun (the _indep folders)
figures/
  fig1_gap.pdf                Figure 1 of the paper as submitted (E-DAIC panel on the whole interview, up to 15 min),
                              md5 a102c1a8872ba59fe73dd58700aae142. The script that drew it is not in this repository
  drafts/fig1_gap_overleaf_2325Z.pdf and .png  Figure 1 of the 23 Sep 23:23 UTC draft (E-DAIC panel on the first
                              300 s), PDF md5 bbefb559a5768dcb07d11ed5d440da18, made by scripts/fig_options.py (option C)
  fig1_gap_grayscale_check.png  grayscale render of the earlier one-axes Figure 1
  fig1_gap_300.pdf, fig1_gap_mid.pdf  two renders of the earlier one-axes figure, unused in the paper
  fig1_gap.png, probe_vs_answer.png, conflict_dumbbell.png  figures of the first release
  checks/                     checks that the option C redraw matches the Overleaf Figure 1 of 23 September
  fig1_9pt/                   a 9 pt version of the 23:23 draft Figure 1 at the column width, made by
                              scripts/make_fig1_9pt.py
  fig1_options/               Figure 1 layout options A to D
  part24/                     PART 24 whole-interview Figure 1, three-panel and one-axes (v4) layouts, made by
                              scripts/make_fig1_whole.py
  probe2/                     an earlier Figure 1 draft from the probe2 runs, made by scripts/probe2/make_fig1_v2.py
folds/                        GroupKFold(5) speaker fold assignments, 64 csv files and README.md, plus 7 csv files in
                              folds/part20/ (section i)
history/speech-health/        earlier project history from the development repository (merge cd8d564), kept for the
                              record and superseded. See history/README.md
lookup/
  master_lookup.csv           243 rows after the header: model, dataset, stream, AUC, n, estimator and source path.
                              Row numbers in this README are file line numbers, with the header as line 1, as GitHub
                              shows them. Lines 236 to 244 hold the PART 23 and PART 24 E-DAIC values the paper prints
  LABELS_DROPPED.tsv          every file whose clinical label was dropped, with the columns dropped (section b)
  SUPERSEDED.csv              source files replaced by the decisions of 23 September 2026
  bootstrap_cis.csv           speaker-bootstrap intervals, 2000 draws
  readout_direction.csv       Qwen2.5-Omni direction test on Pitt and E-DAIC, with readout_direction.json
  new_results_edaic_windows.csv, new_results_pitt_all708.csv  the E-DAIC window and Pitt all-708 reruns
  table1_all_models.csv, table1_omni25.csv  earlier Table 1 extracts, kept for reference
manifests/                    segment lists and window cut points, speech and clinical columns removed
  confounds/                  recording features per clip and the confound AUC table
  part1e/                     speaker crosswalks between Pitt, ADReSSo and ADReSS-2020
  part20/                     PART 20 clip lists, cut lists and hashes, one folder per pod, label column dropped
  part24/                     PART 24 whole-interview E-DAIC clip list and cut hashes, label column dropped
prompts/prompts.json          every prompt string
reports/                      text reports of the audit parts (E-DAIC windows, interviewer measure, Part 13, Part 14)
  part16/, part17/, part20/   run reports, logs and checks of PART 16, PART 17 and PART 20
  part23/, part24/            PART23_SUMMARY.md and PART24_REPORT.md, the notes of those two runs
  part25/                     PART 25 pod logs, run and check logs, and LABELS_DROPPED.tsv for the PART 25 files
  part26/                     PART 26 inventory of encoder states (INVENTORY.md) and the probe minus answer table
                              (PART26_TABLE.md)
  leftovers_23sep/            notes on the leftover checks of 23 September (LEFTOVERS_NOTES.md), the columns dropped
                              from their files (LABELS_DROPPED.tsv) and a drafted audit log entry
                              (DISCREPANCIES_entry_pod3b.txt)
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
  part23/                     PART 23 whole-interview E-DAIC: per-clip answers and probes (A/), truncation check (C/),
                              fine-tune (FT/), verify table (verify/) and per-layer curves
  part24/                     PART 24 whole-interview E-DAIC projector fine-tune: per-clip scores (FT/), pod outputs
                              per fold (pods/), Figure 1 plotted values (fig/), Part 12 triage (part12/) and checks
                              (verify/)
  part25/                     PART 25: probe minus answer gaps on the mean-of-five probe (A/), Holm correction (B/),
                              the other models' cells (C/), encoder layers above the answer (T5/) and checks (verify/)
  part26/                     PART 26 encoder probe cells of Qwen2-Audio, Qwen3-Omni and Kimi-Audio, one folder per
                              model and dataset, with PART26_TABLE.tsv, the inventory (inventory/) and the pod cost
                              (table26/)
  rater/                      agreement of two raters with the sentiment scorer on 80 excerpts. The v2 files are the
                              second rating round, the one the paper prints. The final files are the first round,
                              superseded
  leftovers_23sep/            the leftover checks of 23 September (reports/leftovers_23sep/LEFTOVERS_NOTES.md): M7 and
                              M11 (m7m11/), PART 16 twins (p16twins/), POD3 3b (pod3b/) and T5 refits (t5/), each with
                              a separate verifier (the _verify folders), the lookup table edits and summary rebuild
                              (tables/, tables_check/, tables_rebuild/) and the verified tables (verify/)
scripts/
  fig_options.py              Figure 1 of the 23:23 draft, E-DAIC on the first 300 s: python3 scripts/fig_options.py C out.pdf. It reads the E-DAIC
                              curves from scores/edaic_windows/ and the PC-GITA and Pitt curves from CURVES_DIR
                              (default curves/), which are not in this repository. The two whole-interview scripts
                              below read CURVES_DIR the same way
  make_fig1_v4.py             the earlier one-axes Figure 1 (make_fig1_v3.py is the version before it)
  make_fig1_whole.py          Figure 1 with the whole-interview E-DAIC curves, three-panel (overleaf) or one-axes (v4)
                              layout: python3 scripts/make_fig1_whole.py overleaf out.pdf. Same curves as the
                              submitted figure, without its Δ labels
  make_fig1_v4_whole.py       make_fig1_v4.py with the whole-interview E-DAIC curves and answer line 0.8366
  core/                       probing, per-layer curves, nested repeats, projector fine-tune, zero-shot and transcript scoring
  upstream/                   segment builders and the two conflict rules
  audit_1a1c/, part1e/, confounds/  scripts of the audit
  part14/, edaic_full/        the Part 14 pair builders and the E-DAIC interview-window builder
  part16/                     PART 16: M checks (M/), pods POD1 to POD4B and checks (verify/)
  part17/                     PART 17: T1 to T7b, the Part 12 preparation (part12_prep/) and checks (verify/)
  part20/                     PART 20: one folder per pod, POD1 to POD6, the interviewer-free Pitt cut
                              (pitt_noinv_cut/) and checks (verify/)
  part22/                     PART 22: processor configuration checks (step1/), pod run (pod/) and checks (verify/)
  part23/                     PART 23: A/, C/, FT/ and verify/, as in scores/part23/
  part24/                     PART 24: fine-tune (FT/, pod/), Figure 1 check (fig/), Part 12 (part12/), rater
                              agreement (rater/) and checks (verify/)
  part25/                     PART 25: A/, B/, C/, T5/ and verify/, as in scores/part25/
  part26/                     PART 26: one folder per model and dataset, the inventory check (inventory/), and the
                              table check and pod cost (table26/)
  leftovers_23sep/            the leftover checks of 23 September, one folder per check as in scores/leftovers_23sep/
  probe2/                     make_fig1_v2.py, the script of figures/probe2/
summary/                      summary tables of 23 September (summary_all_v2.csv, ALL_RESULTS_23sep.md) with their
                              build and check scripts. Superseded, see summary/README.md
```

Most per-clip files carry `clip` or `path`, `speaker`, `p_yes` and `mass` or `answer_mass` (the `label` column was
dropped, section b), and each run has a sidecar json with the checkpoint, prompt, n, AUC, dtype and date. Absolute file paths inside the lookup tables, the
sidecars and MANIFEST.txt point to the authors' machines and do not resolve in this repository. The build logs of this
release are not part of the repository.

---

## k. Status of every audit item

DISCREPANCIES.md numbers 176 items. The table gives one row per item. Fixed in the paper: 44. Released here: 128. Open: 4. Items 3, 5, 6, 7, 8, 13 and 14 describe the 14 September draft and are resolved in the submitted tex. Items 4 and 12 also describe that draft. The sentences that resolved them in the 23 Sep 23:23 UTC copy are not in the submitted tex, and their status against it is not yet decided. Quotes are verbatim from the submitted tex (md5 eacf6faf73dec5540d3c088e4aa6f542, PDF md5 708010aebf305980813c67f8e4f8e33e). The previous version of this table quoted the Overleaf copy of 23 September 2026, 23:23 UTC (md5 2923ced559329a4413b75efbaf477168). Rows 1, 4, 9, 12, 42, 65, 100, 112 and 123 still quote that copy: the submitted tex dropped or reworded the sentences they quote, and their status against the submitted tex is not yet decided.

| # | Finding | Status | Quoted sentence or file |
|---|---|---|---|
| 1 | Qwen2.5-Omni cannot take the full E-DAIC files, so the E-DAIC window is capped | fixed in the paper | "The model hears the participant's whole interview, interviewer turns removed and capped at 15 minutes for the 89 longest, with the processor's default input limit lifted so that every window is encoded whole." |
| 2 | The E-DAIC audio is a stitched concatenation of transcript rows | open | Decided by: a recut from the raw recording |
| 3 | The first 30 s of E-DAIC is setup chatter (14 September draft, resolved) | fixed in the paper | "On E-DAIC, the audio answer on the whole interview is within 0.02 of the answer from the whole transcript." / "The E-DAIC answer rises with the audio it hears: 0.66 on the first 30\,s, which is mostly session setup, 0.72 on a middle 30\,s, 0.81 on a middle five minutes and 0.84 on the whole interview, while the encoder probe of Section~\ref{sec:gap}, its layer chosen inside the folds, never exceeds 0.63 on any window." |
| 4 | Scored AD windows contain interviewer speech (14 September draft, resolved) | fixed in the paper | "Each Pitt segment starts at the participant's first timed utterance, and the ADReSSo and ADReSS-2020 windows start at the beginning of the recording, so interviewer prompts inside the window are kept." |
| 5 | The Pitt segments are not thirty seconds long (14 September draft, resolved) | fixed in the paper | "Segments run from 30 to 56\,s (median 33\,s) and the models hear the first 30\,s of each." / "468 segments from 227 participants" |
| 6 | Pitt, ADReSSo and ADReSS-2020 share most speakers (14 September draft, resolved) | fixed in the paper | "The three sets share speakers, so every result is within one set on speaker-disjoint folds and we make no transfer claim among them." |
| 7 | The set called ADReSSo is the ADReSS-M English training split (14 September draft, resolved) | fixed in the paper | "We use the DementiaBank Pitt cookie-theft descriptions \cite{pitt}, with the 468 segments from 227 participants whose fluency score (Conflict set construction below) falls in its top or bottom third, the ADReSSo recordings \cite{adresso} from 237 speakers, as released with the ADReSS-M challenge \cite{adressm}, and the ADReSS-2020 benchmark split \cite{adress2020} including 108 training and 48 test recordings." |
| 8 | NeuroVoz is channel confounded (14 September draft, resolved) | fixed in the paper | "Five recording features from the quietest frames (noise floor, SNR, spectral centroid, roll-off and tilt) separate the NeuroVoz groups at 0.80 and the PC-GITA groups at 0.67, so part of the Parkinson's probe may come from the recording channel." / "Normalising loudness and noise floor brings them to 0.56 on PC-GITA, where the encoder probe falls to 0.83 ($-$0.07 [$-$0.12, $-$0.02]) and the answer stays at 0.57, and only to 0.60 on NeuroVoz, where the probe falls to 0.89 ($-$0.03 [$-$0.061, $-$0.001])." |
| 9 | The E-DAIC binary label is not PHQ-8 at threshold 10 | fixed in the paper | "We use E-DAIC \cite{daic}, the extended dataset released for AVEC 2019, with 275 participants, where the binary label is the dataset's released label based on the eight-item Patient Health Questionnaire (PHQ-8) \cite{Kroenke2009}, stricter than PHQ-8 $\geq 10$ for 20 participants, rather than a clinical diagnosis." |
| 10 | PC-GITA and NeuroVoz have no 30 s window | fixed in the paper | "From the Parkinson's disease datasets, we only use read speech tasks, so the words carry little diagnostic information: PC-GITA \cite{pcgita} (Spanish, ten sentences and one read text per speaker, 1100 clips), NeuroVoz \cite{neurovoz} (Spanish, twelve isolated words, 1270 clips from 107 speakers) and MDVR-KCL \cite{kcl} (English, 37 speakers, most reading one of two passages)." |
| 11 | NeuroVoz is stored twice on disk | released here | manifests/part1e/fp.csv, scripts/part1e/fingerprint.py, scripts/part1e/dupes2.py |
| 12 | The Pitt 468 is the top and bottom thirds of a text fluency score (14 September draft, resolved) | fixed in the paper | "On all 708 Pitt segments, middle third included, the encoder probe reaches 0.80 and the zero-shot answer 0.67." |
| 13 | The E-DAIC valence lexicon dq_lex is not on disk (14 September draft, resolved) | fixed in the paper | "From the raw E-DAIC transcripts, with the interviewer removed, we take every participant segment of at least 25 words and 5\,s of speech and score it with the Twitter-RoBERTa sentiment classifier trained on TweetEval \cite{tweeteval}." |
| 14 | On E-DAIC the answer beats the probe once the model hears interview speech (14 September draft, resolved) | fixed in the paper | "The E-DAIC answer rises with the audio it hears: 0.66 on the first 30\,s, which is mostly session setup, 0.72 on a middle 30\,s, 0.81 on a middle five minutes and 0.84 on the whole interview, while the encoder probe of Section~\ref{sec:gap}, its layer chosen inside the folds, never exceeds 0.63 on any window." / "On E-DAIC, the language model probe reaches 0.74 and the encoder probe 0.60, both below the answer of 0.84 ($-$0.09 [$-$0.16, $-$0.03] and $-$0.24 [$-$0.32, $-$0.15]), while a probe on the answer state matches the answer (0.84)." |
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
| 26 | The E-DAIC agreement arm has 311 distinct segments in 483 rows | fixed in the paper | "E-DAIC \cite{daic}: 483 pairs within 138 speakers (311 distinct agreement segments)." |
| 27 | Audio Flamingo 2 answers Yes on almost every E-DAIC conflict clip | released here | reports/PART14_yes_rates.csv |
| 28 | Audio Flamingo 2 answers Yes on almost every E-DAIC agreement clip | released here | reports/PART14_yes_rates.csv |
| 29 | Qwen3-Omni Pitt zero shot 0.7673 on rerun against 0.7619 shipped | fixed in the paper | "On Qwen3-Omni the encoder probe reaches 0.90 on PC-GITA and 0.81 on Pitt against zero-shot answers of 0.60 and 0.76, so its Alzheimer's gap is only 0.05." |
| 30 | The Qwen3-Omni Pitt projector probe is a single repeat | released here | lookup/master_lookup.csv |
| 31 | The five-repeat file reproduces the Pitt encoder probe 0.7706 exactly | released here | lookup/master_lookup.csv, lookup/SUPERSEDED.csv, scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/README.md |
| 32 | The E-DAIC conflict selection changes under a second sentiment scorer | released here | scores/part16/M/M2_distilbert_sentiment.csv, scores/part16/M/M2_distilbert_sentiment.json, scores/part20/POD3b/sst2_pairs_o25_audio.csv, scores/part20/POD3b/sst2_pairs_o25_audio.json, scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv, reports/part20/rows/POD3b.tsv |
| 33 | The Part 14 builder scripts were only in a temporary folder | released here | scripts/part14/part14_select.py, scripts/part14/part14_sentiment.py |
| 34 | Part 14 clips are reused across pairs, so joins must be positional | fixed in the paper | "E-DAIC \cite{daic}: 483 pairs within 138 speakers (311 distinct agreement segments)." / "This yields 488 conflict segments, of which 483 have an agreement segment from the same speaker, giving 483 pairs from 138 speakers." |
| 35 | Two Qwen2.5-Omni Pitt transcript runs exist | released here | scores/part16/M/M1_pitt_text_arms.csv, scores/part17/T6_pitt_text_sep20_arms.csv, scores/part17/T6_pitt_text_sep20_arms.json, scores/part17/rows/T6.tsv, lookup/master_lookup.csv |
| 36 | The Sep 18 Pitt transcript values reproduce exactly | released here | scores/part17/T6_pitt_text_sep20_arms.csv, scores/part17/rows/T6.tsv |
| 37 | No saved fold file for Pitt in M5 | released here | folds/pitt_groupkfold5_mac.csv, folds/README.md |
| 38 | Audio Flamingo 2 has two Pitt score files that disagree | released here | lookup/SUPERSEDED.csv, lookup/master_lookup.csv, scores/part20/POD4c/work/af2_val10.csv (label column dropped) |
| 39 | Kimi-Audio has four Pitt copies with two score sets | released here | lookup/SUPERSEDED.csv, lookup/master_lookup.csv |
| 40 | Audio Flamingo 3 has three Pitt copies that differ slightly | fixed in the paper | "Audio Flamingo 3 \cite{af3}     & 0.42 & 0.84 & 0.61 & 0.59 \\" |
| 41 | Older 390-row E-DAIC conflict files are a different clip set | released here | manifests/edaic_paradox_manifest_390.csv, reports/PART14_table2_old_vs_new.csv |
| 42 | Only 4 of 12 conflict cells are below chance, all on E-DAIC | fixed in the paper | "Where the words contradict a depression diagnosis, four of six models fall below chance." / "On the depression segments four of the six models fall below chance when the words point away from the label, while every model except Audio Flamingo 2 scores above 0.80 when the words and the label agree." / "On the Pitt dataset the three Qwen models drop by 0.17 to 0.29 from the agreement to the conflict arm but none falls significantly below chance, Kimi-Audio shows no drop, and the two Audio Flamingo models score 0.53 to 0.61 on both arms (Table~\ref{tab:conflict})." |
| 43 | Mass weighting of the readout direction is a no-op on Qwen2-Audio | released here | scores/part16/M/M5_readout_direction_q2a.csv, scores/part17/T3_q2a_direction.csv, scores/part17/T3_q2a_direction_summary.csv, scores/part17/rows/T3.tsv |
| 44 | AUC along the readout direction and the saved p_yes differ in the 5th decimal | released here | scores/part16/M/M5_readout_direction_q2a.csv, scores/part17/T3_q2a_direction.csv, scores/part17/rows/T3.tsv |
| 45 | Apple Accelerate raises spurious floating point flags | released here | scripts/core/part10_readout_direction_v2.py |
| 46 | The first M7 pass used a superseded Pitt probe file | released here | lookup/master_lookup.csv, lookup/SUPERSEDED.csv, scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/README.md |
| 47 | Kimi-Audio has two disagreeing Pitt files | released here | lookup/SUPERSEDED.csv, lookup/master_lookup.csv |
| 48 | On Pitt no conflict cell is entirely below chance | fixed in the paper | "On the Pitt dataset, the three Qwen models drop by 0.17 to 0.29 from the agreement to the conflict arm, but none fall significantly below chance." |
| 49 | Two different KCL ASR transcript sets exist | fixed in the paper | "Pitt and ADReSS-2020 \cite{adress2020} come with manual CHAT transcripts and E-DAIC \cite{daic} with the challenge's automatic transcripts, used with the annotation codes stripped, and ADReSSo \cite{adresso} and the three Parkinson's sets are transcribed with whisper-large-v3 \cite{whisper}." |
| 50 | KCL text AUC differs by one tied pair | fixed in the paper | "MDVR-KCL    &   37 (37)  & 0.71 & 0.76 & 0.70 \\" |
| 51 | sft_full has no answer mass column | released here | scores/edaic_windows/sft_full_oof.csv |
| 52 | sft_mid30 has no answer mass column | released here | scores/edaic_windows/sft_mid30_oof.csv |
| 53 | sft_mid300 has no answer mass column | released here | scores/edaic_windows/sft_mid300_oof.csv |
| 54 | sft_pitt_all has no answer mass column | released here | scores/pitt_all/sft_pitt_all_oof.csv |
| 55 | af3_adress2020_rerun.csv has an empty mass column | released here | lookup/master_lookup.csv |
| 56 | af3_pitt468_rerun.csv has an empty mass column | released here | lookup/master_lookup.csv |
| 57 | No saved fold file for the Pitt LoRA baseline | released here | folds/pitt_groupkfold5_pod.csv, folds/README.md |
| 58 | Two speaker-string forms for the Pitt clips can change GroupKFold | released here | folds/README.md, folds/pitt_groupkfold5_pod_unpaddedids_seed0.csv to folds/pitt_groupkfold5_pod_unpaddedids_seed4.csv |
| 59 | The Pitt LoRA baseline already existed | released here | lookup/master_lookup.csv, scores/part16/POD2/lora_pitt_oof.csv, scores/part20/POD2/balanced_minus_earlier_lora_pitt_perclip.csv |
| 60 | Prior LoRA and projector per-clip files carry no answer mass | released here | scores/part20/POD2/balanced_lora_pitt_oof.csv, scores/part20/POD1/standard_rerun_ft_pitt_oof.csv |
| 61 | Independent recomputation agrees with the paper's Pitt values | released here | reports/part20/rows/POD5.tsv, scores/part17/rows/T5.tsv |
| 62 | No published cell has a median answer mass below 0.15 | released here | scores/part16/M/M9_answer_mass.csv |
| 63 | The answer mass column sits in different positions across files | released here | scores/edaic_windows/o25_full_zeroshot_scores.csv, scores/edaic_windows/af3_full.csv |
| 64 | The Pitt manifest's interviewer column is zero on every row | released here | scores/part16/POD4/cha_overlap_468.csv, manifests/pitt_conflict_manifest_468.csv |
| 65 | The E-DAIC released label is stricter than PHQ-8 >= 10 | fixed in the paper | "We use E-DAIC \cite{daic}, the extended dataset released for AVEC 2019, with 275 participants, where the binary label is the dataset's released label based on the eight-item Patient Health Questionnaire (PHQ-8) \cite{Kroenke2009}, stricter than PHQ-8 $\geq 10$ for 20 participants, rather than a clinical diagnosis." |
| 66 | Retracted claim that 0.7706 is in no file | released here | scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv |
| 67 | No saved fold file for the Pitt 468 in POD4 | released here | folds/pitt_groupkfold5_pod.csv, folds/pitt_groupkfold5_mac.csv, folds/README.md |
| 68 | The KCL WER fallback measures Whisper against itself | fixed in the paper | "ADReSSo \cite{adresso} and the three Parkinson's sets are transcribed with whisper-large-v3 \cite{whisper}." |
| 69 | The KCL 30 s window misses the read passage for some speakers | open | Decided by: a recut of those six windows |
| 70 | KCL speakers read two different passages | fixed in the paper | "From the Parkinson's disease datasets, we only use read speech tasks, so the words carry little diagnostic information: PC-GITA \cite{pcgita} (Spanish, ten sentences and one read text per speaker, 1100 clips), NeuroVoz \cite{neurovoz} (Spanish, twelve isolated words, 1270 clips from 107 speakers) and MDVR-KCL \cite{kcl} (English, 37 speakers, most reading one of two passages)." / "On PC-GITA and NeuroVoz, the transcript sits near chance, since every speaker produces the same words, while on MDVR-KCL, where speakers read one of two passages, it is above chance." |
| 71 | Two bootstrap implementations differ in the last digits | released here | scores/part16/M/M11_kcl_wer.json, scores/part16/M/M11_kcl_wer.csv |
| 72 | A rerun does not reproduce the Pitt zero shot bit for bit | released here | scores/part16/POD4/p16_pitt_orig_zeroshot_scores.csv, scores/part20/POD4/o25_orig_recording_rerun.csv, scores/part20/POD4/o25_orig_recording_rerun.sidecar.json, scores/part20/POD4/check/o25_check10_vs_omni_final.csv, reports/part20/rows/POD4.tsv |
| 73 | Qwen3-Omni Pitt zero-shot drift is a numeric environment effect | fixed in the paper | "On Qwen3-Omni the encoder probe reaches 0.90 on PC-GITA and 0.81 on Pitt against zero-shot answers of 0.60 and 0.76, so its Alzheimer's gap is only 0.05." |
| 74 | No saved fold file for POD3 Pitt and PC-GITA | released here | folds/pitt_groupkfold5_pod_unpaddedids_seed0.csv to folds/pitt_groupkfold5_pod_unpaddedids_seed4.csv, folds/pcgita_groupkfold5_pod_seed0.csv to folds/pcgita_groupkfold5_pod_seed4.csv, folds/README.md |
| 75 | A second job disputed the readout-direction standardisation | released here | lookup/readout_direction.csv, scores/part16/pod_sync/Q3O_direction_FIXED.json, lookup/SUPERSEDED.csv |
| 76 | auc_without_d was standardised with the training fold | released here | scores/part17/T3_q2a_direction.csv, scores/part16/pod_sync/Q3O_direction_FIXED.json, scripts/core/part10_readout_direction_v2.py |
| 77 | The M5 random floor generator | released here | scores/part17/T3_q2a_direction.csv, scores/part17/rows/T3.tsv |
| 78 | Two Pitt transcript runs disagree on the arm contrast | released here | scores/part16/M/M1_pitt_text_arms.csv, scores/part17/T6_pitt_text_sep20_arms.csv, scores/part17/T6_pitt_text_sep20_arms.json, scores/part17/rows/T6.tsv |
| 79 | Some transcript-only lookup AUCs do not reproduce at 4 dp | fixed in the paper | Table 1 rows: "PC-GITA     & 1100 (100) & 0.56 & 0.52 & 0.89 \\" / "NeuroVoz    & 1270 (107) & 0.70 & 0.54 & 0.90 \\" / "MDVR-KCL    &   37 (37)  & 0.71 & 0.76 & 0.70 \\" |
| 80 | Draft prose on answer mass was wrong | released here | scores/part16/M/M9_answer_mass.csv |
| 81 | A fresh LoRA run does not reproduce the older one clip for clip | released here | lookup/master_lookup.csv, scores/part20/POD2/balanced_minus_earlier_lora_pitt_perclip.csv, scores/part20/POD2/control/uniform_lora_pitt_control_oof.csv |
| 82 | Two bootstrap protocols for a single-arm interval | released here | scores/part17/T6_pitt_text_sep20_arms.json, reports/part20/rows/POD2.tsv |
| 83 | NeuroVoz channel normalisation fails the 0.60 bar | fixed in the paper | "Five recording features from the quietest frames (noise floor, SNR, spectral centroid, roll-off and tilt) separate the NeuroVoz groups at 0.80 and the PC-GITA groups at 0.67, so part of the Parkinson's probe may come from the recording channel." / "Normalising loudness and noise floor brings them to 0.56 on PC-GITA, where the encoder probe falls to 0.83 ($-$0.07 [$-$0.12, $-$0.02]) and the answer stays at 0.57, and only to 0.60 on NeuroVoz, where the probe falls to 0.89 ($-$0.03 [$-$0.061, $-$0.001])." |
| 84 | Interviewer presence in the Pitt windows is label correlated | fixed in the paper | "Interviewer speech sits in at least 158 of the 468 Pitt windows, more of it for dementia participants" |
| 85 | PC-GITA is only mildly channel confounded | fixed in the paper | "Normalising loudness and noise floor brings them to 0.56 on PC-GITA, where the encoder probe falls to 0.83 ($-$0.07 [$-$0.12, $-$0.02]) and the answer stays at 0.57, and only to 0.60 on NeuroVoz, where the probe falls to 0.89 ($-$0.03 [$-$0.061, $-$0.001])." |
| 86 | The same zero-shot drift appears on PC-GITA | fixed in the paper | "On Qwen3-Omni the encoder probe reaches 0.90 on PC-GITA and 0.81 on Pitt against zero-shot answers of 0.60 and 0.76, so its Alzheimer's gap is only 0.05." |
| 87 | Which PC-GITA manifest POD4 used | released here | folds/pcgita_groupkfold5_pod.csv, folds/README.md |
| 88 | A kernel swap alone reproduces the drift signature | released here | scores/part16/POD3/q3o_pitt_attn_backend.csv (label column dropped), scores/part16/POD3/q3o_pitt_attn_backend.json |
| 89 | sft_full saved no checkpoints and its training audio was not on disk | released here | scores/edaic_windows/sft_full_oof.csv, scores/edaic_windows/sft_full_sft.json, scripts/edaic_full/build_windows.py |
| 90 | Before and after channel normalisation numbers | fixed in the paper | "Normalising loudness and noise floor brings them to 0.56 on PC-GITA, where the encoder probe falls to 0.83 ($-$0.07 [$-$0.12, $-$0.02]) and the answer stays at 0.57, and only to 0.60 on NeuroVoz, where the probe falls to 0.89 ($-$0.03 [$-$0.061, $-$0.001])." |
| 91 | PC-GITA projector AUC differs in the 4th decimal | released here | scores/part16/POD3/q3o_pcgita_proj_nested_oof.csv (label column dropped) |
| 92 | The full-window E-DAIC probe had no per-clip file | released here | scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv, scores/part17/T7b/T7b_edaic300_meanof5.csv, scores/part17/rows/T7b.tsv, lookup/master_lookup.csv, scores/part16/M/MEDAIC_full_window.csv |
| 93 | ADReSSo released segmentation exists inside the archive | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json |
| 94 | The Mac disk was full, so POD3 states went to the external drive | released here | scores/part16/POD3/q3o_pitt_attn_backend.csv (label column dropped), scores/part16/POD3/q3o_pcgita_proj_nested_oof.csv (label column dropped) |
| 95 | ADReSSo segmentation covers only part of the scored clips | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json |
| 96 | Five ADReSSo segmentation files have no PAR tier | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json |
| 97 | ADReSS-2020 CHAT bullets run past the wav on some files | released here | scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json |
| 98 | No saved folds for ADReSS-2020 or ADReSSo | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json |
| 99 | The published ADReSSo probe is trained on more clips than arm A | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json |
| 100 | The E-DAIC window is cut at 300 s by the processor | fixed in the paper | "The model hears the participant's whole interview, interviewer turns removed and capped at 15 minutes for the 89 longest, with the processor's default input limit lifted so that every window is encoded whole." |
| 101 | The full-window projector per-layer file copies the LM embedding stage | released here | scores/edaic_windows/o25_full_proj_perlayer.csv, scores/edaic_windows/o25_full_llm_perlayer.csv, scores/part17/rows/T7b.tsv |
| 102 | The E-DAIC rebuild reproduces the published run | released here | scores/part16/EDAICFULL/full_zeroshot_scores.csv, scores/edaic_windows/o25_full_zeroshot_scores.csv, scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv, scores/part22/edaic_lifted_perclip.csv |
| 103 | A label-indexing bug in the first probe pass was fixed | released here | scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv, scores/part17/rows/T7b.tsv |
| 104 | Correction: 0.7706 does reproduce | released here | lookup/master_lookup.csv, lookup/SUPERSEDED.csv, scores/part10/pitt_enc_nested5_oof.npz, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/README.md |
| 105 | POD4 used the superseded Pitt encoder baseline | released here | scores/part20/POD5/POD5_nested_enc_mac_seed.csv, scores/part20/POD5/POD5_zs_o25_noinv.csv, reports/part20/rows/POD5.tsv |
| 106 | Two POD4 gap figures used rounded inputs | released here | scores/part20/POD4/o25_noinv_recording.csv, scores/part20/POD4/o25_orig_recording_rerun.csv |
| 107 | Three values circulate for the Qwen3-Omni by-arm probe | released here | lookup/SUPERSEDED.csv, lookup/master_lookup.csv, scores/part17/rows/T4.tsv |
| 108 | POD3's internal verifier compared cells with themselves | released here | scores/part17/rows/T4.tsv |
| 109 | B_text_conflict.json claims a de-duplication that did not happen | fixed in the paper | "E-DAIC \cite{daic}: 483 pairs within 138 speakers (311 distinct agreement segments)." |
| 110 | A LoRA plus projector arm exists | released here | lookup/master_lookup.csv, lookup/SUPERSEDED.csv |
| 111 | The 1d rebuild changes the label and drops the severity gates | fixed in the paper | "Replacing the severity gates of 15 and 4 with a PHQ-8 cutoff of 10 yields 924 pairs from 237 speakers, on which the conflict arm scores 0.18." |
| 112 | E-DAIC's released label is not the standard PHQ-8 cutoff | fixed in the paper | "We use E-DAIC \cite{daic}, the extended dataset released for AVEC 2019, with 275 participants, where the binary label is the dataset's released label based on the eight-item Patient Health Questionnaire (PHQ-8) \cite{Kroenke2009}, stricter than PHQ-8 $\geq 10$ for 20 participants, rather than a clinical diagnosis." |
| 113 | The ADReSSo segmentation exists (correction) | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json |
| 114 | POD4B quoted the wrong Pitt comparator | released here | scores/part16/POD4B/boot_adresso.json, scores/part16/POD4B/boot_adress2020.json, scores/part16/POD4/boot4b.json |
| 115 | window_seconds in the sidecars is hardcoded | released here | lookup/master_lookup.csv, scores/edaic_windows/sft_full_sft.json, scores/edaic_windows/sft_mid300_sft.json |
| 116 | The published ADReSSo probe is the wrong baseline for the 4b deltas | released here | scores/part16/POD4B/adresso_no_interviewer.csv, scores/part16/POD4B/adresso_no_interviewer.sidecar.json, scores/part16/POD4B/adress2020_no_interviewer.csv, scores/part16/POD4B/adress2020_no_interviewer.sidecar.json |
| 117 | POD4B arm A is not bit identical to the paper's cut | released here | scores/part16/POD4B/p16_adresso_orig_zeroshot_scores.csv, scores/part16/POD4B/p16_adress2020_orig_zeroshot_scores.csv |
| 118 | Every E-DAIC Table 2 agreement cell changes on 311 distinct segments | fixed in the paper | "E-DAIC \cite{daic}: 483 pairs within 138 speakers (311 distinct agreement segments)." |
| 119 | The Pitt states file moved on the G-Drive | released here | scores/part17/T3_q2a_direction.json |
| 120 | The first-pass random floor was reseeded after all | released here | scores/part17/T3_q2a_direction.csv, scores/part17/rows/T3.tsv |
| 121 | Final-stage and nested probe directions differ | released here | scores/part17/T3_q2a_direction_summary.csv, scores/part17/rows/T3.tsv |
| 122 | Direction weighting differs between the first pass and the verifier | released here | scores/part17/T3_q2a_direction.csv |
| 123 | One Pitt participant carries two speaker ids | fixed in the paper | "We use the DementiaBank Pitt cookie-theft descriptions \cite{pitt}, with 468 segments from 227 participants (one has sessions in both groups and is grouped under two labels), the top and bottom third of the fluency score defined under Conflict set construction below, the ADReSSo recordings \cite{adresso} from 237 speakers, as released with the ADReSS-M challenge \cite{adressm}, and the ADReSS-2020 benchmark split \cite{adress2020} including 108 training and 48 test recordings." |
| 124 | make_fig1_v5.py does not exist | released here | figures/drafts/fig1_gap_overleaf_2325Z.pdf, scripts/fig_options.py. The 23:23 draft figure is option C of fig_options.py. scripts/make_fig1_v4.py made the earlier one-axes figure. The script that drew the submitted figure is not in this repository |
| 125 | G-Drive paths moved, so recorded paths are stale | open | Decided by: rewriting the 75 stale source paths in lookup/master_lookup.csv that still use the old G-Drive prefix (paper1_local_runs/ at the drive root) |
| 126 | E-DAIC 300 s probe values used a different estimator from Pitt | fixed in the paper | "On E-DAIC, the language model probe reaches 0.74 and the encoder probe 0.60, both below the answer of 0.84 ($-$0.09 [$-$0.16, $-$0.03] and $-$0.24 [$-$0.32, $-$0.15]), while a probe on the answer state matches the answer (0.84)." |
| 127 | The lookup builder cannot reproduce the lookup | released here | lookup/master_lookup.csv |
| 128 | Qwen3-Omni Pitt rows come from two runs | released here | lookup/master_lookup.csv, lookup/SUPERSEDED.csv |
| 129 | Original-run edaic_full lookup rows still say capped at 900 s | released here | Rows 200, 202, 204 and 205 of lookup/master_lookup.csv (zero shot 0.8285, projector 0.5293, answer-state 0.7542 and projector fine tune 0.6421) carry the authors' notes: the first 300 s is what the model heard in those runs, and the fine-tune heard only the first 30 s (item 100). The submitted tex uses the runs with the processor cut lifted instead. Rows 236 to 244 hold them: the PART 23 zero-shot answer 0.8366 and the four mean-of-five probes, the PART 23 fine-tunes 0.7931 (whole window) and 0.8150 (first 300 s), and the PART 24 fine-tunes 0.8589 (seed 0, the Table 1 cell) and 0.8184 (seed 1). The other models' edaic_full rows are unchanged. Related files: lookup/master_lookup.csv. |
| 130 | T4 bookkeeping notes | released here | lookup/master_lookup.csv, lookup/SUPERSEDED.csv |
| 131 | The Pitt transcript by-arm sentence used the Sep 15 run | fixed in the paper | "Pitt        &  468 (227) & 0.66 & 0.66 & 0.77 \\" / "On the Pitt dataset, the transcript alone shows the same arm gap as the audio (0.56 on conflict, 0.71 on agreement)." |
| 132 | The Sep 15 and Sep 20 transcript runs are different scorers | released here | scores/part17/T6_pitt_text_sep20_arms.csv, scores/part17/T6_pitt_text_sep20_arms.json |
| 133 | Pitt speaker id forms | released here | scores/part17/T6_pitt_text_sep20_arms.json, folds/README.md |
| 134 | The E-DAIC mean-of-five probe values could not get intervals on the Mac | released here | scores/part17/T7b/T7b_edaic300_meanof5_perclip.csv, scores/part17/T7b/T7b_edaic300_meanof5.csv, scores/part17/rows/T7b.tsv, lookup/master_lookup.csv |
| 135 | Each dataset has a Mac split and a pod split | released here | folds/README.md, folds/pitt_groupkfold5_mac_seed0.csv to folds/pitt_groupkfold5_mac_seed4.csv, folds/pitt_groupkfold5_pod_seed0_UNVERIFIED.csv to folds/pitt_groupkfold5_pod_seed4_UNVERIFIED.csv, scores/part10/pitt_enc_nested5_oof.npz, scores/part17/verify/T2_verify_omni_pitt_repeats.json |
| 136 | README_probe2 misstates where some probe2 outputs were made | released here | folds/README.md |
| 137 | POD3 Pitt states use unpadded speaker ids | released here | folds/pitt_groupkfold5_pod_unpaddedids_seed0.csv to folds/pitt_groupkfold5_pod_unpaddedids_seed4.csv, folds/README.md |
| 138 | Pod refits on the Mac cannot reach 1e-6 | released here | folds/README.md |
| 139 | Some splits are inferred | released here | folds/README.md |
| 140 | The Qwen2.5-Omni transcript check does not match every check clip | released here | scores/part20/POD3c/val_o25_first25.csv (label column dropped), scores/part20/POD3c/val_o25_first25_run.json |
| 141 | text_score.py counts Yeah and Nope | released here | scores/part20/POD3c/p14_q3o_text.csv, reports/part20/rows/POD3c.tsv |
| 142 | The 10-window Qwen3-Omni check misses 1e-3 | released here | scores/part20/POD4b/q3o_val10_original_windows.csv (label column dropped), scores/part20/POD4b/q3o_val10_original_windows.check.json |
| 143 | Rule 3 adds YES and NO ids for Qwen3-Omni | released here | scores/part20/POD4b/q3o_pitt_noinv_zeroshot_scores.csv (label column dropped) |
| 144 | Correction to the POD4b 10-window counts | released here | scores/part20/POD4b/q3o_val10_original_windows.csv (label column dropped) |
| 145 | The Qwen2.5-Omni audio rescore matches 10 of 20 clips at 1e-3 | released here | scores/part20/POD3b/val20_cut_and_score_check.csv, scores/part20/POD3b/p14_rescore_o25_audio_envcheck.csv, scores/part20/POD3b/p14_rescore_o25_audio_envcheck.json |
| 146 | With DistilBERT arms the Omni conflict AUC is near chance | released here | scores/part20/POD3b/sst2_pairs_o25_audio.csv, scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv, reports/part20/rows/POD3b.tsv |
| 147 | AF3 on a pod reproduces the part10 Pitt copy | fixed in the paper | "Audio Flamingo 3 \cite{af3}     & 0.42 & 0.84 & 0.61 & 0.59 \\" |
| 148 | The language ordering holds only with the full E-DAIC transcript | released here | scores/part20/POD6/A_language_saliency/A_summary.json, scores/part20/POD6/A_language_saliency/A_tfidf_<dataset>_oof.csv, reports/part20/rows/POD6.tsv |
| 149 | On read speech the TF-IDF signal comes from ASR output | open | Decided by: that sentence saying so |
| 150 | Pitt transcript text source | fixed in the paper | "Pitt and ADReSS-2020 \cite{adress2020} come with manual CHAT transcripts and E-DAIC \cite{daic} with the challenge's automatic transcripts, used with the annotation codes stripped, and ADReSSo \cite{adresso} and the three Parkinson's sets are transcribed with whisper-large-v3 \cite{whisper}." |
| 151 | NeuroVoz metadata gaps limit the matched pairs | released here | scores/part20/POD6/B_neurovoz_matched/B_encprobe_nested5_nvmatched_oof.csv, scores/part20/POD6/B_neurovoz_matched/B_encprobe_nested5_nvmatched_oof.sidecar.json |
| 152 | The Kimi-Audio gate fails at 1e-3 but the AUC holds | released here | scores/part20/POD4d/kimi_val10_vs_canonical.csv, scores/part20/POD4d/kimi_val10_vs_canonical.json, scores/part20/POD4d/kimi_pitt_orig_podrerun_overall.csv, reports/part20/rows/POD4d.tsv |
| 153 | Removing the interviewer changes Kimi's agreement arm | released here | scores/part20/POD4d/kimi_pitt_noinv_overall.csv, reports/part20/rows/POD4d.tsv |
| 154 | AF2 ran in the environment that made the paper file | released here | scores/part20/POD4c/work/af2_val10.csv (label column dropped) |
| 155 | The 10-window Qwen2.5-Omni check misses 1e-3 on some windows | released here | scores/part20/POD4/check/o25_check10_vs_omni_final.csv, scores/part20/POD4/check/o25_check10_vs_omni_final.json, scores/part20/POD4/o25_orig_recording_rerun.csv |
| 156 | Qwen2-Audio dtype is fp16 in the paper pipeline | released here | scores/part20/POD4/q2a_orig_recording_rerun.csv, scores/part20/POD4/q2a_orig_recording_rerun_bf16.csv, scores/part20/POD4/q2a_noinv_recording_bf16.csv |
| 157 | Rule 3 adds YES and NO ids | released here | scores/part20/POD4/o25_noinv_recording.csv |
| 158 | The cut manifest's arm column is the cut variant | released here | manifests/pitt_conflict_manifest_468.csv, scores/part20/POD4/o25_noinv_recording.csv |
| 159 | Another job's setup ran on the POD4 pod | released here | scores/part20/POD4/ |
| 160 | All 468 original Pitt windows rescored on the same pod | released here | scores/part20/POD4b/P20_4b_q3o_orig_samepod_overall.perclip.csv, reports/part20/rows/POD4b.tsv |
| 161 | NeuroVoz zero-shot rescore against the paper file | released here | scores/part20/POD6/B_neurovoz_full_check/omni_nvfull_zeroshot_scores_samepod.csv (label column dropped), scores/part20/POD6/B_neurovoz_full_check/B_full_check_summary.json |
| 162 | The earlier LoRA values use the two-logit p_yes | released here | scores/part20/POD2/balanced_lora_pitt_oof.csv, reports/part20/rows/POD2.tsv |
| 163 | Rule 3 also covers YES and NO for the POD3 fine-tune | released here | scores/part20/POD3/p14_ft966_oof.csv |
| 164 | POD5 runs restarted | released here | reports/part20/POD5/sft_attempt1_racefail.log, reports/part20/POD5/sft_attempt2_cpu_throttled.log, reports/part20/POD5/nested_orig_attempt1_killed_cpu_quota.log |
| 165 | The part20 p_yes rule adds YES and NO | released here | scores/part20/POD5/POD5_zs_o25_noinv.csv |
| 166 | Pipeline checks on the POD5 pod | released here | scores/part20/POD5/POD5_direction_origrerun_check.csv, scores/part20/POD5/POD5_nested_enc_pod_seed.csv, reports/part20/rows/POD5.tsv, lookup/readout_direction.csv |
| 167 | A shared scratch folder | released here | scripts/part20/POD5/ and manifests/part20/POD5/ (label column dropped from pod_manifest_noinv.csv) |
| 168 | For AF2 the paper file and the cut cover different audio spans | released here | scores/part20/POD4c/af2_orig30_all.csv (label column dropped), scores/part20/POD4c/af2_diff_orig30_minus_noinv_all.csv (label column dropped), scores/part20/POD4c/af2_diff_origfull_minus_orig30_all.csv (label column dropped), reports/part20/rows/POD4c.tsv |
| 169 | Balanced LoRA drops pooled AUC on Pitt | released here | scores/part20/POD2/balanced_lora_pitt_oof.csv, scores/part20/POD2/control/uniform_lora_pitt_control_oof.csv, scores/part20/POD2/control/uniform_lora_pitt_control_oof.sidecar.json, reports/part20/rows/POD2.tsv |
| 170 | Balanced projector fine-tune fold 3 collapsed once | released here | scores/part20/POD1/balanced_ft_pitt_oof.csv, scores/part20/POD1/balanced_ft_pitt_fold3rerun_oof.csv, scores/part20/POD1/balanced_ft_pitt_r2_oof.csv (label column dropped), reports/part20/rows/POD1.tsv |
| 171 | The standard projector fine-tune reproduces overall but not by arm | fixed in the paper | "Its conflict arm stays at chance in every run (0.45 to 0.57, every interval including 0.5) while its agreement arm rises to 0.82 to 0.90, so the projector recovers the overall AUC through the arm that the words already help." |
| 172 | p_yes definition for the fine-tune comparison | released here | scores/part20/POD1/standard_rerun_ft_pitt_oof.csv, scores/part20/POD1/balanced_ft_pitt_oof.csv |
| 173 | Correction to the p_yes entry | released here | scores/part20/POD1/standard_rerun_ft_pitt_oof.csv, scores/part20/POD1/balanced_ft_pitt_oof.csv |
| 174 | AF2 falls below chance on the interviewer-free Pitt cut | released here | scores/part20/POD4c/af2_noinv_all.csv, scores/part20/POD4c/af2_diff_af2pitt468_minus_noinv_all.csv, reports/part20/rows/POD4c.tsv |
| 175 | The standard projector fine-tune collapses on some folds | fixed in the paper | "On the Pitt dataset the retrained projector reaches 0.77, the level of the answer-state probe of Section~\ref{sec:gap}, and 0.74 to 0.81 across three further seeds." |
| 176 | The Pitt projector fine-tune moves across seeds | released here | scores/part20/POD1b/std_ft_seed0_pitt_oof.csv to scores/part20/POD1b/std_ft_seed3_pitt_oof.csv (label column dropped), scores/part20/POD1b/std_ft_seed_range_pitt.json, reports/part20/rows/POD1b.tsv |
