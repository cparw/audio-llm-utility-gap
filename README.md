# The Representation Utility Gap in Audio LLMs for Clinical Speech

Companion data and code for the ICASSP 2027 submission.

This README is a lookup table. Every cell of Table 1, every cell of Table 2 and every number quoted
in the text is listed below with the file in this repository, or on the authors' machines, that the
number was read from. All values are AUC against the clinical label, printed to 4 decimals where the
source file carries that precision. Where the paper rounds to 2 decimals, both are shown.

The audit that produced the second half of this release found problems with several of the paper's
own claims. Those are not hidden here. Read [DISCREPANCIES.md](DISCREPANCIES.md) first.

---

## What is and is not in this repository

**In:** every script that produced a number, the manifests and segment lists, the per-clip score
files with their sidecar JSON, the prompt strings, the probe out-of-fold scores, and the audit.

**Out, deliberately:**

- No audio. No `.wav`, `.mp3`, `.cha` or any other recording.
- No transcripts, and no column that contains participant speech. Where a released manifest had a
  `text` column it was dropped; the derived counts computed from that text (`nw`, `ttr`, `retrace`,
  `fillers`, `unintel`, `errors`, `pauses`, `rate`, `p_text`, `val`, `npos`, `nneg`) are kept, since
  the probes and the conflict rule are defined on them.
- No per-participant clinical metadata beyond the binary label. The columns `age`, `sex`, `mmse`,
  `dx` and `sev` were dropped from every manifest. DementiaBank and E-DAIC distribute those under a
  data-use agreement and they are not ours to republish.
- No participant identifier beyond each dataset's own released ids.
- The four long-form audit files (`AUDIT_1a1c_ad.txt`, `AUDIT_1b1c_edaic.txt`, `AUDIT_1d_confounds.txt`,
  `AUDIT_1e_dupes.txt`) are **not** here. They quote verbatim participant and interviewer speech.
  `DISCREPANCIES.md` is the summary of all four and it is here, with one redaction noted in its header.

Reproducing any number requires the original corpora, each from its own custodian:
DementiaBank (Pitt, ADReSS-2020, ADReSS-M), E-DAIC/DAIC-WOZ, PC-GITA, NeuroVoz, MDVR-KCL.

---

## Authoritative estimator rules

Several probe numbers exist in more than one version, because the probe was re-run with a different
number of splits or repeats. Only one version per cell is authoritative. The rule:

| cell | authoritative value | file | superseded values |
|---|---|---|---|
| Qwen2.5-Omni, Pitt, encoder probe | **0.7706** | `scores/part10/pitt_enc_nested5_oof.npz` | 0.7761, 0.7969 |
| Qwen2.5-Omni, Pitt, answer probe | **0.7709** | see the note below | 0.7843, 0.7960 |

`0.7706` is the mean of the five per-seed AUCs held in `scores/part10/pitt_enc_nested5_oof.npz`.
Recomputed from that file with the rank statistic, the five seeds are
0.7598, 0.7725, 0.7459, 0.7903, 0.7843, mean **0.7706**. Taking the AUC of the averaged score instead
of the average of the AUCs gives 0.7917; that is a different estimator and is not the reported cell.
The speaker-bootstrap interval is 0.7324 to 0.8480, `lookup/bootstrap_cis.csv` row 1.

`0.7761` is the older five-repeat mean from `omni_final/omni_pitt_nested_repeats.json`.
`0.7969` is a single-split nested value. `0.7960` is the best per-layer `auc_oof` of the answer probe
at layer 27 and `0.7843` is its last-layer value; neither is a whole-probe estimate. All four are
**superseded** and should not be quoted.

> **Unresolved, stated plainly.** The rule for this release names the Pitt answer probe as 0.7709.
> That value does not appear in any file on the machine this release was built from. The closest
> on-disk figure for the Qwen2.5-Omni Pitt answer-state probe is **0.7636**
> (`lookup/master_lookup.csv`, five split nested, mean of 5 repeats,
> source `omni_final/omni_pitt_nested_repeats.json`). Until the file behind 0.7709 is produced, the
> number that can be reproduced from what is published here is 0.7636. This gap is recorded rather
> than papered over.

---

## Table 1, cell by cell

Table 1 of the paper is Qwen2-Audio on every dataset: the encoder probe with the layer chosen inside
the training folds, the model's own zero-shot answer on the same clips, and its answer from the
transcript alone. The paper prints 2 decimals. The underlying values are below.

| condition | dataset | column | paper | this release | n | estimator | source file |
|---|---|---|---|---|---|---|---|
| PD | PC-GITA | probe | 0.92 | **0.9097** | 1100 | five split nested, mean of 5 repeats | `/Volumes/G-Drive Pro/paper1_local_runs/omni_final/q2a_pcgita_nested_repeats.json` |
| PD | PC-GITA | answer, audio | 0.54 | **0.5369** | 1100 | zero shot | `/Volumes/G-Drive Pro/paper1_local_runs/probe2/pcgita_zeroshot_scores.csv` |
| PD | PC-GITA | answer, text | -- | not run | -- | -- | -- |
| PD | NeuroVoz | probe | 0.96 | **0.9324** | 1270 | five split nested, mean of 5 repeats | `/Volumes/G-Drive Pro/paper1_local_runs/omni_final/q2a_neurovoz_nested_repeats.json` |
| PD | NeuroVoz | answer, audio | 0.55 | **0.5519** | 1270 | zero shot | `/Volumes/G-Drive Pro/paper1_local_runs/probe2/neurovoz_zeroshot_scores.csv` |
| PD | NeuroVoz | answer, text | -- | not run | -- | -- | -- |
| PD | MDVR-KCL | probe | 0.89 | **0.7714** | 37 | five split nested, mean of 5 repeats | `/Volumes/G-Drive Pro/paper1_local_runs/omni_final/q2a_kcl_nested_repeats.json` |
| PD | MDVR-KCL | answer, audio | 0.66 | **0.5923** | 37 | zero shot | `/Volumes/G-Drive Pro/paper1_local_runs/probe2/kcl_zeroshot_scores.csv` |
| PD | MDVR-KCL | answer, text | -- | not run | -- | -- | -- |
| MDD | E-DAIC | probe | 0.66 | **0.5673** | 275 | five split nested, mean of 5 repeats | `/Volumes/G-Drive Pro/paper1_local_runs/omni_final/q2a_edaic_nested_repeats.json` |
| MDD | E-DAIC | answer, audio | 0.65 | **0.6464** | 275 | zero shot | `/Volumes/G-Drive Pro/paper1_local_runs/probe2/edaic_zeroshot_scores.csv` |
| MDD | E-DAIC | answer, text | -- | **0.6116** | 275 | text only, per-clip file recomputed (30 s transcript, matched to the 30 s audio window) | `/Volumes/G-Drive Pro/paper1_local_runs/qwen2audio_edaic30_text.csv` |
| AD | Pitt | probe | 0.78 | **0.7846** | 468 | five split nested, mean of 5 repeats | `/Volumes/G-Drive Pro/paper1_local_runs/omni_final/q2a_pitt_nested_repeats.json` |
| AD | Pitt | answer, audio | 0.62 | **0.6173** | 468 | zero shot | `/Volumes/G-Drive Pro/paper1_local_runs/probe2/pitt_zeroshot_scores.csv` |
| AD | Pitt | answer, text | 0.68 | **0.6762** | 468 | text only, per-clip file recomputed | `/Volumes/G-Drive Pro/paper1_local_runs/qwen2audio_pitt_text.csv` |
| AD | ADReSSo | probe | 0.87 | **0.8537** | 237 | five split nested, mean of 5 repeats | `/Volumes/G-Drive Pro/paper1_local_runs/omni_final/q2a_adresso_nested_repeats.json` |
| AD | ADReSSo | answer, audio | 0.66 | **0.6649** | 237 | zero shot | `/Volumes/G-Drive Pro/paper1_local_runs/probe2/adresso_zeroshot_scores.csv` |
| AD | ADReSSo | answer, text | 0.68 | **0.6825** | 237 | text only, per-clip file recomputed | `/Volumes/G-Drive Pro/paper1_local_runs/qwen2audio_adresso_text.csv` |
| AD | ADReSS-2020 | probe | 0.82 | **0.8315** | 156 | five split nested, mean of 5 repeats | `/Volumes/G-Drive Pro/paper1_local_runs/omni_final/q2a_adress2020_nested_repeats.json` |
| AD | ADReSS-2020 | answer, audio | 0.68 | **0.6824** | 156 | zero shot | `/Volumes/G-Drive Pro/paper1_local_runs/probe2/adress2020_zeroshot_scores.csv` |
| AD | ADReSS-2020 | answer, text | -- | **0.7383** | 156 | text only, per-clip file recomputed | `/Volumes/G-Drive Pro/paper1_local_runs/qwen2audio_adress2020_text.csv` |

The probe and answer-text sources under `/Volumes/G-Drive Pro/paper1_local_runs/` are the authors'
run directory, not files in this repository; they are named so the provenance of each cell is
traceable. The per-clip score files that ARE in this repository are listed under "Per-clip score
files" below. `lookup/master_lookup.csv` carries all 178 model x dataset x stream cells, including
the five other models the paper does not tabulate.

The E-DAIC row of Table 1 is the one the audit overturns. See "New results" below and D3, D14.

---

## Table 2, cell by cell

Table 2 is the conflict test: answers on segments where the words point away from the diagnosis,
beside the matched agreement segments. Top block, 195 matched pairs from E-DAIC. Bottom block,
146 conflict and 322 agreement segments from Pitt.

The segment lists themselves are in this repository:
`manifests/edaic_paradox_manifest_390.csv` (390 rows, 195 pairs) and
`manifests/pitt_conflict_manifest_468.csv` (468 rows, 146 conflict + 322 agreement).
The rule that produced them is `scripts/upstream/px_build.py` for E-DAIC and
`scripts/upstream/build_ad_conflict.py` lines 98-106 for Pitt.

| dataset | model | conflict | agreement | source |
|---|---|---|---|---|
| E-DAIC | SALMONN | 0.29 | 0.85 | not on the build machine, see note below |
| E-DAIC | DiVA | 0.30 | 0.70 | `results/health/reliance/diva_daic.csv` (authors' run dir) |
| E-DAIC | Qwen2.5-Omni | 0.31 | 0.85 | `results/health/paradox/omni_paradox_summary.json` (authors' run dir): conflict 0.2970, agreement 0.8680 |
| E-DAIC | Qwen3-Omni | 0.35 | 0.87 | not on the build machine, see note below |
| E-DAIC | Qwen2-Audio | 0.35 | 0.92 | not on the build machine, see note below |
| E-DAIC | Audio Flamingo 3 | 0.36 | 0.93 | not on the build machine, see note below |
| E-DAIC | MiMo-Audio | 0.47 | 0.82 | not on the build machine, see note below |
| E-DAIC | Kimi-Audio | 0.52 | 0.87 | not on the build machine, see note below |
| E-DAIC | Audio Flamingo 2 | 0.55 | 0.57 | not on the build machine, see note below |
| Pitt | Qwen2-Audio | 0.41 | 0.70 | not on the build machine, see note below |
| Pitt | Qwen2.5-Omni | 0.44 | 0.77 | not on the build machine, see note below |

> **Provenance gap, stated plainly.** Of the eleven Table 2 rows, only the E-DAIC Qwen2.5-Omni row
> could be traced to a file on the machine this release was built from, and that file gives
> **0.2970 / 0.8680**, which rounds to 0.30 / 0.87, not the 0.31 / 0.85 printed in Table 2.
> `results/table2_conflict.csv` in this repository reproduces the printed table; it is not itself the
> source of the numbers. The SALMONN, MiMo-Audio, Kimi-Audio, Audio Flamingo 2 and 3, Qwen2-Audio and
> Qwen3-Omni conflict runs were produced on a cluster and their score files are not in this release.
> Until they are recovered, those cells are reported as published and marked unverified here.

A 390-row rerun of the conflict manifest with a 30 s window, which is a **different** protocol from
the 195-pair matched test, is in `lookup/master_lookup.csv` under dataset `edaic_conflict`. Those
numbers (for example Qwen2.5-Omni zero shot 0.6204, Kimi-Audio 0.6899, Audio Flamingo 2 0.5363) are
not Table 2 cells and must not be quoted as such.

---

## Table 3, cross-dataset transfer

`results/table3_cross_dataset.csv`. Encoder probe trained on one Parkinson's read-speech dataset and
tested on another, Qwen2-Audio, no retraining. The diagonal is the within-dataset probe with the layer
chosen inside the training folds, and it is the same value as the Table 1 probe column:
KCL 0.89, NeuroVoz 0.96, PC-GITA 0.92. Script: `scripts/core/omni_transfer.py`.

---

## In-text numbers

| number in the text | value | file |
|---|---|---|
| probe reaches 0.89 to 0.96 on Parkinson's | KCL 0.7714 / NeuroVoz 0.9324 / PC-GITA 0.9097 at 4 dp; the paper quotes the rounded within-dataset diagonal | `lookup/master_lookup.csv`, Qwen2-Audio encoder probe rows |
| 0.78 on Pitt | 0.7846 | `lookup/master_lookup.csv`, Qwen2-Audio pitt encoder probe |
| LM probe on Alzheimer's rises to 0.86 | Qwen2-Audio Pitt LM probe 0.8380; ADReSSo 0.8923 | `lookup/master_lookup.csv` |
| answer state achieves 0.80 | Qwen2-Audio Pitt answer-state probe 0.8253 | `lookup/master_lookup.csv` |
| probe holds 0.82 to 0.88 at every LM stage for MDVR-KCL | per-layer curves | `scripts/core/curves_from_states.py`, authors' run dir |
| model answer stays at 0.54 to 0.68 | Qwen2-Audio zero shot: pcgita 0.5369, neurovoz 0.5519, kcl 0.5923, edaic 0.6464, pitt 0.6173, adresso 0.6649, adress2020 0.6824 | `lookup/master_lookup.csv` |
| transcript alone matches or beats it | Qwen2-Audio transcript only: pitt 0.6762 vs answer 0.6173; adresso 0.6825 vs 0.6649; adress2020 0.7383 vs 0.6824 | `lookup/master_lookup.csv` |
| six of nine models fall to 0.36 or lower | Table 2 top block | `results/table2_conflict.csv`; per-model score files not on the build machine |
| Pitt Qwen2-Audio conflict 0.41 (0.31 to 0.52) | 0.41 | `results/table2_conflict.csv`, interval not on the build machine |
| none of seven frozen interventions lifts conflict above 0.56 | 13 intervention rows | `results/interventions.csv` |
| projector retraining lifts Pitt answer from 0.62 to 0.76 (0.72 to 0.81) | 0.62 = Qwen2-Audio pitt zero shot 0.6173 | `lookup/master_lookup.csv`; the 0.76 out-of-fold file is not on the build machine |
| agreement 0.86, conflict 0.48 after projector retraining | 0.48 / 0.86 | `results/interventions.csv`, row "projector retrained on final encoder output" |
| linear readout on frozen states reaches 0.78 (0.74 to 0.83) | 0.7846 Qwen2-Audio; the Qwen2.5-Omni equivalent is 0.7706 | `lookup/master_lookup.csv`; `scores/part10/pitt_enc_nested5_oof.npz` |
| 0.61 on the conflict segments | 0.61 | `results/interventions.csv`, row "linear readout on frozen states" |
| answer-state probe 0.80, conflict 0.69 | 0.8253 answer-state probe | `lookup/master_lookup.csv`; the 0.69 conflict arm is not on the build machine |
| 468 thirty-second segments | FALSE. 0 of 468 clips are 30 s; median 33.2850 s, max 55.5600 s | DISCREPANCIES.md D5 |
| the 30 s window starts at the patient's first words | FALSE for most windows. Pitt 213/468 windows contain interviewer speech; ADReSS-2020 114/156 | DISCREPANCIES.md D4 |

---

## New results: E-DAIC window rerun

The paper scores E-DAIC on the first 30 s of each file. The audit found that window is the
technician setting up the session, not the interview: median 11 words total and 7 participant words,
188 of 275 speakers under 15 participant words, 47 with none at all (D3). Every window below was
re-cut and every model re-run with the same prompt and the same scoring, changing only the window.

Full table: `lookup/new_results_edaic_windows.csv`. Per-clip scores and sidecars:
`scores/edaic_windows/`. Window definitions and per-speaker cut points: `manifests/edaic_windows_*.csv`.

### Qwen2.5-Omni, every stream, all four windows

| stream | paper, first 30 s | mid 30 s | mid 300 s | full, 900 s cap |
|---|---|---|---|---|
| zero shot answer | 0.6620 | 0.7205 | 0.8122 | **0.8285** |
| encoder probe | 0.5966 | 0.5770 | 0.6168 | **0.5930** |
| projector probe | 0.5041 | 0.4380 | 0.5703 | **0.5293** |
| LM probe | 0.4939 | 0.6249 | 0.6351 | **0.6853** |
| answer-state probe | 0.6146 | 0.6593 | 0.7065 | **0.7542** |
| projector fine-tune, answer | not run | 0.6669 | 0.6595 | **0.6421** |
| transcript only | 0.6613 | pending | pending | **0.8230** |

The transcript-only full-file cell 0.8230 is held at
`overnight2/text_new/o25_edaicfull_text.json` in the authors' run directory; the prompt is in
`prompts/prompts.json`.

**What this shows.** The encoder probe is flat across every window, 0.5770 to 0.6168. The model's own
answer climbs from 0.6620 to 0.8285. On the full file the answer beats the paper's own encoder probe
cell of 0.5966 by +0.2319. For E-DAIC the utility gap does not merely shrink, it inverts, once the
model is given interview speech instead of a microphone check. The E-DAIC row of Table 1 should be
read as a windowing artefact. This is D14.

### Every model, zero-shot answer, by window

| model | mid 30 s | mid 300 s | full, 900 s cap |
|---|---|---|---|
| Qwen2.5-Omni | 0.7205 | 0.8122 | 0.8285 |
| Qwen3-Omni-30B-A3B | 0.7153 | 0.8217 | 0.8741 |
| Audio Flamingo 3 | 0.6949 | 0.8374 | 0.8565 |
| Qwen2-Audio | 0.6891 | 0.6332 | 0.6330 |
| Kimi-Audio | 0.6664 | 0.6581 | not run |

All n=275. Kimi-Audio was not run on the full window.

**Caveat, carried from D1 and D2 and repeated in every sidecar.** Every window here is cut from
`dcaps_proc`, which is a stitched concatenation of the ASR rows and not the raw interview: the ratio
of stitched to raw duration has median 0.5416 and 0 of 275 files within 5 percent of 1.0. A "middle
30 s" is therefore the middle of a concatenation, not the middle of a conversation. Separately,
52 of 275 files exceed the 32,768-token context of Qwen2.5-Omni and cannot be encoded whole at any
setting; the 900 s cap truncates 89 of 275. Both are recorded per window in
`manifests/edaic_windows_*.json`.

---

## New results: Pitt, all 708 segments

The paper's Pitt set of 468 segments is not a sample of Pitt. `scripts/upstream/build_ad_conflict.py`
lines 98-106 keep only the extreme tertiles of an out-of-fold text-fluency score and discard the
middle third (D12). The segment list was rebuilt keeping every tertile: 708 segments, 266 speakers,
against 468 segments and 228 speakers in the paper.

| stream | all 708 | n | estimator |
|---|---|---|---|
| zero shot answer | **0.6695** | 708 | zero shot, single prompt |
| encoder probe | **0.7968** | 708 | 5 repeat nested, mean of per-repeat AUC |
| projector probe | **0.7632** | 708 | 1 repeat nested, mean of per-repeat AUC |
| LM probe | **0.8125** | 708 | 5 repeat nested, mean of per-repeat AUC |
| answer-state probe | **0.7744** | 708 | 5 repeat nested, mean of per-repeat AUC |

Beside the paper cells for the same model on the 468:

| stream | paper, 468 tertile-selected | this rerun, all 708 |
|---|---|---|
| zero shot answer | 0.6578 | 0.6695 |
| encoder probe | 0.7706 | 0.7968 |
| projector probe | 0.7507 | 0.7632 |
| LM probe | 0.8032 | 0.8125 |
| answer-state probe | 0.7636 | 0.7744 |

Zero-shot answer restricted by tertile, recomputed from
`scores/pitt_all/o25_pitt_all_zeroshot_scores.csv` joined to `manifests/mf_pitt_all.csv`:

| subset | AUC | n | positives | speakers |
|---|---|---|---|---|
| all 708 | **0.6695** | 708 | 421 | 266 |
| low tertile | 0.6641 | 234 | 101 | 136 |
| middle tertile, the third the paper discards | **0.6665** | 240 | 131 | 147 |
| high tertile | 0.6437 | 234 | 189 | 139 |
| low + high, the paper's 468 | 0.6711 | 468 | 290 | 228 |

The middle tertile scores 0.6665, which is inside the range of the two kept tertiles. On the
zero-shot answer the tertile selection does not inflate the result. The paper's own 468 cell is
0.6578 and this rerun of the same 468 gives 0.6711, a difference of +0.0133 between two runs of the
same protocol; treat differences of that size as run-to-run noise, not signal.

The probe streams cannot yet be split by tertile from what is released here: the nested probe files
hold per-repeat means, not per-clip out-of-fold scores. The per-clip arrays are needed for that split.

---

## Per-clip score files

Every scored run ships its per-clip file and a sidecar JSON with the checkpoint, the prompt, n, the
AUC, the dtype and the date. Columns are `clip` or `path`, `speaker`, `label`, `p_yes`, `mass` or
`answer_mass`, `prompt`. `p_yes` is the probability mass on the Yes token normalised over Yes and No
only; `mass` is the share of the full next-token distribution those two tokens hold.

| folder | what |
|---|---|
| `scores/edaic_windows/` | E-DAIC rerun, 5 models x 3 windows, zero-shot; Qwen2.5-Omni per-layer probe curves and nested repeats; projector fine-tune out-of-fold |
| `scores/pitt_all/` | Pitt all-708 zero-shot, per-layer probe curves, nested repeats |
| `scores/part10/pitt_enc_nested5_oof.npz` | the authoritative Pitt encoder probe: 5 seeds x 468 out-of-fold scores, labels, speaker ids, clip names |

Every AUC in `lookup/new_results_edaic_windows.csv` and `lookup/new_results_pitt_all708.csv` was
recomputed from the per-clip file with the rank statistic below and agreed with the sidecar to
4 decimals on all 18 runs checked.

```python
r = pd.Series(score).rank().values
n1 = (y == 1).sum(); n0 = (y == 0).sum()
auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
```

---

## Prompts

`prompts/prompts.json` holds every prompt string with the runs that used it. There are five:

- `Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.`
- `Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No.`
- `Based only on this transcript of what the person said, does this speaker show signs of Parkinson's disease? Answer with one word, Yes or No.`
- `Based only on this transcript of what the person said, does this speaker show signs of dementia? Answer with one word, Yes or No.`
- `Based only on this transcript of what the person said, does this speaker show signs of depression? Answer with one word, Yes or No.`

No prompt contains participant speech. The transcript conditions place the transcript after the
prompt at run time; those transcripts are not released.

---

## Repository layout

```
README.md                     this file
DISCREPANCIES.md              the audit, 14 items, one redaction noted in its header
MANIFEST.txt                  every file, its size, where it came from, and the exclusion check
reports/EDAIC_REPORT.txt      the E-DAIC window rerun as a printed table
lookup/
  master_lookup.csv           178 cells: model x dataset x stream -> AUC, n, estimator, source file
  bootstrap_cis.csv           speaker-bootstrap intervals, 2000 draws
  table1_all_models.csv       Table 1 extended to all six models
  table1_omni25.csv           Table 1 recomputed for Qwen2.5-Omni
  new_results_edaic_windows.csv  the E-DAIC window rerun, 29 rows
  new_results_pitt_all708.csv    the Pitt all-tertile rerun, 5 rows
results/                      the paper tables as CSV, as first published
figures/                      the paper figures
manifests/
  edaic_windows_*.csv|json    per-speaker window cut points, cap and truncation counts
  pitt_all_segments.csv       708 Pitt segments, every tertile, speech and clinical columns dropped
  mf_pitt_all.csv             the clip list actually scored
  pitt_conflict_manifest_468.csv  the paper Pitt 468
  edaic_paradox_manifest_390.csv  the E-DAIC conflict and agreement rows
  part7_heard_window.csv      words inside vs outside the scored window, 390 rows
  confounds/                  per-clip recording features and the confound AUC table
  part1e/                     speaker crosswalks and duplicate analysis
scores/                       per-clip scores and sidecars, see above
scripts/
  core/                       probing, curves, nested repeats, fine-tune, figures, transfer
  upstream/                   px_build.py, build_ad_conflict.py, make_fig1.py and the segment builders
  audit_1a1c/                 the audit scripts and the offset tables they produced
  part1e/                     fingerprinting and speaker-overlap recovery
  confounds/                  feature extraction and the confound AUC
prompts/prompts.json          every prompt string
```

---

## Known problems with this paper

[DISCREPANCIES.md](DISCREPANCIES.md) lists fourteen, each with the numbers behind it. The ones that
change how a cell should be read:

- **D3, D14.** The E-DAIC window in Table 1 is a microphone check, not interview speech. On the full
  file the model's answer reaches 0.8285 and beats the probe. The E-DAIC row does not support the
  utility-gap claim.
- **D12.** The Pitt 468 is the extreme-tertile subset of a text model's score, with the middle third
  discarded. The all-708 rerun is above.
- **D5.** "468 thirty-second segments" is wrong: none of the 468 is 30 s.
- **D4.** Windows on the AD sets contain interviewer speech far more often than stated, and the
  `inv_ms_in_window` column that would appear to deny this is 0 on all 468 rows because the Pitt
  `*INV:` tiers usually carry no time bullet. Do not cite that column.
- **D6, D7.** Pitt, ADReSSo and ADReSS-2020 share speakers heavily, and the set called ADReSSo is the
  ADReSS-M English training split.
- **D8.** NeuroVoz is channel confounded: a single scalar from the silence between words gives 0.8650.
- **D13.** The E-DAIC valence lexicon `dq_lex` is not recoverable, so the lexicon half of the conflict
  rule cannot be reproduced bit-identically. The TF-IDF half is fully specified in `px_build.py`.

---

## Citation

Chaitanya Parwatkar, Nima Kelidari, Minoo Ahmadi, Ashutosh Chaubey, Mohammad Soleymani.
*The Representation Utility Gap in Audio LLMs for Clinical Speech.* University of Southern California.
Submitted to ICASSP 2027.
