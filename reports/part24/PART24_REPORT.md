# PART 24 notes: whole interview fine tune, Figure 1, rater sheet and final check

Written 2026-09-23 about 23:10 UTC. Paths are repository paths. A path that starts with `<local data dir>` is a
folder on the authors' machines and is not in this repository.

No file under any omni_final folder was written. The paper tex was only read.

## 1. Headline values (verified)

All values below were recomputed by the independent check (`scripts/part24/verify/verify_ft24_indep.py`, output
`scores/part24/verify/FT24_VERIFIED.tsv`). It checked 41 of 41 rows and every one says yes, with the largest
difference 0.0 at full float precision. AUC is sklearn roc_auc_score. Intervals use 2000 speaker draws with a fresh
default_rng(0) per cell and 2.5/97.5 percentiles. E-DAIC has one clip per speaker, and every cell had 2000 usable
draws.

Qwen2.5-Omni projector fine tune, E-DAIC whole interview, 275 clips (275 speakers, 66 positive), seed 0, the
reported result:

```
E-DAIC whole interview projector FT, pooled out of fold AUC, seed 0     0.8589 [0.8050, 0.9029]   n=275   VERIFIED
FT minus whole interview zero shot, paired, seed 0                      +0.0223 [-0.0064, 0.0547]  n=275   VERIFIED
Whole interview zero shot reference (PART23 per clip file)              0.8366 [0.7786, 0.8866]   n=275   VERIFIED
Fold 0 AUC, seed 0                                                      0.9167 [0.8267, 0.9783]   n=55    VERIFIED
Fold 1 AUC, seed 0                                                      0.8625 [0.7334, 0.9613]   n=55    VERIFIED
Fold 2 AUC, seed 0                                                      0.8417 [0.6955, 0.9556]   n=55    VERIFIED
Fold 3 AUC, seed 0                                                      0.8372 [0.6955, 0.9464]   n=55    VERIFIED
Fold 4 AUC, seed 0                                                      0.8585 [0.7112, 0.9671]   n=55    VERIFIED
```

The paired interval includes 0, so the fine tune does not differ from the whole interview zero shot.

At 2 dp these are 0.86 for the fine tune, 0.84 for the zero shot and +0.02 for the gap.

Extra, not the reported number. Seed 1 on all five folds came from a second set of pods that was launched
pre-emptively (see `reports/part24/SEED1_PREEMPTIVE.txt`). The seed rule did not ask for these runs.

```
EXTRA seed 1 all folds, pooled out of fold AUC                          0.8184 [0.7496, 0.8758]   n=275   VERIFIED
EXTRA seed 1 all folds, FT minus whole interview zero shot, paired     -0.0182 [-0.0519, 0.0139]  n=275   VERIFIED
EXTRA seed 1 fold 0..4 AUC   0.9225 [0.8333, 0.9822] / 0.8117 [0.6620, 0.9288] / 0.8400 [0.6866, 0.9686] / 0.8421 [0.7003, 0.9453] / 0.8353 [0.6547, 0.9643]   n=55 each   VERIFIED
```

Read this with care. Seed 0 and seed 1 differ by 0.04 in pooled AUC, and the sign of the gap to zero shot flips
between them. Both paired intervals include 0. The difference between seeds is larger than the gap itself.

No value from item 3 or item 4 is in the lists above. Item 3 has no rater data, and item 4 had no final tex yet.

## 2. Per fold AUC, median answer mass, seed 1 folds

Answer mass is P(Yes set) + P(No set) under the full vocabulary softmax at the answer position:
- Yes set token ids: 7414, 9454, 9693, 9834, 14004
- No set token ids: 902, 2152, 2308, 2753, 8996

It is logged for every test clip in the `answer_mass` column of the per clip csv. p_yes is the two logit softmax over
' Yes' (7414) and ' No' (2308).

| fold | seed | AUC [95% interval] | median answer mass | n |
|---|---|---|---|---|
| 0 | 0 | 0.9167 [0.8267, 0.9783] | 0.9939 | 55 |
| 1 | 0 | 0.8625 [0.7334, 0.9613] | 0.9934 | 55 |
| 2 | 0 | 0.8417 [0.6955, 0.9556] | 0.9972 | 55 |
| 3 | 0 | 0.8372 [0.6955, 0.9464] | 0.9979 | 55 |
| 4 | 0 | 0.8585 [0.7112, 0.9671] | 0.9970 | 55 |

**Folds that ran seed 1 under the rule: none.** Every seed 0 median mass is at or above 0.9934, far above the 0.5
threshold. All five on-pod MASS_CHECK files say no_rerun, so the seed rule result is the seed 0 result, and no
seedrule csv was built. In PART23, fold 3's median mass was 1.07e-05. It did not collapse this time.

For comparison, the extra seed 1 median masses for folds 0 to 4 are 0.9946, 0.9947, 0.9963, 0.9962 and 0.9960, all
verified.

**Run facts** (from the sidecars, checked by the verifier where noted):
- **Model:** Qwen/Qwen2.5-Omni-7B, HF snapshot ae9e1690543ffd5c0221dc27f79834d0294cba00.
- **Script:** `scripts/part24/pod/sft_whole24.py`, sha256 ad2cf8c9..., the same on all 10 runs.
- **Recipe:** AdamW 1e-4, 3 epochs, one clip per update, cross entropy on the Yes and No logits. Only the projector is
  trained.
- **Hardware and libraries:** NVIDIA H200, torch 2.8.0+cu128, transformers 5.17.0.
- **Training:** 25.7 to 27.6 minutes per fold, with 0 checkpoint restarts.
- **Data (verified):**
  - The fold file sha256 1d57d462... is the same as PART23 (`folds/edaic_groupkfold5_pod.csv`).
  - The window cut list hashes are the same as PART23 (`manifests/part24/cut_sha256_p23.txt`).
  - Each clip appears once.
  - The largest audio_tok is 22500 (900 s), and 89 clips reach it.

**Files:**
- `scores/part24/FT/FT24_whole_seed0_perclip.csv` and its `.sidecar.json` (the reported result)
- `scores/part24/FT/FT24_whole_seed1all_perclip.csv` and its `.sidecar.json` (extra)
- `scores/part24/FT/FT24_values.tsv` and its `.sidecar.json`
- the per fold pod outputs in `scores/part24/pods/`
- verifier files: `scores/part24/verify/FT24_VERIFIED.tsv`, `scores/part24/verify/FT24_VERIFIED.sidecar.json`,
  `scripts/part24/verify/verify_ft24_indep.py` and `scores/part24/verify/verify_ft24_indep.out`

The per clip files in this repository have no label column (see the README). The verifier read the labelled copies on
the authors' machine.

## 3. Figure 1, whole interview version

**Paths**
- `figures/part24/fig1_gap_whole.pdf` (md5 e05e5188...) uses the Overleaf three panel layout and is the drop-in for
  the current file. The same bytes were also copied beside the file Overleaf compiles,
  `<local data dir>/paper1_submission_23sep/zip/figures/fig1_gap_whole.pdf` (md5 identical).
- `figures/part24/fig1_gap_whole_v4layout.pdf` (md5 c3ee6d4d...) is the literal make_fig1_v4.py layout. It is a single
  axes with no gap shading, so it has no gap swatch.
- Script: `scripts/make_fig1_whole.py`. The copy that drew the figure had sha256 aaec628e.... The repository copy
  differs from it only in its paths and its interpreter line. make_fig1_v4.py was not edited.
- Also: the PNGs in `figures/part24/`, and the plotted csvs and sidecars in `scores/part24/fig/`.

**Why two files.** make_fig1_v4.py does not draw the figure the tex uses. The tex includes `fig1_gap.pdf`, and Overleaf
compiles `zip/figures/fig1_gap.pdf` (md5 bbefb559..., the same bytes as `figures/fig1_gap.pdf` here). That file has
three stacked panels, gap shading, a gap swatch, a "proj." band and a "chance" line. v4 has none of these. The
requested E-DAIC panel and gap swatch only exist in the Overleaf layout.

**Checks.** The independent check is `scripts/part24/verify/fig_item2/verify_item2_indep.py` with output
`scores/part24/verify/fig_item2/verify_item2_indep.out`. It uses its own code and does not import the figure step.
The figure step's own check is `scripts/part24/fig/verify_fig1_whole.py` with output `scores/part24/fig/verify_fig1_whole.out`.
- **Smallest font:** 9.0 pt in fig1_gap_whole.pdf, from 28 PyMuPDF spans and 29 content stream text objects, all
  9.0 pt. The figure is 243.780 pt wide, which is the spconf column width, so its scale is 1.00000 and the text stays
  9.0 pt in the paper. The v4layout file runs 9.0 to 10.0 pt. For comparison, the current fig1_gap.pdf goes down to
  7.5 pt ("proj." and "chance"), which prints at 7.49 pt.
- **Fonts:** STIXGeneral-Regular and STIXGeneral-Italic, TrueType, embedded, with no Type 3 fonts and no text drawn
  as paths.
- **Largest per layer difference, figure curves against pod curves (E-DAIC):**
  - Plotted arrays against the pod files `scores/part23/o25_whole_encoder_perlayer.csv` and
    `scores/part23/o25_whole_llm_perlayer.csv` (column auc_oof): **0** on the encoder (32 layers) and **0** on the LLM
    (29 stages).
  - Curves read back from the PDF vertices: 4.9e-09 on the encoder and 5.1e-09 on the LLM (independent check). The
    figure step's own check gives 8.6e-08 and 7.8e-08.
  - The 4 copies of each pod file have the same sha256, and both hashes match the figure sidecar.
  - The PC-GITA and Pitt curves are unchanged from the current figure (difference 0).
  - The E-DAIC curves moved from the current figure by up to 0.0297 on the encoder and 0.0597 on the LLM. That is the
    switch to whole interview curves.
  - The pod curves against the Linux sklearn 1.9.1 recompute differ by at most 0.00058 on the encoder and 0.00203 on
    the LLM (0.19 pt on the printed panel).
- **Answer line:** dotted at 0.836600 as read from the PDF. roc_auc_score on the PART23 zero shot file gives 0.836596.
  **Its label prints "0.84".**
- **Gap swatch:** tinted, at opacity 0.16.
- **Current file intact: yes.**
  - `zip/figures/fig1_gap.pdf` is still md5 bbefb559..., 16467 B, mtime 20:34:38Z, which is before any PART24 file.
    It is byte identical to the overleaf.zip entry.
  - `<local data dir>/paper1_final/figures/fig1_gap.pdf` is still md5 456ca061... (the older v4 output, not what
    Overleaf compiles).
  - The tex is still md5 3f43af01... on the authors' machine and in zip/, with mtime 13:38:06Z. It was not touched.

## 4. Rater sheet

**Status: no rater answers yet, so there are no statistics.** Both sources were empty:
- Local sheet `<local data dir>/release/edaic_rerun/rater_sheet_heard80.csv`: rater1 0 of 80, rater2 0 of 80,
  recounted at 23:03Z.
- The shared online copy of the sheet: rater 1 0 of 80, rater 2 0 of 80, from a read only export at 23:03Z.

Both sheets hold transcript text, so neither is in this repository.

The scripts in `scripts/part24/rater/` are ready and were tested on synthetic sheets:
- `rater_stats.py`: for each rater, agreement with the scorer direction; inter rater agreement; Cohen's kappa. It
  reports these on the 80 items and on the 40 conflict items, with an item bootstrap and a speaker bootstrap.
- `verify_rater_stats.py`: the independent recompute.
- `wait_and_run.sh`: polls every 5 minutes, runs the stats and the verifier when both columns are full, and never
  guesses a cell.

The synthetic test used `scripts/part24/verify/make_synth.py` and the hand recompute
`scripts/part24/verify/hand_check.py`. The synthetic sheets and their outputs are not in this repository, because the
sheets copy the transcript text of the real sheet and the numbers come from invented rater answers.

**Scorer direction** is `sent_heard`, the cardiffnlp twitter-roberta-base-sentiment-latest sign on the heard window, as
the tex describes. The dq_lex direction is also reported. The two differ on 6 items (R005, R008, R018, R041, R032,
R057).

The 80 items come from 48 speakers, and the 40 conflict items from 33.

The waiting script was not started at the time of writing.

## 5. Final check of the tex and PDF (item 4)

**Tooling status: ready, but the watcher was not started.** The trigger had not happened.
`<local data dir>/paper1_final/AudioLLMHealthUtilityGap.tex` still had mtime 1790170686 (13:38:06Z), and no newer
PDF was in that folder. Scripts in `scripts/part24/part12/`:
- `final_check.sh TEX PDF`: prints every MISMATCH and UNMATCHED row with its tex line and sentence, the page count, the
  page 5 text, the smallest font size and font embedding. Each run gets its own folder with a sidecar.
- `wait_final_tex.sh`: checks every 120 s. Four stub tests passed.
- `triage_verify.py`, `report_part12.py` and `pdf_fonts.py`, run with PyMuPDF 1.28.2 in a local virtual environment.

**Dry run** on the current tex (sha256 1e17f743...) and `paper_29.pdf` (sha256 0b1ada4c...), run folder
`scores/part24/part12/final_runs/AudioLLMHealthUtilityGap_20260923_145617/`:
- Part 12 exited 0 in 4 s with 113 MATCH, 20 MISMATCH and 43 UNMATCHED (16 of them method constants). This is the same
  as the 06:38 run. Every row with its tex line is in `final_check.log`.
- The PDF has 5 pages. **Page 5 starts with 15 words of body text:** the last lines of the Conclusions ("...information
  in their representation and future work should focus on further narrowing the utility gap."). Then come
  Acknowledgments, Compliance with Ethical Standards, and References [1] to [28].
- **Smallest font in the PDF: 5.98 pt** (one "1" on page 4). Other sizes under 9 pt:
  - 6.97 pt (34 characters)
  - 7.49 pt (the current figure's "proj." and "chance")
  - 7.97 pt (the corresponding author footnote and the GitHub URL)
  - 8.49 pt (the current figure's labels)
  - 8.88 to 8.99 pt (references and captions)
- The body text is 10.06 pt. There are 14 fonts, all embedded, with 0 Type 3.
- **ICASSP 2027 rule**, checked on 23 September 2026 on two pages:
  - The paper kit (cmsworkshops.com/ICASSP2027/papers/paper_kit.php) limits page 5 to references, funding
    acknowledgements and the ethics statement. It sets a 9 pt minimum font everywhere, figure captions included.
  - The author guidelines (2027.ieeeicassp.org/author-guidelines) also say the 5th page holds references only.
  - So paper_29.pdf breaks the page 5 rule and the 9 pt rule. The new fig1_gap_whole.pdf fixes the 7.49 and 8.49 pt
    figure text, but not the 5.98, 6.97 or 7.97 pt text.

The extracted text files in the run folder have their en dashes written as "to" (for example "pp. 10 to 49").

**Triage of the last run** (06:38 run, 20 MISMATCH): 16 are matcher mispairings where the paper is right, 2 are a wrong
file copy, and 2 are real.

The rows below were recomputed again with separate code (`scripts/part24/verify/triage_indep/indep_triage.py`, output
`scores/part24/verify/triage_indep/indep_triage.out`). Every value matches triage_verify.py to 4 dp, intervals
included.

**REAL**
- **Line 244 (red), "significant performance gain in AD and MDD detection".**
  - The AD part holds, paired FT minus zero shot:
    - Pitt +0.1095 [0.0454, 0.1728]
    - ADReSSo +0.2096 [0.1294, 0.2867]
    - ADReSS +0.1443 [0.0490, 0.2370]
  - The MDD part fails:
    - E-DAIC first 30 s: -0.0537 [-0.1221, 0.0152]
    - The older full window files: -0.1863 [-0.2696, -0.1098]
  - The PART24 whole interview result (+0.0223 [-0.0064, 0.0547]) also includes 0. So no E-DAIC version shows a
    significant gain.
- **Line 97, "speaking rate alone reaches 0.92".** No result file gives 0.92.
  - Speaking rate alone gives 0.6970 [0.6407, 0.7495] on all 708 segments (266 speakers) and 0.7082 [0.6387, 0.7688]
    on the kept 468 (228 speakers), direction free.
  - The only 0.92 is a code comment on line 73 of `scripts/upstream/build_ad_conflict.py`.

**Wrong file copy**
- **Line 172, AF3 Pitt conflict 0.61 and agreement 0.59.** The paper matches the part10 copy: 0.6075 [0.5063, 0.7129]
  (n 146, 100 speakers) and 0.5855 [0.5144, 0.6554] (n 322, 175 speakers). The matcher used the part3 copy: 0.6169
  [0.5188, 0.7209] and 0.5822 [0.5080, 0.6528].

**The 16 mispairings** were recomputed only by triage_verify.py and not by a second check. Their values are in
`scores/part24/part12/triage_last_run.csv` and should be treated as not independently verified.

**Caveat, line 149:** the PC-GITA Delta of 0.35 uses the single split probe. With the five repeat probe mean it would be
0.33.

## 6. Pod cost and shutdown

**Estimate:** $33.16. That is $4.59/hr (H200, SECURE) times the time from pod creation to the watchdog DELETE,
recomputed separately from the watchdog log and the create records with the same total.
- Seed 0 pods p24-ft0 to 4: $16.10 (39.2, 42.6, 43.4, 39.8 and 45.4 min).
- Seed 1 pods p24s1-ft0 to 4: $17.06 (43.1, 46.5, 43.8, 44.4 and 45.0 min). These were launched pre-emptively.
  Because the seed rule never fired, this $17.06 bought only the extra seed 1 numbers.

**Billed figure:** not posted yet when these notes were written. A read only poller checks the RunPod billing API
every 30 minutes.

**No PART 24 pod was left running.** A listing at 23:02:48Z showed no pod whose name starts with "p24".
- The watchdog (`scripts/part24/pod/watchdog24.sh`) deleted all 10 (HTTP 204, then GONE) between 22:32:11Z and
  22:44:28Z.
- It exited on its own at 22:44:28Z with 0 live pods, and made no idle, stall or age kills.
- Every pull had 54 files on the pod and 54 in the tar, and the result sha256 matched.

## 7. Open items at the time of writing

1. Pick the figure.
   - `fig1_gap_whole.pdf` is the Overleaf layout drop-in, all text 9 pt.
   - `fig1_gap_whole_v4layout.pdf` is v4 as literally requested, with no gap swatch.
   - The new figure labels the E-DAIC answer line 0.84, but Table 1 line 125 printed 0.83, which Part 12 matched to the
     older `o25_full_zeroshot.json` 0.8285. The figure and the table must use the same zero shot.
2. Decide the E-DAIC fine tune wording. The tex had Table 1 E-DAIC fine tuned 0.64 (older `sft_full_sft.json` 0.6421),
   and line 226 said E-DAIC "drops".
   - PART24 whole interview: seed 0 gives 0.86, +0.02 [-0.01, +0.05] against zero shot. The extra seed 1 gives 0.82,
     -0.02 [-0.05, +0.01].
   - Neither seed supports "drops" or a gain, and the red MDD claim on line 244 is unsupported either way.
3. Fix the two REAL triage rows and the one file copy row.
   - Line 97: 0.92 should become 0.70, or the number should go.
   - Line 244: say AD only.
   - Line 172: choose the part10 or part3 AF3 copy. If part10, update `lookup/master_lookup.csv` so the matcher stops
     flagging it.
4. Fix page 5 and the fonts under 9 pt. Move the 15 Conclusions words off page 5, and remove the 5.98, 6.97 and
   7.97 pt text.
5. Both raters fill their columns of the rater sheet. Then start `scripts/part24/rater/wait_and_run.sh` in the
   background.
6. Save the final tex and its compiled PDF in the same folder within 10 minutes of each other, then start
   `scripts/part24/part12/wait_final_tex.sh`, or run `scripts/part24/part12/final_check.sh TEX PDF` directly. That
   prints every MISMATCH and UNMATCHED row with its line, the page 5 text and the smallest font.
