# PART 23 notes: independent check of the whole-interview E-DAIC run

The files of this run are in `scores/part23/`: zero-shot, probe and extraction results in `A/`, processor
truncation checks in `C/`, projector fine-tune folds in `FT/` and the checks in `verify/`. The scripts are in
`scripts/part23/`. The per-clip files carry no clinical label. To recompute an AUC, join the label from a licensed
copy of E-DAIC on `pid`.

## Headline numbers

| Row | 300 s value | Whole value | File | Verified |
|---|---|---|---|---|
| Zero-shot AUC | 0.8278 [0.7703, 0.8822] | 0.8366 [0.7786, 0.8866] | `scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv` | whole: yes, 300 s: yes |

Items 1 to 10 (37 sub-values, per-fold AUCs included) were recomputed from the per-clip CSVs with sklearn
`roc_auc_score` and the same bootstrap convention: 28 verified, 0 mismatched, 3 new values with no claim to check
(items 9a and 9b are new computations with no claimed value).

## Item 10, truncation recount

- Duration column: `win_dur_s` in `scores/part23/C/C_trunc_275_derived.csv`.
- Qwen3-Omni (cap 900 s): recomputed cut count 0 (claimed 0, verified). A plain `win_dur_s > 900` test on the raw
  floats gives 9 hits, all at `win_dur_s = 900.0001`. That 0.0001 s is a float artifact of dividing the sample count
  by the sample rate. Counting audio tokens with the real 100 Hz feature frame quantization (`round(dur*100)` frames)
  gives the same frame count for the capped (900.0 s) and raw (900.0001 s) durations on all 9 clips, so 0 windows
  are cut. Verified with the quantization rule.
- AF3 (cap 600 s): recomputed cut count 136 (claimed 136, verified). No boundary cases near 600 s.

## Item 8, fold 3 collapse

- FT whole fold 3 (`scores/part23/FT/FT_whole_fold3_oof.csv`, n=55): median `answer_mass` 0.0 (claimed 0.0000,
  verified). Clips with mass under 0.01: 55 of 55, which is every fold 3 test clip.
- The other FT whole folds and all FT ctrl300 folds have a median mass above 0.99 (checked directly, not in the
  table).

## Item 9, new computations

- FT ctrl300 minus the zero-shot first 300 s answer (`p_yes_300s_T7b`), paired: -0.0128 [-0.0483, 0.0214].
- FT whole pooled AUC with fold 3 removed (220 clips), a sensitivity number only: 0.8274 [0.7569, 0.8898].

## Item 11, per-layer curves on the Mac

The hidden-state shards were available locally (275 files, `enc` (32,1280) and `llm` (29,3584) per clip). They are
not released. For all 32 encoder stages and all 29 language model stages the check refit GroupKFold(5) by speaker,
StandardScaler and LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0), and took auc_oof on the pooled
out-of-fold scores.

- Max abs diff, encoder: 0.0611 (stage 18).
- Max abs diff, language model: 0.0578 (stage 3).
- No stage matches the claimed auc_oof to 4 decimals. Typical diffs are 0.01 to 0.06. The local refits are
  deterministic (a rerun gives identical output), but sklearn raised `RuntimeWarning: overflow/divide-by-zero
  encountered in matmul` inside the lbfgs solver on several stages. That is the sign of a nearly separable, badly
  conditioned fit, with about 220 train rows against 1280 to 3584 features and weak L2 at C=1.0. Fits like this do
  not reproduce bit for bit across BLAS or sklearn versions even with identical code, which plausibly explains the
  gap without a data or method error. The curve shape agrees. The encoder AUC starts at about 0.55 to 0.59 at
  stage 0 and rises to a plateau in the mid 0.6 range. The language model rises from about 0.6 to the high 0.7
  range by stage 28. The per-stage numbers cannot be verified to 4 decimals on this setup.

The per-stage detail is in `scores/part23/verify/PART23_VERIFIED.tsv` (rows `11_encoder_stage*` and
`11_llm_stage*`).

## Per-layer curves (Figure 1 E-DAIC panel), second check on Linux

- Recomputed on a Linux x86 CPU pod with the pods' library versions (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1):
  `scores/part23/verify/curves_linux191_pid.csv`, environment in `scores/part23/verify/curves_linux191_env.txt`.
- Encoder: mean abs diff 0.0002, max 0.0006, Pearson r 1.0000, peak layer 19 in both (0.6876 vs 0.6875).
- Language model: mean abs diff 0.0005, max 0.0020 (stage 22), Pearson r 0.9999, peak stage 28 in both (0.7753).
- 10 of 61 stages match to 4 decimals. The float32 lbfgs fits are ill-conditioned (n=220 train, p=1280 or 3584), so
  small BLAS or thread differences move them. Verified: NO under the 4 decimal rule. The curves agree within 0.002,
  which does not show at the figure's scale. No curve value is printed in the paper text.
- Figure file: `<local data dir>/paper1_submission_23sep/fig/fig1_gap_wholeinterview.pdf` (E-DAIC answer line
  0.8366, verified).
