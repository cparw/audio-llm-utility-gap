# Leftover checks, 23 September 2026

Notes on the leftover checks run on 23 September 2026, between about 22:00 and 00:25 UTC. They close the open items of the
status audit made that day. Every value below was recomputed by a second, separate verifier. The files are in
`scores/leftovers_23sep/` and `scripts/leftovers_23sep/`.

## Summary

1. Every paper value that the audit flagged as untwinned or disagreeing now has a saved twin that agrees at 4 dp,
   recomputed by a separate verifier. Two gaps stay open. First, the T5 curves for eight other datasets and windows:
   five have no states on disk, and three only have states from a different extraction. Second, the exact mean of
   five intervals for the M7 gaps, whose per repeat OOF was never saved.
2. The saved Pitt and E-DAIC first 30 s T5 curves reproduce exactly on Linux pods. The audit's 0/32 came from
   scikit-learn 1.7.2 on the Mac making different GroupKFold folds than 1.9.1 on the pods. The same fold change
   explains the POD3 3b (0.7864 vs 0.8307) and by arm (0.5520, 0.9314) disagreements.
3. The checks ran on 52 CPU pods. All of them were deleted when their results were pulled.

## Status of the audit items

| Item | Status | Why |
|---|---|---|
| P17 T5 curves, Pitt and E-DAIC first 30 s | closed (verified) | Unchanged curves_from_states.py on AVX-512 Linux pods: max abs diff 0 on all 32 encoder layers, the projector, 29 LM and 29 answer stages; 32/32 at 4 dp. Layer counts hold: Pitt 31/32, E-DAIC first 30 s 2/32. Verifier 106/106 plus a second verifier's own refit. |
| Why the audit's spot check got 0/32 on T5 | closed (verified) | sklearn 1.7.2 GroupKFold uses the default argsort, 1.9.1 a stable one. A Mac refit with Mac folds equals the P17 verifier's curve to 1.1e-16. Exact 4 dp also needs OpenBLAS SkylakeX kernels (AVX-512) and 1 BLAS thread. |
| T5 curves, ADReSS-2020, NeuroVoz, E-DAIC full | still open | Only states from a different extraction exist (p_yes differs by up to 0.038, 0.047, 0.030). Refits come within 2.9e-3, 1.1e-3, 3.0e-3 and the counts above the answer are unchanged (32, 32, 0), but no 4 dp twin is possible. |
| T5 curves, ADReSSo, PC-GITA, MDVR-KCL, E-DAIC middle 300 s and middle 30 s | still open | No matching Qwen2.5-Omni states on disk (1461 npz files scanned). The 18 Sep H200 pod and podB/podC states were never pulled. These curves stay done_unverified. The 16 T5 paper value rows for ADReSSo, ADReSS-2020, NeuroVoz and KCL in summary_all_v2 say "yes (twin file)", but that twin only re-reads the saved curve. |
| P16 POD3 3b cosine 0.0051, 6 best stage rows, 2 by arm rows | closed (verified) | lm_head fetched from the HF hub (Qwen3-Omni rev 26291f79, sha256 ef1c8244...). Linux refit gives cosine 0.0051, probe 0.8307, by arm 0.6145 and 0.9349, all 6 best stages, at 4 dp with matching intervals. Two separate verifiers agree. |
| P16 M11 group mean intervals (script vs verifier) | closed (verified) | Different estimators, not an arithmetic error. The script resamples the group's own speakers (the rule); the verifier resampled all 37. Script intervals stand, e.g. B PD 0.3049 [0.1301, 0.4944]. |
| P16 M7 gaps for ADReSSo, ADReSS-2020, PC-GITA, NeuroVoz, KCL on the mean of five basis | closed for the points, open for the intervals | Points are exact: PC-GITA is 0.3306. The intervals are the single split interval shifted to the mean of five; the exact version needs per repeat OOF that was never saved. On re-extracted states the shift is off by at most 0.0019 (NeuroVoz) and 0.0053 (ADReSS-2020). Every shifted interval except KCL excludes zero. |
| P16 VERIFY rule, 580 rows (25 no twin, 14 disagreeing, 15 audit only) | closed (verified) | All 54 now have a saved twin at 4 dp, 40/40 intervals too. 10 are log twins only (MEDAIC 8, POD2 L5 and L6): no states or OOF exist for those, and wall clock and GPU memory cannot be recomputed. |
| Twin files for the M7 Pitt gap and POD4 4b | closed (verified) | `scores/leftovers_23sep/p16twins/M7_pitt_gap_twin.tsv` (+0.1128 [0.0455, 0.1768]) and `scores/leftovers_23sep/p16twins/POD4_4b_descriptives_twin.tsv`. |
| summary_all fix (verified column was a copy) | closed (verified) | `summary_all_v2.csv` fills value_verified from real twin files. It was built 3 minutes before two twin tables were rewritten, so 3 rows were stale (tables_check). Only those rows (plus the rule text on 5 sibling M11 rows) were patched into `scores/leftovers_23sep/tables_rebuild/summary_all_v2_patched.csv`; 1524 rows are identical to v2, and the 3 changed values match the independent verifier outputs. A plain rerun of build_summary_v2.py is not safe: it drops the intervals of 7 M3/M7 rows and relabels the 2 POD2 log twins as recomputes (kept as `summary_all_v2_fullrebuild_NOT_USED.csv`). |
| POD3 3b mismatch in DISCREPANCIES.md, twin the cosine | cosine closed (verified), log entry drafted | The entry is drafted in `reports/leftovers_23sep/DISCREPANCIES_entry_pod3b.txt`. |
| E-DAIC fine tune swing (0.6421 vs 0.8150) | partly closed | master_lookup row 205 now records that the 0.6421 recipe keeps only 30 s (`sft_projector.py` line 59, `x[:16000 * 30]`), so it never heard the 300 s window. PART 24 whole interview: seed 0 0.8589, seed 1 0.8184. |
| Stale notes (master_lookup 200 and 205, T7/T4 "NO INTERVAL", 9 P22_S1 texts, n in p14_o25_sftfull.json) | closed (verified) | Edited in place with a backup beside each. tables_check diffed every file against its backup: only the listed notes changed. The edits are listed in `scores/leftovers_23sep/tables/changes.csv`. |
| PART 23 rows in master_lookup | closed (verified) | 8 rows appended (234 to 242), all values and paired notes recomputed at 4 dp by tables_check. |
| Rater kappa | blocked | Both rater columns are 0/80 (PART 24, 23:03 UTC). |

## Findings from these checks

1. The Qwen3-Omni probe 0.83 is one favourable split. It reproduces exactly on Linux, but over 200 speaker splits that
   differ only in tie order the probe averages 0.8033 (middle 95 percent 0.7786 to 0.8244), and only 2 of 200 reach
   0.8307. The claims around it hold on every split: removing the readout direction changes the probe by -0.003 on
   average, and the cosine stays between 0.0023 and 0.0090, inside the random floor.
2. The Qwen2-Audio readout numbers (0.82 and 0.81) rest on the Mac scikit-learn 1.7.2 folds. On Linux 1.9.1 folds M5
   gives probe 0.8044 against 0.8223 and cosine 0.0134 against 0.0145.
3. PC-GITA best LM stage: the AVX2 kernel gives 0.8961 and the AVX-512 kernel gives 0.8960 (the POD3 value). Pitt also
   drifts on AVX2 (encoder 0.8274, LM 0.8633). The best stage never changes.

## Verified values

Bootstrap rule everywhere: 2000 draws, fresh numpy default_rng(0) per cell, speakers resampled with
rng.choice(N, N, replace=True) over the sorted unique speaker ids, 2.5 and 97.5 percentiles, paired cells share idx.
Pod fits: sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1, Python 3.12.

### T5 layer curves (Qwen2.5-Omni)

```
T5 Pitt encoder curve, pod refit vs saved, max abs diff of auc_oof          0          32/32 layers equal at 4 dp       VERIFIED
T5 Pitt encoder layers above the zero shot answer 0.6578                   31/32      saved curve 31/32                 VERIFIED
T5 Pitt projector, LM stages, answer stages, max abs diff                  0          1/1, 29/29, 29/29 at 4 dp         VERIFIED
T5 E-DAIC first 30 s encoder curve, max abs diff (two pods)                0          32/32 layers equal at 4 dp       VERIFIED
T5 E-DAIC first 30 s encoder layers above the answer 0.6620               2/32       saved curve 2/32                  VERIFIED
T5 E-DAIC first 30 s projector, LM stages, answer stages, max abs diff     0          1/1, 29/29, 29/29 at 4 dp         VERIFIED
T5 ADReSS-2020 encoder, refit from another extraction, max abs diff        2.88e-03   1/32 at 4 dp; above answer 32/32 vs 32/32   VERIFIED (gap size only)
T5 NeuroVoz encoder, refit from another extraction, max abs diff           1.09e-03   5/32 at 4 dp; above answer 32/32 vs 32/32   VERIFIED (gap size only)
T5 E-DAIC full encoder, refit from another extraction, max abs diff        2.97e-03   2/32 at 4 dp; above answer 0/32 vs 0/32     VERIFIED (gap size only)
T5 Mac folds (sklearn 1.7.2) vs pod fold file, clips in the same fold      Pitt 125/468, E-DAIC 20/275; pods 468/468 and 275/275        VERIFIED
T5 Mac refit with Mac folds vs the P17 verifier curve                      1.11e-16 (Pitt and E-DAIC first 30 s)                        VERIFIED
```

### PART 16 twins (p16twins)

```
Qwen3-Omni 3b cosine, probe dir vs Yes minus No readout, all               0.0051                                            VERIFIED
Qwen3-Omni 3b cosine, conflict / agreement                                 -0.0067 / 0.0086                                  VERIFIED
Seed row 14, 3b cosine                                                     0.0051                                            VERIFIED
Qwen3-Omni 3b probe, conflict arm                                          0.6145 [0.5021, 0.7246]                           VERIFIED
Qwen3-Omni 3b probe, agreement arm                                         0.9349 [0.9034, 0.9614]                           VERIFIED
Qwen3-Omni 3b probe without d, conflict                                    0.6196 [0.5094, 0.7283]                           VERIFIED
Qwen3-Omni 3b probe without d, agreement                                   0.9341 [0.9021, 0.9604]                           VERIFIED
POD3 3a Pitt encoder, best stage 30                                        0.8276 [0.7765, 0.8760]                           VERIFIED
POD3 3a Pitt LM, best stage 13                                             0.8634 [0.8179, 0.9062]                           VERIFIED
POD3 3a Pitt answer, best stage 26                                         0.8485 [0.8069, 0.8882]                           VERIFIED
POD3 3a PC-GITA encoder, best stage 14                                     0.9133 [0.8519, 0.9604]                           VERIFIED
POD3 3a PC-GITA LM, best stage 14 (AVX-512)                                0.8960 [0.8341, 0.9463]                           VERIFIED
POD3 3a PC-GITA answer, best stage 15 (AVX-512)                            0.8962 [0.8327, 0.9452]                           VERIFIED
M3 Pitt speaker level AUC (228 speakers)                                   0.6771 [0.6096, 0.7494]                           VERIFIED
M3 Pitt clip minus speaker, paired                                         -0.0193 [-0.0565, 0.0172]                         VERIFIED
M3 PC-GITA speaker level AUC (100 speakers)                                0.5948 [0.4778, 0.7089]                           VERIFIED
M3 PC-GITA clip minus speaker, paired                                      -0.0313 [-0.0820, 0.0179]                         VERIFIED
M3 NeuroVoz speaker level AUC (107 speakers)                               0.7948 [0.7045, 0.8733]                           VERIFIED
M3 NeuroVoz clip minus speaker, paired                                     -0.0997 [-0.1325, -0.0613]                        VERIFIED
M7 Pitt encoder probe, mean of 5                                           0.7706 [0.7157, 0.8220]                           VERIFIED
M7 Pitt probe minus zero shot, paired                                      +0.1128 [0.0455, 0.1768]                          VERIFIED
M7 Pitt probe on the mean of 5 OOF probabilities                           0.7917 [0.7324, 0.8480]                           VERIFIED
M7 Pitt gap on the mean of 5 OOF probabilities, paired                     0.1339 [0.0641, 0.2019]                           VERIFIED
POD4 4b windows resolved to CHAT timings                                   468                                               VERIFIED
POD4 4b windows with any INV speech                                        158                                               VERIFIED
POD4 4b mean INV share / PAR share of the heard window                     0.0200 / 0.5871                                   VERIFIED
POD4 4b mean participant only duration                                     17.61 s                                           VERIFIED
POD4 4b cuts under 10 s of participant speech                              68                                                VERIFIED
M9 inventory files with an answer mass column                              307 of 608                                        VERIFIED
M9 minimum median answer mass                                              0.8662                                            VERIFIED
M5 Qwen2-Audio probe without d, all (Mac folds, where M5 ran)              0.8327 [0.7857, 0.8759]                           VERIFIED
M5 without d, Xproj                                                        0.8223 [0.7743, 0.8672]                           VERIFIED
M5 conflict refit without d                                                0.6869 [0.5806, 0.7872]                           VERIFIED
M5 agreement refit without d                                               0.9346 [0.9057, 0.9603]                           VERIFIED
M5 conflict restricted without d / Xproj                                   0.6326 [0.5277, 0.7336] / 0.6180 [0.5128, 0.7213]   VERIFIED
M5 agreement restricted without d / Xproj                                  0.9070 [0.8702, 0.9383] / 0.8988 [0.8593, 0.9326]   VERIFIED
M5 conflict minus agreement without d, paired (228 speakers)               -0.2745 [-0.3754, -0.1707]                        VERIFIED
M5 random floor, all / conflict / agreement                                0.0124 [0.0005, 0.0343] / 0.0123 [0.0005, 0.0350] / 0.0126 [0.0005, 0.0359]   VERIFIED
MEDAIC full LM / enc / ans / proj nested (log twin only)                   0.6853 / 0.5930 / 0.7542 / 0.5293                 VERIFIED (second record)
MEDAIC full minus answer 0.8285, LM / enc / ans / proj (log twin only)     -0.1432 / -0.2355 / -0.0743 / -0.2992             VERIFIED (second record)
POD2 LoRA minutes per fold (run log)                                       4.13; 4.07; 4.05; 4.03; 4.00                      VERIFIED (second record)
POD2 LoRA peak GPU memory (run log)                                        21.013 GB                                         VERIFIED (second record)
```

### Qwen3-Omni POD3 3b (pod3b)

```
3b cosine, all (fp16 lm_head as POD3 used; exact bf16 differs by 1.8e-10) 0.0051 (0.005109)                                 VERIFIED
3b cosine random floor                                                     0.0177 [0.0008, 0.0498]                           VERIFIED
3b probe OOF, all (Linux folds)                                            0.8307 [0.7833, 0.8741]                           VERIFIED
3b readout direction alone, all                                            0.7656 [0.7088, 0.8179]                           VERIFIED
3b probe without d, all, train fold z / held out z                         0.8305 [0.7826, 0.8737] / 0.8276 [0.7795, 0.8696]   VERIFIED
3b probe, Linux fit on Mac folds (the old Mac 0.7864)                      0.7864 [0.7335, 0.8359]; conflict 0.5520, agreement 0.9314   VERIFIED
3b probe, real sklearn 1.7.2 GroupKFold run on Linux                       0.7973                                            VERIFIED
3b probe over 200 tie order splits                                         mean 0.8033, sd 0.0122, range 0.7713 to 0.8414, 2 of 200 at or above 0.8307   VERIFIED
3b cosine over 200 splits                                                  0.0023 to 0.0090, 0 of 200 above 0.0498           VERIFIED
3b without d over 200 splits, train z / held out z                         0.8061 / 0.7998 (change vs probe -0.0029 / +0.0034)   VERIFIED
Pitt speakers with new fold mates, 1.7.2 vs 1.9.1                          228 of 228 (8 distinct clip counts)               VERIFIED
```

### M7 gaps on the mean of five, and M11 KCL group means (m7m11)

```
M7 ADReSSo probe (0.8752) minus zero shot (0.6150), n 237                  0.2602 [0.1773, 0.3379]   interval approximate   VERIFIED
M7 ADReSS-2020 probe (0.8043) minus zero shot (0.6241), n 156              0.1802 [0.0722, 0.2858]   interval approximate   VERIFIED
M7 PC-GITA probe (0.8941) minus zero shot (0.5635), n 1100                 0.3306 [0.2489, 0.4127]   interval approximate   VERIFIED
M7 NeuroVoz probe (0.9213) minus zero shot (0.6951), n 1270                0.2262 [0.1714, 0.2836]   interval approximate   VERIFIED
M7 MDVR-KCL probe (0.7506) minus zero shot (0.7143), n 37                  0.0363 [-0.2075, 0.2638]  interval approximate   VERIFIED
Check: NeuroVoz re-extracted states, exact mean of five interval           0.2267 [0.1742, 0.2851]   (shifted: [0.1723, 0.2841])   VERIFIED
Check: ADReSS-2020 re-extracted states, exact mean of five interval        0.1783 [0.0750, 0.2770]   (shifted: [0.0750, 0.2823])   VERIFIED
M11 KCL cross system disagreement mean, PD n 16 / control n 21             0.3049 [0.1301, 0.4944] / 0.4779 [0.0917, 1.1479]   VERIFIED
M11 KCL mean token logprob, PD / control                                   -0.1103 [-0.1502, -0.0739] / -0.1356 [-0.2345, -0.0759]   VERIFIED
M11 KCL compression ratio, PD / control                                    1.5938 [1.4484, 1.7269] / 1.4898 [1.3076, 1.6417]   VERIFIED
M11 KCL words in the 30 s window, PD / control                             59.1875 [46.1875, 71.9375] / 55.8095 [43.0464, 67.6202]   VERIFIED
M11 KCL WER vs the other stored transcript, PD / control                   0.3123 [0.1974, 0.4388] / 0.3987 [0.2558, 0.5577]   VERIFIED
M11 KCL disagreement median, PD / control                                  0.1027 [0.0323, 0.3617] / 0.0333 [0.0000, 0.1341]   VERIFIED
M11 KCL disagreement mean on clips with 20+ words, PD / control            0.2100 [0.0671, 0.3837] / 0.1506 [0.0562, 0.2684]   VERIFIED
M11 KCL disagreement pooled, PD / control                                  0.2650 [0.0782, 0.4778] / 0.1800 [0.0620, 0.3636]   VERIFIED
M11 KCL PD minus control, WER / median / 20+ words mean / pooled           -0.0863 [-0.2851, 0.1047] / 0.0694 [-0.1008, 0.4112] / 0.0594 [-0.1355, 0.2708] / 0.0850 [-0.1870, 0.3488]   VERIFIED
```

The fourth decimal of two M7 points is bounded, not fixed: PC-GITA 0.3306 or 0.3307, NeuroVoz 0.2262 or 0.2263. Both
print as 0.33 and 0.23.

### master_lookup PART 23 rows, T7b intervals, summary v2 patch (tables)

```
Qwen2.5-Omni E-DAIC whole interview (900 s cap) zero shot, PART 23         0.8366 [0.7786, 0.8866]   n=275   VERIFIED
Qwen2.5-Omni E-DAIC first 300 s zero shot (rebuild run)                    0.8278 [0.7703, 0.8822]   n=275   VERIFIED
E-DAIC whole, encoder / projector probe, mean of 5                         0.5981 [0.5267, 0.6672] / 0.5866 [0.5166, 0.6585]   VERIFIED
E-DAIC whole, LM / answer state probe, mean of 5                           0.7437 [0.6752, 0.8097] / 0.8359 [0.7819, 0.8832]   VERIFIED
E-DAIC whole, projector fine tune (PART 23)                                0.7931 [0.7263, 0.8536]   VERIFIED
E-DAIC first 300 s, projector fine tune, same pod control                  0.8150 [0.7465, 0.8776]   VERIFIED
Paired: whole minus 300 s zero shot                                        +0.0088 [-0.0320, 0.0510]   VERIFIED
Paired: FT whole minus zero shot whole / FT whole minus FT 300 s           -0.0435 [-0.0939, 0.0027] / -0.0219 [-0.0783, 0.0356]   VERIFIED
Paired: FT 300 s minus zero shot 300 s                                     -0.0128 [-0.0483, 0.0214]   VERIFIED
Paired: whole enc / proj / LM / answer state minus zero shot whole         -0.2385 [-0.3244, -0.1505] / -0.2500 [-0.3339, -0.1623] / -0.0929 [-0.1609, -0.0257] / -0.0007 [-0.0455, 0.0436]   VERIFIED
E-DAIC 300 s LM probe mean of 5, and minus answer 0.8278                   0.7001 [0.6231, 0.7727]; -0.1276 [-0.1993, -0.0629]   VERIFIED
E-DAIC 300 s encoder probe mean of 5, and minus answer                     0.5884 [0.5268, 0.6523]; -0.2394 [-0.3188, -0.1572]   VERIFIED
E-DAIC 300 s answer state / projector, mean of 5                           0.7435 [0.6793, 0.8065] / 0.5610 [0.4871, 0.6339]   VERIFIED
Windows cut: Qwen2.5-Omni at 300 s / Qwen3-Omni / AF3 at 600 s             217 / 0 / 136 of 275   VERIFIED
Audio kept by the 0.6421 E-DAIC fine tune recipe                           30 s (sft_projector.py line 59)   VERIFIED
summary_all_v2 patch: PC-GITA LM best stage / answer best stage            0.8960 [0.8341, 0.9463] / 0.8962 [0.8327, 0.9452]   VERIFIED
summary_all_v2 patch: M11 20+ words mean, PD minus control                 0.0594 [-0.1355, 0.2708]   VERIFIED
```

## Notes on the released files

- The clinical label was removed from every per-clip file here, following the repository policy: the `label` column of
  the csv files and the `label` or `y` arrays of the npz files. Join on the id columns to add it back from a licensed
  copy of each corpus. `reports/leftovers_23sep/LABELS_DROPPED.tsv` lists each file. Because of this, the sha256 lists written on the pods (`output_sha256_*.txt`, `SHA256SUMS.txt`,
  `sha256.txt`) no longer match the npz and csv files that carried a label.
- The raw lm_head rows (`W_yes`, `W_no`) were removed from the two `pod3b_primary_*_vectors.npz` files. The directions
  built from them are kept.
- Hidden states and the inputs staged for the pods are not included. Machine paths are written as
  `<local data dir>/...`.
- The pods ran scikit-learn 1.9.1, numpy 2.1.2 and scipy 1.18.1 on Python 3.12. The Mac ran scikit-learn 1.7.2,
  numpy 2.2.6 and scipy 1.15.3 on Python 3.10.

## Files

- Per task values and sidecars:
  - `scores/leftovers_23sep/t5/t5_values.tsv`
  - `scores/leftovers_23sep/p16twins/twins_p16.tsv`
  - `scores/leftovers_23sep/pod3b/pod3b_values.tsv`
  - `scores/leftovers_23sep/m7m11/m7m11_values.tsv`
  - `scores/leftovers_23sep/tables/values.csv`
- Independent verifiers:
  - `scores/leftovers_23sep/verify/t5_VERIFIED.tsv`
  - `scores/leftovers_23sep/verify/p16twins_VERIFIED.tsv`
  - `scores/leftovers_23sep/verify/pod3b_VERIFIED.tsv`
  - `scores/leftovers_23sep/verify/m7m11_VERIFIED.tsv`
  - `scores/leftovers_23sep/tables_check/`
- Summary v2 patch:
  - `scores/leftovers_23sep/tables_rebuild/summary_all_v2_patched.csv`
  - `summary_v2_patch_values.tsv` and `summary_v2_patch_values.sidecar.json` (3 of 3 match, 1524 rows unchanged)
  - `scripts/leftovers_23sep/tables_rebuild/patch_v2.py` and `verify_patch.py`
