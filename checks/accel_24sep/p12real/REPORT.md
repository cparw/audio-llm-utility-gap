# p12real: Part 12 REAL and WRONG COPY rows on the final tex

Checked 24 Sep, about 00:25 UTC. Read only. Independent code, nothing imported from other checks.
Tex: <local data dir>/paper1_submission_23sep/final_overleaf_2325Z/AudioLLMHealthUtilityGap.tex
Triage: checks/final_2325Z/part12/triage_final_2325Z.csv (CSV rows 8, 9, 26, 46, 47, 48, 49)
AUC is sklearn roc_auc_score. Interval: 2000 speaker bootstrap draws, fresh default_rng(0) per cell, rng.choice of unique speakers with replacement, all their clips, 2.5 and 97.5 percentiles. No draw was skipped in any cell.

## Verdict: all 7 triage calls confirmed. Yes on every item.

| Line | Tex value | Matcher file and value | Right file and value [95% CI], n | Triage call confirmed |
|---|---|---|---|---|
| 125, E-DAIC zero-shot audio | 0.84 | edaic_rerun/variants/o25_full_zeroshot_scores.csv (first 300 s run, json says 0.8285): 0.8285 | part23/A/A2_edaic_whole_zeroshot_perclip.csv, p_yes_whole: 0.8366 [0.7786, 0.8866], 275 clips, 275 speakers, 66 positive | Yes, WRONG COPY |
| 125, E-DAIC fine-tuned audio | 0.86 | edaic_rerun/variants/sft_full_oof.csv (30 s projector FT, json sft_full_sft.json says 0.6421): 0.6421 | part24/FT/FT24_whole_seed0_perclip.csv, p_yes: 0.8589 [0.8050, 0.9029], 275 clips, 275 speakers | Yes, WRONG COPY |
| 171, AF2 Alzheimer's conflict | 0.58 | paper1_local_runs/af2_results/af2_pitt468.csv, conflict arm: 0.6310 | part20/POD4c/af2_orig30_conflict.csv: 0.5835 [0.4730, 0.6913], 146 clips, 100 speakers | Yes, WRONG COPY |
| 171, AF2 Alzheimer's agreement | 0.53 | af2_pitt468.csv, agreement arm: 0.5544 | part20/POD4c/af2_orig30_agreement.csv: 0.5337 [0.4574, 0.6116], 322 clips, 175 speakers | Yes, WRONG COPY |
| 172, AF3 Alzheimer's conflict | 0.61 | overnight2/part3/af3_pitt.csv, conflict arm: 0.6169 (would print 0.62) | overnight2/part10/af3_pitt.csv, conflict arm: 0.6075 (full 0.60748) [0.5063, 0.7129], 146 clips, 100 speakers | Yes, WRONG COPY |
| 172, AF3 Alzheimer's agreement | 0.59 | overnight2/part3/af3_pitt.csv, agreement arm: 0.5822 (would print 0.58) | overnight2/part10/af3_pitt.csv, agreement arm: 0.5855 [0.5144, 0.6554], 322 clips, 175 speakers | Yes, WRONG COPY |
| 149, readout cosine | 0.01 | omni/readout_direction.csv row pitt, cosine: -0.0109 | recomputed from overnight2/part10/o25_pitt_states.npz and omni25_readout.pt: -0.0109 (full -0.01085); random floor 0.0135 | Yes, REAL (sign only) |

Paths: edaic_rerun/, overnight2/ and omni/ files are under the authors' local release folder, <local data dir>/release; part23, part24, part20/POD4c are under scores/ (part20/POD4c is also symlinked from <local data dir>/release/scores/part20/POD4c); af2_pitt468.csv is <local data dir>/paper1_local_runs/af2_results/af2_pitt468.csv.

Every recomputed number equals the triage verified_value to 4 decimals, intervals included.

## Notes per item

Line 125. Table 1 columns are Zero-shot Audio, Zero-shot Text, Fine-tuned Audio, so 0.84 is zero-shot audio and 0.86 is fine-tuned audio. The whole interview files round to the tex (0.8366 to 0.84, 0.8589 to 0.86); the matcher's 300 s and 30 s files do not (0.83, 0.64). Line 81 says the model hears the whole interview, so the whole interview files are the right source. A2 and FT24 hold the same 275 pids with identical labels.

Lines 171 and 172. Table 2 order is Depression Conflict, Depression Agree, Alzheimer's Conflict, Alzheimer's Agree, so the checked cells are columns 4 and 5. The arm clip sets are identical across af2_orig30, af2_pitt468 and af3 part10 (146 conflict, 322 agreement; 100 and 175 speakers), matching the caption. The caption says 30 s segments; af2_orig30 has scored_s = 30 for AF2, which fits the triage reason. For AF3 the part10 copy rounds to both tex cells and part3 rounds to neither (0.62, 0.58), so the paper does use part10.

Line 149. The paper says the cosine is 0.01. The refit (final stage answer state ans[:, -1, :], mass weighted Yes minus No lm_head direction, LogisticRegression C=1, balanced, max_iter 2000, StandardScaler inside the fold, GroupKFold(5) by speaker, coef divided by scale and averaged over folds) gives -0.0109. All 5 per fold cosines are negative (-0.0144, -0.0135, -0.0085, -0.0094, -0.0012). The same refit also reproduces the probe at 0.7709 and the projection AUC at 0.6608 (both printed as 0.77 and 0.66 on line 149). The random floor, mean absolute cosine of 1000 random directions (default_rng(0)) with d, is 0.0135, so the magnitude claim holds. The file's own speaker bootstrap interval for the cosine is [-0.0235, 0.0097] (readout_direction.json, 2000 draws, refit per draw). That bootstrap was not rerun here. Fix: print $-$0.01, or write "0.01 in magnitude". Side fact: with the plain unweighted direction the cosine is +0.0011, but the paper and CSV use the weighted one.

## Files written
- checks/accel_24sep/p12real/p12real_auc.py and p12real_auc.csv, p12real_auc.log
- checks/accel_24sep/p12real/p12real_cosine.py and p12real_cosine.log
