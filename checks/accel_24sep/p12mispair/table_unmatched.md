| row | item | tex | computed | yes/no | how | file |
|---|---|---|---|---|---|---|
| 2 | L74_739 | 47 | 47 | yes | speaker-label units in both arms (raw speaker IDs: 48); per arm {'agreement': 175, 'conflict': 100}; total spk 227; spk x label 228 | <local data dir>/DementiaBank/pitt_conflict_manifest.csv |
| 6 | L81_570 | 89 | 89 | yes | sum capped; max win_dur 900.0001 s; n 275 | RFM/scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv |
| 15 | L92_681 | 99 | 99.17 | yes | conflict clips where greedy first word equals the logit decision (p_yes>0.5): 0.9917; file 'agrees' col 0.9917; Yes/No first word on all 966: 1.0000 | <local data dir>/release/edaic_rerun/part16/POD1/p14_o25_greedy.csv |
| 21 | L97_823 | 33 | 33.3 | yes | median dur | <local data dir>/DementiaBank/pitt_conflict_manifest.csv |
| 23 | L101_219 | 7.5 | 7.5386 | yes | mean age dementia minus control per speaker label (228 spk-label units); clip level 6.5763 | <local data dir>/DementiaBank/pitt_conflict_manifest.csv |
| 24 | L101_395 | 81 | {0: 81, 1: 81} | yes | speaker-label units per label in the whole-dataset matched manifest (rows 330) | <local data dir>/DementiaBank/pitt_agematched_manifest.csv |
| 28 | L140_493 | 0.66 | 0.662 | yes | AUC [0.5831, 0.7315] n=275 | <local data dir>/release/omni_final/omni_edaic_zeroshot_scores.csv |
| 31 | L140_573 | 0.81 | 0.8122 | yes | AUC [0.7479, 0.8707] n=275 | <local data dir>/release/edaic_rerun/variants/o25_mid300_zeroshot_scores.csv |
| 32 | L140_607 | 0.84 | 0.8366 | yes | AUC p_yes_whole | RFM/scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv |
| 33 | L140_674 | 0.63 | 0.6168 | yes | mean-of-5 enc per window (mean, max single repeat): first30 0.5966 (0.6301); mid30 0.5771 (0.6179); mid300 0.6168 (0.6657); full_json 0.5930 (0.6448); first300 0.5884 (0.6060); whole 0.5981 (0.6210) | nested_repeats json files + T7b + A3 (see how) |
| 34 | L149_1433 | 38 | 38 | yes | pairs; max /age_diff/ 5.0 | <local data dir>/release/scores/part20/POD6/B_neurovoz_matched/nv_matched_pairs.csv |
| 35 | L149_1472 | 907 | 907 | yes | clips; speakers 76 | <local data dir>/release/scores/part20/POD6/B_neurovoz_matched/B_channel5_nvmatched.csv |
| 36 | L149_1595 | 0.71 | 0.7112 | yes | AUC p_five_feature_oof | <local data dir>/release/scores/part20/POD6/B_neurovoz_matched/B_channel5_nvmatched.csv |
| 37 | L149_735 | 0.53 to 0.84 | 0.5309 to 0.8439 | yes | egemaps_auc over 7 sets (summary csv) | <local data dir>/release/overnight/egemaps_baseline.csv |
| 39 | L180_1096 | 86 | 86.34 | yes | text decision equals audio decision at 0.5 on 966 clips | <local data dir>/release/edaic_rerun/part16/POD1/p14_o25_1a_joined.csv |
| 40 | L180_1309 | 0.34 | 0.3439 | yes | conflict arm n=483 [0.2900, 0.3979] | <local data dir>/release/edaic_rerun/part16/POD1/p14_o25_audio_p2.csv |
| 41 | L180_1318 | 0.24 | 0.2392 | yes | conflict arm n=483 [0.1940, 0.2865] | <local data dir>/release/edaic_rerun/part16/POD1/p14_o25_audio_p3.csv |
| 44 | L180_1373/L180_1396 | 646 pairs, 0.26 | 646 conflict rows, 147 spk, AUC 0.2619 | yes | conflict arm [0.2188, 0.3075]; total rows 1292; all speakers 147 | <local data dir>/release/edaic_rerun/part16/POD1/p14_thr060_omni.csv |
| 45 | L180_1381/L180_1405 | 281 pairs, 0.19 | 281 conflict rows, 104 spk, AUC 0.1936 | yes | conflict arm [0.1501, 0.2438]; total rows 562; all speakers 104 | <local data dir>/release/edaic_rerun/part16/POD1/p14_thr080_omni.csv |
| 51 | L180_1472/L180_1543 | 924 pairs, 0.18 | 924 conflict rows, 237 spk, AUC 0.1759 | yes | conflict arm [0.1448, 0.2095]; total rows 1848; all speakers 237 | <local data dir>/release/edaic_rerun/part16/POD1/p14_phq10_omni.csv |
| 53 | L180_1616/1618 | 1,334 | 1334 | yes | conflict pairs | <local data dir>/release/scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv |
| 55 | L180_1780 | 420 | 420 | yes | conflict & RoBERTa conflict | <local data dir>/release/scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv |
| 56 | L180_1845 | 855 | 855 | yes | conflict & RoBERTa none | <local data dir>/release/scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv |
| 57 | L180_2058 | 97 | {'agreement': 0.9834, 'conflict': 0.971} | yes | share p_yes>0.5 per arm, E-DAIC pairs (rows incl. repeats) | <local data dir>/release/edaic_rerun/part14/p14_af2.csv |
| 58 | L180_2110 | 87 | {'agreement': 0.8675, 'conflict': 0.94} | yes | share p_yes>0.5 per arm, E-DAIC pairs | <local data dir>/release/edaic_rerun/part14/p14_q2a_zeroshot_scores.csv |
| 59 | L226_1020 | 0.55 | 0.552 | yes | LoRA conflict arm n=146 | <local data dir>/release/edaic_rerun/part16/POD2/lora_pitt_oof.csv |
