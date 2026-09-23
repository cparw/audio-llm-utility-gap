# All results, 23 Sep 2026

Every value below comes from its per-clip file and has an independent check (column 'checked'). CI = 2000-draw speaker bootstrap, 2.5 and 97.5 percentiles.

## A. Paper values (327)

### PART 16 EDAICFULL

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| edaic_full_zeroshot_answer | Qwen2.5-Omni zero-shot answer AUC | 0.8278 [0.7703, 0.8822] | 275 (275) | yes |

### PART 16 M1

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M1_audio_all468 | Pitt Qwen2.5-Omni audio zero-shot answer AUC, all468 arm | 0.6578 [0.6033, 0.7125] | 468 (228) | yes |
| M1_audio_conflict | Pitt Qwen2.5-Omni audio zero-shot answer AUC, conflict arm | 0.5322 [0.4216, 0.6447] | 146 (100) | yes |
| M1_audio_agreement | Pitt Qwen2.5-Omni audio zero-shot answer AUC, agreement arm | 0.7054 [0.6431, 0.7658] | 322 (175) | yes |
| M1_audio_conflict_minus_agreement | Pitt Qwen2.5-Omni audio zero-shot paired conflict minus agreement AUC | -0.1731 [-0.2958, -0.0392] | 468 (228) | yes |
| M1_ALT_text_all468 | ALT Pitt Qwen2.5-Omni transcript-only answer AUC, all468 arm (Sep 20 pod rerun, different prompt run) | 0.6629 [0.5983, 0.7245] | 468 (228) | yes |
| M1_ALT_text_conflict | ALT Pitt Qwen2.5-Omni transcript-only answer AUC, conflict arm (Sep 20 pod rerun, different prompt run) | 0.5560 [0.4516, 0.6680] | 146 (100) | yes |
| M1_ALT_text_agreement | ALT Pitt Qwen2.5-Omni transcript-only answer AUC, agreement arm (Sep 20 pod rerun, different prompt run) | 0.7078 [0.6381, 0.7696] | 322 (175) | yes |

### PART 16 M10

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M10#01:MDVR_KCL_Qwen2.5_Omni_transcript_only_answer_AUC | MDVR-KCL Qwen2.5-Omni transcript-only answer AUC | 0.7634 [0.5936, 0.9118] | 37 (37) | yes |
| M10#02:MDVR_KCL_Qwen2.5_Omni_audio_zero_shot_answer_AUC | MDVR-KCL Qwen2.5-Omni audio zero-shot answer AUC | 0.7143 [0.5210, 0.8810] | 37 (37) | yes |
| M10#03:MDVR_KCL_Qwen2.5_Omni_encoder_probe_AUC_single_saved_OOF_repeat | MDVR-KCL Qwen2.5-Omni encoder probe AUC (single saved OOF repeat) | 0.7768 [0.5994, 0.9265] | 37 (37) | yes |
| M10#04:MDVR_KCL_paired_transcript_minus_audio_answer_AUC | MDVR-KCL paired transcript minus audio answer AUC | 0.0491 [-0.1827, 0.2818] | 37 (37) | yes |
| M10#05:MDVR_KCL_paired_transcript_minus_encoder_probe_AUC | MDVR-KCL paired transcript minus encoder probe AUC | -0.0134 [-0.2383, 0.2381] | 37 (37) | yes |

### PART 16 M11

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M11#27:auc_text_condition_recomputed | auc_text_condition_recomputed | 0.7634 | 37 (37) | yes |
| M11#28:auc_audio_condition_recomputed | auc_audio_condition_recomputed | 0.7143 | 37 (37) | yes |

### PART 16 M3

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M3_pitt_clip | Omni zero shot pitt clip-level AUC | 0.6578 [0.6033, 0.7125] | 468 (228) | yes |
| M3_pcgita_clip | Omni zero shot pcgita clip-level AUC | 0.5635 [0.4931, 0.6331] | 1100 (100) | yes |
| M3_neurovoz_clip | Omni zero shot neurovoz clip-level AUC | 0.6951 [0.6356, 0.7519] | 1270 (107) | yes |

### PART 16 M4

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M4#01:yes_rate_Qwen2.5_Omni_conflict | yes_rate_Qwen2.5-Omni_conflict | 0.2236 [0.1792, 0.2736] | 483 (138) | yes |
| M4#03:yes_rate_Qwen2.5_Omni_agreement | yes_rate_Qwen2.5-Omni_agreement | 0.2919 [0.2073, 0.3771] | 483 (138) | yes |
| M4#05:yes_rate_diff_conflict_minus_agreement_Qwen2.5_Omni | yes_rate_diff_conflict_minus_agreement_Qwen2.5-Omni | -0.0683 [-0.1788, 0.0474] | 966 (138) | yes |
| M4#06:yes_rate_Qwen2_Audio_conflict | yes_rate_Qwen2-Audio_conflict | 0.9400 [0.9199, 0.9612] | 483 (138) | yes |
| M4#08:yes_rate_Qwen2_Audio_agreement | yes_rate_Qwen2-Audio_agreement | 0.8675 [0.8159, 0.9102] | 483 (138) | yes |
| M4#10:yes_rate_diff_conflict_minus_agreement_Qwen2_Audio | yes_rate_diff_conflict_minus_agreement_Qwen2-Audio | 0.0725 [0.0200, 0.1330] | 966 (138) | yes |
| M4#11:yes_rate_Qwen3_Omni_30B_A3B_conflict | yes_rate_Qwen3-Omni-30B-A3B_conflict | 0.3892 [0.3347, 0.4500] | 483 (138) | yes |
| M4#13:yes_rate_Qwen3_Omni_30B_A3B_agreement | yes_rate_Qwen3-Omni-30B-A3B_agreement | 0.4182 [0.3188, 0.5097] | 483 (138) | yes |
| M4#15:yes_rate_diff_conflict_minus_agreement_Qwen3_Omni_30B_A3B | yes_rate_diff_conflict_minus_agreement_Qwen3-Omni-30B-A3B | -0.0290 [-0.1540, 0.1078] | 966 (138) | yes |
| M4#16:yes_rate_AudioFlamingo2_conflict | yes_rate_AudioFlamingo2_conflict | 0.9710 [0.9502, 0.9887] | 483 (138) | yes |
| M4#18:yes_rate_AudioFlamingo2_agreement | yes_rate_AudioFlamingo2_agreement | 0.9834 [0.9629, 0.9979] | 483 (138) | yes |
| M4#20:yes_rate_diff_conflict_minus_agreement_AudioFlamingo2 | yes_rate_diff_conflict_minus_agreement_AudioFlamingo2 | -0.0124 [-0.0280, 0.0020] | 966 (138) | yes |
| M4#21:yes_rate_AudioFlamingo3_conflict | yes_rate_AudioFlamingo3_conflict | 0.7039 [0.6533, 0.7554] | 483 (138) | yes |
| M4#23:yes_rate_AudioFlamingo3_agreement | yes_rate_AudioFlamingo3_agreement | 0.6687 [0.5937, 0.7371] | 483 (138) | yes |
| M4#25:yes_rate_diff_conflict_minus_agreement_AudioFlamingo3 | yes_rate_diff_conflict_minus_agreement_AudioFlamingo3 | 0.0352 [-0.0491, 0.1281] | 966 (138) | yes |
| M4#26:yes_rate_Kimi_Audio_conflict | yes_rate_Kimi-Audio_conflict | 0.8364 [0.7976, 0.8727] | 483 (138) | yes |
| M4#28:yes_rate_Kimi_Audio_agreement | yes_rate_Kimi-Audio_agreement | 0.7578 [0.6869, 0.8221] | 483 (138) | yes |
| M4#30:yes_rate_diff_conflict_minus_agreement_Kimi_Audio | yes_rate_diff_conflict_minus_agreement_Kimi-Audio | 0.0787 [0.0160, 0.1461] | 966 (138) | yes |

### PART 16 M7

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M7_pitt_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, pitt | 0.6578 [0.6033, 0.7125] | 468 (228) | yes |
| M7_adresso_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, adresso | 0.6150 [0.5432, 0.6890] | 237 (237) | yes |
| M7_adresso_gap | paired probe minus zero-shot AUC, adresso | 0.2748 [0.1920, 0.3526] | 237 (237) | yes |
| M7_adress2020_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, adress2020 | 0.6241 [0.5348, 0.7148] | 156 (156) | yes |
| M7_adress2020_gap | paired probe minus zero-shot AUC, adress2020 | 0.1637 [0.0557, 0.2693] | 156 (156) | yes |
| M7_pcgita_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, pcgita | 0.5635 [0.4931, 0.6331] | 1100 (100) | yes |
| M7_pcgita_gap | paired probe minus zero-shot AUC, pcgita | 0.3454 [0.2636, 0.4275] | 1100 (100) | yes |
| M7_neurovoz_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, neurovoz | 0.6951 [0.6356, 0.7519] | 1270 (107) | yes |
| M7_neurovoz_gap | paired probe minus zero-shot AUC, neurovoz | 0.2209 [0.1660, 0.2783] | 1270 (107) | yes |
| M7_kcl_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, kcl | 0.7143 [0.5210, 0.8810] | 37 (37) | yes |
| M7_kcl_gap | paired probe minus zero-shot AUC, kcl | 0.0625 [-0.1813, 0.2900] | 37 (37) | yes |

### PART 16 M7fix

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M7fix#01:Qwen2.5_Omni_Pitt_encoder_probe_5_repeat_CORRECTED_source | Qwen2.5-Omni Pitt encoder probe, 5-repeat (CORRECTED source) | 0.7706 [0.7157, 0.8220] | 468 (228) | yes |
| M7fix#02:Qwen2.5_Omni_Pitt_paired_probe_minus_zero_shot_CORRECTED | Qwen2.5-Omni Pitt paired probe minus zero-shot (CORRECTED) | 0.1128 [0.0455, 0.1768] | 468 (228) | yes |

### PART 16 M8

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M8#01:Qwen2.5_Omni_A_edaic_part14_conflict_src_primary | Qwen2.5-Omni / A_edaic_part14 / conflict / src=primary | 0.2218 [0.1829, 0.2632] | 483 (138) | yes |
| M8#04:Qwen2_Audio_A_edaic_part14_conflict_src_primary | Qwen2-Audio / A_edaic_part14 / conflict / src=primary | 0.2280 [0.1816, 0.2782] | 483 (138) | yes |
| M8#07:Qwen3_Omni_30B_A3B_A_edaic_part14_conflict_src_primary | Qwen3-Omni-30B-A3B / A_edaic_part14 / conflict / src=primary | 0.3008 [0.2454, 0.3641] | 483 (138) | yes |
| M8#10:Audio_Flamingo_2_A_edaic_part14_conflict_src_primary | Audio Flamingo 2 / A_edaic_part14 / conflict / src=primary | 0.5286 [0.4512, 0.6015] | 483 (138) | yes |
| M8#13:Audio_Flamingo_3_A_edaic_part14_conflict_src_primary | Audio Flamingo 3 / A_edaic_part14 / conflict / src=primary | 0.4235 [0.3606, 0.4880] | 483 (138) | yes |
| M8#16:Kimi_Audio_A_edaic_part14_conflict_src_primary | Kimi-Audio / A_edaic_part14 / conflict / src=primary | 0.5258 [0.4501, 0.5977] | 483 (138) | yes |
| M8#19:Qwen2.5_Omni_B_pitt_conflict_src_primary | Qwen2.5-Omni / B_pitt / conflict / src=primary | 0.5322 [0.4216, 0.6447] | 146 (100) | yes |
| M8#20:Qwen2.5_Omni_B_pitt_agreement_src_primary | Qwen2.5-Omni / B_pitt / agreement / src=primary | 0.7054 [0.6431, 0.7658] | 322 (175) | yes |
| M8#21:Qwen2.5_Omni_B_pitt_conflict_minus_agreement_src_primary | Qwen2.5-Omni / B_pitt / conflict_minus_agreement / src=primary | -0.1731 [-0.2958, -0.0392] | 468 (228) | yes |
| M8#22:Qwen2_Audio_B_pitt_conflict_src_primary | Qwen2-Audio / B_pitt / conflict / src=primary | 0.4132 [0.3005, 0.5423] | 146 (100) | yes |
| M8#23:Qwen2_Audio_B_pitt_agreement_src_primary | Qwen2-Audio / B_pitt / agreement / src=primary | 0.6985 [0.6372, 0.7597] | 322 (175) | yes |
| M8#24:Qwen2_Audio_B_pitt_conflict_minus_agreement_src_primary | Qwen2-Audio / B_pitt / conflict_minus_agreement / src=primary | -0.2853 [-0.3950, -0.1485] | 468 (228) | yes |
| M8#25:Qwen3_Omni_30B_A3B_B_pitt_conflict_src_primary | Qwen3-Omni-30B-A3B / B_pitt / conflict / src=primary | 0.6066 [0.4956, 0.7123] | 146 (100) | yes |
| M8#26:Qwen3_Omni_30B_A3B_B_pitt_agreement_src_primary | Qwen3-Omni-30B-A3B / B_pitt / agreement / src=primary | 0.8231 [0.7664, 0.8720] | 322 (175) | yes |
| M8#27:Qwen3_Omni_30B_A3B_B_pitt_conflict_minus_agreement_src_primary | Qwen3-Omni-30B-A3B / B_pitt / conflict_minus_agreement / src=primary | -0.2165 [-0.3242, -0.1052] | 468 (228) | yes |
| M8#28:Audio_Flamingo_2_B_pitt_conflict_src_af2_pitt468 | Audio Flamingo 2 / B_pitt / conflict / src=af2_pitt468 | 0.6310 [0.5336, 0.7260] | 146 (100) | yes |
| M8#29:Audio_Flamingo_2_B_pitt_agreement_src_af2_pitt468 | Audio Flamingo 2 / B_pitt / agreement / src=af2_pitt468 | 0.5544 [0.4760, 0.6296] | 322 (175) | yes |
| M8#30:Audio_Flamingo_2_B_pitt_conflict_minus_agreement_src_af2_pitt468 | Audio Flamingo 2 / B_pitt / conflict_minus_agreement / src=af2_pitt468 | 0.0766 [-0.0307, 0.1869] | 468 (228) | yes |
| M8#34:Audio_Flamingo_3_B_pitt_conflict_src_primary | Audio Flamingo 3 / B_pitt / conflict / src=primary | 0.6169 [0.5188, 0.7209] | 146 (100) | yes |
| M8#35:Audio_Flamingo_3_B_pitt_agreement_src_primary | Audio Flamingo 3 / B_pitt / agreement / src=primary | 0.5822 [0.5080, 0.6528] | 322 (175) | yes |
| M8#36:Audio_Flamingo_3_B_pitt_conflict_minus_agreement_src_primary | Audio Flamingo 3 / B_pitt / conflict_minus_agreement / src=primary | 0.0347 [-0.0698, 0.1474] | 468 (228) | yes |
| M8#37:Kimi_Audio_B_pitt_conflict_src_primary | Kimi-Audio / B_pitt / conflict / src=primary | 0.7197 [0.6201, 0.8147] | 146 (100) | yes |
| M8#38:Kimi_Audio_B_pitt_agreement_src_primary | Kimi-Audio / B_pitt / agreement / src=primary | 0.7227 [0.6560, 0.7896] | 322 (175) | yes |
| M8#39:Kimi_Audio_B_pitt_conflict_minus_agreement_src_primary | Kimi-Audio / B_pitt / conflict_minus_agreement / src=primary | -0.0030 [-0.1134, 0.1065] | 468 (228) | yes |

### PART 16 MEDAIC

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| MEDAIC_full_answer | Omni zero-shot answer AUC, full window | 0.8285 [0.7708, 0.8827] | 275 (275) | yes |
| MEDAIC_full_sft | projector fine-tune OOF AUC, full window | 0.6421 [0.5593, 0.7188] | 275 (275) | yes |
| MEDAIC_full_sft_minus_answer | PAIRED projector fine-tune OOF minus answer, full window (excl0=True) | -0.1863 [-0.2696, -0.1098] | 275 (275) | yes |
| MEDAIC_full_text | transcript-only readout AUC, full window | 0.8230 [0.7651, 0.8705] | 275 (275) | yes |

### PART 16 POD1

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| 1a_o25_text_conflict_auc | text-prompt conflict-arm AUC, o25 | 0.1498 [0.1131, 0.1887] | 483 (138) | yes |
| 1a_o25_pearson_text_audio | Pearson r transcript p_yes vs audio p_yes, o25 | 0.8442 | 966 (138) | yes |
| 1a_o25_same_decision | share text decision == audio decision at 0.5, o25 | 0.8634 | 966 (138) | yes |
| 1a_q2a_text_conflict_auc | text-prompt conflict-arm AUC, q2a | 0.2359 [0.1851, 0.2886] | 483 (138) | yes |
| 1a_q2a_pearson_text_audio | Pearson r transcript p_yes vs audio p_yes, q2a | 0.5155 | 966 (138) | yes |
| 1a_q2a_same_decision | share text decision == audio decision at 0.5, q2a | 0.7122 | 966 (138) | yes |
| 1f_share_yesno_all | share first generated word is Yes/No, all clips | 1 | 966 (138) | yes |
| 1f_agree_gen_logit_all | agreement generated word vs logit decision, Yes/No clips | 0.9865 | 966 (138) | yes |
| 1f_share_yesno_conflict | share first generated word is Yes/No, conflict arm | 1 | 483 (138) | yes |
| 1f_agree_gen_logit_conflict | agreement generated word vs logit decision, conflict arm | 0.9917 | 483 (138) | yes |
| 1f_share_yesno_agreement | share first generated word is Yes/No, agreement arm | 1 | 483 (138) | yes |
| 1f_agree_gen_logit_agreement | agreement generated word vs logit decision, agreement arm | 0.9814 | 483 (138) | yes |
| VALIDATION_conflict_auc | pipeline validation, rebuilt 0.70 set, original wording (reproduces Table 2 audio) conflict arm AUC | 0.2218 [0.1829, 0.2632] | 483 (138) | yes |
| 1d_conflict_auc | 1d PHQ-8>=10 cutoff rebuild, Omni audio conflict arm AUC | 0.1759 [0.1448, 0.2095] | 924 (237) | yes |
| 1d_agreement_auc | 1d PHQ-8>=10 cutoff rebuild, Omni audio agreement arm AUC | 0.9311 [0.9028, 0.9568] | 924 (237) | yes |
| 1e060_conflict_auc | 1e sentiment threshold 0.60, Omni audio conflict arm AUC | 0.2619 [0.2188, 0.3075] | 646 (147) | yes |
| 1e060_agreement_auc | 1e sentiment threshold 0.60, Omni audio agreement arm AUC | 0.9406 [0.9026, 0.9711] | 646 (147) | yes |
| 1e080_conflict_auc | 1e sentiment threshold 0.80, Omni audio conflict arm AUC | 0.1936 [0.1501, 0.2438] | 281 (104) | yes |
| 1e080_agreement_auc | 1e sentiment threshold 0.80, Omni audio agreement arm AUC | 0.9594 [0.9190, 0.9895] | 281 (104) | yes |
| 1b_p2_conflict_auc | 1b prompt wording 2, Omni audio conflict arm AUC | 0.3439 [0.2900, 0.3979] | 483 (138) | yes |
| 1b_p2_agreement_auc | 1b prompt wording 2, Omni audio agreement arm AUC | 0.9119 [0.8505, 0.9594] | 483 (138) | yes |
| 1b_p3_conflict_auc | 1b prompt wording 3, Omni audio conflict arm AUC | 0.2392 [0.1940, 0.2865] | 483 (138) | yes |
| 1b_p3_agreement_auc | 1b prompt wording 3, Omni audio agreement arm AUC | 0.9442 [0.8926, 0.9821] | 483 (138) | yes |
| 1b_p2_conflict_delta | 1b wording 2 conflict AUC change vs original wording | 0.1221 | 483 (138) | yes |
| 1b_p2_agreement_delta | 1b wording 2 agreement AUC change vs original wording | -0.0363 | 483 (138) | yes |
| 1b_p3_conflict_delta | 1b wording 3 conflict AUC change vs original wording | 0.0174 | 483 (138) | yes |
| 1b_p3_agreement_delta | 1b wording 3 agreement AUC change vs original wording | -0.0040 | 483 (138) | yes |
| 1b_max_abs_change | 1b maximum absolute AUC change across both wordings and both arms | 0.1221 | 966 (138) | yes |
| 1e_conflict_auc_range | 1e conflict AUC range across sentiment 0.60/0.70/0.80 | 0.0683 | 0 (0) | yes |

### PART 16 POD2

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| P16.POD2.V1 | Qwen2.5-Omni Pitt projector-FT OOF AUC (recomputed from paper file, rank formula) | 0.7673 [0.7086, 0.8231] | 468 (228) | yes |
| P16.POD2.V2 | Qwen2.5-Omni Pitt zero-shot AUC (recomputed from paper file, rank formula) | 0.6578 [0.6033, 0.7125] | 468 (228) | yes |
| P16.POD2.V3 | projector-FT conflict-arm AUC (recomputed) | 0.5189 [0.4151, 0.6300] | 146 (100) | yes |
| P16.POD2.V4 | projector-FT agreement-arm AUC (recomputed) | 0.8593 [0.8063, 0.9069] | 322 (175) | yes |
| P16.POD2.P1 | LoRA r8 a16 d0.05 on q,k,v,o of all 28 LM layers: trainable params | 5046272 | 468 (228) | yes |
| P16.POD2.P2 | LoRA trainable as % of full Qwen2.5-Omni-7B (10732225408 params) | 0.0470 | 468 (228) | yes |
| P16.POD2.L1 | POD2 fresh LoRA(r8,a16,d0.05,qkvo all 28 LM layers) Qwen2.5-Omni Pitt OOF AUC | 0.8250 [0.7735, 0.8729] | 468 (228) | yes |
| P16.POD2.L2 | POD2 fresh LoRA OOF AUC, conflict arm | 0.5520 [0.4348, 0.6833] | 146 (100) | yes |
| P16.POD2.L3 | POD2 fresh LoRA OOF AUC, agreement arm | 0.9199 [0.8846, 0.9513] | 322 (175) | yes |
| P16.POD2.L4 | POD2 fresh LoRA paired conflict-minus-agreement | -0.3679 [-0.4861, -0.2433] | 468 (228) | yes |
| P16.POD2.D1 | paired delta: LoRA OOF AUC minus projector-FT OOF AUC (same 468 clips, same speakers) | 0.0577 [0.0154, 0.1044] | 468 (228) | yes |
| P16.POD2.D2 | paired delta: LoRA OOF AUC minus zero-shot AUC | 0.1672 [0.1071, 0.2264] | 468 (228) | yes |
| P16.POD2.D3 | paired delta: projector-FT AUC minus zero-shot AUC (both read-only paper files) | 0.1095 [0.0454, 0.1728] | 468 (228) | yes |

### PART 16 POD3

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| 3b_all_cos | Q3O Pitt readout-direction cosine, all clips (random floor 0.0177 [0.0008,0.0498]) | 0.0051 | 468 (228) | yes |
| 3b_all_auc_probe | Q3O Pitt answer-state probe AUC, all clips | 0.8307 [0.7833, 0.8741] | 468 (228) | yes |
| 3b_all_auc_d | Q3O Pitt AUC of the readout direction alone, all clips | 0.7656 [0.7088, 0.8179] | 468 (228) | yes |
| 3a_pitt_enc_nested | Q3O Pitt ENCODER nested probe AUC (5-repeat mean, layer chosen inside the training folds) | 0.8122 [0.7613, 0.8625] | 468 (228) | yes |
| 3a_pcgita_zeroshot_shipped | Q3O PC-GITA zero-shot answer AUC, SHIPPED file recomputed here | 0.5967 | 1100 (100) | yes |
| 3a_pcgita_enc_nested | Q3O PC-GITA ENCODER nested probe AUC (5-repeat mean, layer chosen inside the training folds) | 0.8969 [0.8322, 0.9454] | 1100 (100) | yes |

### PART 16 POD4

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| POD4_4b_pub_zs | Pitt published Omni zero-shot AUC, recomputed rank formula | 0.6578 [0.6033, 0.7125] | 468 (228) | yes |
| POD4_4a_nv_5f_before | NeuroVoz five quiet-frame channel features together, GroupKFold5-by-speaker probe AUC, BEFORE | 0.7977 [0.7248, 0.8611] | 1270 (107) | yes |
| POD4_4a_nv_5f_after | NeuroVoz five channel features together, AFTER -23 LUFS + shaped noise | 0.6025 [0.5479, 0.6580] | 1270 (107) | yes |
| POD4_4a_nv_accept | NeuroVoz acceptance test five-feature AUC below 0.60 after normalisation | FAILED_0.6025 | 1270 (107) | yes |
| POD4_4a_nv_cen_before | NeuroVoz spectral centroid alone, direction-free AUC, BEFORE | 0.7782 [0.7109, 0.8396] | 1270 (107) | yes |
| POD4_4a_nv_zs_before | NeuroVoz Omni zero-shot AUC, BEFORE normalisation, POD4 re-run | 0.7108 [0.6523, 0.7671] | 1270 (107) | yes |
| POD4_4a_nv_enc_before | NeuroVoz Omni encoder nested probe AUC, BEFORE normalisation | 0.9233 [0.8800, 0.9573] | 1270 (107) | yes |
| POD4_4a_pg_5f_before | PC-GITA five quiet-frame channel features together, GroupKFold5-by-speaker probe AUC, BEFORE | 0.6657 [0.5851, 0.7452] | 1100 (100) | yes |
| POD4_4a_pg_5f_after | PC-GITA five channel features together, AFTER -23 LUFS + shaped noise | 0.5590 [0.4922, 0.6213] | 1100 (100) | yes |
| POD4_4a_pg_accept | PC-GITA acceptance test five-feature AUC below 0.60 after normalisation | PASSED_0.5590 | 1100 (100) | yes |
| POD4_4a_nv_zs_after | NeuroVoz Omni zero-shot AUC, AFTER normalisation | 0.6991 [0.6348, 0.7598] | 1270 (107) | yes |
| POD4_4a_nv_enc_after | NeuroVoz Omni encoder nested probe AUC, AFTER normalisation | 0.8924 [0.8481, 0.9323] | 1270 (107) | yes |
| POD4_4a_nv_pd_zs | NeuroVoz PAIRED zero-shot delta, before to after normalisation | -0.0117 [-0.0332, 0.0098] | 1270 (107) | yes |
| POD4_4a_nv_pd_enc | NeuroVoz PAIRED encoder probe delta, before to after normalisation | -0.0308 [-0.0608, -0.0012] | 1270 (107) | yes |
| POD4_4a_pg_enc_before | PC-GITA Omni encoder nested probe AUC, BEFORE normalisation | 0.8959 [0.8339, 0.9448] | 1100 (100) | yes |
| POD4_4a_pg_zs_before | PC-GITA Omni zero-shot AUC, BEFORE normalisation | 0.5728 [0.5002, 0.6440] | 1100 (100) | yes |
| POD4_4a_pg_zs_after | PC-GITA Omni zero-shot AUC, AFTER normalisation | 0.5674 [0.5054, 0.6304] | 1100 (100) | yes |
| POD4_4a_pg_enc_after | PC-GITA Omni encoder nested probe AUC, AFTER normalisation | 0.8298 [0.7546, 0.8968] | 1100 (100) | yes |
| POD4_4a_pg_pd_zs | PC-GITA PAIRED zero-shot delta, before to after normalisation | -0.0054 [-0.0437, 0.0357] | 1100 (100) | yes |
| POD4_4a_pg_pd_enc | PC-GITA PAIRED encoder probe delta, before to after normalisation | -0.0661 [-0.1181, -0.0180] | 1100 (100) | yes |

### PART 16 POD4B

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| 4b_adress2020_published_zs | adress2020 arm published Omni zero-shot AUC | 0.6241 [0.5348, 0.7148] | 156 (156) | yes |

### PART 16 Q3Ofix

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| 3b_fix#01:Qwen3_Omni_Pitt_all_probe | Qwen3-Omni Pitt all: probe | 0.8307 [0.7833, 0.8741] | 468 (228) | yes |
| 3b_fix#02:Qwen3_Omni_Pitt_all_along_readout_direction | Qwen3-Omni Pitt all: along readout direction | 0.7656 [0.7088, 0.8179] | 468 (228) | yes |
| 3b_fix#03:Qwen3_Omni_Pitt_all_probe_minus_readout_dir_CANONICAL | Qwen3-Omni Pitt all: probe minus readout dir (CANONICAL) | 0.8276 [0.7795, 0.8696] | 468 (228) | yes |
| 3b_fix#04:Qwen3_Omni_Pitt_conflict_probe | Qwen3-Omni Pitt conflict: probe | 0.6394 [0.5323, 0.7418] | 146 (100) | yes |
| 3b_fix#05:Qwen3_Omni_Pitt_conflict_along_readout_direction | Qwen3-Omni Pitt conflict: along readout direction | 0.6029 [0.4965, 0.7025] | 146 (100) | yes |
| 3b_fix#06:Qwen3_Omni_Pitt_conflict_probe_minus_readout_dir_CANONICAL | Qwen3-Omni Pitt conflict: probe minus readout dir (CANONICAL) | 0.6532 [0.5483, 0.7527] | 146 (100) | yes |
| 3b_fix#07:Qwen3_Omni_Pitt_agreement_probe | Qwen3-Omni Pitt agreement: probe | 0.9032 [0.8626, 0.9374] | 322 (175) | yes |
| 3b_fix#08:Qwen3_Omni_Pitt_agreement_along_readout_direction | Qwen3-Omni Pitt agreement: along readout direction | 0.8264 [0.7741, 0.8731] | 322 (175) | yes |
| 3b_fix#09:Qwen3_Omni_Pitt_agreement_probe_minus_readout_dir_CANONICAL | Qwen3-Omni Pitt agreement: probe minus readout dir (CANONICAL) | 0.8937 [0.8522, 0.9312] | 322 (175) | yes |

### PART 16 SEED

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| SEED02_1a | Omni transcript-only, conflict arm | 0.1498 [0.1131, 0.1887] | 483 (138) | yes |
| SEED05_1a | Qwen2-Audio transcript-only, conflict arm | 0.2359 [0.1851, 0.2886] | 483 (138) | yes |
| SEED07_1a | Omni text vs audio Pearson r | 0.8442 | 966 (138) | yes |
| SEED08_1a | Omni text vs audio same Yes/No decision | 0.8634 | 966 (138) | yes |
| SEED09_3a | Qwen3-Omni Pitt encoder, nested, 5 repeats (per-repeat [0.8199,0.8016,0.8142,0.8342,0.7914]) | 0.8122 | 468 (228) | yes |
| SEED14_3b | Qwen3-Omni Pitt cos(probe dir, Yes-No unembed) (random floor 0.0177 [0.0008, 0.0498]) | 0.0051 | 468 (228) | yes |
| SEED15_3b | Qwen3-Omni Pitt probe AUC | 0.8307 | 468 (228) | yes |
| SEED16_3b | Qwen3-Omni Pitt AUC along readout direction | 0.7656 | 468 (228) | yes |

### PART 17 T1

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T1_o25_conflict | E-DAIC Table 2 Qwen2.5-Omni audio conflict | 0.2218 [0.1829, 0.2632] | 483 (138) | yes |
| T1_o25_agreement_NEW_311distinct | E-DAIC Table 2 Qwen2.5-Omni audio agreement NEW 311 distinct | 0.9647 [0.9376, 0.9857] | 311 (138) | yes |
| T1_o25_paired_conflict_minus_agreement_NEW_311distinct | E-DAIC Table 2 Qwen2.5-Omni audio SUPPLEMENTARY paired conflict minus agreement, distinct | -0.7429 [-0.7869, -0.6944] | 794 (138) | yes |
| T1_q2a_conflict | E-DAIC Table 2 Qwen2-Audio audio conflict | 0.2280 [0.1816, 0.2782] | 483 (138) | yes |
| T1_q2a_agreement_NEW_311distinct | E-DAIC Table 2 Qwen2-Audio audio agreement NEW 311 distinct | 0.9608 [0.9351, 0.9814] | 311 (138) | yes |
| T1_q2a_paired_conflict_minus_agreement_NEW_311distinct | E-DAIC Table 2 Qwen2-Audio audio SUPPLEMENTARY paired conflict minus agreement, distinct | -0.7328 [-0.7780, -0.6837] | 794 (138) | yes |
| T1_q3o_conflict | E-DAIC Table 2 Qwen3-Omni-30B-A3B audio conflict | 0.3008 [0.2454, 0.3641] | 483 (138) | yes |
| T1_q3o_agreement_NEW_311distinct | E-DAIC Table 2 Qwen3-Omni-30B-A3B audio agreement NEW 311 distinct | 0.9727 [0.9507, 0.9904] | 311 (138) | yes |
| T1_q3o_paired_conflict_minus_agreement_NEW_311distinct | E-DAIC Table 2 Qwen3-Omni-30B-A3B audio SUPPLEMENTARY paired conflict minus agreement, distinct | -0.6719 [-0.7266, -0.6105] | 794 (138) | yes |
| T1_af2_conflict | E-DAIC Table 2 Audio Flamingo 2 audio conflict | 0.5286 [0.4512, 0.6015] | 483 (138) | yes |
| T1_af2_agreement_NEW_311distinct | E-DAIC Table 2 Audio Flamingo 2 audio agreement NEW 311 distinct | 0.6297 [0.5451, 0.7087] | 311 (138) | yes |
| T1_af2_paired_conflict_minus_agreement_NEW_311distinct | E-DAIC Table 2 Audio Flamingo 2 audio SUPPLEMENTARY paired conflict minus agreement, distinct | -0.1011 [-0.1647, -0.0384] | 794 (138) | yes |
| T1_af3_conflict | E-DAIC Table 2 Audio Flamingo 3 audio conflict | 0.4235 [0.3606, 0.4880] | 483 (138) | yes |
| T1_af3_agreement_NEW_311distinct | E-DAIC Table 2 Audio Flamingo 3 audio agreement NEW 311 distinct | 0.8425 [0.7963, 0.8854] | 311 (138) | yes |
| T1_af3_paired_conflict_minus_agreement_NEW_311distinct | E-DAIC Table 2 Audio Flamingo 3 audio SUPPLEMENTARY paired conflict minus agreement, distinct | -0.4190 [-0.4878, -0.3461] | 794 (138) | yes |
| T1_kimi_conflict | E-DAIC Table 2 Kimi-Audio audio conflict | 0.5258 [0.4501, 0.5977] | 483 (138) | yes |
| T1_kimi_agreement_NEW_311distinct | E-DAIC Table 2 Kimi-Audio audio agreement NEW 311 distinct | 0.8936 [0.8479, 0.9329] | 311 (138) | yes |
| T1_kimi_paired_conflict_minus_agreement_NEW_311distinct | E-DAIC Table 2 Kimi-Audio audio SUPPLEMENTARY paired conflict minus agreement, distinct | -0.3678 [-0.4394, -0.3020] | 794 (138) | yes |
| T1_o25t_conflict | E-DAIC Table 2 Qwen2.5-Omni transcript conflict | 0.1498 [0.1131, 0.1887] | 483 (138) | yes |
| T1_o25t_agreement_NEW_311distinct | E-DAIC Table 2 Qwen2.5-Omni transcript agreement NEW 311 distinct | 0.9620 [0.9345, 0.9822] | 311 (138) | yes |
| T1_q2at_conflict | E-DAIC Table 2 Qwen2-Audio transcript conflict | 0.2359 [0.1851, 0.2886] | 483 (138) | yes |
| T1_q2at_agreement_NEW_311distinct | E-DAIC Table 2 Qwen2-Audio transcript agreement NEW 311 distinct | 0.9199 [0.8791, 0.9560] | 311 (138) | yes |

### PART 17 T3

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T3_all_cos | Qwen2-Audio Pitt all: cos(probe direction, Yes-minus-No readout direction) [full fit; used=yes] | 0.0145 | 468 (228) | yes |
| T3_floor_vs_d | Qwen2-Audio Pitt all: random floor mean/cos/ vs readout direction d [fresh default_rng(0), 2000 dirs; used=yes | 0.0123 [0.0005, 0.0321] | 468 (228) | yes |
| T3_all_probe | Qwen2-Audio Pitt all: probe AUC out of fold [full 468 OOF; used=yes] | 0.8223 [0.7742, 0.8671] | 468 (228) | yes |
| T3_all_proj_d | Qwen2-Audio Pitt all: AUC along readout direction (X.d) [full 468 OOF; used=yes] | 0.6174 [0.5539, 0.6800] | 468 (228) | yes |
| T3_all_perp_z | Qwen2-Audio Pitt all: probe AUC with readout direction projected out (within-fold z) [full 468 OOF; used=yes] | 0.8114 [0.7646, 0.8577] | 468 (228) | yes |
| T3_conflict_probe | Qwen2-Audio Pitt conflict: probe AUC out of fold [full 468 OOF restricted to arm (DECIDED); used=yes] | 0.6178 [0.5123, 0.7222] | 146 (100) | yes |
| T3_conflict_proj_d | Qwen2-Audio Pitt conflict: AUC along readout direction (X.d) [full 468 OOF restricted to arm (DECIDED); used=y | 0.4136 [0.3010, 0.5424] | 146 (100) | yes |
| T3_conflict_perp_z | Qwen2-Audio Pitt conflict: probe AUC with readout direction projected out (within-fold z) [full 468 OOF restri | 0.6013 [0.4929, 0.7114] | 146 (100) | yes |
| T3_agreement_probe | Qwen2-Audio Pitt agreement: probe AUC out of fold [full 468 OOF restricted to arm (DECIDED); used=yes] | 0.8988 [0.8598, 0.9328] | 322 (175) | yes |
| T3_agreement_proj_d | Qwen2-Audio Pitt agreement: AUC along readout direction (X.d) [full 468 OOF restricted to arm (DECIDED); used= | 0.6984 [0.6373, 0.7592] | 322 (175) | yes |
| T3_agreement_perp_z | Qwen2-Audio Pitt agreement: probe AUC with readout direction projected out (within-fold z) [full 468 OOF restr | 0.8897 [0.8486, 0.9261] | 322 (175) | yes |
| T3_paired_perp_z | Qwen2-Audio Pitt conflict-agreement: paired difference, perp_z [full 468 OOF restricted to arm (DECIDED), pair | -0.2884 [-0.3884, -0.1815] | 468 (228) | yes |

### PART 17 T4

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T4_pitt_o25_enc_mean5 | Qwen2.5-Omni Pitt encoder probe, mean of five per-repeat AUCs | 0.7706 [0.7157, 0.8220] | 468 (228) | yes |
| T4_edaic_o25_llm_mean5_jsononly | Qwen2.5-Omni E-DAIC 300 s window LM probe, mean of five per-repeat AUCs (json per_repeat list; the per-clip re | 0.7001 [0.6231, 0.7727] | 275 (275) | yes |
| T4_edaic_o25_enc_mean5_jsononly | Qwen2.5-Omni E-DAIC 300 s window encoder probe, mean of five per-repeat AUCs (json per_repeat list; the per-cl | 0.5884 [0.5268, 0.6523] | 275 (275) | yes |
| T4_pcgita_q3o_enc_mean5 | Qwen3-Omni PC-GITA encoder probe, mean of five per-repeat AUCs | 0.8969 [0.8322, 0.9454] | 1100 (100) | yes |
| T4_pitt_q3o_enc_mean5 | Qwen3-Omni Pitt encoder probe, mean of five per-repeat AUCs | 0.8122 [0.7613, 0.8625] | 468 (228) | yes |
| T4_pitt_kimi_conflict | Kimi-Audio Pitt conflict zero-shot answer AUC | 0.7197 [0.6201, 0.8147] | 146 (100) | yes |
| T4_pitt_kimi_agreement | Kimi-Audio Pitt agreement zero-shot answer AUC | 0.7227 [0.6560, 0.7896] | 322 (175) | yes |
| T4_pitt_af2_conflict | Audio Flamingo 2 Pitt conflict zero-shot answer AUC | 0.6310 [0.5336, 0.7260] | 146 (100) | yes |
| T4_pitt_af2_agreement | Audio Flamingo 2 Pitt agreement zero-shot answer AUC | 0.5544 [0.4760, 0.6296] | 322 (175) | yes |
| T4_pitt_q3o_conflict_fullfit | Qwen3-Omni Pitt conflict arm answer-state probe, full-fit probe restricted to the arm | 0.6394 [0.5326, 0.7451] | 146 (100) | yes |
| T4_pitt_q3o_agreement_fullfit | Qwen3-Omni Pitt agreement arm answer-state probe, full-fit probe restricted to the arm | 0.9032 [0.8627, 0.9400] | 322 (175) | yes |
| T4_pitt_o25_lora | Qwen2.5-Omni Pitt LoRA fine tune out-of-fold AUC (125 tied values, ties averaged) | 0.8250 [0.7735, 0.8729] | 468 (228) | yes |
| T4_pitt_o25_lora_plus_proj | Qwen2.5-Omni Pitt LoRA plus projector fine tune out-of-fold AUC (114 tied values, ties averaged) | 0.7696 [0.7135, 0.8253] | 468 (228) | yes |

### PART 17 T5

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T5_pitt_answer | Pitt Qwen2.5-Omni zero-shot answer AUC (rank formula, speaker bootstrap) | 0.6578 [0.6033, 0.7125] | 468 (228) | yes |
| T5_adresso_answer | ADReSSo Qwen2.5-Omni zero-shot answer AUC (rank formula, speaker bootstrap) | 0.6150 [0.5432, 0.6890] | 237 (237) | yes |
| T5_adresso_layers_above | ADReSSo encoder layers (of 32) with probe auc_oof > answer AUC | 32 | 237 (237) | yes |
| T5_adresso_layers_strict | ADReSSo encoder layers (of 32) with auc_mean - auc_std > answer AUC | 32 | 237 (237) | yes |
| T5_adresso_layers_above_ci | ADReSSo encoder layers (of 32) with probe auc_oof > answer AUC upper 97.5 bound | 32 | 237 (237) | yes |
| T5_adresso_probe_max | ADReSSo encoder probe max auc_oof (layer 27) | 0.9030 | 237 (237) | yes |
| T5_adress2020_answer | ADReSS-2020 Qwen2.5-Omni zero-shot answer AUC (rank formula, speaker bootstrap) | 0.6241 [0.5348, 0.7148] | 156 (156) | yes |
| T5_adress2020_layers_above | ADReSS-2020 encoder layers (of 32) with probe auc_oof > answer AUC | 32 | 156 (156) | yes |
| T5_adress2020_layers_strict | ADReSS-2020 encoder layers (of 32) with auc_mean - auc_std > answer AUC | 26 | 156 (156) | yes |
| T5_adress2020_layers_above_ci | ADReSS-2020 encoder layers (of 32) with probe auc_oof > answer AUC upper 97.5 bound | 24 | 156 (156) | yes |
| T5_adress2020_probe_max | ADReSS-2020 encoder probe max auc_oof (layer 25) | 0.8248 | 156 (156) | yes |
| T5_pcgita_answer | PC-GITA Qwen2.5-Omni zero-shot answer AUC (rank formula, speaker bootstrap) | 0.5635 [0.4931, 0.6331] | 1100 (100) | yes |
| T5_neurovoz_answer | NeuroVoz Qwen2.5-Omni zero-shot answer AUC (rank formula, speaker bootstrap) | 0.6951 [0.6356, 0.7519] | 1270 (107) | yes |
| T5_neurovoz_layers_above | NeuroVoz encoder layers (of 32) with probe auc_oof > answer AUC | 32 | 1270 (107) | yes |
| T5_neurovoz_layers_strict | NeuroVoz encoder layers (of 32) with auc_mean - auc_std > answer AUC | 32 | 1270 (107) | yes |
| T5_neurovoz_layers_above_ci | NeuroVoz encoder layers (of 32) with probe auc_oof > answer AUC upper 97.5 bound | 32 | 1270 (107) | yes |
| T5_neurovoz_probe_max | NeuroVoz encoder probe max auc_oof (layer 25) | 0.9335 | 1270 (107) | yes |
| T5_kcl_answer | MDVR-KCL Qwen2.5-Omni zero-shot answer AUC (rank formula, speaker bootstrap) | 0.7143 [0.5210, 0.8810] | 37 (37) | yes |
| T5_kcl_layers_above | MDVR-KCL encoder layers (of 32) with probe auc_oof > answer AUC | 32 | 37 (37) | yes |
| T5_kcl_layers_strict | MDVR-KCL encoder layers (of 32) with auc_mean - auc_std > answer AUC | 15 | 37 (37) | yes |
| T5_kcl_layers_above_ci | MDVR-KCL encoder layers (of 32) with probe auc_oof > answer AUC upper 97.5 bound | 0 | 37 (37) | yes |
| T5_kcl_probe_max | MDVR-KCL encoder probe max auc_oof (layer 19) | 0.8482 | 37 (37) | yes |
| T5_edaic_full_answer | E-DAIC (full, 900 s cap) Qwen2.5-Omni zero-shot answer AUC (rank formula, speaker bootstrap) | 0.8285 [0.7708, 0.8827] | 275 (275) | yes |

### PART 17 T6

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T6_s20_text_all468 | Pitt Qwen2.5-Omni transcript-only answer AUC, all 468 (Sep 20 run) | 0.6629 [0.5983, 0.7245] | 468 (228) | yes |
| T6_s20_text_conflict | Pitt Qwen2.5-Omni transcript-only answer AUC, conflict arm (Sep 20 run) | 0.5560 [0.4516, 0.6680] | 146 (100) | yes |
| T6_s20_text_agreement | Pitt Qwen2.5-Omni transcript-only answer AUC, agreement arm (Sep 20 run) | 0.7078 [0.6381, 0.7696] | 322 (175) | yes |
| T6_s20_text_conflict_minus_agreement | Pitt Qwen2.5-Omni transcript-only PAIRED conflict minus agreement AUC (Sep 20 run) | -0.1518 [-0.2665, -0.0271] | 468 (228) | yes |
| T6_audio_all468 | Pitt Qwen2.5-Omni audio zero-shot answer AUC, all 468 | 0.6578 [0.6033, 0.7125] | 468 (228) | yes |
| T6_audio_conflict | Pitt Qwen2.5-Omni audio zero-shot answer AUC, conflict arm | 0.5322 [0.4216, 0.6447] | 146 (100) | yes |
| T6_audio_agreement | Pitt Qwen2.5-Omni audio zero-shot answer AUC, agreement arm | 0.7054 [0.6431, 0.7658] | 322 (175) | yes |
| T6_audio_conflict_minus_agreement | Pitt Qwen2.5-Omni audio zero-shot PAIRED conflict minus agreement AUC | -0.1731 [-0.2958, -0.0392] | 468 (228) | yes |

### PART 17 T7

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T7_zeroshot_answer | Qwen2.5-Omni zero-shot answer AUC, E-DAIC 300 s window (audio truncated at 300 s by Qwen2_5OmniProcessor; 217/ | 0.8278 [0.7703, 0.8822] | 275 (275) | yes |
| T7_published_llm_meanof5 | LM probe, mean of five per-repeat AUCs (pod run, published [0.6918, 0.7062, 0.6893, 0.7055, 0.7078]); interval | 0.7001 [0.6231, 0.7727] | 275 (275) | yes |
| T7_published_enc_meanof5 | encoder probe, mean of five per-repeat AUCs (pod run, published [0.5964, 0.5826, 0.5962, 0.5608, 0.606]); inte | 0.5884 [0.5268, 0.6523] | 275 (275) | yes |

### PART 17 T7b

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T7b_zeroshot_answer | Qwen2.5-Omni zero-shot answer AUC, E-DAIC 300 s window | 0.8278 [0.7703, 0.8822] | 275 (275) | yes |
| T7b_llm_meanof5 | LM probe, mean of five per-repeat AUCs [0.6918, 0.7062, 0.6893, 0.7055, 0.7078] (nested GroupKFold(5) x 5 repe | 0.7001 [0.6231, 0.7727] | 275 (275) | yes |
| T7b_llm_meanof5_minus_answer | LM probe mean of five per-repeat AUCs minus zero-shot answer AUC, paired (one speaker draw per replicate, all  | -0.1276 [-0.1993, -0.0629] | 275 (275) | yes |
| T7b_enc_meanof5 | encoder probe, mean of five per-repeat AUCs [0.5964, 0.5826, 0.5962, 0.5608, 0.606] (nested GroupKFold(5) x 5  | 0.5884 [0.5268, 0.6523] | 275 (275) | yes |
| T7b_enc_meanof5_minus_answer | encoder probe mean of five per-repeat AUCs minus zero-shot answer AUC, paired (one speaker draw per replicate, | -0.2394 [-0.3188, -0.1572] | 275 (275) | yes |

### PART 20 POD1

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| POD1_std_all | standard projector fine-tune AUC (beside), all 468, p_yes | 0.7673 [0.7086, 0.8231] | 468 (228) | yes |
| POD1_std_conflict | standard projector fine-tune AUC (beside), conflict 146, p_yes | 0.5189 [0.4151, 0.6300] | 146 (100) | yes |
| POD1_std_agreement | standard projector fine-tune AUC (beside), agreement 322, p_yes | 0.8593 [0.8063, 0.9069] | 322 (175) | yes |
| POD1_f3rerun_ref_all | standard fine-tune (paper file) AUC (beside), all 468, p_yes | 0.7673 [0.7086, 0.8231] | 468 (228) | yes |
| POD1_f3rerun_ref_conflict | standard fine-tune (paper file) AUC (beside), conflict 146, p_yes | 0.5189 [0.4151, 0.6300] | 146 (100) | yes |
| POD1_f3rerun_ref_agreement | standard fine-tune (paper file) AUC (beside), agreement 322, p_yes | 0.8593 [0.8063, 0.9069] | 322 (175) | yes |
| POD1_ctrl_ref_all | standard fine-tune (paper file) AUC (beside), all 468, p_yes | 0.7673 [0.7086, 0.8231] | 468 (228) | yes |
| POD1_ctrl_ref_conflict | standard fine-tune (paper file) AUC (beside), conflict 146, p_yes | 0.5189 [0.4151, 0.6300] | 146 (100) | yes |
| POD1_ctrl_ref_agreement | standard fine-tune (paper file) AUC (beside), agreement 322, p_yes | 0.8593 [0.8063, 0.9069] | 322 (175) | yes |
| POD1_bal_r2_ref_all | standard fine-tune (paper file) AUC (beside), all 468, p_yes | 0.7673 [0.7086, 0.8231] | 468 (228) | yes |
| POD1_bal_r2_ref_conflict | standard fine-tune (paper file) AUC (beside), conflict 146, p_yes | 0.5189 [0.4151, 0.6300] | 146 (100) | yes |
| POD1_bal_r2_ref_agreement | standard fine-tune (paper file) AUC (beside), agreement 322, p_yes | 0.8593 [0.8063, 0.9069] | 322 (175) | yes |
| POD1_ctrl_r2_ref_all | standard fine-tune (paper file) AUC (beside), all 468, p_yes | 0.7673 [0.7086, 0.8231] | 468 (228) | yes |
| POD1_ctrl_r2_ref_conflict | standard fine-tune (paper file) AUC (beside), conflict 146, p_yes | 0.5189 [0.4151, 0.6300] | 146 (100) | yes |
| POD1_ctrl_r2_ref_agreement | standard fine-tune (paper file) AUC (beside), agreement 322, p_yes | 0.8593 [0.8063, 0.9069] | 322 (175) | yes |

### PART 20 POD3c

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| POD3c_q3o_audio_conflict483 | Qwen3-Omni audio AUC, conflict483 | 0.3008 [0.2454, 0.3641] | 483 (138) | yes |
| POD3c_q3o_audio_agreement311 | Qwen3-Omni audio AUC, agreement311 | 0.9727 [0.9507, 0.9904] | 311 (138) | yes |
| POD3c_q3o_audio_conf_minus_agree311 | Qwen3-Omni audio, conflict483 minus agreement311 (paired) | -0.6719 [-0.7266, -0.6105] | 794 (138) | yes |

### PART 20 POD4

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| paper_q2a_orig_recording_conflict | paper q2a zero-shot on the original 468 windows, recording wording, conflict (recomputed) | 0.4132 [0.3005, 0.5423] | 146 (100) | yes |
| paper_q2a_orig_recording_agreement | paper q2a zero-shot on the original 468 windows, recording wording, agreement (recomputed) | 0.6985 [0.6372, 0.7597] | 322 (175) | yes |
| paper_o25_orig_recording_overall | paper o25 zero-shot on the original 468 windows, recording wording, overall (recomputed) | 0.6578 [0.6033, 0.7125] | 468 (228) | yes |
| paper_o25_orig_recording_conflict | paper o25 zero-shot on the original 468 windows, recording wording, conflict (recomputed) | 0.5322 [0.4216, 0.6447] | 146 (100) | yes |
| paper_o25_orig_recording_agreement | paper o25 zero-shot on the original 468 windows, recording wording, agreement (recomputed) | 0.7054 [0.6431, 0.7658] | 322 (175) | yes |

### PART 20 POD4b

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| P20_4b_q3o_orig_shipped_conflict | Qwen3-Omni zero-shot AUC, Pitt, original windows, shipped paper file, conflict | 0.6066 [0.4956, 0.7123] | 146 (100) | yes |
| P20_4b_q3o_orig_shipped_agreement | Qwen3-Omni zero-shot AUC, Pitt, original windows, shipped paper file, agreement | 0.8231 [0.7664, 0.8720] | 322 (175) | yes |

### PART 20 POD4c

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| af3_orig_part10_agreement | AF3 zero-shot AUC, original 468 windows (part10 file), agreement (recomputed) | 0.5855 [0.5144, 0.6554] | 322 (175) | yes |
| af3_orig_part10_conflict | AF3 zero-shot AUC, original 468 windows (part10 file), conflict (recomputed) | 0.6075 [0.5063, 0.7129] | 146 (100) | yes |
| af3_orig_part3_agreement | AF3 zero-shot AUC, original 468 windows (part3 file), agreement (recomputed) | 0.5822 [0.5080, 0.6528] | 322 (175) | yes |
| af3_orig_part3_conflict | AF3 zero-shot AUC, original 468 windows (part3 file), conflict (recomputed) | 0.6169 [0.5188, 0.7209] | 146 (100) | yes |
| af2_orig_af2pitt468_agreement | AF2 zero-shot AUC, original 468 windows (af2pitt468 file), agreement (recomputed) | 0.5544 [0.4760, 0.6296] | 322 (175) | yes |
| af2_orig_af2pitt468_conflict | AF2 zero-shot AUC, original 468 windows (af2pitt468 file), conflict (recomputed) | 0.6310 [0.5336, 0.7260] | 146 (100) | yes |

### PART 20 POD4d

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| kimi_pitt_orig_canonical_conflict | Kimi-Audio zero-shot Pitt original windows conflict AUC (canonical paper file, recomputed) [interval: arm-only | 0.7197 [0.6201, 0.8147] | 146 (100) | yes |
| kimi_pitt_orig_canonical_agreement | Kimi-Audio zero-shot Pitt original windows agreement AUC (canonical paper file, recomputed) [interval: arm-onl | 0.7227 [0.6560, 0.7896] | 322 (175) | yes |

### PART 20 POD5

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| POD5_zs_o25_paper_overall | Qwen2.5-Omni zero-shot p_yes (by-product of extraction; POD4 reports it officially), paper 468 windows, overal | 0.6578 [0.6033, 0.7125] | 468 (228) | yes |
| POD5_zs_o25_paper_conflict | Qwen2.5-Omni zero-shot p_yes (by-product of extraction; POD4 reports it officially), paper 468 windows, confli | 0.5322 [0.4216, 0.6447] | 146 (100) | yes |
| POD5_zs_o25_paper_agreement | Qwen2.5-Omni zero-shot p_yes (by-product of extraction; POD4 reports it officially), paper 468 windows, agreem | 0.7054 [0.6431, 0.7658] | 322 (175) | yes |
| POD5_enc_mac_seed_paper_overall | Qwen2.5-Omni nested probe, encoder, 5 renumbered Mac-tie splits pitt_groupkfold5_mac_seed0..4 (paper 0.7706),  | 0.7706 [0.7157, 0.8220] | 468 (228) | yes |
| POD5_enc_pod_seed_original_rerun_overall | Qwen2.5-Omni nested probe, encoder SUPPLEMENT, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper pod json | 0.7761 [0.7245, 0.8273] | 468 (228) | yes |
| POD5_llm_pod_seed_original_rerun_overall | Qwen2.5-Omni nested probe, LM stages at audio positions, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (pape | 0.8032 [0.7567, 0.8496] | 468 (228) | yes |
| POD5_dir_orig_probe | probe AUC, ORIGINAL states rerun (paper 0.7709) | 0.7709 [0.7210, 0.8224] | 468 (228) | yes |
| POD5_dir_orig_after_removal | after-removal AUC, ORIGINAL states rerun (paper 0.7662) | 0.7661 [0.7173, 0.8155] | 468 (228) | yes |
| POD5_dir_orig_cos | cos, ORIGINAL states rerun on this pod (paper -0.0109) | -0.0109 [-0.0234, 0.0099] | 468 (228) | yes |
| POD5_sft_paper_overall | Qwen2.5-Omni projector fine-tune (sft_projector.py recipe), out of fold, pitt_groupkfold5_pod.csv, paper 468 w | 0.7673 [0.7086, 0.8231] | 468 (228) | yes |
| POD5_sft_paper_conflict | Qwen2.5-Omni projector fine-tune (sft_projector.py recipe), out of fold, pitt_groupkfold5_pod.csv, paper 468 w | 0.5189 [0.4151, 0.6300] | 146 (100) | yes |
| POD5_sft_paper_agreement | Qwen2.5-Omni projector fine-tune (sft_projector.py recipe), out of fold, pitt_groupkfold5_pod.csv, paper 468 w | 0.8593 [0.8063, 0.9069] | 322 (175) | yes |

### PART 20 POD6

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| C_af3_part10_conflict | Pitt AF3 zero-shot answer AUC, conflict arm (alternative copy overnight2/part10; matches paper Table 2) | 0.6075 [0.5063, 0.7129] | 146 (100) | yes |
| C_af3_part10_agreement | Pitt AF3 zero-shot answer AUC, agreement arm (alternative copy overnight2/part10; matches paper Table 2) | 0.5855 [0.5144, 0.6554] | 322 (175) | yes |
| C_af3_part10_conflict_minus_agreement | Pitt AF3 conflict minus agreement AUC, one speaker draw per replicate (overnight2/part10 copy) | 0.0220 [-0.0846, 0.1357] | 468 (228) | yes |
| C_M8_Qwen2.5-Omni_primary_conflict | Pitt Qwen2.5-Omni (primary) zero-shot answer AUC, conflict | 0.5322 [0.4216, 0.6447] | 146 (100) | yes |
| C_M8_Qwen2.5-Omni_primary_agreement | Pitt Qwen2.5-Omni (primary) zero-shot answer AUC, agreement | 0.7054 [0.6431, 0.7658] | 322 (175) | yes |
| C_M8_Qwen2.5-Omni_primary_conflict_minus_agreement | Pitt Qwen2.5-Omni (primary) zero-shot answer AUC, conflict minus agreement | -0.1731 [-0.2958, -0.0392] | 468 (228) | yes |
| C_M8_Qwen2-Audio_primary_conflict | Pitt Qwen2-Audio (primary) zero-shot answer AUC, conflict | 0.4132 [0.3005, 0.5423] | 146 (100) | yes |
| C_M8_Qwen2-Audio_primary_agreement | Pitt Qwen2-Audio (primary) zero-shot answer AUC, agreement | 0.6985 [0.6372, 0.7597] | 322 (175) | yes |
| C_M8_Qwen2-Audio_primary_conflict_minus_agreement | Pitt Qwen2-Audio (primary) zero-shot answer AUC, conflict minus agreement | -0.2853 [-0.3950, -0.1485] | 468 (228) | yes |
| C_M8_Qwen3-Omni-30B-A3B_primary_conflict | Pitt Qwen3-Omni-30B-A3B (primary) zero-shot answer AUC, conflict | 0.6066 [0.4956, 0.7123] | 146 (100) | yes |
| C_M8_Qwen3-Omni-30B-A3B_primary_agreement | Pitt Qwen3-Omni-30B-A3B (primary) zero-shot answer AUC, agreement | 0.8231 [0.7664, 0.8720] | 322 (175) | yes |
| C_M8_Qwen3-Omni-30B-A3B_primary_conflict_minus_agreement | Pitt Qwen3-Omni-30B-A3B (primary) zero-shot answer AUC, conflict minus agreement | -0.2165 [-0.3242, -0.1052] | 468 (228) | yes |
| C_M8_AudioFlamingo2_af2_pitt468_conflict | Pitt Audio Flamingo 2 (af2_pitt468) zero-shot answer AUC, conflict | 0.6310 [0.5336, 0.7260] | 146 (100) | yes |
| C_M8_AudioFlamingo2_af2_pitt468_agreement | Pitt Audio Flamingo 2 (af2_pitt468) zero-shot answer AUC, agreement | 0.5544 [0.4760, 0.6296] | 322 (175) | yes |
| C_M8_AudioFlamingo2_af2_pitt468_conflict_minus_agreement | Pitt Audio Flamingo 2 (af2_pitt468) zero-shot answer AUC, conflict minus agreement | 0.0766 [-0.0307, 0.1869] | 468 (228) | yes |
| C_M8_AudioFlamingo3_primary_conflict | Pitt Audio Flamingo 3 (primary) zero-shot answer AUC, conflict | 0.6169 [0.5188, 0.7209] | 146 (100) | yes |
| C_M8_AudioFlamingo3_primary_agreement | Pitt Audio Flamingo 3 (primary) zero-shot answer AUC, agreement | 0.5822 [0.5080, 0.6528] | 322 (175) | yes |
| C_M8_AudioFlamingo3_primary_conflict_minus_agreement | Pitt Audio Flamingo 3 (primary) zero-shot answer AUC, conflict minus agreement | 0.0347 [-0.0698, 0.1474] | 468 (228) | yes |
| C_M8_Kimi-Audio_primary_conflict | Pitt Kimi-Audio (primary) zero-shot answer AUC, conflict | 0.7197 [0.6201, 0.8147] | 146 (100) | yes |
| C_M8_Kimi-Audio_primary_agreement | Pitt Kimi-Audio (primary) zero-shot answer AUC, agreement | 0.7227 [0.6560, 0.7896] | 322 (175) | yes |
| C_M8_Kimi-Audio_primary_conflict_minus_agreement | Pitt Kimi-Audio (primary) zero-shot answer AUC, conflict minus agreement | -0.0030 [-0.1134, 0.1065] | 468 (228) | yes |
| C_T6_s20_text_all468 | Pitt Qwen2.5-Omni transcript-only answer AUC, all 468 (Sep 20 run) | 0.6629 [0.5983, 0.7245] | 468 (228) | yes |
| C_T6_s20_text_conflict | Pitt Qwen2.5-Omni transcript-only answer AUC, conflict arm (Sep 20 run) | 0.5560 [0.4516, 0.6680] | 146 (100) | yes |
| C_T6_s20_text_agreement | Pitt Qwen2.5-Omni transcript-only answer AUC, agreement arm (Sep 20 run) | 0.7078 [0.6381, 0.7696] | 322 (175) | yes |
| C_T6_s20_text_conflict_minus_agreement | Pitt Qwen2.5-Omni transcript-only PAIRED conflict minus agreement AUC (Sep 20 run) | -0.1518 [-0.2665, -0.0271] | 468 (228) | yes |
| C_T6_audio_all468 | Pitt Qwen2.5-Omni audio zero-shot answer AUC, all 468 | 0.6578 [0.6033, 0.7125] | 468 (228) | yes |
| C_T6_audio_conflict | Pitt Qwen2.5-Omni audio zero-shot answer AUC, conflict arm | 0.5322 [0.4216, 0.6447] | 146 (100) | yes |
| C_T6_audio_agreement | Pitt Qwen2.5-Omni audio zero-shot answer AUC, agreement arm | 0.7054 [0.6431, 0.7658] | 322 (175) | yes |
| C_T6_audio_conflict_minus_agreement | Pitt Qwen2.5-Omni audio zero-shot PAIRED conflict minus agreement AUC | -0.1731 [-0.2958, -0.0392] | 468 (228) | yes |
| B_zeroshot_nvfull_ref | Qwen2.5-Omni zero-shot answer AUC, NeuroVoz full 1270 (paper reference, recomputed) | 0.6951 [0.6356, 0.7519] | 1270 (107) | yes |

### PART 22 P22

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| P22_all275_trunc | E-DAIC zero-shot answer AUC, 300 s cut (paper call; reference per-clip file), all 275 E-DAIC windows | 0.8278 [0.7703, 0.8822] | 275 (275) | yes |
| P22_S1_capped900 | E-DAIC full windows capped at 900 s (windows_full.csv capped==1, of 275) [value restored by the PART 22 verifi | 89 | 275 (275) | yes |
| P22_auc_default_all275 | E-DAIC zero-shot AUC, default processor (300 s cut), all275 | 0.8278 [0.7703, 0.8822] | 275 (275) | yes |

## B. PART 23, E-DAIC whole interview
| id | claimed | recomputed | lo | hi | checked |
|---|---|---|---|---|---|
| 1_whole | 0.8366 | 0.8366 | 0.7786 | 0.8866 | yes |
| 1_300s | 0.8278 | 0.8278 | 0.7703 | 0.8822 | yes |
| 1_whole_minus_300s | 0.0088 | 0.0088 | -0.032 | 0.051 | yes |
| 2_encoder_meanof5 | 0.5981 | 0.5981 | 0.5267 | 0.6672 | yes |
| 2_encoder_minus_zswhole | -0.2385 | -0.2385 | -0.3244 | -0.1505 | yes |
| 3_projector_meanof5 | 0.5866 | 0.5866 | 0.5166 | 0.6585 | yes |
| 3_projector_minus_zswhole | -0.25 | -0.25 | -0.3339 | -0.1623 | yes |
| 4_LM_meanof5 | 0.7437 | 0.7437 | 0.6752 | 0.8097 | yes |
| 4_LM_minus_zswhole | -0.0929 | -0.0929 | -0.1609 | -0.0257 | yes |
| 5_answer state_meanof5 | 0.8359 | 0.8359 | 0.7819 | 0.8832 | yes |
| 5_answer state_minus_zswhole | -0.0007 | -0.0007 | -0.0455 | 0.0436 | yes |
| 6_ft_whole_pooled | 0.7931 | 0.7931 | 0.7263 | 0.8536 | yes |
| 6_ft_whole_fold0 | 0.9079 | 0.9079 | nan | nan | yes |
| 6_ft_whole_fold1 | 0.8592 | 0.8592 | nan | nan | yes |
| 6_ft_whole_fold2 | 0.8117 | 0.8117 | nan | nan | yes |
| 6_ft_whole_fold3 | 0.4302 | 0.4302 | nan | nan | yes |
| 6_ft_whole_fold4 | 0.8304 | 0.8304 | nan | nan | yes |
| 6_ft_whole_minus_zswhole | -0.0435 | -0.0435 | -0.0939 | 0.0027 | yes |
| 7_ft_ctrl300_pooled | 0.815 | 0.815 | 0.7465 | 0.8776 | yes |
| 7_ft_ctrl300_fold0 | 0.9758 | 0.9758 | nan | nan | yes |
| 7_ft_ctrl300_fold1 | 0.8233 | 0.8233 | nan | nan | yes |
| 7_ft_ctrl300_fold2 | 0.8358 | 0.8358 | nan | nan | yes |
| 7_ft_ctrl300_fold3 | 0.7442 | 0.7442 | nan | nan | yes |
| 7_ft_ctrl300_fold4 | 0.8043 | 0.8043 | nan | nan | yes |
| 7_ft_whole_minus_ctrl300 | -0.0219 | -0.0219 | -0.0783 | 0.0356 | yes |
| 8_fold3_median_mass | 0.0 | 0.0 | nan | nan | yes |
| 8_fold3_count_lt_0.01 | nan | 55.0 | nan | nan | nan |
| 9a_ft_ctrl300_minus_zs300 | nan | -0.0128 | -0.0483 | 0.0214 | nan |
| 9b_ft_whole_pooled_no_fold3 | nan | 0.8274 | 0.7569 | 0.8898 | nan |
| 10_q3o_cut_count | 0.0 | 0.0 | nan | nan | yes |
| 10_af3_cut_count | 136.0 | 136.0 | nan | nan | yes |
| 11_encoder_max_abs_diff | nan | 0.0611 | nan | nan | nan |
| 11_llm_max_abs_diff | nan | 0.0578 | nan | nan | nan |
| 11_encoder_stage0 | 0.5835 | 0.5859 | nan | nan | no |
| 11_encoder_stage1 | 0.5593 | 0.5468 | nan | nan | no |
| 11_encoder_stage2 | 0.6051 | 0.575 | nan | nan | no |
| 11_encoder_stage3 | 0.5998 | 0.5711 | nan | nan | no |
| 11_encoder_stage4 | 0.6071 | 0.5943 | nan | nan | no |
| 11_encoder_stage5 | 0.597 | 0.5987 | nan | nan | no |
| 11_encoder_stage6 | 0.6047 | 0.601 | nan | nan | no |
| 11_encoder_stage7 | 0.5867 | 0.58 | nan | nan | no |
| 11_encoder_stage8 | 0.5824 | 0.5908 | nan | nan | no |
| 11_encoder_stage9 | 0.5932 | 0.5936 | nan | nan | no |
| 11_encoder_stage10 | 0.6151 | 0.6079 | nan | nan | no |
| 11_encoder_stage11 | 0.6093 | 0.5727 | nan | nan | no |
| 11_encoder_stage12 | 0.6013 | 0.5537 | nan | nan | no |
| 11_encoder_stage13 | 0.6087 | 0.585 | nan | nan | no |
| 11_encoder_stage14 | 0.6098 | 0.5692 | nan | nan | no |
| 11_encoder_stage15 | 0.624 | 0.5993 | nan | nan | no |
| 11_encoder_stage16 | 0.6468 | 0.6124 | nan | nan | no |
| 11_encoder_stage17 | 0.6604 | 0.627 | nan | nan | no |
| 11_encoder_stage18 | 0.6847 | 0.6236 | nan | nan | no |
| 11_encoder_stage19 | 0.6875 | 0.636 | nan | nan | no |
| 11_encoder_stage20 | 0.6752 | 0.6285 | nan | nan | no |
| 11_encoder_stage21 | 0.6674 | 0.6297 | nan | nan | no |
| 11_encoder_stage22 | 0.6613 | 0.6287 | nan | nan | no |
| 11_encoder_stage23 | 0.652 | 0.6375 | nan | nan | no |
| 11_encoder_stage24 | 0.6611 | 0.6435 | nan | nan | no |
| 11_encoder_stage25 | 0.6654 | 0.6595 | nan | nan | no |
| 11_encoder_stage26 | 0.6396 | 0.6441 | nan | nan | no |
| 11_encoder_stage27 | 0.6234 | 0.604 | nan | nan | no |
| 11_encoder_stage28 | 0.6291 | 0.6191 | nan | nan | no |
| 11_encoder_stage29 | 0.6177 | 0.5992 | nan | nan | no |
| 11_encoder_stage30 | 0.598 | 0.587 | nan | nan | no |
| 11_encoder_stage31 | 0.636 | 0.6215 | nan | nan | no |
| 11_llm_stage0 | 0.6094 | 0.5844 | nan | nan | no |
| 11_llm_stage1 | 0.6318 | 0.5928 | nan | nan | no |
| 11_llm_stage2 | 0.6441 | 0.6053 | nan | nan | no |
| 11_llm_stage3 | 0.6641 | 0.6064 | nan | nan | no |
| 11_llm_stage4 | 0.6647 | 0.6126 | nan | nan | no |
| 11_llm_stage5 | 0.6663 | 0.6218 | nan | nan | no |
| 11_llm_stage6 | 0.6341 | 0.5996 | nan | nan | no |
| 11_llm_stage7 | 0.6492 | 0.6151 | nan | nan | no |
| 11_llm_stage8 | 0.6878 | 0.6628 | nan | nan | no |
| 11_llm_stage9 | 0.706 | 0.6659 | nan | nan | no |
| 11_llm_stage10 | 0.7389 | 0.7122 | nan | nan | no |
| 11_llm_stage11 | 0.7607 | 0.7355 | nan | nan | no |
| 11_llm_stage12 | 0.7437 | 0.7233 | nan | nan | no |
| 11_llm_stage13 | 0.7352 | 0.7116 | nan | nan | no |
| 11_llm_stage14 | 0.7444 | 0.7251 | nan | nan | no |
| 11_llm_stage15 | 0.7519 | 0.7347 | nan | nan | no |
| 11_llm_stage16 | 0.7308 | 0.7289 | nan | nan | no |
| 11_llm_stage17 | 0.7163 | 0.7293 | nan | nan | no |
| 11_llm_stage18 | 0.7362 | 0.7445 | nan | nan | no |
| 11_llm_stage19 | 0.7144 | 0.7331 | nan | nan | no |
| 11_llm_stage20 | 0.7186 | 0.731 | nan | nan | no |
| 11_llm_stage21 | 0.7086 | 0.7121 | nan | nan | no |
| 11_llm_stage22 | 0.7093 | 0.7081 | nan | nan | no |
| 11_llm_stage23 | 0.7324 | 0.7263 | nan | nan | no |
| 11_llm_stage24 | 0.7289 | 0.7155 | nan | nan | no |
| 11_llm_stage25 | 0.7292 | 0.7216 | nan | nan | no |
| 11_llm_stage26 | 0.7474 | 0.7302 | nan | nan | no |
| 11_llm_stage27 | 0.7596 | 0.7283 | nan | nan | no |
| 11_llm_stage28 | 0.7753 | 0.7358 | nan | nan | no |

## C. PART 24, E-DAIC whole interview fine-tune
| id | claimed | recomputed | verified |
|---|---|---|---|
| check_fold_file_is_part23 | PART23 FT sidecars fold_file sha256 1d57d4624adf4f40 | PART24 staged + 10 pod copies sha256 1d57d4624adf4f40; all 10 pod sidecars read 1d57d4624adf4f40 | yes |
| check_fold_assignment_seed0 | fold k test set = PART23 fold k (fold file and PART23 FT_whole_fold{k}_oof.csv) | pod files match fold file: True; PART23 oof sets match fold file: True; per-clip fold column matches: True; sizes [55, 55, 55, 55, 55] | yes |
| check_fold_assignment_seed1all | fold k test set = PART23 fold k (fold file and PART23 FT_whole_fold{k}_oof.csv) | pod files match fold file: True; PART23 oof sets match fold file: True; per-clip fold column matches: True; sizes [55, 55, 55, 55, 55] | yes |
| check_each_clip_once_seed0 | 275 clips, each once | per-clip csv rows 275 unique 275 max count 1; pod rows 275 unique 275; set == fold file == manifest == zero-shot: True; pod problems 0 | yes |
| check_each_clip_once_seed1all | 275 clips, each once | per-clip csv rows 275 unique 275 max count 1; pod rows 275 unique 275; set == fold file == manifest == zero-shot: True; pod problems 0 | yes |
| check_perclip_equals_pod_seed0 | per-clip csv copied from the 5 pod oof csvs | 275/275 rows identical in p_yes, answer_mass, label, audio_tok, fold, seed | yes |
| check_perclip_equals_pod_seed1all | per-clip csv copied from the 5 pod oof csvs | 275/275 rows identical in p_yes, answer_mass, label, audio_tok, fold, seed | yes |
| check_labels | labels agree across manifest, zero-shot csv, both per-clip csvs; n_positive 66 | agree: True; n_positive 66; manifest pid==speaker: True | yes |
| seed0_pooled_oof_auc | 0.8588879222850515 [0.8049822386544874, 0.9028700329483457] n=275 (summary 0.8589 0.8050 0.9029) | 0.8588879222850515 [0.8049822386544874, 0.9028700329483457] n=275 usable_draws=2000 | yes |
| seed0_ft_minus_zs_whole_paired | 0.022292301000435 [-0.0063755012224089355, 0.054672203811101744] n=275 (summary 0.0223 -0.0064 0.0547) | 0.022292301000435 [-0.0063755012224089355, 0.054672203811101744] n=275 usable_draws=2000 | yes |
| seed0_fold0_auc | 0.9166666666666667 [0.8266656346749226, 0.9782608695652174] n=55 (summary 0.9167 0.8267 0.9783) | 0.9166666666666667 [0.8266656346749226, 0.9782608695652174] n=55 usable_draws=2000 | yes |
| seed0_fold0_median_answer_mass | 0.9938538074493408 n=55 (summary 0.9939) | 0.9938538074493408 n=55 | yes |
| seed0_fold1_auc | 0.8624999999999999 [0.7334168118466898, 0.9613035161432995] n=55 (summary 0.8625 0.7334 0.9613) | 0.8624999999999999 [0.7334168118466898, 0.9613035161432995] n=55 usable_draws=2000 | yes |
| seed0_fold1_median_answer_mass | 0.9934326112270355 n=55 (summary 0.9934) | 0.9934326112270355 n=55 | yes |
| seed0_fold2_auc | 0.8416666666666667 [0.6954583333333333, 0.9555560394889664] n=55 (summary 0.8417 0.6955 0.9556) | 0.8416666666666667 [0.6954583333333333, 0.9555560394889664] n=55 usable_draws=2000 | yes |
| seed0_fold2_median_answer_mass | 0.9971539359539747 n=55 (summary 0.9972) | 0.9971539359539747 n=55 | yes |
| seed0_fold3_auc | 0.8372093023255813 [0.6955220385674932, 0.9464345238095238] n=55 (summary 0.8372 0.6955 0.9464) | 0.8372093023255813 [0.6955220385674932, 0.9464345238095238] n=55 usable_draws=2000 | yes |
| seed0_fold3_median_answer_mass | 0.9979457128793001 n=55 (summary 0.9979) | 0.9979457128793001 n=55 | yes |
| seed0_fold4_auc | 0.8585271317829457 [0.7112278973669037, 0.9670594545957917] n=55 (summary 0.8585 0.7112 0.9671) | 0.8585271317829457 [0.7112278973669037, 0.9670594545957917] n=55 usable_draws=2000 | yes |
| seed0_fold4_median_answer_mass | 0.997003898024559 n=55 (summary 0.9970) | 0.997003898024559 n=55 | yes |
| ref_zs_whole_auc | 0.8365956212846165 [0.7785777395969262, 0.886634568107391] n=275 (summary 0.8366 0.7786 0.8866) | 0.8365956212846165 [0.7785777395969262, 0.886634568107391] n=275 usable_draws=2000 | yes |
| extra_seed1all_pooled_oof_auc | 0.8183630564013339 [0.7496418164976183, 0.8757892026693909] n=275 (summary 0.8184 0.7496 0.8758) | 0.8183630564013339 [0.7496418164976183, 0.8757892026693909] n=275 usable_draws=2000 | yes |
| extra_seed1all_ft_minus_zs_whole_paired | -0.01823256488328262 [-0.05185012131535606, 0.013896457721323927] n=275 (summary -0.0182 -0.0519 0.0139) | -0.01823256488328262 [-0.05185012131535606, 0.013896457721323927] n=275 usable_draws=2000 | yes |
| extra_seed1all_fold0_auc | 0.9224806201550387 [0.8333333333333334, 0.9822280701754387] n=55 (summary 0.9225) | 0.9224806201550387 [0.8333333333333334, 0.9822280701754387] n=55 usable_draws=2000 | yes |
| extra_seed1all_fold0_median_answer_mass | 0.9945899248123169 n=55 (summary 0.9946) | 0.9945899248123169 n=55 | yes |
| extra_seed1all_fold1_auc | 0.8116666666666668 [0.6620168576342357, 0.9287949776401789] n=55 (summary 0.8117) | 0.8116666666666668 [0.6620168576342357, 0.9287949776401789] n=55 usable_draws=2000 | yes |
| extra_seed1all_fold1_median_answer_mass | 0.994668036699295 n=55 (summary 0.9947) | 0.994668036699295 n=55 | yes |
| extra_seed1all_fold2_auc | 0.8400000000000001 [0.6866487603305786, 0.9686466988296255] n=55 (summary 0.8400) | 0.8400000000000001 [0.6866487603305786, 0.9686466988296255] n=55 usable_draws=2000 | yes |
| extra_seed1all_fold2_median_answer_mass | 0.9962901696562767 n=55 (summary 0.9963) | 0.9962901696562767 n=55 | yes |
| extra_seed1all_fold3_auc | 0.8420542635658915 [0.7003447530270998, 0.945256191319503] n=55 (summary 0.8421) | 0.8420542635658915 [0.7003447530270998, 0.945256191319503] n=55 usable_draws=2000 | yes |
| extra_seed1all_fold3_median_answer_mass | 0.9962337575852871 n=55 (summary 0.9962) | 0.9962337575852871 n=55 | yes |
| extra_seed1all_fold4_auc | 0.8352713178294574 [0.6547193877551021, 0.9642896825396825] n=55 (summary 0.8353) | 0.8352713178294574 [0.6547193877551021, 0.9642896825396825] n=55 usable_draws=2000 | yes |
| extra_seed1all_fold4_median_answer_mass | 0.9959627240896225 n=55 (summary 0.9960) | 0.9959627240896225 n=55 | yes |
| check_seed_rule | no fold triggered (all seed 0 medians > 0.99, rule < 0.5); seed-rule set = seed 0 set; no seedrule csv built | recomputed medians [0.993854, 0.993433, 0.997154, 0.997946, 0.997004] min 0.993433; triggered []; MASS_CHECK files match exactly and say 0:no_rerun 1:no_rerun 2:no_rerun 3:no_rerun 4:no_rerun: True; seed 1 files on p24-ft pods 0; seedrule csv exists False; sidecar seeds_by_fold all 0 | yes |
| check_seed1all_pod_sidecars | seed 1 pod sidecars: seed 1, median answer mass as recomputed | match: True | yes |
| check_pod_RESULT_fold_auc | pod .RESULT auc_p_yes for all 10 runs | all equal recomputed fold AUC: True | yes |
| check_zero_shot_reproduces_0.8366 | 0.8366 (PART23 sidecar 0.8365956212846165 [0.7785777395969262, 0.886634568107391]) | 0.8365956212846165 [0.7785777395969262, 0.886634568107391] from p_yes_whole of A2_edaic_whole_zeroshot_perclip.csv sha256 e13f360198811ff3 | yes |
| check_audio_tok_max | largest audio_tok 22500, 89 clips reach it | max 22500, count 89 | yes |
| check_window_cut_hashes_equal_part23 | cut hashes identical to PART23 | 10 PART24 pod cut lists + 5 PART23 pod cut lists, distinct sha256 1 | yes |
| check_part23_fold3_median_mass | PART23 fold 3 median mass 0 (summary; 0.0000 at 4 dp) | 1.0713833603404055e-05 = 0.0000 at 4 dp; not exactly 0 | yes |
| check_paired_interval_includes_0 | interval includes 0 | [-0.0064, 0.0547] | yes |

## D. Support, released and superseded values (1205)

### PART 16 EDAICFULL ({'SUPERSEDED / NOT USED': 8, 'RELEASED ONLY': 3})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| edaic_full_probe_enc | encoder probe AUC (per-clip OOF averaged over repeats) | 0.5916 [0.5110, 0.6706] | 275 (275) | yes |
| edaic_full_probe_proj | projector probe AUC (per-clip OOF averaged over repeats) | 0.5631 [0.4790, 0.6478] | 275 (275) | yes |
| edaic_full_probe_llm | LM probe AUC (per-clip OOF averaged over repeats) | 0.7088 [0.6232, 0.7874] | 275 (275) | yes |
| edaic_full_probe_ans | answer-state probe AUC (per-clip OOF averaged over repeats) | 0.7748 [0.7041, 0.8387] | 275 (275) | yes |
| edaic_full_gap_llm_minus_answer | LM probe minus zero-shot answer (paired) | -0.1190 [-0.1995, -0.0478] | 275 (275) | yes |
| edaic_full_gap_enc_minus_answer | encoder probe minus zero-shot answer (paired) | -0.2361 [-0.3272, -0.1394] | 275 (275) | yes |
| edaic_full_gap_ans_minus_answer | answer-state probe minus zero-shot answer (paired) | -0.0530 [-0.1046, -0.0049] | 275 (275) | yes |
| edaic_full_gap_proj_minus_answer | projector probe minus zero-shot answer (paired) | -0.2646 [-0.3575, -0.1667] | 275 (275) | yes |
| edaic_first30_zeroshot_answer | Qwen2.5-Omni zero-shot answer AUC, FIRST 30 s window | 0.6620 [0.5831, 0.7315] | 275 (275) | yes |
| edaic_first30_probe_enc | encoder probe AUC, FIRST 30 s window | 0.6324 [0.5520, 0.7104] | 275 (275) | yes |
| edaic_first30_gap_enc_minus_answer | encoder probe minus zero-shot answer (paired), FIRST 30 s window | -0.0297 [-0.1291, 0.0752] | 275 (275) | yes |

### PART 16 M1 ({'SUPERSEDED / NOT USED': 7})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M1_text_all468 | Pitt Qwen2.5-Omni transcript-only answer AUC, all468 arm | 0.6975 [0.6369, 0.7549] | 468 (228) | yes |
| M1_text_conflict | Pitt Qwen2.5-Omni transcript-only answer AUC, conflict arm | 0.4550 [0.3412, 0.5718] | 146 (100) | yes |
| M1_text_agreement | Pitt Qwen2.5-Omni transcript-only answer AUC, agreement arm | 0.7987 [0.7392, 0.8535] | 322 (175) | yes |
| M1_text_conflict_minus_agreement | Pitt Qwen2.5-Omni transcript-only paired conflict minus agreement AUC | -0.3437 [-0.4556, -0.2209] | 468 (228) | yes |
| M1_text_minus_audio_all468 | Pitt Qwen2.5-Omni paired transcript minus audio AUC, all468 arm | 0.0397 [-0.0302, 0.1070] | 468 (228) | yes |
| M1_text_minus_audio_conflict | Pitt Qwen2.5-Omni paired transcript minus audio AUC, conflict arm | -0.0772 [-0.2360, 0.0680] | 146 (100) | yes |
| M1_text_minus_audio_agreement | Pitt Qwen2.5-Omni paired transcript minus audio AUC, agreement arm | 0.0933 [0.0157, 0.1637] | 322 (175) | yes |

### PART 16 M11 ({'RELEASED ONLY': 36})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M11#01:A_asr_reproduction_wer_vs_paper_transcript | A_asr_reproduction_wer_vs_paper_transcript | 0 | 37 (37) | no twin |
| M11#02:B_xsys_disagree_mean_PD | B_xsys_disagree_mean_PD | 0.3049 [0.1301, 0.4944] | 37 (37) | yes |
| M11#03:B_xsys_disagree_mean_control | B_xsys_disagree_mean_control | 0.4779 [0.0917, 1.1479] | 37 (37) | yes |
| M11#04:B_xsys_disagree_diff_PD_minus_control | B_xsys_disagree_diff_PD_minus_control | -0.1730 [-0.8969, 0.3060] | 37 (37) | yes |
| M11#05:B_xsys_disagree_spearman_vs_pyes_text | B_xsys_disagree_spearman_vs_pyes_text | 0.2723 [-0.0814, 0.5558] | 37 (37) | no twin |
| M11#06:B_xsys_disagree_spearman_vs_pyes_audio | B_xsys_disagree_spearman_vs_pyes_audio | -0.0916 [-0.3876, 0.2143] | 37 (37) | no twin |
| M11#07:B_xsys_disagree_auc_alone | B_xsys_disagree_auc_alone | 0.6220 | 37 (37) | no twin |
| M11#08:C_mean_token_logprob_mean_PD | C_mean_token_logprob_mean_PD | -0.1103 [-0.1502, -0.0739] | 37 (37) | yes |
| M11#09:C_mean_token_logprob_mean_control | C_mean_token_logprob_mean_control | -0.1356 [-0.2345, -0.0759] | 37 (37) | yes |
| M11#10:C_mean_token_logprob_diff_PD_minus_control | C_mean_token_logprob_diff_PD_minus_control | 0.0253 [-0.0532, 0.1288] | 37 (37) | yes |
| M11#11:C_mean_token_logprob_spearman_vs_pyes_text | C_mean_token_logprob_spearman_vs_pyes_text | -0.4009 [-0.6720, -0.0835] | 37 (37) | no twin |
| M11#12:C_mean_token_logprob_spearman_vs_pyes_audio | C_mean_token_logprob_spearman_vs_pyes_audio | -0.3101 [-0.5693, 0.0113] | 37 (37) | no twin |
| M11#13:C_mean_token_logprob_auc_alone | C_mean_token_logprob_auc_alone | 0.4792 | 37 (37) | no twin |
| M11#14:C_compression_ratio_mean_PD | C_compression_ratio_mean_PD | 1.5938 [1.4484, 1.7269] | 37 (37) | yes |
| M11#15:C_compression_ratio_mean_control | C_compression_ratio_mean_control | 1.4898 [1.3076, 1.6417] | 37 (37) | yes |
| M11#16:C_compression_ratio_diff_PD_minus_control | C_compression_ratio_diff_PD_minus_control | 0.1040 [-0.1036, 0.3400] | 37 (37) | yes |
| M11#17:C_compression_ratio_spearman_vs_pyes_text | C_compression_ratio_spearman_vs_pyes_text | -0.1390 [-0.4219, 0.1794] | 37 (37) | no twin |
| M11#18:C_compression_ratio_spearman_vs_pyes_audio | C_compression_ratio_spearman_vs_pyes_audio | -0.1305 [-0.4261, 0.1964] | 37 (37) | no twin |
| M11#19:C_compression_ratio_auc_alone | C_compression_ratio_auc_alone | 0.5625 | 37 (37) | no twin |
| M11#20:D_n_words_30s_mean_PD | D_n_words_30s_mean_PD | 59.1875 [46.1875, 71.9375] | 37 (37) | yes |
| M11#21:D_n_words_30s_mean_control | D_n_words_30s_mean_control | 55.8095 [43.0464, 67.6202] | 37 (37) | yes |
| M11#22:D_n_words_30s_diff_PD_minus_control | D_n_words_30s_diff_PD_minus_control | 3.3780 [-13.8212, 22.5698] | 37 (37) | yes |
| M11#23:D_n_words_30s_spearman_vs_pyes_text | D_n_words_30s_spearman_vs_pyes_text | -0.1588 [-0.4443, 0.1670] | 37 (37) | no twin |
| M11#24:D_n_words_30s_spearman_vs_pyes_audio | D_n_words_30s_spearman_vs_pyes_audio | -0.0508 [-0.3683, 0.2864] | 37 (37) | no twin |
| M11#25:D_n_words_30s_auc_alone | D_n_words_30s_auc_alone | 0.5268 | 37 (37) | no twin |
| M11#26:E_auc_from_passage_identity_alone | E_auc_from_passage_identity_alone | 0.6682 | 37 (37) | no twin |
| M11#29:F_wer_vs_local_copy_mean_PD | F_wer_vs_local_copy_mean_PD | 0.3123 [0.1974, 0.4388] | 37 (37) | yes |
| M11#30:F_wer_vs_local_copy_mean_control | F_wer_vs_local_copy_mean_control | 0.3987 [0.2558, 0.5577] | 37 (37) | yes |
| M11#31:B_xsys_disagree_MEDIAN_PD | B_xsys_disagree_MEDIAN_PD | 0.1027 [0.0323, 0.4799] | 16 (16) | yes |
| M11#32:B_xsys_disagree_MEDIAN_control | B_xsys_disagree_MEDIAN_control | 0.0333 [0, 0.2088] | 21 (21) | yes |
| M11#33:B_xsys_disagree_MEDIAN_diff_PD_minus_control | B_xsys_disagree_MEDIAN_diff_PD_minus_control | 0.0694 | 37 (37) | yes |
| M11#34:B_xsys_disagree_ge20w_MEAN_PD | B_xsys_disagree_ge20w_MEAN_PD | 0.2100 [0.0645, 0.3906] | 14 (14) | yes |
| M11#35:B_xsys_disagree_ge20w_MEAN_control | B_xsys_disagree_ge20w_MEAN_control | 0.1506 [0.0481, 0.2765] | 17 (17) | yes |
| M11#36:B_xsys_disagree_ge20w_MEAN_diff_PD_minus_control | B_xsys_disagree_ge20w_MEAN_diff_PD_minus_control | 0.0594 | 37 (37) | yes |
| M11#37:B_xsys_disagree_POOLED_PD | B_xsys_disagree_POOLED_PD | 0.2650 [0.0729, 0.4944] | 16 (16) | yes |
| M11#38:B_xsys_disagree_POOLED_control | B_xsys_disagree_POOLED_control | 0.1800 [0.0595, 0.3764] | 21 (21) | yes |

### PART 16 M2 ({'RELEASED ONLY': 17})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M2.label_all.agreement | sentiment label (pos/neu/neg at 0.70), all segments / raw agreement po | 0.4248 [0.4107, 0.4393] | 6214 (275) | yes |
| M2.label_all.kappa | sentiment label (pos/neu/neg at 0.70), all segments / Cohen kappa | 0.2481 [0.2360, 0.2605] | 6214 (275) | yes |
| M2.label_elig.agreement | sentiment label (pos/neu/neg at 0.70), eligible segments / raw agreement po | 0.4286 [0.4130, 0.4446] | 5523 (275) | yes |
| M2.label_elig.kappa | sentiment label (pos/neu/neg at 0.70), eligible segments / Cohen kappa | 0.2510 [0.2377, 0.2653] | 5523 (275) | yes |
| M2.arm_rule.agreement | rule arm (conflict/agreement/none), eligible segments / raw agreement po | 0.5524 [0.5286, 0.5754] | 5523 (275) | yes |
| M2.arm_rule.kappa | rule arm (conflict/agreement/none), eligible segments / Cohen kappa | 0.3438 [0.3224, 0.3665] | 5523 (275) | yes |
| M2.arm_selected.agreement | selected arm (conflict/agreement/none), eligible segments / raw agreement po | 0.7045 [0.6674, 0.7391] | 5523 (275) | yes |
| M2.arm_selected.kappa | selected arm (conflict/agreement/none), eligible segments / Cohen kappa | 0.3362 [0.3008, 0.3717] | 5523 (275) | yes |
| M2.arm_rule_matched.agreement | rule arm, distilbert at matched threshold 0.995 / raw agreement po | 0.7290 [0.7128, 0.7460] | 5523 (275) | yes |
| M2.arm_rule_matched.kappa | rule arm, distilbert at matched threshold 0.995 / Cohen kappa | 0.4978 [0.4728, 0.5234] | 5523 (275) | yes |
| M2.arm_selected_matched.agreement | selected arm, distilbert at matched threshold 0.995 / raw agreement po | 0.8356 [0.8146, 0.8560] | 5523 (275) | yes |
| M2.arm_selected_matched.kappa | selected arm, distilbert at matched threshold 0.995 / Cohen kappa | 0.4366 [0.3989, 0.4755] | 5523 (275) | yes |
| M2.matched_threshold | distilbert threshold whose conflict count is closest to RoBERTa's 488 | 0.9950 | 644 (149) | yes |
| M2.pairs | distilbert SST-2 rebuild of the Part 14 selection / pairs | 1334 | 2668 (162) | yes |
| M2.clips | distilbert SST-2 rebuild of the Part 14 selection / clips | 2668 | 2668 (162) | yes |
| M2.speakers | distilbert SST-2 rebuild of the Part 14 selection / speakers | 162 | 2668 (162) | yes |
| M2.same_arm_seg_uid | segments with the same (arm, seg_uid) in both selections | 625 | 2302 (162) | yes |

### PART 16 M3 ({'RELEASED ONLY': 6})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M3_pitt_speaker | Omni zero shot pitt speaker-level AUC (mean p_yes) | 0.6771 [0.6096, 0.7494] | 228 (228) | yes |
| M3_pitt_clip_minus_speaker | Omni zero shot pitt paired clip minus speaker AUC | -0.0193 [-0.0565, 0.0172] | 468 (228) | yes |
| M3_pcgita_speaker | Omni zero shot pcgita speaker-level AUC (mean p_yes) | 0.5948 [0.4778, 0.7089] | 100 (100) | yes |
| M3_pcgita_clip_minus_speaker | Omni zero shot pcgita paired clip minus speaker AUC | -0.0313 [-0.0820, 0.0179] | 1100 (100) | yes |
| M3_neurovoz_speaker | Omni zero shot neurovoz speaker-level AUC (mean p_yes) | 0.7948 [0.7045, 0.8733] | 107 (107) | yes |
| M3_neurovoz_clip_minus_speaker | Omni zero shot neurovoz paired clip minus speaker AUC | -0.0997 [-0.1325, -0.0613] | 1270 (107) | yes |

### PART 16 M4 ({'SUPPORT': 12})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M4#02:median_answer_mass_Qwen2.5_Omni_conflict | median_answer_mass_Qwen2.5-Omni_conflict | 0.9965 | 483 (138) | yes |
| M4#04:median_answer_mass_Qwen2.5_Omni_agreement | median_answer_mass_Qwen2.5-Omni_agreement | 0.9969 | 483 (138) | yes |
| M4#07:median_answer_mass_Qwen2_Audio_conflict | median_answer_mass_Qwen2-Audio_conflict | 0.9990 | 483 (138) | yes |
| M4#09:median_answer_mass_Qwen2_Audio_agreement | median_answer_mass_Qwen2-Audio_agreement | 0.9988 | 483 (138) | yes |
| M4#12:median_answer_mass_Qwen3_Omni_30B_A3B_conflict | median_answer_mass_Qwen3-Omni-30B-A3B_conflict | 0.9996 | 483 (138) | yes |
| M4#14:median_answer_mass_Qwen3_Omni_30B_A3B_agreement | median_answer_mass_Qwen3-Omni-30B-A3B_agreement | 0.9998 | 483 (138) | yes |
| M4#17:median_answer_mass_AudioFlamingo2_conflict | median_answer_mass_AudioFlamingo2_conflict | 0.9424 | 483 (138) | yes |
| M4#19:median_answer_mass_AudioFlamingo2_agreement | median_answer_mass_AudioFlamingo2_agreement | 0.9423 | 483 (138) | yes |
| M4#22:median_answer_mass_AudioFlamingo3_conflict | median_answer_mass_AudioFlamingo3_conflict | 0.9249 | 483 (138) | yes |
| M4#24:median_answer_mass_AudioFlamingo3_agreement | median_answer_mass_AudioFlamingo3_agreement | 0.9403 | 483 (138) | yes |
| M4#27:median_answer_mass_Kimi_Audio_conflict | median_answer_mass_Kimi-Audio_conflict | 0.9992 | 483 (138) | yes |
| M4#29:median_answer_mass_Kimi_Audio_agreement | median_answer_mass_Kimi-Audio_agreement | 0.9992 | 483 (138) | yes |

### PART 16 M5 ({'SUPERSEDED / NOT USED': 20, 'SUPPORT': 9})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M5_all_cos | Qwen2-Audio cos(probe_dir, readout_dir) [all] | 0.0145 | 468 (228) | yes |
| M5_all_randfloor | Qwen2-Audio random-direction floor mean/cos/ [all] | 0.0124 [0.0005, 0.0343] | 468 (228) | yes |
| M5_all_auc_probe | Qwen2-Audio auc probe out-of-fold [all] | 0.8223 [0.7742, 0.8671] | 468 (228) | yes |
| M5_all_auc_d | Qwen2-Audio auc along readout direction [all] | 0.6174 [0.5539, 0.6800] | 468 (228) | yes |
| M5_all_auc_without_d | Qwen2-Audio auc probe with readout direction removed (coef proj) [all] | 0.8327 [0.7857, 0.8759] | 468 (228) | yes |
| M5_all_auc_without_d_Xproj | Qwen2-Audio auc probe with readout direction removed (X proj) [all] | 0.8223 [0.7743, 0.8672] | 468 (228) | yes |
| M5_conflict_refit_cos | Qwen2-Audio cos(probe_dir, readout_dir) [conflict_refit] | 0.0054 | 146 (100) | yes |
| M5_conflict_refit_randfloor | Qwen2-Audio random-direction floor mean/cos/ [conflict_refit] | 0.0123 [0.0005, 0.0350] | 146 (100) | yes |
| M5_conflict_refit_auc_probe | Qwen2-Audio auc probe out-of-fold [conflict_refit] | 0.6757 [0.5685, 0.7761] | 146 (100) | yes |
| M5_conflict_refit_auc_d | Qwen2-Audio auc along readout direction [conflict_refit] | 0.4136 [0.3010, 0.5424] | 146 (100) | yes |
| M5_conflict_refit_auc_without_d | Qwen2-Audio auc probe with readout direction removed (coef proj) [conflict_refit] | 0.6869 [0.5806, 0.7872] | 146 (100) | yes |
| M5_conflict_refit_auc_without_d_Xproj | Qwen2-Audio auc probe with readout direction removed (X proj) [conflict_refit] | 0.6759 [0.5687, 0.7766] | 146 (100) | yes |
| M5_agreement_refit_cos | Qwen2-Audio cos(probe_dir, readout_dir) [agreement_refit] | 0.0147 | 322 (175) | yes |
| M5_agreement_refit_randfloor | Qwen2-Audio random-direction floor mean/cos/ [agreement_refit] | 0.0126 [0.0005, 0.0359] | 322 (175) | yes |
| M5_agreement_refit_auc_probe | Qwen2-Audio auc probe out-of-fold [agreement_refit] | 0.9317 [0.9020, 0.9578] | 322 (175) | yes |
| M5_agreement_refit_auc_d | Qwen2-Audio auc along readout direction [agreement_refit] | 0.6984 [0.6373, 0.7592] | 322 (175) | yes |
| M5_agreement_refit_auc_without_d | Qwen2-Audio auc probe with readout direction removed (coef proj) [agreement_refit] | 0.9346 [0.9057, 0.9603] | 322 (175) | yes |
| M5_agreement_refit_auc_without_d_Xproj | Qwen2-Audio auc probe with readout direction removed (X proj) [agreement_refit] | 0.9325 [0.9027, 0.9583] | 322 (175) | yes |
| M5_conflict_restricted_auc_probe | Qwen2-Audio auc probe out-of-fold [conflict_restricted] | 0.6178 [0.5123, 0.7222] | 146 (100) | yes |
| M5_conflict_restricted_auc_d | Qwen2-Audio auc along readout direction [conflict_restricted] | 0.4136 [0.3010, 0.5424] | 146 (100) | yes |
| M5_conflict_restricted_auc_without_d | Qwen2-Audio auc probe with readout direction removed (coef proj) [conflict_restricted] | 0.6326 [0.5277, 0.7336] | 146 (100) | yes |
| M5_conflict_restricted_auc_without_d_Xproj | Qwen2-Audio auc probe with readout direction removed (X proj) [conflict_restricted] | 0.6180 [0.5128, 0.7213] | 146 (100) | yes |
| M5_agreement_restricted_auc_probe | Qwen2-Audio auc probe out-of-fold [agreement_restricted] | 0.8988 [0.8598, 0.9328] | 322 (175) | yes |
| M5_agreement_restricted_auc_d | Qwen2-Audio auc along readout direction [agreement_restricted] | 0.6984 [0.6373, 0.7592] | 322 (175) | yes |
| M5_agreement_restricted_auc_without_d | Qwen2-Audio auc probe with readout direction removed (coef proj) [agreement_restricted] | 0.9070 [0.8702, 0.9383] | 322 (175) | yes |
| M5_agreement_restricted_auc_without_d_Xproj | Qwen2-Audio auc probe with readout direction removed (X proj) [agreement_restricted] | 0.8988 [0.8593, 0.9326] | 322 (175) | yes |
| M5_paired_auc_probe_conflict_minus_agreement | Qwen2-Audio auc_probe conflict minus agreement (paired speaker bootstrap) | -0.2810 [-0.3799, -0.1762] | 468 (228) | yes |
| M5_paired_auc_d_conflict_minus_agreement | Qwen2-Audio auc_d conflict minus agreement (paired speaker bootstrap) | -0.2847 [-0.3947, -0.1475] | 468 (228) | yes |
| M5_paired_auc_without_d_conflict_minus_agreement | Qwen2-Audio auc_without_d conflict minus agreement (paired speaker bootstrap) | -0.2745 [-0.3754, -0.1707] | 468 (228) | yes |

### PART 16 M7 ({'SUPERSEDED / NOT USED': 5, 'SUPPORT': 5, 'RELEASED ONLY': 3})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M7_pitt_probe | Qwen2.5-Omni nested encoder probe AUC, pitt | 0.7969 [0.7412, 0.8512] | 468 (228) | yes |
| M7_pitt_gap | paired probe minus zero-shot AUC, pitt | 0.1391 [0.0680, 0.2084] | 468 (228) | yes |
| M7_adresso_probe | Qwen2.5-Omni nested encoder probe AUC, adresso | 0.8899 [0.8462, 0.9321] | 237 (237) | yes |
| M7_adress2020_probe | Qwen2.5-Omni nested encoder probe AUC, adress2020 | 0.7878 [0.7148, 0.8517] | 156 (156) | yes |
| M7_pcgita_probe | Qwen2.5-Omni nested encoder probe AUC, pcgita | 0.9089 [0.8478, 0.9563] | 1100 (100) | yes |
| M7_neurovoz_probe | Qwen2.5-Omni nested encoder probe AUC, neurovoz | 0.9160 [0.8713, 0.9513] | 1270 (107) | yes |
| M7_kcl_probe | Qwen2.5-Omni nested encoder probe AUC, kcl | 0.7768 [0.5994, 0.9265] | 37 (37) | yes |
| M7_edaic_probe | Qwen2.5-Omni nested encoder probe AUC, edaic | 0.6324 [0.5520, 0.7104] | 275 (275) | yes |
| M7_edaic_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, edaic | 0.6620 [0.5831, 0.7315] | 275 (275) | yes |
| M7_edaic_gap | paired probe minus zero-shot AUC, edaic | -0.0297 [-0.1291, 0.0752] | 275 (275) | yes |
| M7_mean_gap | mean paired probe minus zero-shot AUC across 7 datasets | 0.1681 | 3543 | yes |
| M7_pitt_probe_rep5 | Qwen2.5-Omni encoder probe AUC, pitt, mean of 5-repeat OOF probs (supplement) | 0.7917 [0.7324, 0.8480] | 468 (228) | yes |
| M7_pitt_gap_rep5 | paired probe minus zero-shot AUC, pitt, 5-repeat OOF (supplement) | 0.1339 [0.0641, 0.2019] | 468 (228) | yes |

### PART 16 M8 ({'SUPERSEDED / NOT USED': 18})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M8#02:Qwen2.5_Omni_A_edaic_part14_agreement_src_primary | Qwen2.5-Omni / A_edaic_part14 / agreement / src=primary | 0.9482 [0.9039, 0.9828] | 483 (138) | yes |
| M8#03:Qwen2.5_Omni_A_edaic_part14_conflict_minus_agreement_src_primary | Qwen2.5-Omni / A_edaic_part14 / conflict_minus_agreement / src=primary | -0.7264 [-0.7786, -0.6655] | 966 (138) | yes |
| M8#05:Qwen2_Audio_A_edaic_part14_agreement_src_primary | Qwen2-Audio / A_edaic_part14 / agreement / src=primary | 0.9478 [0.9128, 0.9762] | 483 (138) | yes |
| M8#06:Qwen2_Audio_A_edaic_part14_conflict_minus_agreement_src_primary | Qwen2-Audio / A_edaic_part14 / conflict_minus_agreement / src=primary | -0.7198 [-0.7689, -0.6679] | 966 (138) | yes |
| M8#08:Qwen3_Omni_30B_A3B_A_edaic_part14_agreement_src_primary | Qwen3-Omni-30B-A3B / A_edaic_part14 / agreement / src=primary | 0.9638 [0.9336, 0.9864] | 483 (138) | yes |
| M8#09:Qwen3_Omni_30B_A3B_A_edaic_part14_conflict_minus_agreement_src_primary | Qwen3-Omni-30B-A3B / A_edaic_part14 / conflict_minus_agreement / src=primary | -0.6630 [-0.7192, -0.6030] | 966 (138) | yes |
| M8#11:Audio_Flamingo_2_A_edaic_part14_agreement_src_primary | Audio Flamingo 2 / A_edaic_part14 / agreement / src=primary | 0.6083 [0.5126, 0.6997] | 483 (138) | yes |
| M8#12:Audio_Flamingo_2_A_edaic_part14_conflict_minus_agreement_src_primary | Audio Flamingo 2 / A_edaic_part14 / conflict_minus_agreement / src=primary | -0.0797 [-0.1488, -0.0104] | 966 (138) | yes |
| M8#14:Audio_Flamingo_3_A_edaic_part14_agreement_src_primary | Audio Flamingo 3 / A_edaic_part14 / agreement / src=primary | 0.8095 [0.7447, 0.8696] | 483 (138) | yes |
| M8#15:Audio_Flamingo_3_A_edaic_part14_conflict_minus_agreement_src_primary | Audio Flamingo 3 / A_edaic_part14 / conflict_minus_agreement / src=primary | -0.3860 [-0.4675, -0.3018] | 966 (138) | yes |
| M8#17:Kimi_Audio_A_edaic_part14_agreement_src_primary | Kimi-Audio / A_edaic_part14 / agreement / src=primary | 0.8744 [0.8172, 0.9223] | 483 (138) | yes |
| M8#18:Kimi_Audio_A_edaic_part14_conflict_minus_agreement_src_primary | Kimi-Audio / A_edaic_part14 / conflict_minus_agreement / src=primary | -0.3486 [-0.4248, -0.2795] | 966 (138) | yes |
| M8#31:Audio_Flamingo_2_B_pitt_conflict_src_af2_pitt_audio | Audio Flamingo 2 / B_pitt / conflict / src=af2_pitt_audio | 0.6691 [0.5650, 0.7654] | 146 (100) | yes |
| M8#32:Audio_Flamingo_2_B_pitt_agreement_src_af2_pitt_audio | Audio Flamingo 2 / B_pitt / agreement / src=af2_pitt_audio | 0.4895 [0.4182, 0.5591] | 322 (175) | yes |
| M8#33:Audio_Flamingo_2_B_pitt_conflict_minus_agreement_src_af2_pitt_audio | Audio Flamingo 2 / B_pitt / conflict_minus_agreement / src=af2_pitt_audio | 0.1796 [0.0702, 0.2825] | 468 (228) | yes |
| M8#40:Kimi_Audio_B_pitt_conflict_src_alt_overnight_copy | Kimi-Audio / B_pitt / conflict / src=alt_overnight_copy | 0.7078 [0.6037, 0.8083] | 146 (100) | yes |
| M8#41:Kimi_Audio_B_pitt_agreement_src_alt_overnight_copy | Kimi-Audio / B_pitt / agreement / src=alt_overnight_copy | 0.7014 [0.6309, 0.7730] | 322 (175) | yes |
| M8#42:Kimi_Audio_B_pitt_conflict_minus_agreement_src_alt_overnight_copy | Kimi-Audio / B_pitt / conflict_minus_agreement / src=alt_overnight_copy | 0.0064 [-0.1140, 0.1224] | 468 (228) | yes |

### PART 16 M9 ({'SUPPORT': 100, 'RELEASED ONLY': 2})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| M9.AudioFlamingo2.adress2020.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9676 [0.9664, 0.9689] | 156 (156) | yes |
| M9.AudioFlamingo2.adresso.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9614 [0.9607, 0.9628] | 237 (237) | yes |
| M9.AudioFlamingo2.edaic.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9392 [0.9383, 0.9407] | 275 (275) | yes |
| M9.AudioFlamingo2.edaic_conflict.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9453 [0.9437, 0.9479] | 390 (74) | yes |
| M9.AudioFlamingo2.edaic_conflict_v2.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9423 [0.9409, 0.9434] | 966 (138) | yes |
| M9.AudioFlamingo2.kcl.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9304 [0.9285, 0.9341] | 37 (37) | yes |
| M9.AudioFlamingo2.neurovoz.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9412 [0.9398, 0.9427] | 1270 (107) | yes |
| M9.AudioFlamingo2.pcgita.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9400 [0.9390, 0.9413] | 1100 (100) | yes |
| M9.AudioFlamingo2.pitt.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9626 [0.9614, 0.9639] | 468 (228) | yes |
| M9.AudioFlamingo3.adress2020.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9457 [0.9415, 0.9477] | 156 (156) | yes |
| M9.AudioFlamingo3.adresso.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9579 [0.9541, 0.9614] | 237 (237) | yes |
| M9.AudioFlamingo3.edaic.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9272 [0.9226, 0.9328] | 275 (275) | yes |
| M9.AudioFlamingo3.edaic_conflict.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9332 [0.9239, 0.9435] | 390 (74) | yes |
| M9.AudioFlamingo3.edaic_conflict_v2.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9311 [0.9271, 0.9361] | 966 (138) | yes |
| M9.AudioFlamingo3.edaic_full.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9413 [0.9342, 0.9476] | 275 (275) | yes |
| M9.AudioFlamingo3.edaic_mid30.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9266 [0.9192, 0.9333] | 275 (275) | yes |
| M9.AudioFlamingo3.edaic_mid300.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9398 [0.9334, 0.9441] | 275 (275) | yes |
| M9.AudioFlamingo3.kcl.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9434 [0.9374, 0.9501] | 37 (37) | yes |
| M9.AudioFlamingo3.neurovoz.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9132 [0.9076, 0.9177] | 1270 (107) | yes |
| M9.AudioFlamingo3.pcgita.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.8968 [0.8921, 0.9013] | 1100 (100) | yes |
| M9.AudioFlamingo3.pitt.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9487 [0.9463, 0.9505] | 468 (228) | yes |
| M9.Kimi-Audio.adress2020.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9971 [0.9967, 0.9976] | 156 (156) | yes |
| M9.Kimi-Audio.adresso.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9956 [0.9953, 0.9962] | 237 (237) | yes |
| M9.Kimi-Audio.edaic.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9981 [0.9980, 0.9982] | 275 (275) | yes |
| M9.Kimi-Audio.edaic_conflict.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9992 [0.9990, 0.9993] | 390 (74) | yes |
| M9.Kimi-Audio.edaic_conflict_v2.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9992 [0.9991, 0.9994] | 966 (138) | yes |
| M9.Kimi-Audio.edaic_full.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9983 [0.9981, 0.9984] | 275 (275) | yes |
| M9.Kimi-Audio.edaic_mid30.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9991 [0.9990, 0.9992] | 275 (275) | yes |
| M9.Kimi-Audio.edaic_mid300.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9987 [0.9985, 0.9989] | 275 (275) | yes |
| M9.Kimi-Audio.kcl.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9973 [0.9970, 0.9977] | 37 (37) | yes |
| M9.Kimi-Audio.neurovoz.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9961 [0.9961, 0.9962] | 1270 (107) | yes |
| M9.Kimi-Audio.pcgita.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9959 [0.9958, 0.9960] | 1100 (100) | yes |
| M9.Kimi-Audio.pitt.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9966 [0.9963, 0.9970] | 468 (228) | yes |
| M9.Qwen2-Audio.adress2020.transcript_only | median answer mass P(Yes)+P(No) | 0.9424 [0.9341, 0.9503] | 156 (156) | yes |
| M9.Qwen2-Audio.adresso.transcript_only | median answer mass P(Yes)+P(No) | 0.9110 [0.8975, 0.9225] | 237 (237) | yes |
| M9.Qwen2-Audio.conflict.transcript_only | median answer mass P(Yes)+P(No) | 0.9808 [0.9713, 0.9864] | 390 (74) | yes |
| M9.Qwen2-Audio.edaic.transcript_only | median answer mass P(Yes)+P(No) | 0.9595 [0.9497, 0.9643] | 275 (275) | yes |
| M9.Qwen2-Audio.edaic30.transcript_only | median answer mass P(Yes)+P(No) | 0.9595 [0.9497, 0.9643] | 275 (275) | yes |
| M9.Qwen2-Audio.edaicfull.transcript_only | median answer mass P(Yes)+P(No) | 0.9664 [0.9611, 0.9720] | 275 (275) | yes |
| M9.Qwen2-Audio.pitt.transcript_only | median answer mass P(Yes)+P(No) | 0.9300 [0.9236, 0.9362] | 468 (228) | yes |
| M9.Qwen2-Audio.adress2020.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 [0.9965, 0.9971] | 156 (156) | yes |
| M9.Qwen2-Audio.adresso.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9969 [0.9966, 0.9971] | 237 (237) | yes |
| M9.Qwen2-Audio.edaic.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9959 [0.9952, 0.9964] | 275 (275) | yes |
| M9.Qwen2-Audio.edaic_conflict.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9989 [0.9985, 0.9992] | 390 (74) | yes |
| M9.Qwen2-Audio.edaic_conflict_v2.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9989 [0.9987, 0.9992] | 966 (138) | yes |
| M9.Qwen2-Audio.edaic_full.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9970 [0.9965, 0.9975] | 275 (275) | yes |
| M9.Qwen2-Audio.edaic_mid30.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9990 [0.9987, 0.9993] | 275 (275) | yes |
| M9.Qwen2-Audio.edaic_mid300.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9972 [0.9969, 0.9976] | 275 (275) | yes |
| M9.Qwen2-Audio.kcl.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9991 [0.9989, 0.9994] | 37 (37) | yes |
| M9.Qwen2-Audio.neurovoz.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9992 [0.9992, 0.9992] | 1270 (107) | yes |
| M9.Qwen2-Audio.pcgita.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9994 [0.9994, 0.9994] | 1100 (1100) | yes |
| M9.Qwen2-Audio.pitt.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 [0.9965, 0.9969] | 468 (228) | yes |
| M9.Qwen2.5-Omni.adress2020.transcript_only | median answer mass P(Yes)+P(No) | 0.9968 [0.9967, 0.9969] | 156 (156) | yes |
| M9.Qwen2.5-Omni.adresso.transcript_only | median answer mass P(Yes)+P(No) | 0.9975 [0.9974, 0.9975] | 237 (237) | yes |
| M9.Qwen2.5-Omni.conflict.transcript_only | median answer mass P(Yes)+P(No) | 0.9986 [0.9984, 0.9988] | 390 (74) | yes |
| M9.Qwen2.5-Omni.edaic.transcript_only | median answer mass P(Yes)+P(No) | 0.9986 [0.9985, 0.9986] | 275 (275) | yes |
| M9.Qwen2.5-Omni.edaic30.transcript_only | median answer mass P(Yes)+P(No) | 0.9986 [0.9985, 0.9986] | 275 (275) | yes |
| M9.Qwen2.5-Omni.edaicfull.transcript_only | median answer mass P(Yes)+P(No) | 0.9989 [0.9988, 0.9990] | 275 (275) | yes |
| M9.Qwen2.5-Omni.kcl.transcript_only | median answer mass P(Yes)+P(No) | 0.9980 [0.9979, 0.9981] | 37 (37) | yes |
| M9.Qwen2.5-Omni.neurovoz.transcript_only | median answer mass P(Yes)+P(No) | 0.9973 [0.9973, 0.9974] | 1270 (107) | yes |
| M9.Qwen2.5-Omni.pcgita.transcript_only | median answer mass P(Yes)+P(No) | 0.9978 [0.9978, 0.9978] | 1100 (100) | yes |
| M9.Qwen2.5-Omni.pitt.transcript_only | median answer mass P(Yes)+P(No) | 0.9966 [0.9965, 0.9968] | 468 (228) | yes |
| M9.Qwen2.5-Omni.pitt_all.transcript_only | median answer mass P(Yes)+P(No) | 0.9966 [0.9965, 0.9966] | 708 (266) | yes |
| M9.Qwen2.5-Omni.adress2020.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 [0.9965, 0.9968] | 156 (156) | yes |
| M9.Qwen2.5-Omni.adresso.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 [0.9966, 0.9968] | 237 (237) | yes |
| M9.Qwen2.5-Omni.edaic.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9966 [0.9965, 0.9967] | 275 (275) | yes |
| M9.Qwen2.5-Omni.edaic_conflict.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 [0.9965, 0.9968] | 390 (74) | yes |
| M9.Qwen2.5-Omni.edaic_conflict_v2.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 [0.9966, 0.9968] | 966 (138) | yes |
| M9.Qwen2.5-Omni.edaic_full.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9962 [0.9960, 0.9964] | 275 (275) | yes |
| M9.Qwen2.5-Omni.edaic_mid30.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9958 [0.9956, 0.9960] | 275 (275) | yes |
| M9.Qwen2.5-Omni.edaic_mid300.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9958 [0.9956, 0.9962] | 275 (275) | yes |
| M9.Qwen2.5-Omni.kcl.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9888 [0.9881, 0.9903] | 37 (37) | yes |
| M9.Qwen2.5-Omni.neurovoz.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9952 [0.9952, 0.9953] | 1270 (107) | yes |
| M9.Qwen2.5-Omni.pcgita.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9952 [0.9951, 0.9953] | 1100 (100) | yes |
| M9.Qwen2.5-Omni.pitt.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9966 [0.9965, 0.9967] | 468 (228) | yes |
| M9.Qwen2.5-Omni.pitt_all.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 [0.9966, 0.9967] | 708 (266) | yes |
| M9.Qwen3-Omni-30B-A3B.adress2020.transcript_only | median answer mass P(Yes)+P(No) | 0.9997 [0.9997, 0.9998] | 156 (156) | yes |
| M9.Qwen3-Omni-30B-A3B.adresso.transcript_only | median answer mass P(Yes)+P(No) | 0.9998 [0.9997, 0.9998] | 237 (237) | yes |
| M9.Qwen3-Omni-30B-A3B.conflict.transcript_only | median answer mass P(Yes)+P(No) | 0.9999 [0.9999, 0.9999] | 390 (74) | yes |
| M9.Qwen3-Omni-30B-A3B.edaic.transcript_only | median answer mass P(Yes)+P(No) | 0.9999 [0.9998, 0.9999] | 275 (275) | yes |
| M9.Qwen3-Omni-30B-A3B.edaic30.transcript_only | median answer mass P(Yes)+P(No) | 0.9999 [0.9998, 0.9999] | 275 (275) | yes |
| M9.Qwen3-Omni-30B-A3B.edaicfull.transcript_only | median answer mass P(Yes)+P(No) | 0.9999 [0.9999, 0.9999] | 275 (275) | yes |
| M9.Qwen3-Omni-30B-A3B.kcl.transcript_only | median answer mass P(Yes)+P(No) | 0.9997 [0.9996, 0.9998] | 37 (37) | yes |
| M9.Qwen3-Omni-30B-A3B.neurovoz.transcript_only | median answer mass P(Yes)+P(No) | 0.9996 [0.9996, 0.9996] | 1270 (107) | yes |
| M9.Qwen3-Omni-30B-A3B.pcgita.transcript_only | median answer mass P(Yes)+P(No) | 0.9997 [0.9997, 0.9997] | 1100 (100) | yes |
| M9.Qwen3-Omni-30B-A3B.pitt.transcript_only | median answer mass P(Yes)+P(No) | 0.9997 [0.9997, 0.9997] | 468 (228) | yes |
| M9.Qwen3-Omni-30B-A3B.adress2020.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9995 [0.9995, 0.9996] | 156 (156) | yes |
| M9.Qwen3-Omni-30B-A3B.adresso.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9995 [0.9995, 0.9996] | 237 (237) | yes |
| M9.Qwen3-Omni-30B-A3B.edaic.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9997 [0.9997, 0.9997] | 275 (275) | yes |
| M9.Qwen3-Omni-30B-A3B.edaic_conflict.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9996 [0.9996, 0.9997] | 390 (74) | yes |
| M9.Qwen3-Omni-30B-A3B.edaic_conflict_v2.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9997 [0.9996, 0.9997] | 966 (138) | yes |
| M9.Qwen3-Omni-30B-A3B.edaic_full.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9992 [0.9991, 0.9993] | 275 (275) | yes |
| M9.Qwen3-Omni-30B-A3B.edaic_mid30.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9995 [0.9995, 0.9996] | 275 (275) | yes |
| M9.Qwen3-Omni-30B-A3B.edaic_mid300.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9995 [0.9994, 0.9996] | 275 (275) | yes |
| M9.Qwen3-Omni-30B-A3B.kcl.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9996 [0.9995, 0.9996] | 37 (37) | yes |
| M9.Qwen3-Omni-30B-A3B.neurovoz.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9997 [0.9997, 0.9997] | 1270 (107) | yes |
| M9.Qwen3-Omni-30B-A3B.pcgita.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9997 [0.9997, 0.9997] | 1100 (100) | yes |
| M9.Qwen3-Omni-30B-A3B.pitt.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9995 [0.9995, 0.9995] | 468 (228) | yes |
| M9.MINIMUM | minimum median answer mass across the published grid (Audio Flamingo 3 / pcgita / zero shot answer) | 0.8968 [0.8921, 0.9013] | 1100 (100) | yes |
| M9.SWEEPMIN | minimum median answer mass across every mass-bearing file on disk (Audio Flamingo 2 / kcl30b) | 0.8662 | 37 (37) | yes |
| M9.CELLS | master_lookup cells with an answer mass recorded | 98 | 221 | yes |
| M9.FILES | per-clip score files on disk with an answer mass recorded | 307 | 608 | yes |

### PART 16 MEDAIC ({'RELEASED ONLY': 14, 'SUPERSEDED / NOT USED': 4, 'SUPPORT': 1})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| MEDAIC_full_text_minus_answer | PAIRED transcript-only readout minus answer, full window (excl0=False) | -0.0055 [-0.0566, 0.0431] | 275 (275) | yes |
| MEDAIC_full_llm_nested_NOCI | LM probe nested mean, full window (json aggregate, NOT recomputed, no per-clip file) | 0.6853 | 275 (275) | yes |
| MEDAIC_full_llm_minus_answer_POINT | POINT-ONLY LM probe minus answer, full window (no interval possible) | -0.1432 | 275 (275) | yes |
| MEDAIC_full_enc_nested_NOCI | encoder probe nested mean, full window (json aggregate, NOT recomputed, no per-clip file) | 0.5930 | 275 (275) | yes |
| MEDAIC_full_enc_minus_answer_POINT | POINT-ONLY encoder probe minus answer, full window (no interval possible) | -0.2355 | 275 (275) | yes |
| MEDAIC_full_ans_nested_NOCI | answer-state probe nested mean, full window (json aggregate, NOT recomputed, no per-clip file) | 0.7542 | 275 (275) | yes |
| MEDAIC_full_ans_minus_answer_POINT | POINT-ONLY answer-state probe minus answer, full window (no interval possible) | -0.0743 | 275 (275) | yes |
| MEDAIC_full_proj_nested_NOCI | projector probe nested mean, full window (json aggregate, NOT recomputed, no per-clip file) | 0.5293 | 275 (275) | yes |
| MEDAIC_full_proj_minus_answer_POINT | POINT-ONLY projector probe minus answer, full window (no interval possible) | -0.2992 | 275 (275) | yes |
| MEDAIC_w30_enc | encoder probe nested OOF AUC, 30 s window | 0.6324 [0.5520, 0.7104] | 275 (275) | yes |
| MEDAIC_w30_enc_minus_answer | PAIRED encoder probe minus answer, 30 s window (excl0=False) | -0.0297 [-0.1291, 0.0752] | 275 (275) | yes |
| MEDAIC_w30_llm | LM probe nested OOF AUC, 30 s window | 0.5065 [0.4221, 0.5856] | 275 (275) | yes |
| MEDAIC_w30_llm_minus_answer | PAIRED LM probe minus answer, 30 s window (excl0=True) | -0.1555 [-0.2536, -0.0577] | 275 (275) | yes |
| MEDAIC_w30_ans | answer-state probe nested OOF AUC, 30 s window | 0.6401 [0.5581, 0.7185] | 275 (275) | yes |
| MEDAIC_w30_ans_minus_answer | PAIRED answer-state probe minus answer, 30 s window (excl0=False) | -0.0219 [-0.1053, 0.0663] | 275 (275) | yes |
| MEDAIC_w30_proj | projector probe nested OOF AUC, 30 s window | 0.5182 [0.4411, 0.5999] | 275 (275) | yes |
| MEDAIC_w30_proj_minus_answer | PAIRED projector probe minus answer, 30 s window (excl0=True) | -0.1438 [-0.2490, -0.0320] | 275 (275) | yes |
| MEDAIC_w30_answer | Omni zero-shot answer AUC, 30 s window | 0.6620 [0.5831, 0.7315] | 275 (275) | yes |
| MEDAIC_answer_full_minus_30s | PAIRED answer AUC full minus 30 s, same 275 speakers (excl0=True) | 0.1664 [0.0848, 0.2492] | 275 (275) | yes |

### PART 16 POD1 ({'SUPERSEDED / NOT USED': 3, 'RELEASED ONLY': 3})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| 1a_o25_text_agreement_auc | text-prompt agreement-arm AUC, o25 | 0.9596 [0.9306, 0.9823] | 483 (138) | yes |
| 1a_q2a_text_agreement_auc | text-prompt agreement-arm AUC, q2a | 0.9081 [0.8515, 0.9547] | 483 (138) | yes |
| VALIDATION_agreement_auc | pipeline validation, rebuilt 0.70 set, original wording (reproduces Table 2 audio) agreement arm AUC | 0.9482 [0.9039, 0.9828] | 483 (138) | yes |
| 1c_conflict_auc | 1c sft_full projector RETRAIN, out-of-fold by speaker, Omni audio conflict arm AUC | 0.2283 [0.1650, 0.2980] | 483 (138) | yes |
| 1c_agreement_auc | 1c sft_full projector RETRAIN, out-of-fold by speaker, Omni audio agreement arm AUC | 0.7930 [0.7062, 0.8669] | 483 (138) | yes |
| 1c_edaic_oof_auc_retrain | 1c RETRAIN E-DAIC out-of-fold AUC (original sft_full was 0.6421; input audio differs) | 0.5315 | 275 (275) | yes |

### PART 16 POD2 ({'SUPPORT': 5, 'SUPERSEDED / NOT USED': 4, 'RELEASED ONLY': 2})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| P16.POD2.V5 | projector-FT paired conflict-minus-agreement | -0.3404 [-0.4468, -0.2262] | 468 (228) | yes |
| P16.POD2.X1 | PRE-EXISTING LoRA(r8,qkvo) Pitt OOF AUC, recomputed from file | 0.8202 [0.7690, 0.8679] | 468 (228) | yes |
| P16.POD2.X2 | PRE-EXISTING LoRA conflict-arm AUC, recomputed | 0.5884 [0.4680, 0.7093] | 146 (100) | yes |
| P16.POD2.X3 | PRE-EXISTING LoRA agreement-arm AUC, recomputed | 0.9011 [0.8592, 0.9386] | 322 (175) | yes |
| P16.POD2.X4 | PRE-EXISTING LoRA paired conflict-minus-agreement | -0.3126 [-0.4294, -0.1911] | 468 (228) | yes |
| P16.POD2.P3 | projector-only (thinker.audio_tower.proj) trainable params | 4591104 | 468 (228) | yes |
| P16.POD2.P4 | projector trainable as % of full Qwen2.5-Omni-7B | 0.0428 | 468 (228) | yes |
| P16.POD2.L5 | POD2 LoRA training wall-clock, min per fold (0..4) | 4.13;4.07;4.05;4.03;4.00 | 468 (228) | yes |
| P16.POD2.L6 | POD2 LoRA peak GPU memory allocated, GB | 21.0130 | 468 (228) | yes |
| P16.POD2.L7 | POD2 fresh LoRA OOF AUC using the paper multi-variant p_yes (Yes/yes/YES vs No/no/NO, bare+leading-space) | 0.8220 [0.7702, 0.8711] | 468 (228) | yes |
| P16.POD2.L8 | POD2 fresh LoRA median answer_mass at the first answer position | 0.9988 | 468 (228) | yes |

### PART 16 POD3 ({'RELEASED ONLY': 16, 'SUPERSEDED / NOT USED': 7, 'SUPPORT': 3})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| 3b_all_auc_wo_d | Q3O Pitt probe AUC with the readout direction removed, all clips | 0.8305 [0.7826, 0.8737] | 468 (228) | yes |
| 3b_conflict_cos | Q3O Pitt readout-direction cosine, CONFLICT arm (random floor 0.0177 [0.0008,0.0516]) | -0.0067 | 146 (100) | yes |
| 3b_conflict_auc_probe | Q3O Pitt answer-state probe AUC, CONFLICT arm | 0.6145 [0.5021, 0.7246] | 146 (100) | yes |
| 3b_conflict_auc_d | Q3O Pitt AUC of the readout direction alone, CONFLICT arm | 0.6029 [0.4965, 0.7025] | 146 (100) | yes |
| 3b_conflict_auc_wo_d | Q3O Pitt probe AUC with the readout direction removed, CONFLICT arm | 0.6196 [0.5094, 0.7283] | 146 (100) | yes |
| 3b_agreement_cos | Q3O Pitt readout-direction cosine, AGREEMENT arm (random floor 0.0172 [0.0007,0.0480]) | 0.0086 | 322 (175) | yes |
| 3b_agreement_auc_probe | Q3O Pitt answer-state probe AUC, AGREEMENT arm | 0.9349 [0.9034, 0.9614] | 322 (175) | yes |
| 3b_agreement_auc_d | Q3O Pitt AUC of the readout direction alone, AGREEMENT arm | 0.8264 [0.7741, 0.8731] | 322 (175) | yes |
| 3b_agreement_auc_wo_d | Q3O Pitt probe AUC with the readout direction removed, AGREEMENT arm | 0.9341 [0.9021, 0.9604] | 322 (175) | yes |
| 3a_pitt_zeroshot_pod | Q3O Pitt zero-shot answer AUC, THIS pod (rank formula, deterministic, reproduced twice bit-exact) | 0.7673 | 468 (228) | yes |
| 3a_pitt_zeroshot_shipped | Q3O Pitt zero-shot answer AUC, SHIPPED file recomputed here with the rank formula | 0.7619 | 468 (228) | yes |
| 3a_pitt_enc_best_stage | Q3O Pitt ENCODER best single stage = layer 30 (descriptive max, oracle-selected) | 0.8276 [0.7765, 0.8760] | 468 (228) | yes |
| 3a_pitt_proj_nested | Q3O Pitt PROJECTOR probe AUC (single point, no layer choice) | 0.7820 [0.7228, 0.8368] | 468 (228) | yes |
| 3a_pitt_llm_nested | Q3O Pitt LANGUAGE MODEL nested probe AUC (5-repeat mean, layer chosen inside the training folds) | 0.8445 [0.8003, 0.8878] | 468 (228) | yes |
| 3a_pitt_llm_best_stage | Q3O Pitt LANGUAGE MODEL best single stage = thinker layer 13 (descriptive max, oracle-selected) | 0.8634 [0.8179, 0.9062] | 468 (228) | yes |
| 3a_pitt_ans_nested | Q3O Pitt ANSWER STATE nested probe AUC (5-repeat mean, stage chosen inside the training folds) | 0.8237 [0.7827, 0.8641] | 468 (228) | yes |
| 3a_pitt_ans_best_stage | Q3O Pitt ANSWER STATE best single stage = thinker layer 26 (descriptive max, oracle-selected) | 0.8485 [0.8069, 0.8882] | 468 (228) | yes |
| 3a_pcgita_zeroshot_pod | Q3O PC-GITA zero-shot answer AUC, THIS pod (re-extracted from original audio) | 0.6015 | 1100 (100) | yes |
| 3a_attn_sdpa | Q3O Pitt zero-shot AUC, sdpa attention kernel (mechanism test, all else fixed) | 0.7673 | 468 (228) | yes |
| 3a_attn_eager | Q3O Pitt zero-shot AUC, eager attention kernel (mechanism test, all else fixed) | 0.7679 | 468 (228) | yes |
| 3a_pcgita_enc_best_stage | Q3O PC-GITA ENCODER best single stage = audio encoder layer 14 (descriptive max, oracle-selected) | 0.9133 [0.8519, 0.9604] | 1100 (100) | yes |
| 3a_pcgita_proj_nested | Q3O PC-GITA PROJECTOR probe AUC (single point, no layer choice; 0.8872 in memory, see DISCREPANCIES LOW) | 0.8873 [0.8156, 0.9431] | 1100 (100) | yes |
| 3a_pcgita_llm_nested | Q3O PC-GITA LANGUAGE MODEL nested probe AUC (5-repeat mean, layer chosen inside the training folds) | 0.8772 [0.8117, 0.9309] | 1100 (100) | yes |
| 3a_pcgita_llm_best_stage | Q3O PC-GITA LANGUAGE MODEL best single stage = thinker layer 14 (descriptive max, oracle-selected) | 0.8961 [0.8341, 0.9463] | 1100 (100) | NO (twin gives 0.8961) |
| 3a_pcgita_ans_nested | Q3O PC-GITA ANSWER STATE nested probe AUC (5-repeat mean, stage chosen inside the training folds) | 0.8814 [0.8171, 0.9330] | 1100 (100) | yes |
| 3a_pcgita_ans_best_stage | Q3O PC-GITA ANSWER STATE best single stage = thinker layer 15 (descriptive max, oracle-selected) | 0.8962 [0.8327, 0.9452] | 1100 (100) | yes |

### PART 16 POD4 ({'RELEASED ONLY': 28, 'SUPPORT': 19, 'SUPERSEDED / NOT USED': 1})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| POD4_4b_resolved | Pitt 468 windows resolvable to CHAT timings | 468 | 468 (228) | yes |
| POD4_4b_invwin | Pitt windows containing timed INV speech in the heard 30s | 158 | 468 (228) | yes |
| POD4_4b_invshare | mean INV share of the heard 30s window | 0.0200 | 468 (228) | yes |
| POD4_4b_parshare | mean PAR share of the heard 30s window | 0.5871 | 468 (228) | yes |
| POD4_4b_pardur | mean participant-only duration after re-cut (s) | 17.6135 | 468 (228) | yes |
| POD4_4b_under10s | re-cut windows with under 10s of participant speech | 68 | 468 (228) | yes |
| POD4_4b_pub_enc | Pitt published Omni encoder probe AUC, recomputed rank formula | 0.7969 [0.7412, 0.8512] | 468 (228) | yes |
| POD4_4b_A_zs | Pitt arm A original cut, Omni zero-shot AUC | 0.6497 [0.5959, 0.7056] | 468 (228) | yes |
| POD4_4b_A_enc | Pitt arm A original cut, Omni encoder nested probe AUC | 0.7949 [0.7431, 0.8446] | 468 (228) | yes |
| POD4_4b_B_zs | Pitt arm B interviewer removed PAR-only, Omni zero-shot AUC | 0.7230 [0.6664, 0.7753] | 468 (228) | yes |
| POD4_4b_B_enc | Pitt arm B interviewer removed PAR-only, Omni encoder nested probe AUC | 0.7497 [0.6940, 0.8035] | 468 (228) | yes |
| POD4_4b_C_zs | Pitt arm C duration-matched WITH interviewer, Omni zero-shot AUC | 0.6145 [0.5582, 0.6683] | 468 (228) | yes |
| POD4_4b_C_enc | Pitt arm C duration-matched WITH interviewer, Omni encoder nested probe AUC | 0.7980 [0.7448, 0.8489] | 468 (228) | yes |
| POD4_4b_pd_zs_AB | PAIRED zero-shot delta, original cut to interviewer removed | 0.0732 [0.0312, 0.1165] | 468 (228) | yes |
| POD4_4b_pd_zs_CB | PAIRED zero-shot delta, duration-matched WITH interviewer to interviewer removed | 0.1084 [0.0638, 0.1550] | 468 (228) | yes |
| POD4_4b_pd_enc_AB | PAIRED encoder probe delta, original cut to interviewer removed | -0.0452 [-0.0857, -0.0083] | 468 (228) | yes |
| POD4_4b_pd_enc_CB | PAIRED encoder probe delta, duration-matched WITH interviewer to interviewer removed | -0.0483 [-0.0907, -0.0062] | 468 (228) | yes |
| POD4_4b_pd_zs_AC | PAIRED zero-shot delta, duration control only 30s to 17.6s both WITH interviewer | -0.0352 [-0.0729, -0.0001] | 468 (228) | yes |
| POD4_4b_pd_enc_AC | PAIRED encoder probe delta, duration control only 30s to 17.6s both WITH interviewer | 0.0032 [-0.0367, 0.0394] | 468 (228) | yes |
| POD4_4b_B_zs_conf | Pitt arm B zero-shot AUC, conflict arm | 0.5408 [0.4245, 0.6622] | 146 (100) | yes |
| POD4_4b_B_enc_conf | Pitt arm B encoder probe AUC, conflict arm | 0.6295 [0.5111, 0.7345] | 146 (100) | yes |
| POD4_4b_B_zs_agr | Pitt arm B zero-shot AUC, agreement arm | 0.8007 [0.7500, 0.8501] | 322 (175) | yes |
| POD4_4b_B_enc_agr | Pitt arm B encoder probe AUC, agreement arm | 0.8037 [0.7452, 0.8539] | 322 (175) | yes |
| POD4_4b_A_zs_conf | Pitt arm A zero-shot AUC, conflict arm | 0.5298 [0.4189, 0.6386] | 146 (100) | yes |
| POD4_4b_A_enc_conf | Pitt arm A encoder probe AUC, conflict arm | 0.6161 [0.5060, 0.7281] | 146 (100) | yes |
| POD4_4b_A_zs_agr | Pitt arm A zero-shot AUC, agreement arm | 0.6952 [0.6327, 0.7560] | 322 (175) | yes |
| POD4_4b_A_enc_agr | Pitt arm A encoder probe AUC, agreement arm | 0.8674 [0.8201, 0.9096] | 322 (175) | yes |
| POD4_4a_nv_nf_before | NeuroVoz noise floor alone, direction-free AUC, BEFORE | 0.5410 [0.4731, 0.6049] | 1270 (107) | yes |
| POD4_4a_nv_nf_after | NeuroVoz noise floor alone, direction-free AUC, AFTER | 0.5407 [0.4594, 0.6191] | 1270 (107) | yes |
| POD4_4a_nv_snr_before | NeuroVoz SNR alone, direction-free AUC, BEFORE | 0.5432 [0.4640, 0.6216] | 1270 (107) | yes |
| POD4_4a_nv_snr_after | NeuroVoz SNR alone, direction-free AUC, AFTER | 0.5305 [0.4498, 0.6080] | 1270 (107) | yes |
| POD4_4a_nv_cen_after | NeuroVoz spectral centroid alone, direction-free AUC, AFTER | 0.5603 [0.5051, 0.6188] | 1270 (107) | yes |
| POD4_4a_nv_ro_before | NeuroVoz spectral roll-off alone, direction-free AUC, BEFORE | 0.8005 [0.7370, 0.8574] | 1270 (107) | yes |
| POD4_4a_nv_ro_after | NeuroVoz spectral roll-off alone, direction-free AUC, AFTER | 0.5938 [0.5438, 0.6448] | 1270 (107) | yes |
| POD4_4a_nv_tilt_before | NeuroVoz spectral tilt alone, direction-free AUC, BEFORE | 0.7020 [0.6451, 0.7552] | 1270 (107) | yes |
| POD4_4a_nv_tilt_after | NeuroVoz spectral tilt alone, direction-free AUC, AFTER | 0.5808 [0.5369, 0.6259] | 1270 (107) | yes |
| POD4_4a_pg_nf_before | PC-GITA noise floor alone, direction-free AUC, BEFORE | 0.6627 [0.5885, 0.7362] | 1100 (100) | yes |
| POD4_4a_pg_nf_after | PC-GITA noise floor alone, direction-free AUC, AFTER | 0.5734 [0.5113, 0.6341] | 1100 (100) | yes |
| POD4_4a_pg_snr_before | PC-GITA SNR alone, direction-free AUC, BEFORE | 0.5873 [0.5261, 0.6464] | 1100 (100) | yes |
| POD4_4a_pg_snr_after | PC-GITA SNR alone, direction-free AUC, AFTER | 0.5735 [0.5158, 0.6286] | 1100 (100) | yes |
| POD4_4a_pg_cen_before | PC-GITA spectral centroid alone, direction-free AUC, BEFORE | 0.5539 [0.4997, 0.6104] | 1100 (100) | yes |
| POD4_4a_pg_cen_after | PC-GITA spectral centroid alone, direction-free AUC, AFTER | 0.5591 [0.5029, 0.6136] | 1100 (100) | yes |
| POD4_4a_pg_ro_before | PC-GITA spectral roll-off alone, direction-free AUC, BEFORE | 0.5533 [0.5000, 0.6081] | 1100 (100) | yes |
| POD4_4a_pg_ro_after | PC-GITA spectral roll-off alone, direction-free AUC, AFTER | 0.5657 [0.5102, 0.6189] | 1100 (100) | yes |
| POD4_4a_pg_tilt_before | PC-GITA spectral tilt alone, direction-free AUC, BEFORE | 0.5610 [0.5048, 0.6194] | 1100 (100) | yes |
| POD4_4a_pg_tilt_after | PC-GITA spectral tilt alone, direction-free AUC, AFTER | 0.5526 [0.4987, 0.6030] | 1100 (100) | yes |
| POD4_4b_invcue | Pitt interviewer milliseconds alone as a predictor of dementia, AUC | 0.6427 | 468 (228) | yes |
| POD4_4b_durcue | Pitt participant-only duration alone, direction-free AUC (residual cue in arm B) | 0.6648 | 468 (228) | yes |

### PART 16 POD4B ({'RELEASED ONLY': 47})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| 4b_adress2020_A_orig_zs | adress2020 arm A_orig Omni zero-shot AUC | 0.6272 [0.5379, 0.7165] | 156 (156) | yes |
| 4b_adress2020_A_orig_enc | adress2020 arm A_orig Omni encoder nested probe AUC | 0.7959 [0.7251, 0.8602] | 156 (156) | yes |
| 4b_adress2020_B_paronly_zs | adress2020 arm B_paronly Omni zero-shot AUC | 0.6524 [0.5648, 0.7380] | 156 (156) | yes |
| 4b_adress2020_B_paronly_enc | adress2020 arm B_paronly Omni encoder nested probe AUC | 0.7641 [0.6921, 0.8350] | 156 (156) | yes |
| 4b_adress2020_C_durmatch_zs | adress2020 arm C_durmatch Omni zero-shot AUC | 0.6313 [0.5435, 0.7141] | 156 (156) | yes |
| 4b_adress2020_C_durmatch_enc | adress2020 arm C_durmatch Omni encoder nested probe AUC | 0.8075 [0.7361, 0.8705] | 156 (156) | yes |
| 4b_adress2020_published_enc | adress2020 arm published Omni encoder nested probe AUC | 0.7878 [0.7148, 0.8517] | 156 (156) | yes |
| 4b_adress2020_paired_zs_A_to_B | adress2020 PAIRED zero-shot: original cut -> interviewer removed | 0.0251 [-0.0254, 0.0769] | 156 (156) | yes |
| 4b_adress2020_paired_zs_A_to_C | adress2020 PAIRED zero-shot: original cut -> duration-matched, interviewer KEPT | 0.0041 [-0.0386, 0.0473] | 156 (156) | yes |
| 4b_adress2020_paired_zs_C_to_B | adress2020 PAIRED zero-shot: duration-matched WITH interviewer -> interviewer removed | 0.0210 [-0.0325, 0.0756] | 156 (156) | yes |
| 4b_adress2020_paired_enc_A_to_B | adress2020 PAIRED encoder probe: original cut -> interviewer removed | -0.0317 [-0.0761, 0.0102] | 156 (156) | yes |
| 4b_adress2020_paired_enc_A_to_C | adress2020 PAIRED encoder probe: original cut -> duration-matched, interviewer KEPT | 0.0117 [-0.0229, 0.0449] | 156 (156) | yes |
| 4b_adress2020_paired_enc_C_to_B | adress2020 PAIRED encoder probe: duration-matched WITH interviewer -> interviewer removed | -0.0434 [-0.0888, -0.0033] | 156 (156) | yes |
| 4b_adress2020_gap_A_orig | adress2020 arm A_orig GAP = encoder probe AUC minus zero-shot AUC | 0.1686 [0.0627, 0.2735] | 156 (156) | yes |
| 4b_adress2020_gap_B_paronly | adress2020 arm B_paronly GAP = encoder probe AUC minus zero-shot AUC | 0.1118 [0.0020, 0.2144] | 156 (156) | yes |
| 4b_adress2020_gap_C_durmatch | adress2020 arm C_durmatch GAP = encoder probe AUC minus zero-shot AUC | 0.1762 [0.0739, 0.2792] | 156 (156) | yes |
| 4b_adress2020_gap_published | adress2020 arm published GAP = encoder probe AUC minus zero-shot AUC | 0.1637 [0.0557, 0.2693] | 156 (156) | yes |
| 4b_adress2020_inv_share_mean | adress2020 mean interviewer share of the original window | 0.0853 | 156 (156) | yes |
| 4b_adress2020_inv_share_median | adress2020 median interviewer share of the original window | 0.0501 | 156 (156) | yes |
| 4b_adress2020_inv_any_count | adress2020 windows containing any interviewer speech | 114 | 156 (156) | yes |
| 4b_adress2020_inv_gt5s_count | adress2020 windows containing more than 5 s of interviewer speech | 22 | 156 (156) | yes |
| 4b_adress2020_dur_removed_mean_s | adress2020 mean seconds removed going from arm A to arm B | 2.3600 | 156 (156) | yes |
| 4b_adress2020_par_under10s_count | adress2020 windows with under 10 s of participant speech | 1 | 156 (156) | yes |
| 4b_adresso_A_orig_zs | adresso arm A_orig Omni zero-shot AUC | 0.6291 [0.5404, 0.7134] | 161 (161) | yes |
| 4b_adresso_A_orig_enc | adresso arm A_orig Omni encoder nested probe AUC | 0.8624 [0.8063, 0.9155] | 161 (161) | yes |
| 4b_adresso_B_paronly_zs | adresso arm B_paronly Omni zero-shot AUC | 0.6302 [0.5461, 0.7202] | 161 (161) | yes |
| 4b_adresso_B_paronly_enc | adresso arm B_paronly Omni encoder nested probe AUC | 0.8242 [0.7582, 0.8822] | 161 (161) | yes |
| 4b_adresso_C_durmatch_zs | adresso arm C_durmatch Omni zero-shot AUC | 0.6083 [0.5194, 0.6999] | 161 (161) | yes |
| 4b_adresso_C_durmatch_enc | adresso arm C_durmatch Omni encoder nested probe AUC | 0.8726 [0.8183, 0.9225] | 161 (161) | yes |
| 4b_adresso_published_zs | adresso arm published Omni zero-shot AUC | 0.6320 [0.5448, 0.7177] | 161 (161) | yes |
| 4b_adresso_published_enc | adresso arm published Omni encoder nested probe AUC | 0.9046 [0.8548, 0.9479] | 161 (161) | yes |
| 4b_adresso_paired_zs_A_to_B | adresso PAIRED zero-shot: original cut -> interviewer removed | 0.0011 [-0.0441, 0.0462] | 161 (161) | yes |
| 4b_adresso_paired_zs_A_to_C | adresso PAIRED zero-shot: original cut -> duration-matched, interviewer KEPT | -0.0208 [-0.0610, 0.0196] | 161 (161) | yes |
| 4b_adresso_paired_zs_C_to_B | adresso PAIRED zero-shot: duration-matched WITH interviewer -> interviewer removed | 0.0219 [-0.0253, 0.0683] | 161 (161) | yes |
| 4b_adresso_paired_enc_A_to_B | adresso PAIRED encoder probe: original cut -> interviewer removed | -0.0382 [-0.0693, -0.0094] | 161 (161) | yes |
| 4b_adresso_paired_enc_A_to_C | adresso PAIRED encoder probe: original cut -> duration-matched, interviewer KEPT | 0.0103 [-0.0128, 0.0326] | 161 (161) | yes |
| 4b_adresso_paired_enc_C_to_B | adresso PAIRED encoder probe: duration-matched WITH interviewer -> interviewer removed | -0.0485 [-0.0832, -0.0161] | 161 (161) | yes |
| 4b_adresso_gap_A_orig | adresso arm A_orig GAP = encoder probe AUC minus zero-shot AUC | 0.2333 [0.1450, 0.3262] | 161 (161) | yes |
| 4b_adresso_gap_B_paronly | adresso arm B_paronly GAP = encoder probe AUC minus zero-shot AUC | 0.1940 [0.0976, 0.2900] | 161 (161) | yes |
| 4b_adresso_gap_C_durmatch | adresso arm C_durmatch GAP = encoder probe AUC minus zero-shot AUC | 0.2644 [0.1687, 0.3602] | 161 (161) | yes |
| 4b_adresso_gap_published | adresso arm published GAP = encoder probe AUC minus zero-shot AUC | 0.2726 [0.1783, 0.3674] | 161 (161) | yes |
| 4b_adresso_inv_share_mean | adresso mean interviewer share of the original window | 0.0825 | 161 (161) | yes |
| 4b_adresso_inv_share_median | adresso median interviewer share of the original window | 0.0395 | 161 (161) | yes |
| 4b_adresso_inv_any_count | adresso windows containing any interviewer speech | 105 | 161 (161) | yes |
| 4b_adresso_inv_gt5s_count | adresso windows containing more than 5 s of interviewer speech | 21 | 161 (161) | yes |
| 4b_adresso_dur_removed_mean_s | adresso mean seconds removed going from arm A to arm B | 2.2145 | 161 (161) | yes |
| 4b_adresso_par_under10s_count | adresso windows with under 10 s of participant speech | 4 | 161 (161) | yes |

### PART 16 SEED ({'RELEASED ONLY': 4, 'SUPERSEDED / NOT USED': 3, 'SUPPORT': 2})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| SEED01_1a | Omni transcript-only, 966 conflict clips, pooled | 0.6077 | 966 (138) | yes |
| SEED03_1a | Omni transcript-only, agreement arm | 0.9596 [0.9306, 0.9823] | 483 (138) | yes |
| SEED04_1a | Qwen2-Audio transcript-only, pooled | 0.5995 | 966 (138) | yes |
| SEED06_1a | Qwen2-Audio transcript-only, agreement arm | 0.9081 [0.8515, 0.9547] | 483 (138) | yes |
| SEED10_3a | Qwen3-Omni Pitt projector, nested (1 repeat only) | 0.7820 | 468 (228) | yes |
| SEED11_3a | Qwen3-Omni Pitt LM best, nested, 5 repeats (per-repeat [0.8397,0.8460,0.8393,0.8505,0.8468]) | 0.8445 | 468 (228) | yes |
| SEED12_3a | Qwen3-Omni Pitt answer state, nested, 5 repeats (per-repeat [0.8144,0.8180,0.8221,0.8442,0.8200]) | 0.8237 | 468 (228) | yes |
| SEED13_3a | Qwen3-Omni Pitt zero-shot answer (THIS RUN) | 0.7673 | 468 (228) | yes |
| SEED17_3b | Qwen3-Omni Pitt probe after projecting out readout dir | 0.8305 | 468 (228) | yes |

### PART 17 T1 ({'SUPERSEDED / NOT USED': 8, 'SUPPORT': 2})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T1_o25_agreement_OLD_483rows | E-DAIC Table 2 Qwen2.5-Omni audio agreement OLD 483 rows | 0.9482 [0.9039, 0.9828] | 483 (138) | yes |
| T1_q2a_agreement_OLD_483rows | E-DAIC Table 2 Qwen2-Audio audio agreement OLD 483 rows | 0.9478 [0.9128, 0.9762] | 483 (138) | yes |
| T1_q3o_agreement_OLD_483rows | E-DAIC Table 2 Qwen3-Omni-30B-A3B audio agreement OLD 483 rows | 0.9638 [0.9336, 0.9864] | 483 (138) | yes |
| T1_af2_agreement_OLD_483rows | E-DAIC Table 2 Audio Flamingo 2 audio agreement OLD 483 rows | 0.6083 [0.5126, 0.6997] | 483 (138) | yes |
| T1_af3_agreement_OLD_483rows | E-DAIC Table 2 Audio Flamingo 3 audio agreement OLD 483 rows | 0.8095 [0.7447, 0.8696] | 483 (138) | yes |
| T1_kimi_agreement_OLD_483rows | E-DAIC Table 2 Kimi-Audio audio agreement OLD 483 rows | 0.8744 [0.8172, 0.9223] | 483 (138) | yes |
| T1_o25t_agreement_OLD_483rows | E-DAIC Table 2 Qwen2.5-Omni transcript agreement OLD 483 rows | 0.9596 [0.9306, 0.9823] | 483 (138) | yes |
| T1_o25t_paired_conflict_minus_agreement_NEW_311distinct | E-DAIC Table 2 Qwen2.5-Omni transcript SUPPLEMENTARY paired conflict minus agreement, distinct | -0.8123 [-0.8536, -0.7692] | 794 (138) | yes |
| T1_q2at_agreement_OLD_483rows | E-DAIC Table 2 Qwen2-Audio transcript agreement OLD 483 rows | 0.9081 [0.8515, 0.9547] | 483 (138) | yes |
| T1_q2at_paired_conflict_minus_agreement_NEW_311distinct | E-DAIC Table 2 Qwen2-Audio transcript SUPPLEMENTARY paired conflict minus agreement, distinct | -0.6840 [-0.7516, -0.6161] | 794 (138) | yes |

### PART 17 T2 ({'RELEASED ONLY': 62})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T2_pitt_single_mac_q2a_probe2_pitt | pitt pitt_groupkfold5_mac.csv vs q2a_probe2_pitt (ans/enc/llm), max abs diff per clip vs saved OOF: CONFIRMED  | 0.0000 | 468 (228) | yes |
| T2_pitt_single_mac_q2a_probe2_podruns_pitt | pitt pitt_groupkfold5_mac.csv vs q2a_probe2_podruns_pitt (enc), max abs diff per clip vs saved OOF: CONFIRMED  | 0.0000 | 468 (228) | no twin |
| T2_pitt_single_mac_egemaps_pitt | pitt pitt_groupkfold5_mac.csv vs egemaps_pitt (egemaps), max abs diff per clip vs saved OOF: CONFIRMED exact | 0.0000 | 468 (228) | yes |
| T2_adresso_single_mac_q2a_probe2_adresso | adresso adresso_groupkfold5_mac.csv vs q2a_probe2_adresso (ans/enc/llm), max abs diff per clip vs saved OOF: C | 0.0000 | 237 (237) | yes |
| T2_adresso_single_mac_egemaps_adresso | adresso adresso_groupkfold5_mac.csv vs egemaps_adresso (egemaps), max abs diff per clip vs saved OOF: CONFIRME | 0.0000 | 237 (237) | yes |
| T2_adress2020_single_mac_q2a_probe2_adress2020 | adress2020 adress2020_groupkfold5_mac.csv vs q2a_probe2_adress2020 (ans/enc/llm), max abs diff per clip vs sav | 0.0000 | 156 (156) | yes |
| T2_adress2020_single_mac_egemaps_adress2020 | adress2020 adress2020_groupkfold5_mac.csv vs egemaps_adress2020 (egemaps), max abs diff per clip vs saved OOF: | 0.0000 | 156 (156) | yes |
| T2_pcgita_single_mac_q2a_probe2_pcgita | pcgita pcgita_groupkfold5_mac.csv vs q2a_probe2_pcgita (ans/enc/llm), max abs diff per clip vs saved OOF: CONF | 0.0000 | 1100 (100) | yes |
| T2_pcgita_single_mac_egemaps_pcgita | pcgita pcgita_groupkfold5_mac.csv vs egemaps_pcgita (egemaps), max abs diff per clip vs saved OOF: CONFIRMED e | 0.0000 | 1100 (100) | yes |
| T2_neurovoz_single_mac_q2a_probe2_neurovoz | neurovoz neurovoz_groupkfold5_mac.csv vs q2a_probe2_neurovoz (ans/enc/llm), max abs diff per clip vs saved OOF | 0.0000 | 1270 (107) | yes |
| T2_neurovoz_single_mac_egemaps_neurovoz | neurovoz neurovoz_groupkfold5_mac.csv vs egemaps_neurovoz (egemaps), max abs diff per clip vs saved OOF: CONFI | 0.0000 | 1270 (107) | yes |
| T2_kcl_single_mac_q2a_probe2_kcl | kcl kcl_groupkfold5_mac.csv vs q2a_probe2_kcl (ans/enc/llm), max abs diff per clip vs saved OOF: CONFIRMED exa | 0.0000 | 37 (37) | yes |
| T2_kcl_single_mac_egemaps_kcl | kcl kcl_groupkfold5_mac.csv vs egemaps_kcl (egemaps), max abs diff per clip vs saved OOF: CONFIRMED exact | 0.0000 | 37 (37) | yes |
| T2_edaic_single_mac_q2a_probe2_edaic | edaic edaic_groupkfold5_mac.csv vs q2a_probe2_edaic (ans/enc/llm), max abs diff per clip vs saved OOF: CONFIRM | 0.0000 | 275 (275) | yes |
| T2_edaic_single_mac_egemaps_edaic | edaic edaic_groupkfold5_mac.csv vs egemaps_edaic (egemaps), max abs diff per clip vs saved OOF: CONFIRMED exac | 0.0000 | 275 (275) | yes |
| T2_pitt_single_pod_omni_final_pitt | pitt pitt_groupkfold5_pod.csv vs omni_final_pitt (ans/enc/llm/proj), max abs diff per clip vs saved OOF: CONFI | 0.0613 | 468 (228) | yes |
| T2_pitt_single_pod_q2a_podA_curves_pitt_curve | pitt pitt_groupkfold5_pod.csv vs q2a_podA_curves_pitt (ans/enc/llm), max /dAUC/ vs saved per-layer AUC curve:  | 0.0013 | 468 (228) | no twin |
| T2_pitt_single_pod_omni_podA_curves_pitt_curve | pitt pitt_groupkfold5_pod.csv vs omni_podA_curves_pitt (ans/enc/llm/proj), max /dAUC/ vs saved per-layer AUC c | 0.0009 | 468 (228) | no twin |
| T2_pitt_single_pod_kimi_curves_pitt_curve | pitt pitt_groupkfold5_pod.csv vs kimi_curves_pitt (ans/llm), max /dAUC/ vs saved per-layer AUC curve: AUC-ONLY | 0.0013 | 468 (228) | no twin |
| T2_adresso_single_pod_q3o_new_adresso | adresso adresso_groupkfold5_pod.csv vs q3o_new_adresso (ans/enc/llm/proj), max abs diff per clip vs saved OOF: | 0.0306 | 237 (237) | yes |
| T2_adresso_single_pod_q2a_podA_curves_adresso_curve | adresso adresso_groupkfold5_pod.csv vs q2a_podA_curves_adresso (ans/enc/llm), max /dAUC/ vs saved per-layer AU | 0.0013 | 237 (237) | no twin |
| T2_adress2020_single_pod_q3o_new_adress2020 | adress2020 adress2020_groupkfold5_pod.csv vs q3o_new_adress2020 (ans/enc/llm/proj), max abs diff per clip vs s | 0.0140 | 156 (156) | yes |
| T2_adress2020_single_pod_q2a_podA_curves_adress2020_curve | adress2020 adress2020_groupkfold5_pod.csv vs q2a_podA_curves_adress2020 (ans/enc/llm), max /dAUC/ vs saved per | 0.0016 | 156 (156) | no twin |
| T2_pcgita_single_pod_q3o_new_pcgita | pcgita pcgita_groupkfold5_pod.csv vs q3o_new_pcgita (ans/enc/llm/proj), max abs diff per clip vs saved OOF: CO | 0.0461 | 1100 (100) | yes |
| T2_pcgita_single_pod_q2a_podA_curves_pcgita_curve | pcgita pcgita_groupkfold5_pod.csv vs q2a_podA_curves_pcgita (ans/enc/llm), max /dAUC/ vs saved per-layer AUC c | 0.0006 | 1100 (100) | no twin |
| T2_pcgita_single_pod_kimi_curves_pcgita_curve | pcgita pcgita_groupkfold5_pod.csv vs kimi_curves_pcgita (ans/llm), max /dAUC/ vs saved per-layer AUC curve: AU | 0.0009 | 1100 (100) | no twin |
| T2_neurovoz_single_pod_q2a_probe2_podruns_neurovoz | neurovoz neurovoz_groupkfold5_pod.csv vs q2a_probe2_podruns_neurovoz (ans/enc/llm), max abs diff per clip vs s | 0.2489 | 1270 (107) | no twin |
| T2_neurovoz_single_pod_q3o_new_neurovoz | neurovoz neurovoz_groupkfold5_pod.csv vs q3o_new_neurovoz (ans/enc/llm/proj), max abs diff per clip vs saved O | 0.0447 | 1270 (107) | yes |
| T2_neurovoz_single_pod_q2a_podA_curves_neurovoz_curve | neurovoz neurovoz_groupkfold5_pod.csv vs q2a_podA_curves_neurovoz (ans/enc/llm), max /dAUC/ vs saved per-layer | 0.0006 | 1270 (107) | no twin |
| T2_kcl_single_pod_q2a_probe2_kcl_local | kcl kcl_groupkfold5_pod.csv vs q2a_probe2_kcl_local (enc/llm), max abs diff per clip vs saved OOF: CONFIRMED c | 0.0027 | 37 (37) | yes |
| T2_kcl_single_pod_q3o_new_kcl | kcl kcl_groupkfold5_pod.csv vs q3o_new_kcl (ans/enc/llm/proj), max abs diff per clip vs saved OOF: CONFIRMED c | 0.0028 | 37 (37) | yes |
| T2_kcl_single_pod_q2a_podA_curves_kcl_curve | kcl kcl_groupkfold5_pod.csv vs q2a_podA_curves_kcl (ans/enc/llm), max /dAUC/ vs saved per-layer AUC curve: AUC | 0.0060 | 37 (37) | no twin |
| T2_edaic_single_pod_q2a_probe2_podruns_edaic | edaic edaic_groupkfold5_pod.csv vs q2a_probe2_podruns_edaic (ans/enc/llm), max abs diff per clip vs saved OOF: | 0.2455 | 275 (275) | no twin |
| T2_edaic_single_pod_omni_final_edaic | edaic edaic_groupkfold5_pod.csv vs omni_final_edaic (ans/enc/llm/proj), max abs diff per clip vs saved OOF: CO | 0.0336 | 275 (275) | yes |
| T2_edaic_single_pod_q2a_podA_curves_edaic_curve | edaic edaic_groupkfold5_pod.csv vs q2a_podA_curves_edaic (ans/enc/llm), max /dAUC/ vs saved per-layer AUC curv | 0.0023 | 275 (275) | no twin |
| T2_edaic_single_pod_omni_podA_curves_edaic_curve | edaic edaic_groupkfold5_pod.csv vs omni_podA_curves_edaic (ans/enc/llm/proj), max /dAUC/ vs saved per-layer AU | 0.0019 | 275 (275) | no twin |
| T2_edaic_single_pod_kimi_curves_edaic_curve | edaic edaic_groupkfold5_pod.csv vs kimi_curves_edaic (ans/llm), max /dAUC/ vs saved per-layer AUC curve: AUC-O | 0.0020 | 275 (275) | no twin |
| T2_pitt_seed_pod_q2a_podA_repeats_pitt_auc | pitt pitt_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs q2a_podA_repeats_pitt (ans/enc/llm), max /dAUC/ vs saved  | 0.0080 | 468 (228) | no twin |
| T2_pitt_seed_pod_omni_podA_repeats_pitt_auc | pitt pitt_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs omni_podA_repeats_pitt (enc), max /dAUC/ vs saved per-rep | 0.0003 | 468 (228) | no twin |
| T2_pitt_seed_pod_kimi_repeats_pitt_auc | pitt pitt_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs kimi_repeats_pitt (llm), max /dAUC/ vs saved per-repeat A | 0.0061 | 468 (228) | no twin |
| T2_adresso_seed_pod_q2a_podA_repeats_adresso_auc | adresso adresso_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs q2a_podA_repeats_adresso (ans/enc/llm), max /dAUC/  | 0.0006 | 237 (237) | no twin |
| T2_adresso_seed_pod_q3o_podD2_repeats_adresso_auc | adresso adresso_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs q3o_podD2_repeats_adresso (enc), max /dAUC/ vs save | 0.0001 | 237 (237) | no twin |
| T2_adresso_seed_pod_kimi_repeats_adresso_auc | adresso adresso_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs kimi_repeats_adresso (llm), max /dAUC/ vs saved per | 0.0001 | 237 (237) | no twin |
| T2_adress2020_seed_pod_q2a_podA_repeats_adress2020_auc | adress2020 adress2020_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs q2a_podA_repeats_adress2020 (ans/enc/llm), ma | 0.0004 | 156 (156) | no twin |
| T2_adress2020_seed_pod_q3o_podD2_repeats_adress2020_auc | adress2020 adress2020_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs q3o_podD2_repeats_adress2020 (enc), max /dAUC | 0.0003 | 156 (156) | no twin |
| T2_adress2020_seed_pod_kimi_repeats_adress2020_auc | adress2020 adress2020_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs kimi_repeats_adress2020 (llm), max /dAUC/ vs  | 0.0002 | 156 (156) | no twin |
| T2_pcgita_seed_pod_q3o_POD3_pcgita | pcgita pcgita_groupkfold5_pod_seed0..4.csv vs q3o_POD3_pcgita (ans/enc/llm/proj), max abs diff per clip vs sav | 0.0831 | 1100 (100) | yes |
| T2_pcgita_seed_pod_q2a_podA_repeats_pcgita_auc | pcgita pcgita_groupkfold5_pod_seed0..4.csv vs q2a_podA_repeats_pcgita (enc/llm), max /dAUC/ vs saved per-repea | 0.0031 | 1100 (100) | no twin |
| T2_pcgita_seed_pod_q3o_podD2_repeats_pcgita_auc | pcgita pcgita_groupkfold5_pod_seed0..4.csv vs q3o_podD2_repeats_pcgita (enc), max /dAUC/ vs saved per-repeat A | 0.0012 | 1100 (100) | no twin |
| T2_pcgita_seed_pod_kimi_repeats_pcgita_auc | pcgita pcgita_groupkfold5_pod_seed0..4.csv vs kimi_repeats_pcgita (llm), max /dAUC/ vs saved per-repeat AUC: A | 0.0025 | 1100 (100) | no twin |
| T2_neurovoz_seed_pod_q2a_podA_repeats_neurovoz_auc | neurovoz neurovoz_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs q2a_podA_repeats_neurovoz (enc), max /dAUC/ vs sa | 0.0002 | 1270 (107) | no twin |
| T2_neurovoz_seed_pod_q3o_podD2_repeats_neurovoz_auc | neurovoz neurovoz_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs q3o_podD2_repeats_neurovoz (enc), max /dAUC/ vs s | 0.0001 | 1270 (107) | no twin |
| T2_neurovoz_seed_pod_kimi_repeats_neurovoz_auc | neurovoz neurovoz_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs kimi_repeats_neurovoz (llm), max /dAUC/ vs saved  | 0.0052 | 1270 (107) | no twin |
| T2_kcl_seed_pod_q2a_podA_repeats_kcl_auc | kcl kcl_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs q2a_podA_repeats_kcl (enc), max /dAUC/ vs saved per-repeat  | 0.0000 | 37 (37) | no twin |
| T2_kcl_seed_pod_q3o_podD2_repeats_kcl_auc | kcl kcl_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs q3o_podD2_repeats_kcl (enc), max /dAUC/ vs saved per-repeat | 0.0000 | 37 (37) | no twin |
| T2_kcl_seed_pod_kimi_repeats_kcl_auc | kcl kcl_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs kimi_repeats_kcl (llm), max /dAUC/ vs saved per-repeat AUC: | 0.0000 | 37 (37) | no twin |
| T2_edaic_seed_pod_q2a_podA_repeats_edaic_auc | edaic edaic_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs q2a_podA_repeats_edaic (enc), max /dAUC/ vs saved per-r | 0.0008 | 275 (275) | no twin |
| T2_edaic_seed_pod_omni_podA_repeats_edaic_auc | edaic edaic_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs omni_podA_repeats_edaic (enc), max /dAUC/ vs saved per- | 0.0003 | 275 (275) | no twin |
| T2_edaic_seed_pod_kimi_repeats_edaic_auc | edaic edaic_groupkfold5_pod_seed0..4_UNVERIFIED.csv vs kimi_repeats_edaic (llm), max /dAUC/ vs saved per-repea | 0.0008 | 275 (275) | no twin |
| T2_pitt_seed_mac_omni_part10_pitt_nested5 | pitt pitt_groupkfold5_mac_seed0..4.csv vs omni_part10_pitt_nested5 (enc), max abs diff per clip vs saved OOF:  | 0 | 468 (228) | yes |
| T2_pitt_seed_pod_pod3_q3o_POD3_pitt | pitt pitt_groupkfold5_pod_Control15ids_seed0..4.csv vs q3o_POD3_pitt (ans/enc/llm/proj), max abs diff per clip | 0.0536 | 468 (228) | yes |
| T2_edaic_full_shuffle_part16_EDAICFULL | edaic_full edaic_full_groupkfold5_shuffle_rs0..4.csv vs part16_EDAICFULL (ans/enc/llm/proj), max abs diff per  | 0.0240 | 275 (275) | yes |

### PART 17 T3 ({'RELEASED ONLY': 12, 'SUPERSEDED / NOT USED': 8, 'SUPPORT': 7})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T3_all_cos_bootCI | Qwen2-Audio Pitt all: cos, speaker bootstrap with probe refit per draw [full fit; used=secondary] | 0.0129 [-0.0020, 0.0277] | 468 (228) | yes |
| T3_floor_vs_probe | Qwen2-Audio Pitt all: random floor mean/cos/ vs probe direction [fresh default_rng(0), 2000 dirs; used=seconda | 0.0124 [0.0005, 0.0343] | 468 (228) | yes |
| T3_all_keep_z | Qwen2-Audio Pitt all: matched control: same pooling, d kept (within-fold z) [full 468 OOF; used=secondary] | 0.8119 [0.7649, 0.8577] | 468 (228) | yes |
| T3_conflict_keep_z | Qwen2-Audio Pitt conflict: matched control: same pooling, d kept (within-fold z) [full 468 OOF restricted to a | 0.6029 [0.4961, 0.7123] | 146 (100) | yes |
| T3_agreement_keep_z | Qwen2-Audio Pitt agreement: matched control: same pooling, d kept (within-fold z) [full 468 OOF restricted to  | 0.8899 [0.8491, 0.9264] | 322 (175) | yes |
| T3_paired_probe | Qwen2-Audio Pitt conflict-agreement: paired difference, probe [full 468 OOF restricted to arm (DECIDED), paire | -0.2810 [-0.3799, -0.1762] | 468 (228) | yes |
| T3_paired_proj_d | Qwen2-Audio Pitt conflict-agreement: paired difference, proj_d [full 468 OOF restricted to arm (DECIDED), pair | -0.2847 [-0.3947, -0.1475] | 468 (228) | yes |
| T3_conflict_refit_cos | Qwen2-Audio Pitt conflict: cos(probe direction, arm's own readout direction) [refit within arm; used=no (not u | 0.0054 | 146 (100) | yes |
| T3_conflict_refit_probe | Qwen2-Audio Pitt conflict: probe AUC out of fold [refit within arm; used=no (not used)] | 0.6757 [0.5685, 0.7761] | 146 (100) | yes |
| T3_conflict_refit_proj_d | Qwen2-Audio Pitt conflict: AUC along arm's own readout direction [refit within arm; used=no (not used)] | 0.4136 [0.3010, 0.5424] | 146 (100) | yes |
| T3_conflict_refit_perp_z | Qwen2-Audio Pitt conflict: probe AUC, readout direction projected out (within-fold z) [refit within arm; used= | 0.6744 [0.5730, 0.7703] | 146 (100) | yes |
| T3_agreement_refit_cos | Qwen2-Audio Pitt agreement: cos(probe direction, arm's own readout direction) [refit within arm; used=no (not  | 0.0147 | 322 (175) | yes |
| T3_agreement_refit_probe | Qwen2-Audio Pitt agreement: probe AUC out of fold [refit within arm; used=no (not used)] | 0.9317 [0.9020, 0.9578] | 322 (175) | yes |
| T3_agreement_refit_proj_d | Qwen2-Audio Pitt agreement: AUC along arm's own readout direction [refit within arm; used=no (not used)] | 0.6984 [0.6373, 0.7592] | 322 (175) | yes |
| T3_agreement_refit_perp_z | Qwen2-Audio Pitt agreement: probe AUC, readout direction projected out (within-fold z) [refit within arm; used | 0.9242 [0.8932, 0.9510] | 322 (175) | yes |
| T3_nested_cos | Qwen2-Audio Pitt all: cos(NESTED probe direction, readout direction d) [nested, saved layers; used=supplement] | 0.0256 | 468 (228) | yes |
| T3_nested_floor_vs_probe | Qwen2-Audio Pitt all: random floor mean/cos/ vs nested probe direction [fresh default_rng(0), 2000 dirs; used= | 0.0121 [0.0005, 0.0344] | 468 (228) | yes |
| T3_nested_all_probe | Qwen2-Audio Pitt all: nested probe AUC out of fold [nested, full 468 OOF; used=supplement] | 0.8295 [0.7816, 0.8774] | 468 (228) | yes |
| T3_nested_all_perp_z | Qwen2-Audio Pitt all: nested probe AUC, d projected out (within-fold z) [nested, full 468 OOF; used=supplement | 0.8169 [0.7678, 0.8652] | 468 (228) | yes |
| T3_nested_all_keep_z | Qwen2-Audio Pitt all: nested matched control, d kept (within-fold z) [nested, full 468 OOF; used=supplement] | 0.8199 [0.7709, 0.8680] | 468 (228) | yes |
| T3_nested_conflict_probe | Qwen2-Audio Pitt conflict: nested probe AUC out of fold [nested, full 468 OOF restricted to arm; used=suppleme | 0.6554 [0.5549, 0.7543] | 146 (100) | yes |
| T3_nested_conflict_perp_z | Qwen2-Audio Pitt conflict: nested probe AUC, d projected out (within-fold z) [nested, full 468 OOF restricted  | 0.6398 [0.5413, 0.7403] | 146 (100) | yes |
| T3_nested_conflict_keep_z | Qwen2-Audio Pitt conflict: nested matched control, d kept (within-fold z) [nested, full 468 OOF restricted to  | 0.6374 [0.5391, 0.7365] | 146 (100) | yes |
| T3_nested_agreement_probe | Qwen2-Audio Pitt agreement: nested probe AUC out of fold [nested, full 468 OOF restricted to arm; used=supplem | 0.8975 [0.8531, 0.9349] | 322 (175) | yes |
| T3_nested_agreement_perp_z | Qwen2-Audio Pitt agreement: nested probe AUC, d projected out (within-fold z) [nested, full 468 OOF restricted | 0.8842 [0.8377, 0.9249] | 322 (175) | yes |
| T3_nested_agreement_keep_z | Qwen2-Audio Pitt agreement: nested matched control, d kept (within-fold z) [nested, full 468 OOF restricted to | 0.8888 [0.8438, 0.9283] | 322 (175) | yes |
| T3_nested_paired_perp_z | Qwen2-Audio Pitt conflict-agreement: nested paired difference, perp_z [nested, restricted, paired speaker boot | -0.2444 [-0.3489, -0.1438] | 468 (228) | yes |

### PART 17 T4 ({'SUPERSEDED / NOT USED': 13, 'RELEASED ONLY': 3, 'SUPPORT': 2})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T4_pitt_o25_enc_avgprob_NOTUSED | Qwen2.5-Omni Pitt encoder probe, AUC of averaged probabilities (NOT used) | 0.7917 [0.7324, 0.8480] | 468 (228) | yes |
| T4_edaic_o25_llm_avgprob | Qwen2.5-Omni E-DAIC 300 s window LM probe, AUC of per-clip OOF averaged over 5 repeats | 0.7088 [0.6232, 0.7874] | 275 (275) | yes |
| T4_edaic_o25_llm_minus_answer_avgprob | Qwen2.5-Omni E-DAIC 300 s window LM probe minus zero-shot answer, paired, averaged-prob convention | -0.1190 [-0.1995, -0.0478] | 275 (275) | yes |
| T4_edaic_o25_enc_avgprob | Qwen2.5-Omni E-DAIC 300 s window encoder probe, AUC of per-clip OOF averaged over 5 repeats | 0.5916 [0.5110, 0.6706] | 275 (275) | yes |
| T4_edaic_o25_enc_minus_answer_avgprob | Qwen2.5-Omni E-DAIC 300 s window encoder probe minus zero-shot answer, paired, averaged-prob convention | -0.2361 [-0.3272, -0.1394] | 275 (275) | yes |
| T4_pcgita_q3o_enc_avgprob_NOTUSED | Qwen3-Omni PC-GITA encoder probe, AUC of averaged probabilities (NOT used) | 0.9109 [0.8452, 0.9600] | 1100 (100) | yes |
| T4_pitt_q3o_enc_avgprob_NOTUSED | Qwen3-Omni Pitt encoder probe, AUC of averaged probabilities (NOT used) | 0.8354 [0.7817, 0.8877] | 468 (228) | yes |
| T4_pitt_kimi_pooled | Kimi-Audio Pitt pooled zero-shot answer AUC | 0.7180 [0.6585, 0.7742] | 468 (228) | yes |
| T4_pitt_kimi_SUPERSEDED_overnight_pooled | Kimi-Audio SUPERSEDED overnight/kimi copy Pitt pooled zero-shot answer AUC | 0.6964 [0.6324, 0.7575] | 468 (228) | yes |
| T4_pitt_kimi_SUPERSEDED_overnight_conflict | Kimi-Audio SUPERSEDED overnight/kimi copy Pitt conflict zero-shot answer AUC | 0.7078 [0.6037, 0.8083] | 146 (100) | yes |
| T4_pitt_kimi_SUPERSEDED_overnight_agreement | Kimi-Audio SUPERSEDED overnight/kimi copy Pitt agreement zero-shot answer AUC | 0.7014 [0.6309, 0.7730] | 322 (175) | yes |
| T4_pitt_af2_pooled | Audio Flamingo 2 Pitt pooled zero-shot answer AUC | 0.5724 [0.5066, 0.6424] | 468 (228) | yes |
| T4_pitt_af2_SUPERSEDED_audio_pooled | Audio Flamingo 2 SUPERSEDED af2_pitt_audio Pitt pooled zero-shot answer AUC | 0.5273 [0.4622, 0.5924] | 468 (228) | yes |
| T4_pitt_af2_SUPERSEDED_audio_conflict | Audio Flamingo 2 SUPERSEDED af2_pitt_audio Pitt conflict zero-shot answer AUC | 0.6691 [0.5650, 0.7654] | 146 (100) | yes |
| T4_pitt_af2_SUPERSEDED_audio_agreement | Audio Flamingo 2 SUPERSEDED af2_pitt_audio Pitt agreement zero-shot answer AUC | 0.4895 [0.4182, 0.5591] | 322 (175) | yes |
| T4_lookup_rows_before | master_lookup.csv data rows before Part 17 T4 | 221 |  | yes |
| T4_lookup_rows_after | master_lookup.csv data rows after Part 17 T4 | 231 |  | yes |
| T4_lookup_stale_prefix_rows | master_lookup.csv rows whose source still starts <local data dir>/paper1_local_runs/ (after T4) | 75 |  | yes |

### PART 17 T5 ({'RELEASED ONLY': 15, 'SUPPORT': 12})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T5_pitt_layers_above | Pitt encoder layers (of 32) with probe auc_oof > answer AUC | 31 | 468 (228) | yes |
| T5_pitt_layers_strict | Pitt encoder layers (of 32) with auc_mean - auc_std > answer AUC | 25 | 468 (228) | yes |
| T5_pitt_layers_above_ci | Pitt encoder layers (of 32) with probe auc_oof > answer AUC upper 97.5 bound | 25 | 468 (228) | yes |
| T5_pitt_probe_max | Pitt encoder probe max auc_oof (layer 27) | 0.8083 | 468 (228) | yes |
| T5_pcgita_layers_above | PC-GITA encoder layers (of 32) with probe auc_oof > answer AUC | 32 | 1100 (100) | yes |
| T5_pcgita_layers_strict | PC-GITA encoder layers (of 32) with auc_mean - auc_std > answer AUC | 32 | 1100 (100) | yes |
| T5_pcgita_layers_above_ci | PC-GITA encoder layers (of 32) with probe auc_oof > answer AUC upper 97.5 bound | 32 | 1100 (100) | yes |
| T5_pcgita_probe_max | PC-GITA encoder probe max auc_oof (layer 6) | 0.9234 | 1100 (100) | yes |
| T5_edaic_full_layers_above | E-DAIC (full, 900 s cap) encoder layers (of 32) with probe auc_oof > answer AUC | 0 | 275 (275) | yes |
| T5_edaic_full_layers_strict | E-DAIC (full, 900 s cap) encoder layers (of 32) with auc_mean - auc_std > answer AUC | 0 | 275 (275) | yes |
| T5_edaic_full_layers_above_ci | E-DAIC (full, 900 s cap) encoder layers (of 32) with probe auc_oof > answer AUC upper 97.5 bound | 0 | 275 (275) | yes |
| T5_edaic_full_probe_max | E-DAIC (full, 900 s cap) encoder probe max auc_oof (layer 19) | 0.6896 | 275 (275) | yes |
| T5_edaic_mid300_answer | E-DAIC (middle 300 s) Qwen2.5-Omni zero-shot answer AUC (rank formula, speaker bootstrap) | 0.8122 [0.7479, 0.8707] | 275 (275) | yes |
| T5_edaic_mid300_layers_above | E-DAIC (middle 300 s) encoder layers (of 32) with probe auc_oof > answer AUC | 0 | 275 (275) | yes |
| T5_edaic_mid300_layers_strict | E-DAIC (middle 300 s) encoder layers (of 32) with auc_mean - auc_std > answer AUC | 0 | 275 (275) | yes |
| T5_edaic_mid300_layers_above_ci | E-DAIC (middle 300 s) encoder layers (of 32) with probe auc_oof > answer AUC upper 97.5 bound | 0 | 275 (275) | yes |
| T5_edaic_mid300_probe_max | E-DAIC (middle 300 s) encoder probe max auc_oof (layer 22) | 0.6839 | 275 (275) | yes |
| T5_edaic_mid30_answer | E-DAIC (middle 30 s) Qwen2.5-Omni zero-shot answer AUC (rank formula, speaker bootstrap) | 0.7205 [0.6531, 0.7917] | 275 (275) | yes |
| T5_edaic_mid30_layers_above | E-DAIC (middle 30 s) encoder layers (of 32) with probe auc_oof > answer AUC | 0 | 275 (275) | yes |
| T5_edaic_mid30_layers_strict | E-DAIC (middle 30 s) encoder layers (of 32) with auc_mean - auc_std > answer AUC | 0 | 275 (275) | yes |
| T5_edaic_mid30_layers_above_ci | E-DAIC (middle 30 s) encoder layers (of 32) with probe auc_oof > answer AUC upper 97.5 bound | 0 | 275 (275) | yes |
| T5_edaic_mid30_probe_max | E-DAIC (middle 30 s) encoder probe max auc_oof (layer 2) | 0.6705 | 275 (275) | yes |
| T5_edaic_first30_answer | E-DAIC (first 30 s, omni_final) Qwen2.5-Omni zero-shot answer AUC (rank formula, speaker bootstrap) | 0.6620 [0.5831, 0.7315] | 275 (275) | yes |
| T5_edaic_first30_layers_above | E-DAIC (first 30 s, omni_final) encoder layers (of 32) with probe auc_oof > answer AUC | 2 | 275 (275) | yes |
| T5_edaic_first30_layers_strict | E-DAIC (first 30 s, omni_final) encoder layers (of 32) with auc_mean - auc_std > answer AUC | 0 | 275 (275) | yes |
| T5_edaic_first30_layers_above_ci | E-DAIC (first 30 s, omni_final) encoder layers (of 32) with probe auc_oof > answer AUC upper 97.5 bound | 0 | 275 (275) | yes |
| T5_edaic_first30_probe_max | E-DAIC (first 30 s, omni_final) encoder probe max auc_oof (layer 1) | 0.6686 | 275 (275) | yes |

### PART 17 T6 ({'SUPERSEDED / NOT USED': 9, 'RELEASED ONLY': 8, 'SUPPORT': 5})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T6_s15_text_all468 | Pitt Qwen2.5-Omni transcript-only answer AUC, all 468 (Sep 15 run) | 0.6975 [0.6369, 0.7549] | 468 (228) | yes |
| T6_s15_text_conflict | Pitt Qwen2.5-Omni transcript-only answer AUC, conflict arm (Sep 15 run) | 0.4550 [0.3412, 0.5718] | 146 (100) | yes |
| T6_s15_text_agreement | Pitt Qwen2.5-Omni transcript-only answer AUC, agreement arm (Sep 15 run) | 0.7987 [0.7392, 0.8535] | 322 (175) | yes |
| T6_s15_text_conflict_minus_agreement | Pitt Qwen2.5-Omni transcript-only PAIRED conflict minus agreement AUC (Sep 15 run) | -0.3437 [-0.4556, -0.2209] | 468 (228) | yes |
| T6_s20_text_minus_audio_all468 | PAIRED transcript (Sep 20 run) minus audio AUC, all 468 | 0.0051 [-0.0657, 0.0694] | 468 (228) | yes |
| T6_s20_text_minus_audio_conflict | PAIRED transcript (Sep 20 run) minus audio AUC, conflict arm | 0.0238 [-0.1147, 0.1577] | 146 (100) | yes |
| T6_s20_text_minus_audio_agreement | PAIRED transcript (Sep 20 run) minus audio AUC, agreement arm | 0.0024 [-0.0824, 0.0779] | 322 (175) | yes |
| T6_s20_gap_text_minus_gap_audio | PAIRED (transcript Sep 20 run arm gap) minus (audio arm gap), gap = conflict minus agreement | 0.0214 [-0.1413, 0.1806] | 468 (228) | yes |
| T6_s15_text_minus_audio_all468 | PAIRED transcript (Sep 15 run) minus audio AUC, all 468 | 0.0397 [-0.0302, 0.1070] | 468 (228) | yes |
| T6_s15_text_minus_audio_conflict | PAIRED transcript (Sep 15 run) minus audio AUC, conflict arm | -0.0772 [-0.2360, 0.0680] | 146 (100) | yes |
| T6_s15_text_minus_audio_agreement | PAIRED transcript (Sep 15 run) minus audio AUC, agreement arm | 0.0933 [0.0157, 0.1637] | 322 (175) | yes |
| T6_s15_gap_text_minus_gap_audio | PAIRED (transcript Sep 15 run arm gap) minus (audio arm gap), gap = conflict minus agreement | -0.1705 [-0.3440, -0.0049] | 468 (228) | yes |
| T6_s20_minus_s15_all468 | PAIRED Sep 20 minus Sep 15 transcript AUC, all 468 | -0.0346 [-0.0757, 0.0003] | 468 (228) | yes |
| T6_s20_minus_s15_conflict | PAIRED Sep 20 minus Sep 15 transcript AUC, conflict arm | 0.1010 [0.0270, 0.1766] | 146 (100) | yes |
| T6_s20_minus_s15_agreement | PAIRED Sep 20 minus Sep 15 transcript AUC, agreement arm | -0.0909 [-0.1345, -0.0518] | 322 (175) | yes |
| T6_s20_minus_s15_armgap | PAIRED Sep 20 arm gap minus Sep 15 arm gap (transcript) | 0.1919 [0.1104, 0.2800] | 468 (228) | yes |
| T6_perclip_pearson_s20_s15 | Per-clip p_yes Pearson r, Sep 20 vs Sep 15 transcript run | 0.7640 | 468 (228) | yes |
| T6_perclip_spearman_s20_s15 | Per-clip p_yes Spearman rho, Sep 20 vs Sep 15 transcript run | 0.7876 | 468 (228) | yes |
| T6_perclip_maxabsdiff_s20_s15 | Per-clip p_yes max abs diff, Sep 20 vs Sep 15 transcript run | 0.6147 | 468 (228) | yes |
| T6_perclip_meanabsdiff_s20_s15 | Per-clip p_yes mean abs diff, Sep 20 vs Sep 15 transcript run | 0.3019 | 468 (228) | yes |
| T6_armgap_ratio_s20 | Point ratio transcript arm gap / audio arm gap, Sep 20 run | 0.8765 | 468 (228) | yes |
| T6_armgap_ratio_s15 | Point ratio transcript arm gap / audio arm gap, Sep 15 run | 1.9848 | 468 (228) | yes |

### PART 17 T7 ({'SUPERSEDED / NOT USED': 24, 'SUPPORT': 2})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T7_published_ans_meanof5 | answer-state probe, mean of five per-repeat AUCs (pod run, published [0.755, 0.7302, 0.7418, 0.7225, 0.7679]); | 0.7435 | 275 (275) | yes |
| T7_published_proj_meanof5 | projector probe, mean of five per-repeat AUCs (pod run, published [0.551, 0.5618, 0.5795, 0.5653, 0.5474]); in | 0.5610 | 275 (275) | yes |
| T7_notused_llm_aucofmeanoof | NOT USED: LM probe AUC of OOF averaged over 5 repeats, E-DAIC 300 s window (audio truncated at 300 s by Qwen2_ | 0.7088 [0.6232, 0.7874] | 275 (275) | yes |
| T7_notused_llm_aucofmeanoof_minus_answer | NOT USED: LM probe (AUC of averaged OOF) minus zero-shot answer, paired, E-DAIC 300 s window (audio truncated  | -0.1190 [-0.1995, -0.0478] | 275 (275) | yes |
| T7_notused_enc_aucofmeanoof | NOT USED: encoder probe AUC of OOF averaged over 5 repeats, E-DAIC 300 s window (audio truncated at 300 s by Q | 0.5916 [0.5110, 0.6706] | 275 (275) | yes |
| T7_notused_enc_aucofmeanoof_minus_answer | NOT USED: encoder probe (AUC of averaged OOF) minus zero-shot answer, paired, E-DAIC 300 s window (audio trunc | -0.2361 [-0.3272, -0.1394] | 275 (275) | yes |
| T7_notused_ans_aucofmeanoof | NOT USED: answer-state probe AUC of OOF averaged over 5 repeats, E-DAIC 300 s window (audio truncated at 300 s | 0.7748 [0.7041, 0.8387] | 275 (275) | yes |
| T7_notused_ans_aucofmeanoof_minus_answer | NOT USED: answer-state probe (AUC of averaged OOF) minus zero-shot answer, paired, E-DAIC 300 s window (audio  | -0.0530 [-0.1046, -0.0049] | 275 (275) | yes |
| T7_notused_proj_aucofmeanoof | NOT USED: projector probe AUC of OOF averaged over 5 repeats, E-DAIC 300 s window (audio truncated at 300 s by | 0.5631 [0.4790, 0.6478] | 275 (275) | yes |
| T7_notused_proj_aucofmeanoof_minus_answer | NOT USED: projector probe (AUC of averaged OOF) minus zero-shot answer, paired, E-DAIC 300 s window (audio tru | -0.2646 [-0.3575, -0.1667] | 275 (275) | yes |
| T7_PROXYD_llm_meanof5 | PROXY, NOT the paper's value: Mac rerun with the pod's recorded layer picks; LM probe mean of five per-repeat  | 0.6997 [0.6229, 0.7722] | 275 (275) | yes |
| T7_PROXYD_llm_meanof5_minus_answer | PROXY, NOT the paper's value: Mac rerun with the pod's recorded layer picks; LM probe mean of five per-repeat  | -0.1280 [-0.1999, -0.0632] | 275 (275) | yes |
| T7_PROXYD_enc_meanof5 | PROXY, NOT the paper's value: Mac rerun with the pod's recorded layer picks; encoder probe mean of five per-re | 0.5885 [0.5268, 0.6525] | 275 (275) | yes |
| T7_PROXYD_enc_meanof5_minus_answer | PROXY, NOT the paper's value: Mac rerun with the pod's recorded layer picks; encoder probe mean of five per-re | -0.2392 [-0.3186, -0.1572] | 275 (275) | yes |
| T7_PROXYD_ans_meanof5 | PROXY, NOT the paper's value: Mac rerun with the pod's recorded layer picks; answer-state probe mean of five p | 0.7436 [0.6797, 0.8064] | 275 (275) | yes |
| T7_PROXYD_ans_meanof5_minus_answer | PROXY, NOT the paper's value: Mac rerun with the pod's recorded layer picks; answer-state probe mean of five p | -0.0842 [-0.1341, -0.0370] | 275 (275) | yes |
| T7_PROXYD_proj_meanof5 | PROXY, NOT the paper's value: Mac rerun with the pod's recorded layer picks; projector probe mean of five per- | 0.5611 [0.4872, 0.6338] | 275 (275) | yes |
| T7_PROXYD_proj_meanof5_minus_answer | PROXY, NOT the paper's value: Mac rerun with the pod's recorded layer picks; projector probe mean of five per- | -0.2667 [-0.3493, -0.1811] | 275 (275) | yes |
| T7_PROXYA_llm_meanof5 | PROXY, NOT the paper's value: Mac rerun, layers re-selected on Mac; LM probe mean of five per-repeat AUCs [0.6 | 0.6982 [0.6215, 0.7702] | 275 (275) | yes |
| T7_PROXYA_llm_meanof5_minus_answer | PROXY, NOT the paper's value: Mac rerun, layers re-selected on Mac; LM probe mean of five per-repeat AUCs minu | -0.1296 [-0.2019, -0.0630] | 275 (275) | yes |
| T7_PROXYA_enc_meanof5 | PROXY, NOT the paper's value: Mac rerun, layers re-selected on Mac; encoder probe mean of five per-repeat AUCs | 0.5885 [0.5268, 0.6525] | 275 (275) | yes |
| T7_PROXYA_enc_meanof5_minus_answer | PROXY, NOT the paper's value: Mac rerun, layers re-selected on Mac; encoder probe mean of five per-repeat AUCs | -0.2392 [-0.3186, -0.1572] | 275 (275) | yes |
| T7_PROXYA_ans_meanof5 | PROXY, NOT the paper's value: Mac rerun, layers re-selected on Mac; answer-state probe mean of five per-repeat | 0.7380 [0.6742, 0.8020] | 275 (275) | yes |
| T7_PROXYA_ans_meanof5_minus_answer | PROXY, NOT the paper's value: Mac rerun, layers re-selected on Mac; answer-state probe mean of five per-repeat | -0.0898 [-0.1409, -0.0414] | 275 (275) | yes |
| T7_PROXYA_proj_meanof5 | PROXY, NOT the paper's value: Mac rerun, layers re-selected on Mac; projector probe mean of five per-repeat AU | 0.5611 [0.4872, 0.6338] | 275 (275) | yes |
| T7_PROXYA_proj_meanof5_minus_answer | PROXY, NOT the paper's value: Mac rerun, layers re-selected on Mac; projector probe mean of five per-repeat AU | -0.2667 [-0.3493, -0.1811] | 275 (275) | yes |

### PART 17 T7b ({'SUPERSEDED / NOT USED': 8, 'SUPPORT': 4})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| T7b_ans_meanof5 | answer-state probe, mean of five per-repeat AUCs [0.755, 0.7302, 0.7418, 0.7225, 0.7679] (nested GroupKFold(5) | 0.7435 [0.6793, 0.8065] | 275 (275) | yes |
| T7b_ans_meanof5_minus_answer | answer-state probe mean of five per-repeat AUCs minus zero-shot answer AUC, paired (one speaker draw per repli | -0.0843 [-0.1339, -0.0370] | 275 (275) | yes |
| T7b_proj_meanof5 | projector probe, mean of five per-repeat AUCs [0.551, 0.5618, 0.5795, 0.5653, 0.5474] (nested GroupKFold(5) x  | 0.5610 [0.4871, 0.6339] | 275 (275) | yes |
| T7b_proj_meanof5_minus_answer | projector probe mean of five per-repeat AUCs minus zero-shot answer AUC, paired (one speaker draw per replicat | -0.2668 [-0.3494, -0.1809] | 275 (275) | yes |
| T7b_notused_llm_aucofmeanoof | NOT USED: LM probe AUC of the OOF probabilities averaged over five repeats, E-DAIC 300 s window | 0.7088 [0.6232, 0.7874] | 275 (275) | yes |
| T7b_notused_llm_aucofmeanoof_minus_answer | NOT USED: LM probe (AUC of averaged OOF) minus zero-shot answer, paired, E-DAIC 300 s window | -0.1190 [-0.1995, -0.0478] | 275 (275) | yes |
| T7b_notused_enc_aucofmeanoof | NOT USED: encoder probe AUC of the OOF probabilities averaged over five repeats, E-DAIC 300 s window | 0.5916 [0.5110, 0.6706] | 275 (275) | yes |
| T7b_notused_enc_aucofmeanoof_minus_answer | NOT USED: encoder probe (AUC of averaged OOF) minus zero-shot answer, paired, E-DAIC 300 s window | -0.2361 [-0.3272, -0.1394] | 275 (275) | yes |
| T7b_notused_ans_aucofmeanoof | NOT USED: answer-state probe AUC of the OOF probabilities averaged over five repeats, E-DAIC 300 s window | 0.7748 [0.7041, 0.8387] | 275 (275) | yes |
| T7b_notused_ans_aucofmeanoof_minus_answer | NOT USED: answer-state probe (AUC of averaged OOF) minus zero-shot answer, paired, E-DAIC 300 s window | -0.0530 [-0.1046, -0.0049] | 275 (275) | yes |
| T7b_notused_proj_aucofmeanoof | NOT USED: projector probe AUC of the OOF probabilities averaged over five repeats, E-DAIC 300 s window | 0.5631 [0.4790, 0.6478] | 275 (275) | yes |
| T7b_notused_proj_aucofmeanoof_minus_answer | NOT USED: projector probe (AUC of averaged OOF) minus zero-shot answer, paired, E-DAIC 300 s window | -0.2646 [-0.3575, -0.1667] | 275 (275) | yes |

### PART 20 POD1 ({'RELEASED ONLY': 85, 'SUPPORT': 5})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| POD1_bal_all | balanced projector fine-tune AUC, all 468, p_yes (Yes/No softmax) | 0.7301 [0.6656, 0.7905] | 468 (228) | yes |
| POD1_bal_conflict | balanced projector fine-tune AUC, conflict 146, p_yes (Yes/No softmax) | 0.4693 [0.3686, 0.5854] | 146 (100) | yes |
| POD1_bal_agreement | balanced projector fine-tune AUC, agreement 322, p_yes (Yes/No softmax) | 0.8279 [0.7732, 0.8792] | 322 (175) | yes |
| POD1_paired_conflict | PAIRED balanced minus standard fine-tune AUC, conflict 146, one speaker draw per replicate | -0.0496 [-0.1403, 0.0344] | 146 (100) | yes |
| POD1_paired_all | PAIRED balanced minus standard fine-tune AUC, all 468, one speaker draw per replicate | -0.0372 [-0.0720, -0.0018] | 468 (228) | yes |
| POD1_paired_agreement | PAIRED balanced minus standard fine-tune AUC, agreement 322, one speaker draw per replicate | -0.0314 [-0.0659, 0.0023] | 322 (175) | yes |
| POD1_bal_multi_all | balanced fine-tune AUC, all 468, p_yes_multi (Yes/yes/YES vs No/no/NO) | 0.7407 [0.6761, 0.7994] | 468 (228) | yes |
| POD1_bal_multi_conflict | balanced fine-tune AUC, conflict 146, p_yes_multi (Yes/yes/YES vs No/no/NO) | 0.4671 [0.3641, 0.5818] | 146 (100) | yes |
| POD1_bal_multi_agreement | balanced fine-tune AUC, agreement 322, p_yes_multi (Yes/yes/YES vs No/no/NO) | 0.8433 [0.7925, 0.8924] | 322 (175) | yes |
| POD1_gap_bal | conflict minus agreement AUC gap, balanced fine-tune (one speaker draw over 228 per replicate) | -0.3586 [-0.4671, -0.2461] | 468 (228) | yes |
| POD1_gap_std | conflict minus agreement AUC gap, standard fine-tune (one speaker draw over 228 per replicate) | -0.3404 [-0.4468, -0.2262] | 468 (228) | yes |
| POD1_gap_change | gap change, balanced gap minus standard gap (same draw) | -0.0182 [-0.1116, 0.0756] | 468 (228) | yes |
| POD1_f3rerun_all | balanced fine-tune (fold 3 from lone rerun) AUC, all 468, p_yes | 0.7603 [0.6967, 0.8174] | 468 (228) | yes |
| POD1_f3rerun_conflict | balanced fine-tune (fold 3 from lone rerun) AUC, conflict 146, p_yes | 0.5003 [0.3936, 0.6170] | 146 (100) | yes |
| POD1_f3rerun_agreement | balanced fine-tune (fold 3 from lone rerun) AUC, agreement 322, p_yes | 0.8573 [0.8063, 0.9053] | 322 (175) | yes |
| POD1_f3rerun_paired_conflict | PAIRED balanced fine-tune (fold 3 from lone rerun) minus standard fine-tune (paper file), conflict 146, one sp | -0.0186 [-0.1223, 0.0725] | 146 (100) | yes |
| POD1_f3rerun_paired_all | PAIRED balanced fine-tune (fold 3 from lone rerun) minus standard fine-tune (paper file), all 468, one speaker | -0.0070 [-0.0428, 0.0313] | 468 (228) | yes |
| POD1_f3rerun_paired_agreement | PAIRED balanced fine-tune (fold 3 from lone rerun) minus standard fine-tune (paper file), agreement 322, one s | -0.0020 [-0.0345, 0.0310] | 322 (175) | yes |
| POD1_f3rerun_multi_all | balanced fine-tune (fold 3 from lone rerun) AUC, all 468, p_yes_multi | 0.7621 [0.7006, 0.8192] | 468 (228) | yes |
| POD1_f3rerun_multi_conflict | balanced fine-tune (fold 3 from lone rerun) AUC, conflict 146, p_yes_multi | 0.4999 [0.3923, 0.6151] | 146 (100) | yes |
| POD1_f3rerun_multi_agreement | balanced fine-tune (fold 3 from lone rerun) AUC, agreement 322, p_yes_multi | 0.8603 [0.8099, 0.9065] | 322 (175) | yes |
| POD1_f3rerun_gap | conflict minus agreement gap, balanced fine-tune (fold 3 from lone rerun) | -0.3570 [-0.4649, -0.2459] | 468 (228) | yes |
| POD1_f3rerun_gap_ref | conflict minus agreement gap, standard fine-tune (paper file) | -0.3404 [-0.4468, -0.2262] | 468 (228) | yes |
| POD1_f3rerun_gap_change | gap change, balanced fine-tune (fold 3 from lone rerun) minus standard fine-tune (paper file) (same draw) | -0.0166 [-0.1143, 0.0879] | 468 (228) | yes |
| POD1_ctrl_all | CONTROL standard recipe rerun on POD1 (uniform weights) AUC, all 468, p_yes | 0.7720 [0.7166, 0.8280] | 468 (228) | yes |
| POD1_ctrl_conflict | CONTROL standard recipe rerun on POD1 (uniform weights) AUC, conflict 146, p_yes | 0.4697 [0.3613, 0.5906] | 146 (100) | yes |
| POD1_ctrl_agreement | CONTROL standard recipe rerun on POD1 (uniform weights) AUC, agreement 322, p_yes | 0.8895 [0.8456, 0.9286] | 322 (175) | yes |
| POD1_ctrl_paired_conflict | PAIRED CONTROL standard recipe rerun on POD1 (uniform weights) minus standard fine-tune (paper file), conflict | -0.0492 [-0.1526, 0.0537] | 146 (100) | yes |
| POD1_ctrl_paired_all | PAIRED CONTROL standard recipe rerun on POD1 (uniform weights) minus standard fine-tune (paper file), all 468, | 0.0047 [-0.0349, 0.0460] | 468 (228) | yes |
| POD1_ctrl_paired_agreement | PAIRED CONTROL standard recipe rerun on POD1 (uniform weights) minus standard fine-tune (paper file), agreemen | 0.0302 [-0.0022, 0.0679] | 322 (175) | yes |
| POD1_ctrl_multi_all | CONTROL standard recipe rerun on POD1 (uniform weights) AUC, all 468, p_yes_multi | 0.7724 [0.7166, 0.8285] | 468 (228) | yes |
| POD1_ctrl_multi_conflict | CONTROL standard recipe rerun on POD1 (uniform weights) AUC, conflict 146, p_yes_multi | 0.4726 [0.3635, 0.5924] | 146 (100) | yes |
| POD1_ctrl_multi_agreement | CONTROL standard recipe rerun on POD1 (uniform weights) AUC, agreement 322, p_yes_multi | 0.8896 [0.8455, 0.9290] | 322 (175) | yes |
| POD1_ctrl_gap | conflict minus agreement gap, CONTROL standard recipe rerun on POD1 (uniform weights) | -0.4197 [-0.5321, -0.2948] | 468 (228) | yes |
| POD1_ctrl_gap_ref | conflict minus agreement gap, standard fine-tune (paper file) | -0.3404 [-0.4468, -0.2262] | 468 (228) | yes |
| POD1_ctrl_gap_change | gap change, CONTROL standard recipe rerun on POD1 (uniform weights) minus standard fine-tune (paper file) (sam | -0.0793 [-0.1895, 0.0331] | 468 (228) | yes |
| POD1_balVctrl_all | balanced fine-tune repeat 1 (as reported) AUC, all 468, p_yes | 0.7301 [0.6656, 0.7905] | 468 (228) | yes |
| POD1_balVctrl_conflict | balanced fine-tune repeat 1 (as reported) AUC, conflict 146, p_yes | 0.4693 [0.3686, 0.5854] | 146 (100) | yes |
| POD1_balVctrl_agreement | balanced fine-tune repeat 1 (as reported) AUC, agreement 322, p_yes | 0.8279 [0.7732, 0.8792] | 322 (175) | yes |
| POD1_balVctrl_ref_all | POD1 standard rerun (control) AUC (beside), all 468, p_yes | 0.7720 [0.7166, 0.8280] | 468 (228) | yes |
| POD1_balVctrl_ref_conflict | POD1 standard rerun (control) AUC (beside), conflict 146, p_yes | 0.4697 [0.3613, 0.5906] | 146 (100) | yes |
| POD1_balVctrl_ref_agreement | POD1 standard rerun (control) AUC (beside), agreement 322, p_yes | 0.8895 [0.8456, 0.9286] | 322 (175) | yes |
| POD1_balVctrl_paired_conflict | PAIRED balanced fine-tune repeat 1 (as reported) minus POD1 standard rerun (control), conflict 146, one speake | -0.0004 [-0.0742, 0.0757] | 146 (100) | yes |
| POD1_balVctrl_paired_all | PAIRED balanced fine-tune repeat 1 (as reported) minus POD1 standard rerun (control), all 468, one speaker dra | -0.0420 [-0.0803, -0.0028] | 468 (228) | yes |
| POD1_balVctrl_paired_agreement | PAIRED balanced fine-tune repeat 1 (as reported) minus POD1 standard rerun (control), agreement 322, one speak | -0.0615 [-0.1005, -0.0234] | 322 (175) | yes |
| POD1_balVctrl_multi_all | balanced fine-tune repeat 1 (as reported) AUC, all 468, p_yes_multi | 0.7407 [0.6761, 0.7994] | 468 (228) | yes |
| POD1_balVctrl_multi_conflict | balanced fine-tune repeat 1 (as reported) AUC, conflict 146, p_yes_multi | 0.4671 [0.3641, 0.5818] | 146 (100) | yes |
| POD1_balVctrl_multi_agreement | balanced fine-tune repeat 1 (as reported) AUC, agreement 322, p_yes_multi | 0.8433 [0.7925, 0.8924] | 322 (175) | yes |
| POD1_balVctrl_gap | conflict minus agreement gap, balanced fine-tune repeat 1 (as reported) | -0.3586 [-0.4671, -0.2461] | 468 (228) | yes |
| POD1_balVctrl_gap_ref | conflict minus agreement gap, POD1 standard rerun (control) | -0.4197 [-0.5321, -0.2948] | 468 (228) | yes |
| POD1_balVctrl_gap_change | gap change, balanced fine-tune repeat 1 (as reported) minus POD1 standard rerun (control) (same draw) | 0.0611 [-0.0181, 0.1433] | 468 (228) | yes |
| POD1_bal_r2_all | balanced fine-tune repeat 2 AUC, all 468, p_yes | 0.7360 [0.6726, 0.7966] | 468 (228) | yes |
| POD1_bal_r2_conflict | balanced fine-tune repeat 2 AUC, conflict 146, p_yes | 0.4810 [0.3775, 0.5991] | 146 (100) | yes |
| POD1_bal_r2_agreement | balanced fine-tune repeat 2 AUC, agreement 322, p_yes | 0.8379 [0.7847, 0.8894] | 322 (175) | yes |
| POD1_bal_r2_paired_conflict | PAIRED balanced fine-tune repeat 2 minus standard fine-tune (paper file), conflict 146, one speaker draw per r | -0.0380 [-0.1562, 0.0666] | 146 (100) | yes |
| POD1_bal_r2_paired_all | PAIRED balanced fine-tune repeat 2 minus standard fine-tune (paper file), all 468, one speaker draw per replic | -0.0313 [-0.0748, 0.0104] | 468 (228) | yes |
| POD1_bal_r2_paired_agreement | PAIRED balanced fine-tune repeat 2 minus standard fine-tune (paper file), agreement 322, one speaker draw per  | -0.0214 [-0.0615, 0.0209] | 322 (175) | yes |
| POD1_bal_r2_multi_all | balanced fine-tune repeat 2 AUC, all 468, p_yes_multi | 0.7381 [0.6765, 0.7971] | 468 (228) | yes |
| POD1_bal_r2_multi_conflict | balanced fine-tune repeat 2 AUC, conflict 146, p_yes_multi | 0.4825 [0.3766, 0.5990] | 146 (100) | yes |
| POD1_bal_r2_multi_agreement | balanced fine-tune repeat 2 AUC, agreement 322, p_yes_multi | 0.8405 [0.7897, 0.8909] | 322 (175) | yes |
| POD1_bal_r2_gap | conflict minus agreement gap, balanced fine-tune repeat 2 | -0.3569 [-0.4633, -0.2380] | 468 (228) | yes |
| POD1_bal_r2_gap_ref | conflict minus agreement gap, standard fine-tune (paper file) | -0.3404 [-0.4468, -0.2262] | 468 (228) | yes |
| POD1_bal_r2_gap_change | gap change, balanced fine-tune repeat 2 minus standard fine-tune (paper file) (same draw) | -0.0166 [-0.1313, 0.1009] | 468 (228) | yes |
| POD1_ctrl_r2_all | CONTROL standard recipe rerun on POD1 repeat 2 (uniform weights) AUC, all 468, p_yes | 0.7734 [0.7172, 0.8286] | 468 (228) | yes |
| POD1_ctrl_r2_conflict | CONTROL standard recipe rerun on POD1 repeat 2 (uniform weights) AUC, conflict 146, p_yes | 0.4490 [0.3385, 0.5682] | 146 (100) | yes |
| POD1_ctrl_r2_agreement | CONTROL standard recipe rerun on POD1 repeat 2 (uniform weights) AUC, agreement 322, p_yes | 0.8967 [0.8592, 0.9322] | 322 (175) | yes |
| POD1_ctrl_r2_paired_conflict | PAIRED CONTROL standard recipe rerun on POD1 repeat 2 (uniform weights) minus standard fine-tune (paper file), | -0.0700 [-0.1678, 0.0142] | 146 (100) | yes |
| POD1_ctrl_r2_paired_all | PAIRED CONTROL standard recipe rerun on POD1 repeat 2 (uniform weights) minus standard fine-tune (paper file), | 0.0061 [-0.0337, 0.0493] | 468 (228) | yes |
| POD1_ctrl_r2_paired_agreement | PAIRED CONTROL standard recipe rerun on POD1 repeat 2 (uniform weights) minus standard fine-tune (paper file), | 0.0374 [0.0025, 0.0786] | 322 (175) | yes |
| POD1_ctrl_r2_multi_all | CONTROL standard recipe rerun on POD1 repeat 2 (uniform weights) AUC, all 468, p_yes_multi | 0.7739 [0.7190, 0.8290] | 468 (228) | yes |
| POD1_ctrl_r2_multi_conflict | CONTROL standard recipe rerun on POD1 repeat 2 (uniform weights) AUC, conflict 146, p_yes_multi | 0.4455 [0.3343, 0.5641] | 146 (100) | yes |
| POD1_ctrl_r2_multi_agreement | CONTROL standard recipe rerun on POD1 repeat 2 (uniform weights) AUC, agreement 322, p_yes_multi | 0.8972 [0.8600, 0.9319] | 322 (175) | yes |
| POD1_ctrl_r2_gap | conflict minus agreement gap, CONTROL standard recipe rerun on POD1 repeat 2 (uniform weights) | -0.4477 [-0.5558, -0.3351] | 468 (228) | yes |
| POD1_ctrl_r2_gap_ref | conflict minus agreement gap, standard fine-tune (paper file) | -0.3404 [-0.4468, -0.2262] | 468 (228) | yes |
| POD1_ctrl_r2_gap_change | gap change, CONTROL standard recipe rerun on POD1 repeat 2 (uniform weights) minus standard fine-tune (paper f | -0.1073 [-0.1967, -0.0137] | 468 (228) | yes |
| POD1_balVctrl_r2_all | balanced fine-tune repeat 2 AUC, all 468, p_yes | 0.7360 [0.6726, 0.7966] | 468 (228) | yes |
| POD1_balVctrl_r2_conflict | balanced fine-tune repeat 2 AUC, conflict 146, p_yes | 0.4810 [0.3775, 0.5991] | 146 (100) | yes |
| POD1_balVctrl_r2_agreement | balanced fine-tune repeat 2 AUC, agreement 322, p_yes | 0.8379 [0.7847, 0.8894] | 322 (175) | yes |
| POD1_balVctrl_r2_ref_all | POD1 standard rerun repeat 2 (control) AUC (beside), all 468, p_yes | 0.7734 [0.7172, 0.8286] | 468 (228) | yes |
| POD1_balVctrl_r2_ref_conflict | POD1 standard rerun repeat 2 (control) AUC (beside), conflict 146, p_yes | 0.4490 [0.3385, 0.5682] | 146 (100) | yes |
| POD1_balVctrl_r2_ref_agreement | POD1 standard rerun repeat 2 (control) AUC (beside), agreement 322, p_yes | 0.8967 [0.8592, 0.9322] | 322 (175) | yes |
| POD1_balVctrl_r2_paired_conflict | PAIRED balanced fine-tune repeat 2 minus POD1 standard rerun repeat 2 (control), conflict 146, one speaker dra | 0.0320 [-0.0564, 0.1211] | 146 (100) | yes |
| POD1_balVctrl_r2_paired_all | PAIRED balanced fine-tune repeat 2 minus POD1 standard rerun repeat 2 (control), all 468, one speaker draw per | -0.0374 [-0.0760, -0.0012] | 468 (228) | yes |
| POD1_balVctrl_r2_paired_agreement | PAIRED balanced fine-tune repeat 2 minus POD1 standard rerun repeat 2 (control), agreement 322, one speaker dr | -0.0588 [-0.1016, -0.0204] | 322 (175) | yes |
| POD1_balVctrl_r2_multi_all | balanced fine-tune repeat 2 AUC, all 468, p_yes_multi | 0.7381 [0.6765, 0.7971] | 468 (228) | yes |
| POD1_balVctrl_r2_multi_conflict | balanced fine-tune repeat 2 AUC, conflict 146, p_yes_multi | 0.4825 [0.3766, 0.5990] | 146 (100) | yes |
| POD1_balVctrl_r2_multi_agreement | balanced fine-tune repeat 2 AUC, agreement 322, p_yes_multi | 0.8405 [0.7897, 0.8909] | 322 (175) | yes |
| POD1_balVctrl_r2_gap | conflict minus agreement gap, balanced fine-tune repeat 2 | -0.3569 [-0.4633, -0.2380] | 468 (228) | yes |
| POD1_balVctrl_r2_gap_ref | conflict minus agreement gap, POD1 standard rerun repeat 2 (control) | -0.4477 [-0.5558, -0.3351] | 468 (228) | yes |
| POD1_balVctrl_r2_gap_change | gap change, balanced fine-tune repeat 2 minus POD1 standard rerun repeat 2 (control) (same draw) | 0.0908 [-0.0041, 0.1885] | 468 (228) | yes |

### PART 20 POD1b ({'RELEASED ONLY': 38})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| POD1b_seed1_p_yes_overall | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 1, score p_yes: overall OOF AUC | 0.7396 [0.6891, 0.7921] | 468 (228) | yes |
| POD1b_seed1_p_yes_conflict | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 1, score p_yes: conflict arm OOF AUC (interval: conflic | 0.5364 [0.4306, 0.6426] | 146 (100) | yes |
| POD1b_seed1_p_yes_agreement | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 1, score p_yes: agreement arm OOF AUC (interval: agreem | 0.8213 [0.7716, 0.8689] | 322 (175) | yes |
| POD1b_seed1_p_yes_conflict_minus_agreement | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 1, score p_yes: paired conflict minus agreement AUC (on | -0.2848 [-0.3962, -0.1671] | 468 (228) | yes |
| POD1b_seed1_p_yes_multivariant_overall | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 1, score p_yes_multivariant: overall OOF AUC | 0.7384 [0.6854, 0.7903] | 468 (228) | yes |
| POD1b_seed1_p_yes_multivariant_conflict | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 1, score p_yes_multivariant: conflict arm OOF AUC (inte | 0.5459 [0.4411, 0.6500] | 146 (100) | yes |
| POD1b_seed1_p_yes_multivariant_agreement | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 1, score p_yes_multivariant: agreement arm OOF AUC (int | 0.8168 [0.7649, 0.8650] | 322 (175) | yes |
| POD1b_seed1_p_yes_multivariant_conflict_minus_agreement | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 1, score p_yes_multivariant: paired conflict minus agre | -0.2709 [-0.3839, -0.1562] | 468 (228) | yes |
| POD1b_seed2_p_yes_overall | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 2, score p_yes: overall OOF AUC | 0.8085 [0.7560, 0.8572] | 468 (228) | yes |
| POD1b_seed2_p_yes_conflict | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 2, score p_yes: conflict arm OOF AUC (interval: conflic | 0.5636 [0.4593, 0.6779] | 146 (100) | yes |
| POD1b_seed2_p_yes_agreement | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 2, score p_yes: agreement arm OOF AUC (interval: agreem | 0.9022 [0.8648, 0.9361] | 322 (175) | yes |
| POD1b_seed2_p_yes_conflict_minus_agreement | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 2, score p_yes: paired conflict minus agreement AUC (on | -0.3386 [-0.4461, -0.2271] | 468 (228) | yes |
| POD1b_seed2_p_yes_multivariant_overall | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 2, score p_yes_multivariant: overall OOF AUC | 0.8029 [0.7499, 0.8521] | 468 (228) | yes |
| POD1b_seed2_p_yes_multivariant_conflict | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 2, score p_yes_multivariant: conflict arm OOF AUC (inte | 0.5699 [0.4645, 0.6833] | 146 (100) | yes |
| POD1b_seed2_p_yes_multivariant_agreement | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 2, score p_yes_multivariant: agreement arm OOF AUC (int | 0.8937 [0.8549, 0.9294] | 322 (175) | yes |
| POD1b_seed2_p_yes_multivariant_conflict_minus_agreement | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 2, score p_yes_multivariant: paired conflict minus agre | -0.3238 [-0.4284, -0.2144] | 468 (228) | yes |
| POD1b_seed3_p_yes_overall | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 3, score p_yes: overall OOF AUC | 0.7975 [0.7456, 0.8479] | 468 (228) | yes |
| POD1b_seed3_p_yes_conflict | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 3, score p_yes: conflict arm OOF AUC (interval: conflic | 0.5666 [0.4660, 0.6746] | 146 (100) | yes |
| POD1b_seed3_p_yes_agreement | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 3, score p_yes: agreement arm OOF AUC (interval: agreem | 0.8844 [0.8424, 0.9225] | 322 (175) | yes |
| POD1b_seed3_p_yes_conflict_minus_agreement | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 3, score p_yes: paired conflict minus agreement AUC (on | -0.3178 [-0.4204, -0.2060] | 468 (228) | yes |
| POD1b_seed3_p_yes_multivariant_overall | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 3, score p_yes_multivariant: overall OOF AUC | 0.7940 [0.7408, 0.8450] | 468 (228) | yes |
| POD1b_seed3_p_yes_multivariant_conflict | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 3, score p_yes_multivariant: conflict arm OOF AUC (inte | 0.5694 [0.4692, 0.6769] | 146 (100) | yes |
| POD1b_seed3_p_yes_multivariant_agreement | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 3, score p_yes_multivariant: agreement arm OOF AUC (int | 0.8791 [0.8363, 0.9184] | 322 (175) | yes |
| POD1b_seed3_p_yes_multivariant_conflict_minus_agreement | Qwen2.5-Omni-7B standard Pitt projector fine-tune seed 3, score p_yes_multivariant: paired conflict minus agre | -0.3097 [-0.4128, -0.1970] | 468 (228) | yes |
| POD1b_range_overall_seeds_1_2_3 | range (lo=min, hi=max, value=max minus min) of overall p_yes OOF AUC over seeds 1,2,3 | 0.0688 [0.7396, 0.8085] | 468 (228) | yes |
| POD1b_range_overall_seeds_1_2_3_plus_original | range (lo=min, hi=max, value=max minus min) of overall p_yes OOF AUC over seeds 1,2,3 plus original run (0.767 | 0.0688 [0.7396, 0.8085] | 468 (228) | yes |
| POD1b_range_conflict_seeds_1_2_3 | range (lo=min, hi=max, value=max minus min) of conflict p_yes OOF AUC over seeds 1,2,3 | 0.0301 [0.5364, 0.5666] | 146 (100) | yes |
| POD1b_range_conflict_seeds_1_2_3_plus_original | range (lo=min, hi=max, value=max minus min) of conflict p_yes OOF AUC over seeds 1,2,3 plus original run (0.76 | 0.0476 [0.5189, 0.5666] | 146 (100) | yes |
| POD1b_range_agreement_seeds_1_2_3 | range (lo=min, hi=max, value=max minus min) of agreement p_yes OOF AUC over seeds 1,2,3 | 0.0809 [0.8213, 0.9022] | 322 (175) | yes |
| POD1b_range_agreement_seeds_1_2_3_plus_original | range (lo=min, hi=max, value=max minus min) of agreement p_yes OOF AUC over seeds 1,2,3 plus original run (0.7 | 0.0809 [0.8213, 0.9022] | 322 (175) | yes |
| POD1b_seed0_p_yes_overall | EXTRA (not one of the three new seeds): seed 0 = original run's clip order RandomState(fold*10+ep) + torch.man | 0.7955 [0.7393, 0.8478] | 468 (228) | yes |
| POD1b_seed0_p_yes_conflict | EXTRA (not one of the three new seeds): seed 0 = original run's clip order RandomState(fold*10+ep) + torch.man | 0.5282 [0.4200, 0.6421] | 146 (100) | yes |
| POD1b_seed0_p_yes_agreement | EXTRA (not one of the three new seeds): seed 0 = original run's clip order RandomState(fold*10+ep) + torch.man | 0.8983 [0.8575, 0.9360] | 322 (175) | yes |
| POD1b_seed0_p_yes_conflict_minus_agreement | EXTRA (not one of the three new seeds): seed 0 = original run's clip order RandomState(fold*10+ep) + torch.man | -0.3702 [-0.4831, -0.2530] | 468 (228) | yes |
| POD1b_seed0_p_yes_multivariant_overall | EXTRA (not one of the three new seeds): seed 0 = original run's clip order RandomState(fold*10+ep) + torch.man | 0.7989 [0.7428, 0.8510] | 468 (228) | yes |
| POD1b_seed0_p_yes_multivariant_conflict | EXTRA (not one of the three new seeds): seed 0 = original run's clip order RandomState(fold*10+ep) + torch.man | 0.5344 [0.4269, 0.6503] | 146 (100) | yes |
| POD1b_seed0_p_yes_multivariant_agreement | EXTRA (not one of the three new seeds): seed 0 = original run's clip order RandomState(fold*10+ep) + torch.man | 0.9000 [0.8593, 0.9378] | 322 (175) | yes |
| POD1b_seed0_p_yes_multivariant_conflict_minus_agreement | EXTRA (not one of the three new seeds): seed 0 = original run's clip order RandomState(fold*10+ep) + torch.man | -0.3656 [-0.4796, -0.2501] | 468 (228) | yes |

### PART 20 POD2 ({'RELEASED ONLY': 50, 'SUPPORT': 7})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| POD2.bal.overall.2logit | balanced LoRA out-of-fold AUC, overall, Pitt 468 (column p_yes_2logit) | 0.7200 [0.6494, 0.7866] | 468 (228) | yes |
| POD2.earlier.overall.2logit | earlier LoRA (part16 POD2) out-of-fold AUC, overall, recomputed (column p_yes) | 0.8250 [0.7735, 0.8729] | 468 (228) | yes |
| POD2.diff.overall.2logit | PAIRED balanced minus earlier LoRA AUC, overall (2logit; one speaker draw per replicate scores both) | -0.1050 [-0.1577, -0.0584] | 468 (228) | yes |
| POD2.bal.conflict.2logit | balanced LoRA out-of-fold AUC, conflict, Pitt 468 (column p_yes_2logit) | 0.5770 [0.4586, 0.7006] | 146 (100) | yes |
| POD2.earlier.conflict.2logit | earlier LoRA (part16 POD2) out-of-fold AUC, conflict, recomputed (column p_yes) | 0.5520 [0.4348, 0.6833] | 146 (100) | yes |
| POD2.diff.conflict.2logit | PAIRED balanced minus earlier LoRA AUC, conflict (2logit; one speaker draw per replicate scores both) | 0.0250 [-0.0675, 0.1077] | 146 (100) | yes |
| POD2.bal.agreement.2logit | balanced LoRA out-of-fold AUC, agreement, Pitt 468 (column p_yes_2logit) | 0.7843 [0.7074, 0.8510] | 322 (175) | yes |
| POD2.earlier.agreement.2logit | earlier LoRA (part16 POD2) out-of-fold AUC, agreement, recomputed (column p_yes) | 0.9199 [0.8846, 0.9513] | 322 (175) | yes |
| POD2.diff.agreement.2logit | PAIRED balanced minus earlier LoRA AUC, agreement (2logit; one speaker draw per replicate scores both) | -0.1356 [-0.1977, -0.0848] | 322 (175) | yes |
| POD2.bal.cma.2logit | balanced LoRA conflict minus agreement AUC, paired over all 228 speakers (2logit) | -0.2073 [-0.3389, -0.0778] | 468 (228) | yes |
| POD2.gapchange.2logit | gap change: (conflict minus agreement) balanced minus the same for earlier LoRA, paired (2logit) | 0.1605 [0.0584, 0.2604] | 468 (228) | yes |
| POD2.bal.overall.multivariant | balanced LoRA out-of-fold AUC, overall, Pitt 468 (column p_yes (multi-variant)) | 0.7113 [0.6368, 0.7799] | 468 (228) | yes |
| POD2.earlier.overall.multivariant | earlier LoRA (part16 POD2) out-of-fold AUC, overall, recomputed (column p_yes_multivariant) | 0.8220 [0.7702, 0.8711] | 468 (228) | yes |
| POD2.diff.overall.multivariant | PAIRED balanced minus earlier LoRA AUC, overall (multivariant; one speaker draw per replicate scores both) | -0.1107 [-0.1659, -0.0624] | 468 (228) | yes |
| POD2.bal.conflict.multivariant | balanced LoRA out-of-fold AUC, conflict, Pitt 468 (column p_yes (multi-variant)) | 0.5736 [0.4549, 0.6939] | 146 (100) | yes |
| POD2.earlier.conflict.multivariant | earlier LoRA (part16 POD2) out-of-fold AUC, conflict, recomputed (column p_yes_multivariant) | 0.5516 [0.4333, 0.6814] | 146 (100) | yes |
| POD2.diff.conflict.multivariant | PAIRED balanced minus earlier LoRA AUC, conflict (multivariant; one speaker draw per replicate scores both) | 0.0220 [-0.0733, 0.1054] | 146 (100) | yes |
| POD2.bal.agreement.multivariant | balanced LoRA out-of-fold AUC, agreement, Pitt 468 (column p_yes (multi-variant)) | 0.7724 [0.6938, 0.8396] | 322 (175) | yes |
| POD2.earlier.agreement.multivariant | earlier LoRA (part16 POD2) out-of-fold AUC, agreement, recomputed (column p_yes_multivariant) | 0.9161 [0.8798, 0.9479] | 322 (175) | yes |
| POD2.diff.agreement.multivariant | PAIRED balanced minus earlier LoRA AUC, agreement (multivariant; one speaker draw per replicate scores both) | -0.1437 [-0.2070, -0.0920] | 322 (175) | yes |
| POD2.bal.cma.multivariant | balanced LoRA conflict minus agreement AUC, paired over all 228 speakers (multivariant) | -0.1988 [-0.3306, -0.0711] | 468 (228) | yes |
| POD2.gapchange.multivariant | gap change: (conflict minus agreement) balanced minus the same for earlier LoRA, paired (multivariant) | 0.1657 [0.0597, 0.2684] | 468 (228) | yes |
| POD2.params.lora | balanced LoRA trainable parameters (r=8 on q/k/v/o of 28 LM layers, 112 modules; verified analytically from co | 5046272 | 468 (228) | yes |
| POD2.beside_pod1.proj.overall.2logit | POD1 balanced projector fine-tune AUC, overall (column p_yes), re-verified here for the side-by-side | 0.7301 [0.6656, 0.7905] | 468 (228) | yes |
| POD2.diff_vs_pod1.overall.2logit | PAIRED balanced LoRA minus POD1 balanced projector AUC, overall (2logit: LoRA p_yes_2logit vs POD1 p_yes; one  | -0.0100 [-0.0901, 0.0678] | 468 (228) | yes |
| POD2.beside_pod1.proj.conflict.2logit | POD1 balanced projector fine-tune AUC, conflict (column p_yes), re-verified here for the side-by-side | 0.4693 [0.3686, 0.5854] | 146 (100) | yes |
| POD2.diff_vs_pod1.conflict.2logit | PAIRED balanced LoRA minus POD1 balanced projector AUC, conflict (2logit: LoRA p_yes_2logit vs POD1 p_yes; one | 0.1077 [-0.0263, 0.2383] | 146 (100) | yes |
| POD2.beside_pod1.proj.agreement.2logit | POD1 balanced projector fine-tune AUC, agreement (column p_yes), re-verified here for the side-by-side | 0.8279 [0.7732, 0.8792] | 322 (175) | yes |
| POD2.diff_vs_pod1.agreement.2logit | PAIRED balanced LoRA minus POD1 balanced projector AUC, agreement (2logit: LoRA p_yes_2logit vs POD1 p_yes; on | -0.0436 [-0.1286, 0.0325] | 322 (175) | yes |
| POD2.beside_pod1.proj.overall.multivariant | POD1 balanced projector fine-tune AUC, overall (column p_yes_multi), re-verified here for the side-by-side | 0.7407 [0.6761, 0.7994] | 468 (228) | yes |
| POD2.diff_vs_pod1.overall.multivariant | PAIRED balanced LoRA minus POD1 balanced projector AUC, overall (multivariant: LoRA p_yes vs POD1 p_yes_multi; | -0.0294 [-0.1152, 0.0525] | 468 (228) | yes |
| POD2.beside_pod1.proj.conflict.multivariant | POD1 balanced projector fine-tune AUC, conflict (column p_yes_multi), re-verified here for the side-by-side | 0.4671 [0.3641, 0.5818] | 146 (100) | yes |
| POD2.diff_vs_pod1.conflict.multivariant | PAIRED balanced LoRA minus POD1 balanced projector AUC, conflict (multivariant: LoRA p_yes vs POD1 p_yes_multi | 0.1065 [-0.0324, 0.2410] | 146 (100) | yes |
| POD2.beside_pod1.proj.agreement.multivariant | POD1 balanced projector fine-tune AUC, agreement (column p_yes_multi), re-verified here for the side-by-side | 0.8433 [0.7925, 0.8924] | 322 (175) | yes |
| POD2.diff_vs_pod1.agreement.multivariant | PAIRED balanced LoRA minus POD1 balanced projector AUC, agreement (multivariant: LoRA p_yes vs POD1 p_yes_mult | -0.0709 [-0.1586, 0.0086] | 322 (175) | yes |
| POD2.control.overall.2logit | same-pod uniform-weight LoRA control (w=1, same seeds) out-of-fold AUC, overall (2logit) | 0.8346 [0.7856, 0.8825] | 468 (228) | yes |
| POD2.control_minus_earlier.overall.2logit | PAIRED uniform control minus earlier LoRA AUC, overall (run-to-run variation of the recipe) (2logit) | 0.0096 [-0.0204, 0.0437] | 468 (228) | yes |
| POD2.bal_minus_control.overall.2logit | PAIRED balanced LoRA minus same-pod uniform control AUC, overall (effect of the weights alone) (2logit) | -0.1146 [-0.1751, -0.0564] | 468 (228) | yes |
| POD2.control.conflict.2logit | same-pod uniform-weight LoRA control (w=1, same seeds) out-of-fold AUC, conflict (2logit) | 0.6198 [0.4963, 0.7470] | 146 (100) | yes |
| POD2.control_minus_earlier.conflict.2logit | PAIRED uniform control minus earlier LoRA AUC, conflict (run-to-run variation of the recipe) (2logit) | 0.0678 [-0.0108, 0.1491] | 146 (100) | yes |
| POD2.bal_minus_control.conflict.2logit | PAIRED balanced LoRA minus same-pod uniform control AUC, conflict (effect of the weights alone) (2logit) | -0.0428 [-0.1707, 0.0837] | 146 (100) | yes |
| POD2.control.agreement.2logit | same-pod uniform-weight LoRA control (w=1, same seeds) out-of-fold AUC, agreement (2logit) | 0.9112 [0.8710, 0.9471] | 322 (175) | yes |
| POD2.control_minus_earlier.agreement.2logit | PAIRED uniform control minus earlier LoRA AUC, agreement (run-to-run variation of the recipe) (2logit) | -0.0087 [-0.0311, 0.0130] | 322 (175) | yes |
| POD2.bal_minus_control.agreement.2logit | PAIRED balanced LoRA minus same-pod uniform control AUC, agreement (effect of the weights alone) (2logit) | -0.1268 [-0.1937, -0.0730] | 322 (175) | yes |
| POD2.control.cma.2logit | uniform control conflict minus agreement AUC, paired over all 228 speakers (2logit) | -0.2914 [-0.4204, -0.1653] | 468 (228) | yes |
| POD2.gapchange_bal_minus_control.2logit | gap change: (conflict minus agreement) balanced minus the same for the uniform control, paired (2logit) | 0.0841 [-0.0551, 0.2137] | 468 (228) | yes |
| POD2.control.overall.multivariant | same-pod uniform-weight LoRA control (w=1, same seeds) out-of-fold AUC, overall (multivariant) | 0.8103 [0.7564, 0.8630] | 468 (228) | yes |
| POD2.control_minus_earlier.overall.multivariant | PAIRED uniform control minus earlier LoRA AUC, overall (run-to-run variation of the recipe) (multivariant) | -0.0117 [-0.0359, 0.0134] | 468 (228) | yes |
| POD2.bal_minus_control.overall.multivariant | PAIRED balanced LoRA minus same-pod uniform control AUC, overall (effect of the weights alone) (multivariant) | -0.0991 [-0.1581, -0.0431] | 468 (228) | yes |
| POD2.control.conflict.multivariant | same-pod uniform-weight LoRA control (w=1, same seeds) out-of-fold AUC, conflict (multivariant) | 0.5547 [0.4339, 0.6834] | 146 (100) | yes |
| POD2.control_minus_earlier.conflict.multivariant | PAIRED uniform control minus earlier LoRA AUC, conflict (run-to-run variation of the recipe) (multivariant) | 0.0031 [-0.0573, 0.0633] | 146 (100) | yes |
| POD2.bal_minus_control.conflict.multivariant | PAIRED balanced LoRA minus same-pod uniform control AUC, conflict (effect of the weights alone) (multivariant) | 0.0189 [-0.0934, 0.1316] | 146 (100) | yes |
| POD2.control.agreement.multivariant | same-pod uniform-weight LoRA control (w=1, same seeds) out-of-fold AUC, agreement (multivariant) | 0.8989 [0.8559, 0.9383] | 322 (175) | yes |
| POD2.control_minus_earlier.agreement.multivariant | PAIRED uniform control minus earlier LoRA AUC, agreement (run-to-run variation of the recipe) (multivariant) | -0.0172 [-0.0380, 0.0026] | 322 (175) | yes |
| POD2.bal_minus_control.agreement.multivariant | PAIRED balanced LoRA minus same-pod uniform control AUC, agreement (effect of the weights alone) (multivariant | -0.1264 [-0.1938, -0.0715] | 322 (175) | yes |
| POD2.control.cma.multivariant | uniform control conflict minus agreement AUC, paired over all 228 speakers (multivariant) | -0.3442 [-0.4727, -0.2122] | 468 (228) | yes |
| POD2.gapchange_bal_minus_control.multivariant | gap change: (conflict minus agreement) balanced minus the same for the uniform control, paired (multivariant) | 0.1453 [0.0249, 0.2646] | 468 (228) | yes |

### PART 20 POD3 ({'RELEASED ONLY': 5})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| POD3_ft966_conflict | Omni projector FT on 966 E-DAIC clips, conflict AUC (483 rows) | 0.5932 [0.5087, 0.6831] | 483 (138) | yes |
| POD3_ft966_agreement483 | Omni projector FT on 966 E-DAIC clips, agreement AUC (483 rows) | 0.6228 [0.5222, 0.7238] | 483 (138) | yes |
| POD3_ft966_agreement311 | Omni projector FT on 966 E-DAIC clips, agreement AUC (311 distinct segments) | 0.6686 [0.5727, 0.7586] | 311 (138) | yes |
| POD3_ft966_paired_conf_minus_agr483 | Omni projector FT on 966 clips, paired conflict minus agreement (483 rows) | -0.0296 [-0.1124, 0.0491] | 966 (138) | yes |
| POD3_ft966_paired_conf_minus_agr311 | Omni projector FT on 966 clips, paired conflict minus agreement (311 distinct) | -0.0754 [-0.1513, 0.0010] | 794 (138) | yes |

### PART 20 POD3b ({'RELEASED ONLY': 14, 'SUPPORT': 5})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| POD3b_sst2_o25_audio_conflict | Qwen2.5-Omni audio zero-shot, DistilBERT SST-2 pairs, conflict arm AUC (1334 clips, 1334 distinct segments) | 0.4619 [0.4165, 0.5102] | 1334 (162) | yes |
| POD3b_sst2_o25_audio_agreement_all | Qwen2.5-Omni audio zero-shot, DistilBERT SST-2 pairs, agreement arm AUC, all rows (1334 rows, 799 distinct seg | 0.8316 [0.7766, 0.8795] | 1334 (162) | yes |
| POD3b_sst2_o25_audio_agreement_distinct | Qwen2.5-Omni audio zero-shot, DistilBERT SST-2 pairs, agreement arm AUC, distinct segments | 0.8554 [0.8199, 0.8885] | 799 (162) | yes |
| POD3b_sst2_o25_audio_diff_all | paired conflict minus agreement AUC, all rows (one speaker draw per replicate) | -0.3697 [-0.4269, -0.3073] | 2668 (162) | yes |
| POD3b_sst2_o25_audio_diff_distinct | paired conflict minus agreement AUC, distinct agreement segments | -0.3934 [-0.4420, -0.3410] | 2133 (162) | yes |
| POD3b_p14_rescore_envcheck_conflict | ENV CHECK: Part 14 RoBERTa set re-cut and re-scored on POD3b (tf 5.17.0, H100 NVL), conflict AUC (published 0. | 0.2210 [0.1818, 0.2624] | 483 (138) | yes |
| POD3b_p14_rescore_envcheck_agreement_all | ENV CHECK: Part 14 set re-scored, agreement AUC all rows (published 0.9482) | 0.9467 [0.9017, 0.9827] | 483 (138) | yes |
| POD3b_p14_rescore_envcheck_agreement_distinct | ENV CHECK: Part 14 set re-scored, agreement AUC 311 distinct segments (published 0.9647) | 0.9635 [0.9352, 0.9856] | 311 (138) | yes |
| POD3b_p14_rescore_envcheck_diff_all | ENV CHECK: Part 14 set re-scored, paired conflict minus agreement, all rows (published -0.7264) | -0.7257 [-0.7800, -0.6637] | 966 (138) | yes |
| POD3b_p14_rescore_envcheck_diff_distinct | ENV CHECK: Part 14 set re-scored, paired conflict minus agreement, distinct (published -0.7429) | -0.7425 [-0.7880, -0.6930] | 794 (138) | yes |
| POD3b_diag_conflict_robconflict | DIAG: SST-2 conflict rows whose RoBERTa rule arm is also conflict | 0.1951 [0.1545, 0.2413] | 420 (137) | yes |
| POD3b_diag_conflict_robnone | DIAG: SST-2 conflict rows whose RoBERTa rule arm is none | 0.6150 [0.5593, 0.6755] | 855 (160) | yes |
| POD3b_diag_agreement_robagreement_distinct | DIAG: SST-2 agreement distinct segments whose RoBERTa rule arm is also agreement | 0.9680 [0.9425, 0.9877] | 391 (145) | yes |
| POD3b_diag_agreement_robnone_distinct | DIAG: SST-2 agreement distinct segments whose RoBERTa rule arm is none | 0.7518 [0.6818, 0.8100] | 379 (141) | yes |
| POD3b_diag_conflict_p14speakers | DIAG: SST-2 conflict rows, 138 Part 14 speakers only | 0.4489 [0.4011, 0.4982] | 1236 (138) | yes |
| POD3b_diag_agreement_distinct_p14speakers | DIAG: SST-2 agreement distinct segments, 138 Part 14 speakers only | 0.8587 [0.8201, 0.8912] | 737 (138) | yes |
| POD3b_val20_score_maxabsdiff | VALIDATION: max abs p_yes diff, 20 Part 14 clips re-scored on POD3b (tf 5.17.0) vs published part14 scores; cr | 0.0107 | 20 | yes |
| POD3b_val20_n_within_1e-3 | VALIDATION: clips within 1e-3 of published p_yes, of 20 | 10 | 20 | yes |
| POD3b_val20_cut_identical | VALIDATION: re-cut clips sample-identical to part14_clips (20 of 20; extended 794 of 794, verifier frame-hash) | 794/794 | 794 (138) | yes |

### PART 20 POD3c ({'SUPPORT': 28, 'RELEASED ONLY': 16, 'SUPERSEDED / NOT USED': 2})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| POD3c_q3o_transcript_conflict483 | Qwen3-Omni transcript AUC, conflict483 | 0.2243 [0.1757, 0.2742] | 483 (138) | yes |
| POD3c_q3o_transcript_agreement483 | Qwen3-Omni transcript AUC, agreement483 | 0.9470 [0.9131, 0.9757] | 483 (138) | yes |
| POD3c_q3o_transcript_agreement311 | Qwen3-Omni transcript AUC, agreement311 | 0.9523 [0.9249, 0.9750] | 311 (138) | yes |
| POD3c_q3o_audio_agreement483 | Qwen3-Omni audio AUC, agreement483 | 0.9638 [0.9336, 0.9864] | 483 (138) | yes |
| POD3c_q3o_transcript_conf_minus_agree483 | Qwen3-Omni transcript, conflict483 minus agreement483 (paired) | -0.7227 [-0.7803, -0.6620] | 966 (138) | yes |
| POD3c_q3o_transcript_conf_minus_agree311 | Qwen3-Omni transcript, conflict483 minus agreement311 (paired) | -0.7280 [-0.7805, -0.6731] | 794 (138) | yes |
| POD3c_q3o_audio_conf_minus_agree483 | Qwen3-Omni audio, conflict483 minus agreement483 (paired) | -0.6630 [-0.7192, -0.6030] | 966 (138) | yes |
| POD3c_q3o_text_minus_audio_conflict483 | Qwen3-Omni transcript minus audio AUC, conflict483 (paired) | -0.0765 [-0.1286, -0.0296] | 483 (138) | yes |
| POD3c_q3o_text_minus_audio_agreement483 | Qwen3-Omni transcript minus audio AUC, agreement483 (paired) | -0.0168 [-0.0399, 0.0039] | 483 (138) | yes |
| POD3c_q3o_text_minus_audio_agreement311 | Qwen3-Omni transcript minus audio AUC, agreement311 (paired) | -0.0204 [-0.0402, -0.0034] | 311 (138) | yes |
| POD3c_pearson_r_all966 | Pearson r, Q3O transcript p_yes vs Q3O audio p_yes, all966 | 0.7634 | 966 (138) | yes |
| POD3c_same_decision_share_all966 | same decision share (p_yes>=0.5), transcript vs audio, all966 | 0.8313 | 966 (138) | yes |
| POD3c_text_yes_rate_all966 | transcript yes rate (p_yes>=0.5), all966 | 0.3219 | 966 (138) | yes |
| POD3c_audio_yes_rate_all966 | audio yes rate (p_yes>=0.5), all966 | 0.4037 | 966 (138) | yes |
| POD3c_pearson_r_distinct794 | Pearson r, Q3O transcript p_yes vs Q3O audio p_yes, distinct794 | 0.7399 | 794 (138) | yes |
| POD3c_same_decision_share_distinct794 | same decision share (p_yes>=0.5), transcript vs audio, distinct794 | 0.8224 | 794 (138) | yes |
| POD3c_text_yes_rate_distinct794 | transcript yes rate (p_yes>=0.5), distinct794 | 0.2922 | 794 (138) | yes |
| POD3c_audio_yes_rate_distinct794 | audio yes rate (p_yes>=0.5), distinct794 | 0.3690 | 794 (138) | yes |
| POD3c_pearson_r_conflict483 | Pearson r, Q3O transcript p_yes vs Q3O audio p_yes, conflict483 | 0.6488 | 483 (138) | yes |
| POD3c_same_decision_share_conflict483 | same decision share (p_yes>=0.5), transcript vs audio, conflict483 | 0.7785 | 483 (138) | yes |
| POD3c_text_yes_rate_conflict483 | transcript yes rate (p_yes>=0.5), conflict483 | 0.3126 | 483 (138) | yes |
| POD3c_audio_yes_rate_conflict483 | audio yes rate (p_yes>=0.5), conflict483 | 0.3892 | 483 (138) | yes |
| POD3c_pearson_r_agreement483 | Pearson r, Q3O transcript p_yes vs Q3O audio p_yes, agreement483 | 0.8475 | 483 (138) | yes |
| POD3c_same_decision_share_agreement483 | same decision share (p_yes>=0.5), transcript vs audio, agreement483 | 0.8841 | 483 (138) | yes |
| POD3c_text_yes_rate_agreement483 | transcript yes rate (p_yes>=0.5), agreement483 | 0.3313 | 483 (138) | yes |
| POD3c_audio_yes_rate_agreement483 | audio yes rate (p_yes>=0.5), agreement483 | 0.4182 | 483 (138) | yes |
| POD3c_pearson_r_agreement311 | Pearson r, Q3O transcript p_yes vs Q3O audio p_yes, agreement311 | 0.8516 | 311 (138) | yes |
| POD3c_same_decision_share_agreement311 | same decision share (p_yes>=0.5), transcript vs audio, agreement311 | 0.8907 | 311 (138) | yes |
| POD3c_text_yes_rate_agreement311 | transcript yes rate (p_yes>=0.5), agreement311 | 0.2605 | 311 (138) | yes |
| POD3c_audio_yes_rate_agreement311 | audio yes rate (p_yes>=0.5), agreement311 | 0.3376 | 311 (138) | yes |
| POD3c_tokenset_max_abs_p_yes_ts_minus_rule3 | token set sensitivity (Yeah/Nope added): max_abs_p_yes_ts_minus_rule3 | 0.0000 | 966 (138) | yes |
| POD3c_tokenset_conflict483_auc_ts | token set sensitivity (Yeah/Nope added): conflict483_auc_ts | 0.2248 | 483 (138) | yes |
| POD3c_tokenset_agreement483_auc_ts | token set sensitivity (Yeah/Nope added): agreement483_auc_ts | 0.9471 | 483 (138) | yes |
| POD3c_tokenset_agreement311_auc_ts | token set sensitivity (Yeah/Nope added): agreement311_auc_ts | 0.9522 | 311 (138) | yes |
| POD3c_tokenset_all966_auc_rule3 | token set sensitivity (Yeah/Nope added): all966_auc_rule3 | 0.6284 | 966 (138) | yes |
| POD3c_tokenset_all966_auc_ts | token set sensitivity (Yeah/Nope added): all966_auc_ts | 0.6285 | 966 (138) | yes |
| POD3c_median_mass_text_all966 | transcript Yes+No answer mass, median, all966 | 0.9999 | 966 (138) | yes |
| POD3c_min_mass_text_all966 | transcript Yes+No answer mass, min, all966 | 0.9969 | 966 (138) | yes |
| POD3c_median_mass_text_distinct794 | transcript Yes+No answer mass, median, distinct794 | 0.9999 | 794 (138) | yes |
| POD3c_min_mass_text_distinct794 | transcript Yes+No answer mass, min, distinct794 | 0.9969 | 794 (138) | yes |
| POD3c_median_mass_text_conflict483 | transcript Yes+No answer mass, median, conflict483 | 0.9998 | 483 (138) | yes |
| POD3c_min_mass_text_conflict483 | transcript Yes+No answer mass, min, conflict483 | 0.9969 | 483 (138) | yes |
| POD3c_median_mass_text_agreement483 | transcript Yes+No answer mass, median, agreement483 | 0.9999 | 483 (138) | yes |
| POD3c_min_mass_text_agreement483 | transcript Yes+No answer mass, min, agreement483 | 0.9990 | 483 (138) | yes |
| POD3c_median_mass_text_agreement311 | transcript Yes+No answer mass, median, agreement311 | 0.9999 | 311 (138) | yes |
| POD3c_min_mass_text_agreement311 | transcript Yes+No answer mass, min, agreement311 | 0.9990 | 311 (138) | yes |

### PART 20 POD4 ({'RELEASED ONLY': 63, 'SUPPORT': 1})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| q2a_noinv_recording_overall | q2a_noinv_recording zero-shot AUC, overall | 0.6211 [0.5534, 0.6831] | 468 (228) | yes |
| q2a_noinv_recording_conflict | q2a_noinv_recording zero-shot AUC, conflict | 0.4253 [0.3084, 0.5494] | 146 (100) | yes |
| q2a_noinv_recording_agreement | q2a_noinv_recording zero-shot AUC, agreement | 0.7129 [0.6517, 0.7702] | 322 (175) | yes |
| paper_q2a_orig_recording_overall | paper q2a zero-shot on the original 468 windows, recording wording, overall (recomputed) | 0.6173 [0.5541, 0.6799] | 468 (228) | yes |
| q2a_noinv_recording_PAIRED_paperorig_minus_noinv_overall | PAIRED original (paper file) minus interviewer-free, q2a, overall | -0.0037 [-0.0438, 0.0357] | 468 (228) | yes |
| q2a_noinv_recording_PAIRED_paperorig_minus_noinv_conflict | PAIRED original (paper file) minus interviewer-free, q2a, conflict | -0.0121 [-0.0924, 0.0619] | 146 (100) | yes |
| q2a_noinv_recording_PAIRED_paperorig_minus_noinv_agreement | PAIRED original (paper file) minus interviewer-free, q2a, agreement | -0.0144 [-0.0619, 0.0331] | 322 (175) | yes |
| q2a_noinv_recording_PAIRED_podorig_minus_noinv_overall | PAIRED original (this pod's rerun, same code) minus interviewer-free, q2a, overall | -0.0036 [-0.0441, 0.0359] | 468 (228) | yes |
| q2a_noinv_recording_PAIRED_podorig_minus_noinv_conflict | PAIRED original (this pod's rerun, same code) minus interviewer-free, q2a, conflict | -0.0121 [-0.0921, 0.0609] | 146 (100) | yes |
| q2a_noinv_recording_PAIRED_podorig_minus_noinv_agreement | PAIRED original (this pod's rerun, same code) minus interviewer-free, q2a, agreement | -0.0142 [-0.0620, 0.0337] | 322 (175) | yes |
| o25_noinv_recording_overall | o25_noinv_recording zero-shot AUC, overall | 0.7230 [0.6664, 0.7753] | 468 (228) | yes |
| o25_noinv_recording_conflict | o25_noinv_recording zero-shot AUC, conflict | 0.5408 [0.4245, 0.6622] | 146 (100) | yes |
| o25_noinv_recording_agreement | o25_noinv_recording zero-shot AUC, agreement | 0.8007 [0.7500, 0.8501] | 322 (175) | yes |
| o25_noinv_recording_PAIRED_paperorig_minus_noinv_overall | PAIRED original (paper file) minus interviewer-free, o25, overall | -0.0652 [-0.1081, -0.0228] | 468 (228) | yes |
| o25_noinv_recording_PAIRED_paperorig_minus_noinv_conflict | PAIRED original (paper file) minus interviewer-free, o25, conflict | -0.0086 [-0.0890, 0.0679] | 146 (100) | yes |
| o25_noinv_recording_PAIRED_paperorig_minus_noinv_agreement | PAIRED original (paper file) minus interviewer-free, o25, agreement | -0.0954 [-0.1512, -0.0401] | 322 (175) | yes |
| o25_noinv_recording_PAIRED_podorig_minus_noinv_overall | PAIRED original (this pod's rerun, same code) minus interviewer-free, o25, overall | -0.0732 [-0.1165, -0.0312] | 468 (228) | yes |
| o25_noinv_recording_PAIRED_podorig_minus_noinv_conflict | PAIRED original (this pod's rerun, same code) minus interviewer-free, o25, conflict | -0.0110 [-0.0937, 0.0671] | 146 (100) | yes |
| o25_noinv_recording_PAIRED_podorig_minus_noinv_agreement | PAIRED original (this pod's rerun, same code) minus interviewer-free, o25, agreement | -0.1056 [-0.1613, -0.0489] | 322 (175) | yes |
| o25_orig_recording_rerun_overall | o25_orig_recording_rerun zero-shot AUC, overall | 0.6497 [0.5959, 0.7056] | 468 (228) | yes |
| o25_orig_recording_rerun_conflict | o25_orig_recording_rerun zero-shot AUC, conflict | 0.5298 [0.4189, 0.6386] | 146 (100) | yes |
| o25_orig_recording_rerun_agreement | o25_orig_recording_rerun zero-shot AUC, agreement | 0.6952 [0.6327, 0.7560] | 322 (175) | yes |
| o25_orig_recording_rerun_PAIRED_podrerun_minus_paper_overall | PAIRED this pod's rerun minus paper file, o25, recording wording, original windows, overall | -0.0080 [-0.0160, 0.0002] | 468 (228) | yes |
| o25_orig_recording_rerun_PAIRED_podrerun_minus_paper_conflict | PAIRED this pod's rerun minus paper file, o25, recording wording, original windows, conflict | -0.0024 [-0.0218, 0.0150] | 146 (100) | yes |
| o25_orig_recording_rerun_PAIRED_podrerun_minus_paper_agreement | PAIRED this pod's rerun minus paper file, o25, recording wording, original windows, agreement | -0.0102 [-0.0196, -0.0011] | 322 (175) | yes |
| o25_orig_recording_rerun_PAIRED_podorig_minus_noinv_overall | PAIRED original (this pod's rerun, same code) minus interviewer-free, o25, overall | -0.0732 [-0.1165, -0.0312] | 468 (228) | yes |
| o25_orig_recording_rerun_PAIRED_podorig_minus_noinv_conflict | PAIRED original (this pod's rerun, same code) minus interviewer-free, o25, conflict | -0.0110 [-0.0937, 0.0671] | 146 (100) | yes |
| o25_orig_recording_rerun_PAIRED_podorig_minus_noinv_agreement | PAIRED original (this pod's rerun, same code) minus interviewer-free, o25, agreement | -0.1056 [-0.1613, -0.0489] | 322 (175) | yes |
| o25_orig_voice_overall | o25_orig_voice zero-shot AUC, overall | 0.6473 [0.5931, 0.7039] | 468 (228) | yes |
| o25_orig_voice_conflict | o25_orig_voice zero-shot AUC, conflict | 0.5619 [0.4535, 0.6670] | 146 (100) | yes |
| o25_orig_voice_agreement | o25_orig_voice zero-shot AUC, agreement | 0.6841 [0.6231, 0.7470] | 322 (175) | yes |
| o25_orig_voice_PAIRED_voice_minus_paperrecording_overall | PAIRED voice wording minus recording wording (paper file), o25, original windows, overall | -0.0104 [-0.0356, 0.0155] | 468 (228) | yes |
| o25_orig_voice_PAIRED_voice_minus_paperrecording_conflict | PAIRED voice wording minus recording wording (paper file), o25, original windows, conflict | 0.0297 [-0.0144, 0.0733] | 146 (100) | yes |
| o25_orig_voice_PAIRED_voice_minus_paperrecording_agreement | PAIRED voice wording minus recording wording (paper file), o25, original windows, agreement | -0.0212 [-0.0519, 0.0115] | 322 (175) | yes |
| o25_orig_voice_PAIRED_voice_minus_podrecording_overall | PAIRED voice wording minus recording wording (this pod's rerun, same code), o25, overall | -0.0024 [-0.0265, 0.0221] | 468 (228) | yes |
| o25_orig_voice_PAIRED_voice_minus_podrecording_conflict | PAIRED voice wording minus recording wording (this pod's rerun, same code), o25, conflict | 0.0321 [-0.0099, 0.0732] | 146 (100) | yes |
| o25_orig_voice_PAIRED_voice_minus_podrecording_agreement | PAIRED voice wording minus recording wording (this pod's rerun, same code), o25, agreement | -0.0110 [-0.0407, 0.0198] | 322 (175) | yes |
| q2a_noinv_recording_bf16_overall | q2a_noinv_recording_bf16 zero-shot AUC, overall | 0.6236 [0.5548, 0.6856] | 468 (228) | yes |
| q2a_noinv_recording_bf16_conflict | q2a_noinv_recording_bf16 zero-shot AUC, conflict | 0.4255 [0.3076, 0.5504] | 146 (100) | yes |
| q2a_noinv_recording_bf16_agreement | q2a_noinv_recording_bf16 zero-shot AUC, agreement | 0.7150 [0.6540, 0.7716] | 322 (175) | yes |
| q2a_noinv_recording_bf16_PAIRED_paperorig_minus_noinv_overall | PAIRED original (paper file) minus interviewer-free, q2a, overall | -0.0062 [-0.0457, 0.0324] | 468 (228) | yes |
| q2a_noinv_recording_bf16_PAIRED_paperorig_minus_noinv_conflict | PAIRED original (paper file) minus interviewer-free, q2a, conflict | -0.0123 [-0.0925, 0.0621] | 146 (100) | yes |
| q2a_noinv_recording_bf16_PAIRED_paperorig_minus_noinv_agreement | PAIRED original (paper file) minus interviewer-free, q2a, agreement | -0.0165 [-0.0640, 0.0314] | 322 (175) | yes |
| q2a_noinv_recording_bf16_PAIRED_podorig_minus_noinv_overall | PAIRED original (this pod's rerun, same code) minus interviewer-free, q2a, overall | -0.0030 [-0.0430, 0.0362] | 468 (228) | yes |
| q2a_noinv_recording_bf16_PAIRED_podorig_minus_noinv_conflict | PAIRED original (this pod's rerun, same code) minus interviewer-free, q2a, conflict | -0.0125 [-0.0919, 0.0608] | 146 (100) | yes |
| q2a_noinv_recording_bf16_PAIRED_podorig_minus_noinv_agreement | PAIRED original (this pod's rerun, same code) minus interviewer-free, q2a, agreement | -0.0117 [-0.0598, 0.0354] | 322 (175) | yes |
| q2a_orig_recording_rerun_overall | q2a_orig_recording_rerun zero-shot AUC, overall | 0.6174 [0.5546, 0.6800] | 468 (228) | yes |
| q2a_orig_recording_rerun_conflict | q2a_orig_recording_rerun zero-shot AUC, conflict | 0.4132 [0.3008, 0.5433] | 146 (100) | yes |
| q2a_orig_recording_rerun_agreement | q2a_orig_recording_rerun zero-shot AUC, agreement | 0.6987 [0.6374, 0.7597] | 322 (175) | yes |
| q2a_orig_recording_rerun_PAIRED_podrerun_minus_paper_overall | PAIRED this pod's rerun minus paper file, q2a, recording wording, original windows, overall | 0.0001 [-0.0009, 0.0011] | 468 (228) | yes |
| q2a_orig_recording_rerun_PAIRED_podrerun_minus_paper_conflict | PAIRED this pod's rerun minus paper file, q2a, recording wording, original windows, conflict | 0 [-0.0031, 0.0029] | 146 (100) | yes |
| q2a_orig_recording_rerun_PAIRED_podrerun_minus_paper_agreement | PAIRED this pod's rerun minus paper file, q2a, recording wording, original windows, agreement | 0.0002 [-0.0009, 0.0014] | 322 (175) | yes |
| q2a_orig_recording_rerun_PAIRED_podorig_minus_noinv_overall | PAIRED original (this pod's rerun, same code) minus interviewer-free, q2a, overall | -0.0036 [-0.0441, 0.0359] | 468 (228) | yes |
| q2a_orig_recording_rerun_PAIRED_podorig_minus_noinv_conflict | PAIRED original (this pod's rerun, same code) minus interviewer-free, q2a, conflict | -0.0121 [-0.0921, 0.0609] | 146 (100) | yes |
| q2a_orig_recording_rerun_PAIRED_podorig_minus_noinv_agreement | PAIRED original (this pod's rerun, same code) minus interviewer-free, q2a, agreement | -0.0142 [-0.0620, 0.0337] | 322 (175) | yes |
| q2a_orig_recording_rerun_bf16_overall | q2a_orig_recording_rerun_bf16 zero-shot AUC, overall | 0.6206 [0.5581, 0.6832] | 468 (228) | yes |
| q2a_orig_recording_rerun_bf16_conflict | q2a_orig_recording_rerun_bf16 zero-shot AUC, conflict | 0.4130 [0.3031, 0.5406] | 146 (100) | yes |
| q2a_orig_recording_rerun_bf16_agreement | q2a_orig_recording_rerun_bf16 zero-shot AUC, agreement | 0.7033 [0.6426, 0.7646] | 322 (175) | yes |
| q2a_orig_recording_rerun_bf16_PAIRED_podrerun_minus_paper_overall | PAIRED this pod's rerun minus paper file, q2a, recording wording, original windows, overall | 0.0033 [-0.0008, 0.0072] | 468 (228) | yes |
| q2a_orig_recording_rerun_bf16_PAIRED_podrerun_minus_paper_conflict | PAIRED this pod's rerun minus paper file, q2a, recording wording, original windows, conflict | -0.0002 [-0.0072, 0.0063] | 146 (100) | yes |
| q2a_orig_recording_rerun_bf16_PAIRED_podrerun_minus_paper_agreement | PAIRED this pod's rerun minus paper file, q2a, recording wording, original windows, agreement | 0.0049 [-0.0001, 0.0098] | 322 (175) | yes |
| q2a_orig_recording_rerun_bf16_PAIRED_podorig_minus_noinv_overall | PAIRED original (this pod's rerun, same code) minus interviewer-free, q2a, overall | -0.0030 [-0.0430, 0.0362] | 468 (228) | yes |
| q2a_orig_recording_rerun_bf16_PAIRED_podorig_minus_noinv_conflict | PAIRED original (this pod's rerun, same code) minus interviewer-free, q2a, conflict | -0.0125 [-0.0919, 0.0608] | 146 (100) | yes |
| q2a_orig_recording_rerun_bf16_PAIRED_podorig_minus_noinv_agreement | PAIRED original (this pod's rerun, same code) minus interviewer-free, q2a, agreement | -0.0117 [-0.0598, 0.0354] | 322 (175) | yes |

### PART 20 POD4b ({'RELEASED ONLY': 18, 'SUPPORT': 1})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| P20_4b_q3o_noinv_overall | Qwen3-Omni zero-shot AUC, Pitt, interviewer-free cut (this pod), overall | 0.7802 [0.7265, 0.8270] | 468 (228) | yes |
| P20_4b_q3o_noinv_conflict | Qwen3-Omni zero-shot AUC, Pitt, interviewer-free cut (this pod), conflict | 0.6187 [0.5145, 0.7219] | 146 (100) | yes |
| P20_4b_q3o_noinv_agreement | Qwen3-Omni zero-shot AUC, Pitt, interviewer-free cut (this pod), agreement | 0.8439 [0.7958, 0.8880] | 322 (175) | yes |
| P20_4b_q3o_orig_shipped_overall | Qwen3-Omni zero-shot AUC, Pitt, original windows, shipped paper file, overall | 0.7619 [0.7054, 0.8130] | 468 (228) | yes |
| P20_4b_q3o_orig_rerun_overall | Qwen3-Omni zero-shot AUC, Pitt, original windows, part16 rerun file, overall | 0.7673 [0.7123, 0.8200] | 468 (228) | yes |
| P20_4b_q3o_orig_rerun_conflict | Qwen3-Omni zero-shot AUC, Pitt, original windows, part16 rerun file, conflict | 0.6079 [0.5012, 0.7132] | 146 (100) | yes |
| P20_4b_q3o_orig_rerun_agreement | Qwen3-Omni zero-shot AUC, Pitt, original windows, part16 rerun file, agreement | 0.8265 [0.7729, 0.8740] | 322 (175) | yes |
| P20_4b_q3o_diff_orig_shipped_minus_noinv_overall | PAIRED Qwen3-Omni zero-shot AUC, original windows, shipped paper file minus interviewer-free, overall | -0.0182 [-0.0455, 0.0083] | 468 (228) | yes |
| P20_4b_q3o_diff_orig_shipped_minus_noinv_conflict | PAIRED Qwen3-Omni zero-shot AUC, original windows, shipped paper file minus interviewer-free, conflict | -0.0121 [-0.0664, 0.0422] | 146 (100) | yes |
| P20_4b_q3o_diff_orig_shipped_minus_noinv_agreement | PAIRED Qwen3-Omni zero-shot AUC, original windows, shipped paper file minus interviewer-free, agreement | -0.0207 [-0.0522, 0.0097] | 322 (175) | yes |
| P20_4b_q3o_diff_orig_rerun_minus_noinv_overall | PAIRED Qwen3-Omni zero-shot AUC, original windows, part16 rerun file minus interviewer-free, overall | -0.0128 [-0.0417, 0.0152] | 468 (228) | yes |
| P20_4b_q3o_diff_orig_rerun_minus_noinv_conflict | PAIRED Qwen3-Omni zero-shot AUC, original windows, part16 rerun file minus interviewer-free, conflict | -0.0108 [-0.0665, 0.0488] | 146 (100) | yes |
| P20_4b_q3o_diff_orig_rerun_minus_noinv_agreement | PAIRED Qwen3-Omni zero-shot AUC, original windows, part16 rerun file minus interviewer-free, agreement | -0.0174 [-0.0498, 0.0137] | 322 (175) | yes |
| P20_4b_q3o_orig_samepod_overall | Qwen3-Omni zero-shot AUC, Pitt, original windows re-scored on this pod, overall | 0.7638 [0.7074, 0.8166] | 468 (228) | yes |
| P20_4b_q3o_orig_samepod_conflict | Qwen3-Omni zero-shot AUC, Pitt, original windows re-scored on this pod, conflict | 0.6062 [0.4988, 0.7141] | 146 (100) | yes |
| P20_4b_q3o_orig_samepod_agreement | Qwen3-Omni zero-shot AUC, Pitt, original windows re-scored on this pod, agreement | 0.8221 [0.7686, 0.8718] | 322 (175) | yes |
| P20_4b_q3o_diff_orig_samepod_minus_noinv_overall | PAIRED Qwen3-Omni zero-shot AUC, original windows re-scored on this pod minus interviewer-free, overall | -0.0164 [-0.0452, 0.0117] | 468 (228) | yes |
| P20_4b_q3o_diff_orig_samepod_minus_noinv_conflict | PAIRED Qwen3-Omni zero-shot AUC, original windows re-scored on this pod minus interviewer-free, conflict | -0.0125 [-0.0671, 0.0478] | 146 (100) | yes |
| P20_4b_q3o_diff_orig_samepod_minus_noinv_agreement | PAIRED Qwen3-Omni zero-shot AUC, original windows re-scored on this pod minus interviewer-free, agreement | -0.0217 [-0.0542, 0.0097] | 322 (175) | yes |

### PART 20 POD4c ({'RELEASED ONLY': 24, 'SUPPORT': 3})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| af3_diff_part10_minus_noinv_agreement | AF3 paired AUC difference, original (part10) minus interviewer-free, agreement | -0.0285 [-0.0959, 0.0404] | 322 (175) | yes |
| af3_diff_part10_minus_noinv_all | AF3 paired AUC difference, original (part10) minus interviewer-free, all | -0.0026 [-0.0600, 0.0548] | 468 (228) | yes |
| af3_diff_part10_minus_noinv_conflict | AF3 paired AUC difference, original (part10) minus interviewer-free, conflict | 0.0658 [-0.0359, 0.1653] | 146 (100) | yes |
| af3_diff_part3_minus_noinv_agreement | AF3 paired AUC difference, original (part3) minus interviewer-free, agreement | -0.0318 [-0.0984, 0.0395] | 322 (175) | yes |
| af3_diff_part3_minus_noinv_all | AF3 paired AUC difference, original (part3) minus interviewer-free, all | -0.0034 [-0.0611, 0.0542] | 468 (228) | yes |
| af3_diff_part3_minus_noinv_conflict | AF3 paired AUC difference, original (part3) minus interviewer-free, conflict | 0.0752 [-0.0240, 0.1752] | 146 (100) | yes |
| af3_noinv_agreement | AF3 zero-shot AUC, Pitt interviewer-free cut, agreement | 0.6140 [0.5427, 0.6798] | 322 (175) | yes |
| af3_noinv_all | AF3 zero-shot AUC, Pitt interviewer-free cut, all | 0.5920 [0.5293, 0.6577] | 468 (228) | yes |
| af3_noinv_conflict | AF3 zero-shot AUC, Pitt interviewer-free cut, conflict | 0.5417 [0.4202, 0.6621] | 146 (100) | yes |
| af3_orig_part10_all | AF3 zero-shot AUC, original 468 windows (part10 file), all (recomputed) | 0.5894 [0.5302, 0.6476] | 468 (228) | yes |
| af3_orig_part3_all | AF3 zero-shot AUC, original 468 windows (part3 file), all (recomputed) | 0.5885 [0.5285, 0.6489] | 468 (228) | yes |
| af2_diff_af2pitt468_minus_noinv_agreement | AF2 paired AUC difference, original (af2pitt468) minus interviewer-free, agreement | 0.1579 [0.1052, 0.2093] | 322 (175) | yes |
| af2_diff_af2pitt468_minus_noinv_all | AF2 paired AUC difference, original (af2pitt468) minus interviewer-free, all | 0.1068 [0.0595, 0.1529] | 468 (228) | yes |
| af2_diff_af2pitt468_minus_noinv_conflict | AF2 paired AUC difference, original (af2pitt468) minus interviewer-free, conflict | -0.0167 [-0.1068, 0.0759] | 146 (100) | yes |
| af2_noinv_agreement | AF2 zero-shot AUC, Pitt interviewer-free cut, agreement | 0.3966 [0.3305, 0.4657] | 322 (175) | yes |
| af2_noinv_all | AF2 zero-shot AUC, Pitt interviewer-free cut, all | 0.4655 [0.4038, 0.5265] | 468 (228) | yes |
| af2_noinv_conflict | AF2 zero-shot AUC, Pitt interviewer-free cut, conflict | 0.6477 [0.5357, 0.7444] | 146 (100) | yes |
| af2_orig_af2pitt468_all | AF2 zero-shot AUC, original 468 windows (af2pitt468 file), all (recomputed) | 0.5724 [0.5066, 0.6424] | 468 (228) | yes |
| af2_diff_orig30_minus_noinv_agreement | AF2 paired AUC difference, original first 30 s minus interviewer-free (same span), agreement | 0.1371 [0.0781, 0.1927] | 322 (175) | yes |
| af2_diff_orig30_minus_noinv_all | AF2 paired AUC difference, original first 30 s minus interviewer-free (same span), all | 0.0800 [0.0341, 0.1301] | 468 (228) | yes |
| af2_diff_orig30_minus_noinv_conflict | AF2 paired AUC difference, original first 30 s minus interviewer-free (same span), conflict | -0.0642 [-0.1466, 0.0201] | 146 (100) | yes |
| af2_diff_origfull_minus_orig30_agreement | AF2 paired AUC difference, original full window (af2_pitt468.csv) minus original first 30 s, agreement | 0.0208 [-0.0179, 0.0613] | 322 (175) | yes |
| af2_diff_origfull_minus_orig30_all | AF2 paired AUC difference, original full window (af2_pitt468.csv) minus original first 30 s, all | 0.0268 [-0.0054, 0.0611] | 468 (228) | yes |
| af2_diff_origfull_minus_orig30_conflict | AF2 paired AUC difference, original full window (af2_pitt468.csv) minus original first 30 s, conflict | 0.0475 [-0.0065, 0.1099] | 146 (100) | yes |
| af2_orig30_agreement | AF2 zero-shot AUC, original windows cut to first 30 s (control), agreement | 0.5337 [0.4574, 0.6116] | 322 (175) | yes |
| af2_orig30_all | AF2 zero-shot AUC, original windows cut to first 30 s (control), all | 0.5456 [0.4765, 0.6158] | 468 (228) | yes |
| af2_orig30_conflict | AF2 zero-shot AUC, original windows cut to first 30 s (control), conflict | 0.5835 [0.4730, 0.6913] | 146 (100) | yes |

### PART 20 POD4d ({'RELEASED ONLY': 16, 'SUPPORT': 1})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| kimi_pitt_noinv_overall | Kimi-Audio zero-shot Pitt interviewer-free (PAR-only) overall AUC | 0.5981 [0.5345, 0.6617] | 468 (228) | yes |
| kimi_pitt_noinv_conflict | Kimi-Audio zero-shot Pitt interviewer-free (PAR-only) conflict AUC [interval: arm-only speaker pool] | 0.6972 [0.5905, 0.7946] | 146 (100) | yes |
| kimi_pitt_noinv_agreement | Kimi-Audio zero-shot Pitt interviewer-free (PAR-only) agreement AUC [interval: arm-only speaker pool] | 0.5612 [0.4860, 0.6364] | 322 (175) | yes |
| kimi_pitt_orig_canonical_overall | Kimi-Audio zero-shot Pitt original windows overall AUC (canonical paper file, recomputed) | 0.7180 [0.6585, 0.7742] | 468 (228) | yes |
| kimi_pitt_orig_minus_noinv_overall | Kimi-Audio zero-shot Pitt overall: original (canonical file) minus interviewer-free AUC (paired) | 0.1199 [0.0718, 0.1665] | 468 (228) | yes |
| kimi_pitt_orig_minus_noinv_conflict | Kimi-Audio zero-shot Pitt conflict: original (canonical file) minus interviewer-free AUC (paired) [interval: a | 0.0224 [-0.0495, 0.0996] | 146 (100) | yes |
| kimi_pitt_orig_minus_noinv_agreement | Kimi-Audio zero-shot Pitt agreement: original (canonical file) minus interviewer-free AUC (paired) [interval:  | 0.1615 [0.1068, 0.2168] | 322 (175) | yes |
| kimi_pitt_orig_podrerun_overall | Kimi-Audio zero-shot Pitt original windows overall AUC (this pod's rerun, same env as the cut) | 0.7151 [0.6553, 0.7730] | 468 (228) | yes |
| kimi_pitt_orig_podrerun_conflict | Kimi-Audio zero-shot Pitt original windows conflict AUC (this pod's rerun, same env as the cut) [interval: arm | 0.7215 [0.6210, 0.8153] | 146 (100) | yes |
| kimi_pitt_orig_podrerun_agreement | Kimi-Audio zero-shot Pitt original windows agreement AUC (this pod's rerun, same env as the cut) [interval: ar | 0.7192 [0.6528, 0.7867] | 322 (175) | yes |
| kimi_pitt_podorig_minus_noinv_overall | Kimi-Audio zero-shot Pitt overall: original (pod rerun) minus interviewer-free AUC (paired, same env) | 0.1170 [0.0692, 0.1650] | 468 (228) | yes |
| kimi_pitt_podorig_minus_noinv_conflict | Kimi-Audio zero-shot Pitt conflict: original (pod rerun) minus interviewer-free AUC (paired, same env) [interv | 0.0242 [-0.0519, 0.1016] | 146 (100) | yes |
| kimi_pitt_podorig_minus_noinv_agreement | Kimi-Audio zero-shot Pitt agreement: original (pod rerun) minus interviewer-free AUC (paired, same env) [inter | 0.1580 [0.1027, 0.2136] | 322 (175) | yes |
| kimi_pitt_canon_minus_podorig_overall | Kimi-Audio zero-shot Pitt overall: canonical file minus this pod's rerun of the same original windows (paired, | 0.0029 [-0.0049, 0.0108] | 468 (228) | yes |
| kimi_pitt_canon_minus_podorig_conflict | Kimi-Audio zero-shot Pitt conflict: canonical file minus this pod's rerun of the same original windows (paired | -0.0018 [-0.0215, 0.0160] | 146 (100) | yes |
| kimi_pitt_canon_minus_podorig_agreement | Kimi-Audio zero-shot Pitt agreement: canonical file minus this pod's rerun of the same original windows (paire | 0.0035 [-0.0053, 0.0126] | 322 (175) | yes |
| kimi_val10_vs_canonical | Validation gate FAIL: max /pod p_yes (reference token set) - canonical p_yes/ over the first 10 original windo | 0.0768 | 10 (6) | yes |

### PART 20 POD5 ({'RELEASED ONLY': 69, 'SUPPORT': 1})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| POD5_zs_o25_overall | Qwen2.5-Omni zero-shot p_yes (by-product of extraction; POD4 reports it officially), interviewer-free, overall | 0.7231 [0.6663, 0.7750] | 468 (228) | yes |
| POD5_zs_o25_diff_overall | Qwen2.5-Omni zero-shot p_yes (by-product of extraction; POD4 reports it officially), paired paper minus interv | -0.0654 [-0.1094, -0.0232] | 468 (228) | yes |
| POD5_zs_o25_conflict | Qwen2.5-Omni zero-shot p_yes (by-product of extraction; POD4 reports it officially), interviewer-free, conflic | 0.5426 [0.4273, 0.6661] | 146 (100) | yes |
| POD5_zs_o25_diff_conflict | Qwen2.5-Omni zero-shot p_yes (by-product of extraction; POD4 reports it officially), paired paper minus interv | -0.0103 [-0.0928, 0.0666] | 146 (100) | yes |
| POD5_zs_o25_agreement | Qwen2.5-Omni zero-shot p_yes (by-product of extraction; POD4 reports it officially), interviewer-free, agreeme | 0.7999 [0.7496, 0.8489] | 322 (175) | yes |
| POD5_zs_o25_diff_agreement | Qwen2.5-Omni zero-shot p_yes (by-product of extraction; POD4 reports it officially), paired paper minus interv | -0.0945 [-0.1501, -0.0395] | 322 (175) | yes |
| POD5_enc_mac_seed_overall | Qwen2.5-Omni nested probe, encoder, 5 renumbered Mac-tie splits pitt_groupkfold5_mac_seed0..4 (paper 0.7706),  | 0.7643 [0.7135, 0.8116] | 468 (228) | yes |
| POD5_enc_mac_seed_diff_overall | Qwen2.5-Omni nested probe, encoder, 5 renumbered Mac-tie splits pitt_groupkfold5_mac_seed0..4 (paper 0.7706),  | 0.0063 [-0.0232, 0.0375] | 468 (228) | yes |
| POD5_enc_mac_seed_conflict | Qwen2.5-Omni nested probe, encoder, 5 renumbered Mac-tie splits pitt_groupkfold5_mac_seed0..4 (paper 0.7706),  | 0.6071 [0.5070, 0.7063] | 146 (100) | yes |
| POD5_enc_mac_seed_paper_conflict | Qwen2.5-Omni nested probe, encoder, 5 renumbered Mac-tie splits pitt_groupkfold5_mac_seed0..4 (paper 0.7706),  | 0.5921 [0.4906, 0.7024] | 146 (100) | yes |
| POD5_enc_mac_seed_diff_conflict | Qwen2.5-Omni nested probe, encoder, 5 renumbered Mac-tie splits pitt_groupkfold5_mac_seed0..4 (paper 0.7706),  | -0.0150 [-0.0804, 0.0529] | 146 (100) | yes |
| POD5_enc_mac_seed_agreement | Qwen2.5-Omni nested probe, encoder, 5 renumbered Mac-tie splits pitt_groupkfold5_mac_seed0..4 (paper 0.7706),  | 0.8322 [0.7843, 0.8732] | 322 (175) | yes |
| POD5_enc_mac_seed_paper_agreement | Qwen2.5-Omni nested probe, encoder, 5 renumbered Mac-tie splits pitt_groupkfold5_mac_seed0..4 (paper 0.7706),  | 0.8437 [0.7960, 0.8866] | 322 (175) | yes |
| POD5_enc_mac_seed_diff_agreement | Qwen2.5-Omni nested probe, encoder, 5 renumbered Mac-tie splits pitt_groupkfold5_mac_seed0..4 (paper 0.7706),  | 0.0116 [-0.0161, 0.0408] | 322 (175) | yes |
| POD5_enc_pod_seed_overall | Qwen2.5-Omni nested probe, encoder SUPPLEMENT, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper pod json | 0.7568 [0.7082, 0.8067] | 468 (228) | yes |
| POD5_enc_pod_seed_diff_overall | Qwen2.5-Omni nested probe, encoder SUPPLEMENT, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper pod json | 0.0193 [-0.0089, 0.0507] | 468 (228) | yes |
| POD5_enc_pod_seed_conflict | Qwen2.5-Omni nested probe, encoder SUPPLEMENT, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper pod json | 0.6018 [0.5042, 0.7001] | 146 (100) | yes |
| POD5_enc_pod_seed_original_rerun_conflict | Qwen2.5-Omni nested probe, encoder SUPPLEMENT, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper pod json | 0.5904 [0.4839, 0.7001] | 146 (100) | yes |
| POD5_enc_pod_seed_diff_conflict | Qwen2.5-Omni nested probe, encoder SUPPLEMENT, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper pod json | -0.0114 [-0.0766, 0.0486] | 146 (100) | yes |
| POD5_enc_pod_seed_agreement | Qwen2.5-Omni nested probe, encoder SUPPLEMENT, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper pod json | 0.8234 [0.7746, 0.8659] | 322 (175) | yes |
| POD5_enc_pod_seed_original_rerun_agreement | Qwen2.5-Omni nested probe, encoder SUPPLEMENT, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper pod json | 0.8511 [0.8050, 0.8919] | 322 (175) | yes |
| POD5_enc_pod_seed_diff_agreement | Qwen2.5-Omni nested probe, encoder SUPPLEMENT, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper pod json | 0.0276 [-0.0020, 0.0578] | 322 (175) | yes |
| POD5_llm_pod_seed_overall | Qwen2.5-Omni nested probe, LM stages at audio positions, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (pape | 0.8020 [0.7592, 0.8455] | 468 (228) | yes |
| POD5_llm_pod_seed_diff_overall | Qwen2.5-Omni nested probe, LM stages at audio positions, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (pape | 0.0012 [-0.0265, 0.0307] | 468 (228) | yes |
| POD5_llm_pod_seed_conflict | Qwen2.5-Omni nested probe, LM stages at audio positions, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (pape | 0.5519 [0.4555, 0.6481] | 146 (100) | yes |
| POD5_llm_pod_seed_original_rerun_conflict | Qwen2.5-Omni nested probe, LM stages at audio positions, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (pape | 0.5450 [0.4481, 0.6519] | 146 (100) | yes |
| POD5_llm_pod_seed_diff_conflict | Qwen2.5-Omni nested probe, LM stages at audio positions, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (pape | -0.0070 [-0.0733, 0.0606] | 146 (100) | yes |
| POD5_llm_pod_seed_agreement | Qwen2.5-Omni nested probe, LM stages at audio positions, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (pape | 0.8922 [0.8578, 0.9239] | 322 (175) | yes |
| POD5_llm_pod_seed_original_rerun_agreement | Qwen2.5-Omni nested probe, LM stages at audio positions, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (pape | 0.8935 [0.8573, 0.9260] | 322 (175) | yes |
| POD5_llm_pod_seed_diff_agreement | Qwen2.5-Omni nested probe, LM stages at audio positions, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (pape | 0.0013 [-0.0214, 0.0246] | 322 (175) | yes |
| POD5_ans_pod_seed_overall | Qwen2.5-Omni nested probe, answer state, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper 0.7636), mean  | 0.7777 [0.7332, 0.8188] | 468 (228) | yes |
| POD5_ans_pod_seed_original_rerun_overall | Qwen2.5-Omni nested probe, answer state, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper 0.7636), mean  | 0.7636 [0.7169, 0.8117] | 468 (228) | yes |
| POD5_ans_pod_seed_diff_overall | Qwen2.5-Omni nested probe, answer state, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper 0.7636), mean  | -0.0142 [-0.0420, 0.0163] | 468 (228) | yes |
| POD5_ans_pod_seed_conflict | Qwen2.5-Omni nested probe, answer state, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper 0.7636), mean  | 0.5628 [0.4724, 0.6611] | 146 (100) | yes |
| POD5_ans_pod_seed_original_rerun_conflict | Qwen2.5-Omni nested probe, answer state, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper 0.7636), mean  | 0.5274 [0.4343, 0.6299] | 146 (100) | yes |
| POD5_ans_pod_seed_diff_conflict | Qwen2.5-Omni nested probe, answer state, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper 0.7636), mean  | -0.0353 [-0.0902, 0.0229] | 146 (100) | yes |
| POD5_ans_pod_seed_agreement | Qwen2.5-Omni nested probe, answer state, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper 0.7636), mean  | 0.8601 [0.8242, 0.8942] | 322 (175) | yes |
| POD5_ans_pod_seed_original_rerun_agreement | Qwen2.5-Omni nested probe, answer state, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper 0.7636), mean  | 0.8580 [0.8186, 0.8947] | 322 (175) | yes |
| POD5_ans_pod_seed_diff_agreement | Qwen2.5-Omni nested probe, answer state, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper 0.7636), mean  | -0.0022 [-0.0313, 0.0251] | 322 (175) | yes |
| POD5_proj_pod_seed0_overall | Qwen2.5-Omni nested probe, projector SUPPLEMENT (1 point, seed-0 split as nested_repeats_all.py; paper json 0. | 0.7478 [0.6928, 0.8015] | 468 (228) | yes |
| POD5_proj_pod_seed0_original_rerun_overall | Qwen2.5-Omni nested probe, projector SUPPLEMENT (1 point, seed-0 split as nested_repeats_all.py; paper json 0. | 0.7507 [0.6972, 0.8047] | 468 (228) | yes |
| POD5_proj_pod_seed0_diff_overall | Qwen2.5-Omni nested probe, projector SUPPLEMENT (1 point, seed-0 split as nested_repeats_all.py; paper json 0. | 0.0030 [-0.0418, 0.0489] | 468 (228) | yes |
| POD5_proj_pod_seed0_conflict | Qwen2.5-Omni nested probe, projector SUPPLEMENT (1 point, seed-0 split as nested_repeats_all.py; paper json 0. | 0.6153 [0.5053, 0.7204] | 146 (100) | yes |
| POD5_proj_pod_seed0_original_rerun_conflict | Qwen2.5-Omni nested probe, projector SUPPLEMENT (1 point, seed-0 split as nested_repeats_all.py; paper json 0. | 0.5688 [0.4483, 0.6865] | 146 (100) | yes |
| POD5_proj_pod_seed0_diff_conflict | Qwen2.5-Omni nested probe, projector SUPPLEMENT (1 point, seed-0 split as nested_repeats_all.py; paper json 0. | -0.0465 [-0.1674, 0.0746] | 146 (100) | yes |
| POD5_proj_pod_seed0_agreement | Qwen2.5-Omni nested probe, projector SUPPLEMENT (1 point, seed-0 split as nested_repeats_all.py; paper json 0. | 0.8046 [0.7485, 0.8545] | 322 (175) | yes |
| POD5_proj_pod_seed0_original_rerun_agreement | Qwen2.5-Omni nested probe, projector SUPPLEMENT (1 point, seed-0 split as nested_repeats_all.py; paper json 0. | 0.8237 [0.7692, 0.8741] | 322 (175) | yes |
| POD5_proj_pod_seed0_diff_agreement | Qwen2.5-Omni nested probe, projector SUPPLEMENT (1 point, seed-0 split as nested_repeats_all.py; paper json 0. | 0.0191 [-0.0201, 0.0604] | 322 (175) | yes |
| POD5_dir_probe | answer-state final-stage probe AUC, Mac split, interviewer-free (paper 0.7709) | 0.7958 [0.7483, 0.8408] | 468 (228) | yes |
| POD5_dir_after_removal | probe AUC after removing the readout direction, held-out-fold standardised, interviewer-free (paper 0.7662) | 0.7769 [0.7297, 0.8227] | 468 (228) | yes |
| POD5_dir_kept_control | matched control: same pooling with d kept, interviewer-free | 0.7788 [0.7320, 0.8244] | 468 (228) | yes |
| POD5_dir_probe_minus_removed | probe AUC minus after-removal AUC, paired, interviewer-free | 0.0189 [0.0035, 0.0341] | 468 (228) | yes |
| POD5_dir_probe_conflict | probe AUC, answer state, interviewer-free, conflict arm (full fit restricted to the arm) | 0.5978 [0.4867, 0.7050] | 146 (100) | yes |
| POD5_dir_after_removal_conflict | after_removal AUC, answer state, interviewer-free, conflict arm (full fit restricted to the arm) | 0.5729 [0.4684, 0.6794] | 146 (100) | yes |
| POD5_dir_probe_agreement | probe AUC, answer state, interviewer-free, agreement arm (full fit restricted to the arm) | 0.8719 [0.8287, 0.9121] | 322 (175) | yes |
| POD5_dir_after_removal_agreement | after_removal AUC, answer state, interviewer-free, agreement arm (full fit restricted to the arm) | 0.8545 [0.8084, 0.8977] | 322 (175) | yes |
| POD5_dir_cos | cos(probe direction, mass-weighted Yes-minus-No unembedding), answer state, interviewer-free (paper -0.0109) | 0.0067 [-0.0122, 0.0212] | 468 (228) | yes |
| POD5_dir_floor | random floor, mean /cos/ of 2000 Gaussian directions (fresh default_rng(0)) with the mass-weighted readout dir | 0.0133 [0.0005, 0.0372] | 468 (228) | yes |
| POD5_dir_auc_d | AUC of the projection on the readout direction (gate: vs zero-shot 0.7231), interviewer-free | 0.7282 [0.6728, 0.7808] | 468 (228) | yes |
| POD5_dir_kept_minus_removed | matched control minus after-removal (same held-out-fold pooling), paired, interviewer-free | 0.0020 [-0.0002, 0.0041] | 468 (228) | yes |
| POD5_dir_orig_kept_minus_removed | matched control minus after-removal, paired, ORIGINAL states rerun | -0.0011 [-0.0031, 0.0010] | 468 (228) | yes |
| POD5_dir_probe_diff | probe AUC, paired original(rerun) minus interviewer-free | -0.0248 [-0.0605, 0.0127] | 468 (228) | yes |
| POD5_dir_after_removal_diff | after-removal AUC, paired original(rerun) minus interviewer-free | -0.0108 [-0.0473, 0.0243] | 468 (228) | yes |
| POD5_sft_overall | Qwen2.5-Omni projector fine-tune (sft_projector.py recipe), out of fold, pitt_groupkfold5_pod.csv, interviewer | 0.7912 [0.7404, 0.8423] | 468 (228) | yes |
| POD5_sft_diff_overall | Qwen2.5-Omni projector fine-tune (sft_projector.py recipe), out of fold, pitt_groupkfold5_pod.csv, paired pape | -0.0239 [-0.0744, 0.0270] | 468 (228) | yes |
| POD5_sft_conflict | Qwen2.5-Omni projector fine-tune (sft_projector.py recipe), out of fold, pitt_groupkfold5_pod.csv, interviewer | 0.5639 [0.4609, 0.6685] | 146 (100) | yes |
| POD5_sft_diff_conflict | Qwen2.5-Omni projector fine-tune (sft_projector.py recipe), out of fold, pitt_groupkfold5_pod.csv, paired pape | -0.0450 [-0.1345, 0.0550] | 146 (100) | yes |
| POD5_sft_agreement | Qwen2.5-Omni projector fine-tune (sft_projector.py recipe), out of fold, pitt_groupkfold5_pod.csv, interviewer | 0.8813 [0.8363, 0.9193] | 322 (175) | yes |
| POD5_sft_diff_agreement | Qwen2.5-Omni projector fine-tune (sft_projector.py recipe), out of fold, pitt_groupkfold5_pod.csv, paired pape | -0.0219 [-0.0743, 0.0307] | 322 (175) | yes |
| POD5_sft_rulepyes_overall | projector fine-tune, interviewer-free, overall, part20-rule p_yes (12 single-token variants, full softmax) ins | 0.7882 [0.7357, 0.8393] | 468 (228) | yes |

### PART 20 POD6 ({'RELEASED ONLY': 31, 'SUPERSEDED / NOT USED': 15, 'SUPPORT': 14})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| A_tfidf_pcgita | TF-IDF transcript LR, out-of-fold AUC, PC-GITA (asr_pcgita.csv, _mac folds) | 0.6196 [0.5700, 0.6681] | 1100 (100) | yes |
| A_tfidf_neurovoz | TF-IDF transcript LR, out-of-fold AUC, NeuroVoz (asr_neurovoz.csv, _mac folds) | 0.5814 [0.5422, 0.6206] | 1270 (107) | yes |
| A_tfidf_kcl | TF-IDF transcript LR, out-of-fold AUC, MDVR-KCL (asr_kcl.csv, _mac folds) | 0.4315 [0.2560, 0.6287] | 37 (37) | yes |
| A_tfidf_edaic30 | TF-IDF transcript LR, out-of-fold AUC, E-DAIC first 30 s (edaic30_text.csv, _mac folds) | 0.4844 [0.4028, 0.5624] | 275 (275) | yes |
| A_tfidf_edaicfull | TF-IDF transcript LR, out-of-fold AUC, E-DAIC full interview (edaicfull_text.csv, _mac folds) | 0.6778 [0.5973, 0.7549] | 275 (275) | yes |
| A_tfidf_pitt | TF-IDF transcript LR, out-of-fold AUC, Pitt 468 (pitt_text.csv, _mac folds) | 0.8353 [0.7865, 0.8862] | 468 (228) | yes |
| A_tfidf_adresso | TF-IDF transcript LR, out-of-fold AUC, ADReSSo (adresso_text.csv, _mac folds) | 0.8471 [0.7961, 0.8922] | 237 (237) | yes |
| A_tfidf_adress2020 | TF-IDF transcript LR, out-of-fold AUC, ADReSS-2020 (adress2020_text.csv, _mac folds) | 0.9254 [0.8866, 0.9587] | 156 (156) | yes |
| C_M8_AudioFlamingo2_af2_pitt_audio_conflict | Pitt Audio Flamingo 2 (af2_pitt_audio) zero-shot answer AUC, conflict | 0.6691 [0.5650, 0.7654] | 146 (100) | yes |
| C_M8_AudioFlamingo2_af2_pitt_audio_agreement | Pitt Audio Flamingo 2 (af2_pitt_audio) zero-shot answer AUC, agreement | 0.4895 [0.4182, 0.5591] | 322 (175) | yes |
| C_M8_AudioFlamingo2_af2_pitt_audio_conflict_minus_agreement | Pitt Audio Flamingo 2 (af2_pitt_audio) zero-shot answer AUC, conflict minus agreement | 0.1796 [0.0702, 0.2825] | 468 (228) | yes |
| C_M8_Kimi-Audio_alt_overnight_copy_conflict | Pitt Kimi-Audio (alt_overnight_copy) zero-shot answer AUC, conflict | 0.7078 [0.6037, 0.8083] | 146 (100) | yes |
| C_M8_Kimi-Audio_alt_overnight_copy_agreement | Pitt Kimi-Audio (alt_overnight_copy) zero-shot answer AUC, agreement | 0.7014 [0.6309, 0.7730] | 322 (175) | yes |
| C_M8_Kimi-Audio_alt_overnight_copy_conflict_minus_agreement | Pitt Kimi-Audio (alt_overnight_copy) zero-shot answer AUC, conflict minus agreement | 0.0064 [-0.1140, 0.1224] | 468 (228) | yes |
| C_T6_s15_text_all468 | Pitt Qwen2.5-Omni transcript-only answer AUC, all 468 (Sep 15 run) | 0.6975 [0.6369, 0.7549] | 468 (228) | yes |
| C_T6_s15_text_conflict | Pitt Qwen2.5-Omni transcript-only answer AUC, conflict arm (Sep 15 run) | 0.4550 [0.3412, 0.5718] | 146 (100) | yes |
| C_T6_s15_text_agreement | Pitt Qwen2.5-Omni transcript-only answer AUC, agreement arm (Sep 15 run) | 0.7987 [0.7392, 0.8535] | 322 (175) | yes |
| C_T6_s15_text_conflict_minus_agreement | Pitt Qwen2.5-Omni transcript-only PAIRED conflict minus agreement AUC (Sep 15 run) | -0.3437 [-0.4556, -0.2209] | 468 (228) | yes |
| C_T6_s20_text_minus_audio_all468 | PAIRED transcript (Sep 20 run) minus audio AUC, all 468 | 0.0051 [-0.0657, 0.0694] | 468 (228) | yes |
| C_T6_s20_text_minus_audio_conflict | PAIRED transcript (Sep 20 run) minus audio AUC, conflict arm | 0.0238 [-0.1147, 0.1577] | 146 (100) | yes |
| C_T6_s20_text_minus_audio_agreement | PAIRED transcript (Sep 20 run) minus audio AUC, agreement arm | 0.0024 [-0.0824, 0.0779] | 322 (175) | yes |
| C_T6_s20_gap_text_minus_gap_audio | PAIRED (transcript Sep 20 run arm gap) minus (audio arm gap), gap = conflict minus agreement | 0.0214 [-0.1413, 0.1806] | 468 (228) | yes |
| C_T6_s15_text_minus_audio_all468 | PAIRED transcript (Sep 15 run) minus audio AUC, all 468 | 0.0397 [-0.0302, 0.1070] | 468 (228) | yes |
| C_T6_s15_text_minus_audio_conflict | PAIRED transcript (Sep 15 run) minus audio AUC, conflict arm | -0.0772 [-0.2360, 0.0680] | 146 (100) | yes |
| C_T6_s15_text_minus_audio_agreement | PAIRED transcript (Sep 15 run) minus audio AUC, agreement arm | 0.0933 [0.0157, 0.1637] | 322 (175) | yes |
| C_T6_s15_gap_text_minus_gap_audio | PAIRED (transcript Sep 15 run arm gap) minus (audio arm gap), gap = conflict minus agreement | -0.1705 [-0.3440, -0.0049] | 468 (228) | yes |
| C_T6_s20_minus_s15_all468 | PAIRED Sep 20 minus Sep 15 transcript AUC, all 468 | -0.0346 [-0.0757, 0.0003] | 468 (228) | yes |
| C_T6_s20_minus_s15_conflict | PAIRED Sep 20 minus Sep 15 transcript AUC, conflict arm | 0.1010 [0.0270, 0.1766] | 146 (100) | yes |
| C_T6_s20_minus_s15_agreement | PAIRED Sep 20 minus Sep 15 transcript AUC, agreement arm | -0.0909 [-0.1345, -0.0518] | 322 (175) | yes |
| C_T6_s20_minus_s15_armgap | PAIRED Sep 20 arm gap minus Sep 15 arm gap (transcript) | 0.1919 [0.1104, 0.2800] | 468 (228) | yes |
| C_T6_perclip_pearson_s20_s15 | Per-clip p_yes Pearson r, Sep 20 vs Sep 15 transcript run | 0.7640 | 468 (228) | yes |
| C_T6_perclip_spearman_s20_s15 | Per-clip p_yes Spearman rho, Sep 20 vs Sep 15 transcript run | 0.7876 | 468 (228) | yes |
| C_T6_perclip_maxabsdiff_s20_s15 | Per-clip p_yes max abs diff, Sep 20 vs Sep 15 transcript run | 0.6147 | 468 (228) | yes |
| C_T6_perclip_meanabsdiff_s20_s15 | Per-clip p_yes mean abs diff, Sep 20 vs Sep 15 transcript run | 0.3019 | 468 (228) | yes |
| C_T6_armgap_ratio_s20 | Point ratio transcript arm gap / audio arm gap, Sep 20 run | 0.8765 | 468 (228) | yes |
| C_T6_armgap_ratio_s15 | Point ratio transcript arm gap / audio arm gap, Sep 15 run | 1.9848 | 468 (228) | yes |
| A_tfidf_pitt_rawCHAT | TF-IDF transcript LR, out-of-fold AUC, Pitt 468 on raw CHAT manifest text (sensitivity; not the transcript-con | 0.8452 [0.7975, 0.8918] | 468 (228) | yes |
| B_zeroshot_nvmatched | Qwen2.5-Omni zero-shot answer AUC, NeuroVoz age/sex matched subset (38 pairs; paper per-clip scores restricted | 0.6282 [0.5584, 0.6985] | 907 (76) | yes |
| B_chan_noise_floor_raw_nvmatched | quiet-frame noise_floor alone, raw AUC (PD positive), NeuroVoz matched subset | 0.4640 [0.3873, 0.5423] | 907 (76) | yes |
| B_chan_noise_floor_dirfree_nvmatched | quiet-frame noise_floor alone, direction-free max(AUC,1-AUC), NeuroVoz matched subset | 0.5360 [0.4577, 0.6127] | 907 (76) | yes |
| B_chan_snr_raw_nvmatched | quiet-frame snr alone, raw AUC (PD positive), NeuroVoz matched subset | 0.4294 [0.3413, 0.5181] | 907 (76) | yes |
| B_chan_snr_dirfree_nvmatched | quiet-frame snr alone, direction-free max(AUC,1-AUC), NeuroVoz matched subset | 0.5706 [0.4819, 0.6587] | 907 (76) | yes |
| B_chan_centroid_raw_nvmatched | quiet-frame centroid alone, raw AUC (PD positive), NeuroVoz matched subset | 0.2739 [0.1946, 0.3574] | 907 (76) | yes |
| B_chan_centroid_dirfree_nvmatched | quiet-frame centroid alone, direction-free max(AUC,1-AUC), NeuroVoz matched subset | 0.7261 [0.6426, 0.8054] | 907 (76) | yes |
| B_chan_rolloff_raw_nvmatched | quiet-frame rolloff alone, raw AUC (PD positive), NeuroVoz matched subset | 0.2550 [0.1819, 0.3382] | 907 (76) | yes |
| B_chan_rolloff_dirfree_nvmatched | quiet-frame rolloff alone, direction-free max(AUC,1-AUC), NeuroVoz matched subset | 0.7450 [0.6618, 0.8181] | 907 (76) | yes |
| B_chan_tilt_raw_nvmatched | quiet-frame tilt alone, raw AUC (PD positive), NeuroVoz matched subset | 0.3363 [0.2716, 0.4039] | 907 (76) | yes |
| B_chan_tilt_dirfree_nvmatched | quiet-frame tilt alone, direction-free max(AUC,1-AUC), NeuroVoz matched subset | 0.6637 [0.5961, 0.7284] | 907 (76) | yes |
| B_chan_five_nvmatched | five quiet-frame channel features together, LR out-of-fold AUC, NeuroVoz matched subset (NEW single split, sta | 0.7112 [0.6079, 0.7961] | 907 (76) | yes |
| B_encprobe_mean5_nvmatched | Qwen2.5-Omni encoder nested probe, mean of five renumbered splits (pod tie rule), NeuroVoz matched subset | 0.8986 [0.8430, 0.9428] | 907 (76) | yes |
| B_encprobe_seed0_nvmatched | Qwen2.5-Omni encoder nested probe, renumbered split seed 0, NeuroVoz matched subset (compute gave the point va | 0.8982 [0.8381, 0.9449] | 907 (76) | yes |
| B_encprobe_seed1_nvmatched | Qwen2.5-Omni encoder nested probe, renumbered split seed 1, NeuroVoz matched subset (compute gave the point va | 0.9093 [0.8582, 0.9518] | 907 (76) | yes |
| B_encprobe_seed2_nvmatched | Qwen2.5-Omni encoder nested probe, renumbered split seed 2, NeuroVoz matched subset (compute gave the point va | 0.8951 [0.8308, 0.9457] | 907 (76) | yes |
| B_encprobe_seed3_nvmatched | Qwen2.5-Omni encoder nested probe, renumbered split seed 3, NeuroVoz matched subset (compute gave the point va | 0.9174 [0.8666, 0.9563] | 907 (76) | yes |
| B_encprobe_seed4_nvmatched | Qwen2.5-Omni encoder nested probe, renumbered split seed 4, NeuroVoz matched subset (compute gave the point va | 0.8730 [0.8056, 0.9280] | 907 (76) | yes |
| B_encprobe_single_nvmatched | Qwen2.5-Omni encoder nested probe, single split (stable GroupKFold on speaker labels), NeuroVoz matched subset | 0.8951 [0.8359, 0.9427] | 907 (76) | yes |
| B_zeroshot_nv_complement363 | Qwen2.5-Omni zero-shot answer AUC, the 363 NeuroVoz clips outside the matched subset (paper per-clip file) | 0.8462 [0.7645, 0.9144] | 363 (31) | yes |
| B_encprobe_mean5_nvfull_samepod | Qwen2.5-Omni encoder nested probe, mean of five, FULL NeuroVoz 1270 re-extracted on the same pod/code (check a | 0.9218 [0.8841, 0.9525] | 1270 (107) | yes |
| B_encprobe_single_nvfull_samepod | Qwen2.5-Omni encoder nested probe, single pod split, FULL NeuroVoz 1270 same pod (check against paper 0.9160) | 0.9166 [0.8722, 0.9516] | 1270 (107) | yes |
| B_pairs_nvmatched | NeuroVoz age/sex matched PD-HC pairs (same sex, /age diff/<=5 y, greedy: fewest eligible partners first, ties  | 38 | 907 (76) | yes |

### PART 22 P22 ({'SUPPORT': 72})

| id | what | value [95% CI] | n (speakers) | checked |
|---|---|---|---|---|
| P22_s2_305_tok_full | pid 305 (900.0 s window) audio tokens, paper processor call, full window | 7500 | 1 (1) | yes |
| P22_s2_305_tok_cut300 | pid 305 audio tokens, paper processor call, window cut to first 300 s | 7500 | 1 (1) | yes |
| P22_s2_305_pyes_full | pid 305 p_yes (paper 5+5 id set), full window | 0.5776 | 1 (1) | yes |
| P22_s2_305_pyes_cut300 | pid 305 p_yes (paper 5+5 id set), first 300 s | 0.5776 | 1 (1) | yes |
| P22_s2_305_inputs_identical | pid 305 full vs 300 s cut: input_ids, feature mask and input_features tensors byte-identical | True | 1 (1) | yes |
| P22_s2_679_tok_full | pid 679 (900.0 s window) audio tokens, paper processor call, full window | 7500 | 1 (1) | yes |
| P22_s2_679_tok_cut300 | pid 679 audio tokens, paper processor call, window cut to first 300 s | 7500 | 1 (1) | yes |
| P22_s2_679_pyes_full | pid 679 p_yes (paper 5+5 id set), full window | 0.2339 | 1 (1) | yes |
| P22_s2_679_pyes_cut300 | pid 679 p_yes (paper 5+5 id set), first 300 s | 0.2339 | 1 (1) | yes |
| P22_s2_679_inputs_identical | pid 679 full vs 300 s cut: input_ids, feature mask and input_features tensors byte-identical | True | 1 (1) | yes |
| P22_s2_466_tok_full | pid 466 (900.0 s window) audio tokens, paper processor call, full window | 7500 | 1 (1) | yes |
| P22_s2_466_tok_cut300 | pid 466 audio tokens, paper processor call, window cut to first 300 s | 7500 | 1 (1) | yes |
| P22_s2_466_pyes_full | pid 466 p_yes (paper 5+5 id set), full window | 0.7063 | 1 (1) | yes |
| P22_s2_466_pyes_cut300 | pid 466 p_yes (paper 5+5 id set), first 300 s | 0.7063 | 1 (1) | yes |
| P22_s2_466_inputs_identical | pid 466 full vs 300 s cut: input_ids, feature mask and input_features tensors byte-identical | True | 1 (1) | yes |
| P22_s2_705_tok_full | pid 705 (198.3 s window) audio tokens, paper processor call, full window | 4957 | 1 (1) | yes |
| P22_s2_705_tok_cut300 | pid 705 audio tokens, paper processor call, window cut to first 300 s | 4957 | 1 (1) | yes |
| P22_s2_705_pyes_full | pid 705 p_yes (paper 5+5 id set), full window | 0.9604 | 1 (1) | yes |
| P22_s2_705_pyes_cut300 | pid 705 p_yes (paper 5+5 id set), first 300 s | 0.9604 | 1 (1) | yes |
| P22_s2_705_inputs_identical | pid 705 full vs 300 s cut: input_ids, feature mask and input_features tensors byte-identical | True | 1 (1) | yes |
| P22_s2_overall | step 2 verdict on the 3 longest windows (305, 679, 466): tokens identical and p_yes equal to 6 dp, full vs 300 | TRUNCATION PROVEN | 3 (3) | yes |
| P22_s1_quotes | step 1 md: quoted code blocks equal to the source files line for line (29 transformers 5.17.0 + HF snapshot ae | 31/31 OK, 0 mismatch | 31 | yes |
| P22_s1_tf_version | transformers version used by the paper pod and this pod | 5.17.0 | 1 | yes |
| P22_s1_fe_class | Qwen2_5OmniProcessor feature extractor class | WhisperFeatureExtractor | 1 | yes |
| P22_s1_chunk_length | feature extractor chunk_length (preprocessor_config.json:2) | 300 | 1 | yes |
| P22_s1_sampling_rate | feature extractor sampling_rate | 16000 | 1 | yes |
| P22_s1_max_length | effective max_length = n_samples = chunk_length x sampling_rate (feature_extraction_whisper.py:91, :303) | 4800000 | 1 | yes |
| P22_s1_truncation_default | WhisperFeatureExtractor.__call__ truncation default (feature_extraction_whisper.py:196) | True | 1 | yes |
| P22_s1_cut_line | line that cuts the waveform | feature_extraction_sequence_utils.py:333 | 1 | yes |
| P22_s3_probe_tok_lifted | pid 305 900 s window audio tokens, truncation off (audio_kwargs or flat truncation=False, or max_length 900 s) | 22500 | 1 (1) | yes |
| P22_s3_probe_pyes_lifted | pid 305 p_yes, truncation off, model ran on seq_len 22548 | 0.1733 | 1 (1) | yes |
| P22_s3_tokens_all275 | lifted audio tokens per clip, pod job vs the Mac rerun (275/275 equal); 89 capped windows all 22500; 217 windows  | 275/275 equal | 275 (275) | yes |
| P22_s3_short58_unchanged | 58 windows <= 300 s: p_yes truncation off minus default, max abs (inputs identical) | 0 | 58 (58) | yes |
| P22_s3_perclip_lift_pod_vs_mine | lifted p_yes per clip, pod job vs the independent Mac rescore: max abs diff (0 pos-neg order flips) | 0.0000 | 275 (275) | yes |
| P22_all275_lift | E-DAIC zero-shot answer AUC, whole window (truncation off), all 275 E-DAIC windows | 0.8379 [0.7805, 0.8876] | 275 (275) | yes |
| P22_all275_diff | paired AUC difference, whole window minus 300 s cut (one draw per replicate), all 275 E-DAIC windows | 0.0101 [-0.0309, 0.0518] | 275 (275) | yes |
| P22_long217_trunc | E-DAIC zero-shot answer AUC, 300 s cut (paper call; reference per-clip file), 217 windows longer than 300 s | 0.8380 [0.7758, 0.8930] | 217 (217) | yes |
| P22_long217_lift | E-DAIC zero-shot answer AUC, whole window (truncation off), 217 windows longer than 300 s | 0.8571 [0.8028, 0.9062] | 217 (217) | yes |
| P22_long217_diff | paired AUC difference, whole window minus 300 s cut (one draw per replicate), 217 windows longer than 300 s | 0.0192 [-0.0309, 0.0703] | 217 (217) | yes |
| P22_capped89_trunc | E-DAIC zero-shot answer AUC, 300 s cut (paper call; reference per-clip file), 89 windows capped at 900 s | 0.8169 [0.6985, 0.9161] | 89 (89) | yes |
| P22_capped89_lift | E-DAIC zero-shot answer AUC, whole window (truncation off), 89 windows capped at 900 s | 0.8552 [0.7513, 0.9414] | 89 (89) | yes |
| P22_capped89_diff | paired AUC difference, whole window minus 300 s cut (one draw per replicate), 89 windows capped at 900 s | 0.0383 [-0.0424, 0.1360] | 89 (89) | yes |
| P22_S1_tf_version | transformers version of the paper E-DAIC scoring run (podA SETUP3.log:15); 3 live Part 20 pods also 5.17.0 [va | 5.17.0 | 1 | yes |
| P22_S1_chunk_length | Qwen2.5-Omni WhisperFeatureExtractor chunk_length s (Hub preprocessor_config.json:2; loaded object fe.chunk_le | 300 |  | yes |
| P22_S1_n_samples | feature extractor n_samples = audio cap in 16 kHz samples (feature_extraction_whisper.py:91; loaded object) [v | 4800000 |  | yes |
| P22_S1_trunc_default | WhisperFeatureExtractor.__call__ truncation default (feature_extraction_whisper.py:196); Omni processor passes | True |  | yes |
| P22_S1_tok_300s | audio tokens at the 300 s cap: code formula processing_qwen2_5_omni.py:151-152 vs measured 5.17.0 processor [v | 7500 |  | yes |
| P22_S1_tok_900s_off | audio tokens for 900 s with audio_kwargs truncation False: code formula vs measured 5.17.0 processor [value re | 22500 |  | yes |
| P22_S1_formula_match | E-DAIC full windows whose logged n_audio_tok equals the code formula on win_dur_s (of 275) [value restored by  | 275 | 275 (275) | yes |
| P22_S1_gt300_at_7500 | E-DAIC full windows longer than 300 s that received exactly 7500 audio tokens (of 217) [value restored by the  | 217 | 217 (217) | yes |
| P22_auc_lifted_all275 | E-DAIC zero-shot AUC, truncation=False (whole window), all275 | 0.8379 [0.7805, 0.8876] | 275 (275) | yes |
| P22_auc_diff_all275 | paired AUC lifted minus default, all275 | 0.0101 [-0.0309, 0.0518] | 275 (275) | yes |
| P22_auc_lifted_all275_ruleids | as P22_auc_lifted_all275, rule Yes/No id set | 0.8379 [0.7805, 0.8876] | 275 (275) | yes |
| P22_auc_default_long217 | E-DAIC zero-shot AUC, default processor (300 s cut), long217 | 0.8380 [0.7758, 0.8930] | 217 (217) | yes |
| P22_auc_lifted_long217 | E-DAIC zero-shot AUC, truncation=False (whole window), long217 | 0.8571 [0.8028, 0.9062] | 217 (217) | yes |
| P22_auc_diff_long217 | paired AUC lifted minus default, long217 | 0.0192 [-0.0309, 0.0703] | 217 (217) | yes |
| P22_auc_lifted_long217_ruleids | as P22_auc_lifted_long217, rule Yes/No id set | 0.8571 [0.8028, 0.9062] | 217 (217) | yes |
| P22_auc_default_capped89 | E-DAIC zero-shot AUC, default processor (300 s cut), capped89 | 0.8169 [0.6985, 0.9161] | 89 (89) | yes |
| P22_auc_lifted_capped89 | E-DAIC zero-shot AUC, truncation=False (whole window), capped89 | 0.8552 [0.7513, 0.9414] | 89 (89) | yes |
| P22_auc_diff_capped89 | paired AUC lifted minus default, capped89 | 0.0383 [-0.0424, 0.1360] | 89 (89) | yes |
| P22_auc_lifted_capped89_ruleids | as P22_auc_lifted_capped89, rule Yes/No id set | 0.8552 [0.7513, 0.9414] | 89 (89) | yes |
| P22_gate_maxdiff | validation gate: max /p_yes pod - reference/ over 10 windows, unmodified extract_full.py | 0 | 10 (10) | yes |
| P22_short58_maxdiff | max /p_yes lifted - default/ on the 58 windows <= 300 s | 0 | 58 (58) | yes |
| P22_lift_tok_max | max audio tokens with truncation=False (900 s window) | 22500 | 275 (275) | yes |
| P22_default_tok_max | max audio tokens with the default processor | 7500 | 275 (275) | yes |
| P22_s3_probe_tok_default | pid 305 900 s window audio tokens, default processor | 7500 | 1 (1) | yes |
| P22_s3_probe_model_ran | model forward ran on the 22,500-token input | True | 1 (1) | yes |
| P22_s3_probe_peak_gib | pid 305 peak GPU memory GiB, truncation=False | not measured by verifier | 1 (1) | MISSING verified value (file flag: n/a) |
| P22_s2_305_verdict | pid 305 verdict | TRUNCATION PROVEN | 1 (1) | yes |
| P22_s2_679_verdict | pid 679 verdict | TRUNCATION PROVEN | 1 (1) | yes |
| P22_s2_466_verdict | pid 466 verdict | TRUNCATION PROVEN | 1 (1) | yes |
| P22_s2_705_verdict | pid 705 verdict | control: 198.3 s window, full and 300 s cut byte-identical inputs, p_yes equal | 1 (1) | yes |