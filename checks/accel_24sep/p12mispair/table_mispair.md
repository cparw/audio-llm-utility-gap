| row | item | tex | computed | yes/no | how | file |
|---|---|---|---|---|---|---|
| 0 | L67_321 | 0.89 | 0.8968 | yes | min median Yes+No mass over 70 zero-shot cells = 0.8968 (Audio Flamingo 3 pcgita); recomputed from af3_pcgita.csv = 0.89680315554142; E-DAIC whole median 0.9969; tex says 'at least 0.89'; true floor check (>=0.89) | <local data dir>/release/edaic_rerun/part16/M9_answer_mass.csv ; RFM/scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv |
| 1 | L77_360 | 0.43 to 0.62 | 0.4315 to 0.6196 | yes | AUC(label,p_tfidf) KCL 0.4315 NeuroVoz 0.5814 PC-GITA 0.6196 | <local data dir>/release/scores/part20/POD6/A_language_saliency/A_tfidf_*_oof.csv |
| 2 | L77_398 | 0.68 | 0.6778 | yes | AUC [0.5973, 0.7549] n=275 | <local data dir>/release/scores/part20/POD6/A_language_saliency/A_tfidf_edaicfull_oof.csv |
| 3 | L77_433 | 0.84 to 0.93 | 0.8353 to 0.9254 | yes | Pitt 0.8353 ADReSSo 0.8471 ADReSS-2020 0.9254 | <local data dir>/release/scores/part20/POD6/A_language_saliency/A_tfidf_*_oof.csv |
| 4 | L92_538 | 0.07 | 0.0664 | yes | max /wording - base/ over 14 runs = 0.0664; mean 0.0240 (tex 0.02 on average) | <local data dir>/release/omni_final/omni_{ds}_{zeroshot_scores,p2,p3}.csv |
| 5 | L97_803 | 30 to 56 | 30.0 to 55.6 (median 33.3) | yes | dur column, n=468, sets {'agreement': 322, 'conflict': 146} | <local data dir>/DementiaBank/pitt_conflict_manifest.csv |
| 8 | L140_549 | 0.72 | 0.7205 | yes | AUC [0.6531, 0.7917] n=275 | <local data dir>/release/edaic_rerun/variants/o25_mid30_zeroshot_scores.csv |
| 9 | L149_1096 | 0.67 | 0.6657 | yes | five-feature OOF AUC, PC-GITA, before | <local data dir>/release/edaic_rerun/part16/POD4/channel_norm_pcgita.csv |
| 10 | L149_1224 | 0.56 | 0.5590 | yes | five-feature OOF AUC, PC-GITA, after | <local data dir>/release/edaic_rerun/part16/POD4/channel_norm_pcgita.csv |
| 11 | L149_1283 | 0.07 | -0.0661 | yes | enc probe after minus before 0.8298-0.8959 = -0.0661 [-0.1181, -0.0180] paired spk bootstrap; tex prints -0.07 | <local data dir>/release/edaic_rerun/part16/POD4/channel_norm_pcgita.csv |
| 12 | L149_1292 | 0.12 | -0.1181 | yes | lower bound of row 11 interval; tex prints -0.12 | <local data dir>/release/edaic_rerun/part16/POD4/channel_norm_pcgita.csv |
| 13 | L149_1301 | 0.02 | -0.0180 | yes | upper bound of row 11 interval; tex prints -0.02 | <local data dir>/release/edaic_rerun/part16/POD4/channel_norm_pcgita.csv |
| 14 | L149_1350 | 0.60 | 0.6025 | yes | five-feature OOF AUC, NeuroVoz, after (before 0.7977, tex 0.80) | <local data dir>/release/edaic_rerun/part16/POD4/channel_norm_neurovoz.csv |
| 15 | L149_1402 | 0.03 | -0.0308 | yes | enc probe after minus before 0.8924-0.9233 = -0.0308 [-0.0608, -0.0012]; tex prints -0.03 | <local data dir>/release/edaic_rerun/part16/POD4/channel_norm_neurovoz.csv |
| 16 | L149_1411 | 0.061 | -0.0608 | yes | lower bound (tex gives 3 dp); checked at 3 dp because tex prints -0.061 | <local data dir>/release/edaic_rerun/part16/POD4/channel_norm_neurovoz.csv |
| 17 | L149_1421 | 0.001 | -0.0012 | yes | upper bound (tex gives 3 dp); checked at 3 dp because tex prints -0.001 | <local data dir>/release/edaic_rerun/part16/POD4/channel_norm_neurovoz.csv |
| 18 | L149_1507 | 0.90 | 0.8986 | yes | mean of 5 seed AUCs 0.8986; AUC of averaged p 0.9147; p_single 0.8951; n=907 spk=76 | <local data dir>/release/scores/part20/POD6/B_neurovoz_matched/B_encprobe_nested5_nvmatched_oof.csv |
| 19 | L149_152 | 0.77 to 0.88 | 0.7706 to 0.8752 | yes | Pitt mean of 5 per-repeat AUCs 0.7706 (recomputed); ADReSSo json mean 0.8752 (per-repeat mean 0.8752); ADReSS-2020 0.8043 (0.8043); ADReSSo and ADReSS-2020 read from the repeats json summary, not recomputed per clip | <local data dir>/release/overnight2/part10/pitt_enc_nested5_oof.npz ; omni_final/omni_{adresso,adress2020}_nested_repeats.json |
| 20 | L149_1527 | 0.63 | 0.6282 | yes | AUC [0.5584, 0.6985] n=907 spk=76 | <local data dir>/release/scores/part20/POD6/B_neurovoz_matched/B_zeroshot_nvmatched.csv |
| 21 | L149_2192 | 0.66 | 0.6608 | yes | auc_d (summary csv) | <local data dir>/release/omni/readout_direction.csv (row pitt) |
| 22 | L149_2262 | 0.77 | 0.7709 | yes | auc_probe (summary csv) | <local data dir>/release/omni/readout_direction.csv (row pitt) |
| 23 | L149_2283 | 0.77 | 0.7662 | yes | auc_probe_without_d (summary csv); cosine -0.0109 | <local data dir>/release/omni/readout_direction.csv (row pitt) |
| 25 | L149_2463 | 0.62 | 0.6174 | yes | AUC(label, proj_d) n=468; probe 0.8223; perp_z 0.8114; nested_oof 0.8295; nested_perp_z 0.8169 | <local data dir>/release/edaic_rerun/part17/T3_q2a_direction.csv |
| 26 | L149_2643 | 0.74 | 0.7437 | yes | LM probe mean of five per-repeat AUCs (AUC of averaged p 0.7504) | RFM/scores/part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv |
| 27 | L149_2670 | 0.60 | 0.5981 | yes | encoder probe mean of five per-repeat AUCs (AUC of averaged p 0.6058) | RFM/scores/part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv |
| 28 | L149_2701 | 0.84 | 0.8366 | yes | AUC p_yes_whole n=275 | RFM/scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv |
| 29 | L149_2710 | 0.09 | -0.0929 | yes | LM minus answer -0.0929 [-0.1609, -0.0257] paired, mean-of-five inside each draw; tex prints -0.09 | RFM/scores/part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv + A2 |
| 30 | L149_2719 | 0.16 | -0.1609 | yes | lower bound; tex prints -0.16 | RFM/scores/part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv + A2 |
| 31 | L149_2728 | 0.03 | -0.0257 | yes | upper bound; tex prints -0.03 | RFM/scores/part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv + A2 |
| 32 | L149_2741 | 0.24 | -0.2385 | yes | encoder minus answer -0.2385 [-0.3244, -0.1505]; tex prints -0.24 | RFM/scores/part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv + A2 |
| 33 | L149_2750 | 0.32 | -0.3244 | yes | lower bound; tex prints -0.32 | RFM/scores/part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv + A2 |
| 34 | L149_2759 | 0.15 | -0.1505 | yes | upper bound; tex prints -0.15 | RFM/scores/part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv + A2 |
| 35 | L149_2821 | 0.84 | 0.8359 | yes | answer-state probe mean of five; minus answer -0.0007 [-0.0455, +0.0436] | RFM/scores/part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv |
| 36 | L149_333 | 0.11 | 0.1128 | yes | mean-of-5 enc 0.7706 minus zero-shot 0.6578 = +0.1128 [+0.0455, +0.1768] n=468 spk=228 | <local data dir>/release/overnight2/part10/pitt_enc_nested5_oof.npz + <local data dir>/release/omni_final/omni_pitt_zeroshot_scores.csv |
| 37 | L149_339 | 0.05, 0.18 | 0.0455, 0.1768 | yes | interval of the Pitt gap | same as row 36 |
| 38 | L149_362 | 0.33 | 0.3306 | yes | enc mean-of-5 0.8941 (per-repeat mean 0.8941) minus zero-shot 0.5635; single-split enc OOF file gives 0.9089 (gap 0.3454); point only; no five-repeat per-clip PC-GITA file for an interval | <local data dir>/release/omni_final/omni_pcgita_nested_repeats.json + omni_pcgita_zeroshot_scores.csv |
| 39 | L149_593 | 0.60 | 0.5967 | yes | AUC n=1100 | <local data dir>/release/overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv |
| 40 | L149_639 | 0.05 | 0.0503 | yes | Q3O Pitt enc mean-of-5 0.8122 minus zero-shot 0.7619 | <local data dir>/release/edaic_rerun/part16/POD3/q3o_pitt_encoder_nested_oof.csv + <local data dir>/release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv |
| 41 | L149_796 | 0.01 on MDVR-KCL to 0.18 | +0.0095 (kcl) to +0.1810 (adresso) | yes | probe minus eGeMAPS per PD/AD set: kcl +0.0095, adress2020 +0.0921, adresso +0.1810, pitt +0.1214, pcgita +0.0609, neurovoz +0.0774; eGeMAPS AUC and probe read from the summary csv | <local data dir>/release/overnight/egemaps_baseline.csv |
| 42 | L149_911 | 0.27 | 0.2697 | yes | eGeMAPS 0.8332 minus recomputed zero-shot 0.5635 | <local data dir>/release/overnight/egemaps_baseline.csv + <local data dir>/release/omni_final/omni_pcgita_zeroshot_scores.csv |
| 43 | L149_920 | 0.15 | 0.1488 | yes | eGeMAPS 0.8439 minus recomputed zero-shot 0.6951 | <local data dir>/release/overnight/egemaps_baseline.csv + <local data dir>/release/omni_final/omni_neurovoz_zeroshot_scores.csv |
| 48 | L180_1000 | 0.95 | 0.9523 | yes | agreement, 311 distinct (flag col; dedupe gives 311): 0.9523 [0.9249, 0.9750]; all 483 rows 0.9470 | <local data dir>/release/scores/part20/POD3c/p14_q3o_text.csv |
| 49 | L180_1662 | 0.46 | 0.4619 | yes | conflict n=1334 spk=162 | <local data dir>/release/scores/part20/POD3b/sst2_pairs_o25_audio.csv |
| 50 | L180_1668 | 0.42, 0.51 | 0.4165, 0.5102 | yes | conflict interval | <local data dir>/release/scores/part20/POD3b/sst2_pairs_o25_audio.csv |
| 51 | L180_1723 | 0.39 | 0.3934 | yes | agreement distinct 799 segs 0.8554 minus conflict 0.4619; all agreement rows 0.8316 (gap 0.3697) | <local data dir>/release/scores/part20/POD3b/sst2_pairs_o25_audio.csv |
| 52 | L180_1824 | 0.20 | 0.1951 | yes | conflict & first scorer conflict, n=420; n check 420: True | <local data dir>/release/scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv |
| 53 | L180_1881 | 0.62 | 0.6150 | yes | conflict & first scorer neutral, n=855; n check 855: True | <local data dir>/release/scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv |
| 54 | L180_2394 | 0.53 to 0.61 | 0.5337 to 0.6075 | yes | AF2 conflict 0.5835; AF2 agreement 0.5337; AF3 conflict 0.6075; AF3 agreement 0.5855 | <local data dir>/release/scores/part20/POD4c/af2_orig30_{conflict,agreement}.csv ; <local data dir>/release/overnight2/part10/af3_pitt.csv |
| 55 | L180_2521 | 0.01 | -0.0104 | yes | voice 0.6473 minus recording 0.6578 = -0.0104 [-0.0356, +0.0155]; tex prints -0.01 | <local data dir>/release/scores/part20/POD4/o25_orig_voice.csv + <local data dir>/release/omni_final/omni_pitt_zeroshot_scores.csv |
| 56 | L180_2530 | 0.04 | -0.0356 | yes | lower bound; tex prints -0.04 | <local data dir>/release/scores/part20/POD4/o25_orig_voice.csv + <local data dir>/release/omni_final/omni_pitt_zeroshot_scores.csv |
| 57 | L180_2536 | 0.02 | 0.0155 | yes | upper bound | <local data dir>/release/scores/part20/POD4/o25_orig_voice.csv + <local data dir>/release/omni_final/omni_pitt_zeroshot_scores.csv |
| 58 | L180_931 | 0.15 | 0.1498 | yes | conflict n=483: [0.1131, 0.1887] | <local data dir>/release/edaic_rerun/part16/POD1/p14_o25_1a_joined.csv |
| 59 | L180_952 | 0.96 | 0.9620 | yes | agreement distinct 311: 0.9620; all 483 rows 0.9596 | <local data dir>/release/edaic_rerun/part16/POD1/p14_o25_1a_joined.csv |
| 60 | L180_991 | 0.22 | 0.2243 | yes | conflict n=483: 0.2243 [0.1757, 0.2742] | <local data dir>/release/scores/part20/POD3c/p14_q3o_text.csv |
| 61 | L226_1084 | 0.77 | 0.7696 | yes | AUC n=468 | <local data dir>/release/omni_final/abl2_pitt_both_oof.csv |
| 62 | L226_1154 | +0.02 | 0.0223 | yes | FT seed0 0.8589 minus zero-shot 0.8366 = +0.0223 [-0.0064, +0.0547] n=275 | RFM/scores/part24/FT/FT24_whole_seed0_perclip.csv + RFM/scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv |
| 63 | L226_1164 | 0.01 | -0.0064 | yes | lower bound; tex prints -0.01 | RFM/scores/part24/FT/FT24_whole_seed0_perclip.csv + A2 |
| 64 | L226_1170 | 0.05 | 0.0547 | yes | upper bound | RFM/scores/part24/FT/FT24_whole_seed0_perclip.csv + A2 |
| 65 | L226_1181 | 0.82 | 0.8184 | yes | seed1 0.8184 [0.7496, 0.8758] | RFM/scores/part24/FT/FT24_whole_seed1all_perclip.csv |
| 66 | L226_1340 | 0.59 | 0.5932 | yes | after conflict 0.5932 [0.5087, 0.6831] n=483 (p_yes_sft col 0.5985); before 0.2218 from p14_o25_zeroshot_scores.csv n=483 (tex 0.22) | <local data dir>/release/scores/part20/POD3/p14_ft966_oof.csv |
| 67 | L226_1379 | 0.67 | 0.6686 | yes | after agreement distinct 0.6686 [0.5727, 0.7586] n=311; before 0.9647 from T1_perclip_distinct_agreement.csv n=311 (tex 0.96) | <local data dir>/release/scores/part20/POD3/p14_ft966_oof.csv |
| 68 | L226_469 | 0.74 to 0.81 | 0.7396 to 0.8085 | yes | seeds 1-3 overall 0.7396, 0.8085, 0.7975 | <local data dir>/release/scores/part20/POD1b/std_ft_seed{1,2,3}_pitt_oof.csv |
| 69 | L226_557 | 0.45 to 0.57 | 0.4490 to 0.5666 | yes | conflict arm original omnisft 0.5189, seed1 0.5364, seed2 0.5636, seed3 0.5666, standard_rerun 0.4697, standard_rerun_r2 0.4490; seed0 file (not in the six) conflict 0.5282, agreement 0.8983, overall 0.7955 | six standard runs (see note) |
| 70 | L226_596 | 0.5 | all contain 0.5: True | yes | original omnisft [0.4151, 0.6300]; seed1 [0.4306, 0.6426]; seed2 [0.4593, 0.6779]; seed3 [0.4660, 0.6746]; standard_rerun [0.3613, 0.5906]; standard_rerun_r2 [0.3385, 0.5682]; seed0 (extra) [0.4200, 0.6421]; balanced [0.3686, 0.5854]; balanced_r2 [0.3775, 0.5991]; balanced_fold3rerun [0.3936, 0.6170]; 0.5 is chance, a claim about intervals | all runs above |
| 71 | L226_634 | 0.82 to 0.90 | 0.8213 to 0.9022 | yes | agreement arm original omnisft 0.8593, seed1 0.8213, seed2 0.9022, seed3 0.8844, standard_rerun 0.8895, standard_rerun_r2 0.8967 | six standard runs |
| 72 | L226_888 | 5.0 | 5.0463 | yes | json 5,046,272; arithmetic 5,046,272; parameter count in millions, checked at 1 dp as printed | <local data dir>/release/edaic_rerun/part16/POD2/lora_pitt_params.json |
| 73 | L226_924 | 4.6 | 4.5911 | yes | json 4,591,104; arithmetic 1280x3584+3584 = 4,591,104; parameter count in millions, checked at 1 dp as printed | <local data dir>/release/edaic_rerun/part16/POD2/lora_pitt_params.json |
| 74 | L226_958 | 0.825 | 0.8250 | yes | LoRA AUC n=468 (tex 3 dp); conflict arm 0.5520 (tex 0.55); checked at 3 dp as printed | <local data dir>/release/edaic_rerun/part16/POD2/lora_pitt_oof.csv |
| 75 | L226_973 | 0.06 | 0.0577 | yes | LoRA 0.8250 minus projector 0.7673 = +0.0577 [+0.0154, +0.1044] | <local data dir>/release/edaic_rerun/part16/POD2/lora_pitt_oof.csv + omni_final/omnisft_pitt_oof.csv |
| 76 | L226_979 | 0.02, 0.10 | 0.0154, 0.1044 | yes | interval | as row 75 |
