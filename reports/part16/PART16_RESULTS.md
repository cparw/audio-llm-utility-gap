# PART 16 results

Project notes for the PART 16 runs, started 2026-09-22. One row per value. `file` is the per-clip csv or source the value was computed from.
Bootstrap everywhere: 2000 draws, `numpy.random.default_rng(0)`, speakers resampled (never clips), percentile 2.5/97.5.
PAIRED = the speaker list is drawn once per replicate and both quantities recomputed inside that draw.
AUC recomputed with the rank formula before writing. Zero writes under `omni_final`.
Every value is recomputed by an independent verifier, and a value is final only when both agree to 4 decimals.
Paths in the `file` column point into this repository when the file is here, otherwise to `<local data dir>`.
NA means the value has no interval.

| id | what | value | 95% CI | n | n_spk | file |
|----|------|-------|--------|---|-------|------|
| 1a | Omni transcript-only, 966 conflict clips, pooled | 0.6077 | NA | 966 | 138 | part16/pod_sync/p14_o25_text.csv |
| 1a | Omni transcript-only, conflict arm | 0.1498 | [0.1131, 0.1887] | 483 | 138 | part15/B_text_conflict.csv |
| 1a | Omni transcript-only, agreement arm | 0.9596 | [0.9306, 0.9823] | 483 | 138 | part15/B_text_conflict.csv |
| 1a | Qwen2-Audio transcript-only, pooled | 0.5995 | NA | 966 | 138 | part16/pod_sync/p14_q2a_text.csv |
| 1a | Qwen2-Audio transcript-only, conflict arm | 0.2359 | [0.1851, 0.2886] | 483 | 138 | part15/B_text_conflict.csv |
| 1a | Qwen2-Audio transcript-only, agreement arm | 0.9081 | [0.8515, 0.9547] | 483 | 138 | part15/B_text_conflict.csv |
| 1a | Omni text vs audio Pearson r | 0.8442 | NA | 966 | 138 | part15/B_text_conflict.json |
| 1a | Omni text vs audio same Yes/No decision | 0.8634 | NA | 966 | 138 | part15/B_text_conflict.json |
| 3a | Qwen3-Omni Pitt encoder, nested, 5 repeats | 0.8122 | per-repeat [0.8199,0.8016,0.8142,0.8342,0.7914] | 468 | 228 | part16/pod_sync/q3o_pitt_nested_repeats.json |
| 3a | Qwen3-Omni Pitt projector, nested | 0.7820 | 1 repeat only | 468 | 228 | part16/pod_sync/q3o_pitt_nested_repeats.json |
| 3a | Qwen3-Omni Pitt LM best, nested, 5 repeats | 0.8445 | per-repeat [0.8397,0.8460,0.8393,0.8505,0.8468] | 468 | 228 | part16/pod_sync/q3o_pitt_nested_repeats.json |
| 3a | Qwen3-Omni Pitt answer state, nested, 5 repeats | 0.8237 | per-repeat [0.8144,0.8180,0.8221,0.8442,0.8200] | 468 | 228 | part16/pod_sync/q3o_pitt_nested_repeats.json |
| 3a | Qwen3-Omni Pitt zero-shot answer (THIS RUN) | 0.7673 | NA | 468 | 228 | part16/pod_sync/q3o_pitt_zeroshot_scores.csv |
| 3b | Qwen3-Omni Pitt cos(probe dir, Yes-No unembed) | 0.0051 | random floor 0.0177 [0.0008, 0.0498] | 468 | 228 | part16/pod_sync/readout_direction_q3o.csv |
| 3b | Qwen3-Omni Pitt probe AUC | 0.8307 | NA | 468 | 228 | part16/pod_sync/readout_direction_q3o.csv |
| 3b | Qwen3-Omni Pitt AUC along readout direction | 0.7656 | NA | 468 | 228 | part16/pod_sync/readout_direction_q3o.csv |
| 3b | Qwen3-Omni Pitt probe after projecting out readout dir | 0.8305 | NA | 468 | 228 | part16/pod_sync/readout_direction_q3o.csv |
| edaic_full_zeroshot_answer | Qwen2.5-Omni zero-shot answer AUC | 0.8278 | [0.7703, 0.8822] | 275 | 275 | EDAICFULL/edaic_full_gap.csv |
| edaic_full_probe_enc | encoder probe AUC (per-clip OOF averaged over repeats) | 0.5916 | [0.511, 0.6706] | 275 | 275 | EDAICFULL/edaic_full_gap.csv |
| edaic_full_probe_proj | projector probe AUC (per-clip OOF averaged over repeats) | 0.5631 | [0.479, 0.6478] | 275 | 275 | EDAICFULL/edaic_full_gap.csv |
| edaic_full_probe_llm | LM probe AUC (per-clip OOF averaged over repeats) | 0.7088 | [0.6232, 0.7874] | 275 | 275 | EDAICFULL/edaic_full_gap.csv |
| edaic_full_probe_ans | answer-state probe AUC (per-clip OOF averaged over repeats) | 0.7748 | [0.7041, 0.8387] | 275 | 275 | EDAICFULL/edaic_full_gap.csv |
| edaic_full_gap_llm_minus_answer | LM probe minus zero-shot answer (paired) | -0.119 | [-0.1995, -0.0478] | 275 | 275 | EDAICFULL/edaic_full_gap.csv |
| edaic_full_gap_enc_minus_answer | encoder probe minus zero-shot answer (paired) | -0.2361 | [-0.3272, -0.1394] | 275 | 275 | EDAICFULL/edaic_full_gap.csv |
| edaic_full_gap_ans_minus_answer | answer-state probe minus zero-shot answer (paired) | -0.053 | [-0.1046, -0.0049] | 275 | 275 | EDAICFULL/edaic_full_gap.csv |
| edaic_full_gap_proj_minus_answer | projector probe minus zero-shot answer (paired) | -0.2646 | [-0.3575, -0.1667] | 275 | 275 | EDAICFULL/edaic_full_gap.csv |
| edaic_first30_zeroshot_answer | Qwen2.5-Omni zero-shot answer AUC, FIRST 30 s window | 0.662 | [0.5831, 0.7315] | 275 | 275 | EDAICFULL/edaic_full_gap.csv |
| edaic_first30_probe_enc | encoder probe AUC, FIRST 30 s window | 0.6324 | [0.552, 0.7104] | 275 | 275 | EDAICFULL/edaic_full_gap.csv |
| edaic_first30_gap_enc_minus_answer | encoder probe minus zero-shot answer (paired), FIRST 30 s window | -0.0297 | [-0.1291, 0.0752] | 275 | 275 | EDAICFULL/edaic_full_gap.csv |
| M1_text_all468 | Pitt Qwen2.5-Omni transcript-only answer AUC, all468 arm | 0.6975 | [0.6369, 0.7549] | 468 | 228 | <local data dir>/paper1_local_runs/omni_pitt_text.csv |
| M1_text_conflict | Pitt Qwen2.5-Omni transcript-only answer AUC, conflict arm | 0.4550 | [0.3412, 0.5718] | 146 | 100 | <local data dir>/paper1_local_runs/omni_pitt_text.csv |
| M1_text_agreement | Pitt Qwen2.5-Omni transcript-only answer AUC, agreement arm | 0.7987 | [0.7392, 0.8535] | 322 | 175 | <local data dir>/paper1_local_runs/omni_pitt_text.csv |
| M1_audio_all468 | Pitt Qwen2.5-Omni audio zero-shot answer AUC, all468 arm | 0.6578 | [0.6033, 0.7125] | 468 | 228 | <local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv |
| M1_audio_conflict | Pitt Qwen2.5-Omni audio zero-shot answer AUC, conflict arm | 0.5322 | [0.4216, 0.6447] | 146 | 100 | <local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv |
| M1_audio_agreement | Pitt Qwen2.5-Omni audio zero-shot answer AUC, agreement arm | 0.7054 | [0.6431, 0.7658] | 322 | 175 | <local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv |
| M1_text_conflict_minus_agreement | Pitt Qwen2.5-Omni transcript-only paired conflict minus agreement AUC | -0.3437 | [-0.4556, -0.2209] | 468 | 228 | <local data dir>/paper1_local_runs/omni_pitt_text.csv |
| M1_audio_conflict_minus_agreement | Pitt Qwen2.5-Omni audio zero-shot paired conflict minus agreement AUC | -0.1731 | [-0.2958, -0.0392] | 468 | 228 | <local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv |
| M1_text_minus_audio_all468 | Pitt Qwen2.5-Omni paired transcript minus audio AUC, all468 arm | 0.0397 | [-0.0302, 0.1070] | 468 | 228 | scores/part16/M/M1_pitt_text_arms.csv |
| M1_text_minus_audio_conflict | Pitt Qwen2.5-Omni paired transcript minus audio AUC, conflict arm | -0.0772 | [-0.2360, 0.0680] | 146 | 100 | scores/part16/M/M1_pitt_text_arms.csv |
| M1_text_minus_audio_agreement | Pitt Qwen2.5-Omni paired transcript minus audio AUC, agreement arm | 0.0933 | [0.0157, 0.1637] | 322 | 175 | scores/part16/M/M1_pitt_text_arms.csv |
| M1_ALT_text_all468 | ALT Pitt Qwen2.5-Omni transcript-only answer AUC, all468 arm (Sep 20 pod rerun, different prompt run) | 0.6629 | [0.5983, 0.7245] | 468 | 228 | <local data dir>/Desktop/release/overnight2/text_new/o25_pitt_text.csv |
| M1_ALT_text_conflict | ALT Pitt Qwen2.5-Omni transcript-only answer AUC, conflict arm (Sep 20 pod rerun, different prompt run) | 0.5560 | [0.4516, 0.6680] | 146 | 100 | <local data dir>/Desktop/release/overnight2/text_new/o25_pitt_text.csv |
| M1_ALT_text_agreement | ALT Pitt Qwen2.5-Omni transcript-only answer AUC, agreement arm (Sep 20 pod rerun, different prompt run) | 0.7078 | [0.6381, 0.7696] | 322 | 175 | <local data dir>/Desktop/release/overnight2/text_new/o25_pitt_text.csv |
| M10 | MDVR-KCL Qwen2.5-Omni transcript-only answer AUC | 0.7634 | [0.5936, 0.9118] | 37 | 37 | <local data dir>/Desktop/release/overnight2/text_new/o25_kcl_text.csv |
| M10 | MDVR-KCL Qwen2.5-Omni audio zero-shot answer AUC | 0.7143 | [0.5210, 0.8810] | 37 | 37 | <local data dir>/Desktop/release/omni_final/omni_kcl_zeroshot_scores.csv |
| M10 | MDVR-KCL Qwen2.5-Omni encoder probe AUC (single saved OOF repeat) | 0.7768 | [0.5994, 0.9265] | 37 | 37 | <local data dir>/Desktop/release/omni_final/omni_kcl_enc_nested_oof.csv |
| M10 | MDVR-KCL paired transcript minus audio answer AUC | 0.0491 | [-0.1827, 0.2818] | 37 | 37 | scores/part16/M/M10_kcl_transcript.csv |
| M10 | MDVR-KCL paired transcript minus encoder probe AUC | -0.0134 | [-0.2383, 0.2381] | 37 | 37 | scores/part16/M/M10_kcl_transcript.csv |
| M11 | A_asr_reproduction_wer_vs_paper_transcript | 0.0000 | NA | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | B_xsys_disagree_mean_PD | 0.3049 | [0.1301, 0.4944] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | B_xsys_disagree_mean_control | 0.4779 | [0.0917, 1.1479] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | B_xsys_disagree_diff_PD_minus_control | -0.1730 | [-0.8969, 0.3060] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | B_xsys_disagree_spearman_vs_pyes_text | 0.2723 | [-0.0814, 0.5558] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | B_xsys_disagree_spearman_vs_pyes_audio | -0.0916 | [-0.3876, 0.2143] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | B_xsys_disagree_auc_alone | 0.6220 | NA | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | C_mean_token_logprob_mean_PD | -0.1103 | [-0.1502, -0.0739] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | C_mean_token_logprob_mean_control | -0.1356 | [-0.2345, -0.0759] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | C_mean_token_logprob_diff_PD_minus_control | 0.0253 | [-0.0532, 0.1288] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | C_mean_token_logprob_spearman_vs_pyes_text | -0.4009 | [-0.6720, -0.0835] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | C_mean_token_logprob_spearman_vs_pyes_audio | -0.3101 | [-0.5693, 0.0113] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | C_mean_token_logprob_auc_alone | 0.4792 | NA | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | C_compression_ratio_mean_PD | 1.5938 | [1.4484, 1.7269] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | C_compression_ratio_mean_control | 1.4898 | [1.3076, 1.6417] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | C_compression_ratio_diff_PD_minus_control | 0.1040 | [-0.1036, 0.3400] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | C_compression_ratio_spearman_vs_pyes_text | -0.1390 | [-0.4219, 0.1794] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | C_compression_ratio_spearman_vs_pyes_audio | -0.1305 | [-0.4261, 0.1964] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | C_compression_ratio_auc_alone | 0.5625 | NA | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | D_n_words_30s_mean_PD | 59.1875 | [46.1875, 71.9375] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | D_n_words_30s_mean_control | 55.8095 | [43.0464, 67.6202] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | D_n_words_30s_diff_PD_minus_control | 3.3780 | [-13.8212, 22.5698] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | D_n_words_30s_spearman_vs_pyes_text | -0.1588 | [-0.4443, 0.1670] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | D_n_words_30s_spearman_vs_pyes_audio | -0.0508 | [-0.3683, 0.2864] | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | D_n_words_30s_auc_alone | 0.5268 | NA | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | E_auc_from_passage_identity_alone | 0.6682 | NA | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | auc_text_condition_recomputed | 0.7634 | NA | 37 | 37 | <local data dir>/Desktop/release/podE_final/out/o25_kcl_text.csv |
| M11 | auc_audio_condition_recomputed | 0.7143 | NA | 37 | 37 | <local data dir>/Desktop/release/omni_final/omni_kcl_zeroshot_scores.csv |
| M11 | F_wer_vs_local_copy_mean_PD | 0.3123 | [0.1974, 0.4388] | 37 | 37 | <local data dir>/paper1_local_runs/kcl_read30b_text.csv |
| M11 | F_wer_vs_local_copy_mean_control | 0.3987 | [0.2558, 0.5577] | 37 | 37 | <local data dir>/paper1_local_runs/kcl_read30b_text.csv |
| M11 | B_xsys_disagree_MEDIAN_PD | 0.1027 | [0.0323, 0.4799] | 16 | 16 | scores/part16/M/M11_kcl_wer.csv |
| M11 | B_xsys_disagree_MEDIAN_control | 0.0333 | [0.0000, 0.2088] | 21 | 21 | scores/part16/M/M11_kcl_wer.csv |
| M11 | B_xsys_disagree_MEDIAN_diff_PD_minus_control | 0.0694 | NA | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | B_xsys_disagree_ge20w_MEAN_PD | 0.2100 | [0.0645, 0.3906] | 14 | 14 | scores/part16/M/M11_kcl_wer.csv |
| M11 | B_xsys_disagree_ge20w_MEAN_control | 0.1506 | [0.0481, 0.2765] | 17 | 17 | scores/part16/M/M11_kcl_wer.csv |
| M11 | B_xsys_disagree_ge20w_MEAN_diff_PD_minus_control | 0.0594 | NA | 37 | 37 | scores/part16/M/M11_kcl_wer.csv |
| M11 | B_xsys_disagree_POOLED_PD | 0.2650 | [0.0729, 0.4944] | 16 | 16 | scores/part16/M/M11_kcl_wer.csv |
| M11 | B_xsys_disagree_POOLED_control | 0.1800 | [0.0595, 0.3764] | 21 | 21 | scores/part16/M/M11_kcl_wer.csv |
| M2.label_all.agreement | sentiment label (pos/neu/neg at 0.70), all segments | raw agreement po | 0.4248 | [0.4107, 0.4393] | 6214 | 275 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.label_all.kappa | sentiment label (pos/neu/neg at 0.70), all segments | Cohen kappa | 0.2481 | [0.2360, 0.2605] | 6214 | 275 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.label_elig.agreement | sentiment label (pos/neu/neg at 0.70), eligible segments | raw agreement po | 0.4286 | [0.4130, 0.4446] | 5523 | 275 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.label_elig.kappa | sentiment label (pos/neu/neg at 0.70), eligible segments | Cohen kappa | 0.2510 | [0.2377, 0.2653] | 5523 | 275 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.arm_rule.agreement | rule arm (conflict/agreement/none), eligible segments | raw agreement po | 0.5524 | [0.5286, 0.5754] | 5523 | 275 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.arm_rule.kappa | rule arm (conflict/agreement/none), eligible segments | Cohen kappa | 0.3438 | [0.3224, 0.3665] | 5523 | 275 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.arm_selected.agreement | selected arm (conflict/agreement/none), eligible segments | raw agreement po | 0.7045 | [0.6674, 0.7391] | 5523 | 275 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.arm_selected.kappa | selected arm (conflict/agreement/none), eligible segments | Cohen kappa | 0.3362 | [0.3008, 0.3717] | 5523 | 275 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.arm_rule_matched.agreement | rule arm, distilbert at matched threshold 0.995 | raw agreement po | 0.7290 | [0.7128, 0.7460] | 5523 | 275 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.arm_rule_matched.kappa | rule arm, distilbert at matched threshold 0.995 | Cohen kappa | 0.4978 | [0.4728, 0.5234] | 5523 | 275 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.arm_selected_matched.agreement | selected arm, distilbert at matched threshold 0.995 | raw agreement po | 0.8356 | [0.8146, 0.8560] | 5523 | 275 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.arm_selected_matched.kappa | selected arm, distilbert at matched threshold 0.995 | Cohen kappa | 0.4366 | [0.3989, 0.4755] | 5523 | 275 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.matched_threshold | distilbert threshold whose conflict count is closest to RoBERTa's 488 | 0.995 | NA | 644 | 149 | reports/part16/M2_REPORT.txt |
| M2.pairs | distilbert SST-2 rebuild of the Part 14 selection | pairs | 1334 | NA | 2668 | 162 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.clips | distilbert SST-2 rebuild of the Part 14 selection | clips | 2668 | NA | 2668 | 162 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.speakers | distilbert SST-2 rebuild of the Part 14 selection | speakers | 162 | NA | 2668 | 162 | scores/part16/M/M2_distilbert_sentiment.csv |
| M2.same_arm_seg_uid | segments with the same (arm, seg_uid) in both selections | 625 | NA | 2302 | 162 | scores/part16/M/M2_distilbert_sentiment.csv |
| M3_pitt_clip | Omni zero shot pitt clip-level AUC | 0.6578 | [0.6033, 0.7125] | 468 | 228 | <local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv |
| M3_pitt_speaker | Omni zero shot pitt speaker-level AUC (mean p_yes) | 0.6771 | [0.6096, 0.7494] | 228 | 228 | scores/part16/M/M3_speaker_level.csv |
| M3_pitt_clip_minus_speaker | Omni zero shot pitt paired clip minus speaker AUC | -0.0193 | [-0.0565, 0.0172] | 468 | 228 | scores/part16/M/M3_speaker_level.csv |
| M3_pcgita_clip | Omni zero shot pcgita clip-level AUC | 0.5635 | [0.4931, 0.6331] | 1100 | 100 | <local data dir>/Desktop/release/omni_final/omni_pcgita_zeroshot_scores.csv |
| M3_pcgita_speaker | Omni zero shot pcgita speaker-level AUC (mean p_yes) | 0.5948 | [0.4778, 0.7089] | 100 | 100 | scores/part16/M/M3_speaker_level.csv |
| M3_pcgita_clip_minus_speaker | Omni zero shot pcgita paired clip minus speaker AUC | -0.0313 | [-0.0820, 0.0179] | 1100 | 100 | scores/part16/M/M3_speaker_level.csv |
| M3_neurovoz_clip | Omni zero shot neurovoz clip-level AUC | 0.6951 | [0.6356, 0.7519] | 1270 | 107 | <local data dir>/Desktop/release/omni_final/omni_neurovoz_zeroshot_scores.csv |
| M3_neurovoz_speaker | Omni zero shot neurovoz speaker-level AUC (mean p_yes) | 0.7948 | [0.7045, 0.8733] | 107 | 107 | scores/part16/M/M3_speaker_level.csv |
| M3_neurovoz_clip_minus_speaker | Omni zero shot neurovoz paired clip minus speaker AUC | -0.0997 | [-0.1325, -0.0613] | 1270 | 107 | scores/part16/M/M3_speaker_level.csv |
| M4 | yes_rate_Qwen2.5-Omni_conflict | 0.2236 | [0.1792, 0.2736] | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | median_answer_mass_Qwen2.5-Omni_conflict | 0.9965 | NA | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_Qwen2.5-Omni_agreement | 0.2919 | [0.2073, 0.3771] | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | median_answer_mass_Qwen2.5-Omni_agreement | 0.9969 | NA | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_diff_conflict_minus_agreement_Qwen2.5-Omni | -0.0683 | [-0.1788, 0.0474] | 966 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_Qwen2-Audio_conflict | 0.9400 | [0.9199, 0.9612] | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | median_answer_mass_Qwen2-Audio_conflict | 0.9990 | NA | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_Qwen2-Audio_agreement | 0.8675 | [0.8159, 0.9102] | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | median_answer_mass_Qwen2-Audio_agreement | 0.9988 | NA | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_diff_conflict_minus_agreement_Qwen2-Audio | 0.0725 | [0.0200, 0.1330] | 966 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_Qwen3-Omni-30B-A3B_conflict | 0.3892 | [0.3347, 0.4500] | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | median_answer_mass_Qwen3-Omni-30B-A3B_conflict | 0.9996 | NA | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_Qwen3-Omni-30B-A3B_agreement | 0.4182 | [0.3188, 0.5097] | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | median_answer_mass_Qwen3-Omni-30B-A3B_agreement | 0.9998 | NA | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_diff_conflict_minus_agreement_Qwen3-Omni-30B-A3B | -0.0290 | [-0.1540, 0.1078] | 966 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_AudioFlamingo2_conflict | 0.9710 | [0.9502, 0.9887] | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | median_answer_mass_AudioFlamingo2_conflict | 0.9424 | NA | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_AudioFlamingo2_agreement | 0.9834 | [0.9629, 0.9979] | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | median_answer_mass_AudioFlamingo2_agreement | 0.9423 | NA | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_diff_conflict_minus_agreement_AudioFlamingo2 | -0.0124 | [-0.0280, 0.0020] | 966 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_AudioFlamingo3_conflict | 0.7039 | [0.6533, 0.7554] | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | median_answer_mass_AudioFlamingo3_conflict | 0.9249 | NA | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_AudioFlamingo3_agreement | 0.6687 | [0.5937, 0.7371] | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | median_answer_mass_AudioFlamingo3_agreement | 0.9403 | NA | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_diff_conflict_minus_agreement_AudioFlamingo3 | 0.0352 | [-0.0491, 0.1281] | 966 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_Kimi-Audio_conflict | 0.8364 | [0.7976, 0.8727] | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | median_answer_mass_Kimi-Audio_conflict | 0.9992 | NA | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_Kimi-Audio_agreement | 0.7578 | [0.6869, 0.8221] | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | median_answer_mass_Kimi-Audio_agreement | 0.9992 | NA | 483 | 138 | scores/part16/M/M4_yes_rates.csv |
| M4 | yes_rate_diff_conflict_minus_agreement_Kimi-Audio | 0.0787 | [0.0160, 0.1461] | 966 | 138 | scores/part16/M/M4_yes_rates.csv |
| M5_all_cos | Qwen2-Audio cos(probe_dir, readout_dir) [all] | 0.0145 | NA | 468 | 228 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_all_randfloor | Qwen2-Audio random-direction floor mean|cos| [all] | 0.0124 | [0.0005, 0.0343] | 468 | 228 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_all_auc_probe | Qwen2-Audio auc probe out-of-fold [all] | 0.8223 | [0.7742, 0.8671] | 468 | 228 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_all_auc_d | Qwen2-Audio auc along readout direction [all] | 0.6174 | [0.5539, 0.6800] | 468 | 228 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_all_auc_without_d | Qwen2-Audio auc probe with readout direction removed (coef proj) [all] | 0.8327 | [0.7857, 0.8759] | 468 | 228 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_all_auc_without_d_Xproj | Qwen2-Audio auc probe with readout direction removed (X proj) [all] | 0.8223 | [0.7743, 0.8672] | 468 | 228 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_conflict_refit_cos | Qwen2-Audio cos(probe_dir, readout_dir) [conflict_refit] | 0.0054 | NA | 146 | 100 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_conflict_refit_randfloor | Qwen2-Audio random-direction floor mean|cos| [conflict_refit] | 0.0123 | [0.0005, 0.0350] | 146 | 100 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_conflict_refit_auc_probe | Qwen2-Audio auc probe out-of-fold [conflict_refit] | 0.6757 | [0.5685, 0.7761] | 146 | 100 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_conflict_refit_auc_d | Qwen2-Audio auc along readout direction [conflict_refit] | 0.4136 | [0.3010, 0.5424] | 146 | 100 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_conflict_refit_auc_without_d | Qwen2-Audio auc probe with readout direction removed (coef proj) [conflict_refit] | 0.6869 | [0.5806, 0.7872] | 146 | 100 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_conflict_refit_auc_without_d_Xproj | Qwen2-Audio auc probe with readout direction removed (X proj) [conflict_refit] | 0.6759 | [0.5687, 0.7766] | 146 | 100 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_agreement_refit_cos | Qwen2-Audio cos(probe_dir, readout_dir) [agreement_refit] | 0.0147 | NA | 322 | 175 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_agreement_refit_randfloor | Qwen2-Audio random-direction floor mean|cos| [agreement_refit] | 0.0126 | [0.0005, 0.0359] | 322 | 175 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_agreement_refit_auc_probe | Qwen2-Audio auc probe out-of-fold [agreement_refit] | 0.9317 | [0.9020, 0.9578] | 322 | 175 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_agreement_refit_auc_d | Qwen2-Audio auc along readout direction [agreement_refit] | 0.6984 | [0.6373, 0.7592] | 322 | 175 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_agreement_refit_auc_without_d | Qwen2-Audio auc probe with readout direction removed (coef proj) [agreement_refit] | 0.9346 | [0.9057, 0.9603] | 322 | 175 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_agreement_refit_auc_without_d_Xproj | Qwen2-Audio auc probe with readout direction removed (X proj) [agreement_refit] | 0.9325 | [0.9027, 0.9583] | 322 | 175 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_conflict_restricted_auc_probe | Qwen2-Audio auc probe out-of-fold [conflict_restricted] | 0.6178 | [0.5123, 0.7222] | 146 | 100 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_conflict_restricted_auc_d | Qwen2-Audio auc along readout direction [conflict_restricted] | 0.4136 | [0.3010, 0.5424] | 146 | 100 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_conflict_restricted_auc_without_d | Qwen2-Audio auc probe with readout direction removed (coef proj) [conflict_restricted] | 0.6326 | [0.5277, 0.7336] | 146 | 100 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_conflict_restricted_auc_without_d_Xproj | Qwen2-Audio auc probe with readout direction removed (X proj) [conflict_restricted] | 0.6180 | [0.5128, 0.7213] | 146 | 100 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_agreement_restricted_auc_probe | Qwen2-Audio auc probe out-of-fold [agreement_restricted] | 0.8988 | [0.8598, 0.9328] | 322 | 175 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_agreement_restricted_auc_d | Qwen2-Audio auc along readout direction [agreement_restricted] | 0.6984 | [0.6373, 0.7592] | 322 | 175 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_agreement_restricted_auc_without_d | Qwen2-Audio auc probe with readout direction removed (coef proj) [agreement_restricted] | 0.9070 | [0.8702, 0.9383] | 322 | 175 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_agreement_restricted_auc_without_d_Xproj | Qwen2-Audio auc probe with readout direction removed (X proj) [agreement_restricted] | 0.8988 | [0.8593, 0.9326] | 322 | 175 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_paired_auc_probe_conflict_minus_agreement | Qwen2-Audio auc_probe conflict minus agreement (paired speaker bootstrap) | -0.2810 | [-0.3799, -0.1762] | 468 | 228 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_paired_auc_d_conflict_minus_agreement | Qwen2-Audio auc_d conflict minus agreement (paired speaker bootstrap) | -0.2847 | [-0.3947, -0.1475] | 468 | 228 | scores/part16/M/M5_readout_direction_q2a.csv |
| M5_paired_auc_without_d_conflict_minus_agreement | Qwen2-Audio auc_without_d conflict minus agreement (paired speaker bootstrap) | -0.2745 | [-0.3754, -0.1707] | 468 | 228 | scores/part16/M/M5_readout_direction_q2a.csv |
| M7_pitt_probe | Qwen2.5-Omni nested encoder probe AUC, pitt | 0.7969 | [0.7412, 0.8512] | 468 | 228 | <local data dir>/Desktop/release/omni_final/omni_pitt_enc_nested_oof.csv |
| M7_pitt_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, pitt | 0.6578 | [0.6033, 0.7125] | 468 | 228 | <local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv |
| M7_pitt_gap | paired probe minus zero-shot AUC, pitt | 0.1391 | [0.0680, 0.2084] | 468 | 228 | scores/part16/M/M7_perclip/pitt.csv |
| M7_adresso_probe | Qwen2.5-Omni nested encoder probe AUC, adresso | 0.8899 | [0.8462, 0.9321] | 237 | 237 | <local data dir>/Desktop/release/omni_final/omni_adresso_enc_nested_oof.csv |
| M7_adresso_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, adresso | 0.6150 | [0.5432, 0.6890] | 237 | 237 | <local data dir>/Desktop/release/omni_final/omni_adresso_zeroshot_scores.csv |
| M7_adresso_gap | paired probe minus zero-shot AUC, adresso | 0.2748 | [0.1920, 0.3526] | 237 | 237 | scores/part16/M/M7_perclip/adresso.csv |
| M7_adress2020_probe | Qwen2.5-Omni nested encoder probe AUC, adress2020 | 0.7878 | [0.7148, 0.8517] | 156 | 156 | <local data dir>/Desktop/release/omni_final/omni_adress2020_enc_nested_oof.csv |
| M7_adress2020_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, adress2020 | 0.6241 | [0.5348, 0.7148] | 156 | 156 | <local data dir>/Desktop/release/omni_final/omni_adress2020_zeroshot_scores.csv |
| M7_adress2020_gap | paired probe minus zero-shot AUC, adress2020 | 0.1637 | [0.0557, 0.2693] | 156 | 156 | scores/part16/M/M7_perclip/adress2020.csv |
| M7_pcgita_probe | Qwen2.5-Omni nested encoder probe AUC, pcgita | 0.9089 | [0.8478, 0.9563] | 1100 | 100 | <local data dir>/Desktop/release/omni_final/omni_pcgita_enc_nested_oof.csv |
| M7_pcgita_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, pcgita | 0.5635 | [0.4931, 0.6331] | 1100 | 100 | <local data dir>/Desktop/release/omni_final/omni_pcgita_zeroshot_scores.csv |
| M7_pcgita_gap | paired probe minus zero-shot AUC, pcgita | 0.3454 | [0.2636, 0.4275] | 1100 | 100 | scores/part16/M/M7_perclip/pcgita.csv |
| M7_neurovoz_probe | Qwen2.5-Omni nested encoder probe AUC, neurovoz | 0.9160 | [0.8713, 0.9513] | 1270 | 107 | <local data dir>/Desktop/release/omni_final/omni_neurovoz_enc_nested_oof.csv |
| M7_neurovoz_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, neurovoz | 0.6951 | [0.6356, 0.7519] | 1270 | 107 | <local data dir>/Desktop/release/omni_final/omni_neurovoz_zeroshot_scores.csv |
| M7_neurovoz_gap | paired probe minus zero-shot AUC, neurovoz | 0.2209 | [0.1660, 0.2783] | 1270 | 107 | scores/part16/M/M7_perclip/neurovoz.csv |
| M7_kcl_probe | Qwen2.5-Omni nested encoder probe AUC, kcl | 0.7768 | [0.5994, 0.9265] | 37 | 37 | <local data dir>/Desktop/release/omni_final/omni_kcl_enc_nested_oof.csv |
| M7_kcl_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, kcl | 0.7143 | [0.5210, 0.8810] | 37 | 37 | <local data dir>/Desktop/release/omni_final/omni_kcl_zeroshot_scores.csv |
| M7_kcl_gap | paired probe minus zero-shot AUC, kcl | 0.0625 | [-0.1813, 0.2900] | 37 | 37 | scores/part16/M/M7_perclip/kcl.csv |
| M7_edaic_probe | Qwen2.5-Omni nested encoder probe AUC, edaic | 0.6324 | [0.5520, 0.7104] | 275 | 275 | <local data dir>/Desktop/release/omni_final/omni_edaic_enc_nested_oof.csv |
| M7_edaic_zeroshot | Qwen2.5-Omni zero-shot audio answer AUC, edaic | 0.6620 | [0.5831, 0.7315] | 275 | 275 | <local data dir>/Desktop/release/omni_final/omni_edaic_zeroshot_scores.csv |
| M7_edaic_gap | paired probe minus zero-shot AUC, edaic | -0.0297 | [-0.1291, 0.0752] | 275 | 275 | scores/part16/M/M7_perclip/edaic.csv |
| M7_mean_gap | mean paired probe minus zero-shot AUC across 7 datasets | 0.1681 | NA | 3543 |  | scores/part16/M/M7_gap_per_dataset.csv |
| M7_pitt_probe_rep5 | Qwen2.5-Omni encoder probe AUC, pitt, mean of 5-repeat OOF probs (supplement) | 0.7917 | [0.7324, 0.8480] | 468 | 228 | scores/part10/pitt_enc_nested5_oof.npz |
| M7_pitt_gap_rep5 | paired probe minus zero-shot AUC, pitt, 5-repeat OOF (supplement) | 0.1339 | [0.0641, 0.2019] | 468 | 228 | scores/part16/M/M7_perclip/pitt_rep5mean.csv |
| M7fix | Qwen2.5-Omni Pitt encoder probe, 5-repeat (CORRECTED source) | 0.7706 | [0.7157, 0.8220] | 468 | 228 | scores/part10/pitt_enc_nested5_oof.npz |
| M7fix | Qwen2.5-Omni Pitt paired probe minus zero-shot (CORRECTED) | +0.1128 | [0.0455, 0.1768] | 468 | 228 | scores/part16/M/M7_perclip/pitt_FIXED.csv |
| M8 | Qwen2.5-Omni | A_edaic_part14 | conflict | src=primary | 0.2218 | [0.1829, 0.2632] | 483 | 138 | scores/edaic_conflict_v2/p14_o25_zeroshot_scores.csv |
| M8 | Qwen2.5-Omni | A_edaic_part14 | agreement | src=primary | 0.9482 | [0.9039, 0.9828] | 483 | 138 | scores/edaic_conflict_v2/p14_o25_zeroshot_scores.csv |
| M8 | Qwen2.5-Omni | A_edaic_part14 | conflict_minus_agreement | src=primary | -0.7264 | [-0.7786, -0.6655] | 966 | 138 | scores/edaic_conflict_v2/p14_o25_zeroshot_scores.csv |
| M8 | Qwen2-Audio | A_edaic_part14 | conflict | src=primary | 0.2280 | [0.1816, 0.2782] | 483 | 138 | scores/edaic_conflict_v2/p14_q2a_zeroshot_scores.csv |
| M8 | Qwen2-Audio | A_edaic_part14 | agreement | src=primary | 0.9478 | [0.9128, 0.9762] | 483 | 138 | scores/edaic_conflict_v2/p14_q2a_zeroshot_scores.csv |
| M8 | Qwen2-Audio | A_edaic_part14 | conflict_minus_agreement | src=primary | -0.7198 | [-0.7689, -0.6679] | 966 | 138 | scores/edaic_conflict_v2/p14_q2a_zeroshot_scores.csv |
| M8 | Qwen3-Omni-30B-A3B | A_edaic_part14 | conflict | src=primary | 0.3008 | [0.2454, 0.3641] | 483 | 138 | scores/edaic_conflict_v2/p14_q3o_zeroshot_scores.csv |
| M8 | Qwen3-Omni-30B-A3B | A_edaic_part14 | agreement | src=primary | 0.9638 | [0.9336, 0.9864] | 483 | 138 | scores/edaic_conflict_v2/p14_q3o_zeroshot_scores.csv |
| M8 | Qwen3-Omni-30B-A3B | A_edaic_part14 | conflict_minus_agreement | src=primary | -0.6630 | [-0.7192, -0.6030] | 966 | 138 | scores/edaic_conflict_v2/p14_q3o_zeroshot_scores.csv |
| M8 | Audio Flamingo 2 | A_edaic_part14 | conflict | src=primary | 0.5286 | [0.4512, 0.6015] | 483 | 138 | scores/edaic_conflict_v2/p14_af2.csv |
| M8 | Audio Flamingo 2 | A_edaic_part14 | agreement | src=primary | 0.6083 | [0.5126, 0.6997] | 483 | 138 | scores/edaic_conflict_v2/p14_af2.csv |
| M8 | Audio Flamingo 2 | A_edaic_part14 | conflict_minus_agreement | src=primary | -0.0797 | [-0.1488, -0.0104] | 966 | 138 | scores/edaic_conflict_v2/p14_af2.csv |
| M8 | Audio Flamingo 3 | A_edaic_part14 | conflict | src=primary | 0.4235 | [0.3606, 0.4880] | 483 | 138 | scores/edaic_conflict_v2/p14_af3.csv |
| M8 | Audio Flamingo 3 | A_edaic_part14 | agreement | src=primary | 0.8095 | [0.7447, 0.8696] | 483 | 138 | scores/edaic_conflict_v2/p14_af3.csv |
| M8 | Audio Flamingo 3 | A_edaic_part14 | conflict_minus_agreement | src=primary | -0.3860 | [-0.4675, -0.3018] | 966 | 138 | scores/edaic_conflict_v2/p14_af3.csv |
| M8 | Kimi-Audio | A_edaic_part14 | conflict | src=primary | 0.5258 | [0.4501, 0.5977] | 483 | 138 | scores/edaic_conflict_v2/p14_kimi_zeroshot_scores.csv |
| M8 | Kimi-Audio | A_edaic_part14 | agreement | src=primary | 0.8744 | [0.8172, 0.9223] | 483 | 138 | scores/edaic_conflict_v2/p14_kimi_zeroshot_scores.csv |
| M8 | Kimi-Audio | A_edaic_part14 | conflict_minus_agreement | src=primary | -0.3486 | [-0.4248, -0.2795] | 966 | 138 | scores/edaic_conflict_v2/p14_kimi_zeroshot_scores.csv |
| M8 | Qwen2.5-Omni | B_pitt | conflict | src=primary | 0.5322 | [0.4216, 0.6447] | 146 | 100 | <local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv |
| M8 | Qwen2.5-Omni | B_pitt | agreement | src=primary | 0.7054 | [0.6431, 0.7658] | 322 | 175 | <local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv |
| M8 | Qwen2.5-Omni | B_pitt | conflict_minus_agreement | src=primary | -0.1731 | [-0.2958, -0.0392] | 468 | 228 | <local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv |
| M8 | Qwen2-Audio | B_pitt | conflict | src=primary | 0.4132 | [0.3005, 0.5423] | 146 | 100 | <local data dir>/paper1_local_runs/probe2/pitt_zeroshot_scores.csv |
| M8 | Qwen2-Audio | B_pitt | agreement | src=primary | 0.6985 | [0.6372, 0.7597] | 322 | 175 | <local data dir>/paper1_local_runs/probe2/pitt_zeroshot_scores.csv |
| M8 | Qwen2-Audio | B_pitt | conflict_minus_agreement | src=primary | -0.2853 | [-0.3950, -0.1485] | 468 | 228 | <local data dir>/paper1_local_runs/probe2/pitt_zeroshot_scores.csv |
| M8 | Qwen3-Omni-30B-A3B | B_pitt | conflict | src=primary | 0.6066 | [0.4956, 0.7123] | 146 | 100 | <local data dir>/Desktop/release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv |
| M8 | Qwen3-Omni-30B-A3B | B_pitt | agreement | src=primary | 0.8231 | [0.7664, 0.8720] | 322 | 175 | <local data dir>/Desktop/release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv |
| M8 | Qwen3-Omni-30B-A3B | B_pitt | conflict_minus_agreement | src=primary | -0.2165 | [-0.3242, -0.1052] | 468 | 228 | <local data dir>/Desktop/release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv |
| M8 | Audio Flamingo 2 | B_pitt | conflict | src=af2_pitt468 | 0.6310 | [0.5336, 0.7260] | 146 | 100 | <local data dir>/paper1_local_runs/af2_results/af2_pitt468.csv |
| M8 | Audio Flamingo 2 | B_pitt | agreement | src=af2_pitt468 | 0.5544 | [0.4760, 0.6296] | 322 | 175 | <local data dir>/paper1_local_runs/af2_results/af2_pitt468.csv |
| M8 | Audio Flamingo 2 | B_pitt | conflict_minus_agreement | src=af2_pitt468 | 0.0766 | [-0.0307, 0.1869] | 468 | 228 | <local data dir>/paper1_local_runs/af2_results/af2_pitt468.csv |
| M8 | Audio Flamingo 2 | B_pitt | conflict | src=af2_pitt_audio | 0.6691 | [0.5650, 0.7654] | 146 | 100 | <local data dir>/paper1_local_runs/af2_pitt_audio.csv |
| M8 | Audio Flamingo 2 | B_pitt | agreement | src=af2_pitt_audio | 0.4895 | [0.4182, 0.5591] | 322 | 175 | <local data dir>/paper1_local_runs/af2_pitt_audio.csv |
| M8 | Audio Flamingo 2 | B_pitt | conflict_minus_agreement | src=af2_pitt_audio | 0.1796 | [0.0702, 0.2825] | 468 | 228 | <local data dir>/paper1_local_runs/af2_pitt_audio.csv |
| M8 | Audio Flamingo 3 | B_pitt | conflict | src=primary | 0.6169 | [0.5188, 0.7209] | 146 | 100 | <local data dir>/Desktop/release/overnight2/part3/af3_pitt.csv |
| M8 | Audio Flamingo 3 | B_pitt | agreement | src=primary | 0.5822 | [0.5080, 0.6528] | 322 | 175 | <local data dir>/Desktop/release/overnight2/part3/af3_pitt.csv |
| M8 | Audio Flamingo 3 | B_pitt | conflict_minus_agreement | src=primary | 0.0347 | [-0.0698, 0.1474] | 468 | 228 | <local data dir>/Desktop/release/overnight2/part3/af3_pitt.csv |
| M8 | Kimi-Audio | B_pitt | conflict | src=primary | 0.7197 | [0.6201, 0.8147] | 146 | 100 | <local data dir>/Desktop/release/overnight2/part3/kimi_pitt_zeroshot_scores.csv |
| M8 | Kimi-Audio | B_pitt | agreement | src=primary | 0.7227 | [0.6560, 0.7896] | 322 | 175 | <local data dir>/Desktop/release/overnight2/part3/kimi_pitt_zeroshot_scores.csv |
| M8 | Kimi-Audio | B_pitt | conflict_minus_agreement | src=primary | -0.0030 | [-0.1134, 0.1065] | 468 | 228 | <local data dir>/Desktop/release/overnight2/part3/kimi_pitt_zeroshot_scores.csv |
| M8 | Kimi-Audio | B_pitt | conflict | src=alt_overnight_copy | 0.7078 | [0.6037, 0.8083] | 146 | 100 | <local data dir>/Desktop/release/overnight/kimi/kimi_pitt_zeroshot_scores.csv |
| M8 | Kimi-Audio | B_pitt | agreement | src=alt_overnight_copy | 0.7014 | [0.6309, 0.7730] | 322 | 175 | <local data dir>/Desktop/release/overnight/kimi/kimi_pitt_zeroshot_scores.csv |
| M8 | Kimi-Audio | B_pitt | conflict_minus_agreement | src=alt_overnight_copy | 0.0064 | [-0.1140, 0.1224] | 468 | 228 | <local data dir>/Desktop/release/overnight/kimi/kimi_pitt_zeroshot_scores.csv |
| M9.AudioFlamingo2.adress2020.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9676 | [0.9664, 0.9689] | 156 | 156 | <local data dir>/paper1_local_runs/af2_results/af2_adress2020.csv |
| M9.AudioFlamingo2.adresso.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9614 | [0.9607, 0.9628] | 237 | 237 | <local data dir>/paper1_local_runs/af2_results/af2_adresso237.csv |
| M9.AudioFlamingo2.edaic.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9392 | [0.9383, 0.9407] | 275 | 275 | <local data dir>/paper1_local_runs/af2_results/af2_edaic275.csv |
| M9.AudioFlamingo2.edaic_conflict.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9453 | [0.9437, 0.9479] | 390 | 74 | <local data dir>/Desktop/release/edaic_conflict_new/af2/af2_edaic_conflict_30s.csv |
| M9.AudioFlamingo2.edaic_conflict_v2.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9423 | [0.9409, 0.9434] | 966 | 138 | scores/edaic_conflict_v2/p14_af2.csv |
| M9.AudioFlamingo2.kcl.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9304 | [0.9285, 0.9341] | 37 | 37 | <local data dir>/paper1_local_runs/af2_results/af2_kcl37.csv |
| M9.AudioFlamingo2.neurovoz.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9412 | [0.9398, 0.9427] | 1270 | 107 | <local data dir>/paper1_local_runs/af2_results/af2_neurovoz1270.csv |
| M9.AudioFlamingo2.pcgita.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9400 | [0.9390, 0.9413] | 1100 | 100 | <local data dir>/paper1_local_runs/af2_results/af2_pcgita1100.csv |
| M9.AudioFlamingo2.pitt.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9626 | [0.9614, 0.9639] | 468 | 228 | <local data dir>/paper1_local_runs/af2_results/af2_pitt468.csv |
| M9.AudioFlamingo3.adress2020.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9457 | [0.9415, 0.9477] | 156 | 156 | <local data dir>/Desktop/release/overnight2/af3_new/af3_adress2020.csv |
| M9.AudioFlamingo3.adresso.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9579 | [0.9541, 0.9614] | 237 | 237 | <local data dir>/Desktop/release/overnight2/af3_new/af3_adresso.csv |
| M9.AudioFlamingo3.edaic.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9272 | [0.9226, 0.9328] | 275 | 275 | <local data dir>/Desktop/release/overnight2/part10/af3_edaic.csv |
| M9.AudioFlamingo3.edaic_conflict.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9332 | [0.9239, 0.9435] | 390 | 74 | <local data dir>/Desktop/release/edaic_conflict_new/af3_edaic_conflict.csv |
| M9.AudioFlamingo3.edaic_conflict_v2.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9311 | [0.9271, 0.9361] | 966 | 138 | scores/edaic_conflict_v2/p14_af3.csv |
| M9.AudioFlamingo3.edaic_full.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9413 | [0.9342, 0.9476] | 275 | 275 | scores/edaic_windows/af3_full.csv |
| M9.AudioFlamingo3.edaic_mid30.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9266 | [0.9192, 0.9333] | 275 | 275 | scores/edaic_windows/af3_mid30.csv |
| M9.AudioFlamingo3.edaic_mid300.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9398 | [0.9334, 0.9441] | 275 | 275 | scores/edaic_windows/af3_mid300.csv |
| M9.AudioFlamingo3.kcl.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9434 | [0.9374, 0.9501] | 37 | 37 | <local data dir>/Desktop/release/overnight2/part3/af3_kcl.csv |
| M9.AudioFlamingo3.neurovoz.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9132 | [0.9076, 0.9177] | 1270 | 107 | <local data dir>/Desktop/release/overnight2/af3_new/af3_neurovoz.csv |
| M9.AudioFlamingo3.pcgita.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.8968 | [0.8921, 0.9013] | 1100 | 100 | <local data dir>/Desktop/release/overnight2/part3/af3_pcgita.csv |
| M9.AudioFlamingo3.pitt.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9487 | [0.9463, 0.9505] | 468 | 228 | <local data dir>/Desktop/release/overnight2/part3/af3_pitt.csv |
| M9.Kimi-Audio.adress2020.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9971 | [0.9967, 0.9976] | 156 | 156 | <local data dir>/Desktop/release/overnight2/kimi_new/kimi_adress2020_zeroshot_scores.csv |
| M9.Kimi-Audio.adresso.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9956 | [0.9953, 0.9962] | 237 | 237 | <local data dir>/Desktop/release/overnight2/kimi_new/kimi_adresso_zeroshot_scores.csv |
| M9.Kimi-Audio.edaic.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9981 | [0.9980, 0.9982] | 275 | 275 | <local data dir>/Desktop/release/overnight2/part3/kimi_edaic_zeroshot_scores.csv |
| M9.Kimi-Audio.edaic_conflict.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9992 | [0.9990, 0.9993] | 390 | 74 | <local data dir>/Desktop/release/edaic_conflict_new/kimi/kimi_edaic_conflict_zeroshot_scores.csv |
| M9.Kimi-Audio.edaic_conflict_v2.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9992 | [0.9991, 0.9994] | 966 | 138 | scores/edaic_conflict_v2/p14_kimi_zeroshot_scores.csv |
| M9.Kimi-Audio.edaic_full.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9983 | [0.9981, 0.9984] | 275 | 275 | scores/pitt_all/kimi_full_zeroshot_scores.csv |
| M9.Kimi-Audio.edaic_mid30.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9991 | [0.9990, 0.9992] | 275 | 275 | scores/edaic_windows/kimi_mid30_zeroshot_scores.csv |
| M9.Kimi-Audio.edaic_mid300.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9987 | [0.9985, 0.9989] | 275 | 275 | scores/edaic_windows/kimi_mid300_zeroshot_scores.csv |
| M9.Kimi-Audio.kcl.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9973 | [0.9970, 0.9977] | 37 | 37 | <local data dir>/Desktop/release/overnight2/kimi_new/kimi_kcl_zeroshot_scores.csv |
| M9.Kimi-Audio.neurovoz.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9961 | [0.9961, 0.9962] | 1270 | 107 | <local data dir>/Desktop/release/overnight2/kimi_new/kimi_neurovoz_zeroshot_scores.csv |
| M9.Kimi-Audio.pcgita.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9959 | [0.9958, 0.9960] | 1100 | 100 | <local data dir>/Desktop/release/overnight2/part3/kimi_pcgita_zeroshot_scores.csv |
| M9.Kimi-Audio.pitt.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9966 | [0.9963, 0.9970] | 468 | 228 | <local data dir>/Desktop/release/overnight2/part3/kimi_pitt_zeroshot_scores.csv |
| M9.Qwen2-Audio.adress2020.transcript_only | median answer mass P(Yes)+P(No) | 0.9424 | [0.9341, 0.9503] | 156 | 156 | <local data dir>/paper1_local_runs/qwen2audio_adress2020_text.csv |
| M9.Qwen2-Audio.adresso.transcript_only | median answer mass P(Yes)+P(No) | 0.9110 | [0.8975, 0.9225] | 237 | 237 | <local data dir>/paper1_local_runs/qwen2audio_adresso_text.csv |
| M9.Qwen2-Audio.conflict.transcript_only | median answer mass P(Yes)+P(No) | 0.9808 | [0.9713, 0.9864] | 390 | 74 | <local data dir>/paper1_local_runs/qwen2audio_conflict_text.csv |
| M9.Qwen2-Audio.edaic.transcript_only | median answer mass P(Yes)+P(No) | 0.9595 | [0.9497, 0.9643] | 275 | 275 | <local data dir>/paper1_local_runs/qwen2audio_edaic30_text.csv |
| M9.Qwen2-Audio.edaic30.transcript_only | median answer mass P(Yes)+P(No) | 0.9595 | [0.9497, 0.9643] | 275 | 275 | <local data dir>/paper1_local_runs/qwen2audio_edaic30_text.csv |
| M9.Qwen2-Audio.edaicfull.transcript_only | median answer mass P(Yes)+P(No) | 0.9664 | [0.9611, 0.9720] | 275 | 275 | <local data dir>/paper1_local_runs/qwen2audio_edaicfull_text.csv |
| M9.Qwen2-Audio.pitt.transcript_only | median answer mass P(Yes)+P(No) | 0.9300 | [0.9236, 0.9362] | 468 | 228 | <local data dir>/paper1_local_runs/qwen2audio_pitt_text.csv |
| M9.Qwen2-Audio.adress2020.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 | [0.9965, 0.9971] | 156 | 156 | <local data dir>/paper1_local_runs/probe2/adress2020_zeroshot_scores.csv |
| M9.Qwen2-Audio.adresso.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9969 | [0.9966, 0.9971] | 237 | 237 | <local data dir>/paper1_local_runs/probe2/adresso_zeroshot_scores.csv |
| M9.Qwen2-Audio.edaic.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9959 | [0.9952, 0.9964] | 275 | 275 | <local data dir>/paper1_local_runs/probe2/edaic_zeroshot_scores.csv |
| M9.Qwen2-Audio.edaic_conflict.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9989 | [0.9985, 0.9992] | 390 | 74 | <local data dir>/Desktop/release/edaic_conflict_new/q2a_edaic_conflict_zeroshot_scores.csv |
| M9.Qwen2-Audio.edaic_conflict_v2.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9989 | [0.9987, 0.9992] | 966 | 138 | scores/edaic_conflict_v2/p14_q2a_zeroshot_scores.csv |
| M9.Qwen2-Audio.edaic_full.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9970 | [0.9965, 0.9975] | 275 | 275 | scores/edaic_windows/q2a_full_zeroshot_scores.csv |
| M9.Qwen2-Audio.edaic_mid30.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9990 | [0.9987, 0.9993] | 275 | 275 | scores/edaic_windows/q2a_mid30_zeroshot_scores.csv |
| M9.Qwen2-Audio.edaic_mid300.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9972 | [0.9969, 0.9976] | 275 | 275 | scores/edaic_windows/q2a_mid300_zeroshot_scores.csv |
| M9.Qwen2-Audio.kcl.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9991 | [0.9989, 0.9994] | 37 | 37 | <local data dir>/paper1_local_runs/probe2/kcl_zeroshot_scores.csv |
| M9.Qwen2-Audio.neurovoz.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9992 | [0.9992, 0.9992] | 1270 | 107 | <local data dir>/paper1_local_runs/probe2/neurovoz_zeroshot_scores.csv |
| M9.Qwen2-Audio.pcgita.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9994 | [0.9994, 0.9994] | 1100 | 1100 | <local data dir>/paper1_local_runs/probe2/pcgita_zeroshot_scores.csv |
| M9.Qwen2-Audio.pitt.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 | [0.9965, 0.9969] | 468 | 228 | <local data dir>/paper1_local_runs/probe2/pitt_zeroshot_scores.csv |
| M9.Qwen2.5-Omni.adress2020.transcript_only | median answer mass P(Yes)+P(No) | 0.9968 | [0.9967, 0.9969] | 156 | 156 | <local data dir>/Desktop/release/overnight2/text_new/o25_adress2020_text.csv |
| M9.Qwen2.5-Omni.adresso.transcript_only | median answer mass P(Yes)+P(No) | 0.9975 | [0.9974, 0.9975] | 237 | 237 | <local data dir>/Desktop/release/overnight2/text_new/o25_adresso_text.csv |
| M9.Qwen2.5-Omni.conflict.transcript_only | median answer mass P(Yes)+P(No) | 0.9986 | [0.9984, 0.9988] | 390 | 74 | <local data dir>/Desktop/release/overnight2/text_new/o25_conflict_text.csv |
| M9.Qwen2.5-Omni.edaic.transcript_only | median answer mass P(Yes)+P(No) | 0.9986 | [0.9985, 0.9986] | 275 | 275 | <local data dir>/Desktop/release/overnight2/text_new/o25_edaic30_text.csv |
| M9.Qwen2.5-Omni.edaic30.transcript_only | median answer mass P(Yes)+P(No) | 0.9986 | [0.9985, 0.9986] | 275 | 275 | <local data dir>/Desktop/release/overnight2/text_new/o25_edaic30_text.csv |
| M9.Qwen2.5-Omni.edaicfull.transcript_only | median answer mass P(Yes)+P(No) | 0.9989 | [0.9988, 0.9990] | 275 | 275 | <local data dir>/Desktop/release/overnight2/text_new/o25_edaicfull_text.csv |
| M9.Qwen2.5-Omni.kcl.transcript_only | median answer mass P(Yes)+P(No) | 0.9980 | [0.9979, 0.9981] | 37 | 37 | <local data dir>/Desktop/release/overnight2/text_new/o25_kcl_text.csv |
| M9.Qwen2.5-Omni.neurovoz.transcript_only | median answer mass P(Yes)+P(No) | 0.9973 | [0.9973, 0.9974] | 1270 | 107 | <local data dir>/Desktop/release/overnight2/text_new/o25_neurovoz_text.csv |
| M9.Qwen2.5-Omni.pcgita.transcript_only | median answer mass P(Yes)+P(No) | 0.9978 | [0.9978, 0.9978] | 1100 | 100 | <local data dir>/Desktop/release/overnight2/text_new/o25_pcgita_text.csv |
| M9.Qwen2.5-Omni.pitt.transcript_only | median answer mass P(Yes)+P(No) | 0.9966 | [0.9965, 0.9968] | 468 | 228 | <local data dir>/Desktop/release/overnight2/text_new/o25_pitt_text.csv |
| M9.Qwen2.5-Omni.pitt_all.transcript_only | median answer mass P(Yes)+P(No) | 0.9966 | [0.9965, 0.9966] | 708 | 266 | scores/pitt_all/o25_pitt_all_text.csv |
| M9.Qwen2.5-Omni.adress2020.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 | [0.9965, 0.9968] | 156 | 156 | <local data dir>/paper1_local_runs/omni_final/omni_adress2020_zeroshot_scores.csv |
| M9.Qwen2.5-Omni.adresso.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 | [0.9966, 0.9968] | 237 | 237 | <local data dir>/paper1_local_runs/omni_final/omni_adresso_zeroshot_scores.csv |
| M9.Qwen2.5-Omni.edaic.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9966 | [0.9965, 0.9967] | 275 | 275 | <local data dir>/paper1_local_runs/omni_final/omni_edaic_zeroshot_scores.csv |
| M9.Qwen2.5-Omni.edaic_conflict.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 | [0.9965, 0.9968] | 390 | 74 | <local data dir>/Desktop/release/edaic_conflict_new/o25_edaic_conflict_zeroshot_scores.csv |
| M9.Qwen2.5-Omni.edaic_conflict_v2.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 | [0.9966, 0.9968] | 966 | 138 | scores/edaic_conflict_v2/p14_o25_zeroshot_scores.csv |
| M9.Qwen2.5-Omni.edaic_full.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9962 | [0.9960, 0.9964] | 275 | 275 | scores/edaic_windows/o25_full_zeroshot_scores.csv |
| M9.Qwen2.5-Omni.edaic_mid30.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9958 | [0.9956, 0.9960] | 275 | 275 | scores/edaic_windows/o25_mid30_zeroshot_scores.csv |
| M9.Qwen2.5-Omni.edaic_mid300.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9958 | [0.9956, 0.9962] | 275 | 275 | scores/edaic_windows/o25_mid300_zeroshot_scores.csv |
| M9.Qwen2.5-Omni.kcl.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9888 | [0.9881, 0.9903] | 37 | 37 | <local data dir>/paper1_local_runs/omni_final/omni_kcl_zeroshot_scores.csv |
| M9.Qwen2.5-Omni.neurovoz.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9952 | [0.9952, 0.9953] | 1270 | 107 | <local data dir>/paper1_local_runs/omni_final/omni_neurovoz_zeroshot_scores.csv |
| M9.Qwen2.5-Omni.pcgita.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9952 | [0.9951, 0.9953] | 1100 | 100 | <local data dir>/paper1_local_runs/omni_final/omni_pcgita_zeroshot_scores.csv |
| M9.Qwen2.5-Omni.pitt.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9966 | [0.9965, 0.9967] | 468 | 228 | <local data dir>/paper1_local_runs/omni_final/omni_pitt_zeroshot_scores.csv |
| M9.Qwen2.5-Omni.pitt_all.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9967 | [0.9966, 0.9967] | 708 | 266 | scores/pitt_all/o25_pitt_all_zeroshot_scores.csv |
| M9.Qwen3-Omni-30B-A3B.adress2020.transcript_only | median answer mass P(Yes)+P(No) | 0.9997 | [0.9997, 0.9998] | 156 | 156 | <local data dir>/Desktop/release/overnight2/text_new/q3o_adress2020_text.csv |
| M9.Qwen3-Omni-30B-A3B.adresso.transcript_only | median answer mass P(Yes)+P(No) | 0.9998 | [0.9997, 0.9998] | 237 | 237 | <local data dir>/Desktop/release/overnight2/text_new/q3o_adresso_text.csv |
| M9.Qwen3-Omni-30B-A3B.conflict.transcript_only | median answer mass P(Yes)+P(No) | 0.9999 | [0.9999, 0.9999] | 390 | 74 | <local data dir>/Desktop/release/overnight2/text_new/q3o_conflict_text.csv |
| M9.Qwen3-Omni-30B-A3B.edaic.transcript_only | median answer mass P(Yes)+P(No) | 0.9999 | [0.9998, 0.9999] | 275 | 275 | <local data dir>/Desktop/release/overnight2/text_new/q3o_edaic30_text.csv |
| M9.Qwen3-Omni-30B-A3B.edaic30.transcript_only | median answer mass P(Yes)+P(No) | 0.9999 | [0.9998, 0.9999] | 275 | 275 | <local data dir>/Desktop/release/overnight2/text_new/q3o_edaic30_text.csv |
| M9.Qwen3-Omni-30B-A3B.edaicfull.transcript_only | median answer mass P(Yes)+P(No) | 0.9999 | [0.9999, 0.9999] | 275 | 275 | <local data dir>/Desktop/release/overnight2/text_new/q3o_edaicfull_text.csv |
| M9.Qwen3-Omni-30B-A3B.kcl.transcript_only | median answer mass P(Yes)+P(No) | 0.9997 | [0.9996, 0.9998] | 37 | 37 | <local data dir>/Desktop/release/overnight2/text_new/q3o_kcl_text.csv |
| M9.Qwen3-Omni-30B-A3B.neurovoz.transcript_only | median answer mass P(Yes)+P(No) | 0.9996 | [0.9996, 0.9996] | 1270 | 107 | <local data dir>/Desktop/release/overnight2/text_new/q3o_neurovoz_text.csv |
| M9.Qwen3-Omni-30B-A3B.pcgita.transcript_only | median answer mass P(Yes)+P(No) | 0.9997 | [0.9997, 0.9997] | 1100 | 100 | <local data dir>/Desktop/release/overnight2/text_new/q3o_pcgita_text.csv |
| M9.Qwen3-Omni-30B-A3B.pitt.transcript_only | median answer mass P(Yes)+P(No) | 0.9997 | [0.9997, 0.9997] | 468 | 228 | <local data dir>/Desktop/release/overnight2/text_new/q3o_pitt_text.csv |
| M9.Qwen3-Omni-30B-A3B.adress2020.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9995 | [0.9995, 0.9996] | 156 | 156 | <local data dir>/Desktop/release/overnight2/q3o_new/q3o_adress2020_zeroshot_scores.csv |
| M9.Qwen3-Omni-30B-A3B.adresso.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9995 | [0.9995, 0.9996] | 237 | 237 | <local data dir>/Desktop/release/overnight2/q3o_new/q3o_adresso_zeroshot_scores.csv |
| M9.Qwen3-Omni-30B-A3B.edaic.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9997 | [0.9997, 0.9997] | 275 | 275 | <local data dir>/Desktop/release/overnight2/q3o_new/q3o_edaic_zeroshot_scores.csv |
| M9.Qwen3-Omni-30B-A3B.edaic_conflict.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9996 | [0.9996, 0.9997] | 390 | 74 | <local data dir>/Desktop/release/edaic_conflict_new/q3o_edaic_conflict_zeroshot_scores.csv |
| M9.Qwen3-Omni-30B-A3B.edaic_conflict_v2.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9997 | [0.9996, 0.9997] | 966 | 138 | scores/edaic_conflict_v2/p14_q3o_zeroshot_scores.csv |
| M9.Qwen3-Omni-30B-A3B.edaic_full.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9992 | [0.9991, 0.9993] | 275 | 275 | scores/edaic_windows/q3o_full_zeroshot_scores.csv |
| M9.Qwen3-Omni-30B-A3B.edaic_mid30.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9995 | [0.9995, 0.9996] | 275 | 275 | scores/edaic_windows/q3o_mid30_zeroshot_scores.csv |
| M9.Qwen3-Omni-30B-A3B.edaic_mid300.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9995 | [0.9994, 0.9996] | 275 | 275 | scores/edaic_windows/q3o_mid300_zeroshot_scores.csv |
| M9.Qwen3-Omni-30B-A3B.kcl.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9996 | [0.9995, 0.9996] | 37 | 37 | <local data dir>/Desktop/release/overnight2/q3o_new/q3o_kcl_zeroshot_scores.csv |
| M9.Qwen3-Omni-30B-A3B.neurovoz.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9997 | [0.9997, 0.9997] | 1270 | 107 | <local data dir>/Desktop/release/overnight2/q3o_new/q3o_neurovoz_zeroshot_scores.csv |
| M9.Qwen3-Omni-30B-A3B.pcgita.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9997 | [0.9997, 0.9997] | 1100 | 100 | <local data dir>/Desktop/release/overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv |
| M9.Qwen3-Omni-30B-A3B.pitt.zero_shot_answer | median answer mass P(Yes)+P(No) | 0.9995 | [0.9995, 0.9995] | 468 | 228 | <local data dir>/Desktop/release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv |
| M9.MINIMUM | minimum median answer mass across the published grid (Audio Flamingo 3 / pcgita / zero shot answer) | 0.8968 | [0.8921, 0.9013] | 1100 | 100 | <local data dir>/Desktop/release/overnight2/part3/af3_pcgita.csv |
| M9.SWEEPMIN | minimum median answer mass across every mass-bearing file on disk (Audio Flamingo 2 / kcl30b) | 0.8662 | NA | 37 | 37 | <local data dir>/paper1_local_runs/af2_kcl30b_audio.csv |
| M9.CELLS | master_lookup cells with an answer mass recorded | 98 | NA | 221 |  | scores/part16/M/M9_answer_mass.csv |
| M9.FILES | per-clip score files on disk with an answer mass recorded | 307 | NA | 608 |  | scores/part16/M/M9_inventory.csv |
| MEDAIC_full_answer | Omni zero-shot answer AUC, full window | 0.8285 | [0.7708, 0.8827] | 275 | 275 | edaic_rerun/variants/o25_full_zeroshot_scores.csv |
| MEDAIC_full_sft | projector fine-tune OOF AUC, full window | 0.6421 | [0.5593, 0.7188] | 275 | 275 | edaic_rerun/variants/sft_full_oof.csv |
| MEDAIC_full_sft_minus_answer | PAIRED projector fine-tune OOF minus answer, full window (excl0=True) | -0.1863 | [-0.2696, -0.1098] | 275 | 275 | edaic_rerun/variants/sft_full_oof.csv |
| MEDAIC_full_text | transcript-only readout AUC, full window | 0.8230 | [0.7651, 0.8705] | 275 | 275 | overnight2/text_new/o25_edaicfull_text.csv |
| MEDAIC_full_text_minus_answer | PAIRED transcript-only readout minus answer, full window (excl0=False) | -0.0055 | [-0.0566, 0.0431] | 275 | 275 | overnight2/text_new/o25_edaicfull_text.csv |
| MEDAIC_full_llm_nested_NOCI | LM probe nested mean, full window (json aggregate, NOT recomputed, no per-clip file) | 0.6853 | NA | 275 | 275 | edaic_rerun/variants/o25_full_nested_repeats.json |
| MEDAIC_full_llm_minus_answer_POINT | POINT-ONLY LM probe minus answer, full window (no interval possible) | -0.1432 | NA | 275 | 275 | edaic_rerun/variants/o25_full_nested_repeats.json |
| MEDAIC_full_enc_nested_NOCI | encoder probe nested mean, full window (json aggregate, NOT recomputed, no per-clip file) | 0.5930 | NA | 275 | 275 | edaic_rerun/variants/o25_full_nested_repeats.json |
| MEDAIC_full_enc_minus_answer_POINT | POINT-ONLY encoder probe minus answer, full window (no interval possible) | -0.2355 | NA | 275 | 275 | edaic_rerun/variants/o25_full_nested_repeats.json |
| MEDAIC_full_ans_nested_NOCI | answer-state probe nested mean, full window (json aggregate, NOT recomputed, no per-clip file) | 0.7542 | NA | 275 | 275 | edaic_rerun/variants/o25_full_nested_repeats.json |
| MEDAIC_full_ans_minus_answer_POINT | POINT-ONLY answer-state probe minus answer, full window (no interval possible) | -0.0743 | NA | 275 | 275 | edaic_rerun/variants/o25_full_nested_repeats.json |
| MEDAIC_full_proj_nested_NOCI | projector probe nested mean, full window (json aggregate, NOT recomputed, no per-clip file) | 0.5293 | NA | 275 | 275 | edaic_rerun/variants/o25_full_nested_repeats.json |
| MEDAIC_full_proj_minus_answer_POINT | POINT-ONLY projector probe minus answer, full window (no interval possible) | -0.2992 | NA | 275 | 275 | edaic_rerun/variants/o25_full_nested_repeats.json |
| MEDAIC_w30_enc | encoder probe nested OOF AUC, 30 s window | 0.6324 | [0.5520, 0.7104] | 275 | 275 | omni_final/omni_edaic_enc_nested_oof.csv |
| MEDAIC_w30_enc_minus_answer | PAIRED encoder probe minus answer, 30 s window (excl0=False) | -0.0297 | [-0.1291, 0.0752] | 275 | 275 | omni_final/omni_edaic_enc_nested_oof.csv |
| MEDAIC_w30_llm | LM probe nested OOF AUC, 30 s window | 0.5065 | [0.4221, 0.5856] | 275 | 275 | omni_final/omni_edaic_llm_nested_oof.csv |
| MEDAIC_w30_llm_minus_answer | PAIRED LM probe minus answer, 30 s window (excl0=True) | -0.1555 | [-0.2536, -0.0577] | 275 | 275 | omni_final/omni_edaic_llm_nested_oof.csv |
| MEDAIC_w30_ans | answer-state probe nested OOF AUC, 30 s window | 0.6401 | [0.5581, 0.7185] | 275 | 275 | omni_final/omni_edaic_ans_nested_oof.csv |
| MEDAIC_w30_ans_minus_answer | PAIRED answer-state probe minus answer, 30 s window (excl0=False) | -0.0219 | [-0.1053, 0.0663] | 275 | 275 | omni_final/omni_edaic_ans_nested_oof.csv |
| MEDAIC_w30_proj | projector probe nested OOF AUC, 30 s window | 0.5182 | [0.4411, 0.5999] | 275 | 275 | omni_final/omni_edaic_proj_nested_oof.csv |
| MEDAIC_w30_proj_minus_answer | PAIRED projector probe minus answer, 30 s window (excl0=True) | -0.1438 | [-0.2490, -0.0320] | 275 | 275 | omni_final/omni_edaic_proj_nested_oof.csv |
| MEDAIC_w30_answer | Omni zero-shot answer AUC, 30 s window | 0.6620 | [0.5831, 0.7315] | 275 | 275 | omni_final/omni_edaic_zeroshot_scores.csv |
| MEDAIC_answer_full_minus_30s | PAIRED answer AUC full minus 30 s, same 275 speakers (excl0=True) | 0.1664 | [0.0848, 0.2492] | 275 | 275 | o25_full_zeroshot_scores.csv + omni_edaic_zeroshot_scores.csv |
| 1a_o25_text_conflict_auc | text-prompt conflict-arm AUC, o25 | 0.1498 | [0.1131, 0.1887] | 483 | 138 | scores/part16/POD1/p14_o25_1a_joined.csv |
| 1a_o25_text_agreement_auc | text-prompt agreement-arm AUC, o25 | 0.9596 | [0.9306, 0.9823] | 483 | 138 | scores/part16/POD1/p14_o25_1a_joined.csv |
| 1a_o25_pearson_text_audio | Pearson r transcript p_yes vs audio p_yes, o25 | 0.8442 | NA | 966 | 138 | scores/part16/POD1/p14_o25_1a_joined.csv |
| 1a_o25_same_decision | share text decision == audio decision at 0.5, o25 | 0.8634 | NA | 966 | 138 | scores/part16/POD1/p14_o25_1a_joined.csv |
| 1a_q2a_text_conflict_auc | text-prompt conflict-arm AUC, q2a | 0.2359 | [0.1851, 0.2886] | 483 | 138 | scores/part16/POD1/p14_q2a_1a_joined.csv |
| 1a_q2a_text_agreement_auc | text-prompt agreement-arm AUC, q2a | 0.9081 | [0.8515, 0.9547] | 483 | 138 | scores/part16/POD1/p14_q2a_1a_joined.csv |
| 1a_q2a_pearson_text_audio | Pearson r transcript p_yes vs audio p_yes, q2a | 0.5155 | NA | 966 | 138 | scores/part16/POD1/p14_q2a_1a_joined.csv |
| 1a_q2a_same_decision | share text decision == audio decision at 0.5, q2a | 0.7122 | NA | 966 | 138 | scores/part16/POD1/p14_q2a_1a_joined.csv |
| 1f_share_yesno_all | share first generated word is Yes/No, all clips | 1.0000 | NA | 966 | 138 | scores/part16/POD1/p14_o25_greedy.csv |
| 1f_agree_gen_logit_all | agreement generated word vs logit decision, Yes/No clips | 0.9865 | NA | 966 | 138 | scores/part16/POD1/p14_o25_greedy.csv |
| 1f_share_yesno_conflict | share first generated word is Yes/No, conflict arm | 1.0000 | NA | 483 | 138 | scores/part16/POD1/p14_o25_greedy.csv |
| 1f_agree_gen_logit_conflict | agreement generated word vs logit decision, conflict arm | 0.9917 | NA | 483 | 138 | scores/part16/POD1/p14_o25_greedy.csv |
| 1f_share_yesno_agreement | share first generated word is Yes/No, agreement arm | 1.0000 | NA | 483 | 138 | scores/part16/POD1/p14_o25_greedy.csv |
| 1f_agree_gen_logit_agreement | agreement generated word vs logit decision, agreement arm | 0.9814 | NA | 483 | 138 | scores/part16/POD1/p14_o25_greedy.csv |
| VALIDATION_conflict_auc | pipeline validation, rebuilt 0.70 set, original wording (reproduces Table 2 audio) conflict arm AUC | 0.2218 | [0.1829, 0.2632] | 483 | 138 | scores/part16/POD1/p14_thr070_check.csv |
| VALIDATION_agreement_auc | pipeline validation, rebuilt 0.70 set, original wording (reproduces Table 2 audio) agreement arm AUC | 0.9482 | [0.9039, 0.9828] | 483 | 138 | scores/part16/POD1/p14_thr070_check.csv |
| 1d_conflict_auc | 1d PHQ-8>=10 cutoff rebuild, Omni audio conflict arm AUC | 0.1759 | [0.1448, 0.2095] | 924 | 237 | scores/part16/POD1/p14_phq10_omni.csv |
| 1d_agreement_auc | 1d PHQ-8>=10 cutoff rebuild, Omni audio agreement arm AUC | 0.9311 | [0.9028, 0.9568] | 924 | 237 | scores/part16/POD1/p14_phq10_omni.csv |
| 1e060_conflict_auc | 1e sentiment threshold 0.60, Omni audio conflict arm AUC | 0.2619 | [0.2188, 0.3075] | 646 | 147 | scores/part16/POD1/p14_thr060_omni.csv |
| 1e060_agreement_auc | 1e sentiment threshold 0.60, Omni audio agreement arm AUC | 0.9406 | [0.9026, 0.9711] | 646 | 147 | scores/part16/POD1/p14_thr060_omni.csv |
| 1e080_conflict_auc | 1e sentiment threshold 0.80, Omni audio conflict arm AUC | 0.1936 | [0.1501, 0.2438] | 281 | 104 | scores/part16/POD1/p14_thr080_omni.csv |
| 1e080_agreement_auc | 1e sentiment threshold 0.80, Omni audio agreement arm AUC | 0.9594 | [0.9190, 0.9895] | 281 | 104 | scores/part16/POD1/p14_thr080_omni.csv |
| 1b_p2_conflict_auc | 1b prompt wording 2, Omni audio conflict arm AUC | 0.3439 | [0.2900, 0.3979] | 483 | 138 | scores/part16/POD1/p14_o25_audio_p2.csv |
| 1b_p2_agreement_auc | 1b prompt wording 2, Omni audio agreement arm AUC | 0.9119 | [0.8505, 0.9594] | 483 | 138 | scores/part16/POD1/p14_o25_audio_p2.csv |
| 1b_p3_conflict_auc | 1b prompt wording 3, Omni audio conflict arm AUC | 0.2392 | [0.1940, 0.2865] | 483 | 138 | scores/part16/POD1/p14_o25_audio_p3.csv |
| 1b_p3_agreement_auc | 1b prompt wording 3, Omni audio agreement arm AUC | 0.9442 | [0.8926, 0.9821] | 483 | 138 | scores/part16/POD1/p14_o25_audio_p3.csv |
| 1b_p2_conflict_delta | 1b wording 2 conflict AUC change vs original wording | 0.1221 | NA | 483 | 138 | scores/part16/POD1/p14_o25_audio_p2.csv |
| 1b_p2_agreement_delta | 1b wording 2 agreement AUC change vs original wording | -0.0363 | NA | 483 | 138 | scores/part16/POD1/p14_o25_audio_p2.csv |
| 1b_p3_conflict_delta | 1b wording 3 conflict AUC change vs original wording | 0.0174 | NA | 483 | 138 | scores/part16/POD1/p14_o25_audio_p3.csv |
| 1b_p3_agreement_delta | 1b wording 3 agreement AUC change vs original wording | -0.0040 | NA | 483 | 138 | scores/part16/POD1/p14_o25_audio_p3.csv |
| 1b_max_abs_change | 1b maximum absolute AUC change across both wordings and both arms | 0.1221 | NA | 966 | 138 | scores/part16/POD1/p14_o25_audio_p2.csv |
| 1c_conflict_auc | 1c sft_full projector RETRAIN, out-of-fold by speaker, Omni audio conflict arm AUC | 0.2283 | [0.1650, 0.2980] | 483 | 138 | scores/part16/POD1/p14_o25_sftfull.csv |
| 1c_agreement_auc | 1c sft_full projector RETRAIN, out-of-fold by speaker, Omni audio agreement arm AUC | 0.7930 | [0.7062, 0.8669] | 483 | 138 | scores/part16/POD1/p14_o25_sftfull.csv |
| 1c_edaic_oof_auc_retrain | 1c RETRAIN E-DAIC out-of-fold AUC (original sft_full was 0.6421; input audio differs) | 0.5315 | NA | 275 | 275 | scores/part16/POD1/sft1c_edaic_oof.csv |
| 1e_conflict_auc_range | 1e conflict AUC range across sentiment 0.60/0.70/0.80 | 0.0683 | NA | 0 | 0 | scores/part16/POD1/p14_thr060_omni.csv |
| P16.POD2.V1 | Qwen2.5-Omni Pitt projector-FT OOF AUC (recomputed from paper file, rank formula) | 0.7673 | [0.7086, 0.8231] | 468 | 228 | <local data dir>/Desktop/release/omni_final/omnisft_pitt_oof.csv |
| P16.POD2.V2 | Qwen2.5-Omni Pitt zero-shot AUC (recomputed from paper file, rank formula) | 0.6578 | [0.6033, 0.7125] | 468 | 228 | <local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv |
| P16.POD2.V3 | projector-FT conflict-arm AUC (recomputed) | 0.5189 | [0.4151, 0.6300] | 146 | 100 | <local data dir>/Desktop/release/omni_final/omnisft_pitt_oof.csv |
| P16.POD2.V4 | projector-FT agreement-arm AUC (recomputed) | 0.8593 | [0.8063, 0.9069] | 322 | 175 | <local data dir>/Desktop/release/omni_final/omnisft_pitt_oof.csv |
| P16.POD2.V5 | projector-FT paired conflict-minus-agreement | -0.3404 | [-0.4468, -0.2262] | 468 | 228 | <local data dir>/Desktop/release/omni_final/omnisft_pitt_oof.csv |
| P16.POD2.X1 | PRE-EXISTING LoRA(r8,qkvo) Pitt OOF AUC, recomputed from file | 0.8202 | [0.7690, 0.8679] | 468 | 228 | <local data dir>/Desktop/release/omni_final/abl_pitt_oof.csv |
| P16.POD2.X2 | PRE-EXISTING LoRA conflict-arm AUC, recomputed | 0.5884 | [0.4680, 0.7093] | 146 | 100 | <local data dir>/Desktop/release/omni_final/abl_pitt_oof.csv |
| P16.POD2.X3 | PRE-EXISTING LoRA agreement-arm AUC, recomputed | 0.9011 | [0.8592, 0.9386] | 322 | 175 | <local data dir>/Desktop/release/omni_final/abl_pitt_oof.csv |
| P16.POD2.X4 | PRE-EXISTING LoRA paired conflict-minus-agreement | -0.3126 | [-0.4294, -0.1911] | 468 | 228 | <local data dir>/Desktop/release/omni_final/abl_pitt_oof.csv |
| P16.POD2.P1 | LoRA r8 a16 d0.05 on q,k,v,o of all 28 LM layers: trainable params | 5046272 | NA | 468 | 228 | scores/part16/POD2/lora_pitt_params.json |
| P16.POD2.P2 | LoRA trainable as % of full Qwen2.5-Omni-7B (10732225408 params) | 0.04702 | NA | 468 | 228 | scores/part16/POD2/lora_pitt_params.json |
| P16.POD2.P3 | projector-only (thinker.audio_tower.proj) trainable params | 4591104 | NA | 468 | 228 | scores/part16/POD2/lora_pitt_params.json |
| P16.POD2.P4 | projector trainable as % of full Qwen2.5-Omni-7B | 0.042779 | NA | 468 | 228 | scores/part16/POD2/lora_pitt_params.json |
| P16.POD2.L1 | POD2 fresh LoRA(r8,a16,d0.05,qkvo all 28 LM layers) Qwen2.5-Omni Pitt OOF AUC | 0.8250 | [0.7735, 0.8729] | 468 | 228 | scores/part16/POD2/lora_pitt_oof.csv |
| P16.POD2.L2 | POD2 fresh LoRA OOF AUC, conflict arm | 0.5520 | [0.4348, 0.6833] | 146 | 100 | scores/part16/POD2/lora_pitt_oof.csv |
| P16.POD2.L3 | POD2 fresh LoRA OOF AUC, agreement arm | 0.9199 | [0.8846, 0.9513] | 322 | 175 | scores/part16/POD2/lora_pitt_oof.csv |
| P16.POD2.L4 | POD2 fresh LoRA paired conflict-minus-agreement | -0.3679 | [-0.4861, -0.2433] | 468 | 228 | scores/part16/POD2/lora_pitt_oof.csv |
| P16.POD2.L5 | POD2 LoRA training wall-clock, min per fold (0..4) | 4.13;4.07;4.05;4.03;4.00 | NA | 468 | 228 | scores/part16/POD2/lora_pitt_lora.json |
| P16.POD2.L6 | POD2 LoRA peak GPU memory allocated, GB | 21.013 | NA | 468 | 228 | scores/part16/POD2/lora_pitt_lora.json |
| P16.POD2.L7 | POD2 fresh LoRA OOF AUC using the paper multi-variant p_yes (Yes/yes/YES vs No/no/NO, bare+leading-space) | 0.8220 | [0.7702, 0.8711] | 468 | 228 | scores/part16/POD2/lora_pitt_oof.csv |
| P16.POD2.L8 | POD2 fresh LoRA median answer_mass at the first answer position | 0.998774 | NA | 468 | 228 | scores/part16/POD2/lora_pitt_oof.csv |
| P16.POD2.D1 | paired delta: LoRA OOF AUC minus projector-FT OOF AUC (same 468 clips, same speakers) | 0.0577 | [0.0154, 0.1044] | 468 | 228 | scores/part16/POD2/lora_pitt_oof.csv + omni_final/omnisft_pitt_oof.csv |
| P16.POD2.D2 | paired delta: LoRA OOF AUC minus zero-shot AUC | 0.1672 | [0.1071, 0.2264] | 468 | 228 | scores/part16/POD2/lora_pitt_oof.csv + omni_final/omni_pitt_zeroshot_scores.csv |
| P16.POD2.D3 | paired delta: projector-FT AUC minus zero-shot AUC (both read-only paper files) | 0.1095 | [0.0454, 0.1728] | 468 | 228 | omni_final/omnisft_pitt_oof.csv + omni_final/omni_pitt_zeroshot_scores.csv |
| 3b_all_cos | Q3O Pitt readout-direction cosine, all clips (random floor 0.0177 [0.0008,0.0498]) | 0.0051 | NA | 468 | 228 | scores/part16/POD3/readout_direction_q3o_by_arm.csv |
| 3b_all_auc_probe | Q3O Pitt answer-state probe AUC, all clips | 0.8307 | [0.7833, 0.8741] | 468 | 228 | scores/part16/POD3/readout_direction_q3o_by_arm.csv |
| 3b_all_auc_d | Q3O Pitt AUC of the readout direction alone, all clips | 0.7656 | [0.7088, 0.8179] | 468 | 228 | scores/part16/POD3/readout_direction_q3o_by_arm.csv |
| 3b_all_auc_wo_d | Q3O Pitt probe AUC with the readout direction removed, all clips | 0.8305 | [0.7826, 0.8737] | 468 | 228 | scores/part16/POD3/readout_direction_q3o_by_arm.csv |
| 3b_conflict_cos | Q3O Pitt readout-direction cosine, CONFLICT arm (random floor 0.0177 [0.0008,0.0516]) | -0.0067 | NA | 146 | 100 | scores/part16/POD3/readout_direction_q3o_by_arm.csv |
| 3b_conflict_auc_probe | Q3O Pitt answer-state probe AUC, CONFLICT arm | 0.6145 | [0.5021, 0.7246] | 146 | 100 | scores/part16/POD3/readout_direction_q3o_by_arm.csv |
| 3b_conflict_auc_d | Q3O Pitt AUC of the readout direction alone, CONFLICT arm | 0.6029 | [0.4965, 0.7025] | 146 | 100 | scores/part16/POD3/readout_direction_q3o_by_arm.csv |
| 3b_conflict_auc_wo_d | Q3O Pitt probe AUC with the readout direction removed, CONFLICT arm | 0.6196 | [0.5094, 0.7283] | 146 | 100 | scores/part16/POD3/readout_direction_q3o_by_arm.csv |
| 3b_agreement_cos | Q3O Pitt readout-direction cosine, AGREEMENT arm (random floor 0.0172 [0.0007,0.0480]) | 0.0086 | NA | 322 | 175 | scores/part16/POD3/readout_direction_q3o_by_arm.csv |
| 3b_agreement_auc_probe | Q3O Pitt answer-state probe AUC, AGREEMENT arm | 0.9349 | [0.9034, 0.9614] | 322 | 175 | scores/part16/POD3/readout_direction_q3o_by_arm.csv |
| 3b_agreement_auc_d | Q3O Pitt AUC of the readout direction alone, AGREEMENT arm | 0.8264 | [0.7741, 0.8731] | 322 | 175 | scores/part16/POD3/readout_direction_q3o_by_arm.csv |
| 3b_agreement_auc_wo_d | Q3O Pitt probe AUC with the readout direction removed, AGREEMENT arm | 0.9341 | [0.9021, 0.9604] | 322 | 175 | scores/part16/POD3/readout_direction_q3o_by_arm.csv |
| 3a_pitt_zeroshot_pod | Q3O Pitt zero-shot answer AUC, THIS pod (rank formula, deterministic, reproduced twice bit-exact) | 0.7673 | NA | 468 | 228 | scores/part16/POD3/q3o_pitt_zeroshot_scores.csv |
| 3a_pitt_zeroshot_shipped | Q3O Pitt zero-shot answer AUC, SHIPPED file recomputed here with the rank formula | 0.7619 | NA | 468 | 228 | <local data dir>/Desktop/release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv |
| 3a_pitt_enc_nested | Q3O Pitt ENCODER nested probe AUC (5-repeat mean, layer chosen inside the training folds) | 0.8122 | [0.7613, 0.8625] | 468 | 228 | scores/part16/POD3/q3o_pitt_encoder_perlayer.csv |
| 3a_pitt_enc_best_stage | Q3O Pitt ENCODER best single stage = layer 30 (descriptive max, oracle-selected) | 0.8276 | [0.7765, 0.8760] | 468 | 228 | scores/part16/POD3/q3o_pitt_encoder_perlayer.csv |
| 3a_pitt_proj_nested | Q3O Pitt PROJECTOR probe AUC (single point, no layer choice) | 0.7820 | [0.7228, 0.8368] | 468 | 228 | scores/part16/POD3/q3o_pitt_proj_perlayer.csv |
| 3a_pitt_llm_nested | Q3O Pitt LANGUAGE MODEL nested probe AUC (5-repeat mean, layer chosen inside the training folds) | 0.8445 | [0.8003, 0.8878] | 468 | 228 | scores/part16/POD3/q3o_pitt_llm_perlayer.csv |
| 3a_pitt_llm_best_stage | Q3O Pitt LANGUAGE MODEL best single stage = thinker layer 13 (descriptive max, oracle-selected) | 0.8634 | [0.8179, 0.9062] | 468 | 228 | scores/part16/POD3/q3o_pitt_llm_perlayer.csv |
| 3a_pitt_ans_nested | Q3O Pitt ANSWER STATE nested probe AUC (5-repeat mean, stage chosen inside the training folds) | 0.8237 | [0.7827, 0.8641] | 468 | 228 | scores/part16/POD3/q3o_pitt_ans_perlayer.csv |
| 3a_pitt_ans_best_stage | Q3O Pitt ANSWER STATE best single stage = thinker layer 26 (descriptive max, oracle-selected) | 0.8485 | [0.8069, 0.8882] | 468 | 228 | scores/part16/POD3/q3o_pitt_ans_perlayer.csv |
| 3a_pcgita_zeroshot_pod | Q3O PC-GITA zero-shot answer AUC, THIS pod (re-extracted from original audio) | 0.6015 | NA | 1100 | 100 | scores/part16/POD3/q3o_pcgita_zeroshot_scores.csv |
| 3a_pcgita_zeroshot_shipped | Q3O PC-GITA zero-shot answer AUC, SHIPPED file recomputed here | 0.5967 | NA | 1100 | 100 | <local data dir>/Desktop/release/overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv |
| 3a_attn_sdpa | Q3O Pitt zero-shot AUC, sdpa attention kernel (mechanism test, all else fixed) | 0.7673 | NA | 468 | 228 | scores/part16/POD3/q3o_pitt_attn_backend.csv |
| 3a_attn_eager | Q3O Pitt zero-shot AUC, eager attention kernel (mechanism test, all else fixed) | 0.7679 | NA | 468 | 228 | scores/part16/POD3/q3o_pitt_attn_backend.csv |
| 3a_pcgita_enc_nested | Q3O PC-GITA ENCODER nested probe AUC (5-repeat mean, layer chosen inside the training folds) | 0.8969 | [0.8322, 0.9454] | 1100 | 100 | scores/part16/POD3/q3o_pcgita_encoder_perlayer.csv |
| 3a_pcgita_enc_best_stage | Q3O PC-GITA ENCODER best single stage = audio encoder layer 14 (descriptive max, oracle-selected) | 0.9133 | [0.8519, 0.9604] | 1100 | 100 | scores/part16/POD3/q3o_pcgita_encoder_perlayer.csv |
| 3a_pcgita_proj_nested | Q3O PC-GITA PROJECTOR probe AUC (single point, no layer choice; 0.8872 in memory, see DISCREPANCIES LOW) | 0.8873 | [0.8156, 0.9431] | 1100 | 100 | scores/part16/POD3/q3o_pcgita_proj_perlayer.csv |
| 3a_pcgita_llm_nested | Q3O PC-GITA LANGUAGE MODEL nested probe AUC (5-repeat mean, layer chosen inside the training folds) | 0.8772 | [0.8117, 0.9309] | 1100 | 100 | scores/part16/POD3/q3o_pcgita_llm_perlayer.csv |
| 3a_pcgita_llm_best_stage | Q3O PC-GITA LANGUAGE MODEL best single stage = thinker layer 14 (descriptive max, oracle-selected) | 0.8960 | [0.8341, 0.9463] | 1100 | 100 | scores/part16/POD3/q3o_pcgita_llm_perlayer.csv |
| 3a_pcgita_ans_nested | Q3O PC-GITA ANSWER STATE nested probe AUC (5-repeat mean, stage chosen inside the training folds) | 0.8814 | [0.8171, 0.9330] | 1100 | 100 | scores/part16/POD3/q3o_pcgita_ans_perlayer.csv |
| 3a_pcgita_ans_best_stage | Q3O PC-GITA ANSWER STATE best single stage = thinker layer 15 (descriptive max, oracle-selected) | 0.8962 | [0.8327, 0.9452] | 1100 | 100 | scores/part16/POD3/q3o_pcgita_ans_perlayer.csv |
| POD4_4b_resolved | Pitt 468 windows resolvable to CHAT timings | 468 | NA | 468 | 228 | scores/part16/POD4/cha_overlap_468.csv |
| POD4_4b_invwin | Pitt windows containing timed INV speech in the heard 30s | 158 | NA | 468 | 228 | scores/part16/POD4/cha_overlap_468.csv |
| POD4_4b_invshare | mean INV share of the heard 30s window | 0.0200 | NA | 468 | 228 | scores/part16/POD4/cha_overlap_468.csv |
| POD4_4b_parshare | mean PAR share of the heard 30s window | 0.5871 | NA | 468 | 228 | scores/part16/POD4/cha_overlap_468.csv |
| POD4_4b_pardur | mean participant-only duration after re-cut (s) | 17.61 | NA | 468 | 228 | scores/part16/POD4/cut_report.json |
| POD4_4b_under10s | re-cut windows with under 10s of participant speech | 68 | NA | 468 | 228 | scores/part16/POD4/cut_report.json |
| POD4_4b_pub_zs | Pitt published Omni zero-shot AUC, recomputed rank formula | 0.6578 | [0.6033, 0.7125] | 468 | 228 | <local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv |
| POD4_4b_pub_enc | Pitt published Omni encoder probe AUC, recomputed rank formula | 0.7969 | [0.7412, 0.8512] | 468 | 228 | <local data dir>/Desktop/release/omni_final/omni_pitt_enc_nested_oof.csv |
| POD4_4b_A_zs | Pitt arm A original cut, Omni zero-shot AUC | 0.6497 | [0.5959, 0.7056] | 468 | 228 | scores/part16/POD4/p16_pitt_orig_zeroshot_scores.csv |
| POD4_4b_A_enc | Pitt arm A original cut, Omni encoder nested probe AUC | 0.7949 | [0.7431, 0.8446] | 468 | 228 | scores/part16/POD4/p16_pitt_orig_enc_nested_oof.csv |
| POD4_4b_B_zs | Pitt arm B interviewer removed PAR-only, Omni zero-shot AUC | 0.7230 | [0.6664, 0.7753] | 468 | 228 | scores/part16/POD4/p16_pitt_paronly_zeroshot_scores.csv |
| POD4_4b_B_enc | Pitt arm B interviewer removed PAR-only, Omni encoder nested probe AUC | 0.7497 | [0.6940, 0.8035] | 468 | 228 | scores/part16/POD4/p16_pitt_paronly_enc_nested_oof.csv |
| POD4_4b_C_zs | Pitt arm C duration-matched WITH interviewer, Omni zero-shot AUC | 0.6145 | [0.5582, 0.6683] | 468 | 228 | scores/part16/POD4/p16_pitt_durmatch_zeroshot_scores.csv |
| POD4_4b_C_enc | Pitt arm C duration-matched WITH interviewer, Omni encoder nested probe AUC | 0.7980 | [0.7448, 0.8489] | 468 | 228 | scores/part16/POD4/p16_pitt_durmatch_enc_nested_oof.csv |
| POD4_4b_pd_zs_AB | PAIRED zero-shot delta, original cut to interviewer removed | +0.0732 | NA | 468 | 228 | scores/part16/POD4/boot4b.json |
| POD4_4b_pd_zs_CB | PAIRED zero-shot delta, duration-matched WITH interviewer to interviewer removed | +0.1084 | NA | 468 | 228 | scores/part16/POD4/boot4b.json |
| POD4_4b_pd_enc_AB | PAIRED encoder probe delta, original cut to interviewer removed | -0.0452 | [-0.0857, -0.0083] | 468 | 228 | scores/part16/POD4/boot4b.json |
| POD4_4b_pd_enc_CB | PAIRED encoder probe delta, duration-matched WITH interviewer to interviewer removed | -0.0483 | [-0.0907, -0.0062] | 468 | 228 | scores/part16/POD4/boot4b.json |
| POD4_4b_pd_zs_AC | PAIRED zero-shot delta, duration control only 30s to 17.6s both WITH interviewer | -0.0352 | [-0.0729, -0.0001] | 468 | 228 | scores/part16/POD4/boot4b.json |
| POD4_4b_pd_enc_AC | PAIRED encoder probe delta, duration control only 30s to 17.6s both WITH interviewer | +0.0032 | [-0.0367, +0.0394] | 468 | 228 | scores/part16/POD4/boot4b.json |
| POD4_4b_B_zs_conf | Pitt arm B zero-shot AUC, conflict arm | 0.5408 | [0.4245, 0.6622] | 146 | 100 | scores/part16/POD4/p16_pitt_paronly_zeroshot_scores.csv |
| POD4_4b_B_enc_conf | Pitt arm B encoder probe AUC, conflict arm | 0.6295 | [0.5111, 0.7345] | 146 | 100 | scores/part16/POD4/p16_pitt_paronly_enc_nested_oof.csv |
| POD4_4b_B_zs_agr | Pitt arm B zero-shot AUC, agreement arm | 0.8007 | [0.7500, 0.8501] | 322 | 175 | scores/part16/POD4/p16_pitt_paronly_zeroshot_scores.csv |
| POD4_4b_B_enc_agr | Pitt arm B encoder probe AUC, agreement arm | 0.8037 | [0.7452, 0.8539] | 322 | 175 | scores/part16/POD4/p16_pitt_paronly_enc_nested_oof.csv |
| POD4_4b_A_zs_conf | Pitt arm A zero-shot AUC, conflict arm | 0.5298 | [0.4189, 0.6386] | 146 | 100 | scores/part16/POD4/p16_pitt_orig_zeroshot_scores.csv |
| POD4_4b_A_enc_conf | Pitt arm A encoder probe AUC, conflict arm | 0.6161 | [0.5060, 0.7281] | 146 | 100 | scores/part16/POD4/p16_pitt_orig_enc_nested_oof.csv |
| POD4_4b_A_zs_agr | Pitt arm A zero-shot AUC, agreement arm | 0.6952 | [0.6327, 0.7560] | 322 | 175 | scores/part16/POD4/p16_pitt_orig_zeroshot_scores.csv |
| POD4_4b_A_enc_agr | Pitt arm A encoder probe AUC, agreement arm | 0.8674 | [0.8201, 0.9096] | 322 | 175 | scores/part16/POD4/p16_pitt_orig_enc_nested_oof.csv |
| POD4_4a_nv_5f_before | NeuroVoz five quiet-frame channel features together, GroupKFold5-by-speaker probe AUC, BEFORE | 0.7977 | [0.7248, 0.8611] | 1270 | 107 | scores/part16/POD4/channel_norm_neurovoz.csv |
| POD4_4a_nv_5f_after | NeuroVoz five channel features together, AFTER -23 LUFS + shaped noise | 0.6025 | [0.5479, 0.6580] | 1270 | 107 | scores/part16/POD4/channel_norm_neurovoz.csv |
| POD4_4a_nv_accept | NeuroVoz acceptance test five-feature AUC below 0.60 after normalisation | FAILED_0.6025 | NA | 1270 | 107 | scores/part16/POD4/featprobe_nv.json |
| POD4_4a_nv_nf_before | NeuroVoz noise floor alone, direction-free AUC, BEFORE | 0.5410 | [0.4731, 0.6049] | 1270 | 107 | scores/part16/POD4/channel_norm_neurovoz.csv |
| POD4_4a_nv_nf_after | NeuroVoz noise floor alone, direction-free AUC, AFTER | 0.5407 | [0.4594, 0.6191] | 1270 | 107 | scores/part16/POD4/channel_norm_neurovoz.csv |
| POD4_4a_nv_snr_before | NeuroVoz SNR alone, direction-free AUC, BEFORE | 0.5432 | [0.4640, 0.6216] | 1270 | 107 | scores/part16/POD4/channel_norm_neurovoz.csv |
| POD4_4a_nv_snr_after | NeuroVoz SNR alone, direction-free AUC, AFTER | 0.5305 | [0.4498, 0.6080] | 1270 | 107 | scores/part16/POD4/channel_norm_neurovoz.csv |
| POD4_4a_nv_cen_before | NeuroVoz spectral centroid alone, direction-free AUC, BEFORE | 0.7782 | [0.7109, 0.8396] | 1270 | 107 | scores/part16/POD4/channel_norm_neurovoz.csv |
| POD4_4a_nv_cen_after | NeuroVoz spectral centroid alone, direction-free AUC, AFTER | 0.5603 | [0.5051, 0.6188] | 1270 | 107 | scores/part16/POD4/channel_norm_neurovoz.csv |
| POD4_4a_nv_ro_before | NeuroVoz spectral roll-off alone, direction-free AUC, BEFORE | 0.8005 | [0.7370, 0.8574] | 1270 | 107 | scores/part16/POD4/channel_norm_neurovoz.csv |
| POD4_4a_nv_ro_after | NeuroVoz spectral roll-off alone, direction-free AUC, AFTER | 0.5938 | [0.5438, 0.6448] | 1270 | 107 | scores/part16/POD4/channel_norm_neurovoz.csv |
| POD4_4a_nv_tilt_before | NeuroVoz spectral tilt alone, direction-free AUC, BEFORE | 0.7020 | [0.6451, 0.7552] | 1270 | 107 | scores/part16/POD4/channel_norm_neurovoz.csv |
| POD4_4a_nv_tilt_after | NeuroVoz spectral tilt alone, direction-free AUC, AFTER | 0.5808 | [0.5369, 0.6259] | 1270 | 107 | scores/part16/POD4/channel_norm_neurovoz.csv |
| POD4_4a_nv_zs_before | NeuroVoz Omni zero-shot AUC, BEFORE normalisation, POD4 re-run | 0.7108 | [0.6523, 0.7671] | 1270 | 107 | scores/part16/POD4/p16_nv_before_zeroshot_scores.csv |
| POD4_4a_nv_enc_before | NeuroVoz Omni encoder nested probe AUC, BEFORE normalisation | 0.9233 | [0.8800, 0.9573] | 1270 | 107 | scores/part16/POD4/p16_nv_before_enc_nested_oof.csv |
| POD4_4a_pg_5f_before | PC-GITA five quiet-frame channel features together, GroupKFold5-by-speaker probe AUC, BEFORE | 0.6657 | [0.5851, 0.7452] | 1100 | 100 | scores/part16/POD4/channel_norm_pcgita.csv |
| POD4_4a_pg_5f_after | PC-GITA five channel features together, AFTER -23 LUFS + shaped noise | 0.5590 | [0.4922, 0.6213] | 1100 | 100 | scores/part16/POD4/channel_norm_pcgita.csv |
| POD4_4a_pg_accept | PC-GITA acceptance test five-feature AUC below 0.60 after normalisation | PASSED_0.5590 | NA | 1100 | 100 | scores/part16/POD4/featprobe_pg.json |
| POD4_4a_pg_nf_before | PC-GITA noise floor alone, direction-free AUC, BEFORE | 0.6627 | [0.5885, 0.7362] | 1100 | 100 | scores/part16/POD4/channel_norm_pcgita.csv |
| POD4_4a_pg_nf_after | PC-GITA noise floor alone, direction-free AUC, AFTER | 0.5734 | [0.5113, 0.6341] | 1100 | 100 | scores/part16/POD4/channel_norm_pcgita.csv |
| POD4_4a_pg_snr_before | PC-GITA SNR alone, direction-free AUC, BEFORE | 0.5873 | [0.5261, 0.6464] | 1100 | 100 | scores/part16/POD4/channel_norm_pcgita.csv |
| POD4_4a_pg_snr_after | PC-GITA SNR alone, direction-free AUC, AFTER | 0.5735 | [0.5158, 0.6286] | 1100 | 100 | scores/part16/POD4/channel_norm_pcgita.csv |
| POD4_4a_pg_cen_before | PC-GITA spectral centroid alone, direction-free AUC, BEFORE | 0.5539 | [0.4997, 0.6104] | 1100 | 100 | scores/part16/POD4/channel_norm_pcgita.csv |
| POD4_4a_pg_cen_after | PC-GITA spectral centroid alone, direction-free AUC, AFTER | 0.5591 | [0.5029, 0.6136] | 1100 | 100 | scores/part16/POD4/channel_norm_pcgita.csv |
| POD4_4a_pg_ro_before | PC-GITA spectral roll-off alone, direction-free AUC, BEFORE | 0.5533 | [0.5000, 0.6081] | 1100 | 100 | scores/part16/POD4/channel_norm_pcgita.csv |
| POD4_4a_pg_ro_after | PC-GITA spectral roll-off alone, direction-free AUC, AFTER | 0.5657 | [0.5102, 0.6189] | 1100 | 100 | scores/part16/POD4/channel_norm_pcgita.csv |
| POD4_4a_pg_tilt_before | PC-GITA spectral tilt alone, direction-free AUC, BEFORE | 0.5610 | [0.5048, 0.6194] | 1100 | 100 | scores/part16/POD4/channel_norm_pcgita.csv |
| POD4_4a_pg_tilt_after | PC-GITA spectral tilt alone, direction-free AUC, AFTER | 0.5526 | [0.4987, 0.6030] | 1100 | 100 | scores/part16/POD4/channel_norm_pcgita.csv |
| POD4_4b_invcue | Pitt interviewer milliseconds alone as a predictor of dementia, AUC | 0.6427 | NA | 468 | 228 | scores/part16/POD4/pitt_no_interviewer.csv |
| POD4_4b_durcue | Pitt participant-only duration alone, direction-free AUC (residual cue in arm B) | 0.6648 | NA | 468 | 228 | scores/part16/POD4/pitt_no_interviewer.csv |
| POD4_4a_nv_zs_after | NeuroVoz Omni zero-shot AUC, AFTER normalisation | 0.6991 | [0.6348, 0.7598] | 1270 | 107 | scores/part16/POD4/p16_nv_after_zeroshot_scores.csv |
| POD4_4a_nv_enc_after | NeuroVoz Omni encoder nested probe AUC, AFTER normalisation | 0.8924 | [0.8481, 0.9323] | 1270 | 107 | scores/part16/POD4/p16_nv_after_enc_nested_oof.csv |
| POD4_4a_nv_pd_zs | NeuroVoz PAIRED zero-shot delta, before to after normalisation | -0.0117 | [-0.0332, +0.0098] | 1270 | 107 | scores/part16/POD4/boot4a_nv.json |
| POD4_4a_nv_pd_enc | NeuroVoz PAIRED encoder probe delta, before to after normalisation | -0.0308 | [-0.0608, -0.0012] | 1270 | 107 | scores/part16/POD4/boot4a_nv.json |
| POD4_4a_pg_enc_before | PC-GITA Omni encoder nested probe AUC, BEFORE normalisation | 0.8959 | [0.8339, 0.9448] | 1100 | 100 | scores/part16/POD4/p16_pg_before_enc_nested_oof.csv |
| POD4_4a_pg_zs_before | PC-GITA Omni zero-shot AUC, BEFORE normalisation | 0.5728 | [0.5002, 0.6440] | 1100 | 100 | scores/part16/POD4/p16_pg_before_zeroshot_scores.csv |
| POD4_4a_pg_zs_after | PC-GITA Omni zero-shot AUC, AFTER normalisation | 0.5674 | [0.5054, 0.6304] | 1100 | 100 | scores/part16/POD4/p16_pg_after_zeroshot_scores.csv |
| POD4_4a_pg_enc_after | PC-GITA Omni encoder nested probe AUC, AFTER normalisation | 0.8298 | [0.7546, 0.8968] | 1100 | 100 | scores/part16/POD4/p16_pg_after_enc_nested_oof.csv |
| POD4_4a_pg_pd_zs | PC-GITA PAIRED zero-shot delta, before to after normalisation | -0.0054 | [-0.0437, +0.0357] | 1100 | 100 | scores/part16/POD4/boot4a_pg.json |
| POD4_4a_pg_pd_enc | PC-GITA PAIRED encoder probe delta, before to after normalisation | -0.0661 | [-0.1181, -0.0180] | 1100 | 100 | scores/part16/POD4/boot4a_pg.json |
| 4b_adress2020_A_orig_zs | adress2020 arm A_orig Omni zero-shot AUC | 0.6272 | [0.5379, 0.7165] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_A_orig_enc | adress2020 arm A_orig Omni encoder nested probe AUC | 0.7959 | [0.7251, 0.8602] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_B_paronly_zs | adress2020 arm B_paronly Omni zero-shot AUC | 0.6524 | [0.5648, 0.7380] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_B_paronly_enc | adress2020 arm B_paronly Omni encoder nested probe AUC | 0.7641 | [0.6921, 0.8350] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_C_durmatch_zs | adress2020 arm C_durmatch Omni zero-shot AUC | 0.6313 | [0.5435, 0.7141] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_C_durmatch_enc | adress2020 arm C_durmatch Omni encoder nested probe AUC | 0.8075 | [0.7361, 0.8705] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_published_zs | adress2020 arm published Omni zero-shot AUC | 0.6241 | [0.5348, 0.7148] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_published_enc | adress2020 arm published Omni encoder nested probe AUC | 0.7878 | [0.7148, 0.8517] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_paired_zs_A_to_B | adress2020 PAIRED zero-shot: original cut -> interviewer removed | 0.0251 | [-0.0254, 0.0769] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_paired_zs_A_to_C | adress2020 PAIRED zero-shot: original cut -> duration-matched, interviewer KEPT | 0.0041 | [-0.0386, 0.0473] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_paired_zs_C_to_B | adress2020 PAIRED zero-shot: duration-matched WITH interviewer -> interviewer removed | 0.0210 | [-0.0325, 0.0756] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_paired_enc_A_to_B | adress2020 PAIRED encoder probe: original cut -> interviewer removed | -0.0317 | [-0.0761, 0.0102] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_paired_enc_A_to_C | adress2020 PAIRED encoder probe: original cut -> duration-matched, interviewer KEPT | 0.0117 | [-0.0229, 0.0449] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_paired_enc_C_to_B | adress2020 PAIRED encoder probe: duration-matched WITH interviewer -> interviewer removed | -0.0434 | [-0.0888, -0.0033] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_gap_A_orig | adress2020 arm A_orig GAP = encoder probe AUC minus zero-shot AUC | 0.1686 | [0.0627, 0.2735] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_gap_B_paronly | adress2020 arm B_paronly GAP = encoder probe AUC minus zero-shot AUC | 0.1118 | [0.0020, 0.2144] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_gap_C_durmatch | adress2020 arm C_durmatch GAP = encoder probe AUC minus zero-shot AUC | 0.1762 | [0.0739, 0.2792] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_gap_published | adress2020 arm published GAP = encoder probe AUC minus zero-shot AUC | 0.1637 | [0.0557, 0.2693] | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_inv_share_mean | adress2020 mean interviewer share of the original window | 0.0853 | NA | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_inv_share_median | adress2020 median interviewer share of the original window | 0.0501 | NA | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_inv_any_count | adress2020 windows containing any interviewer speech | 114 | NA | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_inv_gt5s_count | adress2020 windows containing more than 5 s of interviewer speech | 22 | NA | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_dur_removed_mean_s | adress2020 mean seconds removed going from arm A to arm B | 2.3600 | NA | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adress2020_par_under10s_count | adress2020 windows with under 10 s of participant speech | 1 | NA | 156 | 156 | out/adress2020_no_interviewer.csv |
| 4b_adresso_A_orig_zs | adresso arm A_orig Omni zero-shot AUC | 0.6291 | [0.5404, 0.7134] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_A_orig_enc | adresso arm A_orig Omni encoder nested probe AUC | 0.8624 | [0.8063, 0.9155] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_B_paronly_zs | adresso arm B_paronly Omni zero-shot AUC | 0.6302 | [0.5461, 0.7202] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_B_paronly_enc | adresso arm B_paronly Omni encoder nested probe AUC | 0.8242 | [0.7582, 0.8822] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_C_durmatch_zs | adresso arm C_durmatch Omni zero-shot AUC | 0.6083 | [0.5194, 0.6999] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_C_durmatch_enc | adresso arm C_durmatch Omni encoder nested probe AUC | 0.8726 | [0.8183, 0.9225] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_published_zs | adresso arm published Omni zero-shot AUC | 0.6320 | [0.5448, 0.7177] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_published_enc | adresso arm published Omni encoder nested probe AUC | 0.9046 | [0.8548, 0.9479] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_paired_zs_A_to_B | adresso PAIRED zero-shot: original cut -> interviewer removed | 0.0011 | [-0.0441, 0.0462] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_paired_zs_A_to_C | adresso PAIRED zero-shot: original cut -> duration-matched, interviewer KEPT | -0.0208 | [-0.0610, 0.0196] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_paired_zs_C_to_B | adresso PAIRED zero-shot: duration-matched WITH interviewer -> interviewer removed | 0.0219 | [-0.0253, 0.0683] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_paired_enc_A_to_B | adresso PAIRED encoder probe: original cut -> interviewer removed | -0.0382 | [-0.0693, -0.0094] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_paired_enc_A_to_C | adresso PAIRED encoder probe: original cut -> duration-matched, interviewer KEPT | 0.0103 | [-0.0128, 0.0326] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_paired_enc_C_to_B | adresso PAIRED encoder probe: duration-matched WITH interviewer -> interviewer removed | -0.0485 | [-0.0832, -0.0161] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_gap_A_orig | adresso arm A_orig GAP = encoder probe AUC minus zero-shot AUC | 0.2333 | [0.1450, 0.3262] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_gap_B_paronly | adresso arm B_paronly GAP = encoder probe AUC minus zero-shot AUC | 0.1940 | [0.0976, 0.2900] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_gap_C_durmatch | adresso arm C_durmatch GAP = encoder probe AUC minus zero-shot AUC | 0.2644 | [0.1687, 0.3602] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_gap_published | adresso arm published GAP = encoder probe AUC minus zero-shot AUC | 0.2726 | [0.1783, 0.3674] | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_inv_share_mean | adresso mean interviewer share of the original window | 0.0825 | NA | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_inv_share_median | adresso median interviewer share of the original window | 0.0395 | NA | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_inv_any_count | adresso windows containing any interviewer speech | 105 | NA | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_inv_gt5s_count | adresso windows containing more than 5 s of interviewer speech | 21 | NA | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_dur_removed_mean_s | adresso mean seconds removed going from arm A to arm B | 2.2145 | NA | 161 | 161 | out/adresso_no_interviewer.csv |
| 4b_adresso_par_under10s_count | adresso windows with under 10 s of participant speech | 4 | NA | 161 | 161 | out/adresso_no_interviewer.csv |
| 3b_fix | Qwen3-Omni Pitt all: probe | 0.8307 | [0.7833, 0.8741] | 468 | 228 | part16/pod_sync/q3o_perp_cols.npz |
| 3b_fix | Qwen3-Omni Pitt all: along readout direction | 0.7656 | [0.7088, 0.8179] | 468 | 228 | part16/pod_sync/q3o_perp_cols.npz |
| 3b_fix | Qwen3-Omni Pitt all: probe minus readout dir (CANONICAL) | 0.8276 | [0.7795, 0.8696] | 468 | 228 | part16/pod_sync/q3o_perp_cols.npz |
| 3b_fix | Qwen3-Omni Pitt conflict: probe | 0.6394 | [0.5323, 0.7418] | 146 | 100 | part16/pod_sync/q3o_perp_cols.npz |
| 3b_fix | Qwen3-Omni Pitt conflict: along readout direction | 0.6029 | [0.4965, 0.7025] | 146 | 100 | part16/pod_sync/q3o_perp_cols.npz |
| 3b_fix | Qwen3-Omni Pitt conflict: probe minus readout dir (CANONICAL) | 0.6532 | [0.5483, 0.7527] | 146 | 100 | part16/pod_sync/q3o_perp_cols.npz |
| 3b_fix | Qwen3-Omni Pitt agreement: probe | 0.9032 | [0.8626, 0.9374] | 322 | 175 | part16/pod_sync/q3o_perp_cols.npz |
| 3b_fix | Qwen3-Omni Pitt agreement: along readout direction | 0.8264 | [0.7741, 0.8731] | 322 | 175 | part16/pod_sync/q3o_perp_cols.npz |
| 3b_fix | Qwen3-Omni Pitt agreement: probe minus readout dir (CANONICAL) | 0.8937 | [0.8522, 0.9312] | 322 | 175 | part16/pod_sync/q3o_perp_cols.npz |
