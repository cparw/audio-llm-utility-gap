#!/usr/local/bin/python3
"""Triage of every MISMATCH and a verification note for every UNMATCHED row of the final-tex Part 12 run.
Verified values come from verify/values_reverified_mac.csv (the 152-row verifier rerun on the Mac from a fresh
copy of the source files) and verify/values_new_2325Z.csv (verify_new_2325Z.py), plus the file reads named in
each row. usage: build_triage_2325Z.py FLAGGED.csv OUT_DIR"""
import sys, re, pandas as pd
FL, OD = sys.argv[1], sys.argv[2]
R = "<local data dir>/release"; P23 = "part23"; P24 = "part24"
d = pd.read_csv(FL)
T = {  # item_id: (triage, verified value, file, why)
 "L67_321": ("MISPAIRING", "0.8968 (smallest median Yes+No mass over 70 zero-shot cells, AF3 PC-GITA); E-DAIC whole-interview cell 0.9969", R + "/edaic_rerun/part16/M9_answer_mass.csv; " + P23 + "/A/A2_edaic_whole_zeroshot_perclip.csv", "Matcher paired the mass floor with the PC-GITA zero-shot AUC. 'At least 0.89' is a true floor (0.8968 would round to 0.90)."),
 "L77_360": ("MISPAIRING", "KCL 0.4315, NeuroVoz 0.5814, PC-GITA 0.6196", R + "/scores/part20/POD6/A_language_saliency/A_tfidf_{kcl,neurovoz,pcgita}_oof.csv", "Matcher used the Qwen2.5-Omni transcript answers; the sentence is the TF-IDF regression (same as the 06:38 triage)."),
 "L77_398": ("MISPAIRING", "0.6778 [0.5973, 0.7549] n=275", R + "/scores/part20/POD6/A_language_saliency/A_tfidf_edaicfull_oof.csv", "Matcher used the Qwen2.5-Omni transcript answer 0.823 (same as the 06:38 triage)."),
 "L77_433": ("MISPAIRING", "Pitt 0.8353, ADReSSo 0.8471, ADReSS-2020 0.9254", R + "/scores/part20/POD6/A_language_saliency/A_tfidf_{pitt,adresso,adress2020}_oof.csv", "Matcher reused the Parkinson's answer set (same as the 06:38 triage)."),
 "L92_538": ("MISPAIRING", "max 0.0664, mean 0.0240 over 14 wording runs", R + "/omni_final/omni_{ds}_{zeroshot,p2,p3}.json", "Matcher paired the largest wording shift with the E-DAIC answer AUC. Caveat: the E-DAIC wording runs are 30 s windows, not the whole-interview Table 1 cell."),
 "L97_803": ("MISPAIRING", "segments 30.0 to 55.6 s, median 33.3 s", "<local data dir>/DementiaBank/pitt_conflict_manifest.csv", "Matcher paired durations in seconds with zero-shot AUCs."),
 "L125_27": ("WRONG COPY", "0.8366 [0.7786, 0.8866] n=275", P23 + "/A/A2_edaic_whole_zeroshot_perclip.csv (p_yes_whole)", "Right quantity, superseded run: the matcher looks up the first-300 s run (o25_full_zeroshot.json, 0.8285); Table 1 now reports the whole interview (line 81)."),
 "L125_41": ("WRONG COPY", "0.8589 [0.8050, 0.9029] n=275 (seed 0)", P24 + "/FT/FT24_whole_seed0_perclip.csv", "Right quantity, superseded run: the matcher looks up the first-300 s projector FT (sft_full_sft.json, 0.6421); the paper reports the PART 24 whole-interview FT."),
 "L140_549": ("MISPAIRING", "0.7205 [0.6531, 0.7917] n=275", R + "/edaic_rerun/variants/o25_mid30_zeroshot_scores.csv", "Matcher paired the Qwen2.5-Omni middle-30 s answer with Qwen2-Audio (0.633)."),
 "L149_1527": ("MISPAIRING", "0.6282 [0.5584, 0.6985] n=907 spk=76", R + "/scores/part20/POD6/B_neurovoz_matched/B_zeroshot_nvmatched.csv", "Matcher used the Pitt age-matched answer (same as the 06:38 triage)."),
 "L149_2192": ("MISPAIRING", "0.6608 (auc_d)", R + "/omni/readout_direction.csv (row pitt)", "Matcher paired every number of the sentence with the cosine column."),
 "L149_2262": ("MISPAIRING", "0.7709 (auc_probe)", R + "/omni/readout_direction.csv (row pitt)", "Matcher paired it with the cosine."),
 "L149_2283": ("MISPAIRING", "0.7662 (auc_probe_without_d)", R + "/omni/readout_direction.csv (row pitt)", "Matcher paired it with the cosine."),
 "L149_2356": ("REAL (sign only)", "-0.0109 (cosine); random-direction cosine 0.0135", R + "/omni/readout_direction.csv (row pitt)", "The paper prints 0.01 for a cosine of -0.0109. The magnitude claim holds; print -0.01 or say 0.01 in magnitude."),
 "L149_2463": ("MISPAIRING", "0.6174 (proj_d AUC)", R + "/edaic_rerun/part17/T3_q2a_direction.csv", "Matcher paired it with the Qwen3-Omni NeuroVoz probe (0.9372)."),
 "L149_2643": ("MISPAIRING", "0.7437 (LM probe, mean of five, whole interview)", P23 + "/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv", "Matcher used the Qwen3-Omni E-DAIC file (0.5828)."),
 "L149_2741": ("MISPAIRING", "-0.2385 [-0.3244, -0.1505] (encoder minus answer)", P23 + "/.../A3_edaic_whole_meanof5_perclip.csv + A2", "Interval number; matcher paired it with Qwen3-Omni 0.8741."),
 "L149_2701": ("MISPAIRING", "0.8366 (answer, whole interview)", P23 + "/A/A2_edaic_whole_zeroshot_perclip.csv", "Matcher paired it with Qwen3-Omni 0.8741."),
 "L149_2710": ("MISPAIRING", "-0.0929 [-0.1609, -0.0257] (LM minus answer)", P23 + "/.../A3 + A2", "Interval number; matcher paired it with Qwen3-Omni 0.8741."),
 "L149_2728": ("MISPAIRING", "-0.0257 (upper bound of LM minus answer)", P23 + "/.../A3 + A2", "Interval bound; printed with its minus sign in the tex."),
 "L149_1507": ("MISPAIRING", "0.8986 [0.8430, 0.9428] n=907 spk=76", R + "/scores/part20/POD6/B_neurovoz_matched/B_encprobe_nested5_nvmatched_oof.csv", "Matcher used the Pitt age-matched probe (same as the 06:38 triage)."),
 "L149_2750": ("MISPAIRING", "-0.3244 (lower bound of encoder minus answer)", P23 + "/.../A3 + A2", "Interval bound."),
 "L149_2759": ("MISPAIRING", "-0.1505 (upper bound of encoder minus answer)", P23 + "/.../A3 + A2", "Interval bound."),
 "L149_2821": ("MISPAIRING", "0.8359 (answer-state probe, mean of five); minus answer -0.0007 [-0.0455, 0.0436]", P23 + "/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv", "Matcher paired it with Qwen3-Omni 0.8741."),
 "L149_2670": ("MISPAIRING", "0.5981 (encoder probe, mean of five, whole interview)", P23 + "/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv", "Matcher used the Qwen3-Omni E-DAIC file (0.5828)."),
 "L149_1421": ("MISPAIRING", "-0.0308 [-0.0608, -0.0012] (NeuroVoz probe after minus before)", R + "/edaic_rerun/part16/POD4/channel_norm_neurovoz.csv", "Interval bound; matcher paired it with the NeuroVoz probe AUC."),
 "L149_2719": ("MISPAIRING", "-0.1609 (lower bound of LM minus answer)", P23 + "/.../A3 + A2", "Interval bound."),
 "L149_1402": ("MISPAIRING", "-0.0308 (NeuroVoz probe after minus before)", R + "/edaic_rerun/part16/POD4/channel_norm_neurovoz.csv", "Paired difference; matcher paired it with the probe AUC."),
 "L149_152": ("MISPAIRING", "Pitt 0.7706, ADReSSo 0.8752, ADReSS-2020 0.8043", R + "/overnight2/part10/pitt_enc_nested5_oof.npz; paper1_local_runs/omni_final/omni_{adresso,adress2020}_nested_repeats.json", "Matcher used the Parkinson's encoder files."),
 "L149_333": ("MISPAIRING", "0.1128 [0.0455, 0.1768] n=468 spk=228", R + "/overnight2/part10/pitt_enc_nested5_oof.npz + omni_final/omni_pitt_zeroshot_scores.csv", "Matcher paired the Pitt gap with the KCL probe AUC."),
 "L149_339": ("MISPAIRING", "[0.0455, 0.1768]", "same as above", "Interval of the Pitt gap."),
 "L149_362": ("MISPAIRING", "0.3306 = 0.8941 - 0.5635 (point only)", "master_lookup.csv + omni_final/omni_pcgita_zeroshot_scores.csv", "Matcher paired the gap with the probe AUC. No five-repeat per-clip PC-GITA file for an interval; single-split gap is 0.3454."),
 "L149_593": ("MISPAIRING", "0.5967 shipped / 0.6015 pod rerun", R + "/overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv", "Matcher used the Pitt answer (same as the 06:38 triage)."),
 "L149_1411": ("MISPAIRING", "-0.0608 (lower bound, NeuroVoz after minus before)", R + "/edaic_rerun/part16/POD4/channel_norm_neurovoz.csv", "Interval bound."),
 "L149_639": ("MISPAIRING", "0.0503 = 0.8122 - 0.7619", R + "/edaic_rerun/part16/POD3/q3o_pitt_encoder_nested_oof.csv + overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv", "Matcher paired the Qwen3-Omni Pitt gap with its arm drop."),
 "L149_796": ("MISPAIRING", "probe minus eGeMAPS: KCL +0.0095, ADReSSo +0.1810", R + "/overnight/egemaps_baseline.csv", "Matcher paired margins with AUCs."),
 "L149_920": ("MISPAIRING", "0.1488 (eGeMAPS 0.8439 minus answer 0.6951, NeuroVoz)", R + "/overnight/egemaps_baseline.csv + omni_final/omni_neurovoz_zeroshot_scores.csv", "Matcher paired it with the answer AUC."),
 "L149_1096": ("MISPAIRING", "0.6657 (five features, PC-GITA, before)", R + "/edaic_rerun/part16/POD4/channel_norm_pcgita.csv", "Matcher paired it with the single noise-floor feature (0.6872)."),
 "L149_1224": ("MISPAIRING", "0.5590 (five features, PC-GITA, after)", R + "/edaic_rerun/part16/POD4/channel_norm_pcgita.csv", "Matcher paired it with the single noise-floor feature."),
 "L149_1283": ("MISPAIRING", "-0.0661 [-0.1181, -0.0180] (PC-GITA probe after minus before)", R + "/edaic_rerun/part16/POD4/channel_norm_pcgita.csv", "Paired difference."),
 "L149_1292": ("MISPAIRING", "-0.1181 (lower bound)", R + "/edaic_rerun/part16/POD4/channel_norm_pcgita.csv", "Interval bound."),
 "L149_1301": ("MISPAIRING", "-0.0180 (upper bound)", R + "/edaic_rerun/part16/POD4/channel_norm_pcgita.csv", "Interval bound."),
 "L149_1350": ("MISPAIRING", "0.6025 (five features, NeuroVoz, after)", R + "/edaic_rerun/part16/POD4/channel_norm_neurovoz.csv", "Matcher paired it with the NeuroVoz answer."),
 "L149_911": ("MISPAIRING", "0.2697 (eGeMAPS 0.8332 minus answer 0.5635, PC-GITA)", R + "/overnight/egemaps_baseline.csv + omni_final/omni_pcgita_zeroshot_scores.csv", "Matcher paired it with a speaker-level difference."),
 "L171_48": ("WRONG COPY", "0.5835 [0.4730, 0.6913] n=146", R + "/scores/part20/POD4c/af2_orig30_conflict.csv", "Right quantity, other run: the paper uses the AF2 run cut to the first 30 s (as every model hears); the matcher looks up af2_pitt468.csv (0.631)."),
 "L171_55": ("WRONG COPY", "0.5337 [0.4574, 0.6116] n=322", R + "/scores/part20/POD4c/af2_orig30_agreement.csv", "Same as above; matcher 0.5544 from af2_pitt468.csv."),
 "L172_55": ("WRONG COPY", "0.5855 [0.5144, 0.6554] (part10)", R + "/overnight2/part10/af3_pitt.csv", "Paper uses the part10 copy; matcher the part3 copy (0.5822) (same as the 06:38 triage)."),
 "L172_48": ("WRONG COPY", "0.6075 [0.5063, 0.7129] (part10)", R + "/overnight2/part10/af3_pitt.csv", "Paper uses the part10 copy; matcher the part3 copy (0.6169) (same as the 06:38 triage)."),
 "L180_952": ("MISPAIRING", "0.9620 (311 distinct) / 0.9596 (483 rows)", R + "/edaic_rerun/part16/POD1/p14_o25_1a_joined.csv; p14_1a_completion.json", "Matcher used a Pitt transcript file (0.455)."),
 "L180_931": ("MISPAIRING", "0.1498 [0.1131, 0.1887] n=483", R + "/edaic_rerun/part16/POD1/p14_1a_completion.json", "Matcher used a Pitt transcript file (0.455)."),
 "L180_2536": ("MISPAIRING", "-0.0104 [-0.0356, 0.0155]", R + "/scores/part20/POD4/o25_orig_voice.csv + omni_final/omni_pitt_zeroshot_scores.csv", "Interval bound of the voice-wording change."),
 "L180_2530": ("MISPAIRING", "-0.0356 (lower bound)", "same as above", "Interval bound."),
 "L180_2521": ("MISPAIRING", "-0.0104 (voice minus recording wording)", "same as above", "Paired difference."),
 "L180_2394": ("MISPAIRING", "AF2 0.5835 / 0.5337, AF3 0.6075 / 0.5855: range 0.53 to 0.61", R + "/scores/part20/POD4c/af2_orig30_*.csv; overnight2/part10/af3_pitt.csv", "Matcher paired the Flamingo range with the Qwen arm drops."),
 "L180_1668": ("MISPAIRING", "0.4619 [0.4165, 0.5102] n=1334", R + "/scores/part20/POD3b/sst2_pairs_o25_audio.csv", "Interval of the second-scorer conflict arm."),
 "L180_1824": ("MISPAIRING", "0.1951 on 420 segments", R + "/scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv", "Matcher paired it with the Pitt FT file."),
 "L180_1723": ("MISPAIRING", "0.8554 - 0.4619 = 0.3935", R + "/scores/part20/POD3b/sst2_pairs_o25_audio.csv", "Gap to the agreement arm (distinct segments)."),
 "L180_1662": ("MISPAIRING", "0.4619", R + "/scores/part20/POD3b/sst2_pairs_o25_audio.csv", "Matcher paired it with the Pitt FT file."),
 "L180_1000": ("MISPAIRING", "0.9523 (311 distinct) / 0.9470 (483 rows)", "Qwen3-Omni transcript files (06:38 triage)", "Same as the 06:38 triage."),
 "L180_991": ("MISPAIRING", "0.2243 [0.1757, 0.2742] n=483", "Qwen3-Omni transcript file (06:38 triage)", "Same as the 06:38 triage."),
 "L180_1881": ("MISPAIRING", "0.6150 on 855 segments", R + "/scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.csv", "Matcher paired it with the Pitt FT file."),
 "L226_1379": ("MISPAIRING", "agreement 0.6686 [0.5727, 0.7586] n=311 (before 0.9647)", R + "/scores/part20/POD3/p14_ft966_oof.csv; before: edaic_rerun/part17/T1_perclip_distinct_agreement.csv", "Matcher used p14_o25_sftfull.csv, the projector trained on the 275 interviews (agreement 0.7930), not the one trained on the pairs."),
 "L226_1340": ("MISPAIRING", "conflict 0.5932 [0.5087, 0.6831] n=483 (before 0.2218)", R + "/scores/part20/POD3/p14_ft966_oof.csv; before: edaic_rerun/part14/p14_o25_zeroshot_scores.csv", "Matcher used p14_o25_sftfull.csv (conflict 0.2283), a different training set."),
 "L226_1181": ("MISPAIRING", "0.8184 [0.7496, 0.8758] (seed 1)", P24 + "/FT/FT24_whole_seed1all_perclip.csv", "Matcher paired it with the first-300 s zero-shot answer."),
 "L226_1170": ("MISPAIRING", "+0.0547 (upper bound of FT minus zero-shot)", P24 + "/FT/FT24_whole_seed0_perclip.csv + A2", "Interval bound."),
 "L226_1164": ("MISPAIRING", "-0.0064 (lower bound)", P24 + "/FT/FT24_whole_seed0_perclip.csv + A2", "Interval bound; printed with its minus sign in the tex."),
 "L226_1154": ("MISPAIRING", "+0.0223 [-0.0064, 0.0547]", P24 + "/FT/FT24_whole_seed0_perclip.csv + A2", "Matcher paired the paired gain with a first-30 s difference file."),
 "L226_1084": ("MISPAIRING", "0.7696 (LoRA plus projector)", R + "/omni_final/abl2_pitt_both_oof.csv", "Matcher paired every number of the sentence with the Pitt probe."),
 "L226_958": ("MISPAIRING", "0.8250 (LoRA)", R + "/edaic_rerun/part16/POD2/lora_pitt_oof.csv", "Same."),
 "L226_973": ("MISPAIRING", "0.0577 [0.0154, 0.1044] (LoRA minus projector)", R + "/edaic_rerun/part16/POD2/lora_pitt_oof.csv + omni_final/omnisft_pitt_oof.csv", "Same."),
 "L226_924": ("MISPAIRING", "4,591,104 parameters", "arithmetic 1280 x 3584 + 3584 (06:38 triage)", "Parameter count, not an AUC."),
 "L226_888": ("MISPAIRING", "5,046,272 parameters", "json + arithmetic (06:38 triage)", "Parameter count, not an AUC."),
 "L226_634": ("MISPAIRING", "agreement arm 0.8213 to 0.9022 over six standard runs", R + "/omni_final/omnisft_pitt_oof.csv; scores/part20/POD1{,b}/*ft*_pitt*_oof.csv", "Matcher mixed other datasets' FT files."),
 "L226_557": ("MISPAIRING", "conflict arm 0.4490 to 0.5666 over six standard runs", "same six files", "Matcher mixed other datasets' FT files."),
 "L226_469": ("MISPAIRING", "seeds 1 to 3: 0.7396, 0.8085, 0.7975", R + "/scores/part20/POD1b/std_ft_seed{1,2,3}_pitt_oof.csv", "Matcher used the probe files."),
 "L226_979": ("MISPAIRING", "[0.0154, 0.1044]", "as L226_973", "Interval of LoRA minus projector."),
 "L226_596": ("MISPAIRING", "every conflict-arm interval contains 0.5: yes (7 runs incl. balanced)", "six standard FT files + balanced_ft_pitt_oof.csv", "0.5 is the chance level, not an AUC to match."),
}
U = {  # UNMATCHED item_id -> check note (value, file)
}
m = d[d.status == "MISMATCH"].copy(); miss = set(m.item_id) - set(T)
assert not miss, miss
m["triage_2325Z"] = m.item_id.map(lambda k: T[k][0]); m["verified_value"] = m.item_id.map(lambda k: T[k][1])
m["verified_file"] = m.item_id.map(lambda k: T[k][2]); m["why"] = m.item_id.map(lambda k: T[k][3])
m = m.sort_values(["line", "item_id"])
m[["status", "line", "item_id", "paper_value", "file_value", "file_path", "triage_2325Z", "verified_value", "verified_file", "why", "triage", "context"]].to_csv(OD + "/triage_final_2325Z.csv", index=False)
print(m.triage_2325Z.value_counts().to_string())
