import json, hashlib, platform, numpy as np, pandas as pd, scipy
V = "<local data dir>/leftovers_23sep/verify"; W = V + "/m7m11_work"
m7 = json.load(open(W + "/v_m7_results.json")); m11 = json.load(open(W + "/v_m11_results.json"))
claimed = [
 ("M7mo5_adresso_gap", "0.2602", "0.1773", "0.3379"), ("M7mo5_adress2020_gap", "0.1802", "0.0722", "0.2858"),
 ("M7mo5_pcgita_gap", "0.3306", "0.2489", "0.4127"), ("M7mo5_neurovoz_gap", "0.2262", "0.1714", "0.2836"),
 ("M7mo5_kcl_gap", "0.0363", "-0.2075", "0.2638"), ("VAL_neurovoz_pod6_reextract_exact", "0.2267", "0.1742", "0.2851"),
 ("VAL_adress2020_pod4b_reextract_exact", "0.1783", "0.0750", "0.2770"),
 ("M11fix_B_xsys_disagree_mean_PD", "0.3049", "0.1301", "0.4944"), ("M11fix_B_xsys_disagree_mean_control", "0.4779", "0.0917", "1.1479"),
 ("M11fix_C_mean_token_logprob_mean_PD", "-0.1103", "-0.1502", "-0.0739"), ("M11fix_C_mean_token_logprob_mean_control", "-0.1356", "-0.2345", "-0.0759"),
 ("M11fix_C_compression_ratio_mean_PD", "1.5938", "1.4484", "1.7269"), ("M11fix_C_compression_ratio_mean_control", "1.4898", "1.3076", "1.6417"),
 ("M11fix_D_n_words_30s_mean_PD", "59.1875", "46.1875", "71.9375"), ("M11fix_D_n_words_30s_mean_control", "55.8095", "43.0464", "67.6202"),
 ("M11fix_F_wer_vs_local_copy_mean_PD", "0.3123", "0.1974", "0.4388"), ("M11fix_F_wer_vs_local_copy_mean_control", "0.3987", "0.2558", "0.5577"),
 ("M11fix_B_xsys_disagree_MEDIAN_PD", "0.1027", "0.0323", "0.3617"), ("M11fix_B_xsys_disagree_MEDIAN_control", "0.0333", "0.0000", "0.1341"),
 ("M11fix_B_xsys_disagree_ge20w_MEAN_PD", "0.2100", "0.0671", "0.3837"), ("M11fix_B_xsys_disagree_ge20w_MEAN_control", "0.1506", "0.0562", "0.2684"),
 ("M11fix_B_xsys_disagree_POOLED_PD", "0.2650", "0.0782", "0.4778"), ("M11fix_B_xsys_disagree_POOLED_control", "0.1800", "0.0620", "0.3636"),
 ("M11fix_F_wer_vs_local_copy_diff_PD_minus_control", "-0.0863", "-0.2851", "0.1047"),
 ("M11fix_B_xsys_disagree_MEDIAN_diff_PD_minus_control", "0.0694", "-0.1008", "0.4112"),
 ("M11fix_B_xsys_disagree_ge20w_MEAN_diff_PD_minus_control", "0.0594", "-0.1355", "0.2708"),
 ("M11fix_B_xsys_disagree_POOLED_diff_PD_minus_control", "0.0850", "-0.1870", "0.3488")]
f4 = lambda x: f"{x:.4f}".replace("-0.0000", "0.0000")
rec = {}
for ds in ["adresso", "adress2020", "pcgita", "neurovoz", "kcl"]:
    r = m7[ds]; rec[f"M7mo5_{ds}_gap"] = (r["gap_json"], *r["ci_shift_json"])
for k, key in [("VAL_neurovoz_pod6_reextract_exact", "VAL_neurovoz_pod6"), ("VAL_adress2020_pod4b_reextract_exact", "VAL_adress2020_pod4b")]:
    r = m7[key]; rec[k] = (r["gap"], *r["exact_ci"])
for k, v in m11.items(): rec[k] = (v[0], v[1][0], v[1][1])
rows = []
for i, v, lo, hi in claimed:
    a, b, c = rec[i]; rs = f"{f4(a)} [{f4(b)}, {f4(c)}]"; cs = f"{v} [{lo}, {hi}]"
    rows.append(dict(id=i, claimed=cs, recomputed=rs, verified="yes" if (f4(a), f4(b), f4(c)) == (v, lo, hi) else "no"))
df = pd.DataFrame(rows); out = V + "/m7m11_VERIFIED.tsv"; df.to_csv(out, sep="\t", index=False)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
OMNI = "<local data dir>/release/omni_final"
inputs = [f"{OMNI}/omni_{d}_{s}" for d in ["adresso", "adress2020", "pcgita", "neurovoz", "kcl"] for s in ["enc_nested_oof.csv", "zeroshot_scores.csv", "nested_repeats.json"]]
inputs += ["<local data dir>/release/scores/part20/POD6/B_neurovoz_full_check/omni_nvfull_enc_nested5_oof.csv",
           "<local data dir>/release/edaic_rerun/part16/POD4B/out/p16_adress2020_orig_enc_nested_oof.csv",
           "<local data dir>/release/edaic_rerun/part16/POD4B/states/p16_adress2020_orig_states.npz",
           "<local data dir>/release/edaic_rerun/part16/M11_kcl_wer.csv",
           W + "/pull/lo-vm7m11-1/out/vrefit_adress2020_pod4b_oof.csv", W + "/pull/lo-vm7m11-1/out/vrefit_adress2020_pod4b.json",
           "scores/leftovers_23sep/m7m11/lo-m7m11-1/adress2020_pod4b_enc_rep5_oof.csv"]
inputs += [f"<local data dir>/release/folds/adress2020_groupkfold5_pod_seed{s}_UNVERIFIED.csv" for s in range(5)]
side = dict(
    table=out, rows=len(df), verified_yes=int((df.verified == "yes").sum()),
    task_checked="scores/leftovers_23sep/m7m11/m7m11_values.tsv",
    independence="own code, none of the task's scripts run or imported. AUC is a weighted Mann-Whitney over speaker draw counts "
                 "(a speaker drawn k times weighs k), checked equal to sklearn roc_auc_score(sample_weight). Median over the count-weighted multiset; "
                 "pooled = sum(w*(S+D+I))/sum(w*N). ADReSS-2020 POD4B refit: own script (sklearn Pipeline, ProcessPoolExecutor) on own pod.",
    bootstrap="2000 draws; fresh numpy default_rng(0) per cell; idx = rng.choice(N, size=N, replace=True) over the sorted unique speaker ids "
              "(M7: speaker cast to str as in part16 M7_gap_per_dataset.py; M11: that group's speakers for one-group cells, all 37 or the 31 "
              ">=20-word speakers for PD minus control, same idx); 2.5/97.5 np.quantile; one-class draws dropped (none were)",
    m7_convention="point = json 'mean' of omni_<ds>_nested_repeats.json enc minus zero-shot AUC recomputed from per-clip p_yes; "
                  "approx interval = single-split paired interval shifted by (json mean minus single-split AUC)",
    m7_detail={ds: m7[ds] for ds in ["adresso", "adress2020", "pcgita", "neurovoz", "kcl"]},
    val_detail={k: m7[k] for k in ["VAL_neurovoz_pod6", "VAL_adress2020_pod4b"]},
    m11_detail=m11,
    findings=[
      "27/27 claimed values reproduce at 4 dp.",
      "M7 single-split rows reproduce part16 M7_gap_per_dataset.csv exactly (ADReSSo 0.2748 [0.1920, 0.3526], ADReSS-2020 0.1637 [0.0557, 0.2693], PC-GITA 0.3454 [0.2636, 0.4275], NeuroVoz 0.2209 [0.1660, 0.2783], KCL 0.0625 [-0.1813, 0.2900]); zero-shot AUCs 0.6150, 0.6241, 0.5635, 0.6951, 0.7143.",
      "Rounding bounds from the discrete AUC grid k/(n1 n0) plus the json mean: ADReSSo gap 0.260185..0.260211 (0.2602 fixed), ADReSS-2020 0.180178 exact, KCL 0.036310 exact, PC-GITA 0.330594..0.330672 (0.3306 or 0.3307), NeuroVoz 0.226170..0.226267 (0.2262 or 0.2263). Matches the task's not_closed note.",
      "Sensitivity: if the shift uses the mean of the five rounded repeats (0.87524) instead of the json mean (0.8752), the ADReSSo approximate lo moves from 0.1773 to 0.1774. All other M7 endpoints are unchanged at 4 dp.",
      "ADReSS-2020 POD4B refit on my own pod lo-vm7m11-1 (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1, Python 3.12.3): per-repeat AUCs identical to the task's refit (0.811801, 0.794379, 0.799310, 0.806870, 0.799474; mean 0.802367); outer folds equal the Table 1 seed files 156/156 for all 5 seeds; per-clip probabilities differ from the task's refit by at most 0.0020 (float-level solver differences), AUCs unaffected. Exact interval from my OOF [0.0750, 0.2770]; shifted single [0.0750, 0.2823]; max endpoint error 0.0053.",
      "NeuroVoz POD6: per-repeat 0.9287/0.9158/0.9253/0.9129/0.9262, mean 0.9218; exact [0.1742, 0.2851]; shifted single [0.1723, 0.2841]; max endpoint error 0.0019.",
      "M11 all-37 scheme reproduces the old numbers: B mean PD [0.1157, 0.5115], control [0.0814, 1.1692]; MEDIAN PD [0.0323, 0.4799], control [0.0000, 0.2088]; ge20w PD [0.0645, 0.3906], control [0.0481, 0.2765]; POOLED PD [0.0729, 0.4944], control [0.0595, 0.3764]. part16 rows/M11.tsv does carry these and n=37 on the group-mean rows.",
      "Overleaf zip tex (paper1_submission_23sep/zip/AudioLLMHealthUtilityGap.tex) line 149 does print 0.35 on PC-GITA; the mean-of-five gap is 0.3306."],
    pod=dict(name="lo-vm7m11-1", id="xtnh88x0mz6ide", vcpu=32, cost_per_hr=0.96, runtime="about 3 min", pulled_sha256_checked=True, deleted="HTTP 204, GET 404"),
    scripts=[W + "/scripts/" + s for s in ["vcore.py", "v_m7.py", "v_m11.py", "vrefit.py", "v_write.py"]],
    env_mac=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__),
    inputs_sha256={p: sha(p) for p in inputs})
side["outputs_sha256"] = {out: sha(out)}
json.dump(side, open(V + "/m7m11_VERIFIED.sidecar.json", "w"), indent=1, default=float)
print(df.to_string(index=False))
