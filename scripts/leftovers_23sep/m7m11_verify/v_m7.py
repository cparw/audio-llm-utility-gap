import sys, json, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); from vcore import WAUC, speaker_draw_weights, pct, NB
OMNI = "<local data dir>/release/omni_final"
W = "<local data dir>/leftovers_23sep/verify/m7m11_work"
res = {}

def paired(y, spk, score_sets, zs, single=None):
    """score_sets: list of probe score vectors (1 = single split, 5 = repeats). per draw: mean over sets of AUC minus zs AUC"""
    spl = sorted(set(spk))
    fa = [WAUC(y, s) for s in score_sets]; fz = WAUC(y, zs); fs = WAUC(y, single) if single is not None else None
    ones = np.ones(len(y))
    pt_sets = [f(ones) for f in fa]; pt_z = fz(ones); pt_s = fs(ones) if fs else None
    d, dz, ds = [], [], []
    for w, _ in speaker_draw_weights(spk, spl):
        a = [f(w) for f in fa]; z = fz(w)
        if np.isnan(z) or any(np.isnan(a)): continue
        d.append(np.mean(a) - z); dz.append(z)
        if fs: ds.append(fs(w) - z)
    out = dict(n=len(y), n_spk=len(spl), per_set=pt_sets, mean_sets=float(np.mean(pt_sets)), zs=pt_z,
               ci=pct(d), zs_ci=pct(dz), usable=len(d))
    if fs: out.update(single=pt_s, single_ci=pct(ds))
    return out

for ds in ["adresso", "adress2020", "pcgita", "neurovoz", "kcl"]:
    pr = pd.read_csv(f"{OMNI}/omni_{ds}_enc_nested_oof.csv"); zz = pd.read_csv(f"{OMNI}/omni_{ds}_zeroshot_scores.csv")
    js = json.load(open(f"{OMNI}/omni_{ds}_nested_repeats.json"))["enc"]
    m = pr.merge(zz[["clip", "label", "p_yes"]].rename(columns={"label": "label_z"}), on="clip", how="inner")
    assert (m.label == m.label_z).all()
    y = m.label.values.astype(int); spk = m.speaker.astype(str).values
    r = paired(y, spk, [m.p_probe.values], m.p_yes.values)
    single = r["mean_sets"]; lo, hi = r["ci"]
    m5_json = js["mean"]; m5_rep = float(np.mean(js["per_repeat"]))
    n1 = int(y.sum()); n0 = len(y) - n1
    res[ds] = dict(n=r["n"], n_spk=r["n_spk"], n1=n1, n0=n0, zs=r["zs"], zs_k=r["zs"] * n1 * n0, zs_ci=r["zs_ci"], single=single,
                   single_gap=single - r["zs"], single_ci=[lo, hi], mean5_json=m5_json, mean5_of_rounded_repeats=m5_rep,
                   per_repeat=js["per_repeat"], usable=r["usable"],
                   gap_json=m5_json - r["zs"], gap_rep=m5_rep - r["zs"],
                   ci_shift_json=[lo + m5_json - single, hi + m5_json - single],
                   ci_shift_rep=[lo + m5_rep - single, hi + m5_rep - single],
                   # bounds of the true mean of five given each repeat rounded to 4 dp
                   gap_bounds=[m5_rep - 0.00005 - r["zs"], m5_rep + 0.00005 - r["zs"]])
    print(ds, {k: (np.round(v, 4).tolist() if isinstance(v, (list, tuple)) else (round(v, 4) if isinstance(v, float) else v)) for k, v in res[ds].items()}, flush=True)

# NeuroVoz POD6 re-extraction (per-repeat OOF saved), exact mean-of-five paired interval + shifted single
nv = pd.read_csv("<local data dir>/release/scores/part20/POD6/B_neurovoz_full_check/omni_nvfull_enc_nested5_oof.csv")
zn = pd.read_csv(f"{OMNI}/omni_neurovoz_zeroshot_scores.csv")
m = nv.merge(zn[["clip", "p_yes"]], on="clip", how="inner"); assert len(m) == 1270
y = m.label.values.astype(int); spk = m.speaker.astype(str).values
ex = paired(y, spk, [m[f"p_seed{r}"].values for r in range(5)], m.p_yes.values, single=m.p_single.values)
sh = [ex["single_ci"][0] + ex["mean_sets"] - ex["single"], ex["single_ci"][1] + ex["mean_sets"] - ex["single"]]
res["VAL_neurovoz_pod6"] = dict(per_repeat=ex["per_set"], mean5=ex["mean_sets"], zs=ex["zs"], gap=ex["mean_sets"] - ex["zs"], exact_ci=ex["ci"],
                                single=ex["single"], shifted_single_ci=sh, max_endpoint_err=max(abs(sh[0] - ex["ci"][0]), abs(sh[1] - ex["ci"][1])), usable=ex["usable"])
print("VAL_nv", res["VAL_neurovoz_pod6"], flush=True)

# ADReSS-2020 POD4B: my own pod refit, and (cross-check) the task's pulled OOF
mine = pd.read_csv(f"{W}/pull/lo-vm7m11-1/out/vrefit_adress2020_pod4b_oof.csv")
theirs = pd.read_csv("scores/leftovers_23sep/m7m11/lo-m7m11-1/adress2020_pod4b_enc_rep5_oof.csv")
single4b = pd.read_csv("<local data dir>/release/edaic_rerun/part16/POD4B/out/p16_adress2020_orig_enc_nested_oof.csv")
za = pd.read_csv(f"{OMNI}/omni_adress2020_zeroshot_scores.csv")
m = mine.merge(single4b[["clip", "p_probe"]], on="clip").merge(za[["clip", "p_yes"]], on="clip").merge(theirs, on="clip", suffixes=("", "_t"))
assert len(m) == 156
maxdiff = max(float(np.abs(m[f"v_p{r}"] - m[f"p_seed{r}"]).max()) for r in range(5))
foldeq_theirs = {r: int((m[f"v_fold{r}"] == m[f"fold_seed{r}"]).sum()) for r in range(5)}
foldeq_t1 = {}
for r in range(5):
    f = pd.read_csv(f"<local data dir>/release/folds/adress2020_groupkfold5_pod_seed{r}_UNVERIFIED.csv")
    idcol = [c for c in f.columns if "clip" in c.lower() or "file" in c.lower() or "name" in c.lower()][0]
    fc = [c for c in f.columns if c.lower().startswith("fold")][0]
    mm = m.merge(f[[idcol, fc]], left_on="clip", right_on=idcol)
    foldeq_t1[r] = f"{int((mm[f'v_fold{r}'] == mm[fc]).sum())}/{len(mm)}"
y = m.label.values.astype(int); spk = m.speaker.astype(str).values
ex = paired(y, spk, [m[f"v_p{r}"].values for r in range(5)], m.p_yes.values, single=m.p_probe.values)
sh = [ex["single_ci"][0] + ex["mean_sets"] - ex["single"], ex["single_ci"][1] + ex["mean_sets"] - ex["single"]]
res["VAL_adress2020_pod4b"] = dict(per_repeat=ex["per_set"], mean5=ex["mean_sets"], zs=ex["zs"], gap=ex["mean_sets"] - ex["zs"], exact_ci=ex["ci"],
                                   single=ex["single"], shifted_single_ci=sh, max_endpoint_err=max(abs(sh[0] - ex["ci"][0]), abs(sh[1] - ex["ci"][1])),
                                   usable=ex["usable"], my_refit_vs_task_refit_max_abs_pred_diff=maxdiff,
                                   fold_equal_my_vs_task=foldeq_theirs, fold_equal_my_vs_table1_seed_files=foldeq_t1)
print("VAL_a20", res["VAL_adress2020_pod4b"], flush=True)
json.dump(res, open(f"{W}/v_m7_results.json", "w"), indent=1, default=float)
