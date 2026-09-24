"""Independent check of the PART25 A retry outputs, written apart from A_gaps_r2.py (no shared code, no sklearn AUC):
AUC by the Mann-Whitney rank formula (scipy rankdata midranks); speakers taken as sorted(set(...)); bootstrap rebuilt draw
by draw from numpy default_rng(0) with rng.choice(n_spk, size=n_spk, replace=True); every input re-read from the raw files.
Checks per dataset: pulled pod OOF csv == per-clip csv columns; zero-shot column == saved paper file; fold columns == the
release fold files; per-repeat AUCs and mean-of-five; Table 1 values read from omni_final json; draws (npz) == rebuilt
draws (1e-9); interval percentiles; the TSV row (4 dp). Also the Pitt reference from the part10 npz.
usage: A_verify_r2.py [datasets...]  -> verify/A_verify_r2.json"""
import csv, json, os, sys, numpy as np
from scipy.stats import rankdata
R = "<local data dir>/release_from_mac/scores/part25/A/retry_tf554"
OF = "<local data dir>/release/omni_final"; FD = "<local data dir>/release/folds"
POD = {d: f"{R}/pull/p25Ar-{d}" for d in ("pcgita", "neurovoz", "adresso", "adress2020")}
if os.environ.get("A_POD_OVERRIDE"):
    for kv in os.environ["A_POD_OVERRIDE"].split(","): k, v = kv.split("="); POD[k] = v
NM = {"pcgita": "PC-GITA", "neurovoz": "NeuroVoz", "adresso": "ADReSSo", "adress2020": "ADReSS-2020"}
def mw(y, s):
    y = np.asarray(y); r = rankdata(s); n1 = int((y == 1).sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
def rebuild(y, spk, cols, zs):
    keys = sorted(set(spk.tolist())); where = {k: [] for k in keys}
    for i, s in enumerate(spk.tolist()): where[s].append(i)
    g = np.random.default_rng(0); out = []
    for _ in range(2000):
        pick = g.choice(len(keys), size=len(keys), replace=True)
        ii = np.array([j for k in pick for j in where[keys[k]]]); yy = y[ii]
        if yy.min() == yy.max(): out.append(np.nan); continue
        out.append(np.mean([mw(yy, c[ii]) for c in cols]) - mw(yy, zs[ii]))
    return np.array(out)
tsv = {r["dataset"]: r for r in csv.DictReader(open(f"{R}/A_gaps_r2.tsv"), delimiter="\t")}
res = {}; allpass = True
for ds in (sys.argv[1:] or list(POD)):
    c = {}
    pod = list(csv.DictReader(open(f"{POD[ds]}/out/r2_{ds}_enc_nested5_oof.csv")))
    pcl = list(csv.DictReader(open(f"{R}/per_clip/A_{ds}_perclip.csv")))
    c["clip_order_equal"] = [r["clip"] for r in pod] == [r["clip"] for r in pcl]
    c["probe_columns_equal_pod"] = all(float(a[f"p_seed{s}"]) == float(b[f"p_probe_seed{s}"]) for a, b in zip(pod, pcl) for s in range(5))
    zsv = {r["clip"]: r for r in csv.DictReader(open(f"{OF}/omni_{ds}_zeroshot_scores.csv"))}
    c["zeroshot_equal_saved_file"] = all(float(r["p_yes_zeroshot_saved"]) == float(zsv[r["clip"]]["p_yes"]) for r in pcl)
    c["labels_speakers_equal_saved"] = all(r["label"] == zsv[r["clip"]]["label"] and r["speaker"] == zsv[r["clip"]]["speaker"] for r in pcl)
    for s in range(5):
        fn = [f for f in os.listdir(FD) if f.startswith(f"{ds}_groupkfold5_pod_seed{s}")][0]
        ff = {r["clip_id"]: r["fold"] for r in csv.DictReader(open(f"{FD}/{fn}"))}
        c[f"fold_seed{s}_equal_{fn}"] = all(ff[r["clip"]] == r[f"fold_seed{s}"] for r in pcl)
    y = np.array([int(r["label"]) for r in pcl]); spk = np.array([r["speaker"] for r in pcl])
    cols = [np.array([float(r[f"p_probe_seed{s}"]) for r in pcl]) for s in range(5)]
    zs = np.array([float(r["p_yes_zeroshot_saved"]) for r in pcl])
    per = [mw(y, v) for v in cols]; m5 = float(np.mean(per)); za = mw(y, zs)
    T = json.load(open(f"{OF}/omni_{ds}_nested_repeats.json"))["enc"]
    sc = json.load(open(f"{R}/per_clip/A_{ds}_perclip.sidecar.json"))
    c["per_repeat_equal_sidecar_1e-12"] = bool(np.allclose(per, sc["probe_per_repeat_rerun"], atol=1e-12, rtol=0))
    c["table1_read_correctly"] = sc["probe_mean5_table1"] == T["mean"] and sc["probe_per_repeat_table1"] == T["per_repeat"]
    d = rebuild(y, spk, cols, zs); ok = ~np.isnan(d)
    D = np.load(f"{R}/draws/A_{ds}_draws.npz", allow_pickle=True)
    c["draws_equal_rebuilt_1e-9"] = bool(np.array_equal(np.isnan(D["diff"]), ~ok) and np.allclose(D["diff"][ok], d[ok], atol=1e-9, rtol=0))
    c["n_draws_2000"] = int(len(D["diff"])) == 2000
    lo, hi = float(np.percentile(d[ok], 2.5)), float(np.percentile(d[ok], 97.5))
    c["ci_equal_sidecar_1e-9"] = abs(lo - sc["gap_ci"][0]) < 1e-9 and abs(hi - sc["gap_ci"][1]) < 1e-9
    t = tsv[NM[ds]]
    c["tsv_mean5_4dp"] = t["probe_mean5_rerun"] == f"{m5:.4f}"; c["tsv_zeroshot_4dp"] = t["zeroshot_auc"] == f"{za:.4f}"
    c["tsv_gap_4dp"] = t["gap_point_rerun"] == f"{m5 - za:+.4f}"; c["tsv_ci_4dp"] = (t["gap_lo"], t["gap_hi"]) == (f"{lo:.4f}", f"{hi:.4f}")
    c["tsv_table1_4dp"] = t["probe_mean5_table1"] == f"{T['mean']:.4f}"
    c["tsv_table1_gap_4dp"] = t["gap_point_table1"] == f"{T['mean'] - za:+.4f}"
    ps = all(v for k, v in c.items() if isinstance(v, bool)); allpass &= ps
    res[ds] = {"PASS": ps, "checks": c, "recomputed": {"per_repeat": per, "mean5": m5, "mean5_4dp": round(m5, 4), "table1": T["mean"],
               "minus_table1": m5 - T["mean"], "zeroshot": za, "gap": m5 - za, "ci": [lo, hi], "usable": int(ok.sum())}}
    print(ds, "PASS" if ps else "FAIL", {k: v for k, v in c.items() if v is not True}, f"mean5 {m5:.4f} gap {m5-za:+.4f} [{lo:.4f}, {hi:.4f}]", flush=True)
z = np.load("<local data dir>/release/overnight2/part10/pitt_enc_nested5_oof.npz", allow_pickle=True)
zsv = {r["clip"]: float(r["p_yes"]) for r in csv.DictReader(open(f"{OF}/omni_pitt_zeroshot_scores.csv"))}
yP = z["label"].astype(int); sP = z["spk"].astype(str); zsP = np.array([zsv[str(n)] for n in z["name"]])
dP = rebuild(yP, sP, [z["oof"][r] for r in range(5)], zsP); okP = ~np.isnan(dP)
gP = float(np.mean([mw(yP, z["oof"][r]) for r in range(5)]) - mw(yP, zsP))
pr = [round(gP, 4), round(float(np.percentile(dP[okP], 2.5)), 4), round(float(np.percentile(dP[okP], 97.5)), 4)]
res["pitt_reference"] = {"recomputed": pr, "PASS": pr == [0.1128, 0.0455, 0.1768]}; allpass &= res["pitt_reference"]["PASS"]
res["ALL_PASS"] = bool(allpass); print("pitt", pr, "ALL_PASS", allpass)
prev = json.load(open(f"{R}/verify/A_verify_r2.json")) if os.path.exists(f"{R}/verify/A_verify_r2.json") else {}
prev.update(res); prev["ALL_PASS_last_run"] = bool(allpass)
json.dump(prev, open(f"{R}/verify/A_verify_r2.json", "w"), indent=1)
