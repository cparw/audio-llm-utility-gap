import re, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

H = os.path.expanduser("~")
P23 = "scores/part23"
P24 = "scores/part24"
R = H + "/Desktop/release"

def fnum(x):
    if isinstance(x, str):
        m = re.search(r"[-+0-9.eE]+(?=\)?$)", x.replace("np.float64(", "").rstrip(")"))
        return float(x.replace("np.float64(", "").rstrip(")"))
    return float(x)

def boot(spk, y, s, s2=None, B=2000):
    spk = np.asarray(spk).astype(str); y = np.asarray(y); s = np.asarray(s, float)
    u = np.array(sorted(set(spk)))
    idx = {k: np.where(spk == k)[0] for k in u}
    rng = np.random.default_rng(0)
    out = []
    for _ in range(B):
        pick = rng.choice(len(u), size=len(u), replace=True)
        ii = np.concatenate([idx[u[j]] for j in pick])
        if len(set(y[ii])) < 2: continue
        a = roc_auc_score(y[ii], s[ii])
        if s2 is not None: a -= roc_auc_score(y[ii], np.asarray(s2, float)[ii])
        out.append(a)
    return np.percentile(out, 2.5), np.percentile(out, 97.5)

def rep(tag, printed, spk, y, s, s2=None):
    a = roc_auc_score(y, s)
    if s2 is not None: a -= roc_auc_score(y, s2)
    lo, hi = boot(spk, y, s, s2)
    print(f"{tag}: printed {printed} | indep {a:.4f} [{lo:.4f}, {hi:.4f}] n={len(y)} spk={len(set(map(str,spk)))} rounds_to {a:.2f}")
    return a

# 1 M07 E-DAIC zero-shot whole interview
d = pd.read_csv(P23 + "/A/A2_edaic_whole_zeroshot_perclip.csv")
for c in ["p_yes_whole", "p_yes_300s_T7b"]:
    d[c] = d[c].map(fnum)
rep("M07 A2 p_yes_whole", "0.84", d.speaker, d.label, d.p_yes_whole)
print("   A2 p_yes_300s_T7b AUC", round(roc_auc_score(d.label, d.p_yes_300s_T7b), 4))
o = pd.read_csv(R + "/edaic_rerun/variants/o25_full_zeroshot_scores.csv")
print("   matcher file o25_full_zeroshot_scores.csv AUC", round(roc_auc_score(o.label, o.p_yes), 4),
      "json auc", json.load(open(R + "/edaic_rerun/variants/o25_full_zeroshot.json"))["auc"])
m = o.merge(d[["pid", "p_yes_300s_T7b"]], left_on="speaker", right_on="pid")
print("   o25_full p_yes vs A2 p_yes_300s_T7b max abs diff", float((m.p_yes - m.p_yes_300s_T7b).abs().max()))

# 2 M08 E-DAIC FT seed 0
f = pd.read_csv(P24 + "/FT/FT24_whole_seed0_perclip.csv")
rep("M08 FT24 seed0 pooled OOF", "0.86", f.pid, f.label, f.p_yes)
s = pd.read_csv(R + "/edaic_rerun/variants/sft_full_oof.csv")
print("   matcher file sft_full_oof.csv AUC", round(roc_auc_score(s.label, s.p_yes), 4),
      "json", json.load(open(R + "/edaic_rerun/variants/sft_full_sft.json"))["auc_oof"],
      "window_seconds", json.load(open(R + "/edaic_rerun/variants/sft_full_sft.json"))["window_seconds"])

# 3 M25 readout cosine
rd = pd.read_csv(R + "/omni/readout_direction.csv")
print("M25 readout_direction.csv pitt row:", rd[rd.dataset == "pitt"][["cosine", "cosine_random", "auc_d", "auc_probe", "auc_probe_without_d"]].to_dict("records"))

# 4-5 AF2 Pitt first 30 s
for arm, pr, lab in [("conflict", "0.58", "M45"), ("agreement", "0.53", "M46")]:
    a = pd.read_csv(R + f"/scores/part20/POD4c/af2_orig30_{arm}.csv")
    rep(f"{lab} AF2 orig30 {arm}", pr, a.speaker_id, a.label, a.p_yes)
    print("   scored_s unique", sorted(a.scored_s.unique())[:5], "arm col", a.arm.unique())
    b = pd.read_csv("<local data dir>/paper1_local_runs/af2_results/af2_pitt468.csv")
    b = b[b.clip_path.map(lambda p: os.path.basename(p).startswith(arm))]
    print(f"   matcher af2_pitt468.csv {arm} AUC", round(roc_auc_score(b.label, b.p_yes), 4), "n", len(b))

# 6-7 AF3 Pitt part10 vs part3
for part in ["part10", "part3"]:
    a = pd.read_csv(R + f"/overnight2/{part}/af3_pitt.csv")
    a["arm"] = a.clip_path.map(lambda p: os.path.basename(p).split("_")[0])
    for arm, pr, lab in [("conflict", "0.61", "M47"), ("agreement", "0.59", "M48")]:
        b = a[a.arm == arm]
        if part == "part10":
            rep(f"{lab} AF3 {part} {arm}", pr, b.speaker_id, b.label, b.p_yes)
        else:
            print(f"   {lab} matcher AF3 {part} {arm} AUC", round(roc_auc_score(b.label, b.p_yes), 4), "n", len(b))
    print("   arms", a.arm.value_counts().to_dict())

# extra MISPAIRING 8: M01 median answer mass floor
m9 = pd.read_csv(R + "/edaic_rerun/part16/M9_answer_mass.csv")
zs = m9[m9.stream.str.contains("zero", case=False)]
print("M01 M9 zero-shot rows", len(zs), "min median_mass", round(zs.median_mass.min(), 4),
      zs.loc[zs.median_mass.idxmin(), ["model", "dataset"]].tolist(), "| printed 'at least 0.89'")
print("   whole-interview A2 median answer_mass_whole", round(d.answer_mass_whole.map(fnum).median(), 4))

# extra 9: M09 mid 30 s
mm = pd.read_csv(R + "/edaic_rerun/variants/o25_mid30_zeroshot_scores.csv")
rep("M09 o25_mid30", "0.72", mm.speaker, mm.label, mm.p_yes)

# extra 10: M27/M28/M30/M33 A3 mean of five
a3 = pd.read_csv(P23 + "/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv")
a3["zs"] = a3.p_yes_whole.map(fnum)
for k, pr, lab in [("llm", "0.74", "M27"), ("enc", "0.60", "M28")]:
    a3[k] = a3[[f"oof_{k}_r{i}" for i in range(5)]].mean(axis=1)
    per = [roc_auc_score(a3.label, a3[f"oof_{k}_r{i}"]) for i in range(5)]
    print(f"{lab} {k}: mean of 5 per-repeat AUCs {np.mean(per):.4f}; AUC of averaged score {roc_auc_score(a3.label, a3[k]):.4f}; printed {pr}")
rep("M30 llm minus answer (averaged score)", "-0.09 [-0.16,-0.03]", a3.speaker, a3.label, a3["llm"], a3.zs)
rep("M33 enc minus answer (averaged score)", "-0.24 [-0.32,-0.15]", a3.speaker, a3.label, a3["enc"], a3.zs)

# mean-of-5 paired differences, bootstrap of mean over repeats
def boot_m5(k):
    y = a3.label.values; zs = a3.zs.values
    S = [a3[f"oof_{k}_r{i}"].values for i in range(5)]
    pt = np.mean([roc_auc_score(y, s) for s in S]) - roc_auc_score(y, zs)
    u = np.array(sorted(set(a3.speaker.astype(str)))); spk = a3.speaker.astype(str).values
    idx = {x: np.where(spk == x)[0] for x in u}
    rng = np.random.default_rng(0); out = []
    for _ in range(2000):
        ii = np.concatenate([idx[u[j]] for j in rng.choice(len(u), size=len(u), replace=True)])
        out.append(np.mean([roc_auc_score(y[ii], s[ii]) for s in S]) - roc_auc_score(y[ii], zs[ii]))
    print(f"   {k} mean-of-5 minus answer {pt:.4f} [{np.percentile(out,2.5):.4f}, {np.percentile(out,97.5):.4f}]")
boot_m5("llm"); boot_m5("enc")
