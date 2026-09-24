# Independent recompute of the FINAL Table 1 E-DAIC audio cells (0.84 zero-shot, 0.86 fine-tuned) and the line 226 paired +0.02 [-0.01, 0.05].
import re, numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score
A = "scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv"
B = "scores/part24/FT/FT24_whole_seed0_perclip.csv"
num = lambda v: float(re.sub(r"^np\.float64\((.*)\)$", r"\1", str(v)))
a = pd.read_csv(A); b = pd.read_csv(B)
a["zs"] = a["p_yes_whole"].map(num); a["s300"] = a["p_yes_300s_T7b"].map(num)
m = a[["pid","speaker","label","zs","s300"]].merge(b[["pid","label","p_yes"]].rename(columns={"label":"label_ft"}), on="pid")
assert (m.label == m.label_ft).all() and len(m) == 275
def boot(cols, diff=False):
    rng = np.random.default_rng(0); spk = m.speaker.unique(); idx = {s: np.where(m.speaker.values == s)[0] for s in spk}
    vals = []
    for _ in range(2000):
        dr = rng.choice(spk, size=len(spk), replace=True); ii = np.concatenate([idx[s] for s in dr])
        y = m.label.values[ii]
        if len(set(y)) < 2: continue
        if diff: vals.append(roc_auc_score(y, m[cols[0]].values[ii]) - roc_auc_score(y, m[cols[1]].values[ii]))
        else: vals.append(roc_auc_score(y, m[cols[0]].values[ii]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]
y = m.label.values
out = {"n": len(m), "pos": int(y.sum()), "speakers": int(m.speaker.nunique()),
 "zs_whole": [round(roc_auc_score(y, m.zs), 4), boot(["zs"])],
 "zs_300s": [round(roc_auc_score(y, m.s300), 4), boot(["s300"])],
 "ft_seed0": [round(roc_auc_score(y, m.p_yes), 4), boot(["p_yes"])],
 "ft_minus_zs_whole": [round(roc_auc_score(y, m.p_yes) - roc_auc_score(y, m.zs), 4), boot(["p_yes","zs"], True)]}
print(json.dumps(out))
json.dump(out, open("checks/accel_24sep/edaic_cells/edaic_cells.json","w"), indent=1)
