"""Recompute AUC (rank formula) + speaker bootstrap for any pitt per-clip csv.
usage: eval_pitt.py <csv> <label_col> <score_col> <spk_col> <set_col|NONE> <tag>"""
import sys, csv, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import auc_rank, boot_spk, boot_paired_diff

f, lc, sc, kc, setc, tag = sys.argv[1:7]
rows = list(csv.DictReader(open(f)))
y = np.array([int(float(r[lc])) for r in rows])
s = np.array([float(r[sc]) for r in rows])
spk = np.array([r[kc] for r in rows])
out = {"file": os.path.abspath(f), "tag": tag, "n": len(rows), "n_spk": int(len(set(spk)))}
out["auc_all"] = round(auc_rank(y, s), 6)
lo, hi, k = boot_spk(y, s, spk, 2000, 0)
out["auc_all_ci"] = [round(lo, 6), round(hi, 6)]; out["auc_all_draws"] = k
if setc != "NONE":
    st = np.array([r[setc] for r in rows])
    for arm in ["conflict", "agreement"]:
        m = st == arm
        out[f"auc_{arm}"] = round(auc_rank(y[m], s[m]), 6)
        out[f"n_{arm}"] = int(m.sum()); out[f"nspk_{arm}"] = int(len(set(spk[m])))
        l2, h2, k2 = boot_spk(y[m], s[m], spk[m], 2000, 0)
        out[f"auc_{arm}_ci"] = [round(l2, 6), round(h2, 6)]; out[f"auc_{arm}_draws"] = k2
    out["diff_conflict_minus_agreement"] = round(out["auc_conflict"] - out["auc_agreement"], 6)
    dl, dh, dk = boot_paired_diff(y, s, spk, st == "conflict", st == "agreement", 2000, 0)
    out["diff_ci_paired"] = [round(dl, 6), round(dh, 6)]; out["diff_draws"] = dk
print(json.dumps(out, indent=1))
