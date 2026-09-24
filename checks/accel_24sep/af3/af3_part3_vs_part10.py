"""part3 (Sep 19 08:38, same af3_score.py, first 30 s) vs part10 (Sep 19 18:50, first 30 s): per-clip and per-arm AUC with the same bootstrap."""
import csv, os
import numpy as np
from sklearn.metrics import roc_auc_score
OUTD = os.path.dirname(os.path.abspath(__file__))
R = "<local data dir>/release/overnight2"
p3 = {os.path.basename(r["clip_path"]): r for r in csv.DictReader(open(R + "/part3/af3_pitt.csv"))}
p10 = {os.path.basename(r["clip_path"]): r for r in csv.DictReader(open(R + "/part10/af3_pitt.csv"))}
ARM = {os.path.basename(r["segment_path"]): r["set"] for r in csv.DictReader(open("<local data dir>/DementiaBank/pitt_conflict_manifest.csv"))}
assert set(p3) == set(p10) and len(p3) == 468
ks = sorted(p3)
d = np.array([abs(float(p3[k]["p_yes"]) - float(p10[k]["p_yes"])) for k in ks])
print(f"|part3 - part10| p_yes: median {np.median(d):.2e} max {d.max():.4f} share>1e-3 {np.mean(d>1e-3):.3f} share>0.01 {np.mean(d>0.01):.3f}")
print("labels equal:", all(p3[k]["label"] == p10[k]["label"] for k in ks), "speakers equal:", all(p3[k]["speaker_id"] == p10[k]["speaker_id"] for k in ks))
def boot(spk, y, s, s2=None):
    rng = np.random.default_rng(0); u = np.unique(spk); idx = {k: np.flatnonzero(spk == k) for k in u}; v = []
    for _ in range(2000):
        ii = np.concatenate([idx[k] for k in rng.choice(u, size=len(u), replace=True)])
        if not (0 < y[ii].sum() < len(ii)): continue
        a = roc_auc_score(y[ii], s[ii]) - (roc_auc_score(y[ii], s2[ii]) if s2 is not None else 0)
        v.append(a)
    return np.percentile(v, 2.5), np.percentile(v, 97.5), len(v)
rows = []
for arm in ("all", "conflict", "agreement"):
    kk = [k for k in ks if arm == "all" or ARM[k] == arm]
    spk = np.array([p10[k]["speaker_id"] for k in kk]); y = np.array([int(p10[k]["label"]) for k in kk])
    a = np.array([float(p3[k]["p_yes"]) for k in kk]); b = np.array([float(p10[k]["p_yes"]) for k in kk])
    for cell, s, s2 in (("part3_first30s", a, None), ("part3_minus_part10", a, b)):
        val = roc_auc_score(y, s) - (roc_auc_score(y, s2) if s2 is not None else 0)
        lo, hi, nu = boot(spk, y, s, s2)
        rows.append((cell, arm, len(kk), len(set(spk)), round(val, 4), round(lo, 4), round(hi, 4), nu))
        print(f"{cell:20s} {arm:9s} n={len(kk)} spk={len(set(spk))} {val:.4f} [{lo:.4f}, {hi:.4f}] usable={nu}")
with open(os.path.join(OUTD, "af3_part3_vs_part10.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cell", "arm", "n", "n_speakers", "value", "ci_lo", "ci_hi", "usable_draws"]); w.writerows(rows)
