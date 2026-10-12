"""Context cell: the one AF3 Pitt run on disk that fed the WHOLE 30 to 56 s window (no slicing):
<local data dir>/paper1_local_runs/runpod_results/pitt468_af3.csv, written by the Sep 15/16
af3_score.py (score(): librosa.load(p, sr=16000, mono=True) then {"type":"audio","audio":x}, no x[:...]).
Per arm AUC and the paired difference full minus part10 (first 30 s), same bootstrap as boot_af3_pitt.py."""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import rankdata
FULL = "<local data dir>/paper1_local_runs/runpod_results/pitt468_af3.csv"
P10 = "<local data dir>/release/overnight2/part10/af3_pitt.csv"
MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
NB = 2000
def auc(y, s):
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(s); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def boot(spk, y, s, s2=None):
    rng = np.random.default_rng(0)
    u = np.unique(spk); idx = {k: np.where(spk == k)[0] for k in u}; v = []
    for _ in range(NB):
        ii = np.concatenate([idx[k] for k in rng.choice(u, size=len(u), replace=True)])
        a = auc(y[ii], s[ii])
        if s2 is None:
            if not np.isnan(a): v.append(a)
        else:
            b = auc(y[ii], s2[ii])
            if not (np.isnan(a) or np.isnan(b)): v.append(a - b)
    v = np.array(v); return np.percentile(v, 2.5), np.percentile(v, 97.5), len(v)
ARM = {os.path.basename(r["segment_path"]): r["set"] for r in csv.DictReader(open(MAN))}
fu = {os.path.basename(r["clip_path"]): r for r in csv.DictReader(open(FULL))}
p10 = {os.path.basename(r["clip_path"]): r for r in csv.DictReader(open(P10))}
assert set(fu) == set(p10) and len(fu) == 468
for k in fu: assert int(fu[k]["label"]) == int(p10[k]["label"]) and fu[k]["speaker_id"] == p10[k]["speaker_id"]
rows = []
for arm in ("all", "conflict", "agreement"):
    ks = sorted(k for k in fu if arm == "all" or ARM[k] == arm)
    spk = np.array([fu[k]["speaker_id"] for k in ks]); y = np.array([int(fu[k]["label"]) for k in ks])
    a = np.array([float(fu[k]["p_yes"]) for k in ks]); b = np.array([float(p10[k]["p_yes"]) for k in ks])
    for cell, s, s2 in (("fullseg_sep16", a, None), ("fullseg_minus_part10_paired", a, b)):
        val = auc(y, s) if s2 is None else auc(y, s) - auc(y, s2)
        lo, hi, nu = boot(spk, y, s, s2)
        rows.append((cell, arm, len(ks), len(set(spk)), round(val, 4), round(lo, 4), round(hi, 4), nu))
        print(f"{cell:30s} {arm:9s} n={len(ks)} spk={len(set(spk))} AUC={val:.4f} [{lo:.4f}, {hi:.4f}] usable={nu}")
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cell", "arm", "n", "n_speakers", "value", "ci_lo", "ci_hi", "usable_draws"]); w.writerows(rows)
