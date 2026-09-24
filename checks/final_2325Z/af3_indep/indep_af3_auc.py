"""Independent recompute of the AF3 Pitt AUCs and speaker-bootstrap intervals from per-clip files.
AUC: independent Mann-Whitney count (pairs pos>neg plus half ties), checked against sklearn.
Bootstrap: 2000 draws, fresh numpy default_rng(0) per cell, sorted unique speakers resampled with
replacement via rng.integers(0, S, S), rows gathered with multiplicity, 2.5 / 97.5 percentiles.
Paired cells: one speaker draw per iteration, both AUCs on that draw, percentiles of the difference.
Arm: from the clip basename prefix; checked against the DementiaBank manifest 'set' column and the POD4c 'arm' column.
usage: indep_af3_auc.py OUT.csv"""
import csv, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

P10 = "<local data dir>/release/overnight2/part10/af3_pitt.csv"
NOINV = "scores/part20/POD4c/work/af3_noinv468.csv"
FULL = "<local data dir>/paper1_local_runs/runpod_results/pitt468_af3.csv"
MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
B, SEED = 2000, 0

def auc_mw(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    neg_sorted = np.sort(neg)
    lt = np.searchsorted(neg_sorted, pos, side="left")
    le = np.searchsorted(neg_sorted, pos, side="right")
    return float((lt.sum() + 0.5 * (le - lt).sum()) / (len(pos) * len(neg)))

def load(path, key, spk, lab, py):
    d = {}
    for r in csv.DictReader(open(path)):
        k = os.path.basename(r[key])
        assert k not in d, k
        d[k] = (r[spk], int(float(r[lab])), float(r[py]))
    return d

p10 = load(P10, "clip_path", "speaker_id", "label", "p_yes")
ni = load(NOINV, "orig_clip_id", "speaker_id", "label", "p_yes")
fu = load(FULL, "clip_path", "speaker_id", "label", "p_yes")
assert len(p10) == len(ni) == len(fu) == 468 and set(p10) == set(ni) == set(fu)
man_arm = {os.path.basename(r["segment_path"]): r["set"] for r in csv.DictReader(open(MAN))}
ni_arm = {r["orig_clip_id"]: r["arm"] for r in csv.DictReader(open(NOINV))}
clips = sorted(p10)
arm = {c: c.split("_")[0] for c in clips}
n_arm_mismatch = sum(arm[c] != man_arm.get(c) or arm[c] != ni_arm[c] for c in clips)
n_lab_mismatch = sum(len({p10[c][1], ni[c][1], fu[c][1]}) > 1 for c in clips)
n_spk_mismatch = sum(len({p10[c][0], ni[c][0], fu[c][0]}) > 1 for c in clips)
print(f"clips 468 | arm mismatches {n_arm_mismatch} | label mismatches {n_lab_mismatch} | speaker mismatches {n_spk_mismatch}")

def cell_arrays(sub):
    spk = np.array([p10[c][0] for c in sub]); y = np.array([p10[c][1] for c in sub])
    s = {"part10": np.array([p10[c][2] for c in sub]), "noinv": np.array([ni[c][2] for c in sub]),
         "full": np.array([fu[c][2] for c in sub])}
    return spk, y, s

def boot(spk, y, scores_a, scores_b=None):
    rng = np.random.default_rng(SEED)
    uniq = np.array(sorted(set(spk)))
    rows_of = [np.flatnonzero(spk == u) for u in uniq]
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[j] for j in pick])
        yy = y[idx]
        if yy.min() == yy.max(): continue
        a = auc_mw(yy, scores_a[idx])
        vals.append(a if scores_b is None else auc_mw(yy, scores_b[idx]) - a)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)

out = []
for a in ("all", "conflict", "agreement"):
    sub = [c for c in clips if a == "all" or arm[c] == a]
    spk, y, s = cell_arrays(sub)
    for name in ("part10", "noinv", "full"):
        v = auc_mw(y, s[name]); vs = roc_auc_score(y, s[name])
        assert abs(v - vs) < 1e-12, (name, a, v, vs)
        lo, hi, u = boot(spk, y, s[name])
        out.append((name, a, len(sub), len(set(spk)), v, lo, hi, u))
    for (b_, a_) in (("noinv", "part10"), ("full", "part10")):
        v = auc_mw(y, s[b_]) - auc_mw(y, s[a_])
        lo, hi, u = boot(spk, y, s[a_], s[b_])
        out.append((f"{b_}_minus_{a_}_paired", a, len(sub), len(set(spk)), v, lo, hi, u))
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cell", "arm", "n", "n_speakers", "value", "ci_lo", "ci_hi", "usable_draws"])
    for r in out:
        w.writerow(list(r[:4]) + [f"{r[4]:.4f}", f"{r[5]:.4f}", f"{r[6]:.4f}", r[7]])
        print(f"{r[0]:26s} {r[1]:9s} n={r[2]:3d} spk={r[3]:3d}  {r[4]:+.4f} [{r[5]:+.4f}, {r[6]:+.4f}] draws={r[7]}")
print("wrote", sys.argv[1])
