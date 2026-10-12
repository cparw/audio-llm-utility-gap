"""Recompute AF3 Pitt AUCs per arm from per-clip files, with a 2000-draw speaker bootstrap:
fresh numpy default_rng(0) per cell, speakers resampled with replacement, 2.5/97.5 percentiles.
Paired cells use one speaker draw per replicate and recompute both AUCs on it.
Inputs:
  part10  <local data dir>/release/overnight2/part10/af3_pitt.csv   (original 468 windows, the Table 2 cells)
  noinv   POD4c/work/af3_noinv468.csv   (interviewer-free PAR-only cut, the values quoted as 0.5920/0.5417/0.6140)
  arms    <local data dir>/DementiaBank/pitt_conflict_manifest.csv, column 'set', key basename(segment_path)"""
import csv, os, sys
import numpy as np
from scipy.stats import rankdata
P10 = "<local data dir>/release/overnight2/part10/af3_pitt.csv"
NOINV = "scores/part20/POD4c/work/af3_noinv468.csv"
MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
POD4C = "scores/part20/POD4c"
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
p10 = {os.path.basename(r["clip_path"]): r for r in csv.DictReader(open(P10))}
ni = {r["orig_clip_id"]: r for r in csv.DictReader(open(NOINV))}
assert len(p10) == 468 and set(p10) == set(ni) and set(p10) <= set(ARM)
for k in p10:
    assert int(p10[k]["label"]) == int(ni[k]["label"]) and p10[k]["speaker_id"] == ni[k]["speaker_id"]
# cross-check against the POD4c per-arm per-clip files
for arm in ("all", "conflict", "agreement"):
    f = {r["orig_clip_id"]: float(r["p_yes"]) for r in csv.DictReader(open(f"{POD4C}/af3_noinv_{arm}.csv"))}
    g = {r["orig_clip_id"]: float(r["p_yes"]) for r in csv.DictReader(open(f"{POD4C}/af3_orig_part10_{arm}.csv"))}
    assert all(abs(f[k] - float(ni[k]["p_yes"])) < 1e-12 for k in f) and all(abs(g[k] - float(p10[k]["p_yes"])) < 1e-12 for k in g)
    print(f"POD4c per-arm files match the source per-clip files: {arm} noinv n={len(f)} part10 n={len(g)}")
dur = [float(r["dur_s"]) for r in ni.values()]; sc = [float(r["scored_s"]) for r in ni.values()]
print(f"interviewer-free clips: dur_s min {min(dur):.3f} median {np.median(dur):.3f} max {max(dur):.3f}; scored_s max {max(sc):.3f}; clips cut by the 30 s window: {sum(1 for a, b in zip(dur, sc) if b < a)}")
rows = []
for arm in ("all", "conflict", "agreement"):
    ks = sorted(k for k in p10 if arm == "all" or ARM[k] == arm)
    spk = np.array([p10[k]["speaker_id"] for k in ks]); y = np.array([int(p10[k]["label"]) for k in ks])
    a = np.array([float(p10[k]["p_yes"]) for k in ks]); b = np.array([float(ni[k]["p_yes"]) for k in ks])
    for cell, s, s2 in (("part10_original_first30s", a, None), ("noinv_interviewer_free", b, None),
                        ("noinv_minus_part10_paired", b, a)):
        val = auc(y, s) if s2 is None else auc(y, s) - auc(y, s2)
        lo, hi, nu = boot(spk, y, s, s2)
        rows.append((cell, arm, len(ks), len(set(spk)), int(y.sum()), int((y == 0).sum()), round(val, 4), round(lo, 4), round(hi, 4), nu))
        print(f"{cell:28s} {arm:9s} n={len(ks)} spk={len(set(spk))} AUC={val:.4f} [{lo:.4f}, {hi:.4f}] usable={nu}")
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cell", "arm", "n", "n_speakers", "n_pos", "n_neg", "value", "ci_lo", "ci_hi", "usable_draws"]); w.writerows(rows)
