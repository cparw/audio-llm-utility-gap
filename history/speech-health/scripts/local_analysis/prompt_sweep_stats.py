"""Twelve prompt interventions, scored properly.

For each prompt: conflict AUC, agreement AUC, speaker-clustered 95% CI on both,
a label-shuffle null, and the paired difference against the baseline prompt.
Nothing here needs the cluster; it reads the committed per-clip files.
"""
import csv, os, numpy as np
from collections import defaultdict

H = "results/health"

def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if len(set(y)) < 2: return float("nan")
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def boot_ci(y, s, g, B=2000, seed=11):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float); s = np.asarray(s, float); g = np.asarray(g)
    u = np.unique(g); idx = {k: np.where(g == k)[0] for k in u}
    out = []
    for _ in range(B):
        i = np.concatenate([idx[u[k]] for k in rng.choice(len(u), len(u), True)])
        out.append(auc(y[i], s[i]))
    return np.nanpercentile(out, [2.5, 97.5])

def shuffle_null(y, s, g, B=500, seed=12):
    """Permute labels ACROSS speakers, keeping each speaker's clips together.

    Label is a speaker-level property here, so permuting within a speaker is a
    no-op. The correct null reassigns whole speakers to labels.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float); s = np.asarray(s, float); g = np.asarray(g)
    spk = np.unique(g)
    spk_lab = np.array([y[g == k][0] for k in spk])
    idx = {k: np.where(g == k)[0] for k in spk}
    out = []
    for _ in range(B):
        perm = rng.permutation(spk_lab)
        yy = np.empty_like(y)
        for k, lab in zip(spk, perm):
            yy[idx[k]] = lab
        out.append(auc(yy, s))
    return float(np.nanmean(out)), float(np.nanpercentile(out, 95))

def run(model):
    path = f"{H}/posttrain/prompt_intervention_{model}_clips.csv"
    rows = list(csv.DictReader(open(path)))
    byp = defaultdict(list)
    for r in rows: byp[r["prompt_id"]].append(r)
    base = "P00_baseline"
    out = []
    base_conf = None
    for pid in sorted(byp):
        sub = byp[pid]
        res = {"model": model, "prompt": pid}
        for arm in ("conflict", "agreement"):
            a = [r for r in sub if r["set"] == arm]
            y = [int(r["label"]) for r in a]
            s = [float(r["p_dep"]) for r in a]
            g = [r["speaker"] for r in a]
            v = auc(y, s); lo, hi = boot_ci(y, s, g)
            res[f"{arm}_auc"] = round(v, 3)
            res[f"{arm}_lo"] = round(lo, 3)
            res[f"{arm}_hi"] = round(hi, 3)
            if arm == "conflict":
                nm, n95 = shuffle_null(y, s, g)
                res["conflict_null_mean"] = round(nm, 3)
                res["conflict_null_p95"] = round(n95, 3)
                if pid == base: base_conf = v
        res["conflict_vs_baseline"] = round(res["conflict_auc"] - base_conf, 3) if base_conf is not None else None
        out.append(res)
    return out

allrows = []
for m in ("omni", "qwen2audio"):
    allrows += run(m)

os.makedirs(f"{H}/local", exist_ok=True)
with open(f"{H}/local/prompt_sweep_stats.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)

hdr = f"{'model':11s} {'prompt':26s} {'conflict':>8s} {'95% CI':>15s} {'null p95':>9s} {'agreement':>9s} {'vs base':>8s}"
print(hdr); print("-" * len(hdr))
for r in allrows:
    print(f"{r['model']:11s} {r['prompt']:26s} {r['conflict_auc']:8.3f} "
          f"[{r['conflict_lo']:.3f},{r['conflict_hi']:.3f}] {r['conflict_null_p95']:9.3f} "
          f"{r['agreement_auc']:9.3f} {r['conflict_vs_baseline']:+8.3f}")
print(f"\n{len(allrows)} prompt cells -> results/health/local/prompt_sweep_stats.csv")
