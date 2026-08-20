"""Lexical capture outside the conflict set.

The conflict test only exists for depression, because the parkinson's tasks are
fixed read sentences with no naturally contradicting words. This gives the same
measurement on all five datasets, using the text-injection files: score the same
audio with a 'sick' sentence and a 'healthy' sentence injected before the question
and see how far the answer moves.

CAVEAT that must travel with any flip rate: a model saturated on one answer cannot
flip, so a low flip rate is only interpretable for a model that is actually deciding.
"""
import csv, glob, os, sys, numpy as np

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "results", "health", "reliance_qwen")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "health", "local")

def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    m = ~(np.isnan(y) | np.isnan(s)); y, s = y[m], s[m]
    if len(set(y)) < 2: return float("nan")
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def boot(y, s, g, B=2000, seed=5):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float); s = np.asarray(s, float); g = np.asarray(g)
    u = np.unique(g); idx = {k: np.where(g == k)[0] for k in u}
    out = []
    for _ in range(B):
        i = np.concatenate([idx[u[k]] for k in rng.choice(len(u), len(u), True)])
        out.append(auc(y[i], s[i]))
    return np.nanpercentile(out, [2.5, 97.5])

def f(x):
    try: return float(x)
    except Exception: return np.nan

rows = []
for path in sorted(glob.glob(os.path.join(SRC, "*_clips.csv"))):
    r = list(csv.DictReader(open(path)))
    if not r: continue
    y = [f(x["label"]) for x in r]
    g = [x["speaker"] for x in r]
    bd = [f(x["behav_diag"]) for x in r]
    sick = np.array([f(x["rel_sick"]) for x in r])
    heal = np.array([f(x["rel_healthy"]) for x in r])
    flip = np.array([f(x["flip"]) for x in r])
    mass = np.array([f(x["behav_mass"]) for x in r])
    lo, hi = boot(y, bd, g)
    rows.append(dict(
        cell=os.path.basename(path).replace("_clips.csv", ""),
        n=len(r), auc_no_text=round(auc(y, bd), 4),
        ci_low=round(lo, 4), ci_high=round(hi, 4),
        text_shift=round(float(np.nanmean(np.abs(sick - heal))), 4),
        flip_rate=round(float(np.nanmean(flip)), 4),
        answer_mass=round(float(np.nanmean(mass)), 4),
    ))

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "lexical_capture_grid.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
