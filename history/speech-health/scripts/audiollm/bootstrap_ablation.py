"""
Speaker-clustered bootstrap CIs for the audio-ablation benchmark.

Two questions the point estimates cannot answer:
  1) is each condition's AUC actually above chance?
  2) is the difference between conditions (e.g. reversed vs orig) real or noise?

Resamples whole SPEAKERS with replacement (2000x), matching the probing-floor
methodology, so correlated clips from one person move together.
"""
import csv, json, glob, os, argparse
import numpy as np
from sklearn.metrics import roc_auc_score

def auc_safe(y, s):
    try:
        if len(set(y)) < 2: return np.nan
        return roc_auc_score(y, s)
    except Exception:
        return np.nan

def boot(y, S, spk, conds, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    uspk = np.unique(spk)
    idx_by = {u: np.where(spk == u)[0] for u in uspk}
    draws = {c: [] for c in conds}
    diffs = {c: [] for c in conds if c != "orig"}
    for _ in range(n):
        pick = rng.choice(uspk, size=len(uspk), replace=True)
        ii = np.concatenate([idx_by[u] for u in pick])
        yy = y[ii]
        if len(set(yy)) < 2: continue
        base = auc_safe(yy, S["orig"][ii]) if "orig" in S else np.nan
        for c in conds:
            a = auc_safe(yy, S[c][ii]); draws[c].append(a)
            if c != "orig" and not np.isnan(a) and not np.isnan(base):
                diffs[c].append(a - base)
    out = {}
    for c in conds:
        d = np.array([v for v in draws[c] if v == v])
        out[c] = dict(auc_lo=round(float(np.percentile(d, 2.5)), 3),
                      auc_hi=round(float(np.percentile(d, 97.5)), 3)) if len(d) else {}
    for c, d in diffs.items():
        d = np.array(d)
        if len(d):
            lo, hi = np.percentile(d, [2.5, 97.5])
            out[c]["diff_vs_orig"] = round(float(np.mean(d)), 3)
            out[c]["diff_lo"] = round(float(lo), 3)
            out[c]["diff_hi"] = round(float(hi), 3)
            out[c]["diff_significant"] = bool(lo > 0 or hi < 0)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    R = "/project2/msoleyma_946/speech_health/results_chaitanya"
    report = {}
    print("=" * 118)
    print("AUDIO-ABLATION with speaker-clustered bootstrap 95% CIs (no text supplied to the model)")
    print("=" * 118)
    for f in sorted(glob.glob(f"{R}/ablation/*_clips.csv")):
        tag = os.path.basename(f).replace("_clips.csv", "")
        rows = list(csv.DictReader(open(f)))
        if not rows: continue
        conds = [k[2:] for k in rows[0] if k.startswith("p_")]
        y = np.array([int(r["label"]) for r in rows])
        spk = np.array([r["speaker"] for r in rows])
        S = {c: np.array([float(r[f"p_{c}"]) for r in rows]) for c in conds}
        pt = {c: auc_safe(y, S[c]) for c in conds}
        ci = boot(y, S, spk, conds, n=a.n)
        report[tag] = {"n_clips": len(rows), "n_speakers": int(len(set(spk))),
                       "point": {c: (round(pt[c], 3) if pt[c] == pt[c] else None) for c in conds},
                       "ci": ci}
        print(f"\n{tag}   ({len(rows)} clips, {len(set(spk))} speakers)")
        for c in conds:
            d = ci.get(c, {})
            s = f"   {c:9s} AUC {pt[c]:.3f}  95% CI [{d.get('auc_lo')}, {d.get('auc_hi')}]"
            if "diff_vs_orig" in d:
                sig = "SIGNIFICANT" if d["diff_significant"] else "not significant"
                s += f"   vs orig {d['diff_vs_orig']:+.3f} [{d['diff_lo']}, {d['diff_hi']}] {sig}"
            above = "above chance" if d.get("auc_lo", 0) and d["auc_lo"] > 0.5 else "NOT above chance"
            print(s + f"   ({above})")
    os.makedirs(f"{R}/ablation", exist_ok=True)
    json.dump(report, open(f"{R}/ablation/bootstrap_cis.json", "w"), indent=2)
    print(f"\nwrote {R}/ablation/bootstrap_cis.json")

if __name__ == "__main__":
    main()
