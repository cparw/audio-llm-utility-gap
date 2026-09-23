"""Part 20 POD4 analysis. AUC by the rank formula (scipy rankdata, ties averaged).
Bootstrap: 2000 draws, numpy.random.default_rng(0) created fresh per cell, speakers (np.unique order)
resampled with replacement, percentile 2.5/97.5, a draw is usable only if both classes are present.
Per-arm cells: subset the arm first, then resample the speakers of that subset (as in part16 boot.py).
PAIRED: one speaker draw per replicate, both AUCs in that draw, diff = AUC(a) - AUC(b).

usage: analyze.py SPEC_JSON   (spec: result name, per-clip csv, sources, cells, paired; see drive)
writes <outdir>/<name>.sidecar.json and prints one paste line per value.
"""
import csv, json, sys, os, hashlib, datetime
import numpy as np
from scipy.stats import rankdata

NB = 2000

def auc_rank(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def arm_of(r):
    for k in ("arm", "set"):
        if r.get(k) in ("conflict", "agreement"): return r[k]
    c = r["clip"]
    return "conflict" if c.startswith("conflict_") else ("agreement" if c.startswith("agreement_") else "?")

def load(path, col):
    d = {}
    for r in csv.DictReader(open(path)):
        d[r["clip"]] = (r["speaker"], int(float(r["label"])), float(r[col]), arm_of(r))
    return d

def keys_for(d, arm):
    return sorted(k for k in d if arm is None or d[k][3] == arm)

def boot_one(spk, y, s):
    rng = np.random.default_rng(0)
    u = np.unique(spk); idx = {p: np.where(spk == p)[0] for p in u}
    v = []
    for _ in range(NB):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        a = auc_rank(y[ii], s[ii])
        if not np.isnan(a): v.append(a)
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

def boot_pair(spk, y, sa, sb):
    rng = np.random.default_rng(0)
    u = np.unique(spk); idx = {p: np.where(spk == p)[0] for p in u}
    v = []
    for _ in range(NB):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        a = auc_rank(y[ii], sa[ii]); b = auc_rank(y[ii], sb[ii])
        if np.isnan(a) or np.isnan(b): continue
        v.append(a - b)
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

def sha1mb(p):
    with open(p, "rb") as fh: return hashlib.sha256(fh.read(1 << 20)).hexdigest()

def main(spec_path):
    S = json.load(open(spec_path))
    out = dict(result=S["name"], per_clip_csv=S["per_clip_csv"], per_clip_csv_mac=S.get("per_clip_csv_mac"),
               model=S["model"], dtype=S["dtype"], prompt=S["prompt"], command=S["command"], seed=0, n_boot=NB,
               bootstrap="speakers resampled with replacement, fresh numpy.random.default_rng(0) per cell, percentile 2.5/97.5; per arm: subset then resample; PAIRED: one speaker draw per replicate",
               auc="rank formula, scipy.stats.rankdata (average ties)",
               p_yes="P(Yes)/(P(Yes)+P(No)) at the first answer position; paper id set (first token of Yes, Yes, yes, yes, YES / No, No, no, no, NO with and without leading space as in extract_probe_layers.py); p_yes_r3 column = rule-3 single-token set incl. leading-space YES/NO",
               sources=[], cells=[], paired=[], date_utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z")
    for src in S["sources"]:
        e = dict(src)
        if os.path.isfile(src["pod_path"]): e["sha256_first_1MB"] = sha1mb(src["pod_path"])
        out["sources"].append(e)
    lines = []
    for c in S["cells"]:
        d = load(c["file"], c.get("col", "p_yes"))
        k = keys_for(d, c.get("arm"))
        spk = np.array([d[i][0] for i in k]); y = np.array([d[i][1] for i in k]); s = np.array([d[i][2] for i in k])
        a = auc_rank(y, s); lo, hi, nu = boot_one(spk, y, s)
        vs = "above 0.5" if a > 0.5 else "below 0.5"
        ci = "interval excludes 0.5" if (lo > 0.5 or hi < 0.5) else "interval includes 0.5"
        rec = dict(id=c["id"], what=c["what"], value=round(a, 4), lo=round(lo, 4), hi=round(hi, 4), n=len(k),
                   n_spk=int(len(np.unique(spk))), n_pos=int(y.sum()), usable_draws=nu, file=c["file_mac"],
                   col=c.get("col", "p_yes"), arm=c.get("arm") or "overall", two_dp=f"{a:.2f}", vs_half=vs, interval_vs_half=ci)
        out["cells"].append(rec)
        lines.append(f"{rec['id']}: {rec['what']} = {a:.4f} [{lo:.4f}, {hi:.4f}] n={len(k)} n_spk={rec['n_spk']} file={c['file_mac']} || {a:.2f}, {vs} ({ci})")
    for p in S.get("paired", []):
        da = load(p["a"], p.get("col_a", "p_yes")); db = load(p["b"], p.get("col_b", "p_yes"))
        k = sorted(i for i in set(da) & set(db) if p.get("arm") is None or da[i][3] == p["arm"])
        for i in k:
            assert da[i][0] == db[i][0] and da[i][1] == db[i][1] and da[i][3] == db[i][3], i
        spk = np.array([da[i][0] for i in k]); y = np.array([da[i][1] for i in k])
        sa = np.array([da[i][2] for i in k]); sb = np.array([db[i][2] for i in k])
        A, B = auc_rank(y, sa), auc_rank(y, sb); lo, hi, nu = boot_pair(spk, y, sa, sb)
        ex = "interval excludes zero" if (lo > 0 or hi < 0) else "interval includes zero"
        rec = dict(id=p["id"], what=p["what"], value=round(A - B, 4), lo=round(lo, 4), hi=round(hi, 4), n=len(k),
                   n_spk=int(len(np.unique(spk))), usable_draws=nu, a_auc=round(A, 4), b_auc=round(B, 4),
                   file=f"{p['a_mac']} MINUS {p['b_mac']}", arm=p.get("arm") or "overall",
                   two_dp=f"{A-B:+.2f}", vs_half=ex)
        out["paired"].append(rec)
        lines.append(f"{rec['id']}: {rec['what']} = {A-B:+.4f} [{lo:+.4f}, {hi:+.4f}] ({A:.4f} minus {B:.4f}) n={len(k)} n_spk={rec['n_spk']} file={rec['file']} || {A-B:+.2f}, {ex}")
    d = load(S["per_clip_csv"], "p_yes")
    out["n"] = len(d); out["n_speakers"] = int(len({v[0] for v in d.values()}))
    if S.get("mass_col"):
        m = [float(r["mass"]) for r in csv.DictReader(open(S["per_clip_csv"]))]
        out["answer_mass"] = dict(median=round(float(np.median(m)), 4), min=round(float(np.min(m)), 4), n_below_0p5=int(sum(x < 0.5 for x in m)))
    out["paste_lines"] = lines
    json.dump(out, open(S["sidecar"], "w"), indent=1)
    for l in lines: print("PASTE", l, flush=True)
    print("SIDECAR", S["sidecar"], flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
