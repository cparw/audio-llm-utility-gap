"""PART20 POD3b analysis. AUC by rank formula (scipy rankdata, ties averaged); speaker bootstrap 2000 draws,
fresh numpy.random.default_rng(0) per cell, speakers (np.unique) resampled with replacement, percentile 2.5/97.5;
PAIRED = one speaker draw per replicate for both arms.
usage: p20_analyze.py MANIFEST SCORES(glob) OUTSTEM TAG MACDIR REF_JSON"""
import sys, os, json, glob, hashlib, datetime, numpy as np, pandas as pd
from scipy.stats import rankdata
MAN, SC, STEM, TAG, MACDIR = sys.argv[1:6]
EXTRA_SRC = sys.argv[6:]
def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1*(n1+1)/2) / (n1*n0))
def boot(df, fn, B=2000):
    spk = df.speaker_id.values; u = np.unique(spk)
    by = {k: np.where(spk == k)[0] for k in u}
    rng = np.random.default_rng(0); out = []
    for _ in range(B):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([by[k] for k in pick])
        v = fn(df.iloc[ii])
        if np.all(np.isfinite(v)): out.append(v)
    out = np.asarray(out)
    return np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0), len(out)
def sha1mb(p): return hashlib.sha256(open(p, "rb").read(1 << 20)).hexdigest()
m = pd.read_csv(MAN)
files = sorted(glob.glob(SC))
s = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("id")
m = m.merge(s[["id","p_yes","answer_mass","wav_sha256_1mb","n_samples_scored","prompt"]], left_on="seg_uid", right_on="id", how="left").drop(columns="id")
assert m.p_yes.notna().all(), f"missing scores {m.p_yes.isna().sum()}"
m["clip_path_pod"] = [f"/workspace/p20/clips/seg_{int(u[1:]):06d}.wav" for u in m.seg_uid]
C = m[m.set == "conflict"]; A = m[m.set == "agreement"]; AD = A.drop_duplicates("seg_uid")
res = {}
def add(key, what, val, lo, hi, df, kind):
    res[key] = dict(what=what, value=val, lo=float(lo), hi=float(hi), n=int(len(df)), n_distinct_segments=int(df.seg_uid.nunique()),
                    n_speakers=int(df.speaker_id.nunique()), kind=kind)
for key, what, df in (("conflict", "conflict arm AUC", C), ("agreement_all", "agreement arm AUC, all rows", A),
                      ("agreement_distinct", "agreement arm AUC, distinct segments", AD)):
    lo, hi, nu = boot(df, lambda d: np.array([auc(d.label, d.p_yes)]))
    add(key, what, auc(df.label, df.p_yes), lo[0], hi[0], df, "auc"); res[key]["usable_draws"] = nu
    res[key]["n_pos"] = int(df.label.sum())
for key, what, AA in (("diff_conflict_minus_agreement_all", "paired conflict minus agreement (all rows)", A),
                      ("diff_conflict_minus_agreement_distinct", "paired conflict minus agreement (distinct agreement segments)", AD)):
    both = pd.concat([C, AA])
    def fn(d):
        c = d[d.set == "conflict"]; a = d[d.set == "agreement"]
        return np.array([auc(c.label, c.p_yes) - auc(a.label, a.p_yes)])
    lo, hi, nu = boot(both, fn)
    add(key, what, auc(C.label, C.p_yes) - auc(AA.label, AA.p_yes), lo[0], hi[0], both, "diff"); res[key]["usable_draws"] = nu
out = f"{STEM}.csv"
cols = ["seg_uid","set","pair_id","speaker_id","label","sev","start","end","span","speech_s","nw","db_p_pos","db_p_neg",
        "clip_path_pod","n_samples_scored","wav_sha256_1mb","p_yes","answer_mass","prompt"]
cols = [c for c in cols if c in m.columns]
m[cols].to_csv(out, index=False)
macfile = f"{MACDIR}/{os.path.basename(out)}"
lines, rows = [], []
for k, r in res.items():
    if r["kind"] == "auc":
        side = "above 0.5" if r["value"] > 0.5 else "below 0.5"
        excl = "interval excludes 0.5" if (r["lo"] > 0.5 or r["hi"] < 0.5) else "interval includes 0.5"
        two = f"{r['value']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"; vs = side
        lines.append(f"{TAG}_{k}: AUC {r['value']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] n={r['n']} distinct_segments={r['n_distinct_segments']} n_speakers={r['n_speakers']} file={macfile} | {two} {side} ({excl})")
    else:
        vs = "interval excludes zero" if (r["lo"] > 0 or r["hi"] < 0) else "interval includes zero"
        two = f"{r['value']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"
        lines.append(f"{TAG}_{k}: diff {r['value']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] n={r['n']} n_speakers={r['n_speakers']} file={macfile} | {two} {vs}")
    rows.append([f"{TAG}_{k}", r["what"], f"{r['value']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}", str(r["n"]), str(r["n_speakers"]), macfile, two, vs])
srcs = [os.path.abspath(MAN)] + [os.path.abspath(f) for f in files] + [os.path.abspath(x) for x in EXTRA_SRC]
side = {"result_file": out, "result_file_mac": macfile, "results": res, "n_rows": int(len(m)), "n_speakers": int(m.speaker_id.nunique()),
  "n_distinct_segments": int(m.seg_uid.nunique()), "median_answer_mass": float(m.answer_mass.median()), "min_answer_mass": float(m.answer_mass.min()),
  "model_id": "Qwen/Qwen2.5-Omni-7B", "prompt_verbatim": m.prompt.iloc[0], "prompt_unique": int(m.prompt.nunique()),
  "window_seconds": 30, "dtype": "bfloat16", "seed": 0, "bootstrap": "2000 draws, fresh numpy.random.default_rng(0) per cell, speakers resampled with replacement, percentile 2.5/97.5; differences paired (one speaker draw per replicate)",
  "auc": "rank formula, scipy.stats.rankdata ties averaged",
  "scoring_note": "each distinct segment scored once (deterministic forward pass, rerun max diff 0.0 on this pod) and its p_yes joined to every manifest row that uses it",
  "sources": {p: {"sha256_first_1MB": sha1mb(p)} for p in srcs},
  "command": " ".join(["python3"] + sys.argv), "date": datetime.datetime.utcnow().isoformat() + "Z"}
json.dump(side, open(f"{STEM}.json", "w"), indent=1, default=float)
open(f"{STEM}.rows.tsv", "w").write("\n".join("\t".join(r) for r in rows) + "\n")
print("\n".join(lines))
