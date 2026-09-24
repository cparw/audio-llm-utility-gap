#!/usr/local/bin/python3
"""Independent verifier, PART 26, Qwen2-Audio E-DAIC.
Own code: tie-averaged rank AUC (numpy only), own speaker bootstrap, own fold and clip checks.
Reads only. Prints a JSON report."""
import csv, json, hashlib, sys
import numpy as np

CELL = "scores/part26/Qwen2-Audio_E-DAIC"
PERCLIP = f"{CELL}/Qwen2-Audio_E-DAIC_perclip.csv"
DRAWS = f"{CELL}/Qwen2-Audio_E-DAIC_draws.npz"
ZS = "<local data dir>/paper1_local_runs/probe2/edaic_zeroshot_scores.csv"
FOLDDIR = "<local data dir>/release/folds"
FOLDS = {f"seed{k}": f"{FOLDDIR}/edaic_groupkfold5_pod_seed{k}_UNVERIFIED.csv" for k in range(5)}
FOLDS["single"] = f"{FOLDDIR}/edaic_groupkfold5_pod.csv"
MASTER = "<local data dir>/paper1_local_runs/omni_final/q2a_edaic_nested_repeats.json"
P25C_OOF = "<local data dir>/release_from_mac/scores/part25/C/pull/p25C-4/out/q2a_edaic_enc_rep5_oof.csv"
POD_OOF = f"{CELL}/pull/p26-q2a-edaic/out/p26_q2a_edaic_enc_nested5_oof.csv"
CLAIM = dict(probe=0.5673, answer=0.6464, gap=-0.0791, lo=-0.1664, hi=0.0120,
             per_repeat=[0.5130, 0.6021, 0.5555, 0.5558, 0.6098])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def ranks_avg(x):
    # average ranks for ties, 1-based
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    xs = x[o]
    r = np.empty(len(x), float)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def auc(y, s):
    y = np.asarray(y).astype(int)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = ranks_avg(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def auc_pairs(y, s):
    # brute force O(n1*n0) check
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    p = s[y == 1][:, None]; q = s[y == 0][None, :]
    return float(((p > q) + 0.5 * (p == q)).mean())


rep = {"checks": {}, "fail": []}
C = rep["checks"]

# ---------- load per-clip ----------
P = read(PERCLIP)
clips = [r["clip"] for r in P]
spk = np.array([r["speaker"] for r in P])
y = np.array([int(r["label"]) for r in P])
probe = np.array([[float(r[f"p_probe_seed{k}"]) for k in range(5)] for r in P])
foldcols = {f"seed{k}": np.array([int(r[f"fold_seed{k}"]) for r in P]) for k in range(5)}
foldcols["single"] = np.array([int(r["fold_single"]) for r in P])
pzs_col = np.array([float(r["p_yes_zeroshot"]) for r in P])
C["perclip"] = dict(n=len(P), unique_clips=len(set(clips)) == len(clips), n_spk=int(len(np.unique(spk))),
                    n_pos=int(y.sum()), any_nan_probe=bool(np.isnan(probe).any()), sha256=sha(PERCLIP))
if len(set(clips)) != len(clips):
    rep["fail"].append("duplicate clips in perclip")
# clip name vs speaker consistency (E-DAIC clip is <pid>.wav)
C["perclip"]["clip_is_speaker_wav"] = all(c == f"{s}.wav" for c, s in zip(clips, spk))

# ---------- zero-shot file ----------
Z = read(ZS)
zmap = {r["clip"]: r for r in Z}
C["zeroshot"] = dict(n=len(Z), sha256=sha(ZS), unique=len(zmap) == len(Z),
                     same_clip_set=set(zmap) == set(clips))
zy = np.array([int(zmap[c]["label"]) for c in clips])
zs = np.array([str(zmap[c]["speaker"]) for c in clips])
zp = np.array([float(zmap[c]["p_yes"]) for c in clips])
C["zeroshot"]["labels_equal"] = bool((zy == y).all())
C["zeroshot"]["speakers_equal"] = bool((zs == spk).all())
C["zeroshot"]["p_yes_col_max_abs_diff"] = float(np.abs(zp - pzs_col).max())
for k in ["same_clip_set", "labels_equal", "speakers_equal", "unique"]:
    if not C["zeroshot"][k]:
        rep["fail"].append(f"zeroshot {k}")
if C["zeroshot"]["p_yes_col_max_abs_diff"] != 0.0:
    rep["fail"].append("p_yes column differs from zero-shot file")

# ---------- fold files ----------
C["folds"] = {}
for key, fp in FOLDS.items():
    F = read(fp)
    fm = {r["clip_id"]: r for r in F}
    ff = np.array([int(fm[c]["fold"]) for c in clips])
    fs = np.array([str(fm[c]["speaker_id"]) for c in clips])
    sizes = np.bincount(ff, minlength=5).tolist()
    # speaker never in two folds (per fold file)
    spk_fold = {}
    leak = 0
    for s_, f_ in zip(fs, ff):
        if spk_fold.setdefault(s_, f_) != f_:
            leak += 1
    d = dict(file=fp, sha256=sha(fp), n=len(F), same_clip_set=set(fm) == set(clips),
             folds_equal_perclip=bool((ff == foldcols[key]).all()),
             speakers_equal_perclip=bool((fs == spk).all()), fold_sizes=sizes, speaker_leak=leak)
    C["folds"][key] = d
    for k in ["same_clip_set", "folds_equal_perclip", "speakers_equal_perclip"]:
        if not d[k]:
            rep["fail"].append(f"fold {key} {k}")
    if leak:
        rep["fail"].append(f"fold {key} speaker leak")
# repeat splits must differ from each other (five distinct splits)
distinct = len({tuple(foldcols[f"seed{k}"]) for k in range(5)})
C["folds"]["n_distinct_repeat_splits"] = distinct
if distinct != 5:
    rep["fail"].append("repeat splits not distinct")
# sidecar sha claims
side = json.load(open(f"{CELL}/Qwen2-Audio_E-DAIC_perclip.sidecar.json"))
sha_ok = {k: side["probe"]["fold_files"][k]["sha256"] == C["folds"][k]["sha256"] for k in FOLDS}
C["folds"]["sidecar_sha_equal_canonical"] = sha_ok
if not all(sha_ok.values()):
    rep["fail"].append("sidecar fold sha differs from canonical fold file")

# ---------- AUCs ----------
per = [auc(y, probe[:, k]) for k in range(5)]
per_bf = [auc_pairs(y, probe[:, k]) for k in range(5)]
mean5 = float(np.mean(per))
ans = auc(y, zp)
ans_bf = auc_pairs(y, zp)
gap = mean5 - ans
try:
    from sklearn.metrics import roc_auc_score
    sk = [float(roc_auc_score(y, probe[:, k])) for k in range(5)] + [float(roc_auc_score(y, zp))]
except Exception as e:
    sk = str(e)
C["auc"] = dict(per_repeat=per, per_repeat_bruteforce=per_bf, mean5=mean5, answer=ans, answer_bruteforce=ans_bf,
                gap=gap, sklearn_cross=sk, per_repeat_4dp=[round(v, 4) for v in per], mean5_4dp=round(mean5, 4),
                answer_4dp=round(ans, 4), gap_4dp=round(gap, 4))
# master json (read only)
M = json.load(open(MASTER))
C["auc"]["master_enc_per_repeat"] = M["enc"]["per_repeat"]
C["auc"]["master_enc_mean"] = M["enc"]["mean"]
C["auc"]["per_repeat_eq_master_4dp"] = [round(v, 4) for v in per] == [round(v, 4) for v in M["enc"]["per_repeat"]]
C["auc"]["mean_eq_master_4dp"] = round(mean5, 4) == round(M["enc"]["mean"], 4)

# ---------- cross-check OOF against pod OOF and PART 25 C refit ----------
for tag, fp, pc, fc in [("pod_oof", POD_OOF, "p_seed{k}", "fold_seed{k}"), ("p25C_oof", P25C_OOF, "p_seed{k}", "fold_seed{k}")]:
    R = read(fp); rm = {r["clip"]: r for r in R}
    pp = np.array([[float(rm[c][pc.format(k=k)]) for k in range(5)] for c in clips])
    ffold = np.array([[int(rm[c][fc.format(k=k)]) for k in range(5)] for c in clips])
    C[tag] = dict(file=fp, sha256=sha(fp), n=len(R), same_clips=set(rm) == set(clips),
                  max_abs_p_diff=float(np.abs(pp - probe).max()),
                  folds_equal=bool((ffold == np.stack([foldcols[f"seed{k}"] for k in range(5)], 1)).all()),
                  labels_equal=bool((np.array([int(rm[c]["label"]) for c in clips]) == y).all()))

# ---------- speaker bootstrap, own implementation ----------
u = np.unique(spk)
rows_of = {s: np.flatnonzero(spk == s) for s in u}
rng = np.random.default_rng(0)
B = 2000
dd = np.empty(B); mm = np.empty(B); qq = np.empty(B); ok = np.ones(B, bool)
idx_all = np.empty((B, len(u)), np.int64)
for b in range(B):
    pick = rng.choice(len(u), size=len(u), replace=True)
    idx_all[b] = pick
    ii = np.concatenate([rows_of[u[j]] for j in pick])
    yb = y[ii]
    if yb.min() == yb.max():
        ok[b] = False; continue
    m = np.mean([auc(yb, probe[ii, k]) for k in range(5)])
    q = auc(yb, zp[ii])
    mm[b] = m; qq[b] = q; dd[b] = m - q
lo, hi = np.percentile(dd[ok], [2.5, 97.5])
C["bootstrap"] = dict(n_draws=B, usable=int(ok.sum()), lo=float(lo), hi=float(hi), lo_4dp=round(float(lo), 4),
                      hi_4dp=round(float(hi), 4), probe_ci=np.percentile(mm[ok], [2.5, 97.5]).tolist(),
                      answer_ci=np.percentile(qq[ok], [2.5, 97.5]).tolist(),
                      frac_draws_gap_le0=float((dd[ok] <= 0).mean()))
# compare with saved draws file
D = np.load(DRAWS, allow_pickle=False)
C["bootstrap"]["saved_keys"] = list(D.files)
C["bootstrap"]["saved_speakers_eq_sorted_unique"] = bool((D["speakers"] == u).all())
C["bootstrap"]["saved_idx_eq_own"] = bool((D["speaker_idx"].astype(np.int64) == idx_all).all())
C["bootstrap"]["saved_diff_max_abs"] = float(np.abs(D["diff"] - dd).max())
C["bootstrap"]["saved_mean5_max_abs"] = float(np.abs(D["mean5"] - mm).max())
C["bootstrap"]["saved_zs_max_abs"] = float(np.abs(D["zeroshot"] - qq).max())
C["bootstrap"]["saved_ci"] = D["ci_diff"].tolist()
C["bootstrap"]["saved_point"] = float(D["point_diff"])

# ---------- compare with claim at 4 dp ----------
cmp = dict(
    per_repeat=[round(v, 4) for v in per] == [round(v, 4) for v in CLAIM["per_repeat"]],
    probe=round(mean5, 4) == CLAIM["probe"],
    answer=round(ans, 4) == CLAIM["answer"],
    gap=round(gap, 4) == CLAIM["gap"],
    lo=round(float(lo), 4) == CLAIM["lo"],
    hi=round(float(hi), 4) == CLAIM["hi"],
)
rep["claim_match_4dp"] = cmp
for k, v in cmp.items():
    if not v:
        rep["fail"].append(f"claim {k} mismatch")
# internal agreements
if max(abs(a - b) for a, b in zip(per, per_bf)) > 1e-12 or abs(ans - ans_bf) > 1e-12:
    rep["fail"].append("rank vs bruteforce AUC disagree")
if not C["auc"]["per_repeat_eq_master_4dp"]:
    rep["fail"].append("per repeat != master json 4dp")
rep["PASS"] = len(rep["fail"]) == 0
import platform
rep["versions"] = dict(python=platform.python_version(), numpy=np.__version__)
print(json.dumps(rep, indent=1, default=float))
