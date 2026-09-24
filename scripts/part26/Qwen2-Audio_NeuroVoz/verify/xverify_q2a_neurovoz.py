"""PART26 independent verifier, Qwen2-Audio x NeuroVoz.
Written from scratch; does not import or copy the producer's gap or verify scripts.
Recomputes from the per-clip OOF csv and the zero-shot per-clip csv:
  per-repeat AUCs (own Mann-Whitney rank AUC), their mean, answer AUC, point gap,
  2000-draw speaker bootstrap with a fresh numpy default_rng(0), 2.5/97.5 percentiles.
Checks fold columns against the saved release fold files, clip/speaker/label alignment
across every file, speaker grouping of folds, and compares to the claimed numbers and the npz draws.
Read only on every input. Writes only xverify_q2a_neurovoz.json next to this script."""
import csv, json, hashlib, os, sys
import numpy as np
from scipy.stats import rankdata

CELL = "scores/part26/Qwen2-Audio_NeuroVoz"
PERCLIP = f"{CELL}/per_clip/p26_q2a_neurovoz_perclip.csv"
DRAWS = f"{CELL}/draws/p26_q2a_neurovoz_draws.npz"
POD_OOF = f"{CELL}/pull/p26-q2a-neurovoz/out/p26_q2a_neurovoz_enc_nested5_oof.csv"
ZS = "<local data dir>/paper1_local_runs/probe2/neurovoz_zeroshot_scores.csv"
FOLDDIR = "<local data dir>/release/folds"
STAGED = f"{CELL}/stage/q2a_neurovoz_encstates.npz"
CLAIM = {"probe": "0.9324", "answer": "0.5519", "gap": "+0.3805", "lo": "+0.3278", "hi": "+0.4325",
         "per_repeat": "0.9310 0.9296 0.9283 0.9386 0.9346"}
OUT = f"{CELL}/verify/xverify_q2a_neurovoz.json"

res = {"checks": {}, "values": {}}
def chk(name, ok, detail=None):
    res["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  {detail}"), flush=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def read_csv(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))

def my_auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s, method="average")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

# ---------- load
pc = read_csv(PERCLIP); zs = read_csv(ZS); pod = read_csv(POD_OOF)
clips = [r["clip"] for r in pc]
n = len(clips)
chk("perclip_1270_rows_unique_clips", n == 1270 and len(set(clips)) == n, f"n={n} unique={len(set(clips))}")
tags = [f"seed{s}" for s in range(5)]
P = np.array([[float(r[f"p_probe_{t}"]) for t in tags] for r in pc])   # n x 5
F = np.array([[int(r[f"fold_{t}"]) for t in tags] for r in pc])
y = np.array([int(r["label"]) for r in pc]); spk = np.array([r["speaker"] for r in pc])
chk("no_nan_probe_scores", np.isfinite(P).all() and ((P >= 0) & (P <= 1)).all())

# zero-shot alignment by clip
zmap = {r["clip"]: r for r in zs}
chk("zeroshot_same_clip_set", set(zmap) == set(clips) and len(zmap) == len(zs) == n, f"zs rows={len(zs)}")
zy = np.array([int(zmap[c]["label"]) for c in clips]); zspk = np.array([zmap[c]["speaker"] for c in clips])
zp = np.array([float(zmap[c]["p_yes"]) for c in clips])
chk("zeroshot_labels_equal", np.array_equal(zy, y))
chk("zeroshot_speakers_equal", np.array_equal(zspk, spk))
pz = np.array([float(r["p_yes_zeroshot"]) for r in pc])
chk("perclip_zeroshot_column_equals_zs_file", np.array_equal(pz, zp), f"max|d|={np.abs(pz-zp).max():.3g}")
chk("zeroshot_finite", np.isfinite(zp).all())

# pod OOF equals per-clip file
pm = {r["clip"]: r for r in pod}
chk("pod_oof_same_clips", set(pm) == set(clips) and len(pod) == n)
dmax = max(abs(float(pm[c][f"p_{t}"]) - P[i, j]) for i, c in enumerate(clips) for j, t in enumerate(tags))
fsame = all(int(pm[c][f"fold_{t}"]) == F[i, j] for i, c in enumerate(clips) for j, t in enumerate(tags))
chk("perclip_probe_equals_pod_oof", dmax == 0.0 and fsame, f"max|d|={dmax}")
chk("pod_oof_speaker_label_equal", all(pm[c]["speaker"] == spk[i] and int(pm[c]["label"]) == y[i] for i, c in enumerate(clips)))

# staged states (what the pod fit on): names, speakers, labels line up
z = np.load(STAGED, allow_pickle=True)
sn = z["name"].astype(str); ss = z["spk"].astype(str); sl = z["label"].astype(int)
chk("states_order_equals_perclip", np.array_equal(sn, np.array(clips)) and np.array_equal(ss, spk) and np.array_equal(sl, y))

# ---------- folds vs saved release fold files
fold_info = {}
for j, t in enumerate(tags):
    fp = os.path.join(FOLDDIR, f"neurovoz_groupkfold5_pod_{t}_UNVERIFIED.csv")
    fr = {r["clip_id"]: r for r in read_csv(fp)}
    same_set = set(fr) == set(clips) and len(fr) == n
    eq_fold = same_set and all(int(fr[c]["fold"]) == F[i, j] for i, c in enumerate(clips))
    eq_spk = same_set and all(fr[c]["speaker_id"] == spk[i] for i, c in enumerate(clips))
    # speaker grouping: each speaker in one fold only
    grouped = all(len(set(F[spk == s, j])) == 1 for s in np.unique(spk))
    sizes = np.bincount(F[:, j], minlength=5).tolist()
    classes_ok = all(len(set(y[F[:, j] == k])) == 2 for k in range(5))
    fold_info[t] = {"file": fp, "sha256": sha(fp), "sizes": sizes}
    chk(f"fold_{t}_equals_release_file", eq_fold and eq_spk, f"{os.path.basename(fp)}")
    chk(f"fold_{t}_speaker_grouped_5folds_both_classes", grouped and sorted(set(F[:, j])) == [0, 1, 2, 3, 4] and classes_ok, f"sizes={sizes}")

# own rebuild of the pod tie rule (renumber speakers with default_rng(seed).permutation, greedy fill by descending size,
# equal sizes ordered by descending renumbered id, first least-loaded fold)
def rebuild(seed):
    u = np.unique(spk); perm = np.random.default_rng(seed).permutation(u)
    newid = {s: i for i, s in enumerate(perm)}
    g = np.array([newid[s] for s in spk])
    ids = sorted(set(g.tolist())); cnt = {i: int((g == i).sum()) for i in ids}
    order = sorted(ids, key=lambda i: (-cnt[i], -i))
    load = [0] * 5; fg = {}
    for i in order:
        f = min(range(5), key=lambda k: (load[k], k)); fg[i] = f; load[f] += cnt[i]
    return np.array([fg[i] for i in g])
for j, t in enumerate(tags):
    chk(f"fold_{t}_equals_own_rebuild_of_tie_rule", np.array_equal(rebuild(j), F[:, j]))
res["fold_files"] = fold_info

# ---------- point estimates
per = [my_auc(y, P[:, j]) for j in range(5)]
mean5 = float(np.mean(per)); ans = my_auc(y, zp); gap = mean5 - ans
res["values"].update({"per_repeat": per, "mean5": mean5, "answer": ans, "gap": gap,
                      "n": n, "n_pos": int(y.sum()), "n_speakers": int(len(np.unique(spk)))})
f4 = lambda v: f"{v:.4f}"; s4 = lambda v: f"{v:+.4f}"
chk("per_repeat_4dp", " ".join(f4(v) for v in per) == CLAIM["per_repeat"], " ".join(f4(v) for v in per))
chk("probe_mean5_4dp", f4(mean5) == CLAIM["probe"], f"{mean5:.6f}")
chk("answer_4dp", f4(ans) == CLAIM["answer"], f"{ans:.6f}")
chk("gap_4dp", s4(gap) == CLAIM["gap"], f"{gap:+.6f}")

# ---------- speaker bootstrap, 2000 draws, fresh default_rng(0)
u = np.unique(spk); rows = {s: np.where(spk == s)[0] for s in u}
rng = np.random.default_rng(0)
B = 2000; dg = np.full(B, np.nan); dp = np.full((B, 5), np.nan); dz = np.full(B, np.nan)
for b in range(B):
    pick = rng.choice(len(u), size=len(u), replace=True)
    idx = np.concatenate([rows[u[k]] for k in pick])
    yy = y[idx]
    if yy.min() == yy.max(): continue
    dp[b] = [my_auc(yy, P[idx, j]) for j in range(5)]
    dz[b] = my_auc(yy, zp[idx]); dg[b] = dp[b].mean() - dz[b]
ok = np.isfinite(dg); lo, hi = np.percentile(dg[ok], [2.5, 97.5])
res["values"].update({"usable_draws": int(ok.sum()), "lo": float(lo), "hi": float(hi),
                      "probe_ci": np.percentile(dp[ok].mean(1), [2.5, 97.5]).tolist(),
                      "answer_ci": np.percentile(dz[ok], [2.5, 97.5]).tolist(),
                      "speaker_order": "np.unique of speaker strings", "n_draws": B})
chk("usable_draws_2000", int(ok.sum()) == 2000, int(ok.sum()))
chk("lo_4dp", s4(lo) == CLAIM["lo"], f"{lo:+.6f}")
chk("hi_4dp", s4(hi) == CLAIM["hi"], f"{hi:+.6f}")
chk("interval_above_zero", lo > 0)

# compare with saved draws npz
d = np.load(DRAWS, allow_pickle=True)
res["npz_keys"] = {k: [list(d[k].shape), str(d[k].dtype)] for k in d.files}
print("npz keys", res["npz_keys"])
sg = d["diff"].astype(float); both = np.isfinite(sg) & ok
md = float(np.abs(sg[both] - dg[both]).max()); nan_eq = bool(np.array_equal(np.isfinite(sg), ok))
chk("draws_npz_diff_equals_own", md < 1e-9 and nan_eq, f"max|d|={md:.3g}")
chk("draws_npz_per_repeat_equals_own", float(np.nanmax(np.abs(d["per_repeat"] - dp))) < 1e-9,
    f"max|d|={float(np.nanmax(np.abs(d['per_repeat'] - dp))):.3g}")
chk("draws_npz_zeroshot_equals_own", float(np.nanmax(np.abs(d["zeroshot"] - dz))) < 1e-9)
chk("draws_npz_speaker_list_equals_np_unique", np.array_equal(d["speakers"].astype(str), u))
rng3 = np.random.default_rng(0); mypicks = np.array([rng3.choice(len(u), size=len(u), replace=True) for _ in range(B)])
chk("draws_npz_speaker_idx_equals_own", np.array_equal(d["speaker_idx"].astype(int), mypicks))
chk("npz_point_diff_equals_own", abs(float(d["point_diff"]) - gap) < 1e-12, f"{float(d['point_diff']):.10f}")
res["values"]["npz_point_diff_master"] = float(d["point_diff_master"])

# sensitivity only (not a pass/fail): numeric speaker order
un = np.array(sorted(u, key=int)); rng2 = np.random.default_rng(0); g2 = []
for b in range(B):
    idx = np.concatenate([rows[un[k]] for k in rng2.choice(len(un), size=len(un), replace=True)])
    yy = y[idx]
    if yy.min() == yy.max(): continue
    g2.append(np.mean([my_auc(yy, P[idx, j]) for j in range(5)]) - my_auc(yy, zp[idx]))
res["sensitivity_numeric_speaker_order"] = np.percentile(g2, [2.5, 97.5]).tolist()
print("sensitivity numeric order CI", res["sensitivity_numeric_speaker_order"])

res["inputs_sha256"] = {"perclip": sha(PERCLIP), "zeroshot": sha(ZS), "draws": sha(DRAWS), "pod_oof": sha(POD_OOF)}
res["all_pass"] = all(v["pass"] for v in res["checks"].values())
res["claim"] = CLAIM
json.dump(res, open(OUT, "w"), indent=1, default=float)
print("ALL_PASS", res["all_pass"])
