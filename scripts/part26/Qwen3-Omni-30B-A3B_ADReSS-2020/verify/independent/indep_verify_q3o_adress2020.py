"""Independent verifier, PART26 cell Qwen3-Omni-30B-A3B x ADReSS-2020.
Written from scratch; does not import or call p26_gap.py / p26_verify.py / sklearn.metrics.
AUC = Mann-Whitney via average ranks (numpy only), cross-checked by brute pair counting."""
import csv, json, hashlib, sys, numpy as np

CELL = "scores/part26/Qwen3-Omni-30B-A3B_ADReSS-2020"
PERCLIP = CELL + "/per_clip/p26_q3o_adress2020_perclip.csv"
DRAWS = CELL + "/draws/p26_q3o_adress2020_draws.npz"
PODOOF = CELL + "/pull/p26-q3o-adress2020/out/p26_q3o_adress2020_enc_nested5_oof.csv"
PODJSON = CELL + "/pull/p26-q3o-adress2020/out/p26_q3o_adress2020_enc_nested5.json"
PODFOLDDIR = CELL + "/pull/p26-q3o-adress2020/folds"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_adress2020_zeroshot_scores.csv"
FOLDDIR = "<local data dir>/release/folds"
STATES = "<local data dir>/release/overnight2/q3o/q3o_adress2020_states.npz"
CLAIM = dict(probe=0.8445, per_repeat=[0.8337, 0.8555, 0.8458, 0.8401, 0.8475], answer=0.7097,
             gap=0.1348, lo=0.0417, hi=0.2335)

def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()

def rows(p): return list(csv.DictReader(open(p, newline="")))

def auc_rank(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0: return np.nan
    o = np.argsort(s, kind="mergesort"); ss = s[o]; r = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

def auc_pairs(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    p = s[y == 1][:, None]; n = s[y == 0][None, :]
    return float(((p > n) + 0.5 * (p == n)).mean())

out = {}; ok = {}
# ---- per-clip file
pc = rows(PERCLIP)
clips = [r["clip"] for r in pc]; spk = np.array([r["speaker"] for r in pc]); y = np.array([int(r["label"]) for r in pc])
P = np.array([[float(r[f"p_probe_seed{s}"]) for s in range(5)] for r in pc])
F = np.array([[int(r[f"fold_seed{s}"]) for s in range(5)] for r in pc])
zcol = np.array([float(r["p_yes_zeroshot"]) for r in pc])
out["n"] = len(pc); out["n_unique_clips"] = len(set(clips)); out["n_spk"] = len(set(spk)); out["n_pos"] = int(y.sum())
ok["perclip_156_unique"] = len(pc) == 156 and len(set(clips)) == 156
ok["no_nan_scores"] = bool(np.isfinite(P).all() and np.isfinite(zcol).all())

# ---- release fold files: join by clip, speaker must match, fold must match
sidecar = json.load(open(CELL + "/per_clip/p26_q3o_adress2020_perclip.sidecar.json"))
fold_ok = True; spk_ok = True; sha_ok = True; podfold_sha_ok = True
for s in range(5):
    fn = f"adress2020_groupkfold5_pod_seed{s}_UNVERIFIED.csv"; fp = f"{FOLDDIR}/{fn}"
    fr = {r["clip_id"]: r for r in rows(fp)}
    if set(fr) != set(clips) or len(fr) != 156: fold_ok = False
    for i, c in enumerate(clips):
        if fr[c]["speaker_id"] != spk[i]: spk_ok = False
        if int(fr[c]["fold"]) != F[i, s]: fold_ok = False
    if sha(fp) != sidecar["probe"]["fold_file_sha256"][f"seed{s}"]: sha_ok = False
    if sha(f"{PODFOLDDIR}/{fn}") != sha(fp): podfold_sha_ok = False
ok["perclip_folds_equal_release_fold_files"] = fold_ok
ok["fold_file_speakers_equal_perclip"] = spk_ok
ok["release_fold_sha_equal_sidecar"] = sha_ok
ok["pod_pulled_fold_files_equal_release_bytes"] = podfold_sha_ok

# ---- rebuild folds from scratch (renumber, then greedy fill with stable-reversed order)
def rebuild(spk, seed, k=5):
    u = np.unique(spk); perm = np.random.default_rng(seed).permutation(u)
    newid = {sp: i for i, sp in enumerate(perm)}; g = np.array([newid[sp] for sp in spk])
    ug, inv = np.unique(g, return_inverse=True); cnt = np.bincount(inv)
    order = np.argsort(cnt, kind="stable")[::-1]; fg = np.zeros(len(ug), int); load = np.zeros(k)
    for gi in order:
        f = int(np.argmin(load)); fg[gi] = f; load[f] += cnt[gi]
    return fg[inv]
reb = np.stack([rebuild(spk, s) for s in range(5)], 1)
ok["folds_rebuilt_from_scratch_equal"] = bool(np.array_equal(reb, F))
out["fold_sizes"] = {f"seed{s}": np.bincount(F[:, s]).tolist() for s in range(5)}

# ---- pod OOF file equals per-clip probe columns and folds
po = rows(PODOOF)
ok["pod_oof_rows_same_order"] = [r["clip"] for r in po] == clips and [r["speaker"] for r in po] == list(spk) \
    and [int(r["label"]) for r in po] == list(y)
ok["pod_oof_scores_equal_perclip"] = all(float(po[i][f"p_seed{s}"]) == P[i, s] for i in range(156) for s in range(5))
ok["pod_oof_folds_equal_perclip"] = all(int(po[i][f"fold_seed{s}"]) == F[i, s] for i in range(156) for s in range(5))
pj = json.load(open(PODJSON))
ok["pod_json_fold_files_are_release_seed_files"] = all(
    pj["tags"][f"seed{s}"]["fold_file"] == f"adress2020_groupkfold5_pod_seed{s}_UNVERIFIED.csv"
    and pj["tags"][f"seed{s}"]["fold_file_sha256"] == sha(f"{FOLDDIR}/adress2020_groupkfold5_pod_seed{s}_UNVERIFIED.csv")
    for s in range(5))

# ---- states align
z = np.load(STATES, allow_pickle=True)
ok["states_names_spk_label_align"] = list(z["name"].astype(str)) == clips and list(z["spk"].astype(str)) == list(spk) \
    and np.array_equal(z["label"].astype(int), y)
out["states_sha256"] = sha(STATES); ok["states_sha_equal_sidecar"] = out["states_sha256"] == sidecar["states"]["sha256_mac"]

# ---- zero-shot file, joined by clip
zr = {r["clip"]: r for r in rows(ZS)}
ok["zs_clips_equal"] = set(zr) == set(clips) and len(zr) == 156
ok["zs_speaker_label_align"] = all(zr[c]["speaker"] == spk[i] and int(zr[c]["label"]) == y[i] for i, c in enumerate(clips))
zs = np.array([float(zr[c]["p_yes"]) for c in clips])
ok["zs_column_equal_file"] = bool(np.array_equal(zs, zcol))
out["zs_sha256"] = sha(ZS); ok["zs_sha_equal_sidecar"] = out["zs_sha256"] == sidecar["answer"]["sha256"]

# ---- point estimates
per = [auc_rank(y, P[:, s]) for s in range(5)]; per_pairs = [auc_pairs(y, P[:, s]) for s in range(5)]
ans = auc_rank(y, zs); ans_pairs = auc_pairs(y, zs)
mean5 = float(np.mean(per)); gap = mean5 - ans
out.update(per_repeat=per, per_repeat_4dp=[round(a, 4) for a in per], mean5=mean5, answer=ans, gap=gap,
           rank_vs_pairs_maxdiff=float(max(abs(a - b) for a, b in zip(per + [ans], per_pairs + [ans_pairs]))))
ok["per_repeat_4dp_match"] = [round(a, 4) for a in per] == CLAIM["per_repeat"]
ok["probe_4dp_match"] = round(mean5, 4) == CLAIM["probe"]
ok["answer_4dp_match"] = round(ans, 4) == CLAIM["answer"]
ok["gap_4dp_match"] = round(gap, 4) == CLAIM["gap"]
ok["rank_auc_equals_pair_auc"] = out["rank_vs_pairs_maxdiff"] < 1e-12

# ---- speaker bootstrap, fresh default_rng(0), 2000 draws
U = np.unique(spk); rows_of = [np.where(spk == u)[0] for u in U]
rng = np.random.default_rng(0); d = []; usable = 0; idx_all = []
for b in range(2000):
    si = rng.choice(len(U), size=len(U), replace=True); idx_all.append(si)
    ii = np.concatenate([rows_of[k] for k in si]); yy = y[ii]
    if yy.min() == yy.max(): d.append(np.nan); continue
    usable += 1
    d.append(np.mean([auc_rank(yy, P[ii, s]) for s in range(5)]) - auc_rank(yy, zs[ii]))
d = np.array(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
out.update(boot_usable=usable, lo=float(lo), hi=float(hi), p_le0=float(np.mean(d[np.isfinite(d)] <= 0)))
ok["boot_2000_usable"] = usable == 2000
ok["lo_4dp_match"] = round(lo, 4) == CLAIM["lo"]; ok["hi_4dp_match"] = round(hi, 4) == CLAIM["hi"]
ok["interval_above_zero"] = lo > 0
# alternative draw generator (rng.integers) for information
rng2 = np.random.default_rng(0); same_gen = all(np.array_equal(rng2.integers(0, len(U), len(U)), s) for s in idx_all)
out["choice_equals_integers_stream"] = bool(same_gen)

# ---- compare with saved draws npz
dz = np.load(DRAWS, allow_pickle=True)
ok["npz_speakers_equal_unique"] = list(dz["speakers"].astype(str)) == list(U)
ok["npz_speaker_idx_equal"] = bool(np.array_equal(dz["speaker_idx"].astype(int), np.stack(idx_all)))
out["npz_diff_maxabs"] = float(np.nanmax(np.abs(dz["diff"] - d)))
ok["npz_draws_equal_1e-10"] = out["npz_diff_maxabs"] < 1e-10
ok["npz_ci_equal_1e-10"] = bool(np.allclose(dz["ci_diff"], [lo, hi], atol=1e-10, rtol=0))

out["checks"] = ok; out["all_pass"] = all(ok.values())
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
for k, v in ok.items(): print(("PASS " if v else "FAIL ") + k)
print(json.dumps({k: out[k] for k in ("per_repeat_4dp", "mean5", "answer", "gap", "lo", "hi", "boot_usable", "p_le0",
                                      "npz_diff_maxabs", "rank_vs_pairs_maxdiff", "choice_equals_integers_stream", "fold_sizes")}, default=float))
print("ALL_PASS", out["all_pass"])
