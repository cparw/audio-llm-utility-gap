# Independent verifier, PART 26, Qwen2-Audio on MDVR-KCL.
# Written from scratch. Reads only; writes one json next to itself.
import csv, json, hashlib, sys, os
import numpy as np
from sklearn.metrics import roc_auc_score

CELL = "scores/part26/Qwen2-Audio_MDVR-KCL"
PERCLIP = f"{CELL}/Q2A_kcl_perclip.csv"
DRAWS = f"{CELL}/Q2A_kcl_draws.npz"
PODOOF = f"{CELL}/pull/p26-q2a-kcl/out/p26_q2a_kcl_enc_nested5_oof.csv"
ZS = "<local data dir>/paper1_local_runs/probe2/kcl_zeroshot_scores.csv"
FOLDDIR = "<local data dir>/release/folds"
T1 = "<local data dir>/release/omni_final/q2a_kcl_nested_repeats.json"
CLAIM = {"probe": 0.7714, "answer": 0.5923, "gap": 0.1792, "lo": -0.0503, "hi": 0.4305,
         "per_repeat": [0.8065, 0.8095, 0.6786, 0.7679, 0.7946]}

def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()

def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))

def auc_mw(y, s):
    # Mann-Whitney AUC with average ranks for ties, no sklearn
    y = np.asarray(y); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

out = {"checks": {}, "values": {}}
C = out["checks"]; V = out["values"]

# 1. per-clip file
pc = rows(PERCLIP)
clips = [r["clip"] for r in pc]; spk = np.array([r["speaker"] for r in pc]); y = np.array([int(r["label"]) for r in pc])
C["n_rows_37"] = len(pc) == 37
C["clips_unique"] = len(set(clips)) == len(clips)
C["speakers_unique_37"] = len(set(spk)) == 37
C["speaker_prefix_of_clip"] = all(c.split("_")[0] == s for c, s in zip(clips, spk))
C["label_matches_clip_tag"] = all((("_pd_" in c) == (l == 1)) and (("_hc_" in c) == (l == 0)) for c, l in zip(clips, y))
V["n"] = len(pc); V["n_pos"] = int(y.sum()); V["n_spk"] = int(len(set(spk)))
P = np.array([[float(r[f"p_probe_seed{s}"]) for s in range(5)] for r in pc])
FO = np.array([[int(r[f"fold_seed{s}"]) for s in range(5)] for r in pc])
C["no_nan_probe"] = bool(np.isfinite(P).all())

# 2. folds vs saved release fold files
fold_ok = {}; fold_sha = {}
for s in range(5):
    fp = f"{FOLDDIR}/kcl_groupkfold5_pod_seed{s}_UNVERIFIED.csv"
    fr = {r["clip_id"]: (r["speaker_id"], int(r["fold"])) for r in rows(fp)}
    fold_sha[f"seed{s}"] = sha(fp)
    same_set = set(fr) == set(clips) and len(fr) == 37
    same_spk = all(fr[c][0] == sp for c, sp in zip(clips, spk))
    same_fold = all(fr[c][1] == FO[i, s] for i, c in enumerate(clips))
    sizes = sorted(np.bincount(FO[:, s], minlength=5).tolist(), reverse=True)
    # each speaker in exactly one outer fold
    grouped = all(len(set(FO[spk == u, s])) == 1 for u in set(spk))
    fold_ok[f"seed{s}"] = {"clip_set": same_set, "speakers": same_spk, "fold_ids": same_fold,
                           "sizes": sizes, "speaker_grouped": grouped}
fp = f"{FOLDDIR}/kcl_groupkfold5_pod.csv"
fr = {r["clip_id"]: (r["speaker_id"], int(r["fold"])) for r in rows(fp)}
fold_sha["single"] = sha(fp)
fold_ok["single"] = {"fold_ids": all(fr[c][1] == int(r["fold_single"]) for c, r in zip(clips, pc))}
V["fold_checks"] = fold_ok; V["fold_sha256"] = fold_sha
C["folds_match_saved_files"] = all(all(v for k, v in d.items() if k != "sizes") for d in fold_ok.values())
# the five repeats must be different splits (not the same file five times)
C["five_repeat_splits_distinct"] = len({tuple(FO[:, s]) for s in range(5)}) == 5

# 3. zero-shot file
zr = {r["clip"]: r for r in rows(ZS)}
V["zs_sha256"] = sha(ZS)
C["zs_clip_set_equal"] = set(zr) == set(clips) and len(zr) == 37
C["zs_speaker_equal"] = all(zr[c]["speaker"] == s for c, s in zip(clips, spk))
C["zs_label_equal"] = all(int(zr[c]["label"]) == l for c, l in zip(clips, y))
zs = np.array([float(zr[c]["p_yes"]) for c in clips])
C["zs_column_in_perclip_equal_file"] = bool(np.array_equal(zs, np.array([float(r["p_yes_zeroshot"]) for r in pc])))

# 4. pod OOF file equals per-clip file
po = {r["clip"]: r for r in rows(PODOOF)}
C["pod_oof_equals_perclip"] = (set(po) == set(clips) and all(
    po[c]["speaker"] == r["speaker"] and int(po[c]["label"]) == int(r["label"]) and
    all(float(po[c][f"p_seed{s}"]) == float(r[f"p_probe_seed{s}"]) and int(po[c][f"fold_seed{s}"]) == int(r[f"fold_seed{s}"])
        for s in range(5)) for c, r in zip(clips, pc)))

# 5. point estimates
per_mw = [auc_mw(y, P[:, s]) for s in range(5)]
per_sk = [float(roc_auc_score(y, P[:, s])) for s in range(5)]
mean5 = float(np.mean(per_mw)); ans = auc_mw(y, zs); ans_sk = float(roc_auc_score(y, zs))
gap = mean5 - ans
V.update(per_repeat=per_mw, per_repeat_sklearn=per_sk, mean5=mean5, answer=ans, answer_sklearn=ans_sk, gap=gap)
C["mw_equals_sklearn"] = max(abs(a - b) for a, b in zip(per_mw, per_sk)) < 1e-12 and abs(ans - ans_sk) < 1e-12
t1 = json.load(open(T1))["enc"]
C["per_repeat_equals_table1_json_4dp"] = [round(a, 4) for a in per_mw] == t1["per_repeat"]
C["mean_equals_table1_json_4dp"] = round(mean5, 4) == t1["mean"]

# 6. bootstrap, 2000 draws, fresh default_rng(0), unique speakers with replacement, all their clips
uspk = np.unique(spk); nspk = len(uspk)
clips_of = [np.where(spk == u)[0] for u in uspk]
rng = np.random.default_rng(0)
B = 2000; diff = np.full(B, np.nan); m5 = np.full(B, np.nan); za = np.full(B, np.nan); pr = np.full((B, 5), np.nan)
IDX = np.zeros((B, nspk), int); dropped = 0
for b in range(B):
    idx = rng.choice(nspk, size=nspk, replace=True); IDX[b] = idx
    rr = np.concatenate([clips_of[i] for i in idx]); yb = y[rr]
    if yb.min() == yb.max(): dropped += 1; continue
    pr[b] = [auc_mw(yb, P[rr, s]) for s in range(5)]
    m5[b] = pr[b].mean(); za[b] = auc_mw(yb, zs[rr]); diff[b] = m5[b] - za[b]
ok = np.isfinite(diff)
lo, hi = np.percentile(diff[ok], [2.5, 97.5])
V.update(boot_usable=int(ok.sum()), boot_dropped=dropped, lo=float(lo), hi=float(hi),
         probe_ci=[float(x) for x in np.percentile(m5[ok], [2.5, 97.5])],
         answer_ci=[float(x) for x in np.percentile(za[ok], [2.5, 97.5])],
         frac_draws_le0=float((diff[ok] <= 0).mean()))

# 7. saved draws
d = np.load(DRAWS, allow_pickle=False)
C["saved_speakers_equal_unique_order"] = list(d["speakers"]) == list(uspk)
C["saved_speaker_idx_equal_mine"] = bool(np.array_equal(d["speaker_idx"], IDX))
C["saved_diff_draws_match_mine_1e-12"] = bool(np.nanmax(np.abs(d["diff"] - diff)) < 1e-12)
C["saved_per_repeat_draws_match_mine_1e-12"] = bool(np.nanmax(np.abs(d["per_repeat"] - pr)) < 1e-12)
C["saved_zeroshot_draws_match_mine_1e-12"] = bool(np.nanmax(np.abs(d["zeroshot"] - za)) < 1e-12)
C["saved_point_diff_matches"] = abs(float(d["point_diff"]) - gap) < 1e-12
V["saved_draws_max_abs_diff"] = float(np.nanmax(np.abs(d["diff"] - diff)))

# 8. compare to the claim at 4 dp
r4 = lambda v: round(float(v), 4)
cmp = {"probe": (r4(mean5), CLAIM["probe"]), "answer": (r4(ans), CLAIM["answer"]), "gap": (r4(gap), CLAIM["gap"]),
       "lo": (r4(lo), CLAIM["lo"]), "hi": (r4(hi), CLAIM["hi"]),
       "per_repeat": ([r4(a) for a in per_mw], CLAIM["per_repeat"])}
V["claim_compare_4dp"] = cmp
C["claim_matches_4dp"] = all(a == b for a, b in cmp.values())
out["all_pass"] = all(C.values())
out["files"] = {"perclip": PERCLIP, "perclip_sha256": sha(PERCLIP), "draws": DRAWS, "draws_sha256": sha(DRAWS),
                "zeroshot": ZS, "pod_oof": PODOOF, "table1_json": T1}
out["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_q2a_kcl_mine.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
