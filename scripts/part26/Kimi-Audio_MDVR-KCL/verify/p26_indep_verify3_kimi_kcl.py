"""Independent verifier for PART26 Kimi-Audio MDVR-KCL cell.
Written from scratch; uses only the per-clip OOF file, the zero-shot file,
the pod OOF file, the release fold files and the draws npz."""
import csv, json, hashlib, sys
from fractions import Fraction
import numpy as np

ROOT = "scores/part26/Kimi-Audio_MDVR-KCL"
PERCLIP = ROOT + "/per_clip/p26_kimi_kcl_perclip.csv"
DRAWS = ROOT + "/draws/p26_kimi_kcl_draws.npz"
PODOOF = ROOT + "/pull/p26-kimi-kcl-cpu/out/p26_kimi_kcl_enc_nested5_oof.csv"
ZS = "<local data dir>/release/overnight2/kimi_new/kimi_kcl_zeroshot_scores.csv"
FOLDDIR = "<local data dir>/release/folds"
FOLDS = {s: f"{FOLDDIR}/kcl_groupkfold5_pod_seed{s}_UNVERIFIED.csv" for s in range(5)}

CLAIM = dict(probe=0.7631, answer=0.7113, gap=0.0518, lo=-0.1637, hi=0.2901,
             per_repeat=[0.6935, 0.7946, 0.8512, 0.7173, 0.7589],
             probe_frac=Fraction(641, 840), answer_frac=Fraction(239, 336), gap_frac=Fraction(29, 560))

out = {}
checks = {}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def rows(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))

# my own exact AUC: count pairs (pos > neg) + 0.5 ties over n_pos*n_neg
def auc_exact(y, s):
    y = list(y); s = list(s)
    pos = [s[i] for i in range(len(y)) if y[i] == 1]
    neg = [s[i] for i in range(len(y)) if y[i] == 0]
    num = Fraction(0)
    for a in pos:
        for b in neg:
            if a > b:
                num += 1
            elif a == b:
                num += Fraction(1, 2)
    return num / (len(pos) * len(neg))

# fast rank based AUC for bootstrap (average ranks for ties); counts duplicates as separate clips
def auc_rank(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r1 = ranks[y == 1].sum()
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)

# ---------- load
pc = rows(PERCLIP)
zs = rows(ZS)
pod = rows(PODOOF)
folds = {s: rows(FOLDS[s]) for s in range(5)}
out["sha256"] = {"perclip": sha(PERCLIP), "draws": sha(DRAWS), "pod_oof": sha(PODOOF), "zeroshot": sha(ZS),
                 **{f"fold_seed{s}": sha(FOLDS[s]) for s in range(5)}}

# ---------- G1: clips, speakers, labels line up
n = len(pc)
clips = [r["clip"] for r in pc]
spk = [r["speaker"] for r in pc]
lab = [int(r["label"]) for r in pc]
g1 = {}
g1["n_clips"] = n
g1["unique_clips"] = len(set(clips)) == n
g1["n_speakers"] = len(set(spk))
g1["n_pos"] = sum(lab)
zs_map = {r["clip"]: r for r in zs}
g1["zs_same_clip_set"] = set(zs_map) == set(clips) and len(zs) == n
g1["zs_same_order"] = [r["clip"] for r in zs] == clips
g1["zs_speaker_match"] = all(zs_map[c]["speaker"] == s for c, s in zip(clips, spk))
g1["zs_label_match"] = all(int(zs_map[c]["label"]) == l for c, l in zip(clips, lab))
# label vs clip name (hc -> 0, pd -> 1)
g1["label_matches_clip_name"] = all((("_pd_" in c) == (l == 1)) and (("_hc_" in c) == (l == 0)) for c, l in zip(clips, lab))
g1["speaker_is_clip_prefix"] = all(c.split("_")[0] == s for c, s in zip(clips, spk))
pod_map = {r["clip"]: r for r in pod}
g1["pod_same_clip_set"] = set(pod_map) == set(clips) and len(pod) == n
g1["pod_speaker_label_match"] = all(pod_map[c]["speaker"] == s and int(pod_map[c]["label"]) == l for c, s, l in zip(clips, spk, lab))
checks["G1_clips_speakers_labels"] = all(v for k, v in g1.items() if isinstance(v, bool)) and n == 37 and g1["n_speakers"] == 37 and g1["n_pos"] == 16
out["G1"] = g1

# ---------- G2: folds match saved fold files
g2 = {}
ok2 = True
for s in range(5):
    fm = {r["clip_id"]: r for r in folds[s]}
    same_set = set(fm) == set(clips) and len(folds[s]) == n
    spk_ok = all(fm[c]["speaker_id"] == sp for c, sp in zip(clips, spk))
    fold_pc = all(int(fm[c]["fold"]) == int(r[f"fold_seed{s}"]) for c, r in zip(clips, pc))
    fold_pod = all(int(fm[c]["fold"]) == int(pod_map[c][f"fold_seed{s}"]) for c in clips)
    sizes = sorted(np.bincount([int(fm[c]["fold"]) for c in clips]).tolist(), reverse=True)
    g2[f"seed{s}"] = dict(same_clip_set=same_set, speaker_match=spk_ok, perclip_fold_match=fold_pc,
                         pod_oof_fold_match=fold_pod, fold_sizes=sizes)
    ok2 &= same_set and spk_ok and fold_pc and fold_pod and sizes == [8, 8, 7, 7, 7]
# rebuild each seed file from its stated definition: renumber speakers with default_rng(seed).permutation(unique ids),
# then GroupKFold(5) with stable argsort of group sizes reversed, lightest fold first
def rebuild(seed):
    uniq = np.unique(np.array(spk))
    perm = np.random.default_rng(seed).permutation(uniq)
    newnum = {sp: i for i, sp in enumerate(perm)}
    g = np.array([newnum[x] for x in spk])
    ug, inv = np.unique(g, return_inverse=True)
    cnt = np.bincount(inv)
    idx = np.argsort(cnt, kind="stable")[::-1]
    load = np.zeros(5)
    g2f = np.zeros(len(ug), dtype=int)
    for gi in idx:
        f = int(np.argmin(load)); load[f] += cnt[gi]; g2f[gi] = f
    return g2f[inv]
reb = {}
for s in range(5):
    fm = {r["clip_id"]: int(r["fold"]) for r in folds[s]}
    rb = rebuild(s)
    reb[f"seed{s}"] = bool(all(int(rb[i]) == fm[c] for i, c in enumerate(clips)))
g2["rebuild_from_definition_matches"] = reb
# folds are speaker disjoint trivially (1 clip per speaker) but check anyway
g2["speaker_disjoint"] = all(len({int(r[f"fold_seed{s}"]) for r in pc if r["speaker"] == sp}) == 1 for s in range(5) for sp in set(spk))
checks["G2_folds_match_saved_files"] = bool(ok2 and g2["speaker_disjoint"])
out["G2"] = g2

# ---------- G3: per-clip scores equal pod OOF and zero-shot file exactly
g3 = {}
g3["probe_scores_equal_pod_oof"] = all(float(r[f"p_probe_seed{s}"]) == float(pod_map[r["clip"]][f"p_seed{s}"]) for r in pc for s in range(5))
g3["single_equal_pod_oof"] = all(float(r["p_probe_single"]) == float(pod_map[r["clip"]]["p_single"]) for r in pc)
g3["zeroshot_equal_file"] = all(float(r["p_yes_zeroshot"]) == float(zs_map[r["clip"]]["p_yes"]) for r in pc)
g3["no_nan"] = all(np.isfinite(float(r[k])) for r in pc for k in [f"p_probe_seed{s}" for s in range(5)] + ["p_yes_zeroshot"])
checks["G3_scores_trace_to_sources"] = all(g3.values())
out["G3"] = g3

# ---------- G4: per-repeat AUCs and mean, exact
P = np.array([[float(r[f"p_probe_seed{s}"]) for s in range(5)] for r in pc])
Z = np.array([float(r["p_yes_zeroshot"]) for r in pc])
Y = np.array(lab)
per_exact = [auc_exact(Y, P[:, s]) for s in range(5)]
mean_exact = sum(per_exact) / 5
ans_exact = auc_exact(Y, Z)
gap_exact = mean_exact - ans_exact
from sklearn.metrics import roc_auc_score
per_sk = [roc_auc_score(Y, P[:, s]) for s in range(5)]
ans_sk = roc_auc_score(Y, Z)
g4 = dict(per_repeat_exact=[str(f) for f in per_exact], per_repeat=[float(f) for f in per_exact],
          per_repeat_4dp=[round(float(f), 4) for f in per_exact], per_repeat_sklearn=per_sk,
          mean5_exact=str(mean_exact), mean5=float(mean_exact), mean5_4dp=round(float(mean_exact), 4))
g4["match_claim_per_repeat_4dp"] = g4["per_repeat_4dp"] == CLAIM["per_repeat"]
g4["match_claim_mean_4dp"] = round(float(mean_exact), 4) == CLAIM["probe"]
g4["match_claim_fraction"] = mean_exact == CLAIM["probe_frac"]
g4["sklearn_agrees"] = max(abs(a - float(b)) for a, b in zip(per_sk, per_exact)) < 1e-12
checks["G4_per_repeat_and_mean"] = g4["match_claim_per_repeat_4dp"] and g4["match_claim_mean_4dp"] and g4["match_claim_fraction"] and g4["sklearn_agrees"]
out["G4"] = g4

# ---------- G5: answer AUC
g5 = dict(answer_exact=str(ans_exact), answer=float(ans_exact), answer_4dp=round(float(ans_exact), 4), sklearn=ans_sk)
g5["match_claim_4dp"] = round(float(ans_exact), 4) == CLAIM["answer"]
g5["match_claim_fraction"] = ans_exact == CLAIM["answer_frac"]
g5["sklearn_agrees"] = abs(ans_sk - float(ans_exact)) < 1e-12
# also compute from the zero-shot file directly, in its own order
g5["answer_from_raw_zs_file"] = float(auc_exact([int(r["label"]) for r in zs], [float(r["p_yes"]) for r in zs]))
g5["raw_file_agrees"] = g5["answer_from_raw_zs_file"] == float(ans_exact)
checks["G5_answer_auc"] = g5["match_claim_4dp"] and g5["match_claim_fraction"] and g5["sklearn_agrees"] and g5["raw_file_agrees"]
out["G5"] = g5

# ---------- G6: gap point
g6 = dict(gap_exact=str(gap_exact), gap=float(gap_exact), gap_4dp=round(float(gap_exact), 4))
g6["match_claim_4dp"] = round(float(gap_exact), 4) == CLAIM["gap"]
g6["match_claim_fraction"] = gap_exact == CLAIM["gap_frac"]
checks["G6_gap_point"] = g6["match_claim_4dp"] and g6["match_claim_fraction"]
out["G6"] = g6

# ---------- G7: speaker bootstrap, 2000 draws, fresh default_rng(0)
uspk = np.unique(np.array(spk))
spk_arr = np.array(spk)
clips_of = [np.where(spk_arr == u)[0] for u in uspk]
rng = np.random.default_rng(0)
B = 2000
mydiff = np.full(B, np.nan); mymean = np.full(B, np.nan); myzs = np.full(B, np.nan); myper = np.full((B, 5), np.nan)
myidx = np.zeros((B, len(uspk)), dtype=np.int64)
for b in range(B):
    pick = rng.choice(len(uspk), size=len(uspk), replace=True)
    myidx[b] = pick
    ii = np.concatenate([clips_of[k] for k in pick])
    yb = Y[ii]
    if yb.min() == yb.max():
        continue
    pr = [auc_rank(yb, P[ii, s]) for s in range(5)]
    myper[b] = pr
    mymean[b] = float(np.mean(pr))
    myzs[b] = auc_rank(yb, Z[ii])
    mydiff[b] = mymean[b] - myzs[b]
usable = int(np.isfinite(mydiff).sum())
lo, hi = np.percentile(mydiff[np.isfinite(mydiff)], [2.5, 97.5])
# cross check with rng.integers stream (informational only)
rng2 = np.random.default_rng(0)
same_as_integers = all(np.array_equal(rng2.integers(0, len(uspk), len(uspk)), myidx[b]) for b in range(5))
g7 = dict(usable=usable, lo=float(lo), hi=float(hi), lo_4dp=round(float(lo), 4), hi_4dp=round(float(hi), 4),
          mean5_ci=[float(x) for x in np.percentile(mymean[np.isfinite(mymean)], [2.5, 97.5])],
          answer_ci=[float(x) for x in np.percentile(myzs[np.isfinite(myzs)], [2.5, 97.5])],
          frac_draws_gap_le_0=float(np.mean(mydiff[np.isfinite(mydiff)] <= 0)),
          choice_stream_equals_integers_stream_first5=bool(same_as_integers))
g7["match_claim_lo_4dp"] = g7["lo_4dp"] == CLAIM["lo"]
g7["match_claim_hi_4dp"] = g7["hi_4dp"] == CLAIM["hi"]
g7["usable_2000"] = usable == 2000
g7["interval_includes_zero"] = bool(lo <= 0 <= hi)
# compare with saved draws
d = np.load(DRAWS, allow_pickle=False)
g7["saved_speakers_equal_np_unique"] = bool(np.array_equal(d["speakers"], uspk))
g7["saved_speaker_idx_equal"] = bool(np.array_equal(d["speaker_idx"].astype(np.int64), myidx))
g7["saved_diff_maxabs"] = float(np.nanmax(np.abs(d["diff"] - mydiff)))
g7["saved_mean5_maxabs"] = float(np.nanmax(np.abs(d["mean5"] - mymean)))
g7["saved_zeroshot_maxabs"] = float(np.nanmax(np.abs(d["zeroshot"] - myzs)))
g7["saved_per_repeat_maxabs"] = float(np.nanmax(np.abs(d["per_repeat"] - myper)))
g7["saved_point_diff"] = float(d["point_diff"])
g7["saved_point_diff_matches"] = abs(float(d["point_diff"]) - float(gap_exact)) < 1e-12
slo, shi = np.percentile(d["diff"], [2.5, 97.5])
g7["saved_draws_lo_hi"] = [float(slo), float(shi)]
g7["saved_draws_ok"] = (g7["saved_speaker_idx_equal"] and g7["saved_diff_maxabs"] < 1e-9 and g7["saved_mean5_maxabs"] < 1e-9
                        and g7["saved_zeroshot_maxabs"] < 1e-9 and g7["saved_per_repeat_maxabs"] < 1e-9 and g7["saved_point_diff_matches"])
checks["G7_bootstrap"] = g7["match_claim_lo_4dp"] and g7["match_claim_hi_4dp"] and g7["usable_2000"] and g7["interval_includes_zero"] and g7["saved_draws_ok"] and g7["saved_speakers_equal_np_unique"]
out["G7"] = g7

out["checks"] = checks
out["all_pass"] = all(checks.values())
out["versions"] = dict(python=sys.version.split()[0], numpy=np.__version__)
print(json.dumps(out, indent=1, default=str))
