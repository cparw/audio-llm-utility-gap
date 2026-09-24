#!/usr/local/bin/python3
# Independent verifier, PART 26, Kimi-Audio x ADReSSo encoder nested probe.
# Written from scratch: own rank AUC (Mann-Whitney with midranks), own bootstrap loop.
import csv, hashlib, json, sys
import numpy as np

D = "scores/part26/Kimi-Audio_ADReSSo"
PERCLIP = D + "/per_clip/Kimi-Audio_ADReSSo_perclip.csv"
PODOOF = D + "/pull/p26-kimi-adresso-cpu2/out/p26_kimi_adresso_enc_nested5_oof.csv"
DRAWS = D + "/draws/Kimi-Audio_ADReSSo_draws.npz"
ZS = "<local data dir>/release/overnight2/kimi_new/kimi_adresso_zeroshot_scores.csv"
FOLDDIR = "<local data dir>/release/folds/"
FOLDS = {f"seed{s}": FOLDDIR + f"adresso_groupkfold5_pod_seed{s}_UNVERIFIED.csv" for s in range(5)}
FOLDS["single"] = FOLDDIR + "adresso_groupkfold5_pod.csv"

CLAIM = dict(probe=0.8682, answer=0.7531, gap=0.1151, lo=0.0481, hi=0.1831,
             per_repeat=[0.8490, 0.8806, 0.8926, 0.8775, 0.8415], single=0.8945)

out = {"checks": {}}
ok_all = True
def chk(name, cond, detail=None):
    global ok_all
    out["checks"][name] = {"pass": bool(cond), "detail": detail}
    ok_all = ok_all and bool(cond)
    print(("PASS " if cond else "FAIL ") + name, "" if detail is None else detail)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def rows(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))

def auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

pc = rows(PERCLIP)
zs = rows(ZS)
pod = rows(PODOOF)
n = len(pc)
out["n"] = n
out["sha256"] = {"perclip": sha(PERCLIP), "zeroshot": sha(ZS), "pod_oof": sha(PODOOF), "draws": sha(DRAWS),
                 **{"fold_" + k: sha(v) for k, v in FOLDS.items()}}

# clips, speakers, labels line up with the zero-shot file
chk("n_237", n == 237 and len(zs) == 237 and len(pod) == 237, [n, len(zs), len(pod)])
chk("clip_order_equal_zeroshot", [r["clip"] for r in pc] == [r["clip"] for r in zs])
chk("speaker_equal_zeroshot", [r["speaker"] for r in pc] == [r["speaker"] for r in zs])
chk("label_equal_zeroshot", [r["label"] for r in pc] == [r["label"] for r in zs])
chk("clips_unique", len(set(r["clip"] for r in pc)) == n)
nspk = len(set(r["speaker"] for r in pc)); npos = sum(int(r["label"]) for r in pc)
out["n_speakers"] = nspk; out["n_pos"] = npos
# each speaker has one label
spk_lab = {}
cons = True
for r in pc:
    if spk_lab.setdefault(r["speaker"], r["label"]) != r["label"]:
        cons = False
chk("speaker_label_consistent", cons, {"n_speakers": nspk, "n_pos": npos})
chk("pyes_copy_equals_zeroshot", all(float(a["p_yes_zeroshot"]) == float(b["p_yes"]) for a, b in zip(pc, zs)))

# per-clip csv equals pod OOF csv
podmap = {r["clip"]: r for r in pod}
eq = True; maxd = 0.0
for r in pc:
    q = podmap[r["clip"]]
    if q["speaker"] != r["speaker"] or q["label"] != r["label"]:
        eq = False
    for s in range(5):
        d = abs(float(r[f"p_probe_seed{s}"]) - float(q[f"p_seed{s}"])); maxd = max(maxd, d)
        if r[f"fold_seed{s}"] != q[f"fold_seed{s}"]:
            eq = False
    maxd = max(maxd, abs(float(r["p_probe_single"]) - float(q["p_single"])))
    if r["fold_single"] != q["fold_single"]:
        eq = False
chk("perclip_equals_pod_oof", eq and maxd == 0.0, {"max_abs_diff": maxd})

# folds match the saved fold files clip by clip, and folds are speaker grouped
for k, p in FOLDS.items():
    fr = rows(p)
    fmap = {r["clip_id"]: r for r in fr}
    col = "fold_single" if k == "single" else f"fold_{k}"
    same_set = set(fmap) == set(r["clip"] for r in pc)
    same = same_set and all(fmap[r["clip"]]["fold"] == r[col] and fmap[r["clip"]]["speaker_id"] == r["speaker"] for r in pc)
    spk_fold = {}
    grouped = True
    for r in fr:
        if spk_fold.setdefault(r["speaker_id"], r["fold"]) != r["fold"]:
            grouped = False
    nf = len(set(r["fold"] for r in fr))
    chk(f"folds_match_{k}", same and grouped and nf == 5 and len(fr) == n,
        {"file": p, "n_rows": len(fr), "n_folds": nf, "grouped": grouped})

y = np.array([int(r["label"]) for r in pc])
P = np.array([[float(r[f"p_probe_seed{s}"]) for s in range(5)] for r in pc])
psingle = np.array([float(r["p_probe_single"]) for r in pc])
pyes = np.array([float(r["p_yes"]) for r in zs])
chk("all_scores_finite", np.isfinite(P).all() and np.isfinite(pyes).all() and np.isfinite(psingle).all())

per = [auc(y, P[:, s]) for s in range(5)]
mean5 = float(np.mean(per))
ans = auc(y, pyes)
single = auc(y, psingle)
gap = mean5 - ans
out["recomputed"] = {"per_repeat": per, "mean5": mean5, "answer": ans, "gap": gap, "single": single}
try:
    from sklearn.metrics import roc_auc_score
    sk = [roc_auc_score(y, P[:, s]) for s in range(5)] + [roc_auc_score(y, pyes)]
    out["sklearn_crosscheck_maxdiff"] = float(np.max(np.abs(np.array(sk) - np.array(per + [ans]))))
except Exception as e:
    out["sklearn_crosscheck_maxdiff"] = str(e)

r4 = lambda x: round(x + 0.0, 4)
chk("per_repeat_4dp", [r4(a) for a in per] == CLAIM["per_repeat"], [r4(a) for a in per])
chk("mean5_4dp", r4(mean5) == CLAIM["probe"], r4(mean5))
chk("answer_4dp", r4(ans) == CLAIM["answer"], r4(ans))
chk("gap_4dp", r4(gap) == CLAIM["gap"], r4(gap))
chk("single_4dp", r4(single) == CLAIM["single"], r4(single))

# speaker bootstrap, fresh default_rng(0), unique speakers in np.unique order
spk = np.array([r["speaker"] for r in pc])
uspk = np.unique(spk)
rows_of = {s: np.where(spk == s)[0] for s in uspk}
rng = np.random.default_rng(0)
B = 2000
diffs = np.full(B, np.nan); m5s = np.full(B, np.nan); zss = np.full(B, np.nan)
idx_all = np.zeros((B, len(uspk)), dtype=np.int64)
for b in range(B):
    idx = rng.choice(len(uspk), size=len(uspk), replace=True)
    idx_all[b] = idx
    ii = np.concatenate([rows_of[uspk[j]] for j in idx])
    yy = y[ii]
    if yy.min() == yy.max():
        continue
    m = np.mean([auc(yy, P[ii, s]) for s in range(5)])
    a = auc(yy, pyes[ii])
    m5s[b] = m; zss[b] = a; diffs[b] = m - a
usable = int(np.isfinite(diffs).sum())
lo, hi = np.percentile(diffs[np.isfinite(diffs)], [2.5, 97.5])
out["bootstrap"] = {"usable": usable, "lo": float(lo), "hi": float(hi),
                    "p_le0": float(np.mean(diffs[np.isfinite(diffs)] <= 0))}
chk("usable_2000", usable == 2000, usable)
chk("lo_4dp", r4(lo) == CLAIM["lo"], r4(lo))
chk("hi_4dp", r4(hi) == CLAIM["hi"], r4(hi))

# compare with the saved draws file
z = np.load(DRAWS, allow_pickle=False)
dd = float(np.nanmax(np.abs(z["diff"] - diffs)))
out["saved_draws_compare"] = {
    "diff_maxabs": dd,
    "mean5_maxabs": float(np.nanmax(np.abs(z["mean5"] - m5s))),
    "zeroshot_maxabs": float(np.nanmax(np.abs(z["zeroshot"] - zss))),
    "speaker_idx_equal": bool(np.array_equal(z["speaker_idx"].astype(np.int64), idx_all)),
    "speakers_equal": bool(np.array_equal(z["speakers"], uspk)),
    "point_diff": float(z["point_diff"]), "point_mean5": float(z["point_mean5"]), "point_zeroshot": float(z["point_zeroshot"]),
}
chk("saved_draws_match", dd < 1e-9 and out["saved_draws_compare"]["speaker_idx_equal"] and out["saved_draws_compare"]["speakers_equal"],
    out["saved_draws_compare"])
chk("saved_points_match", abs(float(z["point_diff"]) - gap) < 1e-12 and abs(float(z["point_mean5"]) - mean5) < 1e-12
    and abs(float(z["point_zeroshot"]) - ans) < 1e-12)

out["all_pass"] = ok_all
print(json.dumps({k: v for k, v in out.items() if k != "checks"}, indent=1, default=float))
print("ALL_PASS", ok_all)
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
