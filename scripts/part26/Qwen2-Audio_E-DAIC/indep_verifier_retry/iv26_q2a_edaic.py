#!/usr/local/bin/python3
"""Independent verifier, PART 26 cell Qwen2-Audio / E-DAIC (first 30 s), encoder nested probe minus zero-shot answer.
Written from scratch. Reads only; writes its json next to itself (argv[1] = output json path).
AUC: own Mann-Whitney rank implementation (average ranks for ties), cross-checked with sklearn roc_auc_score.
"""
import csv, json, sys, hashlib, datetime
import numpy as np

CELL = "scores/part26/Qwen2-Audio_E-DAIC"
PERCLIP = f"{CELL}/Qwen2-Audio_E-DAIC_perclip.csv"
DRAWS = f"{CELL}/Qwen2-Audio_E-DAIC_draws.npz"
SIDECAR = f"{CELL}/Qwen2-Audio_E-DAIC_perclip.sidecar.json"
PODOOF = f"{CELL}/pull/p26-q2a-edaic/out/p26_q2a_edaic_enc_nested5_oof.csv"
PODJSON = f"{CELL}/pull/p26-q2a-edaic/out/p26_q2a_edaic_enc_nested5.json"
ZS = "<local data dir>/paper1_local_runs/probe2/edaic_zeroshot_scores.csv"
STATES = "<local data dir>/paper1_local_runs/probe2/edaic_states.npz"
MASTER = "<local data dir>/paper1_local_runs/omni_final/q2a_edaic_nested_repeats.json"
FOLDDIR = "<local data dir>/release/folds"
FOLDS = {f"seed{s}": f"{FOLDDIR}/edaic_groupkfold5_pod_seed{s}_UNVERIFIED.csv" for s in range(5)}
FOLDS["single"] = f"{FOLDDIR}/edaic_groupkfold5_pod.csv"
CLAIM = {"probe": 0.5673, "answer": 0.6464, "gap": -0.0791, "lo": -0.1664, "hi": 0.0120,
         "per_repeat": [0.5130, 0.6021, 0.5555, 0.5558, 0.6098]}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def rcsv(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort"); ss = s[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0: return np.nan
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

out = {"verifier": "iv26_q2a_edaic.py (written from scratch)", "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "numpy": np.__version__, "checks": {}}
C = out["checks"]
fail = []
def check(name, ok, detail=None):
    C[name] = {"ok": bool(ok), "detail": detail}
    if not ok: fail.append(name)

# ---- load
pc = rcsv(PERCLIP); zs = rcsv(ZS); po = rcsv(PODOOF)
clips = [r["clip"] for r in pc]
check("perclip_275_unique_clips", len(clips) == 275 and len(set(clips)) == 275, len(clips))
pcd = {r["clip"]: r for r in pc}; zsd = {r["clip"]: r for r in zs}; pod = {r["clip"]: r for r in po}
check("zs_clip_set_equal", set(zsd) == set(pcd) and len(zs) == 275, len(zs))
check("pod_oof_clip_set_equal", set(pod) == set(pcd) and len(po) == 275, len(po))

spk = np.array([pcd[c]["speaker"] for c in clips]); y = np.array([int(pcd[c]["label"]) for c in clips])
check("speaker_is_clip_stem", all(c == f"{s}.wav" for c, s in zip(clips, spk)))
check("one_clip_per_speaker", len(set(spk)) == 275, len(set(spk)))
check("zs_speaker_label_match", all(zsd[c]["speaker"] == pcd[c]["speaker"] and int(zsd[c]["label"]) == int(pcd[c]["label"]) for c in clips))
check("pod_speaker_label_match", all(pod[c]["speaker"] == pcd[c]["speaker"] and int(pod[c]["label"]) == int(pcd[c]["label"]) for c in clips))
out["n"] = 275; out["n_pos"] = int(y.sum()); out["n_speakers"] = int(len(set(spk)))

# states (read only): names, spk, labels, p_yes
st = np.load(STATES, allow_pickle=False)
nm = [str(x) for x in st["name"]]; sidx = {n: i for i, n in enumerate(nm)}
check("states_clip_set_equal", set(nm) == set(clips), len(nm))
check("states_label_spk_match", all(int(st["label"][sidx[c]]) == int(pcd[c]["label"]) and str(st["spk"][sidx[c]]) == pcd[c]["speaker"] for c in clips))
check("states_p_yes_equals_zs_file", max(abs(float(st["p_yes"][sidx[c]]) - float(zsd[c]["p_yes"])) for c in clips) == 0.0)

# per-clip scores vs pod OOF and zs file
P = np.array([[float(pcd[c][f"p_probe_seed{s}"]) for s in range(5)] for c in clips])
Ppod = np.array([[float(pod[c][f"p_seed{s}"]) for s in range(5)] for c in clips])
check("perclip_probe_equals_pod_oof", np.array_equal(P, Ppod), float(np.max(np.abs(P - Ppod))))
check("perclip_single_equals_pod_oof", all(float(pcd[c]["p_probe_single"]) == float(pod[c]["p_single"]) for c in clips))
z = np.array([float(zsd[c]["p_yes"]) for c in clips])
check("perclip_zs_equals_zs_file", all(float(pcd[c]["p_yes_zeroshot"]) == float(zsd[c]["p_yes"]) for c in clips))

# ---- folds vs saved files, group disjointness, and the documented tie rule
fold_info = {}
for tag, p in FOLDS.items():
    fr = rcsv(p); fd = {r["clip_id"]: r for r in fr}
    ok_set = set(fd) == set(clips) and len(fr) == 275
    ok_spk = all(fd[c]["speaker_id"] == pcd[c]["speaker"] for c in clips)
    f_saved = np.array([int(fd[c]["fold"]) for c in clips])
    f_pc = np.array([int(pcd[c][f"fold_{tag}"]) for c in clips])
    f_pod = np.array([int(pod[c][f"fold_{tag}"]) for c in clips])
    disjoint = all(len(set(f_saved[spk == s])) == 1 for s in set(spk))
    sizes = np.bincount(f_saved, minlength=5).tolist()
    # rebuild: renumber (seed tags) then GroupKFold greedy fill with stable argsort reversed
    u = np.unique(spk)
    if tag.startswith("seed"):
        perm = np.random.default_rng(int(tag[4:])).permutation(u); g = {s: i for i, s in enumerate(perm)}
    else:
        g = {s: i for i, s in enumerate(u)}
    gid = np.array([g[s] for s in spk]); ug, cnt = np.unique(gid, return_counts=True)
    order = np.argsort(cnt, kind="stable")[::-1]; load = np.zeros(5); gf = {}
    for gi in order:
        k = int(np.argmin(load)); gf[ug[gi]] = k; load[k] += cnt[gi]
    f_rule = np.array([gf[x] for x in gid])
    fold_info[tag] = {"file": p, "sha256": sha(p), "sizes": sizes, "perclip_equals_saved": bool(np.array_equal(f_pc, f_saved)),
                      "pod_oof_equals_saved": bool(np.array_equal(f_pod, f_saved)), "speaker_disjoint": disjoint,
                      "rebuilt_tie_rule_equals_saved": bool(np.array_equal(f_rule, f_saved)),
                      "rebuilt_mismatch": int((f_rule != f_saved).sum())}
    check(f"folds_{tag}", ok_set and ok_spk and np.array_equal(f_pc, f_saved) and np.array_equal(f_pod, f_saved) and disjoint, fold_info[tag])
out["folds"] = fold_info
# sidecar and pod json name these same fold files and hashes
sc = json.load(open(SIDECAR)); pj = json.load(open(PODJSON))
check("sidecar_fold_hashes_match_saved", all(sc["probe"]["fold_files"][t]["sha256"] == fold_info[t]["sha256"] for t in FOLDS))
check("pod_json_fold_names_match", all(pj["tags"][t]["fold_file"] == FOLDS[t].split("/")[-1] for t in FOLDS))

# ---- point estimates
from sklearn.metrics import roc_auc_score
per = [auc(y, P[:, s]) for s in range(5)]
per_sk = [roc_auc_score(y, P[:, s]) for s in range(5)]
check("own_auc_equals_sklearn", max(abs(a - b) for a, b in zip(per, per_sk)) < 1e-12)
mean5 = float(np.mean(per)); ans = auc(y, z); gap = mean5 - ans
single = auc(y, np.array([float(pcd[c]["p_probe_single"]) for c in clips]))
out["recomputed"] = {"per_repeat": per, "mean5": mean5, "answer": ans, "gap": gap, "single_split": single}
master = json.load(open(MASTER))["enc"]
check("per_repeat_match_master_4dp", [round(v, 4) for v in per] == master["per_repeat"], master["per_repeat"])
check("mean_match_master_4dp", round(mean5, 4) == master["mean"], master["mean"])
check("per_repeat_match_pod_json", np.allclose(per, pj["per_repeat_auc"], atol=1e-12, rtol=0))

# ---- speaker bootstrap
rng = np.random.default_rng(0)
u = np.unique(spk); rows = {s: np.where(spk == s)[0] for s in u}
B = 2000; D = np.empty(B); M = np.empty(B); Zb = np.empty(B); R = np.empty((B, 5)); IDX = np.empty((B, len(u)), int)
for b in range(B):
    idx = rng.choice(len(u), size=len(u), replace=True); IDX[b] = idx
    r = np.concatenate([rows[u[i]] for i in idx]); yb = y[r]
    R[b] = [auc(yb, P[r, s]) for s in range(5)]; M[b] = R[b].mean(); Zb[b] = auc(yb, z[r]); D[b] = M[b] - Zb[b]
usable = int(np.isfinite(D).sum())
lo, hi = np.percentile(D[np.isfinite(D)], [2.5, 97.5])
out["bootstrap"] = {"B": B, "usable": usable, "lo": float(lo), "hi": float(hi),
                    "probe_ci": np.percentile(M, [2.5, 97.5]).tolist(), "zs_ci": np.percentile(Zb, [2.5, 97.5]).tolist()}

# ---- compare with saved draws
d = np.load(DRAWS, allow_pickle=False)
cmp = {"speakers_equal": bool(np.array_equal(d["speakers"].astype(str), u)),
       "speaker_idx_equal": bool(np.array_equal(d["speaker_idx"], IDX)),
       "diff_maxabs": float(np.max(np.abs(d["diff"] - D))), "mean5_maxabs": float(np.max(np.abs(d["mean5"] - M))),
       "zeroshot_maxabs": float(np.max(np.abs(d["zeroshot"] - Zb))), "per_repeat_maxabs": float(np.max(np.abs(d["per_repeat"] - R))),
       "point_diff_saved": float(d["point_diff"]), "ci_saved": d["ci_diff"].tolist()}
out["draws_compare"] = cmp
check("draws_match", cmp["speakers_equal"] and cmp["speaker_idx_equal"] and max(cmp["diff_maxabs"], cmp["mean5_maxabs"], cmp["zeroshot_maxabs"], cmp["per_repeat_maxabs"]) < 1e-10, cmp)

# ---- claim vs recomputed, 4 dp
r4 = lambda v: round(float(v), 4)
cl = {"probe": (r4(mean5), CLAIM["probe"]), "answer": (r4(ans), CLAIM["answer"]), "gap": (r4(gap), CLAIM["gap"]),
      "lo": (r4(lo), CLAIM["lo"]), "hi": (r4(hi), CLAIM["hi"]), "per_repeat": ([r4(v) for v in per], CLAIM["per_repeat"])}
out["claim_vs_recomputed_4dp"] = {k: {"recomputed": a, "claimed": b, "equal": a == b} for k, (a, b) in cl.items()}
for k, (a, b) in cl.items(): check(f"claim_{k}_4dp", a == b, {"recomputed": a, "claimed": b})

out["failed"] = fail; out["verified"] = not fail
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print(json.dumps({"verified": out["verified"], "failed": fail, "recomputed": out["recomputed"], "boot": out["bootstrap"],
                  "draws": cmp, "folds": {t: {k: v for k, v in fi.items() if k not in ("file", "sha256")} for t, fi in fold_info.items()}}, indent=1, default=float))
