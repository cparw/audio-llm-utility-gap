"""Independent verifier, PART26 Qwen2-Audio MDVR-KCL. Written from scratch; does not import or copy the cell's scripts.
Reads: per-clip OOF csv, zero-shot per-clip csv, release fold files, saved draws npz, states npz (read only)."""
import csv, json, hashlib, sys
import numpy as np

CELL = "scores/part26/Qwen2-Audio_MDVR-KCL"
PERCLIP = f"{CELL}/Q2A_kcl_perclip.csv"
DRAWS = f"{CELL}/Q2A_kcl_draws.npz"
PODOOF = f"{CELL}/pull/p26-q2a-kcl/out/p26_q2a_kcl_enc_nested5_oof.csv"
ZS = "<local data dir>/paper1_local_runs/probe2/kcl_zeroshot_scores.csv"
FOLDDIR = "<local data dir>/release/folds"
STATES = "<local data dir>/paper1_local_runs/probe2/kcl_states.npz"
T1 = "<local data dir>/release/omni_final/q2a_kcl_nested_repeats.json"
CLAIM = dict(probe=0.7714, answer=0.5923, gap=0.1792, lo=-0.0503, hi=0.4305,
             per_repeat=[0.8065, 0.8095, 0.6786, 0.7679, 0.7946])

out = {"checks": {}, "values": {}}
def chk(name, ok, detail=None):
    out["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  {detail}"))

def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()

def auc(y, s):
    """Mann-Whitney AUC with average ranks for ties (own code)."""
    y = np.asarray(y); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort"); ss = s[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def rows(p): return list(csv.DictReader(open(p)))

# ---------- load
pc = rows(PERCLIP); zs = rows(ZS); pod = rows(PODOOF)
clips = [r["clip"] for r in pc]
spk = np.array([r["speaker"] for r in pc]); y = np.array([int(r["label"]) for r in pc])
P = np.array([[float(r[f"p_probe_seed{s}"]) for s in range(5)] for r in pc])  # (n,5)
F = np.array([[int(r[f"fold_seed{s}"]) for s in range(5)] for r in pc])
chk("perclip_sha256_matches_sidecar", sha(PERCLIP) == "415f4be0a06360880ef20fa74ea0a5ce8bf6a2d2c77df7f167d1b619e2625a5a")
chk("draws_sha256_matches_sidecar", sha(DRAWS) == "cf8be0386700a860dd9a6fde0a134e24de5b015f24f17e796a0acbb2decc1d83")
chk("zeroshot_sha256_matches_C_build_cells", sha(ZS) == "52fed914ffe7987c314d5a5f413aa3780938322be28c385cb226dd2402b104c0")
chk("n_clips_37_unique", len(clips) == 37 and len(set(clips)) == 37, f"n={len(clips)} npos={int(y.sum())} nspk={len(set(spk))}")

# ---------- folds vs release fold files
exp_sha = {0: "e2653f93cf66ffc151e058ff09a985a61a22f84adff79f01dfccbfee68dfafae",
           1: "8c1b07503c586ea751795d461d12c6b4bc4e9d1fe7df6c8297370b30cb56e1cf",
           2: "110399d28ef21477a1524cd95ee38a81543a5db63a67eb4d9727f247e8581c1b",
           3: "b672b2efdb52e31168e340f5cb48ede4b74eac14e2f6960a2982ba90504b4f36",
           4: "e41ef57489f25b6acaa97ac0e1ec7e0daeef3189653db8edba898d9b06cc3711"}
for s in range(5):
    fp = f"{FOLDDIR}/kcl_groupkfold5_pod_seed{s}_UNVERIFIED.csv"
    fr = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in rows(fp)}
    same_set = set(fr) == set(clips) and len(fr) == len(clips)
    fold_eq = same_set and all(fr[c][0] == F[i, s] for i, c in enumerate(clips))
    spk_eq = same_set and all(fr[c][1] == spk[i] for i, c in enumerate(clips))
    chk(f"seed{s}_fold_file_sha_matches_sidecar", sha(fp) == exp_sha[s])
    chk(f"seed{s}_folds_equal_release_file", fold_eq and spk_eq, f"clips_same={same_set}")
    # every speaker in exactly one fold (no speaker leakage across outer folds)
    ok = all(len(set(F[spk == u, s])) == 1 for u in np.unique(spk))
    chk(f"seed{s}_speaker_grouped", ok, f"fold sizes {np.bincount(F[:, s]).tolist()}")
# rebuild the seed splits from first principles: default_rng(seed) renumbering + largest-first greedy fill, stable ties
def greedy(groups, k):
    u, inv = np.unique(groups, return_inverse=True); cnt = np.bincount(inv)
    order = np.argsort(cnt, kind="stable")[::-1]; load = np.zeros(k); fg = np.zeros(len(u), int)
    for gi in order:
        f = int(np.argmin(load)); fg[gi] = f; load[f] += cnt[gi]
    return fg[inv]
for s in range(5):
    u = np.unique(spk); perm = {v: i for i, v in enumerate(np.random.default_rng(s).permutation(u))}
    g = np.array([perm[v] for v in spk])
    chk(f"seed{s}_fold_file_rebuilt_from_rng", np.array_equal(greedy(g, 5), F[:, s]))

# ---------- perclip probe columns equal pod OOF csv
pmap = {r["clip"]: r for r in pod}
okp = set(pmap) == set(clips) and all(
    float(pmap[c][f"p_seed{s}"]) == P[i, s] and int(pmap[c][f"fold_seed{s}"]) == F[i, s]
    and pmap[c]["speaker"] == spk[i] and int(pmap[c]["label"]) == y[i]
    for i, c in enumerate(clips) for s in range(5))
chk("perclip_probe_equals_pod_oof_exact", okp)

# ---------- zero-shot alignment
zmap = {r["clip"]: r for r in zs}
chk("zeroshot_clip_set_equal", set(zmap) == set(clips) and len(zs) == len(clips))
chk("zeroshot_speakers_equal", all(zmap[c]["speaker"] == spk[i] for i, c in enumerate(clips)))
chk("zeroshot_labels_equal", all(int(zmap[c]["label"]) == y[i] for i, c in enumerate(clips)))
A = np.array([float(zmap[c]["p_yes"]) for c in clips])
chk("perclip_zeroshot_column_equals_file", np.array_equal(A, np.array([float(r["p_yes_zeroshot"]) for r in pc])))
# states cross-check (read only): names, speakers, labels, p_yes
z = np.load(STATES, allow_pickle=True)
smap = {n: i for i, n in enumerate(z["name"].astype(str))}
okst = set(smap) == set(clips) and all(
    str(z["spk"][smap[c]]) == spk[i] and int(z["label"][smap[c]]) == y[i] and float(z["p_yes"][smap[c]]) == A[i]
    for i, c in enumerate(clips))
chk("states_names_speakers_labels_pyes_equal", okst)
# label from filename (hc=0, pd=1) sanity
chk("label_matches_filename_tag", all((("_pd_" in c) == (y[i] == 1)) for i, c in enumerate(clips)))

# ---------- point values
per = [auc(y, P[:, s]) for s in range(5)]; mean5 = float(np.mean(per)); ans = auc(y, A); gap = mean5 - ans
from sklearn.metrics import roc_auc_score
per_sk = [roc_auc_score(y, P[:, s]) for s in range(5)]; ans_sk = roc_auc_score(y, A)
chk("own_auc_equals_sklearn", max(abs(a - b) for a, b in zip(per + [ans], per_sk + [ans_sk])) < 1e-12)
out["values"].update(per_repeat=per, mean5=mean5, answer=ans, gap=gap)
chk("per_repeat_4dp_equal_claim", [round(a, 4) for a in per] == CLAIM["per_repeat"], [round(a, 4) for a in per])
chk("probe_4dp_equal_claim", round(mean5, 4) == CLAIM["probe"], round(mean5, 6))
chk("answer_4dp_equal_claim", round(ans, 4) == CLAIM["answer"], round(ans, 6))
chk("gap_4dp_equal_claim", round(gap, 4) == CLAIM["gap"], round(gap, 6))
t1 = json.load(open(T1))["enc"]
chk("per_repeat_equal_table1_json_4dp", [round(a, 4) for a in per] == t1["per_repeat"] and round(mean5, 4) == t1["mean"],
    f"table1 {t1['per_repeat']} mean {t1['mean']}")

# ---------- speaker bootstrap, own implementation
u = np.unique(spk); idx_of = {v: np.where(spk == v)[0] for v in u}
rng = np.random.default_rng(0); B = 2000
diffs = []; mids = []; zsd = []; drawn = []; degen = 0
for b in range(B):
    pick = rng.choice(len(u), size=len(u), replace=True); drawn.append(pick)
    ii = np.concatenate([idx_of[u[k]] for k in pick]); yy = y[ii]
    if yy.min() == yy.max(): degen += 1; continue
    m = np.mean([auc(yy, P[ii, s]) for s in range(5)]); a = auc(yy, A[ii])
    diffs.append(m - a); mids.append(m); zsd.append(a)
diffs = np.array(diffs); lo, hi = np.percentile(diffs, [2.5, 97.5])
out["values"].update(lo=float(lo), hi=float(hi), usable=len(diffs), degenerate=degen)
chk("bootstrap_2000_usable", len(diffs) == 2000, f"usable {len(diffs)} degenerate {degen}")
chk("lo_4dp_equal_claim", round(lo, 4) == CLAIM["lo"], round(lo, 6))
chk("hi_4dp_equal_claim", round(hi, 4) == CLAIM["hi"], round(hi, 6))
# alternative draw call: rng.integers gives the same indices?
rng2 = np.random.default_rng(0); same_int = all(np.array_equal(rng2.integers(0, len(u), len(u)), d) for d in drawn)
out["values"]["choice_equals_integers"] = bool(same_int)
# vs saved draws
d = np.load(DRAWS, allow_pickle=True)
chk("saved_speakers_order_equal", np.array_equal(d["speakers"].astype(str), u))
chk("saved_speaker_idx_equal_own", np.array_equal(d["speaker_idx"], np.array(drawn)))
chk("saved_diff_equal_own", np.allclose(d["diff"], diffs, atol=1e-12, rtol=0), f"max abs {np.max(np.abs(d['diff'] - diffs)):.2e}")
chk("saved_zeroshot_draws_equal_own", np.allclose(d["zeroshot"], zsd, atol=1e-12, rtol=0))
chk("saved_mean5_draws_equal_own", np.allclose(d["mean5"], mids, atol=1e-12, rtol=0))
chk("saved_point_diff_equal", abs(float(d["point_diff"]) - gap) < 1e-12)
chk("interval_above_zero_false", bool(lo > 0) is False, "interval includes zero")

json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
allpass = all(v["pass"] for v in out["checks"].values())
print("ALL PASS" if allpass else "SOME FAIL", json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in out["values"].items() if k != "per_repeat"}))
print("per_repeat", [round(a, 6) for a in per])
