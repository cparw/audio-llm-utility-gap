"""Independent verifier, PART 26, Qwen3-Omni-30B-A3B on Pitt.
Own code: rank AUC, own joins, own speaker bootstrap. Reads only."""
import csv, hashlib, json, sys
import numpy as np

D = "scores/part26/Qwen3-Omni-30B-A3B_Pitt"
OOF = D + "/pull/p26-q3o-pitt/out/p26_q3o_pitt_enc_nested5_oof.csv"
PERCLIP = D + "/per_clip/Qwen3-Omni-30B-A3B_Pitt_perclip.csv"
DRAWS = D + "/draws/Qwen3-Omni-30B-A3B_Pitt_draws.npz"
POD3 = "<local data dir>/release/edaic_rerun/part16/POD3/q3o_pitt_encoder_nested_oof.csv"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv"
FOLDS = "<local data dir>/release/folds/pitt_groupkfold5_pod_Control15ids_seed{}.csv"

def read(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def auc_rank(y, s):
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
    npos = y.sum(); nneg = len(y) - npos
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

def auc_pairs(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    p = s[y == 1][:, None]; n = s[y == 0][None, :]
    return ((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size)

out = {}
fails = []
def check(name, ok, detail=None):
    out[name] = {"pass": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)

# ---- load
_, oof = read(OOF)
_, zs = read(ZS)
_, pc = read(PERCLIP)
_, p3 = read(POD3)
out["sha256"] = {"oof": sha(OOF), "zs": sha(ZS), "perclip": sha(PERCLIP), "draws": sha(DRAWS), "pod3": sha(POD3)}

clips = [r["clip"] for r in oof]
check("oof_n468_unique", len(clips) == 468 and len(set(clips)) == 468, len(clips))
zmap = {r["clip"]: r for r in zs}
check("zs_n468_unique", len(zs) == 468 and len(zmap) == 468, len(zs))
check("same_clip_set_oof_zs", set(clips) == set(zmap), len(set(clips) ^ set(zmap)))

y = np.array([int(r["label"]) for r in oof])
yz = np.array([int(zmap[c]["label"]) for c in clips])
check("labels_equal_oof_vs_zs", np.array_equal(y, yz), int((y != yz).sum()))
spk = np.array([r["speaker"] for r in oof])
spk_z = np.array([zmap[c]["speaker"] for c in clips])
# speaker bijection
fwd = {}; bwd = {}; bij = True
for a, b in zip(spk, spk_z):
    if fwd.setdefault(a, b) != b or bwd.setdefault(b, a) != a:
        bij = False
check("speaker_bijection_oof_vs_zs", bij, {"n_spk_oof": len(set(spk)), "n_spk_zs": len(set(spk_z))})
# ids differ only by zero padding
def norm(s):
    import re
    m = re.match(r"([A-Za-z]+)0*(\d+)$", s)
    return (m.group(1), int(m.group(2))) if m else s
check("speaker_ids_equal_up_to_zero_padding", all(norm(a) == norm(b) for a, b in zip(spk, spk_z)))
# clip name encodes speaker? check prefix group per speaker constant label
lab_by_spk = {}
cons = True
for s_, l_ in zip(spk, y):
    if lab_by_spk.setdefault(s_, l_) != l_:
        cons = False
check("label_constant_within_speaker", cons)
out["n"] = int(len(y)); out["n_pos"] = int(y.sum()); out["n_spk"] = int(len(set(spk)))

# ---- folds vs release files, speaker disjointness
P = np.array([[float(r["p_seed%d" % k]) for k in range(5)] for r in oof])
FO = np.array([[int(r["fold_seed%d" % k]) for k in range(5)] for r in oof])
fold_detail = {}
for k in range(5):
    _, fr = read(FOLDS.format(k))
    fm = {r["clip_id"]: r for r in fr}
    ok_set = set(fm) == set(clips) and len(fr) == 468
    ok_fold = all(int(fm[c]["fold"]) == FO[i, k] for i, c in enumerate(clips))
    ok_spk = all(fm[c]["speaker_id"] == spk[i] for i, c in enumerate(clips))
    # speaker disjoint across outer folds
    sf = {}
    disj = True
    for s_, f_ in zip(spk, FO[:, k]):
        if sf.setdefault(s_, f_) != f_:
            disj = False
    sizes = np.bincount(FO[:, k], minlength=5).tolist()
    fold_detail[k] = {"file_sha256": sha(FOLDS.format(k)), "clipset": ok_set, "fold_equal": ok_fold,
                      "speaker_equal": ok_spk, "speaker_disjoint": disj, "sizes": sizes}
    check("fold_seed%d_matches_release_file" % k, ok_set and ok_fold and ok_spk, fold_detail[k])
    check("fold_seed%d_speaker_disjoint" % k, disj)
out["folds"] = fold_detail

# ---- per-repeat AUCs
pr = [auc_rank(y, P[:, k]) for k in range(5)]
pr2 = [auc_pairs(y, P[:, k]) for k in range(5)]
check("rank_auc_equals_pairwise_auc", max(abs(a - b) for a, b in zip(pr, pr2)) < 1e-12)
mean5 = float(np.mean(pr))
out["per_repeat_auc"] = pr; out["mean5"] = mean5
claimed_pr = [0.819905, 0.801569, 0.814161, 0.834212, 0.791379]
check("per_repeat_match_claim_4dp", all(round(a, 4) == round(b, 4) for a, b in zip(pr, claimed_pr)) and
      max(abs(a - b) for a, b in zip(pr, claimed_pr)) < 5e-6, [round(a, 6) for a in pr])
check("mean5_match_claim_4dp", round(mean5, 4) == 0.8122 and abs(mean5 - 0.812245) < 5e-6, mean5)

# ---- answer AUC
pz = np.array([float(zmap[c]["p_yes"]) for c in clips])
ans = auc_rank(y, pz)
out["answer_auc"] = ans
check("answer_match_claim_4dp", round(ans, 4) == 0.7619 and abs(ans - 0.761933) < 5e-6, ans)
gap = mean5 - ans
out["gap_point"] = gap
check("gap_match_claim_4dp", round(gap, 4) == 0.0503 and abs(gap - 0.050312) < 5e-6, gap)

# ---- per-clip file consistency with OOF and zero-shot
pcm = {r["clip"]: r for r in pc}
ok = len(pc) == 468 and set(pcm) == set(clips)
mx = 0.0; mxz = 0.0; fok = True; lok = True; sok = True
for i, c in enumerate(clips):
    r = pcm[c]
    for k in range(5):
        mx = max(mx, abs(float(r["p_probe_seed%d" % k]) - P[i, k]))
        fok &= int(r["fold_seed%d" % k]) == FO[i, k]
    mxz = max(mxz, abs(float(r["p_yes_zeroshot"]) - pz[i]))
    lok &= int(r["label"]) == y[i]
    sok &= (r["speaker"] == spk[i]) and (r["speaker_padded"] == spk_z[i])
check("perclip_csv_equals_oof_and_zs", ok and mx == 0 and mxz == 0 and fok and lok and sok,
      {"max_probe_diff": mx, "max_zs_diff": mxz, "folds": fok, "labels": lok, "speakers": sok})

# ---- POD3 source cross-check (Table 1 origin)
p3m = {r["clip"]: r for r in p3}
out["pod3_columns"] = list(p3[0].keys())

# ---- speaker bootstrap, my own implementation
def boot(spk_ids, seed=0, B=2000, use_choice=False):
    uniq = np.unique(spk_ids)
    rows = [np.flatnonzero(spk_ids == u) for u in uniq]
    rng = np.random.default_rng(seed)
    d = np.empty(B); m5 = np.empty(B); zz = np.empty(B); idxs = np.empty((B, len(uniq)), int)
    for b in range(B):
        idx = rng.choice(len(uniq), size=len(uniq), replace=True) if use_choice else rng.integers(0, len(uniq), len(uniq))
        idxs[b] = idx
        ii = np.concatenate([rows[j] for j in idx])
        yy = y[ii]
        m5[b] = np.mean([auc_rank(yy, P[ii, k]) for k in range(5)])
        zz[b] = auc_rank(yy, pz[ii])
        d[b] = m5[b] - zz[b]
    return uniq, idxs, d, m5, zz

uA, iA, dA, mA, zA = boot(spk)                       # probe-file ids (Control15 style)
uA2, iA2, dA2, _, _ = boot(spk, use_choice=True)
check("rng_integers_equals_rng_choice", np.array_equal(iA, iA2))
loA, hiA = np.percentile(dA, [2.5, 97.5])
out["boot_probe_ids"] = {"lo": loA, "hi": hiA, "p_le0": float((dA <= 0).mean()),
                         "probe_ci": np.percentile(mA, [2.5, 97.5]).tolist(),
                         "zs_ci": np.percentile(zA, [2.5, 97.5]).tolist()}
check("interval_match_claim_4dp", round(loA, 4) == -0.0087 and round(hiA, 4) == 0.1095, [loA, hiA])
uB, iB, dB, _, _ = boot(spk_z)                       # zero-padded ids
loB, hiB = np.percentile(dB, [2.5, 97.5])
out["boot_padded_ids"] = {"lo": loB, "hi": hiB}
check("padded_sensitivity_match_claim_4dp", round(loB, 4) == -0.0058 and round(hiB, 4) == 0.1108, [loB, hiB])

# ---- compare with saved draws
z = np.load(DRAWS, allow_pickle=True)
check("saved_speakers_equal_mine", np.array_equal(z["speakers"], uA))
check("saved_speaker_idx_equal_mine", np.array_equal(z["speaker_idx"].astype(int), iA))
check("saved_diff_equal_mine", float(np.max(np.abs(z["diff"] - dA))) < 1e-12, float(np.max(np.abs(z["diff"] - dA))))
check("saved_mean5_equal_mine", float(np.max(np.abs(z["mean5"] - mA))) < 1e-12)
check("saved_zeroshot_equal_mine", float(np.max(np.abs(z["zeroshot"] - zA))) < 1e-12)
check("saved_point_equal_mine", abs(float(z["point_diff"]) - gap) < 1e-12)
check("saved_padded_diff_equal_mine", float(np.max(np.abs(z["sens_padded_ids_diff"] - dB))) < 1e-12)
check("saved_draws_finite_2000", z["diff"].shape == (2000,) and np.isfinite(z["diff"]).all())

out["fails"] = fails
out["all_pass"] = not fails
json.dump(out, sys.stdout, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
