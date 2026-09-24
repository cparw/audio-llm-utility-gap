"""Independent verifier, PART26 cell Qwen3-Omni-30B-A3B x ADReSS-2020.
Written from scratch. Reads only; writes a json next to this script."""
import csv, json, hashlib, sys, os
import numpy as np
from scipy.stats import rankdata

CELL = "scores/part26/Qwen3-Omni-30B-A3B_ADReSS-2020"
PERCLIP = CELL + "/per_clip/p26_q3o_adress2020_perclip.csv"
DRAWS = CELL + "/draws/p26_q3o_adress2020_draws.npz"
PODOOF = CELL + "/pull/p26-q3o-adress2020/out/p26_q3o_adress2020_enc_nested5_oof.csv"
PODJSON = CELL + "/pull/p26-q3o-adress2020/out/p26_q3o_adress2020_enc_nested5.json"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_adress2020_zeroshot_scores.csv"
STATES = "<local data dir>/release/overnight2/q3o/q3o_adress2020_states.npz"
FOLDDIR = "<local data dir>/release/folds"
P25C = "<local data dir>/release_from_mac/scores/part25/C/perclip/C_q3o_adress2020_enc_perclip.csv"
CLAIM = dict(probe=0.8445, per=[0.8337, 0.8555, 0.8458, 0.8401, 0.8475], answer=0.7097, gap=0.1348, lo=0.0417, hi=0.2335)

out = {"checks": {}, "values": {}}
def chk(name, ok, **info):
    out["checks"][name] = bool(ok)
    if info: out.setdefault("info", {})[name] = info
    print(("PASS " if ok else "FAIL ") + name, info if info else "")

def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()

def rows(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))

# my AUC: Mann-Whitney U with midranks
def auc_rank(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0: return np.nan
    r = rankdata(s)  # average ranks for ties
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

# second AUC: explicit pair counting (ties count half)
def auc_pairs(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    p = s[y == 1][:, None]; n = s[y == 0][None, :]
    return ((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size)

# ---------- load per-clip file ----------
R = rows(PERCLIP)
clips = [r["clip"] for r in R]; spk = np.array([r["speaker"] for r in R]); y = np.array([int(r["label"]) for r in R])
P = np.array([[float(r[f"p_probe_seed{s}"]) for s in range(5)] for r in R])  # n x 5
FO = np.array([[int(r[f"fold_seed{s}"]) for s in range(5)] for r in R])
Z = np.array([float(r["p_yes_zeroshot"]) for r in R])
n = len(R)
chk("perclip_n156_unique_clips", n == 156 and len(set(clips)) == n, n=n)
chk("perclip_one_clip_per_speaker_156", len(np.unique(spk)) == 156, n_spk=int(len(np.unique(spk))))
chk("perclip_78_pos", int(y.sum()) == 78, n_pos=int(y.sum()))
chk("perclip_no_nan", np.isfinite(P).all() and np.isfinite(Z).all())
chk("clip_name_is_speaker_wav", all(c == s + ".wav" for c, s in zip(clips, spk)))

# ---------- zero-shot file alignment ----------
ZR = {r["clip"]: r for r in rows(ZS)}
chk("zeroshot_sha256_as_sidecar", sha(ZS) == "d938609e6e30af5a8313db9b634cb3ebd011c466d9d9a1ddc1057cb0c361713b")
chk("zeroshot_same_clip_set", set(ZR) == set(clips) and len(ZR) == n)
chk("zeroshot_speaker_label_match", all(ZR[c]["speaker"] == s and int(ZR[c]["label"]) == l for c, s, l in zip(clips, spk, y)))
zs_file = np.array([float(ZR[c]["p_yes"]) for c in clips])
chk("zeroshot_column_equals_file_exact", np.array_equal(zs_file, Z), max_abs=float(np.abs(zs_file - Z).max()))
prompts = {ZR[c]["prompt"] for c in clips}
chk("zeroshot_single_prompt", len(prompts) == 1, prompt=list(prompts)[0])

# ---------- states npz alignment ----------
st = np.load(STATES, allow_pickle=True)
sn = st["name"].astype(str); ss = st["spk"].astype(str); sl = st["label"].astype(int)
chk("states_rows_align_perclip", list(sn) == clips and list(ss) == list(spk) and np.array_equal(sl, y))
chk("states_enc_shape_156_32_1280", tuple(st["enc"].shape) == (156, 32, 1280), shape=list(st["enc"].shape))

# ---------- pod OOF file equals per-clip ----------
PR = {r["clip"]: r for r in rows(PODOOF)}
chk("podoof_same_clips", set(PR) == set(clips))
same = all(float(PR[c][f"p_seed{s}"]) == P[i, s] and int(PR[c][f"fold_seed{s}"]) == FO[i, s]
           and PR[c]["speaker"] == spk[i] and int(PR[c]["label"]) == y[i] for i, c in enumerate(clips) for s in range(5))
chk("podoof_probe_and_fold_columns_equal_perclip", same)

# ---------- folds vs saved release fold files ----------
pj = json.load(open(PODJSON))
fold_info = {}
for s in range(5):
    fp = os.path.join(FOLDDIR, f"adress2020_groupkfold5_pod_seed{s}_UNVERIFIED.csv")
    fr = {r["clip_id"]: r for r in rows(fp)}
    ok_set = set(fr) == set(clips) and len(fr) == n
    ok_spk = all(fr[c]["speaker_id"] == spk[i] for i, c in enumerate(clips))
    ok_fold = all(int(fr[c]["fold"]) == FO[i, s] for i, c in enumerate(clips))
    h = sha(fp)
    ok_sha = h == pj["tags"][f"seed{s}"]["fold_file_sha256"] and pj["tags"][f"seed{s}"]["fold_file"] == os.path.basename(fp)
    sizes = np.bincount(FO[:, s], minlength=5).tolist()
    # speaker disjointness across folds
    disj = all(len(set(FO[spk == u, s])) == 1 for u in np.unique(spk))
    # every fold has both classes
    both = all(len(set(y[FO[:, s] == k])) == 2 for k in range(5))
    fold_info[f"seed{s}"] = dict(sha256=h, sizes=sizes)
    chk(f"seed{s}_fold_file_matches_perclip_clips_speakers_folds", ok_set and ok_spk and ok_fold)
    chk(f"seed{s}_fold_file_sha256_equals_pod_json", ok_sha)
    chk(f"seed{s}_folds_speaker_disjoint_both_classes", disj and both, sizes=sizes)

# rebuild the renumbered pod-tie-rule folds from scratch (my own greedy fill)
def my_gkf(groups, k):
    u, inv = np.unique(groups, return_inverse=True); cnt = np.bincount(inv)
    order = np.argsort(cnt, kind="stable")[::-1]
    load = np.zeros(k); fg = np.empty(len(u), int)
    for gi in order:
        f = int(np.argmin(load)); fg[gi] = f; load[f] += cnt[gi]
    return fg[inv]
for s in range(5):
    rng = np.random.default_rng(s); u = np.unique(spk); perm = rng.permutation(u)
    newid = {sp: i for i, sp in enumerate(perm)}
    g = np.array([newid[x] for x in spk])
    chk(f"seed{s}_folds_rebuilt_from_scratch_equal", np.array_equal(my_gkf(g, 5), FO[:, s]))

# check fold and layer consistency with the pod json and checkpoint
CK = json.load(open(CELL + "/pull/p26-q3o-adress2020/out/p26_q3o_adress2020_ckpt.json"))
CLAIM_LAYERS = {"seed0": [19, 18, 31, 25, 29], "seed1": [18, 19, 29, 26, 25], "seed2": [28, 18, 30, 19, 19],
                "seed3": [30, 22, 22, 29, 19], "seed4": [19, 29, 19, 20, 30]}
for s in range(5):
    t = CK[f"seed{s}"]
    idx = [clips.index(c) for c in clips]
    chk(f"seed{s}_ckpt_fold_and_oof_equal_perclip", t["fold"] == FO[:, s].tolist() and
        all(float(t["oof"][i]) == P[i, s] for i in range(n)))
    chk(f"seed{s}_layers_equal_claim", t["layers"] == CLAIM_LAYERS[f"seed{s}"] == pj["tags"][f"seed{s}"]["layers"], layers=t["layers"])
    ok_arg = all(int(np.argmax(t["inner"][k])) == t["layers"][k] and len(t["inner"][k]) == 32 for k in range(5))
    chk(f"seed{s}_layers_are_argmax_of_saved_inner_scores", ok_arg)

# ---------- AUCs ----------
per_rank = [auc_rank(y, P[:, s]) for s in range(5)]
per_pair = [auc_pairs(y, P[:, s]) for s in range(5)]
mean5 = float(np.mean(per_rank))
ans_rank = auc_rank(y, Z); ans_pair = auc_pairs(y, Z)
gap = mean5 - ans_rank
from sklearn.metrics import roc_auc_score
per_sk = [roc_auc_score(y, P[:, s]) for s in range(5)]
chk("auc_rank_equals_pairs_equals_sklearn", max(abs(a - b) for a, b in zip(per_rank, per_pair)) < 1e-12
    and max(abs(a - b) for a, b in zip(per_rank, per_sk)) < 1e-12 and abs(ans_rank - ans_pair) < 1e-12)
out["values"].update(per_repeat=per_rank, mean5=mean5, answer=ans_rank, gap=gap)
chk("per_repeat_4dp_match_claim", [round(a, 4) for a in per_rank] == CLAIM["per"], mine=[round(a, 4) for a in per_rank])
chk("mean5_4dp_match_claim", round(mean5, 4) == CLAIM["probe"], mine=round(mean5, 4))
chk("answer_4dp_match_claim", round(ans_rank, 4) == CLAIM["answer"], mine=round(ans_rank, 4))
chk("gap_4dp_match_claim", round(gap, 4) == CLAIM["gap"], mine=round(gap, 4))
chk("per_repeat_equal_pod_json_1e-12", max(abs(a - b) for a, b in zip(per_rank, pj["per_repeat_auc"])) < 1e-12)

# ---------- speaker bootstrap ----------
rng = np.random.default_rng(0)
U = np.unique(spk); ns = len(U)
rows_of = [np.where(spk == u)[0] for u in U]
D = np.full(2000, np.nan); M5 = np.full(2000, np.nan); ZB = np.full(2000, np.nan); IDX = np.zeros((2000, ns), int)
usable = 0
for b in range(2000):
    idx = rng.choice(ns, size=ns, replace=True); IDX[b] = idx
    r = np.concatenate([rows_of[i] for i in idx])
    yb = y[r]
    if yb.min() == yb.max(): continue
    m = np.mean([auc_rank(yb, P[r, s]) for s in range(5)]); zb = auc_rank(yb, Z[r])
    M5[b] = m; ZB[b] = zb; D[b] = m - zb; usable += 1
ok = np.isfinite(D)
lo, hi = np.percentile(D[ok], [2.5, 97.5])
out["values"].update(lo=float(lo), hi=float(hi), usable=int(usable), p_le0=float((D[ok] <= 0).mean()))
chk("usable_2000", usable == 2000, usable=usable)
chk("lo_4dp_match_claim", round(float(lo), 4) == CLAIM["lo"], mine=round(float(lo), 4))
chk("hi_4dp_match_claim", round(float(hi), 4) == CLAIM["hi"], mine=round(float(hi), 4))

# compare with saved draws
dz = np.load(DRAWS, allow_pickle=True)
chk("draws_speakers_equal_unique_order", list(dz["speakers"]) == list(U))
chk("draws_speaker_idx_equal_mine", np.array_equal(dz["speaker_idx"].astype(int), IDX))
chk("draws_diff_equal_mine_1e-12", np.nanmax(np.abs(dz["diff"] - D)) < 1e-12, max_abs=float(np.nanmax(np.abs(dz["diff"] - D))))
chk("draws_mean5_zeroshot_equal_mine_1e-12", np.nanmax(np.abs(dz["mean5"] - M5)) < 1e-12 and np.nanmax(np.abs(dz["zeroshot"] - ZB)) < 1e-12)
chk("draws_ci_equal_mine_1e-12", np.abs(dz["ci_diff"] - np.array([lo, hi])).max() < 1e-12)
chk("draws_point_equal_mine_1e-12", abs(float(dz["point_diff"]) - gap) < 1e-12 and abs(float(dz["point_probe_mean5"]) - mean5) < 1e-12
    and abs(float(dz["point_zeroshot"]) - ans_rank) < 1e-12)

# ---------- informational: PART25 C per-clip for the same cell ----------
try:
    C = {r["clip"]: r for r in rows(P25C)}
    cols = [k for k in next(iter(C.values())).keys()]
    out["info_part25C_columns"] = cols
except Exception as e:
    out["info_part25C_error"] = repr(e)

out["fold_files"] = fold_info
out["all_pass"] = all(out["checks"].values())
out["n_checks"] = len(out["checks"])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "iv_q3o_adress2020.json"), "w"), indent=1)
print("ALL PASS" if out["all_pass"] else "SOME FAIL", out["n_checks"], "checks")
print(json.dumps(out["values"], indent=1))
