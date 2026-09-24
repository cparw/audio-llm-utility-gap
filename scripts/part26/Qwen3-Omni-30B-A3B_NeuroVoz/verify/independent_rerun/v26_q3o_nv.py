"""Independent verifier, PART26 Qwen3-Omni-30B-A3B on NeuroVoz. Own code: rank AUC, own join, own bootstrap."""
import csv, json, hashlib, sys, numpy as np

B = "scores/part26/Qwen3-Omni-30B-A3B_NeuroVoz"
PERCLIP = B + "/per_clip/Qwen3-Omni-30B-A3B_NeuroVoz_perclip.csv"
PODOOF = B + "/pull/p26-q3o-neurovoz/out/p26_q3o_neurovoz_enc_nested5_oof.csv"
PODJSON = B + "/pull/p26-q3o-neurovoz/out/p26_q3o_neurovoz_enc_nested5.json"
CKPT = B + "/pull/p26-q3o-neurovoz/out/p26_q3o_neurovoz_ckpt.json"
DRAWS = B + "/draws/Qwen3-Omni-30B-A3B_NeuroVoz_draws.npz"
STATES = B + "/stage/q3o_neurovoz_encstates.npz"
SRC_STATES = "<local data dir>/release/overnight2/q3o/q3o_neurovoz_states.npz"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_neurovoz_zeroshot_scores.csv"
FD = "<local data dir>/release/folds/"
FF = {f"seed{s}": FD + f"neurovoz_groupkfold5_pod_seed{s}_UNVERIFIED.csv" for s in range(5)}
FF["single"] = FD + "neurovoz_groupkfold5_pod.csv"
CLAIM = dict(probe=0.9372, answer=0.6393, gap=0.2980, lo=0.2360, hi=0.3578,
             per=[0.9329, 0.9370, 0.9414, 0.9371, 0.9377])
TAGS5 = [f"seed{s}" for s in range(5)]
out = {"checks": {}}
def chk(name, ok, **info):
    out["checks"][name] = {"ok": bool(ok), **info}
    print(("PASS " if ok else "FAIL ") + name, info if info else "", flush=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def auc(y, s):
    """Mann-Whitney AUC with midranks for ties, written from scratch."""
    y = np.asarray(y); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort"); ss = s[order]
    r = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = int((y == 1).sum()); nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

def pairwise_auc(y, s):
    p = s[y == 1]; n = s[y == 0]
    gt = (p[:, None] > n[None, :]).sum(); eq = (p[:, None] == n[None, :]).sum()
    return (gt + 0.5 * eq) / (len(p) * len(n))

def readcsv(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))

# ---------- load
pc = readcsv(PERCLIP); pod = readcsv(PODOOF); zs = readcsv(ZS)
folds = {t: {r["clip_id"]: r for r in readcsv(p)} for t, p in FF.items()}
z = np.load(STATES, allow_pickle=True); zsrc = np.load(SRC_STATES, allow_pickle=True)
names = z["name"].astype(str); spk = z["spk"].astype(str); lab = z["label"].astype(int)

clips = [r["clip"] for r in pc]
chk("perclip_rows_1270_unique", len(pc) == 1270 and len(set(clips)) == 1270, n=len(pc))
chk("perclip_same_clip_order_as_states", clips == names.tolist())
chk("perclip_speaker_equals_states", [r["speaker"] for r in pc] == spk.tolist())
chk("perclip_label_equals_states", [int(r["label"]) for r in pc] == lab.tolist())
chk("states_counts", len(np.unique(spk)) == 107 and lab.sum() == 621 and z["enc"].shape == (1270, 32, 1280),
    n_spk=int(len(np.unique(spk))), n_pos=int(lab.sum()), enc_shape=list(z["enc"].shape))
# each speaker has one label
spk_lab = {}
for s_, l_ in zip(spk, lab): spk_lab.setdefault(s_, set()).add(int(l_))
chk("each_speaker_single_label", all(len(v) == 1 for v in spk_lab.values()))
# clip name encodes speaker number and HC/PD label
def parse(n):
    base = n.rsplit(".", 1)[0]; grp = base.split("_")[0]; num = base.split("_")[-1]
    return grp, str(int(num))
bad_spk = sum(parse(n)[1] != s_ for n, s_ in zip(names, spk))
bad_lab = sum((parse(n)[0] == "PD") != bool(l_) for n, l_ in zip(names, lab))
chk("clip_name_matches_speaker_and_label", bad_spk == 0 and bad_lab == 0, bad_spk=bad_spk, bad_lab=bad_lab,
    groups=sorted(set(parse(n)[0] for n in names)))
# stage states byte-equal to source states for shared keys
same = {k: bool(np.array_equal(z[k], zsrc[k])) for k in z.files}
chk("stage_states_equal_source_states", all(same.values()), keys=same)

# ---------- per-clip vs pod raw OOF
podc = {r["clip"]: r for r in pod}
chk("pod_oof_same_clips", set(podc) == set(clips) and len(pod) == 1270)
maxd = 0.0; fold_mis = 0; meta_mis = 0
for r in pc:
    q = podc[r["clip"]]
    for s in range(5):
        maxd = max(maxd, abs(float(r[f"p_probe_seed{s}"]) - float(q[f"p_seed{s}"])))
        fold_mis += int(r[f"fold_seed{s}"] != q[f"fold_seed{s}"])
    maxd = max(maxd, abs(float(r["p_probe_single_rerun"]) - float(q["p_single"])))
    fold_mis += int(r["fold_single"] != q["fold_single"])
    meta_mis += int(r["speaker"] != q["speaker"] or r["label"] != q["label"])
chk("perclip_equals_pod_oof", maxd == 0.0 and fold_mis == 0 and meta_mis == 0, max_abs=maxd, fold_mis=fold_mis, meta_mis=meta_mis)

# ---------- folds vs saved release fold files
fsha = {t: sha(p) for t, p in FF.items()}
pj = json.load(open(PODJSON)); ck = json.load(open(CKPT))
for t in TAGS5 + ["single"]:
    col = "fold_single" if t == "single" else f"fold_{t}"
    ff = folds[t]
    ok_set = set(ff) == set(clips)
    mis = sum(int(ff[r["clip"]]["fold"]) != int(r[col]) for r in pc)
    spkmis = sum(ff[r["clip"]]["speaker_id"] != r["speaker"] for r in pc)
    # speaker disjointness and 5 folds
    sf = {}
    for r in pc: sf.setdefault(r["speaker"], set()).add(int(r[col]))
    disjoint = all(len(v) == 1 for v in sf.values())
    nf = sorted(set(int(r[col]) for r in pc))
    sha_ok = pj["tags"][t]["fold_file_sha256"] == fsha[t] and pj["tags"][t]["fold_file"] == FF[t].rsplit("/", 1)[1]
    ck_ok = ck[t]["fold"] == [int(folds[t][n]["fold"]) for n in names]
    chk(f"folds_{t}", ok_set and mis == 0 and spkmis == 0 and disjoint and nf == [0, 1, 2, 3, 4] and sha_ok and ck_ok,
        file=FF[t].rsplit("/", 1)[1], clip_set_equal=ok_set, fold_mismatch=mis, speaker_mismatch=spkmis,
        speaker_disjoint=disjoint, folds=nf, pod_json_sha_and_name_match=sha_ok, ckpt_fold_match=ck_ok, sha256=fsha[t])
# the five repeat partitions are genuinely different from each other
parts = {t: tuple(int(folds[t][n]["fold"]) for n in names) for t in TAGS5}
chk("five_repeat_partitions_distinct", len(set(parts.values())) == 5)

# ---------- per-repeat AUCs
y = np.array([int(r["label"]) for r in pc])
P = np.array([[float(r[f"p_probe_seed{s}"]) for s in range(5)] for r in pc])
per = [auc(y, P[:, s]) for s in range(5)]
per_pw = [pairwise_auc(y, P[:, s]) for s in range(5)]
from sklearn.metrics import roc_auc_score
per_sk = [roc_auc_score(y, P[:, s]) for s in range(5)]
mean5 = float(np.mean(per))
out["per_repeat"] = per; out["mean5"] = mean5
chk("per_repeat_three_ways_agree", max(abs(a - b) for a, b in zip(per, per_pw)) < 1e-12 and max(abs(a - b) for a, b in zip(per, per_sk)) < 1e-12)
chk("per_repeat_4dp_match_claim", [round(a, 4) for a in per] == CLAIM["per"], mine=[round(a, 4) for a in per], claim=CLAIM["per"])
chk("per_repeat_match_pod_json", max(abs(a - b) for a, b in zip(per, pj["per_repeat_auc"])) < 1e-12)
chk("mean5_4dp_match_claim", round(mean5, 4) == CLAIM["probe"], mine=mean5)
chk("no_missing_oof", np.isfinite(P).all() and ((P >= 0) & (P <= 1)).all())
ps = np.array([float(r["p_probe_single_rerun"]) for r in pc])
out["single_auc"] = auc(y, ps)
chk("single_split_auc", abs(out["single_auc"] - 0.9195802287180326) < 1e-12, mine=out["single_auc"])
# layers in pod json equal claim and equal argmax of the recorded inner scores
claim_layers = {"seed0": [31, 29, 31, 31, 19], "seed1": [29, 31, 31, 28, 29], "seed2": [17, 26, 17, 14, 21],
                "seed3": [30, 27, 26, 17, 21], "seed4": [21, 20, 29, 26, 25], "single": [23, 28, 26, 22, 13]}
lay_ok = all(pj["tags"][t]["layers"] == claim_layers[t] for t in claim_layers)
arg_ok = all(int(np.argmax(pj["tags"][t]["inner"][k])) == pj["tags"][t]["layers"][k] for t in claim_layers for k in range(5))
chk("layers_match_claim_and_inner_argmax", lay_ok and arg_ok)

# ---------- zero-shot answer AUC
zc = {r["clip"]: r for r in zs}
chk("zeroshot_same_clips", set(zc) == set(clips) and len(zs) == 1270)
zlab_mis = sum(int(zc[c]["label"]) != int(r["label"]) for c, r in zip(clips, pc))
zspk_mis = sum(zc[c]["speaker"] != r["speaker"] for c, r in zip(clips, pc))
chk("zeroshot_labels_speakers_align", zlab_mis == 0 and zspk_mis == 0, lab_mis=zlab_mis, spk_mis=zspk_mis)
Z = np.array([float(zc[c]["p_yes"]) for c in clips])
chk("zeroshot_equals_perclip_col", np.max(np.abs(Z - np.array([float(r["p_yes_zeroshot"]) for r in pc]))) == 0.0)
chk("zeroshot_equals_states_p_yes", np.max(np.abs(Z - z["p_yes"])) == 0.0)
chk("zeroshot_file_sha", sha(ZS) == "c5bc6852aad557adfe861b8ad8925ab53a0f94b7dccc2a16a9c92567b8594a14")
ans = auc(y, Z); out["answer"] = ans
chk("answer_4dp_match_claim", round(ans, 4) == CLAIM["answer"] and abs(ans - roc_auc_score(y, Z)) < 1e-12, mine=ans)
gap = mean5 - ans; out["gap"] = gap
chk("gap_4dp_match_claim", round(gap, 4) == CLAIM["gap"], mine=gap)

# ---------- speaker bootstrap, own implementation
uspk = np.unique(spk)                       # sorted string order
rows_of = {s_: np.where(spk == s_)[0] for s_ in uspk}
rng = np.random.default_rng(0)
D = np.empty(2000); M = np.empty(2000); A = np.empty(2000); PR = np.empty((2000, 5)); IDX = np.empty((2000, len(uspk)), int)
for b in range(2000):
    ii = rng.choice(len(uspk), size=len(uspk), replace=True); IDX[b] = ii
    rows = np.concatenate([rows_of[uspk[k]] for k in ii])
    yb = y[rows]
    PR[b] = [auc(yb, P[rows, s]) for s in range(5)]
    M[b] = PR[b].mean(); A[b] = auc(yb, Z[rows]); D[b] = M[b] - A[b]
lo, hi = np.percentile(D, [2.5, 97.5])
out.update(lo=float(lo), hi=float(hi), mean5_ci=np.percentile(M, [2.5, 97.5]).tolist(), ans_ci=np.percentile(A, [2.5, 97.5]).tolist())
chk("ci_4dp_match_claim", round(lo, 4) == CLAIM["lo"] and round(hi, 4) == CLAIM["hi"], lo=float(lo), hi=float(hi))
chk("ci_above_zero", lo > 0)
# rng.integers path gives the same indices (robustness of the stream choice)
rng2 = np.random.default_rng(0)
same_int = all(np.array_equal(rng2.integers(0, len(uspk), len(uspk)), IDX[b]) for b in range(2000))
out["integers_stream_identical"] = bool(same_int)
# compare with the saved draws
d = np.load(DRAWS, allow_pickle=True)
chk("saved_draw_speakers_order", d["speakers"].tolist() == uspk.tolist())
chk("saved_draw_indices_equal_mine", np.array_equal(d["speaker_idx"], IDX))
dd = {k: float(np.max(np.abs(d[k] - v))) for k, v in (("diff", D), ("mean5", M), ("zeroshot", A), ("per_repeat", PR))}
chk("saved_draws_equal_mine", max(dd.values()) < 1e-12, max_abs=dd)
chk("saved_points_equal_mine", abs(float(d["point_diff"]) - gap) < 1e-12 and abs(float(d["point_mean5"]) - mean5) < 1e-12
    and abs(float(d["point_zeroshot"]) - ans) < 1e-12)
chk("draws_file_sha_in_sidecar", sha(DRAWS) == "3d4a42271896bbc726865cb7105219672402ff87ecc801b58791c7e5b6eb47f5")
chk("perclip_file_sha_in_sidecar", sha(PERCLIP) == "9ba110a6ed04cc78ef6781957d2b5ba2e85664a9dc6c6fd227cda066f84a8ff7")
chk("pod_oof_sha_in_sidecar", sha(PODOOF) == "eef487681a232056ee40b6b4b9d2a0f6c896cef5f6a41b83810c6ba4936272d7")

# sensitivity: integer speaker order would change draws; report only
ui = np.unique(spk.astype(int)).astype(str)
rng3 = np.random.default_rng(0); D3 = np.empty(2000)
for b in range(2000):
    ii = rng3.choice(len(ui), size=len(ui), replace=True)
    rows = np.concatenate([rows_of[ui[k]] for k in ii]); yb = y[rows]
    D3[b] = np.mean([auc(yb, P[rows, s]) for s in range(5)]) - auc(yb, Z[rows])
out["sensitivity_int_speaker_order_ci"] = np.percentile(D3, [2.5, 97.5]).tolist()

out["all_pass"] = all(v["ok"] for v in out["checks"].values())
out["n_checks"] = len(out["checks"]); out["failed"] = [k for k, v in out["checks"].items() if not v["ok"]]
import platform, sklearn
out["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print("SUMMARY", json.dumps({k: out[k] for k in ("per_repeat", "mean5", "answer", "gap", "lo", "hi", "all_pass", "n_checks", "failed", "sensitivity_int_speaker_order_ci", "integers_stream_identical")}, default=float))
