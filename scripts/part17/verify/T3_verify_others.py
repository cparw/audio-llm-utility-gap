"""PART 17 T3 verifier, second script: (a) the main job's 50 participant-grouped partitions, replicated
from the scheme it states (GroupKFold(5, shuffle=True, random_state=0..49)); (b) Qwen2.5-Omni Part 10 values
recomputed from o25_pitt_states.npz + omni25_readout.pt; (c) Qwen3-Omni values recomputed from the per-clip
q3o_perp_cols.npz; (d) comparison of the main job's per-clip csv and summary csv with my T3_verify_scores.npz.
Own AUC (pairs, ties 0.5, trapezoid cross-check) and bootstrap. Writes only under part17/verify/.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import csv, json, warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
OUT = "<local data dir>/release/edaic_rerun/part17/verify"
MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def auc_pairs(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]
    d = pos[:, None] - neg[None, :]
    return (np.count_nonzero(d > 0) + 0.5 * np.count_nonzero(d == 0)) / (len(pos) * len(neg))


def auc_trap(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=np.float64)
    o = np.argsort(-s, kind="mergesort"); ss, yy = s[o], y[o]
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
    tp = np.cumsum(yy)[last].astype(float); fp = (last + 1) - tp
    tpr = np.r_[0.0, tp / tp[-1]]; fpr = np.r_[0.0, fp / fp[-1]]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def auc(y, s):
    a, b = auc_pairs(y, s), auc_trap(y, s)
    assert abs(a - b) < 1e-12
    return a


def boot(y, spk, idx, scores, paired_arm=None):
    u = np.unique(spk[idx]); rows = {s: idx[spk[idx] == s] for s in u}
    rng = np.random.default_rng(0); out = {k: [] for k in scores}
    for _ in range(2000):
        r = np.concatenate([rows[s] for s in rng.choice(u, size=len(u), replace=True)])
        if paired_arm is None:
            if y[r].min() == y[r].max():
                continue
            for k, v in scores.items():
                out[k].append(auc_pairs(y[r], v[r]))
        else:
            rc, ra = r[paired_arm[r]], r[~paired_arm[r]]
            for k, v in scores.items():
                out[k].append(auc_pairs(y[rc], v[rc]) - auc_pairs(y[ra], v[ra]))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)) for k, v in out.items()}


def fit(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    return sc, LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


man = list(csv.DictReader(open(MAN)))
arm_of = {os.path.basename(r["segment_path"]): r["set"] for r in man}
R = {}

# ---------------------------------------------------------------- (a) partitions, main job's stated scheme
say("(a) 50 participant-grouped partitions, GroupKFold(5, shuffle=True, random_state=s), s=0..49")
S = np.load(f"{OUT}/T3_verify_scores.npz", allow_pickle=True)
z = np.load("<local data dir>/paper work/paper1_local_runs/probe2/pitt_states.npz", allow_pickle=True)
X = z["ans"][:, -1, :].astype(np.float64); y = z["label"].astype(int); spk = z["spk"].astype(str)
names = z["name"].astype(str); isC = np.array([arm_of[n] == "conflict" for n in names])
num = np.array(["".join(ch for ch in s if ch.isdigit()) for s in spk])
d = S["d"]


def part(s):
    oof = np.zeros(len(y)); pz = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=s).split(X, y, groups=num):
        sc, lr = fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
        wf = lr.coef_.ravel() / sc.scale_
        b = X[te] @ (wf - (wf @ d) / (d @ d) * d)
        pz[te] = (b - b.mean()) / b.std()
    return auc(y, oof), auc(y[isC], pz[isC]), auc(y[isC], pz[isC]) - auc(y[~isC], pz[~isC])


P = np.array(Parallel(n_jobs=14)(delayed(part)(s) for s in range(50)))
for j, nm in enumerate(["all_probe", "conflict_perp", "paired_perp"]):
    v = P[:, j]
    R[f"partitions_{nm}"] = [float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    say(f"  {nm:14s} median {np.median(v):.4f} [{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]")

# ---------------------------------------------------------------- (b) Qwen2.5-Omni Part 10
say("\n(b) Qwen2.5-Omni-7B, recomputed from overnight2/part10/o25_pitt_states.npz + omni25_readout.pt")
import torch
P10 = "<local data dir>/release/overnight2/part10"
oz = np.load(f"{P10}/o25_pitt_states.npz", allow_pickle=True)
ro = torch.load(f"{P10}/omni25_readout.pt", map_location="cpu")
Wy = ro["lm_head_yes"].to(torch.float64).numpy(); Wn = ro["lm_head_no"].to(torch.float64).numpy()
Xo = oz["ans"][:, -1, :].astype(np.float64); yo = oz["label"].astype(int); so = oz["spk"].astype(str)
no = oz["name"].astype(str)
assert all(n in arm_of for n in no)
isCo = np.array([arm_of[n] == "conflict" for n in no])
L = Xo @ np.vstack([Wy, Wn]).T; L -= L.max(1, keepdims=True); Pq = np.exp(L); Pq /= Pq.sum(1, keepdims=True)
mp = Pq.mean(0); wy = mp[:5] / mp[:5].sum(); wn = mp[5:] / mp[5:].sum()
do = wy @ Wy - wn @ Wn
say(f"  n={len(yo)} n_spk={len(np.unique(so))} conflict={isCo.sum()} ({len(np.unique(so[isCo]))} spk) "
    f"agreement={(~isCo).sum()} ({len(np.unique(so[~isCo]))} spk)")
say(f"  p_yes check: max|mine - saved p_yes| {np.max(np.abs(Pq[:, :5].sum(1) - oz['p_yes'])):.4f}")
oof = np.zeros(len(yo)); pz = np.zeros(len(yo)); kz = np.zeros(len(yo)); Wf = []
for tr, te in GroupKFold(n_splits=5).split(Xo, yo, groups=so):
    sc, lr = fit(Xo[tr], yo[tr])
    oof[te] = lr.predict_proba(sc.transform(Xo[te]))[:, 1]
    wf = lr.coef_.ravel() / sc.scale_; Wf.append(wf)
    a = Xo[te] @ wf; b = Xo[te] @ (wf - (wf @ do) / (do @ do) * do)
    pz[te] = (b - b.mean()) / b.std(); kz[te] = (a - a.mean()) / a.std()
wbar = np.mean(Wf, axis=0)
G = np.random.default_rng(0).standard_normal((1000, len(do)))
fl = np.mean(np.abs(G @ do) / (np.linalg.norm(G, axis=1) * np.linalg.norm(do)))
cells = {"all": np.arange(len(yo)), "conflict": np.nonzero(isCo)[0], "agreement": np.nonzero(~isCo)[0]}
sc_o = {"probe": oof, "along": Xo @ do, "perp": pz, "keep": kz}
R["omni_cos"] = cos(wbar, do); R["omni_floor_1000_vs_d"] = float(fl)
say(f"  cos {cos(wbar, do):.4f}  floor vs d (1000 dirs, rng(0), {len(do)} dims) {fl:.4f}")
for cn, ix in cells.items():
    bb = boot(yo, so, ix, sc_o)
    for k, v in sc_o.items():
        R[f"omni_{cn}_{k}"] = [auc(yo[ix], v[ix]), *bb[k]]
        say(f"  {cn:9s} {k:6s} {auc(yo[ix], v[ix]):.4f} [{bb[k][0]:.4f}, {bb[k][1]:.4f}] draws {bb[k][2]}")
bp = boot(yo, so, np.arange(len(yo)), sc_o, paired_arm=isCo)
for k, v in sc_o.items():
    val = auc(yo[isCo], v[isCo]) - auc(yo[~isCo], v[~isCo])
    R[f"omni_paired_{k}"] = [val, *bp[k]]
    say(f"  paired    {k:6s} {val:.4f} [{bp[k][0]:.4f}, {bp[k][1]:.4f}]")

# ---------------------------------------------------------------- (c) Qwen3-Omni per-clip file
say("\n(c) Qwen3-Omni-30B, AUCs recomputed from part16/pod_sync/q3o_perp_cols.npz (per clip)")
qz = np.load("<local data dir>/release/edaic_rerun/part16/pod_sync/q3o_perp_cols.npz", allow_pickle=True)
yq = qz["label"].astype(int); sq = qz["spk"].astype(str); nq = qz["name"].astype(str)
isCq = np.array([arm_of[n] == "conflict" for n in nq])
say(f"  n={len(yq)} n_spk={len(np.unique(sq))} conflict={isCq.sum()} ({len(np.unique(sq[isCq]))} spk) "
    f"agreement={(~isCq).sum()} ({len(np.unique(sq[~isCq]))} spk); names equal Q2A order: {np.array_equal(nq, names)}")
numq = np.array(["".join(ch for ch in s if ch.isdigit()) for s in sq])
say(f"  q3o speaker labels for participant 172: {sorted(set(sq[numq.astype(int) == 172]))}")
cq = {"all": np.arange(len(yq)), "conflict": np.nonzero(isCq)[0], "agreement": np.nonzero(~isCq)[0]}
scq = {"probe": qz["oof"], "along": qz["proj_d"], "perp": qz["perp_z"], "perp_trainz": qz["perp_trainz"], "perp_raw": qz["perp_raw"]}
for cn, ix in cq.items():
    bb = boot(yq, sq, ix, scq)
    for k, v in scq.items():
        R[f"q3o_{cn}_{k}"] = [auc(yq[ix], v[ix]), *bb[k]]
        say(f"  {cn:9s} {k:11s} {auc(yq[ix], v[ix]):.4f} [{bb[k][0]:.4f}, {bb[k][1]:.4f}] draws {bb[k][2]}")

# ---------------------------------------------------------------- (d) main job's per-clip csv and summary csv
say("\n(d) main job per-clip csv and summary csv vs my scores")
AG = "<local data dir>/release/edaic_rerun/part17"
rows = list(csv.DictReader(open(f"{AG}/T3_q2a_direction.csv")))
say(f"  T3_q2a_direction.csv rows {len(rows)}; unique spk {len(set(r['spk'] for r in rows))}; "
    f"conflict {sum(r['arm'] == 'conflict' for r in rows)}; names match npz order {[r['name'] for r in rows] == list(names)}")
for col, mine in [("oof_probe", S["oof"]), ("proj_d", S["proj"]), ("perp_z", S["perp_zheld"]), ("keep_z", S["keep_zheld"]),
                  ("nested_oof", S["nested_oof"]), ("nested_perp_z", S["nested_perp"])]:
    theirs = np.array([float(r[col]) for r in rows])
    say(f"  {col:14s} max|main - mine| = {np.max(np.abs(theirs - mine)):.2e}")
say(f"  fold column identical: {np.array_equal(np.array([int(r['fold']) for r in rows]), S['fold'])}; "
    f"nested_layer identical: {np.array_equal(np.array([int(r['nested_layer']) for r in rows]), S['nested_layer'])}")
json.dump(R, open(f"{OUT}/T3_verify_others.json", "w"), indent=1, default=float)
open(f"{OUT}/T3_verify_others.log", "w").write("\n".join(LOG) + "\n")
