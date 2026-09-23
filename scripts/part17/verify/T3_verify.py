"""PART 17 T3 INDEPENDENT VERIFIER. Qwen2-Audio readout-direction test on Pitt.

Written from scratch. Reads ONLY the original sources:
  pitt_states.npz, pitt_zeroshot_scores.csv, pitt_nested.json, pitt_ans_nested_oof.csv (probe2),
  pitt_conflict_manifest.csv (arms), the Qwen2-Audio tokenizer.json / vocab.json and the lm_head shard.
The main job's outputs (T3_q2a_direction*.csv/json) are opened ONLY at the end, to compare.

AUC: explicit Mann-Whitney over all positive x negative pairs, ties 0.5, cross-checked with a trapezoid
over the full empirical ROC. Intervals: 2000 draws, fresh numpy.random.default_rng(0) per cell, speakers
resampled with replacement, percentiles 2.5/97.5; paired = one speaker draw per replicate.
Writes only under part17/verify/. usage: /usr/local/bin/python3 T3_verify.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import csv, json, struct, hashlib, time, sys, warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
T0 = time.time()
OUTDIR = "<local data dir>/release/edaic_rerun/part17/verify"
P2 = "<local data dir>/paper work/paper1_local_runs/probe2"
STATES = f"{P2}/pitt_states.npz"
ZS = f"{P2}/pitt_zeroshot_scores.csv"
NESTJ = f"{P2}/pitt_nested.json"
NESTOOF = f"{P2}/pitt_ans_nested_oof.csv"
MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
SNAP = ("<hf cache>/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/"
        "0a095220c30b7b31434169c3086508ef3ea5bf0a")
SHARD = f"{SNAP}/model-00005-of-00005.safetensors"
TOKJ = f"{SNAP}/tokenizer.json"
VOCABJ = f"{SNAP}/vocab.json"
NDRAW = 2000
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def sha1mb(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read(1 << 20)).hexdigest()


# ------------------------------------------------------------------ AUC, two independent ways
def auc_pairs(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    d = pos[:, None] - neg[None, :]
    return (np.count_nonzero(d > 0) + 0.5 * np.count_nonzero(d == 0)) / (len(pos) * len(neg))


def auc_trap(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    o = np.argsort(-s, kind="mergesort")
    ss, yy = s[o], y[o]
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]   # last index of each distinct threshold
    tp = np.cumsum(yy)[last].astype(float)
    fp = (last + 1) - tp
    tpr = np.r_[0.0, tp / tp[-1]]
    fpr = np.r_[0.0, fp / fp[-1]]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def auc(y, s):
    a, b = auc_pairs(y, s), auc_trap(y, s)
    assert abs(a - b) < 1e-12, (a, b)
    return a


def f4(v):
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


# ------------------------------------------------------------------ bootstrap
def spk_rows(spk_arr, idx):
    u = np.unique(spk_arr[idx])
    return u, {s: idx[spk_arr[idx] == s] for s in u}


def boot_cell(y, spk_arr, idx, scores):
    """fresh default_rng(0); speakers of this cell resampled with replacement; same draw for every score."""
    u, rows_of = spk_rows(spk_arr, idx)
    rng = np.random.default_rng(0)
    out = {k: [] for k in scores}
    for _ in range(NDRAW):
        dr = rng.choice(u, size=len(u), replace=True)
        r = np.concatenate([rows_of[s] for s in dr])
        yb = y[r]
        if yb.min() == yb.max():
            continue
        for k, sc in scores.items():
            out[k].append(auc_pairs(yb, sc[r]))
    res = {}
    for k, v in out.items():
        v = np.asarray(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        res[k] = (float(lo), float(hi), int(len(v)))
    return res


def boot_paired(y, spk_arr, arm_c, scores):
    u, rows_of = spk_rows(spk_arr, np.arange(len(y)))
    rng = np.random.default_rng(0)
    out = {k: [] for k in scores}
    for _ in range(NDRAW):
        dr = rng.choice(u, size=len(u), replace=True)
        r = np.concatenate([rows_of[s] for s in dr])
        rc, ra = r[arm_c[r]], r[~arm_c[r]]
        yc, ya = y[rc], y[ra]
        if len(rc) == 0 or len(ra) == 0 or yc.min() == yc.max() or ya.min() == ya.max():
            continue
        for k, sc in scores.items():
            out[k].append(auc_pairs(yc, sc[rc]) - auc_pairs(ya, sc[ra]))
    res = {}
    for k, v in out.items():
        v = np.asarray(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        res[k] = (float(lo), float(hi), int(len(v)))
    return res


# ------------------------------------------------------------------ probe
def fit(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return sc, lr


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def floor(vec, n=NDRAW):
    G = np.random.default_rng(0).standard_normal((n, len(vec)))
    c = np.abs(G @ vec) / (np.linalg.norm(G, axis=1) * np.linalg.norm(vec))
    lo, hi = np.percentile(c, [2.5, 97.5])
    return float(c.mean()), float(lo), float(hi)


def run_probe(X, y, groups, d, n_splits=5):
    """GroupKFold probe, returns oof prob, fold id, per-fold raw directions, and every projection variant."""
    n = len(y)
    oof = np.zeros(n)
    fold = np.full(n, -1)
    V = {k: np.zeros(n) for k in ("perp_zheld", "keep_zheld", "perp_ztrain", "keep_ztrain",
                                  "perp_raw", "keep_raw", "perp_scaled_icpt", "keep_scaled_icpt",
                                  "perp_zheld_ddof1")}
    W = []
    for f, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups=groups)):
        sc, lr = fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
        fold[te] = f
        wf = lr.coef_.ravel() / sc.scale_
        W.append(wf)
        wp = wf - (wf @ d) / (d @ d) * d
        a_te, b_te = X[te] @ wf, X[te] @ wp
        a_tr, b_tr = X[tr] @ wf, X[tr] @ wp
        V["keep_raw"][te], V["perp_raw"][te] = a_te, b_te
        V["keep_zheld"][te] = (a_te - a_te.mean()) / a_te.std()
        V["perp_zheld"][te] = (b_te - b_te.mean()) / b_te.std()
        V["perp_zheld_ddof1"][te] = (b_te - b_te.mean()) / b_te.std(ddof=1)
        V["keep_ztrain"][te] = (a_te - a_tr.mean()) / a_tr.std()
        V["perp_ztrain"][te] = (b_te - b_tr.mean()) / b_tr.std()
        # scaler-space variant: remove d in standardised coordinates, keep intercept and scaler mean
        c = lr.coef_.ravel()
        ds = d * sc.scale_
        cp = c - (c @ ds) / (ds @ ds) * ds
        Z = sc.transform(X[te])
        V["keep_scaled_icpt"][te] = Z @ c + lr.intercept_[0]
        V["perp_scaled_icpt"][te] = Z @ cp + lr.intercept_[0]
    wbar = np.mean(np.stack(W), axis=0)
    return oof, fold, W, wbar, V


# ================================================================== 0. load
say("=" * 100)
say("PART 17 T3 INDEPENDENT VERIFIER  (Qwen2-Audio, Pitt, readout direction)")
say("=" * 100)
src = {"states": STATES, "zeroshot": ZS, "nested_json": NESTJ, "nested_oof": NESTOOF, "manifest": MAN,
       "shard": SHARD, "tokenizer": TOKJ, "vocab": VOCABJ}
SHA = {}
for k, p in src.items():
    SHA[k] = sha1mb(p)
    say(f"[src] {k:12s} {os.path.getsize(p):>14,d} B sha256(1MB)={SHA[k][:16]}  {p}")
PART16_STATES_SHA = "5d8272dc673cdbded7c0a84207d5306ffdd30b6d1fee6a99b549b05d7087494c"
say(f"[src] states sha256(1MB) equals Part16 sidecar value: {SHA['states'] == PART16_STATES_SHA}")
say(f"[src] old path exists? <local data dir>/paper1_local_runs/probe2/pitt_states.npz -> "
    f"{os.path.exists('<local data dir>/paper1_local_runs/probe2/pitt_states.npz')}")

z = np.load(STATES, allow_pickle=True)
ANS = z["ans"]
y = z["label"].astype(int)
spk = z["spk"].astype(str)
names = z["name"].astype(str)
p_yes_npz = z["p_yes"].astype(np.float64)
N = len(y)
say(f"[states] ans {ANS.shape} {ANS.dtype}; n={N} n_speakers={len(np.unique(spk))} pos={y.sum()} neg={N - y.sum()}")

# arms from the manifest (join on segment basename)
man = list(csv.DictReader(open(MAN)))
arm_of = {os.path.basename(r["segment_path"]): r["set"] for r in man}
lab_of = {os.path.basename(r["segment_path"]): int(r["label"]) for r in man}
assert all(nm in arm_of for nm in names), "npz clip missing from manifest"
arm = np.array([arm_of[nm] for nm in names])
assert all(lab_of[nm] == int(l) for nm, l in zip(names, y)), "label mismatch manifest vs npz"
assert all(nm.split("_")[0] == a for nm, a in zip(names, arm)), "name prefix vs manifest set mismatch"
isC = arm == "conflict"
idx_all, idx_c, idx_a = np.arange(N), np.nonzero(isC)[0], np.nonzero(~isC)[0]
say(f"[arms] conflict n={len(idx_c)} spk={len(np.unique(spk[idx_c]))}; agreement n={len(idx_a)} "
    f"spk={len(np.unique(spk[idx_a]))}  (manifest join, labels agree, name prefix agrees)")
num = np.array(["".join(ch for ch in s if ch.isdigit()) for s in spk])
dual = sorted({n_ for n_ in num if len(set(spk[num == n_])) > 1})
say(f"[172] numeric participant ids under >1 npz speaker label: {dual}; real participants={len(np.unique(num))}; "
    f"clips of those: {int(np.isin(num, dual).sum())}")

zs = list(csv.DictReader(open(ZS)))
assert [r["clip"] for r in zs] == list(names), "zeroshot csv order differs from npz"
p_yes_csv = np.array([float(r["p_yes"]) for r in zs])
say(f"[zeroshot] csv rows={len(zs)}; AUC(p_yes csv)={auc(y, p_yes_csv):.6f} "
    f"(trapezoid {auc_trap(y, p_yes_csv):.6f}); max|csv-npz p_yes|={np.max(np.abs(p_yes_csv - p_yes_npz)):.2e}")

# ================================================================== 1. tokens and lm_head
from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOKJ)
vocab = json.load(open(VOCABJ))
def single_ids(forms):
    out = []
    for f in forms:
        ids = tok.encode(f, add_special_tokens=False).ids
        if len(ids) == 1:
            out.append(ids[0])
    return sorted(set(out))
YES = single_ids(["Yes", " Yes", "yes", " yes", "YES", " YES"])
NO = single_ids(["No", " No", "no", " no", "NO", " NO"])
# cross-check with vocab.json byte-level entries
for w in ["Yes", "ĠYes", "yes", "Ġyes", "YES", "ĠYES", "No", "ĠNo", "no", "Ġno", "NO", "ĠNO"]:
    assert w in vocab, w
assert sorted(vocab[w] for w in ["Yes", "ĠYes", "yes", "Ġyes", "YES", "ĠYES"]) == YES
assert sorted(vocab[w] for w in ["No", "ĠNo", "no", "Ġno", "NO", "ĠNO"]) == NO
say(f"[tokens] YES ids {YES}  NO ids {NO}  (tokenizers.encode single-token forms == vocab.json lookup)")
# extract2.py's own sets (first token of each form) for the p_yes post-norm check
def first_ids(forms):
    return sorted({tok.encode(f, add_special_tokens=False).ids[0] for f in forms})
YES_X2 = first_ids(["Yes", " Yes", "yes", " yes", "YES"])
NO_X2 = first_ids(["No", " No", "no", " no", "NO"])
say(f"[tokens] extract2.py sets: yes {YES_X2} no {NO_X2}")

with open(SHARD, "rb") as fh:
    hlen = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(hlen))
meta = hdr["language_model.lm_head.weight"]
assert meta["dtype"] == "BF16"
V_, H_ = meta["shape"]
raw = np.memmap(SHARD, dtype=np.uint16, mode="r", offset=8 + hlen + meta["data_offsets"][0], shape=(V_, H_))
def bf16_rows(r):
    return (np.asarray(raw[r]).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
say(f"[lm_head] shard header {list(k for k in hdr if k != '__metadata__')} shape {V_}x{H_} BF16, decoded by hand")

X = ANS[:, -1, :].astype(np.float64)   # final LM stage at the answer position
ALLID = sorted(set(YES + NO + YES_X2 + NO_X2))
Wsel = bf16_rows(ALLID)
col = {t: i for i, t in enumerate(ALLID)}
L_sel = X @ Wsel.T                                  # logits of the needed ids
# full-vocabulary logsumexp in chunks
lse = np.full(N, -np.inf)
Xf = X.astype(np.float32)
for s0 in range(0, V_, 16384):
    Wc = (np.asarray(raw[s0:s0 + 16384]).astype(np.uint32) << 16).view(np.float32)
    Lc = (Xf @ Wc.T).astype(np.float64)
    m = np.maximum(lse, Lc.max(axis=1))
    lse = m + np.log(np.exp(lse - m) + np.exp(Lc - m[:, None]).sum(axis=1))
P_full = np.exp(L_sel - lse[:, None])               # full-vocab probability of each needed id
say(f"[softmax] full vocab: mean Yes mass (6 ids) {P_full[:, [col[t] for t in YES]].sum(1).mean():.6f}, "
    f"mean No mass {P_full[:, [col[t] for t in NO]].sum(1).mean():.6f}")
# post-norm check: reproduce extract2's p_yes from the saved final state
py = P_full[:, [col[t] for t in YES_X2]].sum(1)
pn = P_full[:, [col[t] for t in NO_X2]].sum(1)
p_yes_mine = py / (py + pn + 1e-12)
say(f"[post-norm] p_yes from X@W vs saved p_yes: max|diff|={np.max(np.abs(p_yes_mine - p_yes_npz)):.4f} "
    f"corr={np.corrcoef(p_yes_mine, p_yes_npz)[0, 1]:.6f}; AUC mine {auc(y, p_yes_mine):.4f} vs saved {auc(y, p_yes_npz):.4f}")

Wy, Wn = bf16_rows(YES), bf16_rows(NO)
def direction(idx, mode):
    if mode == "full":
        py_ = P_full[np.ix_(idx, [col[t] for t in YES])]
        pn_ = P_full[np.ix_(idx, [col[t] for t in NO])]
    else:   # 12-way softmax over the Yes/No logits only
        Lyn = L_sel[np.ix_(idx, [col[t] for t in YES + NO])]
        Lyn = Lyn - Lyn.max(1, keepdims=True)
        E = np.exp(Lyn)
        Pq = E / E.sum(1, keepdims=True)
        py_, pn_ = Pq[:, :len(YES)], Pq[:, len(YES):]
    wy = py_.mean(0); wy = wy / wy.sum()
    wn = pn_.mean(0); wn = wn / wn.sum()
    dd = wy @ Wy - wn @ Wn
    return dd / np.linalg.norm(dd), wy, wn
d_all, wy_all, wn_all = direction(idx_all, "full")
d_all12, wy12, wn12 = direction(idx_all, "12")
d_plain = Wy.mean(0) - Wn.mean(0); d_plain /= np.linalg.norm(d_plain)
say(f"[direction] full-vocab weights yes {dict(zip(YES, np.round(wy_all, 6)))} no {dict(zip(NO, np.round(wn_all, 6)))}")
say(f"[direction] cos(d_fullvocab, d_12way)={cos(d_all, d_all12):.10f}  cos(d, d_plain)={cos(d_all, d_plain):.6f}")
auc_d_all = auc(y, X @ d_all)
say(f"[gate] AUC(X.d)={auc_d_all:.6f} vs zero-shot file {auc(y, p_yes_csv):.6f}, delta {auc_d_all - auc(y, p_yes_csv):+.6f}")

# ================================================================== 2. final-stage probe, 228 speaker groups
say("\n[probe] final stage ans[:,-1,:] float64, GroupKFold(5) by npz speaker, scaler in training fold, LR(2000, balanced)")
oof, fold, Wf, wbar, V = run_probe(X, y, spk, d_all)
fs = [int((fold == f).sum()) for f in range(5)]
say(f"  fold sizes {fs}  speakers/fold {[len(np.unique(spk[fold == f])) for f in range(5)]}")
cos_all = cos(wbar, d_all)
say(f"  cos(mean-of-folds probe dir, d) = {cos_all:.6f}; per fold {[round(cos(w, d_all), 4) for w in Wf]}; "
    f"mean of per-fold cos {np.mean([cos(w, d_all) for w in Wf]):.6f}")
sc_full, lr_full = fit(X, y)
w_fullfit = lr_full.coef_.ravel() / sc_full.scale_
say(f"  cos(single full-data fit dir, d) = {cos(w_fullfit, d_all):.6f}  (alternative definition, not the reported one)")
say(f"  cos with d_12way instead: {cos(wbar, d_all12):.6f}")

proj = X @ d_all
R = {}
def rec(key, val, lo=None, hi=None, nd=None, n=None, ns=None):
    R[key] = {"value": val, "lo": lo, "hi": hi, "draws": nd, "n": n, "n_speakers": ns}

say("\n  standardisation variants of the d-removed projection (point AUC; all / conflict / agreement):")
for k in ["perp_zheld", "perp_ztrain", "perp_raw", "perp_scaled_icpt", "perp_zheld_ddof1",
          "keep_zheld", "keep_ztrain", "keep_raw", "keep_scaled_icpt"]:
    say(f"    {k:18s} {auc(y, V[k]):.4f} / {auc(y[idx_c], V[k][idx_c]):.4f} / {auc(y[idx_a], V[k][idx_a]):.4f}")

# ================================================================== 3. floors
fl_d = floor(d_all)
fl_d12 = floor(d_all12)
fl_w = floor(wbar)
say(f"\n[floor] fresh rng(0), 2000 dirs, 4096 dims: vs d {fl_d[0]:.10f} [{fl_d[1]:.4f}, {fl_d[2]:.4f}]; "
    f"vs d_12way {fl_d12[0]:.10f}; vs probe dir {fl_w[0]:.4f} [{fl_w[1]:.4f}, {fl_w[2]:.4f}]; "
    f"analytic sqrt(2/(pi*4096))={np.sqrt(2 / (np.pi * 4096)):.4f}")
# rng.choice vs rng.integers stream equivalence (documents that the bootstrap draw is scheme-independent)
_u = np.arange(228)
_a = np.random.default_rng(0).choice(_u, size=228, replace=True)
_b = _u[np.random.default_rng(0).integers(0, 228, size=228)]
say(f"[rng] rng.choice(u, replace=True) == u[rng.integers(0,n,n)] for fresh rng(0): {np.array_equal(_a, _b)}")

# ================================================================== 4. point values + intervals, decided cells
cells = {"ALL": idx_all, "CONFLICT": idx_c, "AGREEMENT": idx_a}
SC = {"probe": oof, "along": proj, "perp": V["perp_zheld"], "keep": V["keep_zheld"], "perp_ztrain": V["perp_ztrain"]}
say("\n[decided] full fit (468 OOF) restricted to arm; within-held-out-fold z")
for cn, ix in cells.items():
    b = boot_cell(y, spk, ix, SC)
    ns = len(np.unique(spk[ix]))
    for k in SC:
        val = auc(y[ix], SC[k][ix])
        lo, hi, nd = b[k]
        rec(f"{cn}_{k}", val, lo, hi, nd, len(ix), ns)
        say(f"  {cn:9s} {k:12s} {val:.4f} [{lo:.4f}, {hi:.4f}] draws {nd}  n={len(ix)} n_spk={ns}")
bp = boot_paired(y, spk, isC, SC)
for k in SC:
    val = auc(y[idx_c], SC[k][idx_c]) - auc(y[idx_a], SC[k][idx_a])
    lo, hi, nd = bp[k]
    rec(f"PAIRED_{k}", val, lo, hi, nd, N, len(np.unique(spk)))
    say(f"  PAIRED    {k:12s} {val:.4f} [{lo:.4f}, {hi:.4f}] draws {nd}")
rec("cos_all", cos_all, n=N, ns=228)
rec("floor_vs_d", fl_d[0], fl_d[1], fl_d[2], NDRAW)
rec("floor_vs_probe", fl_w[0], fl_w[1], fl_w[2], NDRAW)

# ================================================================== 5. refit within arm (secondary, NOT USED)
say("\n[refit within arm, NOT USED]")
for cn, ix in (("CONFLICT", idx_c), ("AGREEMENT", idx_a)):
    d_arm, _, _ = direction(ix, "full")
    o2, f2, W2, wb2, V2 = run_probe(X[ix], y[ix], spk[ix], d_arm)
    fla = floor(d_arm)
    c2 = cos(wb2, d_arm)
    yy = y[ix]
    sc2 = {"probe": o2, "along": X[ix] @ d_arm, "perp": V2["perp_zheld"]}
    # bootstrap in the arm's own index space
    b2 = boot_cell(yy, spk[ix], np.arange(len(ix)), sc2)
    rec(f"{cn}_refit_cos", c2)
    rec(f"{cn}_refit_floor_vs_d", fla[0], fla[1], fla[2], NDRAW)
    say(f"  {cn}: cos(d_arm, d_all)={cos(d_arm, d_all):.10f}; cos(probe,d_arm)={c2:.4f}; floor vs d_arm {fla[0]:.10f} "
        f"[{fla[1]:.4f}, {fla[2]:.4f}]")
    for k in sc2:
        val = auc(yy, sc2[k])
        rec(f"{cn}_refit_{k}", val, *b2[k], len(ix), len(np.unique(spk[ix])))
        say(f"    {k:6s} {val:.4f} [{b2[k][0]:.4f}, {b2[k][1]:.4f}] draws {b2[k][2]}")

# ================================================================== 6. nested supplement
say("\n[nested] ans layers, float32 as nested_probe_local.py, outer GroupKFold(5), inner GroupKFold(4) mean inner AUC")
X32 = ANS.astype(np.float32)
def inner_score(tr, l):
    s = 0.0
    Xt, yt, gt = X32[tr][:, l], y[tr], spk[tr]
    for itr, ite in GroupKFold(n_splits=4).split(Xt, yt, groups=gt):
        sc_, lr_ = fit(Xt[itr], yt[itr])
        p = lr_.predict_proba(sc_.transform(Xt[ite]))[:, 1]
        s += auc_pairs(yt[ite], p) if len(set(yt[ite])) > 1 else 0.5
    return s / 4
outer = list(GroupKFold(n_splits=5).split(X32, y, groups=spk))
assert all(np.array_equal(np.sort(te), np.nonzero(fold == f)[0]) for f, (tr, te) in enumerate(outer))
chosen = []
n_oof = np.zeros(N)
n_perp = np.zeros(N)
n_keep = np.zeros(N)
n_layer = np.zeros(N, dtype=int)
Wn_ = []
for f, (tr, te) in enumerate(outer):
    sc_in = Parallel(n_jobs=14)(delayed(inner_score)(tr, l) for l in range(ANS.shape[1]))
    best = int(np.argmax(sc_in))
    chosen.append(best)
    sc_, lr_ = fit(X32[tr][:, best], y[tr])
    n_oof[te] = lr_.predict_proba(sc_.transform(X32[te][:, best]))[:, 1]
    wf = (lr_.coef_.ravel() / sc_.scale_).astype(np.float64)
    Wn_.append(wf)
    wp = wf - (wf @ d_all) / (d_all @ d_all) * d_all
    Xl = ANS[te, best, :].astype(np.float64)
    a_, b_ = Xl @ wf, Xl @ wp
    n_keep[te] = (a_ - a_.mean()) / a_.std()
    n_perp[te] = (b_ - b_.mean()) / b_.std()
    n_layer[te] = best
    say(f"  outer fold {f}: chosen layer {best} (inner mean AUC {sc_in[best]:.4f}); cos(fold dir, d)={cos(wf, d_all):.4f}")
nj = json.load(open(NESTJ))
ref = list(csv.DictReader(open(NESTOOF)))
assert [r["clip"] for r in ref] == list(names)
ref_p = np.array([float(r["p_probe"]) for r in ref])
say(f"  chosen {chosen} vs pitt_nested.json {nj['ans']['chosen_layers']}: {chosen == nj['ans']['chosen_layers']}; "
    f"max|oof - pitt_ans_nested_oof.csv|={np.max(np.abs(n_oof - ref_p)):.2e}; "
    f"AUC nested oof {auc(y, n_oof):.6f} vs json {nj['ans']['auc_nested_oof']:.6f}")
wbar_n = np.mean(np.stack(Wn_), axis=0)
cos_n = cos(wbar_n, d_all)
cos_n_meanfold = float(np.mean([cos(w, d_all) for w in Wn_]))
unit_mean = np.mean(np.stack([w / np.linalg.norm(w) for w in Wn_]), axis=0)
fl_n = floor(wbar_n)
say(f"  cos(mean of fold dirs, d)={cos_n:.4f}; mean of per-fold cos={cos_n_meanfold:.4f}; "
    f"cos(mean of unit fold dirs, d)={cos(unit_mean, d_all):.4f}; floor vs nested dir {fl_n[0]:.4f} [{fl_n[1]:.4f}, {fl_n[2]:.4f}]")
rec("NESTED_cos", cos_n)
rec("NESTED_cos_meanfold", cos_n_meanfold)
rec("NESTED_floor", fl_n[0], fl_n[1], fl_n[2], NDRAW)
SCN = {"probe": n_oof, "perp": n_perp, "keep": n_keep}
for cn, ix in cells.items():
    b = boot_cell(y, spk, ix, SCN)
    for k in SCN:
        val = auc(y[ix], SCN[k][ix])
        rec(f"NESTED_{cn}_{k}", val, *b[k], len(ix), len(np.unique(spk[ix])))
        say(f"  NESTED {cn:9s} {k:6s} {val:.4f} [{b[k][0]:.4f}, {b[k][1]:.4f}] draws {b[k][2]}")
bpn = boot_paired(y, spk, isC, SCN)
for k in SCN:
    val = auc(y[idx_c], SCN[k][idx_c]) - auc(y[idx_a], SCN[k][idx_a])
    rec(f"NESTED_PAIRED_{k}", val, *bpn[k], N, 228)
    say(f"  NESTED PAIRED    {k:6s} {val:.4f} [{bpn[k][0]:.4f}, {bpn[k][1]:.4f}] draws {bpn[k][2]}")

# ================================================================== 7. participant 172 regrouped (227 groups)
say("\n[172] regroup by real participant number (227 groups), same d, final stage")
o3, f3, W3, wb3, V3 = run_probe(X, y, num, d_all)
rg = {"probe_all": auc(y, o3), "perp_all": auc(y, V3["perp_zheld"]),
      "perp_conflict": auc(y[idx_c], V3["perp_zheld"][idx_c]),
      "perp_agreement": auc(y[idx_a], V3["perp_zheld"][idx_a]),
      "paired_perp": auc(y[idx_c], V3["perp_zheld"][idx_c]) - auc(y[idx_a], V3["perp_zheld"][idx_a]),
      "cos": cos(wb3, d_all)}
for k, v in rg.items():
    rec(f"REGROUP_{k}", v)
say("  " + "  ".join(f"{k} {v:.4f}" for k, v in rg.items()))
# my own 50 random participant-grouped partitions (scheme: shuffle participant ids with rng(s), assign round robin 5 folds)
say("  50 random 5-fold partitions grouped by real participant (my scheme: rng(s).permutation of ids, round robin)")
uid = np.unique(num)
def one_partition(s):
    perm = np.random.default_rng(s).permutation(uid)
    fmap = {p: i % 5 for i, p in enumerate(perm)}
    fo = np.array([fmap[p] for p in num])
    pz = np.zeros(N)
    for f in range(5):
        te, tr = np.nonzero(fo == f)[0], np.nonzero(fo != f)[0]
        sc_, lr_ = fit(X[tr], y[tr])
        wf = lr_.coef_.ravel() / sc_.scale_
        wp = wf - (wf @ d_all) / (d_all @ d_all) * d_all
        b_ = X[te] @ wp
        pz[te] = (b_ - b_.mean()) / b_.std()
    return auc(y[idx_c], pz[idx_c]), auc(y[idx_c], pz[idx_c]) - auc(y[idx_a], pz[idx_a])
parts = Parallel(n_jobs=14)(delayed(one_partition)(s) for s in range(50))
pc = np.array([p[0] for p in parts]); pp = np.array([p[1] for p in parts])
say(f"  conflict perp: median {np.median(pc):.4f} [{np.percentile(pc, 2.5):.4f}, {np.percentile(pc, 97.5):.4f}]; "
    f"paired: median {np.median(pp):.4f} [{np.percentile(pp, 2.5):.4f}, {np.percentile(pp, 97.5):.4f}]")
R["PARTITIONS_mine"] = {"conflict_perp_median": float(np.median(pc)), "conflict_perp_lo": float(np.percentile(pc, 2.5)),
                        "conflict_perp_hi": float(np.percentile(pc, 97.5)), "paired_median": float(np.median(pp)),
                        "paired_lo": float(np.percentile(pp, 2.5)), "paired_hi": float(np.percentile(pp, 97.5))}

# ================================================================== 8. cos bootstrap with probe refit per draw (extra)
say("\n[cos bootstrap, refit per draw] fresh rng(0) speaker draws, full-data fit per draw, cos(coef/scale, d)")
u_all, rows_all = spk_rows(spk, idx_all)
rng = np.random.default_rng(0)
draw_rows = []
for _ in range(NDRAW):
    dr = rng.choice(u_all, size=len(u_all), replace=True)
    draw_rows.append(np.concatenate([rows_all[s] for s in dr]))
def cos_draw(r):
    if y[r].min() == y[r].max():
        return np.nan
    sc_, lr_ = fit(X[r], y[r])
    return cos(lr_.coef_.ravel() / sc_.scale_, d_all)
cb = np.array(Parallel(n_jobs=14, batch_size=8)(delayed(cos_draw)(r) for r in draw_rows))
cb = cb[~np.isnan(cb)]
rec("cos_boot_refit", float(cb.mean()), float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5)), len(cb))
say(f"  mean {cb.mean():.4f} [{np.percentile(cb, 2.5):.4f}, {np.percentile(cb, 97.5):.4f}] usable {len(cb)}")

# ================================================================== save
np.savez(f"{OUTDIR}/T3_verify_scores.npz", names=names, spk=spk, y=y, arm=arm, oof=oof, proj=proj,
         perp_zheld=V["perp_zheld"], perp_ztrain=V["perp_ztrain"], keep_zheld=V["keep_zheld"], fold=fold,
         nested_oof=n_oof, nested_perp=n_perp, nested_layer=n_layer, d=d_all)
json.dump({"results": R, "sha256_first_1MB": SHA, "sources": src, "yes_ids": YES, "no_ids": NO,
           "chosen_nested": chosen, "runtime_s": time.time() - T0,
           "command": "/usr/local/bin/python3 " + os.path.abspath(__file__)},
          open(f"{OUTDIR}/T3_verify.json", "w"), indent=1, default=float)
open(f"{OUTDIR}/T3_verify.log", "w").write("\n".join(LOG) + "\n")
say(f"\nruntime {time.time() - T0:.0f}s")
