"""PART 17 task 3: Qwen2-Audio readout-direction test on Pitt, corrected method, full report.

Corrected method (canonical, release/scripts/part10_readout_direction_v2.py lines 261-274):
  per training fold, w_fold = coef_ / scaler.scale_ (raw state space),
  w_perp = w_fold - (w_fold.d / d.d) d, project the HELD-OUT speakers onto w_perp,
  standardise WITHIN the held-out fold: perp_z[te] = (b - b.mean()) / b.std(), then pool.
Random floor: numpy.random.default_rng(0) created fresh per cell, 2000 Gaussian directions
  in the 4096-dim space, |cos| with the reference vector (both the readout direction d and the
  probe direction are reported, see sidecar).
Mass-weighted readout direction: as in part16/M5 first pass and p15/direction_q3o.py:
  full-vocabulary softmax of X @ lm_head.T per clip, mean mass of each Yes id and each No id
  over the clips, Yes weights renormalised to sum 1, No weights renormalised to sum 1,
  d = sum_i wy_i W[yes_i] - sum_j wn_j W[no_j], unit length.

Written from scratch for Part 17. Does not import the first pass or the verifier.
lm_head read from the safetensors shard by parsing the header by hand (the verifier's approach).
AUC by the rank formula (scipy.stats.rankdata, ties averaged). Never copied from a json.
Writes only under edaic_rerun/part17/ and appends to DISCREPANCIES.md. ZERO writes under omni_final.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import sys, json, csv, time, hashlib, warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
np.seterr(all="ignore")   # Apple Accelerate raises spurious FPU flags on matmul (see part10 blas_note)

T0 = time.time()
# ------------------------------------------------------------------ paths
STATES_TASK = "<local data dir>/paper1_local_runs/probe2/pitt_states.npz"
PROBE2 = "<local data dir>/paper work/paper1_local_runs/probe2"
NPZ = os.path.join(PROBE2, "pitt_states.npz")
NESTED_JSON = os.path.join(PROBE2, "pitt_nested.json")
NESTED_OOF = os.path.join(PROBE2, "pitt_ans_nested_oof.csv")
ZS_CSV = os.path.join(PROBE2, "pitt_zeroshot_scores.csv")
MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
SNAP = ("<hf cache>/hub/models--Qwen--Qwen2-Audio-7B-Instruct/"
        "snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a")
SHARD = os.path.join(SNAP, "model-00005-of-00005.safetensors")
TOKJ = os.path.join(SNAP, "tokenizer.json")
CFG = os.path.join(SNAP, "config.json")
IDX = os.path.join(SNAP, "model.safetensors.index.json")
MID = "Qwen/Qwen2-Audio-7B-Instruct"
REL = "<local data dir>/release"
VERIFY_OUT = f"{REL}/edaic_rerun/part16/verify/M5_verify_out.json"
FIRST_SUM = f"{REL}/edaic_rerun/part16/M5_readout_direction_q2a_summary.csv"
OMNI_CSV = f"{REL}/omni/readout_direction.csv"
OMNI_JSON = f"{REL}/omni/readout_direction.json"
OMNI_H = f"{REL}/edaic_rerun/part15/H_readout_arms.csv"
Q3O_FIX = f"{REL}/edaic_rerun/part16/Q3O_direction_FIXED.json"
Q3O_TSV = f"{REL}/edaic_rerun/part16/rows/Q3Ofix.tsv"
Q3O_COS = f"{REL}/edaic_rerun/part16/pod_sync/readout_direction_q3o_FIXED.json"
DISC = f"{REL}/edaic_rerun/DISCREPANCIES.md"
OUT = f"{REL}/edaic_rerun/part17"
PERCLIP = os.path.join(OUT, "T3_q2a_direction.csv")
SUMCSV = os.path.join(OUT, "T3_q2a_direction_summary.csv")
SIDE = os.path.join(OUT, "T3_q2a_direction.json")
TSV = os.path.join(OUT, "rows", "T3.tsv")
LOG = os.path.join(OUT, "T3_q2a_direction.log")
for p in (PERCLIP, SUMCSV, SIDE, TSV, LOG):
    assert "omni_final" not in p, p
SEED, DRAWS = 0, 2000

_logf = open(LOG, "w")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); _logf.write(s + "\n"); _logf.flush()

def sha1mb(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()

DISC_WRITTEN = []; _DISC_QUEUE = []
def disc(sev, title, body):
    """queued; appended to DISCREPANCIES.md only after every output file is written (append only)."""
    _DISC_QUEUE.append(f"\n## [{sev}] [PART17] T3 {title} ({time.strftime('%Y-%m-%d %H:%M')})\n{body}\n")
    DISC_WRITTEN.append(f"[{sev}] {title}")
def flush_disc():
    with open(DISC, "a") as f:
        for s in _DISC_QUEUE:
            f.write(s)

# ------------------------------------------------------------------ rule 4: rank-formula AUC
def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=np.float64)
    n1 = int(y.sum()); n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = st.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

# ------------------------------------------------------------------ rule 5: speaker bootstrap
def boot(y, s, spk):
    rng = np.random.default_rng(SEED)            # fresh per cell
    y = np.asarray(y); s = np.asarray(s, float); spk = np.asarray(spk)
    us = np.unique(spk); idx = {u: np.flatnonzero(spk == u) for u in us}
    v = []
    for _ in range(DRAWS):
        pick = rng.choice(us, size=len(us), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        a = auc(y[ii], s[ii])
        if a == a:
            v.append(a)
    v = np.asarray(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v))

def boot_paired(y, s, spk, arm):
    """ONE speaker draw per replicate, conflict AUC and agreement AUC both recomputed inside it."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y); s = np.asarray(s, float); spk = np.asarray(spk); arm = np.asarray(arm)
    us = np.unique(spk); idx = {u: np.flatnonzero(spk == u) for u in us}
    v = []
    for _ in range(DRAWS):
        pick = rng.choice(us, size=len(us), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        yc, sc_, ac = y[ii], s[ii], arm[ii]
        a = auc(yc[ac == "conflict"], sc_[ac == "conflict"])
        b = auc(yc[ac == "agreement"], sc_[ac == "agreement"])
        if a == a and b == b:
            v.append(a - b)
    v = np.asarray(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v))

def floor(ref):
    """fresh default_rng(0), 2000 Gaussian directions in len(ref) dims, |cos| with ref."""
    rng = np.random.default_rng(SEED)
    G = rng.standard_normal((DRAWS, len(ref)))
    c = np.abs(G @ ref) / (np.linalg.norm(G, axis=1) * np.linalg.norm(ref))
    return float(c.mean()), float(np.percentile(c, 2.5)), float(np.percentile(c, 97.5)), int(len(c)), c

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ------------------------------------------------------------------ sources
log("=" * 100)
log("PART 17 T3  Qwen2-Audio readout-direction test, Pitt, corrected within-fold standardisation")
log("=" * 100)
if not os.path.exists(STATES_TASK):
    log(f"[path] task path does not resolve: {STATES_TASK}")
    log(f"[path] using {NPZ}  (paper1_local_runs now lives under 'paper work/')")
for p in (NPZ, NESTED_JSON, NESTED_OOF, ZS_CSV, MAN, SHARD, TOKJ, CFG, IDX):
    log(f"  exists={os.path.exists(p)}  {os.path.getsize(p):>13,} B  {p}")

z = np.load(NPZ, allow_pickle=True)
name = z["name"].astype(str); y = z["label"].astype(int); spk = z["spk"].astype(str)
p_yes_saved = z["p_yes"].astype(np.float64)
ANS = z["ans"]                                   # (468, 33, 4096) float32
X = ANS[:, -1, :].astype(np.float64)             # final LM stage at the first answer position
N, D = X.shape
NSPK = int(len(np.unique(spk)))
log(f"[states] ans {ANS.shape} {ANS.dtype} -> X = ans[:, -1, :] {X.shape}; n={N} n_spk={NSPK} "
    f"pos={int(y.sum())} neg={int((y == 0).sum())}")

man = pd.read_csv(MAN, dtype={"spk": str})
man["base"] = man["segment_path"].astype(str).map(os.path.basename)
assert man["base"].is_unique
m_arm = dict(zip(man["base"], man["set"])); m_lab = dict(zip(man["base"], man["label"].astype(int)))
m_spk = dict(zip(man["base"], man["spk"]))
nb = np.array([os.path.basename(v) for v in name])
assert set(nb) == set(man["base"]), "npz names do not match manifest basenames"
arm = np.array([m_arm[v] for v in nb])
assert (np.array([m_lab[v] for v in nb]) == y).all(), "label mismatch npz vs manifest"
# speaker grouping check: npz spk vs manifest spk must be a one-to-one relabelling
pairs = set(zip(spk, [m_spk[v] for v in nb]))
spk_1to1 = (len(pairs) == NSPK == len(set(m_spk[v] for v in nb)))
MC, MA = arm == "conflict", arm == "agreement"
log(f"[arms] conflict={int(MC.sum())} ({len(np.unique(spk[MC]))} spk)  "
    f"agreement={int(MA.sum())} ({len(np.unique(spk[MA]))} spk)  npz-vs-manifest speaker map one-to-one={spk_1to1}")
assert MC.sum() == 146 and MA.sum() == 322

# ------------------------------------------------------------------ token ids (single-token encodings)
from tokenizers import Tokenizer
tk = Tokenizer.from_file(TOKJ)
def ids(words):
    out = []
    for w in words:
        for form in (w, " " + w):
            t = tk.encode(form, add_special_tokens=False).ids
            if len(t) == 1:
                out.append(int(t[0]))
    return sorted(set(out))
YES, NO = ids(["Yes", "yes", "YES"]), ids(["No", "no", "NO"])
log(f"[tokens] yes ids {YES}  {[tk.id_to_token(i) for i in YES]}")
log(f"[tokens] no  ids {NO}  {[tk.id_to_token(i) for i in NO]}")

# ------------------------------------------------------------------ lm_head, header parsed by hand
idx_map = json.load(open(IDX))["weight_map"]
assert idx_map["language_model.lm_head.weight"] == os.path.basename(SHARD)
cfg = json.load(open(CFG))
TIE = cfg.get("tie_word_embeddings", cfg.get("text_config", {}).get("tie_word_embeddings", "ABSENT"))
with open(SHARD, "rb") as fh:
    hlen = int.from_bytes(fh.read(8), "little")
    hdr = json.loads(fh.read(hlen).decode("utf-8"))
info = hdr["language_model.lm_head.weight"]
assert info["dtype"] == "BF16"
V, H = info["shape"]; assert H == D
start = 8 + hlen + info["data_offsets"][0]
assert info["data_offsets"][1] - info["data_offsets"][0] == V * H * 2
raw = np.memmap(SHARD, dtype="<u2", mode="r", offset=start, shape=(V, H))
W = np.empty((V, H), dtype=np.float32)
for a0 in range(0, V, 16384):
    blk = np.asarray(raw[a0:a0 + 16384]).astype(np.uint32) << np.uint32(16)
    W[a0:a0 + 16384] = blk.view(np.float32)
del raw
log(f"[lm_head] {SHARD}  shape ({V}, {H}) bf16 -> float32, header parsed by hand; "
    f"tensors in shard: {[k for k in hdr if k != '__metadata__']}; tie_word_embeddings={TIE}")
# cross-check the hand decode against torch's bf16 decode on the Yes/No rows
try:
    import torch
    from safetensors import safe_open
    with safe_open(SHARD, framework="pt") as f:
        tW = f.get_slice("language_model.lm_head.weight")
        chk = np.stack([tW[i:i + 1].float().numpy()[0] for i in YES + NO])
    BF16_CHECK = float(np.abs(chk - W[YES + NO]).max())
except Exception as e:
    BF16_CHECK = f"not run: {e}"
log(f"[lm_head] hand bf16 decode vs torch decode on the 12 Yes/No rows: max|diff| = {BF16_CHECK}")

# ------------------------------------------------------------------ full-vocabulary softmax masses
lse = np.full(N, -np.inf)
for a0 in range(0, V, 16384):
    blk = X @ W[a0:a0 + 16384].astype(np.float64).T
    m = blk.max(axis=1)
    lse = np.logaddexp(lse, m + np.log(np.exp(blk - m[:, None]).sum(axis=1)))
WY = W[YES].astype(np.float64); WN = W[NO].astype(np.float64)
PY = np.exp(X @ WY.T - lse[:, None]); PN = np.exp(X @ WN.T - lse[:, None])
log(f"[softmax] full vocab: mean total Yes mass {PY.sum(1).mean():.6f}, mean total No mass {PN.sum(1).mean():.6f}")

def dvec(mask):
    wy = PY[mask].mean(0); wn = PN[mask].mean(0)
    wy = wy / wy.sum(); wn = wn / wn.sum()
    d = wy @ WY - wn @ WN
    return d / np.linalg.norm(d), wy, wn

d_all, wy_all, wn_all = dvec(np.ones(N, bool))
log("[direction] Yes weights " + "  ".join(f"{i}:{w:.6f}" for i, w in zip(YES, wy_all)))
log("[direction] No  weights " + "  ".join(f"{i}:{w:.6f}" for i, w in zip(NO, wn_all)))
# the verifier's variant: softmax over the 12 Yes/No logits only
L12 = np.concatenate([X @ WY.T, X @ WN.T], 1); L12 = np.exp(L12 - L12.max(1, keepdims=True))
L12 /= L12.sum(1, keepdims=True)
m12 = L12.mean(0); ny = len(YES)
d12 = (m12[:ny] / m12[:ny].sum()) @ WY - (m12[ny:] / m12[ny:].sum()) @ WN; d12 /= np.linalg.norm(d12)
d_plain = WY.mean(0) - WN.mean(0)
d_top = WY[int(np.argmax(wy_all))] - WN[int(np.argmax(wn_all))]
COS_D = {"full_vs_12way": cosine(d_all, d12), "full_vs_top": cosine(d_all, d_top),
         "full_vs_plain": cosine(d_all, d_plain)}
log(f"[direction] cos(d_fullvocab, d_12way verifier) = {COS_D['full_vs_12way']:.10f}   "
    f"cos(d, d_top) = {COS_D['full_vs_top']:.10f}   cos(d, d_plain) = {COS_D['full_vs_plain']:.6f}")

proj_d = X @ d_all
zs = pd.read_csv(ZS_CSV)
zs_name_col = [c for c in zs.columns if c in ("clip", "name")][0]
zs = zs.set_index(zs_name_col).loc[name]
AUC_ZS = auc(zs["label"].astype(int).values, zs["p_yes"].values)
AUC_PYES_NPZ = auc(y, p_yes_saved)
RHO = float(st.spearmanr(proj_d, p_yes_saved).statistic)
log(f"[gate] AUC(zero-shot p_yes, {os.path.basename(ZS_CSV)}) = {AUC_ZS:.4f}; AUC(X.d) = {auc(y, proj_d):.4f}; "
    f"delta {auc(y, proj_d) - AUC_ZS:+.4f} (part10 gate |delta| <= 0.01); spearman(X.d, saved p_yes) = {RHO:.6f}")

# ------------------------------------------------------------------ probe with canonical perp_z
def probe(Xm, ym, sm, d):
    oof = np.zeros(len(ym)); perp_z = np.zeros(len(ym)); keep_z = np.zeros(len(ym))
    fold = np.full(len(ym), -1); ws = []; pf_perp = []; pf_keep = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(Xm, ym, groups=sm)):
        assert not (set(sm[tr]) & set(sm[te])), "speaker leakage"
        sc = StandardScaler().fit(Xm[tr])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xm[tr]), ym[tr])
        oof[te] = lr.predict_proba(sc.transform(Xm[te]))[:, 1]
        wf = lr.coef_.ravel() / sc.scale_
        ws.append(wf)
        wp = wf - (np.dot(wf, d) / np.dot(d, d)) * d
        a, b = Xm[te] @ wf, Xm[te] @ wp
        keep_z[te] = (a - a.mean()) / a.std()          # matched control, d kept
        perp_z[te] = (b - b.mean()) / b.std()          # CANONICAL: standardised within the held-out fold
        fold[te] = k
        pf_perp.append(auc(ym[te], b)); pf_keep.append(auc(ym[te], a))
    w_raw = np.mean(np.stack(ws), 0)
    return dict(oof=oof, perp_z=perp_z, keep_z=keep_z, fold=fold, w_raw=w_raw, w_folds=ws,
                w_dir=w_raw / np.linalg.norm(w_raw), pf_perp=pf_perp, pf_keep=pf_keep,
                fold_sizes=np.bincount(fold).tolist(),
                spk_per_fold=[int(len(set(sm[fold == k]))) for k in range(5)])

log("\n[probe] all 468: GroupKFold(5) by speaker, StandardScaler in-fold, "
    "LogisticRegression(max_iter=2000, class_weight='balanced')")
P = probe(X, y, spk, d_all)
log(f"  fold sizes {P['fold_sizes']}  speakers/fold {P['spk_per_fold']}")
COS = cosine(P["w_dir"], d_all)
COS_FOLDS = [cosine(w, d_all) for w in P["w_folds"]]
fl_d = floor(d_all); fl_w = floor(P["w_dir"])
# the first pass drew per direction with rng.normal in a loop; check that stream equals the block draw
rng_chk = np.random.default_rng(SEED)
loop = np.stack([rng_chk.normal(size=D) for _ in range(DRAWS)])
STREAM_SAME = bool(np.array_equal(loop, np.random.default_rng(SEED).standard_normal((DRAWS, D))))
del loop
E_ABS = float(np.sqrt(2.0 / (np.pi * D)))
log(f"  cos(probe direction, d) = {COS:.4f}   per fold {[round(c, 4) for c in COS_FOLDS]}")
log(f"  floor vs d          : mean|cos| {fl_d[0]:.4f} [{fl_d[1]:.4f}, {fl_d[2]:.4f}] ({fl_d[3]} draws)")
log(f"  floor vs probe dir  : mean|cos| {fl_w[0]:.4f} [{fl_w[1]:.4f}, {fl_w[2]:.4f}] ({fl_w[3]} draws)")
log(f"  analytic E|cos| for a random direction in {D} dims = sqrt(2/(pi*{D})) = {E_ABS:.4f}")
log(f"  rng.normal loop stream == standard_normal block stream (fresh rng(0)): {STREAM_SAME}")

# ------------------------------------------------------------------ cosine CI, probe refit per draw (secondary)
rng_c = np.random.default_rng(SEED)
us = np.unique(spk); sidx = {u: np.flatnonzero(spk == u) for u in us}
cos_b = []
for _ in range(DRAWS):
    pick = rng_c.choice(us, size=len(us), replace=True)
    ii = np.concatenate([sidx[u] for u in pick])
    if len(np.unique(y[ii])) < 2:
        continue
    sc = StandardScaler().fit(X[ii])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[ii]), y[ii])
    cos_b.append(cosine(lr.coef_.ravel() / sc.scale_, d_all))
cos_b = np.asarray(cos_b)
COS_CI = (float(np.percentile(cos_b, 2.5)), float(np.percentile(cos_b, 97.5)), int(len(cos_b)), float(cos_b.mean()))
log(f"  cos speaker bootstrap, probe refit per draw on the resampled speakers: mean {COS_CI[3]:.4f} "
    f"[{COS_CI[0]:.4f}, {COS_CI[1]:.4f}] ({COS_CI[2]} usable draws)")

# ------------------------------------------------------------------ cells
RES = []   # summary rows
def add(cid, cell, metric, estimator, used, v, lo=None, hi=None, draws=None, n=None, nspk=None, note=""):
    RES.append(dict(id=cid, cell=cell, metric=metric, estimator=estimator, used=used, value=v,
                    lo=lo, hi=hi, draws_used=draws, n=n, n_spk=nspk, note=note))

def auc_cell(cid, cell, metric, estimator, used, yv, sv, spv, note=""):
    v = auc(yv, sv); lo, hi, dr = boot(yv, sv, spv)
    add(cid, cell, metric, estimator, used, v, lo, hi, dr, int(len(yv)), int(len(np.unique(spv))), note)
    return v, lo, hi, dr

ALLM = np.ones(N, bool)
add("T3_all_cos", "all", "cos(probe direction, Yes-minus-No readout direction)", "full fit", "yes",
    COS, None, None, None, N, NSPK, "probe direction = mean over folds of coef/scale, unit length")
add("T3_all_cos_bootCI", "all", "cos, speaker bootstrap with probe refit per draw", "full fit", "secondary",
    COS_CI[3], COS_CI[0], COS_CI[1], COS_CI[2], N, NSPK, "value column is the bootstrap mean; point estimate is T3_all_cos")
add("T3_floor_vs_d", "all", "random floor mean|cos| vs readout direction d", "fresh default_rng(0), 2000 dirs",
    "yes", fl_d[0], fl_d[1], fl_d[2], fl_d[3], N, NSPK, "verifier's and part10's reference vector")
add("T3_floor_vs_probe", "all", "random floor mean|cos| vs probe direction", "fresh default_rng(0), 2000 dirs",
    "secondary", fl_w[0], fl_w[1], fl_w[2], fl_w[3], N, NSPK, "task text and q3o script's reference vector")

CELLS = {}
for cell, m in (("all", ALLM), ("conflict", MC), ("agreement", MA)):
    est = "full 468 OOF" if cell == "all" else "full 468 OOF restricted to arm (DECIDED)"
    r = {}
    r["probe"] = auc_cell(f"T3_{cell}_probe", cell, "probe AUC out of fold", est, "yes", y[m], P["oof"][m], spk[m])
    r["proj_d"] = auc_cell(f"T3_{cell}_proj_d", cell, "AUC along readout direction (X.d)", est, "yes",
                           y[m], proj_d[m], spk[m])
    r["perp_z"] = auc_cell(f"T3_{cell}_perp_z", cell, "probe AUC with readout direction projected out (within-fold z)",
                           est, "yes", y[m], P["perp_z"][m], spk[m])
    r["keep_z"] = auc_cell(f"T3_{cell}_keep_z", cell, "matched control: same pooling, d kept (within-fold z)",
                           est, "secondary", y[m], P["keep_z"][m], spk[m])
    CELLS[cell] = r

# paired conflict minus agreement, decided estimator
PAIRED = {}
for key, s in (("perp_z", P["perp_z"]), ("probe", P["oof"]), ("proj_d", proj_d)):
    diff = auc(y[MC], s[MC]) - auc(y[MA], s[MA])
    lo, hi, dr = boot_paired(y, s, spk, arm)
    PAIRED[key] = (diff, lo, hi, dr)
    add(f"T3_paired_{key}", "conflict-agreement", f"paired difference, {key}",
        "full 468 OOF restricted to arm (DECIDED), paired speaker bootstrap",
        "yes" if key == "perp_z" else "secondary", diff, lo, hi, dr, N, NSPK)

# ------------------------------------------------------------------ refit within arm (NOT USED)
REFIT = {}; refit_cols = {}
for cell, m in (("conflict", MC), ("agreement", MA)):
    d_a, _, _ = dvec(m)
    Pa = probe(X[m], y[m], spk[m], d_a)
    pa = X[m] @ d_a
    ca = cosine(Pa["w_dir"], d_a)
    fa = floor(d_a)
    r = dict(cos=ca, cos_d_all=cosine(d_all, d_a), floor=fa[:4], fold_sizes=Pa["fold_sizes"])
    add(f"T3_{cell}_refit_cos", cell, "cos(probe direction, arm's own readout direction)", "refit within arm",
        "no (not used)", ca, None, None, None, int(m.sum()), int(len(np.unique(spk[m]))))
    r["probe"] = auc_cell(f"T3_{cell}_refit_probe", cell, "probe AUC out of fold", "refit within arm",
                          "no (not used)", y[m], Pa["oof"], spk[m])
    r["proj_d"] = auc_cell(f"T3_{cell}_refit_proj_d", cell, "AUC along arm's own readout direction",
                           "refit within arm", "no (not used)", y[m], pa, spk[m])
    r["perp_z"] = auc_cell(f"T3_{cell}_refit_perp_z", cell, "probe AUC, readout direction projected out (within-fold z)",
                           "refit within arm", "no (not used)", y[m], Pa["perp_z"], spk[m])
    REFIT[cell] = r
    refit_cols[cell] = (m, Pa, pa)

# ------------------------------------------------------------------ nested probe direction (supplement)
nj = json.load(open(NESTED_JSON)); LAY = [int(v) for v in nj["ans"]["chosen_layers"]]
X32 = ANS.astype(np.float32)
def inner_auc(Xl, yt, st_):
    """inner GroupKFold(4) on the outer-training speakers at one layer, mean inner AUC (as nested_probe_local.py)."""
    import warnings as _w; _w.filterwarnings("ignore"); np.seterr(all="ignore")
    s = 0.0
    for itr, ite in GroupKFold(n_splits=4).split(Xl, yt, groups=st_):
        sc = StandardScaler().fit(Xl[itr])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xl[itr]), yt[itr])
        p = lr.predict_proba(sc.transform(Xl[ite]))[:, 1]
        s += auc(yt[ite], p) if len(set(yt[ite])) > 1 else 0.5
    return s / 4
from joblib import Parallel, delayed
n_oof = np.zeros(N); n_perp = np.zeros(N); n_keep = np.zeros(N); n_layer = np.full(N, -1)
n_ws = []; lay_mine = []
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X32, y, groups=spk)):
    Xtr = X32[tr]
    inner = Parallel(n_jobs=12)(delayed(inner_auc)(np.ascontiguousarray(Xtr[:, l]), y[tr], spk[tr])
                                for l in range(X32.shape[1]))
    best = int(np.argmax(inner)); lay_mine.append(best)
    L = LAY[k]
    sc = StandardScaler().fit(X32[tr][:, L])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X32[tr][:, L]), y[tr])
    n_oof[te] = lr.predict_proba(sc.transform(X32[te][:, L]))[:, 1]
    wf = lr.coef_.ravel().astype(np.float64) / sc.scale_.astype(np.float64)
    n_ws.append(wf)
    wp = wf - (np.dot(wf, d_all) / np.dot(d_all, d_all)) * d_all
    XL = ANS[te, L, :].astype(np.float64)
    a, b = XL @ wf, XL @ wp
    n_keep[te] = (a - a.mean()) / a.std(); n_perp[te] = (b - b.mean()) / b.std(); n_layer[te] = L
nfile = pd.read_csv(NESTED_OOF).set_index("clip").loc[name]
NEST_MAXDIFF = float(np.abs(nfile["p_probe"].values - n_oof).max())
n_wdir = np.mean(np.stack(n_ws), 0); n_wdir /= np.linalg.norm(n_wdir)
N_COS = cosine(n_wdir, d_all); N_COS_F = [cosine(w, d_all) for w in n_ws]
N_FLOOR = floor(n_wdir)
log(f"\n[nested] saved chosen layers {LAY}; my inner GroupKFold(4) re-selection {lay_mine}; "
    f"OOF vs {os.path.basename(NESTED_OOF)} max|diff| = {NEST_MAXDIFF:.3e}")
log(f"[nested] cos(nested probe direction, d) = {N_COS:.4f}   per fold {[round(c, 4) for c in N_COS_F]}")
add("T3_nested_cos", "all", "cos(NESTED probe direction, readout direction d)", "nested, saved layers", "supplement",
    N_COS, None, None, None, N, NSPK, f"per-fold layers {LAY}; d is the final-stage readout direction")
add("T3_nested_floor_vs_probe", "all", "random floor mean|cos| vs nested probe direction",
    "fresh default_rng(0), 2000 dirs", "supplement", N_FLOOR[0], N_FLOOR[1], N_FLOOR[2], N_FLOOR[3], N, NSPK)
NEST = {}
for cell, m in (("all", ALLM), ("conflict", MC), ("agreement", MA)):
    est = "nested, full 468 OOF" + ("" if cell == "all" else " restricted to arm")
    NEST[cell] = dict(
        probe=auc_cell(f"T3_nested_{cell}_probe", cell, "nested probe AUC out of fold", est, "supplement",
                       y[m], n_oof[m], spk[m]),
        perp_z=auc_cell(f"T3_nested_{cell}_perp_z", cell, "nested probe AUC, d projected out (within-fold z)", est,
                        "supplement", y[m], n_perp[m], spk[m]),
        keep_z=auc_cell(f"T3_nested_{cell}_keep_z", cell, "nested matched control, d kept (within-fold z)", est,
                        "supplement", y[m], n_keep[m], spk[m]))
dn = auc(y[MC], n_perp[MC]) - auc(y[MA], n_perp[MA]); lo_, hi_, dr_ = boot_paired(y, n_perp, spk, arm)
NEST["paired"] = (dn, lo_, hi_, dr_)
add("T3_nested_paired_perp_z", "conflict-agreement", "nested paired difference, perp_z",
    "nested, restricted, paired speaker bootstrap", "supplement", dn, lo_, hi_, dr_, N, NSPK)

# ------------------------------------------------------------------ per-clip csv
per = pd.DataFrame(dict(name=name, spk=spk, label=y, arm=arm, oof_probe=P["oof"], proj_d=proj_d,
                        perp_z=P["perp_z"], fold=P["fold"], keep_z=P["keep_z"], p_yes_saved=p_yes_saved))
for col in ("oof_probe_armrefit", "proj_d_armrefit", "perp_z_armrefit", "fold_armrefit"):
    per[col] = np.nan
for cell, (m, Pa, pa) in refit_cols.items():
    per.loc[m, "oof_probe_armrefit"] = Pa["oof"]; per.loc[m, "proj_d_armrefit"] = pa
    per.loc[m, "perp_z_armrefit"] = Pa["perp_z"]; per.loc[m, "fold_armrefit"] = Pa["fold"]
per["nested_layer"] = n_layer; per["nested_oof"] = n_oof
per["nested_perp_z"] = n_perp; per["nested_keep_z"] = n_keep
per.to_csv(PERCLIP, index=False, float_format="%.10g")
pd.DataFrame(RES).to_csv(SUMCSV, index=False, float_format="%.10g")

# ------------------------------------------------------------------ compare with the verifier
V_ = json.load(open(VERIFY_OUT)); vr = {r["cell"]: r for r in V_["rows"]}
CMP = []
def cmp(what, mine, theirs):
    a, b = f"{mine:.4f}", f"{theirs:.4f}"
    CMP.append(dict(what=what, mine=a, verifier=b, ok=(a == b)))
cmp("all cos", COS, vr["all"]["cos"])
cmp("floor mean (vs d)", fl_d[0], vr["all"]["rand_mean_abs"]); cmp("floor lo", fl_d[1], vr["all"]["rand_lo"])
cmp("floor hi", fl_d[2], vr["all"]["rand_hi"])
for cell, vk in (("all", "all"), ("conflict", "conflict_restricted"), ("agreement", "agreement_restricted")):
    for key, vkey in (("probe", "auc_probe"), ("proj_d", "auc_d"), ("perp_z", "auc_without_d")):
        v, lo, hi, _ = CELLS[cell][key]
        cmp(f"{cell} {key}", v, vr[vk][vkey]); cmp(f"{cell} {key} lo", lo, vr[vk][vkey + "_lo"])
        cmp(f"{cell} {key} hi", hi, vr[vk][vkey + "_hi"])
for cell in ("conflict", "agreement"):
    vk = f"{cell}_refit"
    cmp(f"{cell} refit cos", REFIT[cell]["cos"], vr[vk]["cos"])
    for key, vkey in (("probe", "auc_probe"), ("proj_d", "auc_d"), ("perp_z", "auc_without_d")):
        v, lo, hi, _ = REFIT[cell][key]
        cmp(f"{cell} refit {key}", v, vr[vk][vkey]); cmp(f"{cell} refit {key} lo", lo, vr[vk][vkey + "_lo"])
        cmp(f"{cell} refit {key} hi", hi, vr[vk][vkey + "_hi"])
for key, vkey in (("perp_z", "auc_without_d"), ("probe", "auc_probe"), ("proj_d", "auc_d")):
    cmp(f"paired {key}", PAIRED[key][0], V_["paired"][vkey]["diff"])
    cmp(f"paired {key} lo", PAIRED[key][1], V_["paired"][vkey]["lo"])
    cmp(f"paired {key} hi", PAIRED[key][2], V_["paired"][vkey]["hi"])
NBAD = sum(1 for c in CMP if not c["ok"])

# first pass (training-fold z) for the record
fp = pd.read_csv(FIRST_SUM).set_index("cell")

# ------------------------------------------------------------------ three backbones, quoted from file
om = pd.read_csv(OMNI_CSV).set_index("dataset").loc["pitt"]
omj = [d for d in json.load(open(OMNI_JSON))["datasets"] if d["dataset"] == "pitt"][0]
omh = pd.read_csv(OMNI_H).set_index("row")
q3 = pd.read_csv(Q3O_TSV, sep="\t")
q3c = json.load(open(Q3O_COS)); q3f = json.load(open(Q3O_FIX))
def q3v(s):
    r = q3[q3["what"] == s].iloc[0]; return float(r["value"]), float(r["lo"]), float(r["hi"])

# ------------------------------------------------------------------ discrepancies
if not os.path.exists(STATES_TASK):
    disc("LOW", "states path moved",
         f"- `{STATES_TASK}` no longer resolves; `paper1_local_runs` now sits under `<local data dir>/paper work/`. "
         f"Same file: sha256 of first 1 MB `{sha1mb(NPZ)}` equals the Part 16 M5 sidecar value. "
         f"Every Part 16 sidecar that cites `<local data dir>/paper1_local_runs/...` now points at a dead path.")
disc("LOW", "first-pass random floor was NOT an advancing generator",
     f"- The Part 17 brief says the first pass drew its floor from an advancing generator. Its `rand_floor` "
     f"creates `default_rng(0)` fresh on every call, and the rng.normal-in-a-loop stream equals the "
     f"standard_normal block stream bit for bit (checked here: {STREAM_SAME}). Its three floors differ because it "
     f"measured |cos| against each cell's own PROBE direction (three different probes), while the verifier measured "
     f"against the READOUT direction d (three near-identical vectors). Both are valid floors: vs d "
     f"{fl_d[0]:.4f} [{fl_d[1]:.4f}, {fl_d[2]:.4f}], vs probe direction {fl_w[0]:.4f} [{fl_w[1]:.4f}, {fl_w[2]:.4f}]; "
     f"analytic E|cos| in {D} dims = {E_ABS:.4f}. The part10 canonical floor (Omni) is vs d, 1000 draws; "
     f"the q3o floor is vs the probe direction, 2000 draws. The three backbones' floors are therefore not "
     f"computed against the same reference vector; the difference is immaterial (all sit at sqrt(2/(pi*dim))).")
disc("MEDIUM", "'nested probe direction' vs the verifier's run",
     f"- The authors' wording asks for the cosine with the NESTED probe direction 'from the verifier's corrected run'. "
     f"The verifier's run (and Omni part10, and q3o) uses the FINAL-STAGE probe on ans[:, -1, :], not the nested "
     f"probe. The nested answer probe picks layers {LAY} (none is the final stage 32). Both are reported in "
     f"part17/T3_q2a_direction_summary.csv: final-stage cos {COS:.4f} (headline, comparable across the three "
     f"backbones) and nested cos {N_COS:.4f} (supplement; d is the final-stage readout direction applied at "
     f"mid layers, a logit-lens reading). The paper text must say which one it quotes.")
disc("LOW", "direction weighting differs between first pass and verifier",
     f"- First pass and q3o weight the Yes/No rows by FULL-vocabulary softmax mass; the verifier and part10 "
     f"(Omni) use the softmax over the Yes/No logits only. For Qwen2-Audio Pitt the two unit directions agree "
     f"to cos {COS_D['full_vs_12way']:.10f}; the reported 4-decimal values are {'identical' if NBAD == 0 else 'NOT all identical, see T3 sidecar'}.")
if NBAD:
    disc("MEDIUM", "own run vs verifier disagree at 4 dp",
         "- " + "; ".join(f"{c['what']}: mine {c['mine']} verifier {c['verifier']}" for c in CMP if not c["ok"]))

# ------------------------------------------------------------------ fragment tsv
def f4(v): return "" if v is None else f"{float(v):.4f}"
with open(TSV, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["id", "what", "value", "lo", "hi", "n", "n_spk", "file"])
    for r in RES:
        w.writerow([r["id"], f"Qwen2-Audio Pitt {r['cell']}: {r['metric']} [{r['estimator']}; used={r['used']}]",
                    f4(r["value"]), f4(r["lo"]), f4(r["hi"]), r["n"], r["n_spk"], PERCLIP])

# ------------------------------------------------------------------ sidecar
CMD = " ".join([sys.executable] + [os.path.abspath(sys.argv[0])] + sys.argv[1:])
side = {
    "part": "17 task 3", "date": time.strftime("%Y-%m-%d %H:%M"), "model_id": MID,
    "question": "Qwen2-Audio readout-direction test on Pitt with the canonical within-fold standardisation",
    "command": CMD, "python": sys.version.split()[0], "script": os.path.abspath(sys.argv[0]),
    "source_files": {"states": NPZ, "states_path_in_brief": STATES_TASK, "manifest": MAN, "lm_head_shard": SHARD,
                     "tokenizer": TOKJ, "config": CFG, "index": IDX, "zeroshot_scores": ZS_CSV,
                     "nested_json": NESTED_JSON, "nested_oof": NESTED_OOF, "verifier_out": VERIFY_OUT,
                     "first_pass_summary": FIRST_SUM, "omni_csv": OMNI_CSV, "omni_json": OMNI_JSON,
                     "omni_part_h": OMNI_H, "q3o_fixed_json": Q3O_FIX, "q3o_tsv": Q3O_TSV, "q3o_cos_json": Q3O_COS},
    "sha256_first_1MB": {k: sha1mb(p) for k, p in (("states", NPZ), ("manifest", MAN), ("lm_head_shard", SHARD),
                         ("tokenizer", TOKJ), ("config", CFG), ("zeroshot_scores", ZS_CSV), ("nested_json", NESTED_JSON),
                         ("nested_oof", NESTED_OOF), ("verifier_out", VERIFY_OUT), ("omni_csv", OMNI_CSV),
                         ("omni_part_h", OMNI_H), ("q3o_fixed_json", Q3O_FIX), ("q3o_tsv", Q3O_TSV),
                         ("q3o_cos_json", Q3O_COS))},
    "n": N, "n_speakers": NSPK, "n_conflict": int(MC.sum()), "n_speakers_conflict": int(len(np.unique(spk[MC]))),
    "n_agreement": int(MA.sum()), "n_speakers_agreement": int(len(np.unique(spk[MA]))),
    "seed": SEED, "draws": DRAWS,
    "bootstrap": "numpy.random.default_rng(0) created fresh per cell; speakers resampled with replacement "
                 "(rng.choice over np.unique(spk)); percentiles 2.5/97.5; per-clip scores held fixed; "
                 "paired = one speaker draw per replicate with both arm AUCs recomputed inside it",
    "auc": "rank formula, scipy.stats.rankdata with ties averaged: (sum ranks of positives - n1(n1+1)/2)/(n1 n0)",
    "features": "X = ans[:, -1, :] of pitt_states.npz (final LM stage, first answer position), float64",
    "yes_ids": YES, "no_ids": NO,
    "yes_tokens": [tk.id_to_token(i) for i in YES], "no_tokens": [tk.id_to_token(i) for i in NO],
    "token_rule": "Yes/yes/YES and No/no/NO, bare and leading-space, kept only when the tokenizer encodes the form as ONE token",
    "lm_head": {"tensor": "language_model.lm_head.weight", "shape": [V, H], "dtype": "BF16",
                "tie_word_embeddings": TIE, "decode": "header parsed by hand, bf16 bits shifted into float32",
                "bf16_decode_vs_torch_maxdiff_on_yesno_rows": BF16_CHECK},
    "direction_definition": ("mass-weighted: p = softmax over the FULL vocabulary of X @ lm_head.T per clip; mean p of "
                             "each Yes id over the cell's clips, renormalised to sum 1 over the Yes ids; same for No; "
                             "d = sum_i wy_i W[yes_i] - sum_j wn_j W[no_j], unit length. Same rule as part16 M5 first pass "
                             "and p15/direction_q3o.py."),
    "direction_weights_all": {"yes": dict(zip(map(str, YES), map(float, wy_all))),
                              "no": dict(zip(map(str, NO), map(float, wn_all)))},
    "direction_cosines": COS_D,
    "probe": "GroupKFold(5) by speaker, StandardScaler fit on the training fold, LogisticRegression(max_iter=2000, "
             "class_weight='balanced'); probe direction = mean over folds of coef_/scaler.scale_, unit length",
    "perp_z_definition": "per training fold w_perp = w_fold - (w_fold.d/d.d) d; b = X[held-out] @ w_perp; "
                         "perp_z[held-out] = (b - b.mean())/b.std() WITHIN the held-out fold; pooled; AUC on pooled",
    "keep_z_definition": "matched control: identical, with d NOT removed",
    "arm_estimator": "DECIDED: full 468-clip OOF values restricted to the arm (Omni Part H estimator). "
                     "Refit-within-arm reported secondary, NOT USED.",
    "floor": {"rng": "numpy.random.default_rng(0) fresh per cell", "draws": DRAWS, "dim": D,
              "vs_d": fl_d[:4], "vs_probe_direction": fl_w[:4], "analytic_E_abs_cos": E_ABS,
              "loop_stream_equals_block_stream": STREAM_SAME},
    "cos_bootstrap": {"what": "speaker bootstrap, probe refit on each resample (no CV), cos(coef/scale, d)",
                      "mean": COS_CI[3], "lo": COS_CI[0], "hi": COS_CI[1], "usable": COS_CI[2], "status": "secondary"},
    "gate": {"auc_zeroshot_file": AUC_ZS, "auc_saved_p_yes_npz": AUC_PYES_NPZ, "auc_proj_d": auc(y, proj_d),
             "spearman_proj_d_vs_saved_p_yes": RHO},
    "fold_sizes": P["fold_sizes"], "speakers_per_fold": P["spk_per_fold"],
    "per_fold_auc_perp_raw": P["pf_perp"], "per_fold_auc_keep_raw": P["pf_keep"], "per_fold_cos": COS_FOLDS,
    "nested": {"saved_layers": LAY, "my_reselected_layers": lay_mine, "oof_maxdiff_vs_file": NEST_MAXDIFF,
               "cos": N_COS, "per_fold_cos": N_COS_F, "floor_vs_nested_dir": N_FLOOR[:4],
               "status": "supplement; d is the final-stage readout direction applied at the nested layers"},
    "refit_within_arm_not_used": {k: {"cos": v["cos"], "cos_d_all_vs_d_arm": v["cos_d_all"], "floor_vs_d_arm": v["floor"],
                                      "fold_sizes": v["fold_sizes"]} for k, v in REFIT.items()},
    "compare_with_verifier_4dp": CMP, "n_disagree_with_verifier": NBAD,
    "discrepancies_appended": DISC_WRITTEN,
    "outputs": {"per_clip_csv": PERCLIP, "summary_csv": SUMCSV, "sidecar": SIDE, "fragment": TSV, "log": LOG},
    "runtime_s": round(time.time() - T0, 1),
}
json.dump(side, open(SIDE, "w"), indent=1, default=float)

# ------------------------------------------------------------------ print
def line(label, v, lo, hi, n, ns, extra=""):
    iv = f"[{lo:.4f}, {hi:.4f}]" if lo is not None else "[no interval]"
    return f"{label}: {v:.4f} {iv}, n={n}, n_speakers={ns}{extra} | {PERCLIP}"
log("\n" + "=" * 100)
log("RESULTS (paste lines)")
log("=" * 100)
log(line("Q2A Pitt cos(probe direction, Yes-minus-No readout direction), final stage", COS, None, None, N, NSPK))
log(line("Q2A Pitt cos, speaker bootstrap with probe refit per draw (secondary; mean)", COS_CI[3], COS_CI[0], COS_CI[1], N, NSPK,
         f", usable {COS_CI[2]}/{DRAWS}"))
log(line("Q2A Pitt random floor mean|cos| vs readout direction d (fresh rng(0), 2000 dirs)", fl_d[0], fl_d[1], fl_d[2], N, NSPK))
log(line("Q2A Pitt random floor mean|cos| vs probe direction (fresh rng(0), 2000 dirs)", fl_w[0], fl_w[1], fl_w[2], N, NSPK))
for cell, lab in (("all", "ALL"), ("conflict", "CONFLICT (restricted, decided)"), ("agreement", "AGREEMENT (restricted, decided)")):
    r = CELLS[cell]; m = ALLM if cell == "all" else (MC if cell == "conflict" else MA)
    n_, ns_ = int(m.sum()), int(len(np.unique(spk[m])))
    for key, desc in (("probe", "probe AUC (OOF)"), ("proj_d", "AUC along readout direction"),
                      ("perp_z", "probe AUC after projecting d out (within-fold z)"),
                      ("keep_z", "matched control, d kept (within-fold z)")):
        v, lo, hi, dr = r[key]
        log(line(f"Q2A Pitt {lab} {desc}", v, lo, hi, n_, ns_, f", usable {dr}/{DRAWS}"))
for key, desc in (("perp_z", "after-projection AUC"), ("probe", "probe AUC"), ("proj_d", "along-d AUC")):
    v, lo, hi, dr = PAIRED[key]
    log(line(f"Q2A Pitt PAIRED conflict minus agreement, {desc}", v, lo, hi, N, NSPK, f", usable {dr}/{DRAWS}"))
log("-- refit within arm, NOT USED --")
for cell in ("conflict", "agreement"):
    m = MC if cell == "conflict" else MA; n_, ns_ = int(m.sum()), int(len(np.unique(spk[m])))
    log(line(f"Q2A Pitt {cell.upper()} refit (not used) cos", REFIT[cell]["cos"], None, None, n_, ns_))
    for key in ("probe", "proj_d", "perp_z"):
        v, lo, hi, dr = REFIT[cell][key]
        log(line(f"Q2A Pitt {cell.upper()} refit (not used) {key}", v, lo, hi, n_, ns_))
log("-- nested probe direction, supplement --")
log(line(f"Q2A Pitt NESTED cos(nested probe direction, d), layers {LAY}", N_COS, None, None, N, NSPK))
log(line("Q2A Pitt random floor vs nested probe direction", N_FLOOR[0], N_FLOOR[1], N_FLOOR[2], N, NSPK))
for cell in ("all", "conflict", "agreement"):
    m = ALLM if cell == "all" else (MC if cell == "conflict" else MA); n_, ns_ = int(m.sum()), int(len(np.unique(spk[m])))
    for key in ("probe", "perp_z"):
        v, lo, hi, dr = NEST[cell][key]
        log(line(f"Q2A Pitt NESTED {cell} {key}", v, lo, hi, n_, ns_))
v, lo, hi, dr = NEST["paired"]
log(line("Q2A Pitt NESTED PAIRED conflict minus agreement, perp_z", v, lo, hi, N, NSPK))

log("\nFIRST PASS (training-fold z, SUPERSEDED) vs CORRECTED (within-fold z), auc_without_d:")
for cell, fk in (("all", "all"), ("conflict", "conflict_restricted"), ("agreement", "agreement_restricted")):
    log(f"  {cell:<10} first pass {float(fp.loc[fk, 'auc_without_d']):.4f}  ->  corrected {CELLS[cell]['perp_z'][0]:.4f}")

log(f"\nVERIFIER CHECK: {len(CMP) - NBAD}/{len(CMP)} values identical to M5_verify_out.json at 4 dp")
for c in CMP:
    if not c["ok"]:
        log(f"  XX {c['what']}: mine {c['mine']} verifier {c['verifier']}")

log("\n" + "=" * 100)
log("THREE BACKBONES, Pitt, all / conflict / agreement  (probe / along d / probe minus d, within-fold z)")
log("=" * 100)
log(f"Qwen2.5-Omni-7B : cos {float(om['cosine']):.4f}, floor {float(om['cosine_random']):.4f} (vs d, 1000 dirs); "
    f"all {float(om['auc_probe']):.4f} / {float(om['auc_d']):.4f} / {float(om['auc_probe_without_d']):.4f} | {OMNI_CSV}")
log(f"                  conflict probe {float(omh.loc['omni_part10_probe_conflict','value']):.4f}, "
    f"agreement probe {float(omh.loc['omni_part10_probe_agreement','value']):.4f}; along-d and minus-d by arm: "
    f"not in any file | {OMNI_H}")
qa, qc, qg = [q3v(f"Qwen3-Omni Pitt {c}: probe") for c in ("all", "conflict", "agreement")]
qad, qcd, qgd = [q3v(f"Qwen3-Omni Pitt {c}: along readout direction") for c in ("all", "conflict", "agreement")]
qaw, qcw, qgw = [q3v(f"Qwen3-Omni Pitt {c}: probe minus readout dir (CANONICAL)") for c in ("all", "conflict", "agreement")]
log(f"Qwen3-Omni-30B  : cos {q3c['cosine']:.4f}, floor {q3c['cosine_random_mean_abs']:.4f} "
    f"[{q3c['cosine_random_ci'][0]:.4f}, {q3c['cosine_random_ci'][1]:.4f}] (vs probe dir, 2000 dirs) | {Q3O_COS}")
log(f"                  all {qa[0]:.4f} / {qad[0]:.4f} / {qaw[0]:.4f}; conflict {qc[0]:.4f} / {qcd[0]:.4f} / {qcw[0]:.4f}; "
    f"agreement {qg[0]:.4f} / {qgd[0]:.4f} / {qgw[0]:.4f} | {Q3O_TSV}  (canonical json value {q3f['conventions']['CANONICAL_foldz']:.4f})")
A = CELLS
log(f"Qwen2-Audio-7B  : cos {COS:.4f}, floor {fl_d[0]:.4f} [{fl_d[1]:.4f}, {fl_d[2]:.4f}] (vs d, 2000 dirs); "
    f"all {A['all']['probe'][0]:.4f} / {A['all']['proj_d'][0]:.4f} / {A['all']['perp_z'][0]:.4f}; "
    f"conflict {A['conflict']['probe'][0]:.4f} / {A['conflict']['proj_d'][0]:.4f} / {A['conflict']['perp_z'][0]:.4f}; "
    f"agreement {A['agreement']['probe'][0]:.4f} / {A['agreement']['proj_d'][0]:.4f} / {A['agreement']['perp_z'][0]:.4f} | {PERCLIP}")
for p in (PERCLIP, SUMCSV, SIDE, TSV):
    assert os.path.getsize(p) > 0, p
flush_disc()
log(f"\nwrote {PERCLIP}\nwrote {SUMCSV}\nwrote {SIDE}\nwrote {TSV}\nwrote {LOG}")
log(f"discrepancies appended to {DISC}: {DISC_WRITTEN}")
log(f"runtime {time.time() - T0:.1f}s")
