"""INDEPENDENT VERIFICATION of PART16 M5 (Qwen2-Audio readout-direction test on Pitt).

Written from scratch. Does NOT read, import or exec M5_readout_direction_q2a.py.
Goes back to the ORIGINAL sources:
  states   : <local data dir>/paper1_local_runs/probe2/pitt_states.npz
  manifest : manifests/pitt_conflict_manifest_468.csv
  lm_head  : ~/.cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/
             0a095220c30b7b31434169c3086508ef3ea5bf0a/model-00005-of-00005.safetensors

AUC here is NOT a rank-sum one-liner. Two independent implementations:
  auc_u    : Mann-Whitney U by explicit pairwise concordance count, ties = 0.5
  auc_trap : trapezoid over the full ROC built from descending unique thresholds
Both are cross-checked against each other on every headline value.

Writes only under part16/verify/. ZERO writes under release/omni_final.
"""
import os, json, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
np.seterr(all="ignore")
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

NPZ = "<local data dir>/paper1_local_runs/probe2/pitt_states.npz"
MAN = "manifests/pitt_conflict_manifest_468.csv"
SNAP = ("<local data dir>/.cache/huggingface/hub/"
        "models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a")
SHARD = os.path.join(SNAP, "model-00005-of-00005.safetensors")
OUT = "<local data dir>/Desktop/release/edaic_rerun/part16/verify"
THEIR_CSV = "scores/part16/M/M5_readout_direction_q2a.csv"
THEIR_SUM = "scores/part16/M/M5_readout_direction_q2a_summary.csv"

DRAWS = 2000
SEED = 0
t0 = time.time()

# ---------------------------------------------------------------- AUC, route A
def auc_u(y, s):
    """Mann-Whitney U by explicit concordant-pair count. Ties contribute 0.5.
    No ranking, no sorting: a direct n1 x n0 comparison."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=np.float64)
    a = s[y == 1]; b = s[y == 0]
    if a.size == 0 or b.size == 0:
        return float("nan")
    # chunk so the outer product never blows up
    wins = 0.0
    step = max(1, int(4_000_000 // max(b.size, 1)))
    for i in range(0, a.size, step):
        blk = a[i:i + step][:, None]
        wins += np.count_nonzero(blk > b[None, :]) + 0.5 * np.count_nonzero(blk == b[None, :])
    return float(wins / (a.size * b.size))

# ---------------------------------------------------------------- AUC, route B
def auc_trap(y, s):
    """Trapezoid area under the full ROC curve, thresholds = descending unique scores."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=np.float64)
    P = int((y == 1).sum()); N = int((y == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    thr = np.unique(s)[::-1]
    tpr = [0.0]; fpr = [0.0]
    for t in thr:
        sel = s >= t
        tpr.append(float(np.count_nonzero(y[sel] == 1)) / P)
        fpr.append(float(np.count_nonzero(y[sel] == 0)) / N)
    tpr = np.array(tpr); fpr = np.array(fpr)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))

def auc_both(y, s, tag=""):
    a = auc_u(y, s); b = auc_trap(y, s)
    if not (np.isnan(a) and np.isnan(b)):
        assert abs(a - b) < 1e-12, f"my two AUC routes disagree on {tag}: {a} vs {b}"
    return a

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

R4 = lambda v: float(f"{float(v):.4f}")

# ---------------------------------------------------------------- sources
print("=" * 78)
print("SOURCES")
for p in (NPZ, MAN, SHARD):
    print(f"  exists={os.path.exists(p)}  {os.path.getsize(p):>12,} B  {p}")

z = np.load(NPZ, allow_pickle=True)
name = z["name"].astype(str)
y = z["label"].astype(int)
spk = z["spk"].astype(str)
p_yes_saved = z["p_yes"].astype(np.float64)
ANS = z["ans"]
S = ANS[:, -1, :].astype(np.float64)
n = len(y)
print(f"  npz n={n}  ans stage used = ans[:,-1,:] (stage {ANS.shape[1]-1} of {ANS.shape[1]})  "
      f"n_spk={len(np.unique(spk))}  pos={int(y.sum())} neg={int((1-y).sum())}")

# arms straight from the manifest, not from the npz name prefix
import csv as _csv
with open(MAN, newline="") as f:
    man = list(_csv.DictReader(f))
arm_of = {r["segment_path"].rsplit("/", 1)[-1]: r["set"] for r in man}
lab_of = {r["segment_path"].rsplit("/", 1)[-1]: int(r["label"]) for r in man}
arm = np.array([arm_of[x] for x in name])
assert (np.array([lab_of[x] for x in name]) == y).all(), "manifest labels disagree with npz labels"
print(f"  manifest rows={len(man)}  conflict={(arm=='conflict').sum()}  agreement={(arm=='agreement').sum()}")

# ---------------------------------------------------------------- token ids, built here
tok = json.load(open(os.path.join(SNAP, "tokenizer.json")))
vocab = tok["model"]["vocab"]
def ids_for(words):
    out = []
    for w in words:
        for form in (w, "Ġ" + w):       # bare and leading-space (GPT2 byte-level 'G with dot')
            if form in vocab:
                out.append(int(vocab[form]))
    return sorted(set(out))
yes_ids = ids_for(["Yes", "yes", "YES"])
no_ids = ids_for(["No", "no", "NO"])
print(f"  yes_ids={yes_ids}")
print(f"  no_ids ={no_ids}")

with safe_open(SHARD, framework="np") as f:
    keys = list(f.keys())
    assert keys == ["language_model.lm_head.weight"], keys
    sl = f.get_slice("language_model.lm_head.weight")
    shape = sl.get_shape(); dt = sl.get_dtype()

def read_bf16_rows(path, tensor, rows):
    """Read named rows of a bf16 safetensors tensor by hand: header, byte offset,
    then bf16 -> float32 by moving the 16 bits into the top half of a float32."""
    with open(path, "rb") as fh:
        hlen = int.from_bytes(fh.read(8), "little")
        hdr = json.loads(fh.read(hlen).decode("utf-8"))
        base = 8 + hlen
        info = hdr[tensor]
        assert info["dtype"] == "BF16", info["dtype"]
        nrow, ncol = info["shape"]
        start = base + info["data_offsets"][0]
        out = np.empty((len(rows), ncol), dtype=np.float64)
        for j, i in enumerate(rows):
            fh.seek(start + i * ncol * 2)
            raw = np.frombuffer(fh.read(ncol * 2), dtype="<u2").astype(np.uint32)
            out[j] = ((raw << np.uint32(16)).astype(np.uint32)).view(np.float32).astype(np.float64)
        return out, (nrow, ncol)

Wy, shp = read_bf16_rows(SHARD, "language_model.lm_head.weight", yes_ids)
Wn, _ = read_bf16_rows(SHARD, "language_model.lm_head.weight", no_ids)
print(f"  lm_head shape={shape} (header says {shp})  dtype={dt}  keys_in_shard={len(keys)}")
print(f"  |Wy| rows={Wy.shape}  |Wn| rows={Wn.shape}  max|W|={max(np.abs(Wy).max(), np.abs(Wn).max()):.6f}")
cfg = json.load(open(os.path.join(SNAP, "config.json")))
print(f"  config tie_word_embeddings = {cfg.get('tie_word_embeddings', 'ABSENT')}")

# ---------------------------------------------------------------- direction
def yesno_mass(Smat):
    both = np.concatenate([Smat @ Wy.T, Smat @ Wn.T], axis=1)
    m = both.max(axis=1, keepdims=True)
    e = np.exp(both - m)
    return e / e.sum(axis=1, keepdims=True)

def direction(Smat):
    mp = yesno_mass(Smat).mean(axis=0)
    ny = len(yes_ids)
    wy = mp[:ny] / mp[:ny].sum()
    wn = mp[ny:] / mp[ny:].sum()
    d = wy @ Wy - wn @ Wn
    return d / np.linalg.norm(d), wy, wn

d_all, wy_all, wn_all = direction(S)
d_plain = Wy.mean(0) - Wn.mean(0)
d_top = Wy[int(np.argmax(wy_all))] - Wn[int(np.argmax(wn_all))]
print("\nDIRECTION")
print(f"  yes weights {np.round(wy_all,6).tolist()}")
print(f"  no  weights {np.round(wn_all,6).tolist()}")
print(f"  cos(d_weighted, d_top)   = {cosine(d_all, d_top):.8f}")
print(f"  cos(d_weighted, d_plain) = {cosine(d_all, d_plain):.8f}")

# ---------------------------------------------------------------- probe
def probe(Smat, yv, spkv, d):
    """Returns oof probability, per-fold coefficient-projection z score,
    the mean raw coefficient, fold sizes and speakers per fold."""
    oof = np.zeros(len(yv)); perp_z = np.zeros(len(yv)); folds = np.full(len(yv), -1)
    wlist = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(Smat, yv, groups=spkv)):
        assert not (set(spkv[tr]) & set(spkv[te])), "speaker leakage"
        sc = StandardScaler().fit(Smat[tr])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Smat[tr]), yv[tr])
        oof[te] = lr.predict_proba(sc.transform(Smat[te]))[:, 1]
        folds[te] = k
        wf = lr.coef_.ravel() / sc.scale_
        wlist.append(wf)
        wp = wf - (np.dot(wf, d) / np.dot(d, d)) * d
        b = Smat[te] @ wp
        perp_z[te] = (b - b.mean()) / b.std()
    return oof, perp_z, np.mean(np.stack(wlist), 0), np.bincount(folds).tolist(), \
           [len(set(spkv[folds == k])) for k in range(5)]

def probe_xproj(Smat, yv, spkv, d):
    """Literal reading: project d out of X first, then run the whole probe."""
    dn = d / np.linalg.norm(d)
    Xp = Smat - np.outer(Smat @ dn, dn)
    oof = np.zeros(len(yv))
    for tr, te in GroupKFold(n_splits=5).split(Xp, yv, groups=spkv):
        sc = StandardScaler().fit(Xp[tr])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xp[tr]), yv[tr])
        oof[te] = lr.predict_proba(sc.transform(Xp[te]))[:, 1]
    return oof

# ---------------------------------------------------------------- bootstrap
def boot_ci(yv, sv, spkv, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    us = np.unique(spkv)
    idx = {u: np.where(spkv == u)[0] for u in us}
    vals = []
    for _ in range(draws):
        pick = rng.choice(us, size=len(us), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        v = auc_u(yv[ii], sv[ii])
        if v == v:
            vals.append(v)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)

def boot_paired(yv, sv, spkv, armv, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    us = np.unique(spkv)
    idx = {u: np.where(spkv == u)[0] for u in us}
    vals = []
    for _ in range(draws):
        pick = rng.choice(us, size=len(us), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        ma = armv[ii] == "conflict"; mb = armv[ii] == "agreement"
        va = auc_u(yv[ii][ma], sv[ii][ma]); vb = auc_u(yv[ii][mb], sv[ii][mb])
        if va == va and vb == vb:
            vals.append(va - vb)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)

def rand_floor(d, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((draws, len(d)))
    c = np.abs((G @ d) / (np.linalg.norm(G, axis=1) * np.linalg.norm(d)))
    return float(c.mean()), float(np.percentile(c, 2.5)), float(np.percentile(c, 97.5)), draws

# ---------------------------------------------------------------- run: all
print("\nPROBE: all 468")
oof_all, perpz_all, wraw_all, fs_all, spf_all = probe(S, y, spk, d_all)
xproj_all = probe_xproj(S, y, spk, d_all)
score_d_all = S @ d_all
print(f"  fold sizes {fs_all}  speakers/fold {spf_all}")

cells = {}
cells["all"] = dict(mask=np.ones(n, bool), cos=cosine(wraw_all, d_all), d=d_all,
                    probe=oof_all, dscore=score_d_all, without=perpz_all, xproj=xproj_all)

# arm refits: own direction, own folds, own scaler
for a in ("conflict", "agreement"):
    m = arm == a
    d_a, _, _ = direction(S[m])
    oof_a, perpz_a, wraw_a, fs_a, spf_a = probe(S[m], y[m], spk[m], d_a)
    xp_a = probe_xproj(S[m], y[m], spk[m], d_a)
    print(f"PROBE: {a}_refit n={m.sum()}  fold sizes {fs_a}  speakers/fold {spf_a}  "
          f"cos(d_all,d_{a})={cosine(d_all,d_a):.10f}")
    cells[f"{a}_refit"] = dict(mask=m, cos=cosine(wraw_a, d_a), d=d_a,
                               probe=oof_a, dscore=S[m] @ d_a, without=perpz_a, xproj=xp_a)
    cells[f"{a}_restricted"] = dict(mask=m, cos=None, d=d_all,
                                    probe=oof_all[m], dscore=score_d_all[m],
                                    without=perpz_all[m], xproj=xproj_all[m])

# ---------------------------------------------------------------- summarise
their_sum = {r["cell"]: r for r in _csv.DictReader(open(THEIR_SUM))}
rows = []
order = ["all", "conflict_refit", "agreement_refit", "conflict_restricted", "agreement_restricted"]
for cname in order:
    c = cells[cname]
    m = c["mask"]; yv = y[m]; sv = spk[m]
    rec = dict(cell=cname, n=int(m.sum()), n_spk=int(len(np.unique(sv))))
    if c["cos"] is not None:
        rec["cos"] = c["cos"]
        rm, rlo, rhi, rn = rand_floor(c["d"])
        rec.update(rand_mean_abs=rm, rand_lo=rlo, rand_hi=rhi, rand_used=rn)
    for key, sc in (("auc_probe", c["probe"]), ("auc_d", c["dscore"]),
                    ("auc_without_d", c["without"]), ("auc_without_d_Xproj", c["xproj"])):
        v = auc_both(yv, sc, f"{cname}/{key}")
        lo, hi, used = boot_ci(yv, np.asarray(sc, float), sv)
        rec[key] = v; rec[key + "_lo"] = lo; rec[key + "_hi"] = hi; rec[key + "_draws"] = used
    rows.append(rec)

paired = {}
for key, sc in (("auc_probe", oof_all), ("auc_d", score_d_all), ("auc_without_d", perpz_all)):
    va = auc_u(y[arm == "conflict"], np.asarray(sc)[arm == "conflict"])
    vb = auc_u(y[arm == "agreement"], np.asarray(sc)[arm == "agreement"])
    lo, hi, used = boot_paired(y, np.asarray(sc, float), spk, arm)
    paired[key] = dict(diff=va - vb, lo=lo, hi=hi, draws=used)

# ---------------------------------------------------------------- sanity vs saved p_yes
from scipy.stats import spearmanr
sp = float(spearmanr(score_d_all, p_yes_saved).statistic)
auc_saved = auc_both(y, p_yes_saved, "saved_p_yes")
print(f"\nSANITY  spearman(S.d, saved p_yes) = {sp:.7f}   auc(saved p_yes) = {auc_saved:.6f}")

# ---------------------------------------------------------------- compare
print("\n" + "=" * 78)
print("COMPARISON  (mine vs theirs, 4 decimals)")
disagree = []
checked = []
def cmp(what, mine, theirs, extra=""):
    a = f"{float(mine):.4f}"; b = f"{float(theirs):.4f}"
    ok = (a == b)
    if not ok:
        disagree.append((what, a, b))
    checked.append(dict(what=what, mine=a, theirs=b, ok=ok))
    print(f"  {'OK ' if ok else 'XX '} {what:<48} mine={a:>8}  theirs={b:>8} {extra}")

for rec in rows:
    t = their_sum[rec["cell"]]
    cname = rec["cell"]
    cmp(f"{cname} n", rec["n"], float(t["n"]))
    cmp(f"{cname} n_spk", rec["n_spk"], float(t["n_spk"]))
    if "cos" in rec:
        cmp(f"{cname} cos", rec["cos"], float(t["cos"]))
        cmp(f"{cname} rand_mean_abs", rec["rand_mean_abs"], float(t["rand_mean_abs"]))
        cmp(f"{cname} rand_lo", rec["rand_lo"], float(t["rand_lo"]))
        cmp(f"{cname} rand_hi", rec["rand_hi"], float(t["rand_hi"]))
    for key in ("auc_probe", "auc_d", "auc_without_d", "auc_without_d_Xproj"):
        cmp(f"{cname} {key}", rec[key], float(t[key]))
        cmp(f"{cname} {key}_lo", rec[key + "_lo"], float(t[key + "_lo"]))
        cmp(f"{cname} {key}_hi", rec[key + "_hi"], float(t[key + "_hi"]))

their_json = json.load(open("scores/part16/M/M5_readout_direction_q2a.json"))
tp = their_json["paired_conflict_minus_agreement"]
for key in ("auc_probe", "auc_d", "auc_without_d"):
    cmp(f"paired {key} diff", paired[key]["diff"], tp[key]["diff"])
    cmp(f"paired {key} lo", paired[key]["lo"], tp[key]["lo"])
    cmp(f"paired {key} hi", paired[key]["hi"], tp[key]["hi"])

# per-clip csv columns: do their saved scores reproduce mine?
their = list(_csv.DictReader(open(THEIR_CSV)))
tmap = {r["clip"]: r for r in their}
assert set(tmap) == set(name), "per-clip csv clip set differs from npz names"
col = lambda k: np.array([float(tmap[x][k]) for x in name])
print("\nPER-CLIP COLUMN AGREEMENT (max abs diff against my recomputation)")
for k, mine in (("p_probe_oof_all", oof_all), ("score_along_d_all", score_d_all),
                ("z_probe_oof_without_d_all", perpz_all),
                ("p_probe_oof_without_d_Xproj_all", xproj_all),
                ("p_yes_scorer_saved", p_yes_saved)):
    print(f"  {k:<36} max|diff| = {np.abs(col(k) - mine).max():.3e}")

print("\n" + "=" * 78)
print(f"DISAGREEMENTS AT 4dp: {len(disagree)}")
for w, a, b in disagree:
    print(f"  XX {w}: mine={a} theirs={b}")

json.dump(dict(rows=rows, paired=paired, checked=checked,
               disagree=disagree, spearman=sp, auc_saved_p_yes=auc_saved,
               yes_ids=yes_ids, no_ids=no_ids,
               cos_dw_dtop=cosine(d_all, d_top), cos_dw_dplain=cosine(d_all, d_plain),
               runtime_s=round(time.time() - t0, 1)),
          open(os.path.join(OUT, "M5_verify_out.json"), "w"), indent=1, default=float)
print(f"\nwrote {OUT}/M5_verify_out.json   runtime {time.time()-t0:.1f}s")
