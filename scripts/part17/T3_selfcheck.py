"""PART 17 T3 self-check, separate code path from T3_q2a_direction.py.

1. Every AUC in T3_q2a_direction_summary.csv recomputed from the per-clip csv with a different AUC
   implementation (explicit pairwise Mann-Whitney) and a different bootstrap implementation
   (speaker multiplicity WEIGHTS instead of concatenated resamples; same default_rng(0) draws).
2. perp_z / keep_z / oof re-derived from the states with a differently written probe loop
   (sklearn Pipeline) and the verifier-style 12-way Yes/No-softmax direction.
3. Participant 172 sensitivity: pitt_states.npz groups speakers as 'Control172' and 'Dementia172'
   (one participant, two diagnosis groups). Regroup by the manifest's participant id (227 groups)
   and report the headline values again.
Writes only part17/T3_selfcheck.json and .log. ZERO writes under omni_final.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys, json, time, hashlib, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
T0 = time.time()
OUT = "<local data dir>/release/edaic_rerun/part17"
PERCLIP = f"{OUT}/T3_q2a_direction.csv"; SUMCSV = f"{OUT}/T3_q2a_direction_summary.csv"
NPZ = "<local data dir>/paper work/paper1_local_runs/probe2/pitt_states.npz"
MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
SHARD = ("<hf cache>/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/"
         "0a095220c30b7b31434169c3086508ef3ea5bf0a/model-00005-of-00005.safetensors")
YES = [7414, 9454, 9693, 9834, 14004, 14080]; NO = [902, 2152, 2308, 2753, 5664, 8996]
LOGP = f"{OUT}/T3_selfcheck.log"; JS = f"{OUT}/T3_selfcheck.json"
assert "omni_final" not in LOGP + JS
lf = open(LOGP, "w")
def log(*a):
    s = " ".join(map(str, a)); print(s, flush=True); lf.write(s + "\n"); lf.flush()
def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read(1 << 20)); return h.hexdigest()

def auc_mw(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    a, b = s[y == 1], s[y == 0]
    if a.size == 0 or b.size == 0: return float("nan")
    return float(((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (a.size * b.size))

def auc_w(y, s, w):
    """weighted Mann-Whitney: clip i counted w_i times."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float); w = np.asarray(w, float)
    k = w > 0; y, s, w = y[k], s[k], w[k]
    a, wa, b, wb = s[y == 1], w[y == 1], s[y == 0], w[y == 0]
    if wa.sum() == 0 or wb.sum() == 0: return float("nan")
    W = wa[:, None] * wb[None, :]
    return float(((a[:, None] > b[None, :]) * W).sum() + 0.5 * ((a[:, None] == b[None, :]) * W).sum()) / (wa.sum() * wb.sum())

def boot_w(y, s, spk):
    rng = np.random.default_rng(0); us = np.unique(spk); pos = {u: i for i, u in enumerate(us)}
    code = np.array([pos[v] for v in spk]); v = []
    for _ in range(2000):
        pick = rng.choice(us, size=len(us), replace=True)
        cnt = np.bincount(np.array([pos[p] for p in pick]), minlength=len(us))
        a = auc_w(y, s, cnt[code])
        if a == a: v.append(a)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

def boot_w_paired(y, s, spk, arm):
    rng = np.random.default_rng(0); us = np.unique(spk); pos = {u: i for i, u in enumerate(us)}
    code = np.array([pos[v] for v in spk]); v = []
    c, g = arm == "conflict", arm == "agreement"
    for _ in range(2000):
        cnt = np.bincount(np.array([pos[p] for p in rng.choice(us, size=len(us), replace=True)]), minlength=len(us))
        w = cnt[code]; a = auc_w(y[c], s[c], w[c]); b = auc_w(y[g], s[g], w[g])
        if a == a and b == b: v.append(a - b)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

# ---------------------------------------------------------------- 1. summary vs per-clip
pc = pd.read_csv(PERCLIP); sm = pd.read_csv(SUMCSV).set_index("id")
y = pc["label"].values.astype(int); spk = pc["spk"].astype(str).values; arm = pc["arm"].values
masks = {"all": np.ones(len(pc), bool), "conflict": arm == "conflict", "agreement": arm == "agreement"}
colmap = {"probe": "oof_probe", "proj_d": "proj_d", "perp_z": "perp_z", "keep_z": "keep_z"}
checks = []
def chk(cid, v, lo, hi):
    r = sm.loc[cid]
    for nm, mine, theirs in (("value", v, r["value"]), ("lo", lo, r["lo"]), ("hi", hi, r["hi"])):
        if mine is None: continue
        ok = f"{mine:.4f}" == f"{float(theirs):.4f}"
        checks.append(dict(id=cid, field=nm, selfcheck=f"{mine:.4f}", main=f"{float(theirs):.4f}", ok=ok))
for cell, m in masks.items():
    for key, col in colmap.items():
        v = auc_mw(y[m], pc[col].values[m]); lo, hi, _ = boot_w(y[m], pc[col].values[m], spk[m])
        chk(f"T3_{cell}_{key}", v, lo, hi)
    for key, col in (("probe", "nested_oof"), ("perp_z", "nested_perp_z"), ("keep_z", "nested_keep_z")):
        v = auc_mw(y[m], pc[col].values[m]); lo, hi, _ = boot_w(y[m], pc[col].values[m], spk[m])
        chk(f"T3_nested_{cell}_{key}", v, lo, hi)
for cell in ("conflict", "agreement"):
    m = masks[cell]
    for key, col in (("probe", "oof_probe_armrefit"), ("proj_d", "proj_d_armrefit"), ("perp_z", "perp_z_armrefit")):
        v = auc_mw(y[m], pc[col].values[m]); lo, hi, _ = boot_w(y[m], pc[col].values[m], spk[m])
        chk(f"T3_{cell}_refit_{key}", v, lo, hi)
for key, col in (("perp_z", "perp_z"), ("probe", "oof_probe"), ("proj_d", "proj_d")):
    s = pc[col].values; c, g = masks["conflict"], masks["agreement"]
    v = auc_mw(y[c], s[c]) - auc_mw(y[g], s[g]); lo, hi, _ = boot_w_paired(y, s, spk, arm)
    chk(f"T3_paired_{key}", v, lo, hi)
s = pc["nested_perp_z"].values; c, g = masks["conflict"], masks["agreement"]
v = auc_mw(y[c], s[c]) - auc_mw(y[g], s[g]); lo, hi, _ = boot_w_paired(y, s, spk, arm)
chk("T3_nested_paired_perp_z", v, lo, hi)
nbad = sum(not c["ok"] for c in checks)
log(f"[1] summary AUCs/intervals re-derived from per-clip csv (pairwise MW + weighted bootstrap): "
    f"{len(checks) - nbad}/{len(checks)} identical at 4 dp")
for c in checks:
    if not c["ok"]: log(f"   XX {c}")

# ---------------------------------------------------------------- 2. re-derive columns from the states
z = np.load(NPZ, allow_pickle=True)
X = z["ans"][:, -1, :].astype(np.float64); name = z["name"].astype(str)
assert (name == pc["name"].values).all() and (z["label"].astype(int) == y).all() and (z["spk"].astype(str) == spk).all()
with open(SHARD, "rb") as fh:
    hl = int.from_bytes(fh.read(8), "little"); hd = json.loads(fh.read(hl))
    inf = hd["language_model.lm_head.weight"]; ncol = inf["shape"][1]; base = 8 + hl + inf["data_offsets"][0]
    rows = []
    for i in YES + NO:
        fh.seek(base + i * ncol * 2)
        u = np.frombuffer(fh.read(ncol * 2), dtype="<u2").astype(np.uint32) << np.uint32(16)
        rows.append(u.view(np.float32).astype(np.float64))
Wr = np.stack(rows); WY, WN = Wr[:6], Wr[6:]
Lg = X @ Wr.T; P12 = np.exp(Lg - Lg.max(1, keepdims=True)); P12 /= P12.sum(1, keepdims=True); mp = P12.mean(0)
d = (mp[:6] / mp[:6].sum()) @ WY - (mp[6:] / mp[6:].sum()) @ WN; d /= np.linalg.norm(d)
log(f"[2] max|X.d12 - proj_d column| = {np.abs(X @ d - pc['proj_d'].values).max():.3e}")

def rederive(groups):
    oof = np.zeros(len(y)); pz = np.zeros(len(y)); kz = np.zeros(len(y)); W = []; fold = np.zeros(len(y), int)
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups)):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
        w = pipe[-1].coef_[0] / pipe[0].scale_; W.append(w)
        r = X[te] @ (w - (w @ d) * d); q = X[te] @ w
        pz[te] = (r - r.mean()) / r.std(); kz[te] = (q - q.mean()) / q.std(); fold[te] = k
    wd = np.mean(W, 0); return oof, pz, kz, float(wd @ d / np.linalg.norm(wd)), fold
oof2, pz2, kz2, cos2, fold2 = rederive(spk)
log(f"[2] re-derived vs per-clip columns, max|diff|: oof {np.abs(oof2 - pc['oof_probe']).max():.3e}  "
    f"perp_z {np.abs(pz2 - pc['perp_z']).max():.3e}  keep_z {np.abs(kz2 - pc['keep_z']).max():.3e}  "
    f"fold mismatches {(fold2 != pc['fold'].values).sum()}  cos {cos2:.4f}")

# ---------------------------------------------------------------- 3. participant 172 sensitivity
man = pd.read_csv(MAN, dtype={"spk": str}); man["base"] = man["segment_path"].map(os.path.basename)
pid = dict(zip(man["base"], man["spk"])); part = np.array([pid[n] for n in name])
dup = sorted(set(p for p in part if len(set(spk[part == p])) > 1))
where = {str(p): {str(s_): sorted(set(pc["fold"].values[spk == s_].tolist())) for s_ in sorted(set(spk[part == p]))} for p in dup}
log(f"[3] participants carrying two npz speaker ids: {dup}; main-run folds: {where}")
oof3, pz3, kz3, cos3, _ = rederive(part)
sens = {"n_groups": int(len(np.unique(part))), "cos": cos3}
for cell, m in masks.items():
    for key, sc_ in (("probe", oof3), ("perp_z", pz3), ("proj_d", pc["proj_d"].values)):
        v = auc_mw(y[m], sc_[m]); lo, hi, n_ = boot_w(y[m], sc_[m], part[m])
        sens[f"{cell}_{key}"] = (v, lo, hi, n_, int(m.sum()), int(len(np.unique(part[m]))))
        log(f"[3] grouped by participant ({len(np.unique(part[m]))} groups) {cell} {key}: {v:.4f} [{lo:.4f}, {hi:.4f}] n={int(m.sum())}")
c, g = masks["conflict"], masks["agreement"]
v = auc_mw(y[c], pz3[c]) - auc_mw(y[g], pz3[g]); lo, hi, n_ = boot_w_paired(y, pz3, part, arm)
sens["paired_perp_z"] = (v, lo, hi, n_)
log(f"[3] grouped by participant, paired conflict-agreement perp_z: {v:.4f} [{lo:.4f}, {hi:.4f}]; cos {cos3:.4f}")
SENS_CSV = f"{OUT}/T3_selfcheck_participant_grouped.csv"
pd.DataFrame(dict(name=name, spk=spk, participant=part, label=y, arm=arm, oof_probe=oof3, proj_d=pc["proj_d"].values,
                  perp_z=pz3, keep_z=kz3)).to_csv(SENS_CSV, index=False, float_format="%.10g")
log(f"[3] per-clip scores of the participant-grouped run: {SENS_CSV}")

# ---------------------------------------------------------------- 4. fold-partition sensitivity
# 50 random speaker partitions, grouped by participant (no leakage): GroupKFold(5, shuffle=True, random_state=r)
def run_part(r):
    oof = np.zeros(len(y)); pz = np.zeros(len(y))
    for tr, te in GroupKFold(5, shuffle=True, random_state=r).split(X, y, part):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
        w = pipe[-1].coef_[0] / pipe[0].scale_; b = X[te] @ (w - (w @ d) * d); pz[te] = (b - b.mean()) / b.std()
    return dict(all_probe=auc_mw(y, oof), all_perp_z=auc_mw(y, pz),
                conflict_probe=auc_mw(y[c], oof[c]), conflict_perp_z=auc_mw(y[c], pz[c]),
                agreement_probe=auc_mw(y[g], oof[g]), agreement_perp_z=auc_mw(y[g], pz[g]),
                paired_perp_z=auc_mw(y[c], pz[c]) - auc_mw(y[g], pz[g]))
PARTS = [run_part(r) for r in range(50)]
pdist = {}
for k in PARTS[0]:
    a = np.array([q[k] for q in PARTS])
    pdist[k] = dict(median=float(np.median(a)), p2_5=float(np.percentile(a, 2.5)), p97_5=float(np.percentile(a, 97.5)),
                    min=float(a.min()), max=float(a.max()))
    log(f"[4] 50 random participant partitions, {k}: median {np.median(a):.4f}, 2.5-97.5% [{np.percentile(a, 2.5):.4f}, "
        f"{np.percentile(a, 97.5):.4f}], min {a.min():.4f}, max {a.max():.4f}")
PART_CSV = f"{OUT}/T3_selfcheck_partitions.csv"
pd.DataFrame(PARTS).assign(random_state=range(50)).to_csv(PART_CSV, index=False, float_format="%.10g")
log(f"[4] per-partition AUCs: {PART_CSV}")

json.dump(dict(command=" ".join([sys.executable, os.path.abspath(sys.argv[0])]),
               sources={"per_clip": PERCLIP, "summary": SUMCSV, "states": NPZ, "manifest": MAN, "shard": SHARD},
               sha256_first_1MB={"per_clip": sha(PERCLIP), "summary": sha(SUMCSV), "states": sha(NPZ),
                                 "manifest": sha(MAN), "shard": sha(SHARD)},
               n=int(len(y)), n_speakers=int(len(np.unique(spk))), seed=0, draws=2000,
               model_id="Qwen/Qwen2-Audio-7B-Instruct",
               checks=checks, n_disagree=nbad,
               rederive={"oof_maxdiff": float(np.abs(oof2 - pc['oof_probe']).max()),
                         "perp_z_maxdiff": float(np.abs(pz2 - pc['perp_z']).max()),
                         "keep_z_maxdiff": float(np.abs(kz2 - pc['keep_z']).max()), "cos": cos2},
               participant_172={"participants_with_two_ids": dup, "main_run_folds": where, "sensitivity": sens, "per_clip": SENS_CSV},
               partition_sensitivity={"what": "50 partitions GroupKFold(5, shuffle=True, random_state=0..49) by participant; AUC of pooled OOF", "dist": pdist, "per_partition_csv": PART_CSV},
               runtime_s=round(time.time() - T0, 1)), open(JS, "w"), indent=1, default=float)
log(f"wrote {JS}  runtime {time.time() - T0:.1f}s")
