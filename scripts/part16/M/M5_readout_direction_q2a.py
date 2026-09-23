"""PART 16 M5 (old Part D): the readout-direction test on Qwen2-Audio.

Mirrors p15/direction_q3o.py exactly, swapping the backbone to Qwen2-Audio-7B-Instruct.
States are local (probe2/pitt_states.npz); lm_head comes from the cached HF snapshot,
read straight out of the safetensors shard (no 16GB model instantiation needed).

Nothing is fabricated: every number printed is computed here from files on disk.
"""
import os, sys, json, csv, time, hashlib, warnings
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

T0 = time.time()
NPZ  = "<local data dir>/paper1_local_runs/probe2/pitt_states.npz"
MAN  = "manifests/pitt_conflict_manifest_468.csv"
SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/"
                          "snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a")
SHARD = SNAP + "/model-00005-of-00005.safetensors"
MID  = "Qwen/Qwen2-Audio-7B-Instruct"
OUT  = "<local data dir>/Desktop/release/edaic_rerun/part16"
DISC = "<local data dir>/Desktop/release/edaic_rerun/DISCREPANCIES.md"
PROMPT = ("Based only on this recording, does this speaker show signs of dementia? "
          "Answer with one word, Yes or No.")
SEED, DRAWS = 0, 2000

def log(*a): print(*a, flush=True)

def sha1mb(p):
    h = hashlib.sha256()
    with open(p, "rb") as f: h.update(f.read(1 << 20))
    return h.hexdigest()

def disc(sev, text):
    os.makedirs(os.path.dirname(DISC), exist_ok=True)
    with open(DISC, "a") as f:
        f.write(f"\n- **[{sev}] PART16 M5 ({time.strftime('%Y-%m-%d %H:%M')})** {text}\n")

# ---------------- rule 5: rank-formula AUC ----------------
def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = st.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

# ---------------- rule 6: speaker bootstrap ----------------
def boot_ci(y, s, spk, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s, float); spk = np.asarray(spk)
    us = np.unique(spk); idx = {u: np.where(spk == u)[0] for u in us}
    vals = []
    for _ in range(draws):
        pick = rng.choice(us, size=len(us), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        v = auc(y[ii], s[ii])
        if v == v: vals.append(v)
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

def boot_ci_paired_diff(y, s, spk, arm, a="conflict", b="agreement", draws=DRAWS, seed=SEED):
    """PAIRED: one speaker list per replicate, BOTH arm AUCs recomputed inside that draw."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s, float); spk = np.asarray(spk); arm = np.asarray(arm)
    us = np.unique(spk); idx = {u: np.where(spk == u)[0] for u in us}
    vals = []
    for _ in range(draws):
        pick = rng.choice(us, size=len(us), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        ma = arm[ii] == a; mb = arm[ii] == b
        va = auc(y[ii][ma], s[ii][ma]); vb = auc(y[ii][mb], s[ii][mb])
        if va == va and vb == vb: vals.append(va - vb)
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

# ---------------- load states ----------------
z = np.load(NPZ, allow_pickle=True)
ans = z["ans"]; y = z["label"].astype(int); spk = z["spk"].astype(str); name = z["name"].astype(str)
p_yes_saved = z["p_yes"].astype(float)
X = ans[:, -1, :].astype(np.float64)          # final LM stage at the first answer position
N, D = X.shape
log(f"[states] {NPZ}")
log(f"[states] ans {ans.shape} -> X {X.shape}  (stage {ans.shape[1]-1} of {ans.shape[1]})  "
    f"n={N} n_spk={len(np.unique(spk))} pos={int(y.sum())} neg={int((y==0).sum())}")

# ---------------- arms ----------------
man = pd.read_csv(MAN, dtype={"spk": str})
man["base"] = man["segment_path"].str.rsplit("/", n=1).str[-1]
arm_of = dict(zip(man["base"], man["set"])); lab_of = dict(zip(man["base"], man["label"].astype(int)))
assert set(name) == set(man["base"]), "npz names != manifest basenames"
arm = np.array([arm_of[n] for n in name])
assert (np.array([lab_of[n] for n in name]) == y).all(), "label mismatch npz vs manifest"
mc, ma = arm == "conflict", arm == "agreement"
log(f"[arms] {MAN}  conflict={int(mc.sum())} ({len(np.unique(spk[mc]))} spk)  "
    f"agreement={int(ma.sum())} ({len(np.unique(spk[ma]))} spk)")

# ---------------- rule 7: folds ----------------
FOLD_NOTE = ("no saved fold file found under <local data dir>/Desktop/release or "
             "<local data dir>/paper1_local_runs (searched *fold*.csv/json/npz and the fold column "
             "of every *_oof.csv in probe2/ and part15/ - none carry one); folds recomputed "
             "deterministically with GroupKFold(n_splits=5) grouped by speaker on the clips in npz order, "
             "which is the same scheme p15/direction_q3o.py and part15/H_readout_arms.py used, so the three "
             "backbones stay comparable")
log(f"[folds] {FOLD_NOTE}")
disc("MEDIUM", "No saved fold assignment file exists for Pitt. " + FOLD_NOTE +
     ". Affects M5_readout_direction_q2a.csv (auc_probe, auc_probe_without_d and both arm variants).")

# ---------------- lm_head ----------------
from safetensors import safe_open
from transformers import AutoTokenizer
with safe_open(SHARD, "pt") as f:
    keys = list(f.keys())
    assert keys == ["language_model.lm_head.weight"], keys
    W = f.get_tensor("language_model.lm_head.weight").float().numpy()
cfg = json.load(open(SNAP + "/config.json"))
TIE = cfg.get("tie_word_embeddings", cfg.get("text_config", {}).get("tie_word_embeddings", None))
WHICH = ("untied language_model.lm_head.weight, read directly from "
         "model-00005-of-00005.safetensors (the shard holds that one tensor and nothing else); "
         f"config carries no tie_word_embeddings key (value={TIE}), and a distinct lm_head tensor is "
         "present in the checkpoint index, so the head is NOT tied to the input embedding")
log(f"[lm_head] {WHICH}")
log(f"[lm_head] W {W.shape} from {SHARD}")
assert W.shape[1] == D

tok = AutoTokenizer.from_pretrained(MID, local_files_only=True)
def ids(ws):
    out = []
    for w in ws:
        for form in (w, " " + w):
            t = tok.encode(form, add_special_tokens=False)
            if len(t) == 1: out.append(t[0])
    return sorted(set(out))
YES, NO = ids(["Yes", "yes", "YES"]), ids(["No", "no", "NO"])
log(f"[tokens] yes ids {YES}")
log(f"[tokens] no  ids {NO}")

# ---------------- mass-weighted readout direction ----------------
# softmax(X @ W.T) then take the Yes / No columns. Done as a chunked logsumexp so the
# full 468 x 156032 float64 logit block never has to be held twice; identical result.
lse = np.full(N, -np.inf)
CH = 16384
for a0 in range(0, W.shape[0], CH):
    blk = X @ W[a0:a0 + CH].astype(np.float64).T
    m = blk.max(axis=1)
    cur = m + np.log(np.exp(blk - m[:, None]).sum(axis=1))
    lse = np.logaddexp(lse, cur)
    del blk
sc_yes = np.exp(X @ W[YES].astype(np.float64).T - lse[:, None])
sc_no  = np.exp(X @ W[NO].astype(np.float64).T  - lse[:, None])
log(f"[softmax] mean total Yes mass {sc_yes.sum(1).mean():.6f}  mean total No mass {sc_no.sum(1).mean():.6f}")

def dvec_from(rows_mask):
    wy = sc_yes[rows_mask].mean(0); wn = sc_no[rows_mask].mean(0)
    wy = wy / max(wy.sum(), 1e-12); wn = wn / max(wn.sum(), 1e-12)
    d = (W[YES].astype(np.float64).T @ wy) - (W[NO].astype(np.float64).T @ wn)
    return d / np.linalg.norm(d)

ALL = np.ones(N, bool)
dvec = dvec_from(ALL)

# ---------------- probe ----------------
def run_probe(Xm, ym, spkm):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(ym)); folds = np.full(len(ym), -1); ws = []
    for k, (tr, te) in enumerate(gkf.split(Xm, ym, groups=spkm)):
        assert not (set(spkm[tr]) & set(spkm[te])), "speaker leak"
        ss = StandardScaler().fit(Xm[tr])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(ss.transform(Xm[tr]), ym[tr])
        oof[te] = lr.predict_proba(ss.transform(Xm[te]))[:, 1]
        folds[te] = k
        ws.append(lr.coef_[0] / ss.scale_)
    w = np.mean(ws, axis=0); w = w / np.linalg.norm(w)
    return oof, w, folds

def probe_without_d_coef(Xm, ym, spkm, d):
    """q3o method: project d out of the fold COEFFICIENT, score with the residual weights."""
    gkf = GroupKFold(n_splits=5); oof2 = np.zeros(len(ym))
    for tr, te in gkf.split(Xm, ym, groups=spkm):
        ss = StandardScaler().fit(Xm[tr]); Xt = ss.transform(Xm[tr]); Xe = ss.transform(Xm[te])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xt, ym[tr])
        wf = lr.coef_[0]; wp = wf - (wf @ d) / (d @ d) * d
        oof2[te] = (Xe @ wp - (Xt @ wp).mean()) / ((Xt @ wp).std() + 1e-12)
    return oof2

def probe_without_d_X(Xm, ym, spkm, d):
    """literal reading: project d out of X, then re-run the probe from scratch."""
    Xp = Xm - np.outer(Xm @ d, d)
    return run_probe(Xp, ym, spkm)[0]

def rand_floor(w, dim, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed); r = []
    for _ in range(draws):
        v = rng.normal(size=dim); v /= np.linalg.norm(v); r.append(abs(float(w @ v)))
    r = np.array(r)
    return float(r.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), len(r)

def cell(tag, m, d):
    Xm, ym, sm = X[m], y[m], spk[m]
    oof, w, folds = run_probe(Xm, ym, sm)
    cos = float(w @ d)
    rmean, rlo, rhi, rused = rand_floor(w, D)
    s_d = Xm @ d
    oof_wo   = probe_without_d_coef(Xm, ym, sm, d)
    oof_wo_X = probe_without_d_X(Xm, ym, sm, d)
    r = dict(cell=tag, n=int(m.sum()), n_spk=int(len(np.unique(sm))),
             cos=cos, rand_mean_abs=rmean, rand_lo=rlo, rand_hi=rhi, rand_used=rused,
             auc_probe=auc(ym, oof), auc_d=auc(ym, s_d),
             auc_without_d=auc(ym, oof_wo), auc_without_d_Xproj=auc(ym, oof_wo_X),
             fold_sizes=np.bincount(folds).tolist(),
             spk_per_fold=[int(len(set(sm[folds == k]))) for k in range(5)])
    for k, s in (("auc_probe", oof), ("auc_d", s_d), ("auc_without_d", oof_wo),
                 ("auc_without_d_Xproj", oof_wo_X)):
        lo, hi, used = boot_ci(ym, s, sm)
        r[k + "_lo"], r[k + "_hi"], r[k + "_draws"] = lo, hi, used
    return r, oof, s_d, oof_wo, oof_wo_X

log("\n[run] all 468 ...")
R_all, oof_all, sd_all, oofwo_all, oofwoX_all = cell("all", ALL, dvec)

# arm cells: the FULL pipeline refit inside each arm, with that arm's own mass-weighted direction
log("[run] conflict arm 146 (own direction, own folds) ...")
d_c = dvec_from(mc); R_c, oof_c, sd_c, oofwo_c, oofwoX_c = cell("conflict_refit", mc, d_c)
log("[run] agreement arm 322 (own direction, own folds) ...")
d_a = dvec_from(ma); R_a, oof_a, sd_a, oofwo_a, oofwoX_a = cell("agreement_refit", ma, d_a)
log(f"[dirs] cos(d_all,d_conflict)={float(dvec@d_c):.4f}  cos(d_all,d_agreement)={float(dvec@d_a):.4f}  "
    f"cos(d_conflict,d_agreement)={float(d_c@d_a):.4f}")

# arm cells: the ALL-data probe / direction simply RESTRICTED to each arm (continuity with part15 H)
def restrict(tag, m):
    r = dict(cell=tag, n=int(m.sum()), n_spk=int(len(np.unique(spk[m]))),
             cos="", rand_mean_abs="", rand_lo="", rand_hi="", rand_used="",
             auc_probe=auc(y[m], oof_all[m]), auc_d=auc(y[m], sd_all[m]),
             auc_without_d=auc(y[m], oofwo_all[m]), auc_without_d_Xproj=auc(y[m], oofwoX_all[m]),
             fold_sizes="", spk_per_fold="")
    for k, s in (("auc_probe", oof_all), ("auc_d", sd_all), ("auc_without_d", oofwo_all),
                 ("auc_without_d_Xproj", oofwoX_all)):
        lo, hi, used = boot_ci(y[m], s[m], spk[m])
        r[k + "_lo"], r[k + "_hi"], r[k + "_draws"] = lo, hi, used
    return r
R_cr = restrict("conflict_restricted", mc)
R_ar = restrict("agreement_restricted", ma)

# paired conflict-minus-agreement on the all-data scores
PD = {}
for k, s in (("auc_probe", oof_all), ("auc_d", sd_all), ("auc_without_d", oofwo_all)):
    lo, hi, used = boot_ci_paired_diff(y, s, spk, arm)
    PD[k] = dict(diff=auc(y[mc], s[mc]) - auc(y[ma], s[ma]), lo=lo, hi=hi, draws=used)

# sanity: direction score vs the scorer's own saved p_yes
rho = float(st.spearmanr(sd_all, p_yes_saved).statistic)
auc_pyes = auc(y, p_yes_saved)
log(f"[sanity] spearman(X@dvec, saved p_yes) = {rho:.4f}   auc(saved p_yes) = {auc_pyes:.4f}")

# ---------------- outputs ----------------
ROWS = [R_all, R_c, R_a, R_cr, R_ar]
per = pd.DataFrame(dict(
    clip=name, speaker=spk, label=y, arm=arm,
    p_probe_oof_all=oof_all, score_along_d_all=sd_all,
    z_probe_oof_without_d_all=oofwo_all, p_probe_oof_without_d_Xproj_all=oofwoX_all,
    p_yes_scorer_saved=p_yes_saved))
per["p_probe_oof_armrefit"] = np.nan
per.loc[mc, "p_probe_oof_armrefit"] = oof_c
per.loc[ma, "p_probe_oof_armrefit"] = oof_a
per["score_along_d_armrefit"] = np.nan
per.loc[mc, "score_along_d_armrefit"] = sd_c
per.loc[ma, "score_along_d_armrefit"] = sd_a
# the arm-refit residual scores too, so a second check can reproduce EVERY row of rows/M5.tsv
# from this one per-clip csv with independent code
per["z_probe_oof_without_d_armrefit"] = np.nan
per.loc[mc, "z_probe_oof_without_d_armrefit"] = oofwo_c
per.loc[ma, "z_probe_oof_without_d_armrefit"] = oofwo_a
per["p_probe_oof_without_d_Xproj_armrefit"] = np.nan
per.loc[mc, "p_probe_oof_without_d_Xproj_armrefit"] = oofwoX_c
per.loc[ma, "p_probe_oof_without_d_Xproj_armrefit"] = oofwoX_a
PERP = os.path.join(OUT, "M5_readout_direction_q2a.csv")
per.to_csv(PERP, index=False)

SUMP = os.path.join(OUT, "M5_readout_direction_q2a_summary.csv")
pd.DataFrame(ROWS).to_csv(SUMP, index=False)

side = {
 "part": "16 M5 (old Part D)", "date": time.strftime("%Y-%m-%d"),
 "question": "readout-direction test on Qwen2-Audio, to sit beside Qwen2.5-Omni and Qwen3-Omni",
 "model_id": MID, "lm_head_source": SHARD, "lm_head_tied": False, "which_head_used": WHICH,
 "lm_head_shape": list(W.shape), "vocab_size": int(W.shape[0]), "hidden": int(D),
 "prompt": PROMPT,
 "prompt_note": ("the prompt that produced the states, from probe2/extract2.py line 20, condition 'ad'. "
                 "This run does not re-run the model; it reuses those states."),
 "yes_ids": YES, "no_ids": NO,
 "token_id_note": ("ids built as direction_q3o.py does: ['Yes','yes','YES'] and ['No','no','NO'], each in bare "
                   "and leading-space form, single-token encodings only, deduped and sorted. The original "
                   "extractor extract2.py used a slightly wider list and took t[0] of any encoding; that "
                   "affects the saved p_yes column, not the direction computed here."),
 "source_files": {"states": NPZ, "manifest": MAN, "hf_snapshot": SNAP, "shard": SHARD,
                  "reference_q3o": "<local data dir>/scratch/"
                                   "p15/direction_q3o.py"},
 "sha256_first_1MB": {"states": sha1mb(NPZ), "manifest": sha1mb(MAN), "shard": sha1mb(SHARD),
                      "config": sha1mb(SNAP + "/config.json")},
 "n": N, "n_speakers": int(len(np.unique(spk))),
 "n_conflict": int(mc.sum()), "n_speakers_conflict": int(len(np.unique(spk[mc]))),
 "n_agreement": int(ma.sum()), "n_speakers_agreement": int(len(np.unique(spk[ma]))),
 "seed": SEED, "draws": DRAWS,
 "bootstrap": {"draws": DRAWS, "rng": "numpy.random.default_rng(0), reseeded per cell",
               "unit": "speaker, resampled with replacement (never clips)",
               "percentiles": [2.5, 97.5], "refit_per_draw": False,
               "paired": "conflict-minus-agreement draws the speaker list ONCE per replicate and "
                         "recomputes BOTH arm AUCs inside that draw"},
 "folds": "GroupKFold(n_splits=5) grouped by speaker, clips in npz order; scaler fit inside the training fold only",
 "folds_note": FOLD_NOTE,
 "features": "ans[:, -1, :] from pitt_states.npz, the final LM stage at the first answer position",
 "direction": "mass-weighted Yes minus No lm_head rows, normalised to unit length",
 "auc": "recomputed from the per-clip scores with the rank formula (scipy rankdata), never copied from a json",
 "auc_without_d_definition": ("auc_without_d is the q3o definition (project d out of the fold COEFFICIENT) so "
                              "the three backbones are comparable; auc_without_d_Xproj is the literal reading "
                              "(project d out of X, re-run the probe) and is reported beside it"),
 "arm_note": ("*_refit rows run the whole pipeline inside the arm, with that arm's own mass-weighted direction "
              "and its own GroupKFold(5); *_restricted rows are the all-468 probe and direction evaluated on "
              "that arm's clips only, which is what part15/H_readout_arms.csv reports"),
 "direction_agreement": {"cos_all_conflict": round(float(dvec @ d_c), 4),
                         "cos_all_agreement": round(float(dvec @ d_a), 4),
                         "cos_conflict_agreement": round(float(d_c @ d_a), 4)},
 "sanity": {"spearman_score_along_d_vs_saved_p_yes": round(rho, 4),
            "auc_saved_p_yes_zeroshot": round(auc_pyes, 4)},
 "paired_conflict_minus_agreement": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                         for kk, vv in v.items()} for k, v in PD.items()},
 "results": ROWS,
 "outputs": {"per_clip_csv": PERP, "summary_csv": SUMP,
             "sidecar": os.path.join(OUT, "M5_readout_direction_q2a.json"),
             "rows_tsv": os.path.join(OUT, "rows", "M5.tsv")},
 "python": sys.version.split()[0],
 "command": f"/usr/local/bin/python3 {os.path.abspath(__file__)}",
 "runtime_s": round(time.time() - T0, 1),
}
SIDE = os.path.join(OUT, "M5_readout_direction_q2a.json")
json.dump(side, open(SIDE, "w"), indent=1, default=str)

TSV = os.path.join(OUT, "rows", "M5.tsv")
with open(TSV, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["id", "what", "value", "lo", "hi", "n", "n_spk", "file"])
    def put(i, what, v, lo, hi, n, ns): w.writerow([i, what, f"{v:.4f}",
        (f"{lo:.4f}" if lo != "" else ""), (f"{hi:.4f}" if hi != "" else ""), n, ns, PERP])
    for R in ROWS:
        t = R["cell"]
        if R["cos"] != "":
            put(f"M5_{t}_cos", f"Qwen2-Audio cos(probe_dir, readout_dir) [{t}]", R["cos"], "", "", R["n"], R["n_spk"])
            put(f"M5_{t}_randfloor", f"Qwen2-Audio random-direction floor mean|cos| [{t}]",
                R["rand_mean_abs"], R["rand_lo"], R["rand_hi"], R["n"], R["n_spk"])
        for k, lbl in (("auc_probe", "auc probe out-of-fold"), ("auc_d", "auc along readout direction"),
                       ("auc_without_d", "auc probe with readout direction removed (coef proj)"),
                       ("auc_without_d_Xproj", "auc probe with readout direction removed (X proj)")):
            put(f"M5_{t}_{k}", f"Qwen2-Audio {lbl} [{t}]", R[k], R[k + "_lo"], R[k + "_hi"], R["n"], R["n_spk"])
    for k, v in PD.items():
        put(f"M5_paired_{k}_conflict_minus_agreement",
            f"Qwen2-Audio {k} conflict minus agreement (paired speaker bootstrap)",
            v["diff"], v["lo"], v["hi"], N, len(np.unique(spk)))

# ---------------- print ----------------
log("")
log("=" * 118)
R = R_all
log("DIR n=%d cos=%.4f rand_mean_abs=%.4f rand_ci=[%.4f,%.4f] auc_probe=%.4f auc_d=%.4f auc_without_d=%.4f"
    % (R["n"], R["cos"], R["rand_mean_abs"], R["rand_lo"], R["rand_hi"],
       R["auc_probe"], R["auc_d"], R["auc_without_d"]))
log("")
log("Qwen2-Audio: cos=%.4f, rand mean|cos|=%.4f [%.4f,%.4f], auc_probe=%.4f, auc_d=%.4f, auc_without_d=%.4f "
    "| n=%d, n_speakers=%d | %s"
    % (R["cos"], R["rand_mean_abs"], R["rand_lo"], R["rand_hi"], R["auc_probe"], R["auc_d"],
       R["auc_without_d"], R["n"], R["n_spk"], PERP))
log("")
for R in ROWS:
    log("[%-21s] n=%3d n_spk=%3d  cos=%s  auc_probe=%.4f [%.4f,%.4f]  auc_d=%.4f [%.4f,%.4f]  "
        "auc_without_d=%.4f [%.4f,%.4f]  auc_without_d_Xproj=%.4f [%.4f,%.4f]"
        % (R["cell"], R["n"], R["n_spk"], (f"{R['cos']:.4f}" if R["cos"] != "" else "  n/a "),
           R["auc_probe"], R["auc_probe_lo"], R["auc_probe_hi"],
           R["auc_d"], R["auc_d_lo"], R["auc_d_hi"],
           R["auc_without_d"], R["auc_without_d_lo"], R["auc_without_d_hi"],
           R["auc_without_d_Xproj"], R["auc_without_d_Xproj_lo"], R["auc_without_d_Xproj_hi"]))
log("")
for k, v in PD.items():
    log("[paired] %-20s conflict-agreement = %+.4f [%+.4f,%+.4f]  usable draws %d/%d"
        % (k, v["diff"], v["lo"], v["hi"], v["draws"], DRAWS))
log("")
log(f"[draws] bootstrap usable draws, all-cell: " +
    ", ".join(f"{k}={R_all[k+'_draws']}/{DRAWS}" for k in
              ("auc_probe", "auc_d", "auc_without_d", "auc_without_d_Xproj")) +
    f", random-floor={R_all['rand_used']}/{DRAWS}")
log(f"[folds] all-cell fold sizes {R_all['fold_sizes']} speakers/fold {R_all['spk_per_fold']}")
log("")
log("WROTE " + PERP)
log("WROTE " + SUMP)
log("WROTE " + SIDE)
log("WROTE " + TSV)
log(f"[done] {time.time()-T0:.1f}s")
