#!/usr/local/bin/python3
"""Independent check of part25/T5 r2 outputs (T5_counts.tsv, T5_curve_match.tsv, perclip_r2, boot_r2).

Written separately from scripts/T5_counts_r2.py. Different code path:
  AUC by the Mann-Whitney rank formula (scipy.stats.rankdata, ties averaged), not sklearn;
  csv files read with pandas; speaker groups built with pandas groupby;
  bootstrap redrawn from a fresh numpy default_rng(0) per cell and compared draw by draw with the saved npz;
  refit counts recounted from the per-layer csv files found by glob on the pull folders;
  curve match recomputed against the saved curves;
  counts also compared with the first attempt (attempt1_2345Z) and the leftovers T5 verifier (t5_VERIFIED.tsv).
Writes only verify/T5_verify_r2.log and verify/T5_verify_r2.sidecar.json.
"""
import glob, hashlib, json, os, re, sys
from datetime import datetime, timezone
import numpy as np, pandas as pd
from scipy.stats import rankdata

T5D = "<local data dir>/release_from_mac/scores/part25/T5"
PULL = "<local data dir>/release_from_mac/scores/leftovers_23sep/t5/pull"
OF = "<local data dir>/release/omni_final"
LOG = open(f"{T5D}/verify/T5_verify_r2.log", "w")
n_ok = n_bad = 0
def chk(name, a, b, tol=0.0):
    global n_ok, n_bad
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        good = (np.isnan(a) and np.isnan(b)) or abs(float(a) - float(b)) <= tol
    else:
        good = str(a) == str(b)
    if good: n_ok += 1
    else: n_bad += 1
    line = f"{'OK ' if good else 'BAD'} {name}: t5={a} verify={b}"
    print(line, file=LOG, flush=True)
    if not good: print(line, flush=True)

def mw(lab, sc):
    lab = np.asarray(lab).astype(int); sc = np.asarray(sc, float)
    P = int(lab.sum()); N = len(lab) - P
    if P == 0 or N == 0: return np.nan
    r = rankdata(sc)
    return (r[lab == 1].sum() - P * (P + 1) / 2.0) / (P * N)

def parse_np(v):
    s = str(v)
    m = re.match(r"^np\.float64\((.*)\)$", s)
    return float(m.group(1)) if m else float(s)

T = pd.read_csv(f"{T5D}/T5_counts.tsv", sep="\t", dtype=str, keep_default_na=False)
print(f"rows {len(T)}", file=LOG)
info = []
for _, R in T.iterrows():
    ds = R["dataset"]
    zf, pc = R["answer_file"], R["answer_column"]
    Z = pd.read_csv(zf, dtype=str)
    spcol = "speaker" if "speaker" in Z.columns else "pid"
    lab = Z["label"].map(parse_np).astype(int).to_numpy()
    p = Z[pc].map(parse_np).to_numpy()
    spk = Z[spcol].astype(str).str.strip().to_numpy()
    chk(f"{ds} n_clips", int(R["n_clips"]), len(lab))
    chk(f"{ds} n_pos", int(R["n_pos"]), int(lab.sum()))
    ans = mw(lab, p)
    chk(f"{ds} answer_auc", float(R["answer_auc"]), ans, 1e-12)
    # bootstrap, own implementation
    groups = pd.Series(np.arange(len(spk))).groupby(spk).apply(lambda s: s.to_numpy())
    names = np.array(sorted(groups.index)); chk(f"{ds} n_speakers", int(R["n_speakers"]), len(names))
    rng = np.random.default_rng(0)
    B = np.load(R["boot_npz"], allow_pickle=True)
    ad = np.empty(2000); sidx_all = np.empty((2000, len(names)), dtype=np.int64)
    for b in range(2000):
        sidx = rng.choice(len(names), size=len(names), replace=True); sidx_all[b] = sidx
        ix = np.concatenate([groups[names[i]] for i in sidx])
        ad[b] = mw(lab[ix], p[ix])
    ok = ~np.isnan(ad)
    chk(f"{ds} boot usable", int(R["boot_usable"]), int(ok.sum()))
    lo, hi = np.percentile(ad[ok], [2.5, 97.5])
    chk(f"{ds} answer_lo", float(R["answer_lo"]), lo, 1e-12)
    chk(f"{ds} answer_hi", float(R["answer_hi"]), hi, 1e-12)
    chk(f"{ds} saved speaker idx equal", True, bool((B["speaker_idx"] == sidx_all).all()))
    chk(f"{ds} saved answer draws maxabs", 0.0, float(np.nanmax(np.abs(B["answer_draws"] - ad))), 1e-12)
    # saved curve counts
    S = pd.read_csv(R["saved_curve"])
    so = S["auc_oof"].to_numpy()
    chk(f"{ds} saved_above_answer", int(R["saved_above_answer"]), int((so > ans).sum()))
    chk(f"{ds} saved_above_hi", int(R["saved_above_hi"]), int((so > hi).sum()))
    chk(f"{ds} saved_min", float(R["saved_min"]), float(so.min()), 1e-15)
    chk(f"{ds} saved_max", float(R["saved_max"]), float(so.max()), 1e-15)
    # per clip csv answer column
    PC = pd.read_csv(R["perclip_csv"], dtype={"speaker": str, "clip": str})
    chk(f"{ds} perclip answer AUC", ans, mw(PC["label"], PC["p_yes_answer"]), 1e-12)
    if R["refit_status"] != "refit":
        chk(f"{ds} no refit files exist for this dataset in pull", 0,
            len([f for f in glob.glob(f"{PULL}/*/out/*perlayer.csv") if os.path.basename(f).startswith(ds)]))
        info.append(f"{ds}: no refit; saved {int((so > ans).sum())}/32 above answer {ans:.4f}, {int((so > hi).sum())}/32 above hi {hi:.4f}")
        continue
    rf = R["refit_file"]; pod = R["refit_pod"]
    job = os.path.basename(rf).replace("_encoder_perlayer.csv", "")
    C = pd.read_csv(rf); ro = C["auc_oof"].to_numpy(); rm = C["auc_mean"].to_numpy()
    chk(f"{ds} refit n_layers", 32, len(ro))
    chk(f"{ds} refit_above_answer", int(R["refit_above_answer"]), int((ro > ans).sum()))
    chk(f"{ds} refit_above_hi", int(R["refit_above_hi"]), int((ro > hi).sum()))
    chk(f"{ds} refit_mean_above_answer", int(R["refit_mean_above_answer"]), int((rm > ans).sum()))
    chk(f"{ds} refit_mean_above_hi", int(R["refit_mean_above_hi"]), int((rm > hi).sum()))
    chk(f"{ds} refit_not_above_answer", R["refit_not_above_answer"], ";".join(str(i) for i in range(32) if not ro[i] > ans))
    chk(f"{ds} refit_not_above_hi", R["refit_not_above_hi"], ";".join(str(i) for i in range(32) if not ro[i] > hi))
    chk(f"{ds} refit_min", float(R["refit_min"]), float(ro.min()), 1e-15)
    chk(f"{ds} refit_max", float(R["refit_max"]), float(ro.max()), 1e-15)
    chk(f"{ds} every_layer_on_refit", R["every_layer_on_refit"], "holds" if (ro > ans).all() else f"fails ({int((ro > ans).sum())}/32)")
    chk(f"{ds} refit_vs_saved_maxabs_oof", float(R["refit_vs_saved_maxabs_oof"]), float(np.abs(ro - so).max()), 1e-15)
    chk(f"{ds} refit_vs_saved_match4dp_oof", int(R["refit_vs_saved_match4dp_oof"]), int((np.round(ro, 4) == np.round(so, 4)).sum()))
    # every refit file of the job, same folds (exclude macfile), recount
    files = glob.glob(f"{PULL}/*/out/{job}_encoder_perlayer.csv") + glob.glob(f"{PULL}/*/out/{job}__*_enc.csv")
    same = [f for f in files if "__macfile_" not in f]; mac = [f for f in files if "__macfile_" in f]
    ca = [int((pd.read_csv(f)["auc_oof"].to_numpy() > ans).sum()) for f in same]
    ch = [int((pd.read_csv(f)["auc_oof"].to_numpy() > hi).sum()) for f in same]
    chk(f"{ds} range_above_answer_same_folds", R["range_above_answer_same_folds"], f"{min(ca)} to {max(ca)} over {len(ca)} files")
    chk(f"{ds} range_above_hi_same_folds", R["range_above_hi_same_folds"], f"{min(ch)} to {max(ch)} over {len(ch)} files")
    if mac:
        ma = [int((pd.read_csv(f)["auc_oof"].to_numpy() > ans).sum()) for f in mac]
        mh = [int((pd.read_csv(f)["auc_oof"].to_numpy() > hi).sum()) for f in mac]
        chk(f"{ds} range_above_answer_mac_folds", R["range_above_answer_mac_folds"], f"{min(ma)} to {max(ma)} over {len(ma)} files")
        chk(f"{ds} range_above_hi_mac_folds", R["range_above_hi_mac_folds"], f"{min(mh)} to {max(mh)} over {len(mh)} files")
    # per clip refit scores: pooled AUC equals the refit csv, and equals the vrefit (second code path) oof when present
    oofc = [c for c in PC.columns if c.startswith("refit_oof_enc_L")]
    chk(f"{ds} perclip refit columns", 32, len(oofc))
    pooled = np.array([mw(PC["label"], PC[c]) for c in oofc])
    chk(f"{ds} perclip pooled AUC vs refit csv maxabs", 0.0, float(np.abs(pooled - ro).max()), 1e-12)
    vr = f"{PULL}/{pod}/out/{job}__vrefit_enc.csv"
    if os.path.exists(vr):
        v = pd.read_csv(vr)["auc_oof"].to_numpy()
        info.append(f"{ds}: vrefit (second code path, same pod) auc_oof vs refit csv maxabs {np.abs(v - ro).max():.2e}; "
                    f"vrefit counts {int((v > ans).sum())}/32 above answer, {int((v > hi).sum())}/32 above hi")
    # paired delta: own recompute from per clip csv
    X = PC[oofc].to_numpy(); L = PC["label"].to_numpy()
    spk_pc = PC["speaker"].astype(str).to_numpy()
    chk(f"{ds} perclip speakers equal answer file", True, bool((spk_pc == spk).all()))
    pr = np.full((2000, 32), np.nan)
    for b in range(2000):
        ix = np.concatenate([groups[names[i]] for i in sidx_all[b]])
        if L[ix].min() != L[ix].max():
            pr[b] = [mw(L[ix], X[ix, l]) for l in range(32)]
    dl = pr - ad[:, None]
    chk(f"{ds} delta draws maxabs vs npz", 0.0, float(np.nanmax(np.abs(dl - B["delta_draws"]))), 1e-12)
    dlo = np.nanpercentile(dl, 2.5, axis=0); dhi = np.nanpercentile(dl, 97.5, axis=0)
    chk(f"{ds} paired_delta_lo_gt0", int(R["paired_delta_lo_gt0"]), int((dlo > 0).sum()))
    chk(f"{ds} paired_delta_hi_lt0", int(R["paired_delta_hi_lt0"]), int((dhi < 0).sum()))
    info.append(f"{ds}: refit {int((ro > ans).sum())}/32 above answer {ans:.4f}, {int((ro > hi).sum())}/32 above hi {hi:.4f}; "
                f"paired delta lower bound > 0 on {int((dlo > 0).sum())}/32 layers")

# curve match table
M = pd.read_csv(f"{T5D}/T5_curve_match.tsv", sep="\t")
for _, r in M.iterrows():
    a, b = pd.read_csv(r["refit_file"]), pd.read_csv(r["saved_file"])
    for col in ("auc_oof", "auc_mean", "auc_std"):
        d = np.abs(a[col].to_numpy() - b[col].to_numpy())
        chk(f"curve {r['curve']} {r['stream']} {r['pod']} {col} maxabs", float(r[f"{col}_maxabs"]), float(d.max()), 1e-15)
        chk(f"curve {r['curve']} {r['stream']} {r['pod']} {col} match4dp", int(r[f"{col}_match4dp"]),
            int((np.round(a[col].to_numpy(), 4) == np.round(b[col].to_numpy(), 4)).sum()))
# every exact-script curve in the pull folders is in the table
allcurves = sorted(glob.glob(f"{PULL}/*/out/*_*_perlayer.csv"))
chk("curve table covers every exact-script curve file", len(allcurves), len(M))

# compare with the first attempt and with the leftovers verifier
A1 = pd.read_csv(f"{T5D}/attempt1_2345Z/T5_counts.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("dataset")
MAP = {"neurovoz": "neurovoz", "kcl": "kcl", "adresso": "adresso", "adress2020": "adress2020",
       "edaic_full_o25": "edaic_full", "edaic_full_statesrun": "edaic_full_statesrun", "edaic_lifted_p22": "edaic_full_lifted",
       "edaic_first30": "edaic_first30", "pitt": "pitt", "pcgita": "pcgita"}
for _, R in T.iterrows():
    k = MAP.get(R["dataset"])
    if k is None or k not in A1.index: continue
    a = A1.loc[k]
    for col in ("answer_auc", "answer_lo", "answer_hi"):
        chk(f"attempt1 {k} {col}", float(a[col]), float(R[col]), 1e-12)
    for col in ("saved_above_answer", "saved_above_hi"):
        chk(f"attempt1 {k} {col}", a[col], R[col])
    if R["refit_status"] == "refit":
        for col in ("refit_above_answer", "refit_above_hi", "refit_mean_above_answer", "refit_mean_above_hi"):
            chk(f"attempt1 {k} {col}", a[col], R[col])
LV = pd.read_csv("scores/leftovers_23sep/verify/t5_VERIFIED.tsv", sep="\t")
lv = dict(zip(LV["id"], LV["recomputed"]))
TT = T.set_index("dataset")
chk("leftovers verifier adress2020 above answer 32/32", "32/32" in lv["T5_adress2020_enc_pod7"], TT.loc["adress2020", "refit_above_answer"] == "32")
chk("leftovers verifier neurovoz above answer 32/32", "32/32" in lv["T5_neurovoz_enc_pod8"], TT.loc["neurovoz", "refit_above_answer"] == "32")
chk("leftovers verifier edaic_full above answer 0/32", "0/32" in lv["T5_edaic_full_enc_pod9"], TT.loc["edaic_full_statesrun", "refit_above_answer"] == "0")
chk("leftovers verifier pitt above answer 31/32", "31/32" in lv["T5_pitt_enc_pod10_layers_above"], TT.loc["pitt", "refit_above_answer"] == "31")
chk("leftovers verifier edaic first30 above 2/32", "2/32" in lv["T5_edaic_first30_enc_match4dp"], TT.loc["edaic_first30", "refit_above_answer"] == "2")

# omni_final untouched
newest = max(os.path.getmtime(os.path.join(dp, f)) for dp, _, fs in os.walk(OF) for f in fs)
nd = datetime.fromtimestamp(newest, timezone.utc)
chk("omni_final newest mtime before this run (2026-09-24T02:40Z)", True, nd < datetime(2026, 9, 24, 2, 40, tzinfo=timezone.utc))
info.append(f"omni_final newest mtime {nd.isoformat()}")

print(f"TOTAL checks {n_ok + n_bad}, BAD {n_bad}", file=LOG, flush=True)
print(f"TOTAL checks {n_ok + n_bad}, BAD {n_bad}")
for s in info: print("INFO", s); print("INFO", s, file=LOG)
sh = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(dict(task="independent check of part25/T5 r2 outputs", date_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               script=os.path.abspath(__file__), script_sha256=sh(__file__),
               checked={p: sh(f"{T5D}/{p}") for p in ("T5_counts.tsv", "T5_curve_match.tsv", "T5_counts_sidecar.json")},
               result=f"TOTAL checks {n_ok + n_bad}, BAD {n_bad}", n_ok=n_ok, n_bad=n_bad, info=info),
          open(f"{T5D}/verify/T5_verify_r2.sidecar.json", "w"), indent=1)
