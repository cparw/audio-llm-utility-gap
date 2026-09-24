#!/usr/bin/env python3
# One RNG stream for the whole file: append new rows at the END; inserting a row shifts every interval after it.
"""Bootstrap intervals for every per-clip answer, probe and fix score the paper quotes.
2000 clip-level resamples, seed 0, percentile interval. Paired rows resample the same clips in
both arms and report the difference. Regenerates results/health/bootstrap_cis.csv in full."""
import csv, os, numpy as np
from sklearn.metrics import roc_auc_score as auc
D = os.path.expanduser("~/Desktop/Hard drive data for paper")
OUT = f"{D}/speech-health-repo/results/health/bootstrap_cis.csv"
rng = np.random.default_rng(0); B = 2000
def rd(p): return list(csv.DictReader(open(p)))
def col(rows, names):
    for nm in names:
        if nm in rows[0]: return np.array([float(r[nm]) for r in rows])
    raise KeyError(f"{names} not in {list(rows[0])}")
out = []
def ci(name, y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); n = len(y); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        v.append(auc(y[i], p[i]))
    out.append((name, n, auc(y, p), np.percentile(v, 2.5), np.percentile(v, 97.5)))
    print(f"{name}: n={n} {out[-1][2]:.3f} [{out[-1][3]:.3f}, {out[-1][4]:.3f}]", flush=True)
def paired(name, y, p1, p0):
    y, p1, p0 = map(lambda a: np.asarray(a, float), (y, p1, p0)); n = len(y); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        v.append(auc(y[i], p1[i]) - auc(y[i], p0[i]))
    d = auc(y, p1) - auc(y, p0)
    out.append((name, n, d, np.percentile(v, 2.5), np.percentile(v, 97.5)))
    print(f"{name}: n={n} {d:+.3f} [{out[-1][3]:+.3f}, {out[-1][4]:+.3f}]  P(diff<=0)={np.mean(np.array(v)<=0):.3f}", flush=True)
# Pitt (468 segments, manifest order)
man = [r for r in rd(f"{D}/DementiaBank/pitt_conflict_manifest.csv") if r.get("segment_path")]
y = np.array([int(r["label"]) for r in man]); paths = [r["segment_path"] for r in man]
arm = np.array([r["set"] for r in man])
amap = {r["segment_path"]: float(r["p_main"]) for r in rd(f"{D}/DementiaBank/ad_scores_qwen2audio.csv")}
p_ans = np.array([amap[p] for p in paths])
proj = np.load(f"{D}/DementiaBank/projector_retrain_oof.npy")
z = np.load(f"{D}/DementiaBank/ad_encoder_states.npz", allow_pickle=True)
pos = {str(p): i for i, p in enumerate(z["path"])}
probe = np.load(f"{D}/DementiaBank/ad_probe_oof.npy"); probe_m = np.array([probe[pos[p]] for p in paths])
assert (z["label"].astype(int)[[pos[p] for p in paths]] == y).all()
raw = {r["segment_path"]: float(r["p_yes"]) for r in rd(f"{D}/DementiaBank/ad_textonly_scores.csv")}
cln = {r["segment_path"]: float(r["p_yes"]) for r in rd(f"{D}/DementiaBank/ad_textonly_clean_scores.csv")}
p_raw = np.array([raw[p] for p in paths]); p_cln = np.array([cln[p] for p in paths])
ci("pitt answer", y, p_ans); ci("pitt projector retrained", y, proj); ci("pitt readout strict", y, probe_m)
ci("pitt transcript only (raw)", y, p_raw); ci("pitt transcript only (clean)", y, p_cln)
paired("pitt projector minus answer (paired)", y, proj, p_ans)
paired("pitt readout minus answer (paired)", y, probe_m, p_ans)
paired("pitt audio answer minus transcript only clean (paired)", y, p_ans, p_cln)
c = arm == "conflict"
ci("pitt answer, conflict arm", y[c], p_ans[c]); ci("pitt readout strict, conflict arm", y[c], probe_m[c]); ci("pitt projector retrained, conflict arm", y[c], proj[c])
paired("pitt readout minus answer, conflict arm (paired)", y[c], probe_m[c], p_ans[c])
paired("pitt projector minus answer, conflict arm (paired)", y[c], proj[c], p_ans[c])

# no-plug control: projector retrained on its default input (final encoder layer)
_np = f"{D}/DementiaBank/projector_noplug_oof.npy"
if not os.path.exists(_np): _np = f"{D}/DementiaBank/projector_noplug_oof_partial.npy"
nop = np.load(_np); assert (nop != 0).all(), "no-plug oof incomplete"
ci("pitt projector retrained, no plug (final layer)", y, nop)
paired("pitt projector no plug minus answer (paired)", y, nop, p_ans)
paired("pitt projector layer 24 minus no plug (paired)", y, proj, nop)
ci("pitt projector no plug, conflict arm", y[c], nop[c])
paired("pitt projector layer 24 minus no plug, conflict arm (paired)", y[c], proj[c], nop[c])
# ADReSSo, patient-onset window
ar = rd(f"{D}/adresso/adresso_scores_qwen2audio_patient.csv"); at = rd(f"{D}/adresso/adresso_textonly_v3_scores.csv")
tmap = {r["spk"]: r for r in at}
ya = np.array([int(r["label"]) for r in ar]); pa = col(ar, ["p_main", "p_yes"])
pt = np.array([float(tmap[r["spk"]].get("p_yes", tmap[r["spk"]].get("p_main"))) for r in ar])
ci("adresso answer (patient window)", ya, pa); ci("adresso transcript only (whisper-large-v3)", ya, pt)
paired("adresso transcript minus answer (paired)", ya, pt, pa)
# ADReSS-2020, patient-onset window
a20 = rd(f"{D}/adress2020/adress2020_answers_patient.csv")
y20 = np.array([int(r["label"]) for r in a20]); p20 = col(a20, ["p_yes", "p_main"]); sp = np.array([r["split"] for r in a20])
ci("adress2020 answer pooled (patient window)", y20, p20)
ci("adress2020 answer, official test split", y20[sp == "test"], p20[sp == "test"])
ci("adress2020 answer, official train split", y20[sp == "train"], p20[sp == "train"])
# MDVR-KCL read, projector run
kc = [r for r in rd(f"{D}/KCL/kcl_local_scores.csv") if r["task"] == "read"]; print("kcl read rows", len(kc))
yk = col(kc, ["label", "y"]); pk = col(kc, ["p_voice", "p_main", "p_yes"]); ok = np.load(f"{D}/KCL/kcl_projector_oof.npy")
print("kcl n", len(yk), "oof n", len(ok))
ci("kcl answer", yk, pk); ci("kcl projector retrained", yk, ok); paired("kcl projector minus answer (paired)", yk, ok, pk)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["quantity", "n", "mean", "ci_low", "ci_high"])
    for r in out: w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}"])
print("wrote", OUT)
