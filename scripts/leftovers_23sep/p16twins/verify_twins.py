"""Verifier for twins_p16.tsv, written separately from the twin code.
(1) 'original' column: re-read from summary_all.csv (value_computed) for every id present there, and from the PART 16 rows.
(2) 'recomputed' column: re-derived from the SAVED per-clip outputs with sklearn.metrics.roc_auc_score (not the rank formula),
    for every AUC row that has a saved per-clip output; CIs re-derived for the paper-value rows with a vectorised bootstrap.
Writes verify_twins.tsv (id, check, twin_value, verifier_value, agree_4dp)."""
import os, csv, json, sys, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

OUT = "<local data dir>/leftovers_23sep/p16twins"
P16 = "<local data dir>/release/edaic_rerun/part16"
R = "<local data dir>/release"
PULL = f"{OUT}/pull"
T = {r["id"]: r for r in csv.DictReader(open(f"{OUT}/twins_p16.tsv"), delimiter="\t")}
S = pd.read_csv("<local data dir>/release_from_mac/summary/summary_all.csv", dtype=str)
S = S[S["part"] == "16"]
checks = []
def chk(id_, what, a, b, tol=None):
    try:
        fa, fb = float(str(a).replace("+", "")), float(str(b).replace("+", ""))
        ok = (f"{fa:.4f}" == f"{fb:.4f}") if tol is None else abs(fa - fb) <= tol
    except Exception:
        ok = str(a) == str(b)
    checks.append(dict(id=id_, check=what, twin_value=a, verifier_value=b, agree_4dp="yes" if ok else "no"))

# (1) originals against summary_all.csv
alias = {"SEED14_3b": "SEED14_3b", "M7fix#01 Pitt encoder probe mean of 5": None, "M7fix#02 Pitt paired probe minus zero-shot": None}
for id_, r in T.items():
    sid = id_
    if id_.startswith("M7fix#01"): sid = "M7fix#01:Qwen2.5_Omni_Pitt_encoder_probe_5_repeat_CORRECTED_source"
    if id_.startswith("M7fix#02"): sid = "M7fix#02:Qwen2.5_Omni_Pitt_paired_probe_minus_zero_shot_CORRECTED"
    m = S[S["id"] == sid]
    if len(m): chk(id_, "original == summary_all value_computed", r["original"], m.iloc[0]["value_computed"])

def boot(fn, spk, draws=2000):
    us = np.unique(spk); rows = [np.flatnonzero(spk == u) for u in us]; g = np.random.default_rng(0); v = []
    for _ in range(draws):
        ii = np.concatenate([rows[k] for k in g.choice(len(us), size=len(us), replace=True)])
        x = fn(ii)
        if x is not None: v.append(x)
    return np.percentile(v, 2.5), np.percentile(v, 97.5)

# (2a) POD3 3b by arm, from the saved per-clip outputs of the pod run
for arm in ("all", "conflict", "agreement"):
    z = np.load(f"{PULL}/lo-p16twins-1/out/twin_dir_{arm}_perclip.npz", allow_pickle=True)
    y = z["label"]; spk = z["spk"]
    for col, tid in (("oof", f"3b_{arm}_auc_probe"), ("oof_trainz", f"3b_{arm}_auc_wo_d"), ("s_d", f"3b_{arm}_auc_d")):
        if tid in T:
            chk(tid, "roc_auc_score on saved per-clip scores", T[tid]["recomputed"], roc_auc_score(y, z[col]))
            if arm != "all" and col in ("oof", "oof_trainz"):
                def f(ii, c=z[col]):
                    return roc_auc_score(y[ii], c[ii]) if y[ii].min() != y[ii].max() else None
                lo, hi = boot(f, spk)
                chk(tid, "CI re-derived (roc_auc_score)", T[tid]["recomputed_ci"], f"[{lo:.4f}, {hi:.4f}]")
    if f"3b_{arm}_cos" in T:
        chk(f"3b_{arm}_cos", "cosine from saved directions", T[f"3b_{arm}_cos"]["recomputed"], float(z["probe_dir"] @ z["d"]))

# (2b) POD3 best stage, from the saved best-stage OOF (5 repeats) next to the json the table cites
SUBSET = {"llmtop4": [13, 14, 15, 19], "anstop4": [15, 18, 19, 27]}
for tid, r in T.items():
    if not tid.endswith("_best_stage"): continue
    ds, st = tid.split("_")[1], tid.split("_")[2]
    f = r["file"].replace(".json", "_best_oof.npz"); z = np.load(f, allow_pickle=True)
    y = z["label"]; spk = z["spk"]
    v = np.mean([roc_auc_score(y, o) for o in z["oof"]])
    chk(tid, "mean of 5 roc_auc_score on saved best-stage OOF", r["recomputed"], v)
    lo, hi = boot(lambda ii: float(np.mean([roc_auc_score(y[ii], o[ii]) for o in z["oof"]])) if y[ii].min() != y[ii].max() else None, spk)
    chk(tid, "CI re-derived (roc_auc_score)", r["recomputed_ci"], f"[{lo:.4f}, {hi:.4f}]")
    tag = os.path.basename(r["file"]).split("_")[2]
    stage = int(z["stage"]) if tag not in SUBSET else SUBSET[tag][int(z["stage"])]
    sj = f"{P16}/POD3/q3o_pitt_perlayer_summary.json" if ds == "pitt" else f"{P16}/POD3/q3o_pcgita_{ {'enc':'encoder','llm':'llm','ans':'ans'}[st] }_perlayer_summary.json"
    chk(tid, "best stage equals POD3 best stage", stage, json.load(open(sj))["streams"][st]["best_stage"])

# (2c) M7 Pitt fix and rep5
z = np.load(f"{R}/overnight2/part10/pitt_enc_nested5_oof.npz", allow_pickle=True)
oof, y, spk, name = z["oof"], z["label"].astype(int), z["spk"].astype(str), z["name"].astype(str)
zs = pd.read_csv(f"{R}/omni_final/omni_pitt_zeroshot_scores.csv"); zmap = dict(zip(zs["clip"], zs["p_yes"]))
q = np.array([zmap[n] for n in name])
probe = np.mean([roc_auc_score(y, o) for o in oof]); zsa = roc_auc_score(y, q)
chk("M7fix#01 Pitt encoder probe mean of 5", "roc_auc_score", T["M7fix#01 Pitt encoder probe mean of 5"]["recomputed"], probe)
chk("M7fix#02 Pitt paired probe minus zero-shot", "roc_auc_score", T["M7fix#02 Pitt paired probe minus zero-shot"]["recomputed"], probe - zsa)
lo, hi = boot(lambda ii: float(np.mean([roc_auc_score(y[ii], o[ii]) for o in oof]) - roc_auc_score(y[ii], q[ii])) if y[ii].min() != y[ii].max() else None, spk)
chk("M7fix#02 Pitt paired probe minus zero-shot", "paired CI re-derived", T["M7fix#02 Pitt paired probe minus zero-shot"]["recomputed_ci"], f"[{lo:.4f}, {hi:.4f}]")
chk("M7_pitt_probe_rep5", "roc_auc_score", T["M7_pitt_probe_rep5"]["recomputed"], roc_auc_score(y, oof.mean(0)))
chk("M7_pitt_gap_rep5", "roc_auc_score", T["M7_pitt_gap_rep5"]["recomputed"], roc_auc_score(y, oof.mean(0)) - zsa)

# (2d) M3 speaker level
for ds in ("pitt", "pcgita", "neurovoz"):
    d = pd.read_csv(f"{R}/omni_final/omni_{ds}_zeroshot_scores.csv")
    sp = d.groupby("speaker").agg(label=("label", "max"), p=("p_yes", "mean"))
    c = roc_auc_score(d.label, d.p_yes); s_ = roc_auc_score(sp.label, sp.p)
    chk(f"M3_{ds}_speaker", "roc_auc_score", T[f"M3_{ds}_speaker"]["recomputed"], s_)
    chk(f"M3_{ds}_clip_minus_speaker", "roc_auc_score", T[f"M3_{ds}_clip_minus_speaker"]["recomputed"], c - s_)

# (2e) M5 per-clip rows
m5 = pd.read_csv(f"{P16}/M5_readout_direction_q2a.csv")
C = m5.arm == "conflict"; A = m5.arm == "agreement"; ALL = m5.arm.notna()
for tid, mk, col in (("M5_all_auc_without_d", ALL, "z_probe_oof_without_d_all"), ("M5_all_auc_without_d_Xproj", ALL, "p_probe_oof_without_d_Xproj_all"),
                     ("M5_conflict_refit_auc_without_d", C, "z_probe_oof_without_d_armrefit"), ("M5_agreement_refit_auc_without_d", A, "z_probe_oof_without_d_armrefit"),
                     ("M5_conflict_restricted_auc_without_d", C, "z_probe_oof_without_d_all"), ("M5_conflict_restricted_auc_without_d_Xproj", C, "p_probe_oof_without_d_Xproj_all"),
                     ("M5_agreement_restricted_auc_without_d", A, "z_probe_oof_without_d_all"), ("M5_agreement_restricted_auc_without_d_Xproj", A, "p_probe_oof_without_d_Xproj_all")):
    chk(tid, "roc_auc_score", T[tid]["recomputed"], roc_auc_score(m5.label[mk], m5[col][mk]))
chk("M5_paired_auc_without_d_conflict_minus_agreement", "roc_auc_score", T["M5_paired_auc_without_d_conflict_minus_agreement"]["recomputed"],
    roc_auc_score(m5.label[C], m5.z_probe_oof_without_d_all[C]) - roc_auc_score(m5.label[A], m5.z_probe_oof_without_d_all[A]))

# (2f) POD4 4b descriptives, recounted with pandas query
co = pd.read_csv(f"{P16}/POD4/cha_overlap_468.csv"); cr = pd.DataFrame(json.load(open(f"{P16}/POD4/cut_report.json")))
chk("POD4_4b_resolved", "count", T["POD4_4b_resolved"]["recomputed"], co.shape[0])
chk("POD4_4b_invwin", "count", T["POD4_4b_invwin"]["recomputed"], int(co.query("inv_ms_win > 0").shape[0]))
chk("POD4_4b_invshare", "mean", T["POD4_4b_invshare"]["recomputed"], co.eval("inv_ms_win / win_span_ms").mean())
chk("POD4_4b_parshare", "mean", T["POD4_4b_parshare"]["recomputed"], co.eval("par_ms_win / win_span_ms").mean())
chk("POD4_4b_pardur", "mean (2 dp)", T["POD4_4b_pardur"]["recomputed"], f"{cr[cr.ok == 1].par_s.mean():.2f}")
chk("POD4_4b_under10s", "count", T["POD4_4b_under10s"]["recomputed"], int((cr[cr.ok == 1].par_s < 10).sum()))

with open(f"{OUT}/verify_twins.tsv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "check", "twin_value", "verifier_value", "agree_4dp"], delimiter="\t"); w.writeheader(); w.writerows(checks)
bad = [c for c in checks if c["agree_4dp"] != "yes"]
print(f"verifier checks {len(checks)}, agree {len(checks)-len(bad)}")
for b in bad: print("DISAGREE", b)
