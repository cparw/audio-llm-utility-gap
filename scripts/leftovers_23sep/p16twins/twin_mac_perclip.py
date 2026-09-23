"""p16twins, Mac side: independent recomputes of the PART 16 values that need only per-clip files, logs or file headers.
Written from scratch (does not import the PART 16 scripts). Bootstrap rule: 2000 draws, fresh numpy default_rng(0) per cell,
idx = rng.choice(N, size=N, replace=True) over the sorted unique speaker ids, 2.5 / 97.5 percentiles, paired = same idx.
Output: twin_mac_perclip.json (list of rows: id, original, recomputed, lo, hi, file, how, closes)."""
import os, re, json, glob, csv, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata

R = "<local data dir>/release"
P16 = f"{R}/edaic_rerun/part16"
G = "<local data dir>/paper work"
OUTJ = sys.argv[1]
rows = []

def auc(y, s):
    y = np.asarray(y).astype(int); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(np.asarray(s, dtype=np.float64))
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def spk_index(spk):
    us = np.unique(spk); return us, [np.flatnonzero(spk == u) for u in us]

def add(id_, orig, val, lo=None, hi=None, olo=None, ohi=None, file="", how="", closes="", fmt="{:.4f}"):
    rows.append(dict(id=id_, original=orig, original_lo=olo, original_hi=ohi, recomputed=val, lo=lo, hi=hi,
                     file=file, how=how, closes=closes, fmt=fmt))

def remap(p):
    p = p.replace("<local data dir>/paper1_local_runs/", "<local data dir>/paper1_local_runs/")
    return p

# ------------------------------------------------------------------ M3 speaker level (6 rows) + clip (3 rows)
M3 = {r["id"]: r for r in csv.DictReader(open(f"{P16}/rows/M3.tsv"), delimiter="\t")}
for ds in ("pitt", "pcgita", "neurovoz"):
    f = f"{R}/omni_final/omni_{ds}_zeroshot_scores.csv"
    d = pd.read_csv(f)
    y = d.label.astype(int).to_numpy(); p = d.p_yes.astype(float).to_numpy(); spk = d.speaker.astype(str).to_numpy()
    us, rws = spk_index(spk)
    lab_s = np.array([y[r].max() for r in rws]); mp_s = np.array([p[r].mean() for r in rws])
    assert all(len(set(y[r])) == 1 for r in rws)
    a_c, a_s = auc(y, p), auc(lab_s, mp_s)
    g = np.random.default_rng(0); bc, bs, bd = [], [], []
    for _ in range(2000):
        idx = g.choice(len(us), size=len(us), replace=True)
        ii = np.concatenate([rws[k] for k in idx])
        if len(set(lab_s[idx])) < 2 or len(set(y[ii])) < 2: continue
        c, s_ = auc(y[ii], p[ii]), auc(lab_s[idx], mp_s[idx]); bc.append(c); bs.append(s_); bd.append(c - s_)
    pc = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    for suf, val, ci_ in (("clip", a_c, pc(bc)), ("speaker", a_s, pc(bs)), ("clip_minus_speaker", a_c - a_s, pc(bd))):
        o = M3[f"M3_{ds}_{suf}"]
        add(f"M3_{ds}_{suf}", o["value"], val, *ci_, olo=o["lo"], ohi=o["hi"], file=f,
            how=f"speaker mean p_yes, paired speaker bootstrap over {len(us)} sorted speakers, {len(bd)} usable draws",
            closes=("audit_p16 item 6: M3 PC-GITA and NeuroVoz paired, no twin" if suf == "clip_minus_speaker" and ds != "pitt"
                    else "audit_p16 item 6: M3 twinned by the audit only (no saved twin)" if suf != "clip" else "extra (already twinned)"))

# ------------------------------------------------------------------ M7 Pitt fix and rep5 (4 rows)
z = np.load(f"{R}/overnight2/part10/pitt_enc_nested5_oof.npz", allow_pickle=True)
oof, y, spk, name = z["oof"], z["label"].astype(int), z["spk"].astype(str), z["name"].astype(str)
zsf = f"{R}/omni_final/omni_pitt_zeroshot_scores.csv"
zs = pd.read_csv(zsf); zmap = dict(zip(zs["clip"].map(os.path.basename), zs.p_yes.astype(float)))
zsv = np.array([zmap[os.path.basename(n)] for n in name])
us, rws = spk_index(spk)
per = [auc(y, o) for o in oof]; probe = float(np.mean(per)); zs_auc = auc(y, zsv); pm = oof.mean(0); a_pm = auc(y, pm)
g = np.random.default_rng(0); bp, bd = [], []
for _ in range(2000):
    ii = np.concatenate([rws[k] for k in g.choice(len(us), size=len(us), replace=True)]); yy = y[ii]
    if yy.min() == yy.max(): continue
    a = float(np.mean([auc(yy, o[ii]) for o in oof])); b = auc(yy, zsv[ii]); bp.append(a); bd.append(a - b)
g = np.random.default_rng(0); bp5, bd5 = [], []
for _ in range(2000):
    ii = np.concatenate([rws[k] for k in g.choice(len(us), size=len(us), replace=True)]); yy = y[ii]
    if yy.min() == yy.max(): continue
    a = auc(yy, pm[ii]); b = auc(yy, zsv[ii]); bp5.append(a); bd5.append(a - b)
q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
npzf = f"{R}/overnight2/part10/pitt_enc_nested5_oof.npz"
add("M7fix#01 Pitt encoder probe mean of 5", "0.7706", probe, *q(bp), olo="0.7157", ohi="0.8220", file=npzf,
    how=f"mean of per-repeat AUCs {[round(v,4) for v in per]}, speaker bootstrap of the mean", closes="audit_p16 item 6: M7 fix twinned by the audit only; ledger LEFT me 3 (save the M7 twin file)")
add("M7fix#02 Pitt paired probe minus zero-shot", "+0.1128", probe - zs_auc, *q(bd), olo="0.0455", ohi="0.1768", file=npzf + " + " + zsf,
    how=f"(mean of 5 per-repeat AUCs) minus zero-shot AUC {zs_auc:.4f}, paired speaker bootstrap", closes="task: M7 Pitt gap twin; audit_p16 item 6 PAPER VALUE with no saved twin; ledger LEFT me 3")
add("M7_pitt_probe_rep5", "0.7917", a_pm, *q(bp5), olo="0.7324", ohi="0.8480", file=npzf,
    how="AUC of the mean of the 5 OOF probabilities", closes="audit_p16 item 6: M7 rep5 twinned by the audit only")
add("M7_pitt_gap_rep5", "0.1339", a_pm - zs_auc, *q(bd5), olo="0.0641", ohi="0.2019", file=npzf + " + " + zsf,
    how="AUC(mean prob) minus zero-shot, paired speaker bootstrap", closes="audit_p16 item 6: M7 rep5 twinned by the audit only")

# ------------------------------------------------------------------ POD4 4b descriptives (6 rows)
co = pd.read_csv(f"{P16}/POD4/cha_overlap_468.csv"); cr = json.load(open(f"{P16}/POD4/cut_report.json"))
ok = [r for r in cr if r.get("ok") == 1]; ps = np.array([r["par_s"] for r in ok])
f1 = f"{P16}/POD4/cha_overlap_468.csv"; f2 = f"{P16}/POD4/cut_report.json"
cl = "task: POD4 4b descriptives twins; audit_p16 item 3 and item 6 (twinned by the audit only)"
add("POD4_4b_resolved", "468", len(co), file=f1, how="rows resolved to CHAT timings", closes=cl, fmt="{:d}")
add("POD4_4b_invwin", "158", int((co.inv_ms_win > 0).sum()), file=f1, how="windows with inv_ms_win > 0", closes=cl, fmt="{:d}")
add("POD4_4b_invshare", "0.0200", float((co.inv_ms_win / co.win_span_ms).mean()), file=f1, how="mean inv_ms_win / win_span_ms", closes=cl)
add("POD4_4b_parshare", "0.5871", float((co.par_ms_win / co.win_span_ms).mean()), file=f1, how="mean par_ms_win / win_span_ms", closes=cl)
add("POD4_4b_pardur", "17.61", float(ps.mean()), file=f2, how=f"mean par_s over {len(ok)} ok cuts", closes=cl, fmt="{:.2f}")
add("POD4_4b_under10s", "68", int((ps < 10).sum()), file=f2, how="cuts with par_s < 10 s", closes=cl, fmt="{:d}")

# ------------------------------------------------------------------ M9 FILES and SWEEPMIN
inv = pd.read_csv(f"{P16}/M9_inventory.csv")
MASS = ["answer_mass", "mass", "denom", "p_sum", "psum"]
has, missing, flips, med = 0, [], [], []
for _, r in inv.iterrows():
    p = remap(r["file"])
    if not os.path.exists(p): missing.append(r["file"]); continue
    h = [c.strip().strip('"').lower() for c in open(p, errors="replace").readline().strip().split(",")]
    mc = [c for c in MASS if c in h]; rec = ("p_yes_raw" in h and "p_no_raw" in h)
    hm = bool(mc or rec); has += hm
    if hm != bool(r["has_mass"]): flips.append(r["file"])
    if hm:
        d = pd.read_csv(p)
        cols = {c.strip().strip('"').lower(): c for c in d.columns}
        if mc: v = pd.to_numeric(d[cols[mc[0]]], errors="coerce").to_numpy(float)
        else: v = (pd.to_numeric(d[cols["p_yes_raw"]], errors="coerce") + pd.to_numeric(d[cols["p_no_raw"]], errors="coerce")).to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v): med.append((float(np.median(v)), len(v), r["file"]))
med.sort()
add("M9.FILES", "307", has, file=f"{P16}/M9_inventory.csv",
    how=f"reopened the {len(inv)} inventory files on disk (old G-Drive prefix remapped), counted headers with a mass column "
        f"({', '.join(MASS)}) or p_yes_raw+p_no_raw; {len(missing)} files missing, {len(flips)} disagree with the stored flag",
    closes="audit_p16 item 6: M9 FILES count, no twin", fmt="{:d}")
add("M9.SWEEPMIN", "0.8662", med[0][0], file=remap(med[0][2]),
    how=f"minimum over {len(med)} mass-bearing files of the median answer mass; argmin n={med[0][1]}",
    closes="audit_p16 item 6: M9 sweep min twinned by the audit only")

# ------------------------------------------------------------------ MEDAIC json aggregates (8 rows): log twin + answer recompute
lg = open(f"{R}/edaic_rerun/logs/podA/o25_full_rep.log").read()
ans_f = f"{R}/edaic_rerun/variants/o25_full_zeroshot_scores.csv"
a = pd.read_csv(ans_f); ans_auc = auc(a.label.astype(int), a.p_yes.astype(float))
jf = f"{R}/edaic_rerun/variants/o25_full_nested_repeats.json"; js = json.load(open(jf))
MED = {r["id"]: r for r in csv.DictReader(open(f"{P16}/rows/MEDAIC.tsv"), delimiter="\t")}
for st in ("llm", "enc", "ans", "proj"):
    m = re.search(rf"^{st}: mean [0-9.]+ over \d+ splits\s+\[([^\]]+)\]", lg, re.M)
    reps = [float(x) for x in m.group(1).split(",")]
    assert np.allclose(reps, js[st]["per_repeat"])
    mean_ = float(np.mean(reps))
    add(f"MEDAIC_full_{st}_nested_NOCI", MED[f"MEDAIC_full_{st}_nested_NOCI"]["value"], mean_,
        file=f"{R}/edaic_rerun/logs/podA/o25_full_rep.log",
        how=f"mean of the per-repeat AUCs printed in the pod log {reps} (same list as the json); NOT a refit: no per-clip OOF or states exist",
        closes="audit_p16 item 6: MEDAIC json aggregates, no twin (log twin only)")
    add(f"MEDAIC_full_{st}_minus_answer_POINT", MED[f"MEDAIC_full_{st}_minus_answer_POINT"]["value"], mean_ - ans_auc,
        file=f"{R}/edaic_rerun/logs/podA/o25_full_rep.log + {ans_f}",
        how=f"log mean minus answer AUC {ans_auc:.6f} recomputed from the per-clip zero-shot file; probe side not refittable",
        closes="audit_p16 item 6: MEDAIC json aggregates, no twin (log twin only)", fmt="{:+.4f}")

# ------------------------------------------------------------------ POD2 L5 and L6: run-log twin
lg2 = open(f"{P16}/POD2/lora.log").read()
fm = [float(x) for x in re.findall(r"fold \d done ([0-9.]+) min", lg2)]
pk = [float(x) for x in re.findall(r"peak ([0-9.]+) GB", lg2)]
add("P16.POD2.L5", "4.13;4.07;4.05;4.03;4.00", ";".join(f"{x:.2f}" for x in fm), file=f"{P16}/POD2/lora.log",
    how="per-fold minutes read from the pod training log (a second record, not the json); a wall-clock time cannot be recomputed",
    closes="audit_p16 item 6: POD2 L5 run log, no twin (log twin only)", fmt="{}")
add("P16.POD2.L6", "21.013", max(pk), file=f"{P16}/POD2/lora.log",
    how=f"max of the per-fold peak GPU GB in the pod log {pk}; a GPU memory peak cannot be recomputed",
    closes="audit_p16 item 6: POD2 L6 run log, no twin (log twin only)", fmt="{:.3f}")

# ------------------------------------------------------------------ M5 AUC rows from the per-clip file (9 rows)
m5f = f"{P16}/M5_readout_direction_q2a.csv"; m5 = pd.read_csv(m5f)
M5 = {r["id"]: r for r in csv.DictReader(open(f"{P16}/rows/M5.tsv"), delimiter="\t")}
y5 = m5.label.astype(int).to_numpy(); s5 = m5.speaker.astype(str).to_numpy(); arm5 = m5.arm.to_numpy()
def ci5(mask, col):
    yy, ss, sp = y5[mask], m5[col].to_numpy(float)[mask], s5[mask]
    us, rws = spk_index(sp); g = np.random.default_rng(0); v = []
    for _ in range(2000):
        ii = np.concatenate([rws[k] for k in g.choice(len(us), size=len(us), replace=True)])
        a_ = auc(yy[ii], ss[ii])
        if a_ == a_: v.append(a_)
    return auc(yy, ss), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
ALL = np.ones(len(m5), bool); C = arm5 == "conflict"; A = arm5 == "agreement"
spec = [("M5_all_auc_without_d", ALL, "z_probe_oof_without_d_all"),
        ("M5_all_auc_without_d_Xproj", ALL, "p_probe_oof_without_d_Xproj_all"),
        ("M5_conflict_refit_auc_without_d", C, "z_probe_oof_without_d_armrefit"),
        ("M5_agreement_refit_auc_without_d", A, "z_probe_oof_without_d_armrefit"),
        ("M5_conflict_restricted_auc_without_d", C, "z_probe_oof_without_d_all"),
        ("M5_conflict_restricted_auc_without_d_Xproj", C, "p_probe_oof_without_d_Xproj_all"),
        ("M5_agreement_restricted_auc_without_d", A, "z_probe_oof_without_d_all"),
        ("M5_agreement_restricted_auc_without_d_Xproj", A, "p_probe_oof_without_d_Xproj_all")]
for id_, mk, col in spec:
    v, lo, hi = ci5(mk, col); o = M5[id_]
    add(id_, o["value"], v, lo, hi, olo=o["lo"], ohi=o["hi"], file=m5f, how=f"AUC of per-clip column {col}, speaker bootstrap",
        closes="audit_p16 item 6: M5 flagged verified though the M5 verifier disagrees (verifier used the canonical estimator)")
us, rws = spk_index(s5); g = np.random.default_rng(0); v = []; col = m5["z_probe_oof_without_d_all"].to_numpy(float)
for _ in range(2000):
    ii = np.concatenate([rws[k] for k in g.choice(len(us), size=len(us), replace=True)])
    c = ii[C[ii]]; a_ = ii[A[ii]]; x1, x2 = auc(y5[c], col[c]), auc(y5[a_], col[a_])
    if x1 == x1 and x2 == x2: v.append(x1 - x2)
o = M5["M5_paired_auc_without_d_conflict_minus_agreement"]
add("M5_paired_auc_without_d_conflict_minus_agreement", o["value"], auc(y5[C], col[C]) - auc(y5[A], col[A]),
    float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), olo=o["lo"], ohi=o["hi"], file=m5f,
    how="conflict minus agreement AUC of z_probe_oof_without_d_all, paired speaker bootstrap over 228 speakers",
    closes="audit_p16 item 6: M5 flagged verified though the M5 verifier disagrees", fmt="{:+.4f}")

json.dump(rows, open(OUTJ, "w"), indent=1, default=str)
for r in rows:
    f = r["fmt"]; v = r["recomputed"]
    vs = f.format(v) if f != "{}" else str(v)
    print(f"{r['id']:58s} orig {str(r['original']):>26s}  mine {vs:>26s}  "
          f"{'' if r['lo'] is None else '[%.4f, %.4f]' % (r['lo'], r['hi'])}  orig_ci [{r['original_lo']}, {r['original_hi']}]")
