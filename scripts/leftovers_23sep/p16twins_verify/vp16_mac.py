"""p16twins VERIFIER, Mac part. Written from scratch; reads only per-clip / per-repeat / log files.
AUC = Mann-Whitney count via searchsorted (ties one half). Bootstrap: 2000 draws, fresh default_rng(0) per cell,
idx = rng.choice(N, N, replace=True) over the sorted unique speaker ids, percentiles 2.5 / 97.5, paired = same idx."""
import os, re, json, glob
import numpy as np, pandas as pd

REL = "<local data dir>/release"
P16 = REL + "/edaic_rerun/part16"
OUTJ = "<local data dir>/scratch/vp16/mac_results.json"
R = {}

def auc(lab, sc):
    lab = np.asarray(lab).astype(int); sc = np.asarray(sc, dtype=np.float64)
    pos = np.sort(sc[lab == 1]); neg = np.sort(sc[lab == 0])
    if len(pos) == 0 or len(neg) == 0: return np.nan
    lt = np.searchsorted(neg, pos, "left"); le = np.searchsorted(neg, pos, "right")
    return float((lt.sum() + 0.5 * (le - lt).sum()) / (len(pos) * len(neg)))

def boot(spk, fn, draws=2000):
    """fn(ii) -> tuple of stats on clip index array ii, or None to skip. Returns array (usable, k)."""
    spk = np.asarray(spk).astype(str); u = np.unique(spk)
    rows = {s: np.flatnonzero(spk == s) for s in u}
    rng = np.random.default_rng(0); out = []
    for _ in range(draws):
        k = rng.choice(len(u), size=len(u), replace=True)
        ii = np.concatenate([rows[u[j]] for j in k])
        v = fn(ii)
        if v is not None: out.append(v)
    return np.array(out)

def pc(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def two(y): return len(np.unique(y)) == 2

# ---------------- M3 speaker level, Qwen2.5-Omni zero shot ----------------
for ds in ("pitt", "pcgita", "neurovoz"):
    f = f"{REL}/omni_final/omni_{ds}_zeroshot_scores.csv"
    d = pd.read_csv(f, dtype={"speaker": str})
    y = d.label.to_numpy(int); p = d.p_yes.to_numpy(float); s = d.speaker.astype(str).to_numpy()
    u = np.unique(s); code = np.searchsorted(u, s)
    sl = np.zeros(len(u), int); np.maximum.at(sl, code, y)            # speaker label = max over clips
    cnt = np.bincount(code, minlength=len(u)); sm = np.bincount(code, weights=p, minlength=len(u)) / cnt
    lab_consistent = bool(all(len(set(y[code == k])) == 1 for k in range(len(u))))
    a_c, a_s = auc(y, p), auc(sl, sm)
    def fn(ii, y=y, p=p, code=code, sl=sl, sm=sm):
        # speakers drawn: recover them from the clip draw via the same idx; speaker-level uses one row per draw
        return None
    # paired bootstrap written explicitly over speaker draws
    rng = np.random.default_rng(0); rows = {k: np.flatnonzero(code == k) for k in range(len(u))}
    B = []
    for _ in range(2000):
        k = rng.choice(len(u), size=len(u), replace=True)
        ii = np.concatenate([rows[j] for j in k])
        if not two(sl[k]) or not two(y[ii]): continue
        ac, as_ = auc(y[ii], p[ii]), auc(sl[k], sm[k])
        B.append((ac, as_, ac - as_))
    B = np.array(B)
    R[f"M3_{ds}_speaker"] = dict(value=a_s, ci=pc(B[:, 1]), n_spk=int(len(u)), draws=int(len(B)), file=f,
                                 labels_consistent=lab_consistent, clip_auc=a_c)
    R[f"M3_{ds}_clip_minus_speaker"] = dict(value=a_c - a_s, ci=pc(B[:, 2]), draws=int(len(B)), file=f)

# ---------------- M7 Pitt (part10 5-repeat OOF vs Omni zero shot) ----------------
z = np.load(f"{REL}/overnight2/part10/pitt_enc_nested5_oof.npz", allow_pickle=True)
oof = np.asarray(z["oof"], float); y = np.asarray(z["label"]).astype(int)
spk = np.asarray(z["spk"]).astype(str); name = np.asarray(z["name"]).astype(str)
zs = pd.read_csv(f"{REL}/omni_final/omni_pitt_zeroshot_scores.csv")
zmap = dict(zip(zs["clip"].map(os.path.basename), zs.p_yes))
q = np.array([zmap[os.path.basename(n)] for n in name], float)
zs2 = pd.read_csv(f"{P16}/POD4/omni_pitt_zeroshot_scores.csv") if os.path.exists(f"{P16}/POD4/omni_pitt_zeroshot_scores.csv") else None
same_zs = None if zs2 is None else bool(np.array_equal(zs2.sort_values("clip").p_yes.to_numpy(), zs.sort_values("clip").p_yes.to_numpy()))
per = [auc(y, oof[r]) for r in range(oof.shape[0])]
pm = float(np.mean(per)); pbar = oof.mean(0); a_bar = auc(y, pbar); a_z = auc(y, q)
def fn(ii):
    if not two(y[ii]): return None
    m5 = np.mean([auc(y[ii], oof[r][ii]) for r in range(oof.shape[0])])
    az = auc(y[ii], q[ii]); ab = auc(y[ii], pbar[ii])
    return (m5, m5 - az, ab, ab - az)
B = boot(spk, fn)
R["M7fix#01 Pitt encoder probe mean of 5"] = dict(value=pm, ci=pc(B[:, 0]), per_repeat=per, draws=int(len(B)))
R["M7fix#02 Pitt paired probe minus zero-shot"] = dict(value=pm - a_z, ci=pc(B[:, 1]), zeroshot=a_z, draws=int(len(B)),
                                                       zs_files_identical=same_zs)
R["M7_pitt_probe_rep5"] = dict(value=a_bar, ci=pc(B[:, 2]), draws=int(len(B)))
R["M7_pitt_gap_rep5"] = dict(value=a_bar - a_z, ci=pc(B[:, 3]), draws=int(len(B)))

# ---------------- POD4 4b descriptives ----------------
c = pd.read_csv(f"{P16}/POD4/cha_overlap_468.csv")
ok = c.dropna(subset=["par_ms_win", "inv_ms_win", "win_span_ms"])
R["POD4_4b_resolved"] = dict(value=int(len(ok)), rows=int(len(c)), unique_clips=int(c["clip"].nunique()))
R["POD4_4b_invwin"] = dict(value=int((ok.inv_ms_win > 0).sum()))
R["POD4_4b_invshare"] = dict(value=float((ok.inv_ms_win / ok.win_span_ms).mean()), from_col=float(ok.inv_share_win.mean()))
R["POD4_4b_parshare"] = dict(value=float((ok.par_ms_win / ok.win_span_ms).mean()), from_col=float(ok.par_share_win.mean()))
cr = pd.DataFrame(json.load(open(f"{P16}/POD4/cut_report.json")))
cok = cr[cr.ok == 1]
R["POD4_4b_pardur"] = dict(value=float(cok.par_s.mean()), n_cuts=int(len(cok)), n_rows=int(len(cr)))
R["POD4_4b_under10s"] = dict(value=int((cok.par_s < 10).sum()))

# ---------------- M9 files and sweep minimum ----------------
inv = pd.read_csv(f"{P16}/M9_inventory.csv")
SCORE = {"p_yes", "p_probe", "p", "p_egemaps", "p_text", "p_audio", "p_dep", "p_after"}
MASS = ["answer_mass", "mass", "denom", "p_sum", "psum"]
def remap(p):
    if os.path.exists(p): return p
    for a, b in (("<local data dir>/paper1_local_runs/", "<local data dir>/paper1_local_runs/"),):
        if p.startswith(a) and os.path.exists(b + p[len(a):]): return b + p[len(a):]
    return None
n_ok = n_missing = n_score = n_mass = n_disagree = 0; medians = []
for _, r in inv.iterrows():
    p = remap(r.file)
    if p is None: n_missing += 1; continue
    n_ok += 1
    with open(p, errors="replace") as fh: h = fh.readline().strip()
    cols = [x.strip().strip('"').lower() for x in h.split(",")]
    if SCORE & set(cols): n_score += 1
    mc = [m for m in MASS if m in cols]; rec = "p_yes_raw" in cols and "p_no_raw" in cols
    has = bool(mc or rec); n_mass += has; n_disagree += (has != bool(r.has_mass))
    if has:
        df = pd.read_csv(p)
        lc = {k.lower(): k for k in df.columns}
        if mc: v = pd.to_numeric(df[lc[mc[0]]], errors="coerce").to_numpy(float)
        else: v = (pd.to_numeric(df[lc["p_yes_raw"]], errors="coerce") + pd.to_numeric(df[lc["p_no_raw"]], errors="coerce")).to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v): medians.append((float(np.median(v)), p, len(v)))
medians.sort()
R["M9.FILES"] = dict(value=int(n_mass), of=int(len(inv)), opened=n_ok, missing=n_missing, with_score_col=n_score, disagree_with_flag=n_disagree)
R["M9.SWEEPMIN"] = dict(value=medians[0][0], file=medians[0][1], n=medians[0][2], files_swept=len(medians),
                        next=[(round(a, 4), os.path.basename(b)) for a, b, _ in medians[1:4]])

# ---------------- M5 AUC rows from the per-clip columns ----------------
f5 = f"{P16}/M5_readout_direction_q2a.csv"
m5 = pd.read_csv(f5); y5 = m5.label.to_numpy(int); s5 = m5.speaker.astype(str).to_numpy(); arm = m5.arm.to_numpy(str)
def cell_auc(col, mask):
    yy, ss, sp = y5[mask], m5[col].to_numpy(float)[mask], s5[mask]
    B = boot(sp, lambda ii: (auc(yy[ii], ss[ii]),) if two(yy[ii]) else None)
    return dict(value=auc(yy, ss), ci=pc(B[:, 0]), n=int(mask.sum()), draws=int(len(B)), column=col)
ALL = np.ones(len(y5), bool); MC = arm == "conflict"; MA = arm == "agreement"
R["M5_all_auc_without_d"] = cell_auc("z_probe_oof_without_d_all", ALL)
R["M5_all_auc_without_d_Xproj"] = cell_auc("p_probe_oof_without_d_Xproj_all", ALL)
R["M5_conflict_refit_auc_without_d"] = cell_auc("z_probe_oof_without_d_armrefit", MC)
R["M5_agreement_refit_auc_without_d"] = cell_auc("z_probe_oof_without_d_armrefit", MA)
R["M5_conflict_restricted_auc_without_d"] = cell_auc("z_probe_oof_without_d_all", MC)
R["M5_conflict_restricted_auc_without_d_Xproj"] = cell_auc("p_probe_oof_without_d_Xproj_all", MC)
R["M5_agreement_restricted_auc_without_d"] = cell_auc("z_probe_oof_without_d_all", MA)
R["M5_agreement_restricted_auc_without_d_Xproj"] = cell_auc("p_probe_oof_without_d_Xproj_all", MA)
sc = m5.z_probe_oof_without_d_all.to_numpy(float)
def fnp(ii):
    a, b = MC[ii], MA[ii]
    if not two(y5[ii][a]) or not two(y5[ii][b]): return None
    return (auc(y5[ii][a], sc[ii][a]) - auc(y5[ii][b], sc[ii][b]),)
B = boot(s5, fnp)
R["M5_paired_auc_without_d_conflict_minus_agreement"] = dict(value=auc(y5[MC], sc[MC]) - auc(y5[MA], sc[MA]), ci=pc(B[:, 0]),
                                                            n_spk=int(len(np.unique(s5))), draws=int(len(B)))

# ---------------- MEDAIC log twins ----------------
lg = open(f"{REL}/edaic_rerun/logs/podA/o25_full_rep.log").read()
ans = pd.read_csv(f"{REL}/edaic_rerun/variants/o25_full_zeroshot_scores.csv")
a_ans = auc(ans.label.to_numpy(int), ans.p_yes.to_numpy(float))
js = json.load(open(f"{REL}/edaic_rerun/variants/o25_full_nested_repeats.json"))
for st in ("llm", "enc", "ans", "proj"):
    mm = re.search(rf"^{st}: mean ([0-9.]+) over (\d+) splits\s+\[([^\]]*)\]", lg, re.M)
    lst = [float(x) for x in mm.group(3).split(",")]
    R[f"MEDAIC_full_{st}_nested_NOCI"] = dict(value=float(np.mean(lst)), log_printed_mean=float(mm.group(1)), per_repeat_log=lst)
    R[f"MEDAIC_full_{st}_minus_answer_POINT"] = dict(value=float(np.mean(lst)) - a_ans, answer_auc=a_ans, n_answer=int(len(ans)))
R["_MEDAIC_json_keys"] = dict(keys=list(js.keys())[:20])

# ---------------- POD2 L5 / L6 run log ----------------
ll = open(f"{P16}/POD2/lora.log").read()
fd = re.findall(r"^fold (\d) done ([0-9.]+) min, \d+ steps, peak ([0-9.]+) GB", ll, re.M)
R["P16.POD2.L5"] = dict(value=";".join(f"{float(m):.2f}" for _, m, _ in fd), folds=[int(k) for k, _, _ in fd])
R["P16.POD2.L6"] = dict(value=max(float(g) for _, _, g in fd), per_fold=[float(g) for _, _, g in fd])
lj = f"{P16}/POD2/lora_pitt_lora.json"
if os.path.exists(lj):
    R["_POD2_json"] = dict(text=open(lj).read()[:3000])

json.dump(R, open(OUTJ, "w"), indent=1, default=str)
for k, v in R.items():
    if k.startswith("_"): continue
    val = v["value"]; ci = v.get("ci")
    print(k, round(val, 4) if isinstance(val, float) else val, [round(x, 4) for x in ci] if ci else "")
