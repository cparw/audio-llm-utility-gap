#!/usr/local/bin/python3
"""
M7 - The core gap, per dataset, paired.
Qwen2.5-Omni nested ENCODER probe OOF vs zero-shot AUDIO answer p_yes, same clips,
2000-draw paired speaker bootstrap, seven datasets.
READ ONLY on omni_final. All writes under part16/.
"""
import os, sys, json, hashlib, subprocess
import numpy as np
import pandas as pd
import scipy.stats as st

OMNI = "<local data dir>/Desktop/release/omni_final"
OUT  = "<local data dir>/Desktop/release/edaic_rerun/part16"
PERCLIP = os.path.join(OUT, "M7_perclip")
ROWS = os.path.join(OUT, "rows")
DISC = "<local data dir>/Desktop/release/edaic_rerun/DISCREPANCIES.md"
MASTER = "lookup/master_lookup.csv"
INDEX  = "<local data dir>/Desktop/release/overnight/perclip/INDEX.csv"
CMD = "/usr/local/bin/python3 " + os.path.abspath(__file__)

DATASETS = ["pitt", "adresso", "adress2020", "pcgita", "neurovoz", "kcl", "edaic"]
NBOOT = 2000
SEED = 0

os.makedirs(PERCLIP, exist_ok=True); os.makedirs(ROWS, exist_ok=True)

disc_lines = []
def disc(sev, msg):
    disc_lines.append("- **%s** [PART16/M7] %s" % (sev, msg))
    print("  DISCREPANCY %s: %s" % (sev, msg))

def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = st.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def sha1mb(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()

# ---------------- master lookup for cross-check ----------------
ml = pd.read_csv(MASTER)
ml_o = ml[ml["model"] == "Qwen2.5-Omni"]
def ml_get(ds, stream):
    r = ml_o[(ml_o["dataset"] == ds) & (ml_o["stream"] == stream)]
    if len(r) == 0: return None, None, None
    return float(r.iloc[0]["auc"]), int(r.iloc[0]["n"]), str(r.iloc[0]["estimator"])

idx = pd.read_csv(INDEX)
def idx_get(exp, ds):
    r = idx[(idx["experiment"] == exp) & (idx["dataset"] == ds)]
    if len(r) == 0: return None
    return float(r.iloc[0]["auc_recomputed"])

rows = []
sidecar = {
    "task": "M7 paired probe minus zero-shot, per dataset, Qwen2.5-Omni",
    "model_id": "Qwen/Qwen2.5-Omni-7B",
    "seed": SEED, "n_boot": NBOOT,
    "bootstrap": "speaker resampling with replacement, PAIRED (one speaker draw per replicate, both AUCs inside it), percentile 2.5/97.5, rng reseeded per dataset cell",
    "auc": "recomputed from per-clip scores with the rank (Mann-Whitney) formula, scipy rankdata ties averaged",
    "folds": "NOT recomputed - the saved nested out-of-fold predictions in omni_final/omni_<ds>_enc_nested_oof.csv were used as-is (they already carry the saved fold assignment baked into the OOF prediction). No GroupKFold was run.",
    "shell_command": CMD,
    "datasets": {},
}

print("=" * 110)
print("M7  paired encoder-probe minus zero-shot-answer, Qwen2.5-Omni, seven datasets")
print("=" * 110)

for ds in DATASETS:
    fp = os.path.join(OMNI, "omni_%s_enc_nested_oof.csv" % ds)
    fz = os.path.join(OMNI, "omni_%s_zeroshot_scores.csv" % ds)
    if not os.path.exists(fp):
        disc("HIGH", "%s: no omni_%s_enc_nested_oof.csv - dataset skipped" % (ds, ds)); continue
    if not os.path.exists(fz):
        disc("HIGH", "%s: no omni_%s_zeroshot_scores.csv - dataset skipped" % (ds, ds)); continue

    dp = pd.read_csv(fp)
    dz = pd.read_csv(fz)
    np_, nz_ = len(dp), len(dz)

    m = dp.merge(dz[["clip", "p_yes", "prompt"]], on="clip", how="inner", suffixes=("", "_z"))
    n = len(m)
    lost_p = np_ - n
    lost_z = nz_ - n
    if lost_p or lost_z:
        disc("MEDIUM", "%s: join kept %d clips; %d lost from probe file (n=%d), %d lost from zeroshot file (n=%d)"
             % (ds, n, lost_p, np_, lost_z, nz_))

    # label consistency check
    lz = dz.set_index("clip")["label"]
    lbl_mismatch = int((m["label"].values != lz.reindex(m["clip"]).values).sum())
    if lbl_mismatch:
        disc("HIGH", "%s: %d clips disagree on label between probe oof and zeroshot csv" % (ds, lbl_mismatch))

    y = m["label"].values.astype(int)
    sp = m["p_probe"].values.astype(float)
    sz = m["p_yes"].values.astype(float)
    spk = m["speaker"].astype(str).values
    nspk = len(np.unique(spk))
    prompt = str(m["prompt"].iloc[0])

    a_probe = auc(y, sp)
    a_zero  = auc(y, sz)
    d_obs   = a_probe - a_zero

    # ---- paired speaker bootstrap ----
    rng = np.random.default_rng(SEED)
    uspk = np.unique(spk)
    byspk = {s: np.flatnonzero(spk == s) for s in uspk}
    bp, bz, bd = [], [], []
    for _ in range(NBOOT):
        draw = rng.choice(uspk, size=len(uspk), replace=True)
        ii = np.concatenate([byspk[s] for s in draw])
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        ap = auc(yy, sp[ii]); az = auc(yy, sz[ii])
        if np.isnan(ap) or np.isnan(az):
            continue
        bp.append(ap); bz.append(az); bd.append(ap - az)
    usable = len(bd)
    bp = np.array(bp); bz = np.array(bz); bd = np.array(bd)
    p_lo, p_hi = np.percentile(bp, [2.5, 97.5])
    z_lo, z_hi = np.percentile(bz, [2.5, 97.5])
    d_lo, d_hi = np.percentile(bd, [2.5, 97.5])
    excl = bool(d_lo > 0 or d_hi < 0)
    if usable < NBOOT:
        disc("LOW", "%s: %d of %d bootstrap draws usable (%d degenerate, one class only)"
             % (ds, usable, NBOOT, NBOOT - usable))

    # ---- cross-check vs master_lookup and INDEX ----
    ml_probe, ml_n, ml_est = ml_get(ds, "encoder probe")
    ml_zero,  ml_nz, ml_estz = ml_get(ds, "zero shot answer")
    ix_probe = idx_get("probe_omni_enc", ds)
    ix_zero  = idx_get("zeroshot_omni", ds)

    if ml_probe is not None and round(a_probe, 4) != round(ml_probe, 4):
        disc("HIGH", "%s encoder probe: recomputed from omni_%s_enc_nested_oof.csv = %.4f, master_lookup.csv = %.4f (estimator there: %s). The OOF csv is the SINGLE-SPLIT nested run; master_lookup carries the mean of 5 repeats, so these are different estimators of the same quantity, not a data error."
             % (ds, ds, a_probe, ml_probe, ml_est))
    if ml_zero is not None and round(a_zero, 4) != round(ml_zero, 4):
        disc("HIGH", "%s zero shot answer: recomputed = %.4f, master_lookup.csv = %.4f" % (ds, a_zero, ml_zero))
    if ml_n is not None and ml_n != n:
        disc("HIGH", "%s: n after join = %d, master_lookup n = %d" % (ds, n, ml_n))
    if ix_probe is not None and round(a_probe, 4) != round(ix_probe, 4):
        disc("HIGH", "%s encoder probe: recomputed %.4f != perclip INDEX.csv auc_recomputed %.4f (same source csv)" % (ds, a_probe, ix_probe))
    if ix_zero is not None and round(a_zero, 4) != round(ix_zero, 4):
        disc("HIGH", "%s zero shot: recomputed %.4f != perclip INDEX.csv auc_recomputed %.4f (same source csv)" % (ds, a_zero, ix_zero))

    # ---- per clip out ----
    out = m[["clip", "speaker", "label", "p_probe", "p_yes"]].copy()
    out["diff_rank_pct"] = st.rankdata(sp) / len(sp) - st.rankdata(sz) / len(sz)
    pc = os.path.join(PERCLIP, "%s.csv" % ds)
    out.to_csv(pc, index=False)

    rows.append(dict(dataset=ds, n=n, n_spk=nspk,
                     probe_auc=round(a_probe, 4), probe_lo=round(p_lo, 4), probe_hi=round(p_hi, 4),
                     zeroshot_auc=round(a_zero, 4), zero_lo=round(z_lo, 4), zero_hi=round(z_hi, 4),
                     paired_diff=round(d_obs, 4), diff_lo=round(d_lo, 4), diff_hi=round(d_hi, 4),
                     excludes_zero=excl, boot_usable=usable, boot_total=NBOOT,
                     n_probe_file=np_, n_zeroshot_file=nz_, lost_probe=lost_p, lost_zeroshot=lost_z,
                     ml_probe_auc=ml_probe, ml_zeroshot_auc=ml_zero, ml_n=ml_n,
                     idx_probe_auc=ix_probe, idx_zeroshot_auc=ix_zero,
                     probe_stage="enc (nested encoder probe, saved OOF)",
                     probe_file=fp, zeroshot_file=fz, perclip_file=pc))

    sidecar["datasets"][ds] = dict(
        probe_file=fp, zeroshot_file=fz, perclip_out=pc,
        sha256_first1mb={fp: sha1mb(fp), fz: sha1mb(fz)},
        n=n, n_speakers=nspk, n_probe_file=np_, n_zeroshot_file=nz_,
        lost_from_probe=lost_p, lost_from_zeroshot=lost_z,
        label_mismatches=lbl_mismatch, prompt=prompt,
        probe_auc=round(a_probe, 6), zeroshot_auc=round(a_zero, 6),
        paired_diff=round(d_obs, 6), diff_ci=[round(d_lo, 6), round(d_hi, 6)],
        boot_usable=usable, boot_total=NBOOT,
        master_lookup_probe=ml_probe, master_lookup_zeroshot=ml_zero,
        master_lookup_probe_estimator=ml_est,
        index_csv_probe=ix_probe, index_csv_zeroshot=ix_zero)

    print("%-11s n=%4d n_spk=%4d | probe %.4f [%.4f,%.4f] | zeroshot %.4f [%.4f,%.4f] | diff %+.4f [%+.4f,%+.4f] excl0=%s (%d/%d draws)"
          % (ds, n, nspk, a_probe, p_lo, p_hi, a_zero, z_lo, z_hi, d_obs, d_lo, d_hi, excl, usable, NBOOT))

df = pd.DataFrame(rows)
csv_out = os.path.join(OUT, "M7_gap_per_dataset.csv")
df.to_csv(csv_out, index=False)

mean_diff = float(df["paired_diff"].mean())
n_excl = int(df["excludes_zero"].sum())
sidecar["mean_paired_diff_across_datasets"] = round(mean_diff, 4)
sidecar["n_datasets_excluding_zero"] = n_excl
sidecar["n_datasets"] = len(df)
sidecar["out_csv"] = csv_out

print("-" * 110)
print("MEAN paired difference across %d datasets = %+.4f ; %d of %d have 95%% intervals excluding zero"
      % (len(df), mean_diff, n_excl, len(df)))

# ---- rows fragment ----
tsv = os.path.join(ROWS, "M7.tsv")
with open(tsv, "w") as f:
    f.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n")
    for r in rows:
        f.write("M7_%s_probe\tQwen2.5-Omni nested encoder probe AUC, %s\t%.4f\t%.4f\t%.4f\t%d\t%d\t%s\n"
                % (r["dataset"], r["dataset"], r["probe_auc"], r["probe_lo"], r["probe_hi"], r["n"], r["n_spk"], r["probe_file"]))
        f.write("M7_%s_zeroshot\tQwen2.5-Omni zero-shot audio answer AUC, %s\t%.4f\t%.4f\t%.4f\t%d\t%d\t%s\n"
                % (r["dataset"], r["dataset"], r["zeroshot_auc"], r["zero_lo"], r["zero_hi"], r["n"], r["n_spk"], r["zeroshot_file"]))
        f.write("M7_%s_gap\tpaired probe minus zero-shot AUC, %s\t%.4f\t%.4f\t%.4f\t%d\t%d\t%s\n"
                % (r["dataset"], r["dataset"], r["paired_diff"], r["diff_lo"], r["diff_hi"], r["n"], r["n_spk"], r["perclip_file"]))
    f.write("M7_mean_gap\tmean paired probe minus zero-shot AUC across %d datasets\t%.4f\t\t\t%d\t\t%s\n"
            % (len(df), mean_diff, int(df["n"].sum()), csv_out))

# ---- sidecar ----
side = os.path.join(OUT, "M7_gap_per_dataset.json")
with open(side, "w") as f:
    json.dump(sidecar, f, indent=1)

# ---- discrepancies ----
if disc_lines:
    with open(DISC, "a") as f:
        f.write("\n## PART16 M7 - paired encoder probe vs zero-shot answer (Qwen2.5-Omni, 7 datasets)\n")
        f.write("run: %s\n\n" % CMD)
        f.write("\n".join(disc_lines) + "\n")

print("\nwrote: %s" % csv_out)
print("wrote: %s" % side)
print("wrote: %s" % tsv)
print("wrote per-clip joins: %s/<ds>.csv" % PERCLIP)
print("discrepancies appended: %d -> %s" % (len(disc_lines), DISC))
