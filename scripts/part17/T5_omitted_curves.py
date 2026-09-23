#!/usr/bin/env python3
"""PART 17 task 5. Do the four datasets left out of Figure 1 follow their condition's pattern?

For every dataset: the Qwen2.5-Omni per-layer ENCODER probe curve (32 audio encoder transformer layers,
column auc_oof = pooled out-of-fold AUC, GroupKFold(5) by speaker, the column Figure 1 plots) against the
dataset's zero-shot answer AUC (recomputed here from per-clip p_yes with the rank formula).

Figure 1 script: make_fig1_v5.py is not on disk. The newest script is
  <local data dir>/release/edaic_rerun/make_fig1_v4.py
which wrote paper1_final/figures/fig1_gap.pdf (variant full) and whose grayscale render is fig1_gray_v5-1.png.
It plots: PC-GITA and Pitt from release/overnight/curves/omni/<ds>_encoder.csv column "auc" (a mirror of
omni_final auc_oof), E-DAIC from edaic_rerun/variants/o25_<variant>_encoder_perlayer.csv column auc_oof,
32 encoder layers + 29 LM stages, answer lines hard-coded PC-GITA 0.5635, Pitt 0.6578,
E-DAIC full 0.8285 / mid30 0.7205 / mid300 0.8122.

Counting rule: over the 32 encoder rows (layer 0..31, one row per transformer layer, no pre-transformer
stage), count layers with probe auc_oof STRICTLY greater than the full-precision recomputed answer AUC.
Sensitivities: (a) auc_mean - auc_std > answer (fold mean minus fold std, np.nanstd ddof 0 in the probe
script); (b) auc_oof > upper 97.5 percentile of the answer AUC's speaker bootstrap interval.

Writes only under release/edaic_rerun/part17/. Reads omni_final read-only.
"""
import os, sys, json, csv, hashlib, datetime
import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUT = "<local data dir>/release/edaic_rerun/part17"
OMNI = "<local data dir>/release/omni_final"
MIRROR = "<local data dir>/release/overnight/curves/omni"
VAR = "<local data dir>/release/edaic_rerun/variants"
MASTER = "<local data dir>/release/master/master_lookup.csv"
FIGSCRIPT = "<local data dir>/release/edaic_rerun/make_fig1_v4.py"
DISC = "<local data dir>/release/edaic_rerun/DISCREPANCIES.md"
CMD = " ".join(["/usr/local/bin/python3"] + [os.path.abspath(sys.argv[0])] + sys.argv[1:])
B = 2000
N_ENC = 32

assert os.path.abspath(OUT).startswith("<local data dir>/release/edaic_rerun/part17")

# the answer values Figure 1 draws (hard-coded in make_fig1_v4.py lines 51-53)
FIG_ANSWER = {"pcgita": 0.5635, "pitt": 0.6578, "edaic_full": 0.8285, "edaic_mid30": 0.7205, "edaic_mid300": 0.8122}

# (key, display, condition, plotted, curve file, curve column, zero-shot per-clip file, master_lookup dataset key)
DS = [
    ("pitt", "Pitt", "AD", "plotted", f"{MIRROR}/pitt_encoder.csv", "auc", f"{OMNI}/omni_pitt_zeroshot_scores.csv", "pitt"),
    ("adresso", "ADReSSo", "AD", "omitted", f"{OMNI}/omni_adresso_encoder_perlayer.csv", "auc_oof", f"{OMNI}/omni_adresso_zeroshot_scores.csv", "adresso"),
    ("adress2020", "ADReSS-2020", "AD", "omitted", f"{OMNI}/omni_adress2020_encoder_perlayer.csv", "auc_oof", f"{OMNI}/omni_adress2020_zeroshot_scores.csv", "adress2020"),
    ("pcgita", "PC-GITA", "PD", "plotted", f"{MIRROR}/pcgita_encoder.csv", "auc", f"{OMNI}/omni_pcgita_zeroshot_scores.csv", "pcgita"),
    ("neurovoz", "NeuroVoz", "PD", "omitted", f"{OMNI}/omni_neurovoz_encoder_perlayer.csv", "auc_oof", f"{OMNI}/omni_neurovoz_zeroshot_scores.csv", "neurovoz"),
    ("kcl", "MDVR-KCL", "PD", "omitted", f"{OMNI}/omni_kcl_encoder_perlayer.csv", "auc_oof", f"{OMNI}/omni_kcl_zeroshot_scores.csv", "kcl"),
    ("edaic_full", "E-DAIC (full, 900 s cap)", "MDD", "plotted (fig1_gap.pdf)", f"{VAR}/o25_full_encoder_perlayer.csv", "auc_oof", f"{VAR}/o25_full_zeroshot_scores.csv", "edaic_full"),
    ("edaic_mid300", "E-DAIC (middle 300 s)", "MDD", "plotted variant (fig1_gap_300.pdf)", f"{VAR}/o25_mid300_encoder_perlayer.csv", "auc_oof", f"{VAR}/o25_mid300_zeroshot_scores.csv", "edaic_mid300"),
    ("edaic_mid30", "E-DAIC (middle 30 s)", "MDD", "plotted variant (fig1_gap_mid.pdf)", f"{VAR}/o25_mid30_encoder_perlayer.csv", "auc_oof", f"{VAR}/o25_mid30_zeroshot_scores.csv", "edaic_mid30"),
    ("edaic_first30", "E-DAIC (first 30 s, omni_final)", "MDD", "not plotted, reference", f"{OMNI}/omni_edaic_encoder_perlayer.csv", "auc_oof", f"{OMNI}/omni_edaic_zeroshot_scores.csv", "edaic"),
]
# full-stat file for each curve (auc_mean, auc_std, auc_oof); for the mirrors it is the omni_final source
STATFILE = {"pitt": f"{OMNI}/omni_pitt_encoder_perlayer.csv", "pcgita": f"{OMNI}/omni_pcgita_encoder_perlayer.csv"}


def sha1mb(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read(1 << 20)).hexdigest()


def rank_auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    n1 = int(y.sum()); n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)  # ties averaged
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spk_boot(df):
    rng = np.random.default_rng(0)  # fresh per cell
    spk = df["speaker"].astype(str).values
    uniq = np.unique(spk)
    idx_by = {u: np.where(spk == u)[0] for u in uniq}
    y = df["label"].values.astype(int); s = df["p_yes"].values.astype(float)
    vals = []
    for _ in range(B):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by[u] for u in draw])
        a = rank_auc(y[ii], s[ii])
        if not np.isnan(a):
            vals.append(a)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int(len(vals))


def disc(sev, text):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(DISC, "a") as fh:
        fh.write(f"\n- [PART17] [{sev}] T5 ({stamp}): {text}\n")


master = pd.read_csv(MASTER)
m_o25 = master[(master["model"] == "Qwen2.5-Omni") & (master["stream"] == "zero shot answer")]

curve_rows, summary_rows, perclip, sidecar_entries, frag = [], [], [], [], []
disc_notes = [("MEDIUM", "make_fig1_v5.py does not exist on disk (Desktop, release, paper1_final, public repo clones searched). "
                         "The newest figure script is edaic_rerun/make_fig1_v4.py; it wrote paper1_final/figures/fig1_gap.pdf "
                         "(E-DAIC variant full, answer 0.8285) at 22:25 on 22 Sep, and fig1_gray_v5-1.png is the grayscale render of "
                         "that output. Which E-DAIC variant sits in Overleaf cannot be checked from disk. T5 used make_fig1_v4.py.")]
for key, disp, cond, plotted, cfile, ccol, zfile, mkey in DS:
    cur = pd.read_csv(cfile)
    zs = pd.read_csv(zfile)
    print(f"[open] {cfile}  columns {list(cur.columns)}  rows {len(cur)}")
    print(f"[open] {zfile}  columns {list(zs.columns)}  rows {len(zs)}")
    assert len(cur) == N_ENC, f"{cfile} has {len(cur)} rows, expected {N_ENC}"
    assert list(cur["layer"]) == list(range(N_ENC)), f"{cfile} layer index is not 0..31"
    stat = pd.read_csv(STATFILE.get(key, cfile))
    assert len(stat) == N_ENC and list(stat["layer"]) == list(range(N_ENC))
    if key in STATFILE:  # mirror must equal omni_final auc_oof
        d = float(np.max(np.abs(cur["auc"].values - stat["auc_oof"].values)))
        print(f"[check] mirror {os.path.basename(cfile)} 'auc' vs {os.path.basename(STATFILE[key])} auc_oof: max abs diff {d:.2e}")
        if d > 1e-12:
            disc_notes.append(("MEDIUM", f"mirror {cfile} column auc differs from {STATFILE[key]} auc_oof by up to {d:.3e}"))
    probe = cur[ccol].values.astype(float)
    amean = stat["auc_mean"].values.astype(float); astd = stat["auc_std"].values.astype(float)

    y = zs["label"].values.astype(int); s = zs["p_yes"].values.astype(float)
    assert not np.isnan(s).any()
    ans = rank_auc(y, s)
    lo, hi, used = spk_boot(zs)
    n = len(zs); nspk = zs["speaker"].astype(str).nunique(); npos = int(y.sum())

    mrow = m_o25[m_o25["dataset"] == mkey]
    m_val = float(mrow["auc"].iloc[0]) if len(mrow) else np.nan
    m_n = int(mrow["n"].iloc[0]) if len(mrow) else -1
    m_match = (len(mrow) == 1) and abs(round(ans, 4) - m_val) < 1e-9 and m_n == n
    if not m_match:
        disc_notes.append(("LOW" if len(mrow) else "MEDIUM",
                           f"{disp}: recomputed zero-shot answer {ans:.4f} n={n} vs master_lookup Qwen2.5-Omni/{mkey} "
                           f"{m_val} n={m_n}"))
    fig_ans = FIG_ANSWER.get(key)
    if fig_ans is not None and abs(round(ans, 4) - fig_ans) > 1e-9:
        disc_notes.append(("MEDIUM", f"{disp}: Figure 1 draws answer {fig_ans} but per-clip recompute is {ans:.4f}"))

    above = probe > ans
    strict = (amean - astd) > ans
    above_hi = probe > hi
    n_above, n_strict, n_above_hi = int(above.sum()), int(strict.sum()), int(above_hi.sum())
    # count with the figure's own 4-decimal answer value, to show rounding never flips a layer
    n_above_fig = int((probe > fig_ans).sum()) if fig_ans is not None else None
    margin = probe - ans
    closest = int(np.argmin(np.abs(margin)))
    imax = int(np.argmax(probe)); imin = int(np.argmin(probe))
    first_above = int(np.argmax(above)) if above.any() else None

    for l in range(N_ENC):
        curve_rows.append(dict(dataset=key, layer=l, probe_auc=round(float(probe[l]), 6), answer_auc=round(ans, 6),
                               above=int(above[l]), condition=cond, plotted=plotted, curve_column=ccol,
                               auc_mean=round(float(amean[l]), 6), auc_std=round(float(astd[l]), 6),
                               mean_minus_std_above=int(strict[l]), above_answer_ci_hi=int(above_hi[l]),
                               curve_file=cfile))
    for c, sp, lab, p in zip(zs["clip"], zs["speaker"], zs["label"], zs["p_yes"]):
        perclip.append(dict(dataset=key, clip=c, speaker=sp, label=int(lab), p_yes=float(p), source=zfile))

    srow = dict(dataset=key, display=disp, condition=cond, plotted=plotted, n_clips=n, n_speakers=nspk, n_pos=npos,
                answer_auc=round(ans, 4), answer_lo=round(lo, 4), answer_hi=round(hi, 4), boot_usable=used,
                master_lookup_value=m_val, master_lookup_n=m_n, master_match=bool(m_match),
                figure_answer=fig_ans if fig_ans is not None else "",
                probe_min=round(float(probe.min()), 4), probe_min_layer=imin,
                probe_median=round(float(np.median(probe)), 4),
                probe_max=round(float(probe.max()), 4), probe_max_layer=imax,
                layers_above=n_above, layers_total=N_ENC,
                layers_above_with_figure_value=n_above_fig if n_above_fig is not None else "",
                layers_mean_minus_std_above=n_strict, layers_above_answer_ci_hi=n_above_hi,
                first_layer_above=first_above if first_above is not None else "",
                closest_layer=closest, closest_margin=round(float(margin[closest]), 4),
                curve_file=cfile, curve_column=ccol, stat_file=STATFILE.get(key, cfile), zeroshot_file=zfile)
    summary_rows.append(srow)
    srcs = sorted({cfile, STATFILE.get(key, cfile), zfile, MASTER})
    sidecar_entries.append(dict(result=f"T5_{key}", dataset=key, condition=cond, plotted=plotted,
                                sources={p: sha1mb(p) for p in srcs}, n=n, n_speakers=nspk, seed=0,
                                bootstrap_draws=B, bootstrap_usable=used,
                                model_id="Qwen/Qwen2.5-Omni-7B", command=CMD))
    base = f"{OUT}/T5_omitted_summary.csv"
    frag.append((f"T5_{key}_answer", f"{disp} Qwen2.5-Omni zero-shot answer AUC (rank formula, speaker bootstrap)",
                 f"{ans:.4f}", f"{lo:.4f}", f"{hi:.4f}", n, nspk, zfile))
    frag.append((f"T5_{key}_layers_above", f"{disp} encoder layers (of 32) with probe auc_oof > answer AUC",
                 str(n_above), "", "", n, nspk, f"{OUT}/T5_omitted_curves.csv"))
    frag.append((f"T5_{key}_layers_strict", f"{disp} encoder layers (of 32) with auc_mean - auc_std > answer AUC",
                 str(n_strict), "", "", n, nspk, f"{OUT}/T5_omitted_curves.csv"))
    frag.append((f"T5_{key}_layers_above_ci", f"{disp} encoder layers (of 32) with probe auc_oof > answer AUC upper 97.5 bound",
                 str(n_above_hi), "", "", n, nspk, f"{OUT}/T5_omitted_curves.csv"))
    frag.append((f"T5_{key}_probe_max", f"{disp} encoder probe max auc_oof (layer {imax})",
                 f"{probe.max():.4f}", "", "", n, nspk, cfile))

# ---- write outputs
pd.DataFrame(curve_rows).to_csv(f"{OUT}/T5_omitted_curves.csv", index=False)
pd.DataFrame(summary_rows).to_csv(f"{OUT}/T5_omitted_summary.csv", index=False)
pd.DataFrame(perclip).to_csv(f"{OUT}/T5_zeroshot_perclip.csv", index=False)
os.makedirs(f"{OUT}/rows", exist_ok=True); os.makedirs(f"{OUT}/sidecars", exist_ok=True)
with open(f"{OUT}/rows/T5.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["id", "what", "value", "lo", "hi", "n", "n_spk", "file"])
    w.writerows(frag)
side = dict(task="PART17 task 5, omitted dataset encoder curves vs zero-shot answer",
            date=datetime.date.today().isoformat(), command=CMD, seed=0, bootstrap_draws=B,
            bootstrap="speakers resampled with replacement, numpy default_rng(0) fresh per dataset, percentile 2.5/97.5",
            auc="rank formula, scipy.stats.rankdata ties averaged",
            count_rule="over encoder rows layer 0..31 (32 transformer layers, no pre-transformer stage): probe auc_oof > full-precision recomputed answer AUC, strict",
            figure_script=FIGSCRIPT, figure_script_sha256_1mb=sha1mb(FIGSCRIPT),
            figure_script_note="make_fig1_v5.py is not on disk; make_fig1_v4.py wrote fig1_gap.pdf (variant full) and the fig1_gray_v5 render",
            outputs=[f"{OUT}/T5_omitted_curves.csv", f"{OUT}/T5_omitted_summary.csv", f"{OUT}/T5_zeroshot_perclip.csv", f"{OUT}/rows/T5.tsv"],
            results=sidecar_entries)
with open(f"{OUT}/T5_omitted_curves.json", "w") as fh:
    json.dump(side, fh, indent=1)
for e in sidecar_entries:
    with open(f"{OUT}/sidecars/{e['result']}.json", "w") as fh:
        json.dump(dict(e, outputs=side["outputs"], count_rule=side["count_rule"], figure_script=FIGSCRIPT), fh, indent=1)

for sev, t in disc_notes:
    disc(sev, t)

# ---- print
print()
print("dataset | cond | plotted | n | n_spk | answer [95% spk CI] | probe min/med/max (max layer) | above/32 | strict/32 | above CI-hi/32")
for r in summary_rows:
    print(f"T5 {r['display']} ({r['condition']}, {r['plotted']}): answer {r['answer_auc']:.4f} "
          f"[{r['answer_lo']:.4f}, {r['answer_hi']:.4f}] usable {r['boot_usable']}/{B}; encoder probe min {r['probe_min']:.4f} "
          f"(L{r['probe_min_layer']}) median {r['probe_median']:.4f} max {r['probe_max']:.4f} (L{r['probe_max_layer']}); "
          f"layers above answer {r['layers_above']}/32, mean-std above {r['layers_mean_minus_std_above']}/32, "
          f"above answer CI hi {r['layers_above_answer_ci_hi']}/32; master_lookup {r['master_lookup_value']} match {r['master_match']}; "
          f"n {r['n_clips']}, n_speakers {r['n_speakers']}, {OUT}/T5_omitted_summary.csv")
print("discrepancies appended:", len(disc_notes))
for sev, t in disc_notes:
    print("  ", sev, t)
