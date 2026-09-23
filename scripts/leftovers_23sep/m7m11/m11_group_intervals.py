#!/usr/local/bin/python3
"""m7m11 (b): M11 MDVR-KCL group-level intervals, cause of the script vs verifier disagreement, and the fix.

Cause (read from the code, confirmed by recomputing both ways below):
  part16/M11_kcl_wer.py boot_mean() resamples ONLY the speakers of the group being summarised
  (16 PD or 21 control, a fresh default_rng(0) per cell, N = group size).
  part16/M11_verify.py ci() and part16/verify/M11_verify.py boot_ci_grouped(mode="all37") resample ALL 37
  speakers from one default_rng(0) stream and then average whichever rows of that group the draw happened to
  contain (group size varies from draw to draw). These are two different bootstraps, so the group-mean intervals
  differ although every per-clip value agrees. part16/M11_robust.py (MEDIAN, >=20 words, POOLED rows) used the
  all-37 scheme too, so rows/M11.tsv mixes the two schemes.
Rule used for the final intervals (the brief's rule, and the main script's): a fresh default_rng(0) per cell,
  idx = rng.choice(N, size=N, replace=True) over the unique sorted speaker ids of the cell. For a one-group cell
  the cell's speakers are that group's speakers (and for the >=20-word rows, that group's speakers inside the
  subset). For a between-group difference, N = the cell's speakers (all 37; the 31 inside the >=20-word subset) and both groups use the same idx.
KCL has one clip per speaker, so speaker = clip.
"""
import os, json, hashlib, platform
import numpy as np, pandas as pd, scipy, scipy.stats as st

CSV = "<local data dir>/release/edaic_rerun/part16/M11_kcl_wer.csv"
OLD = "<local data dir>/release/edaic_rerun/part16/rows/M11.tsv"
OUT = "<local data dir>/leftovers_23sep/m7m11"
NB, SEED = 2000, 0
CLOSES = "STATUS_LEDGER 5.4 / audit_p16 item 4 (M11 group-mean CIs, script vs verifier)"

d = pd.read_csv(CSV)
assert len(d) == 37 and d.spk.nunique() == 37
spk = d.spk.astype(str).values
grp = d.group.values
old = pd.read_csv(OLD, sep="\t", dtype=str)
oldmap = {r.what: r for r in old.itertuples()}


def draws_over(rowsel):
    """rowsel: boolean mask of rows in the cell. Fresh rng(0); idx over the unique sorted speaker ids of the cell."""
    u = np.unique(spk[rowsel])
    rows = {s: np.flatnonzero((spk == s) & rowsel) for s in u}
    rng = np.random.default_rng(SEED)
    return [np.concatenate([rows[u[i]] for i in rng.choice(len(u), size=len(u), replace=True)]) for _ in range(NB)], len(u)


ALL = np.ones(37, bool)
def pct(v):
    v = np.asarray([x for x in v if x is not None and not np.isnan(x)], float)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def stat_mean(col):
    v = d[col].values.astype(float)
    return lambda ii: float(v[ii].mean()) if len(ii) else np.nan
def stat_median(col):
    v = d[col].values.astype(float)
    return lambda ii: float(np.median(v[ii])) if len(ii) else np.nan
def stat_pooled(ii):
    if not len(ii): return np.nan
    num = (d.xsys_S.values[ii] + d.xsys_D.values[ii] + d.xsys_I.values[ii]).sum(); den = d.xsys_n_ref_words.values[ii].sum()
    return float(num / den) if den else np.nan

GE20 = d.xsys_n_ref_words.values >= 20
CELLS = [  # tsv key prefix, statistic factory, subset mask, label
    ("B_xsys_disagree_mean", stat_mean("xsys_disagree"), ALL, "cross-system disagreement (large-v3 vs medium), mean"),
    ("C_mean_token_logprob_mean", stat_mean("mean_token_logprob"), ALL, "Whisper large-v3 mean per-token logprob, mean"),
    ("C_compression_ratio_mean", stat_mean("compression_ratio"), ALL, "zlib compression ratio, mean"),
    ("D_n_words_30s_mean", stat_mean("n_words_30s"), ALL, "words in the 30 s window, mean"),
    ("F_wer_vs_local_copy_mean", stat_mean("wer_local"), ALL, "large-v3 vs the other stored transcript copy, WER mean"),
    ("B_xsys_disagree_MEDIAN", stat_median("xsys_disagree"), ALL, "cross-system disagreement, median"),
    ("B_xsys_disagree_ge20w_MEAN", stat_mean("xsys_disagree"), GE20, "cross-system disagreement, mean over clips with >= 20 words"),
    ("B_xsys_disagree_POOLED", stat_pooled, ALL, "cross-system disagreement, pooled sum(S+D+I)/sum(N)"),
]

rows = []
def oldci(key):
    r = oldmap.get(key)
    if r is None: return None, None, None, None, None
    f = lambda x: None if (x is None or (isinstance(x, float) and np.isnan(x)) or str(x) in ("", "nan")) else float(x)
    return f(r.value), f(r.lo), f(r.hi), r.n, r.n_spk

for key, fn, sub, label in CELLS:
    vals = {}
    for g in ("PD", "control"):
        cell = sub & (grp == g)
        v = fn(np.flatnonzero(cell)); vals[g] = v
        # rule: within the cell's own speakers
        D, nsp = draws_over(cell)
        lo, hi, us = pct([fn(ii) for ii in D])
        # verifier / M11_robust scheme: all 37 speakers, then keep this cell's rows
        Dall, _ = draws_over(ALL)
        lo37, hi37, us37 = pct([fn(ii[cell[ii]]) if cell[ii].any() else np.nan for ii in Dall])
        tkey = f"{key}_{g}" if key.endswith(("MEDIAN", "MEAN", "POOLED")) else f"{key}_{g}"
        ov, olo, ohi, on, ons = oldci(tkey)
        rows.append(dict(id=f"M11fix_{tkey}", what=f"MDVR-KCL {g}: {label}", value=round(v, 4), lo=round(lo, 4), hi=round(hi, 4),
                         n=int(cell.sum()), n_spk=nsp, usable_draws=us, rule="within-cell speakers (final)",
                         lo_all37=round(lo37, 4), hi_all37=round(hi37, 4), usable_all37=us37,
                         old_value=ov, old_lo=olo, old_hi=ohi, old_n=on, old_n_spk=ons,
                         old_matches_rule=(olo is not None and round(lo, 4) == olo and round(hi, 4) == ohi),
                         old_matches_all37=(olo is not None and round(lo37, 4) == olo and round(hi37, 4) == ohi),
                         value_full=v, lo_full=lo, hi_full=hi, source=CSV, closes=CLOSES))
    # paired between-group difference: the cell's speakers (all 37, or the 31 inside the >=20-word subset), same idx for both groups
    Dall, n37 = draws_over(sub)
    diffs = []
    for ii in Dall:
        a = ii[(grp[ii] == "PD") & sub[ii]]; b = ii[(grp[ii] == "control") & sub[ii]]
        diffs.append(fn(a) - fn(b) if len(a) and len(b) else np.nan)
    dlo, dhi, dus = pct(diffs)
    D37, _ = draws_over(ALL)   # old scheme for the record: all 37 speakers, subset rows kept
    d37 = []
    for ii in D37:
        a = ii[(grp[ii] == "PD") & sub[ii]]; b = ii[(grp[ii] == "control") & sub[ii]]
        d37.append(fn(a) - fn(b) if len(a) and len(b) else np.nan)
    dlo37, dhi37, dus37 = pct(d37)
    dv = vals["PD"] - vals["control"]
    dkey = f"{key}_diff_PD_minus_control" if not key.endswith("_mean") else key[:-5] + "_diff_PD_minus_control"
    ov, olo, ohi, on, ons = oldci(dkey)
    rows.append(dict(id=f"M11fix_{dkey}", what=f"MDVR-KCL PD minus control: {label}", value=round(dv, 4), lo=round(dlo, 4), hi=round(dhi, 4),
                     n=int(sub.sum()), n_spk=n37, usable_draws=dus, rule="the cell's speakers (37, or 31 for the >=20-word subset), same idx for both groups (paired)",
                     lo_all37=round(dlo37, 4), hi_all37=round(dhi37, 4), usable_all37=dus37,
                     old_value=ov, old_lo=olo, old_hi=ohi, old_n=on, old_n_spk=ons,
                     old_matches_rule=(olo is not None and round(dlo, 4) == olo and round(dhi, 4) == ohi),
                     old_matches_all37=(olo is not None and round(dlo37, 4) == olo and round(dhi37, 4) == ohi),
                     value_full=dv, lo_full=dlo, hi_full=dhi, source=CSV, closes=CLOSES))

df = pd.DataFrame(rows)
df.to_csv(OUT + "/m11_group_intervals.csv", index=False)
for r in rows:
    print(f"{r['id']:52s} {r['value']:>9.4f} rule[{r['lo']:.4f}, {r['hi']:.4f}] n={r['n']:>2} | all37[{r['lo_all37']:.4f}, {r['hi_all37']:.4f}] | old {r['old_lo']},{r['old_hi']} n={r['old_n']} | old=rule {r['old_matches_rule']} old=all37 {r['old_matches_all37']}")

side = dict(task="m7m11 (b): M11 KCL group-level intervals, cause and fix",
            cause=("M11_kcl_wer.py boot_mean() resamples only the speakers of the summarised group (N=16 PD or 21 control, fresh rng(0) per cell); "
                   "M11_verify.py and verify/M11_verify.py (mode all37) resample all 37 speakers from one rng(0) stream and keep the group's rows, "
                   "so the group size varies per draw. Different estimators, not an arithmetic error. M11_robust.py used the all-37 scheme, so rows/M11.tsv mixed both."),
            which_was_wrong=("The verifier (and M11_robust.py) for one-group cells: they did not resample the speakers the original script's cell used, "
                             "which the brief's rule requires. The main script's group-mean intervals stand. The six M11_robust.py group intervals "
                             "(MEDIAN, >=20 words, POOLED) are recomputed here under the same rule, and the group rows now carry n = 16 or 21 "
                             "(rows/M11.tsv printed n = 37 for the group means)."),
            rule="fresh numpy default_rng(0) per cell; idx = rng.choice(N, size=N, replace=True) over the unique sorted speaker ids of the cell; "
                 "one-group cell: that group's speakers (>=20-word rows: that group's speakers inside the subset); difference: the cell's speakers (37, or the 31 inside the >=20-word subset), same idx; "
                 "2000 draws; percentiles 2.5/97.5; a draw is dropped only if a group is empty (count in usable_draws)",
            new_rows="paired PD minus control intervals for MEDIAN, >=20 words and POOLED did not exist before (rows/M11.tsv had the point only); added under the paired rule",
            command="/usr/local/bin/python3 '" + os.path.abspath(__file__) + "'",
            env=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__, machine=platform.machine()),
            inputs_sha256={CSV: hashlib.sha256(open(CSV, "rb").read()).hexdigest(), OLD: hashlib.sha256(open(OLD, "rb").read()).hexdigest()},
            n=37, n_pd=int((grp == "PD").sum()), n_control=int((grp == "control").sum()),
            out_csv=OUT + "/m11_group_intervals.csv")
json.dump(side, open(OUT + "/m11_group_intervals.sidecar.json", "w"), indent=1)
print("wrote", OUT + "/m11_group_intervals.csv")
