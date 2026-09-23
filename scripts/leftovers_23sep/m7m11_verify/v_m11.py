# Independent M11 verifier. Stats are computed from speaker draw COUNTS (weights), not from concatenated rows.
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); from vcore import speaker_draw_weights, pct, NB
W = "<local data dir>/leftovers_23sep/verify/m7m11_work"
d = pd.read_csv("<local data dir>/release/edaic_rerun/part16/M11_kcl_wer.csv")
d["spk"] = d.spk.astype(str)
d["xsys_err"] = d.xsys_S + d.xsys_D + d.xsys_I

def wmean(v, w): return float((v * w).sum() / w.sum())
def wmedian(v, w):
    # median of the multiset where row i appears w_i times (w integer counts); np.median convention (mean of 2 middles)
    rep = np.repeat(v, w.astype(int)); return float(np.median(rep))
def wpooled(sub, w): return float((sub.xsys_err.values * w).sum() / (sub.xsys_n_ref_words.values * w).sum())

STATS = {
    "mean": lambda sub, col, w: wmean(sub[col].values.astype(float), w),
    "median": lambda sub, col, w: wmedian(sub[col].values.astype(float), w),
    "pooled": lambda sub, col, w: wpooled(sub, w),
}

def one_group(sub, col, stat):
    f = STATS[stat]; spl = sorted(set(sub.spk)); pt = f(sub, col, np.ones(len(sub)))
    b = [f(sub, col, w) for w, _ in speaker_draw_weights(sub.spk.values, spl) if w.sum() > 0]
    return pt, pct(b), len(b), len(sub), len(spl)

def paired(sub, col, stat):
    f = STATS[stat]; spl = sorted(set(sub.spk)); isPD = (sub.group.values == "PD"); isC = (sub.group.values == "control")
    pt = f(sub[isPD], col, np.ones(isPD.sum())) - f(sub[isC], col, np.ones(isC.sum()))
    b = []
    for w, _ in speaker_draw_weights(sub.spk.values, spl):
        wp, wc = w[isPD], w[isC]
        if wp.sum() == 0 or wc.sum() == 0: continue
        b.append(f(sub[isPD], col, wp) - f(sub[isC], col, wc))
    return pt, pct(b), len(b), len(sub), len(spl)

def all37_group(sub_all, col, stat, g, keep_mask=None):
    """the old (all-37) scheme, for the record: resample all 37 speakers, keep the group's rows"""
    f = STATS[stat]; spl = sorted(set(d.spk)); mk = (d.group.values == g) & (keep_mask if keep_mask is not None else True)
    b = []
    for w, _ in speaker_draw_weights(d.spk.values, spl):
        ww = w[mk]
        if ww.sum() == 0: continue
        b.append(f(d[mk], col, ww))
    return pct(b)

ge = d[d.xsys_n_ref_words >= 20]
cells = {}
for key, col in [("B_xsys_disagree", "xsys_disagree"), ("C_mean_token_logprob", "mean_token_logprob"),
                 ("C_compression_ratio", "compression_ratio"), ("D_n_words_30s", "n_words_30s"), ("F_wer_vs_local_copy", "wer_local")]:
    for g in ("PD", "control"):
        cells[f"M11fix_{key}_mean_{g}"] = one_group(d[d.group == g], col, "mean") + (all37_group(d, col, "mean", g),)
    cells[f"M11fix_{key}_diff_PD_minus_control"] = paired(d, col, "mean")
for g in ("PD", "control"):
    cells[f"M11fix_B_xsys_disagree_MEDIAN_{g}"] = one_group(d[d.group == g], "xsys_disagree", "median") + (all37_group(d, "xsys_disagree", "median", g),)
    cells[f"M11fix_B_xsys_disagree_ge20w_MEAN_{g}"] = one_group(ge[ge.group == g], "xsys_disagree", "mean") + (all37_group(d, "xsys_disagree", "mean", g, (d.xsys_n_ref_words >= 20).values),)
    cells[f"M11fix_B_xsys_disagree_POOLED_{g}"] = one_group(d[d.group == g], "xsys_disagree", "pooled") + (all37_group(d, "xsys_disagree", "pooled", g),)
cells["M11fix_B_xsys_disagree_MEDIAN_diff_PD_minus_control"] = paired(d, "xsys_disagree", "median")
cells["M11fix_B_xsys_disagree_ge20w_MEAN_diff_PD_minus_control"] = paired(ge, "xsys_disagree", "mean")
cells["M11fix_B_xsys_disagree_POOLED_diff_PD_minus_control"] = paired(d, "xsys_disagree", "pooled")
for k, v in cells.items():
    extra = f"  all37 [{v[5][0]:.4f}, {v[5][1]:.4f}]" if len(v) > 5 else ""
    print(f"{k:58s} {v[0]:.4f} [{v[1][0]:.4f}, {v[1][1]:.4f}] usable {v[2]} n {v[3]} spk {v[4]}{extra}")
json.dump(cells, open(f"{W}/v_m11_results.json", "w"), indent=1, default=float)
