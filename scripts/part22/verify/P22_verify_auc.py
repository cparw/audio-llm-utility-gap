#!/usr/bin/env python3
"""PART 22 independent AUC verifier (Mac side). Written from scratch; imports nothing from the pod job.

Inputs
  --labels   part16 edaic_full_perclip.csv (pid,label,p_yes_answer,...) = truncated baseline 0.8278
  --windows  windows_full.csv (pid, win_dur_s, capped)
  --mine_lift, --mine_default   my own per-clip scores from P22_verify.py score (pid, p_yes_paper, p_yes_rule, ...)
  --pod      pod job lifted_all275.csv (pid, label, def_p_yes, lift_p_yes, ...)
AUC: explicit Mann-Whitney over all pos x neg pairs (ties 0.5) and trapezoid ROC; both must agree.
CI : 2000 draws, numpy default_rng(0) fresh per cell, speakers resampled with replacement (one clip per speaker here),
     2.5/97.5 percentiles; paired differences use one draw per replicate shared by both arms.
"""
import argparse, csv, json, sys
import numpy as np

D = 2000


def auc_pairs(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def auc_trapz(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    P, N = (y == 1).sum(), (y == 0).sum()
    tp, fp = [0.0], [0.0]
    for t in np.unique(s)[::-1]:
        tp.append(((s >= t) & (y == 1)).sum() / P); fp.append(((s >= t) & (y == 0)).sum() / N)
    tp, fp = np.array(tp), np.array(fp)
    return float(np.sum((fp[1:] - fp[:-1]) * (tp[1:] + tp[:-1]) / 2))


def boot(spk, fns):
    spk = np.asarray(spk).astype(str)
    uniq = np.unique(spk)
    groups = [np.flatnonzero(spk == u) for u in uniq]
    rng = np.random.default_rng(0)
    st = {k: [] for k in fns}; bad = 0
    for _ in range(D):
        d = rng.integers(0, len(uniq), len(uniq))
        r = np.concatenate([groups[i] for i in d])
        v = {k: f(r) for k, f in fns.items()}
        if any(np.isnan(z) for z in v.values()):
            bad += 1; continue
        for k, z in v.items():
            st[k].append(z)
    return {k: np.array(v) for k, v in st.items()}, len(uniq), bad


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def rd(path, key="pid"):
    return {r[key]: r for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--mine_lift", required=True)
    ap.add_argument("--mine_default", default=None)
    ap.add_argument("--pod", default=None)
    ap.add_argument("--col", default="p_yes_paper")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    L = rd(a.labels); W = rd(a.windows); ML = rd(a.mine_lift)
    MD = rd(a.mine_default) if a.mine_default else {}
    AG = rd(a.pod) if a.pod else {}
    pids = sorted(L, key=int)
    rep = dict(n_labels=len(pids), n_mine_lift=len(ML), n_mine_default=len(MD), n_pod=len(AG), col=a.col)
    miss = [p for p in pids if p not in ML or ML[p].get(a.col, "") == ""]
    rep["mine_lift_missing_or_error"] = miss
    if miss:
        raise SystemExit(f"missing or errored lift rows: {miss}")
    y = np.array([int(L[p]["label"]) for p in pids])
    s_tr = np.array([float(L[p]["p_yes_answer"]) for p in pids])
    s_li = np.array([float(ML[p][a.col]) if p not in miss else np.nan for p in pids])
    dur = np.array([float(W[p]["win_dur_s"]) for p in pids])
    cap = np.array([int(W[p]["capped"]) for p in pids])
    ntok = np.array([int(ML[p]["n_audio_tok"]) if p not in miss else -1 for p in pids])
    rep["n_pos"] = int(y.sum()); rep["n_neg"] = int((1 - y).sum())
    rep["n_gt300"] = int((dur > 300).sum()); rep["n_capped"] = int(cap.sum()); rep["n_le300"] = int((dur <= 300).sum())
    def _exp(p):
        n = int(round(float(ML[p]["dur_in_s"]) * 16000))
        F = -(-n // 160) if n < 300 * 16000 else n // 160
        return ((F - 1) // 2 + 1 - 2) // 2 + 1
    exp_tok = np.array([_exp(p) if p not in miss else -1 for p in pids])
    rep["lift_tokens_equal_formula_n"] = int((ntok == exp_tok).sum())
    rep["lift_tokens_capped_all_22500"] = bool(np.all(ntok[cap == 1] == 22500))
    rep["lift_tokens_max"] = int(ntok.max()); rep["lift_tokens_gt7500_n"] = int((ntok > 7500).sum())
    rep["lift_seq_max"] = int(max(int(ML[p]["seq_len"]) for p in pids if p not in miss))

    short = dur <= 300
    rep["short_maxabs_lift_minus_trunc"] = float(np.nanmax(np.abs(s_li[short] - s_tr[short])))
    rep["short_identical_to_1e-12_n"] = int((np.abs(s_li[short] - s_tr[short]) < 1e-12).sum())

    if MD:
        s_md = np.array([float(MD[p][a.col]) for p in pids])
        rep["mine_default_vs_part16_maxabs"] = float(np.max(np.abs(s_md - s_tr)))
        rep["mine_default_auc"] = auc_pairs(y, s_md)
    if AG:
        s_ag_l = np.array([float(AG[p]["lift_p_yes"]) for p in pids if p in AG])
        s_mi_l = np.array([s_li[i] for i, p in enumerate(pids) if p in AG])
        rep["pod_lift_vs_mine_maxabs"] = float(np.nanmax(np.abs(s_ag_l - s_mi_l)))
        rep["pod_lift_vs_mine_n"] = int(len(s_ag_l))
        s_ag_d = np.array([float(AG[p]["def_p_yes"]) for p in pids if p in AG])
        s_tr_ag = np.array([s_tr[i] for i, p in enumerate(pids) if p in AG])
        rep["pod_default_vs_part16_maxabs"] = float(np.max(np.abs(s_ag_d - s_tr_ag)))
        tok_ag = np.array([int(AG[p]["lift_n_audio_tok"]) for p in pids if p in AG])
        tok_mi = np.array([ntok[i] for i, p in enumerate(pids) if p in AG])
        rep["pod_lift_tokens_equal_mine_n"] = int((tok_ag == tok_mi).sum())

    cells = {}
    for name, m in [("all275", np.ones(len(pids), bool)), ("gt300_217", dur > 300), ("capped900_89", cap == 1),
                    ("le300_58", dur <= 300)]:
        yy, t, l = y[m], s_tr[m], s_li[m]
        spk = np.array(pids)[m]
        c = dict(n=int(m.sum()), n_pos=int(yy.sum()), n_spk=int(len(np.unique(spk))))
        c["trunc_auc_pairs"] = auc_pairs(yy, t); c["trunc_auc_trapz"] = auc_trapz(yy, t)
        c["lift_auc_pairs"] = auc_pairs(yy, l); c["lift_auc_trapz"] = auc_trapz(yy, l)
        c["diff_lift_minus_trunc"] = c["lift_auc_pairs"] - c["trunc_auc_pairs"]
        st, K, bad = boot(spk, {"t": lambda r, yy=yy, t=t: auc_pairs(yy[r], t[r]),
                                "l": lambda r, yy=yy, l=l: auc_pairs(yy[r], l[r])})
        c["trunc_ci"] = ci(st["t"]); c["lift_ci"] = ci(st["l"]); c["diff_ci"] = ci(st["l"] - st["t"])
        c["boot_K"] = K; c["boot_dropped"] = bad; c["boot_kept"] = int(len(st["t"]))
        # separate single-cell bootstraps (fresh rng per cell) for the unpaired intervals
        st1, _, b1 = boot(spk, {"x": lambda r, yy=yy, t=t: auc_pairs(yy[r], t[r])})
        st2, _, b2 = boot(spk, {"x": lambda r, yy=yy, l=l: auc_pairs(yy[r], l[r])})
        c["trunc_ci_single"] = ci(st1["x"]); c["lift_ci_single"] = ci(st2["x"])
        c["pairs_equal_trapz_1e-9"] = bool(abs(c["trunc_auc_pairs"] - c["trunc_auc_trapz"]) < 1e-9 and
                                          abs(c["lift_auc_pairs"] - c["lift_auc_trapz"]) < 1e-9)
        cells[name] = c
    out = dict(report=rep, cells=cells)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
