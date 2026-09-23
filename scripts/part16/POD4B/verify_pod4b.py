#!/usr/bin/env python3
"""Independent Mac-side verification: recompute every AUC with the rank formula
from the synced per-clip files, and re-run the speaker bootstrap. Nothing is
copied out of a json."""
import csv, json, sys
import numpy as np
NB = 2000
def auc_rank(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if len(np.unique(y)) < 2: return np.nan
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    rr = np.arange(1, len(s)+1, dtype=float); i = 0
    while i < len(ss):
        j = i
        while j+1 < len(ss) and ss[j+1] == ss[i]: j += 1
        rr[i:j+1] = rr[i:j+1].mean(); i = j+1
    r = np.empty(len(s)); r[o] = rr
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return (r[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)
def boot_ci(spk, y, s, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(spk)
    ib = {p: np.where(spk == p)[0] for p in u}; v = []
    for _ in range(NB):
        ii = np.concatenate([ib[p] for p in rng.choice(u, size=len(u), replace=True)])
        a = auc_rank(y[ii], s[ii])
        if not np.isnan(a): v.append(a)
    v = np.array(v); return float(np.percentile(v,2.5)), float(np.percentile(v,97.5)), len(v)
def boot_paired(spk, y, s1, s2, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(spk)
    ib = {p: np.where(spk == p)[0] for p in u}; d = []
    for _ in range(NB):
        ii = np.concatenate([ib[p] for p in rng.choice(u, size=len(u), replace=True)])
        a1, a2 = auc_rank(y[ii], s1[ii]), auc_rank(y[ii], s2[ii])
        if np.isnan(a1) or np.isnan(a2): continue
        d.append(a2-a1)
    d = np.array(d); return float(np.percentile(d,2.5)), float(np.percentile(d,97.5)), len(d)

rows_out = []
for CORPUS, N in (("adress2020", 156), ("adresso", 161)):
    pc = list(csv.DictReader(open(f"out/{CORPUS}_no_interviewer.csv")))
    assert len(pc) == N, (CORPUS, len(pc))
    spk = np.array([r["speaker"] for r in pc]); y = np.array([int(r["label"]) for r in pc])
    S = {}
    for arm in ("A", "B", "C"):
        S[f"{arm}_zs"]  = np.array([float(r[f"p_yes_{arm}"]) for r in pc])
        S[f"{arm}_enc"] = np.array([float(r[f"p_probe_{arm}"]) for r in pc])
    S["pub_zs"]  = np.array([float(r["p_yes_published"]) for r in pc])
    S["pub_enc"] = np.array([float(r["p_probe_published"]) for r in pc])
    nspk = len(np.unique(spk))
    lbl = {"A":"A_orig","B":"B_paronly","C":"C_durmatch","pub":"published"}
    print(f"\n########## {CORPUS}  n={len(pc)} speakers={nspk}  AD={int(y.sum())} control={int((y==0).sum())}")
    for arm in ("A","B","C","pub"):
        for st, nm in (("zs","zero-shot"), ("enc","encoder nested probe")):
            k = f"{arm}_{st}"; a = auc_rank(y, S[k]); lo, hi, nu = boot_ci(spk, y, S[k])
            print(f"  {lbl[arm]:12s} {nm:22s} AUC {a:.4f}  [{lo:.4f}, {hi:.4f}]  draws={nu}")
            rows_out.append((f"4b_{CORPUS}_{lbl[arm]}_{st}",
                             f"{CORPUS} arm {lbl[arm]} Omni {nm} AUC",
                             a, lo, hi, len(pc), nspk, f"out/{CORPUS}_no_interviewer.csv", nu))
    print("  --- paired arm contrasts (same speaker draw, both quantities) ---")
    for st, nm in (("zs","zero-shot"), ("enc","encoder probe")):
        for a1, a2, desc in (("A","B","original cut -> interviewer removed"),
                             ("A","C","original cut -> duration-matched, interviewer KEPT"),
                             ("C","B","duration-matched WITH interviewer -> interviewer removed")):
            k1, k2 = f"{a1}_{st}", f"{a2}_{st}"
            d = auc_rank(y, S[k2]) - auc_rank(y, S[k1])
            lo, hi, nu = boot_paired(spk, y, S[k1], S[k2])
            ex = "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
            print(f"  PAIRED {nm:14s} {a1}->{a2}  d {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {ex}   ({desc})")
            rows_out.append((f"4b_{CORPUS}_paired_{st}_{a1}_to_{a2}",
                             f"{CORPUS} PAIRED {nm}: {desc}", d, lo, hi, len(pc), nspk,
                             f"out/{CORPUS}_no_interviewer.csv", nu))
    print("  --- the gap: encoder probe minus zero-shot, under each arm ---")
    for arm in ("A","B","C","pub"):
        d = auc_rank(y, S[f"{arm}_enc"]) - auc_rank(y, S[f"{arm}_zs"])
        lo, hi, nu = boot_paired(spk, y, S[f"{arm}_zs"], S[f"{arm}_enc"])
        ex = "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
        print(f"  GAP {lbl[arm]:12s} d {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {ex}")
        rows_out.append((f"4b_{CORPUS}_gap_{lbl[arm]}",
                         f"{CORPUS} arm {lbl[arm]} GAP = encoder probe AUC minus zero-shot AUC",
                         d, lo, hi, len(pc), nspk, f"out/{CORPUS}_no_interviewer.csv", nu))
    # interviewer presence
    inv = np.array([float(r["inv_ms_in_window"]) for r in pc])
    shr = np.array([float(r["inv_share_of_window"]) for r in pc])
    par = np.array([float(r["par_ms_in_window"]) for r in pc])
    dA  = np.array([float(r["dur_A_orig_s"]) for r in pc])
    dB  = np.array([float(r["dur_B_paronly_s"]) for r in pc])
    print(f"  --- interviewer actually present in the original window ---")
    print(f"  mean INV share {shr.mean():.4f}  median {np.median(shr):.4f}  max {shr.max():.4f}")
    print(f"  windows containing ANY interviewer speech: {(inv>0).sum()}/{len(pc)}   >5 s: {(inv>5000).sum()}/{len(pc)}")
    print(f"  arm A dur mean {dA.mean():.2f}s  arm B dur mean {dB.mean():.2f}s  removed mean {(dA-dB).mean():.2f}s")
    print(f"  windows under 10 s of participant speech: {(dB<10).sum()}/{len(pc)}")
    rows_out += [
      (f"4b_{CORPUS}_inv_share_mean", f"{CORPUS} mean interviewer share of the original window",
       float(shr.mean()), "", "", len(pc), nspk, f"out/{CORPUS}_no_interviewer.csv", ""),
      (f"4b_{CORPUS}_inv_share_median", f"{CORPUS} median interviewer share of the original window",
       float(np.median(shr)), "", "", len(pc), nspk, f"out/{CORPUS}_no_interviewer.csv", ""),
      (f"4b_{CORPUS}_inv_any_count", f"{CORPUS} windows containing any interviewer speech",
       int((inv>0).sum()), "", "", len(pc), nspk, f"out/{CORPUS}_no_interviewer.csv", ""),
      (f"4b_{CORPUS}_inv_gt5s_count", f"{CORPUS} windows containing more than 5 s of interviewer speech",
       int((inv>5000).sum()), "", "", len(pc), nspk, f"out/{CORPUS}_no_interviewer.csv", ""),
      (f"4b_{CORPUS}_dur_removed_mean_s", f"{CORPUS} mean seconds removed going from arm A to arm B",
       round(float((dA-dB).mean()),4), "", "", len(pc), nspk, f"out/{CORPUS}_no_interviewer.csv", ""),
      (f"4b_{CORPUS}_par_under10s_count", f"{CORPUS} windows with under 10 s of participant speech",
       int((dB<10).sum()), "", "", len(pc), nspk, f"out/{CORPUS}_no_interviewer.csv", ""),
    ]

with open("reports/part16/rows/POD4B.tsv", "w") as f:
    f.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n")
    for r in rows_out:
        v = f"{r[2]:.4f}" if isinstance(r[2], float) else str(r[2])
        lo = f"{r[3]:.4f}" if isinstance(r[3], float) else ""
        hi = f"{r[4]:.4f}" if isinstance(r[4], float) else ""
        f.write(f"{r[0]}\t{r[1]}\t{v}\t{lo}\t{hi}\t{r[5]}\t{r[6]}\t{r[7]}\n")
print(f"\nwrote rows/POD4B.tsv : {len(rows_out)} values")
