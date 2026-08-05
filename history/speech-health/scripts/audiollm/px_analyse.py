"""Multi-model generalisation of the DAIC words-vs-voice paradox.

For every model x prompt:
  - conflict / agreement / pooled block metrics (acc, bal-acc, AUC vs clinical label,
    AUC vs lexical direction)
  - paired conflict-minus-agreement difference with speaker-clustered bootstrap + McNemar
  - speaker-level label-shuffle control on the conflict set
  - threshold-free headline: pooled AUC(lexical) - AUC(label), speaker-clustered paired boot

Design note that matters for reading the tables:
  Inside the conflict subset the lexical direction is by construction the exact
  complement of the clinical label, so AUC_lex == 1 - AUC_label there; inside the
  agreement subset they are identical, so AUC_lex == AUC_label. Those two columns
  carry no extra information within a subset. On the POOLED 390-clip set the two
  targets are exactly orthogonal (label=1: 138 words-yes / 138 words-no;
  label=0: 57 / 57), so pooled AUC_label and pooled AUC_lex ARE independent
  quantities. The pooled pair is the real test.
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
warnings.filterwarnings("ignore")

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
P = f"{R}/committed/paradox"
OUT = f"{R}/final"
os.makedirs(OUT, exist_ok=True)
RNG_SEED = 0
NBOOT = 4000
NPERM = 2000
GRID = np.linspace(0.01, 0.99, 197)

MODELS = [("qwen2audio", "qwen2-audio"), ("omni", "qwen2.5-omni"),
          ("af3", "audio-flamingo-3"), ("kimi", "kimi-audio")]
PROMPTS = [("main", "p_main"), ("voice", "p_voice")]

M = pd.read_csv(f"{P}/manifest.csv")           # 390 pair rows
assert len(M) == 390 and M.pair_id.nunique() == 195


def bal_acc(y, p, t):
    pr = (p >= t).astype(int)
    if len(set(y)) < 2:
        return float("nan")
    return 0.5 * ((pr[y == 1] == 1).mean() + (pr[y == 0] == 0).mean())


def auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def pick_thr(y, p):
    return float(GRID[int(np.argmax([bal_acc(y, p, t) for t in GRID]))])


def block(d, pcol, t):
    y, p = d.label.values, d[pcol].values
    pr = (p >= t).astype(int)
    return dict(n=len(d), spk=int(d.speaker_id.nunique()),
                mean_p=float(p.mean()), sd_p=float(p.std()),
                acc=float((pr == y).mean()), bal_acc=bal_acc(y, p, t),
                auc_label=auc(y, p), auc_lex=auc(d.words_say_depressed.values, p),
                pred_dep_rate=float(pr.mean()))


def analyse(short, nice, pcol, pname):
    f = f"{OUT}/px_scores_{short}.csv"
    if not os.path.exists(f):
        print("MISSING", f, flush=True)
        return None
    S = pd.read_csv(f)
    seg = S[S.task_type == "paradox_segment"][["filepath", pcol]]
    iv = S[S.task_type == "interview"].copy()

    D = M.merge(seg, on="filepath", how="inner", validate="many_to_one")
    if len(D) != 390:
        print("WARN %s: joined %d/390 rows" % (short, len(D)), flush=True)

    # --- threshold from the model's own whole interviews, same prompt, same 30 s window
    t_all = pick_thr(iv.label.values, iv[pcol].values)
    px_spk = set(D.speaker_id.astype(str))
    ivh = iv[~iv.speaker_id.astype(str).isin(px_spk)]        # 201 held-out speakers
    t_ho = pick_thr(ivh.label.values, ivh[pcol].values)
    iv_auc = auc(iv.label.values, iv[pcol].values)

    res = dict(model=nice, short=short, prompt=pname,
               interview30_auc=iv_auc, interview30_n=int(len(iv)),
               thr_session=t_all, thr_heldout_speakers=t_ho,
               thr_heldout_n_spk=int(ivh.speaker_id.nunique()))

    for tname, t in [("thr_session", t_all), ("thr0.50", 0.5), ("thr_heldout", t_ho)]:
        for s in ["conflict", "agreement"]:
            res[f"{s}|{tname}"] = block(D[D.set == s], pcol, t)
        res[f"pooled|{tname}"] = block(D, pcol, t)

    t = t_all
    C = D[D.set == "conflict"].copy()
    C["pred"] = (C[pcol] >= t).astype(int)
    res["follow_label"] = float((C.pred == C.label).mean())
    res["follow_words"] = float((C.pred == C.words_say_depressed).mean())

    # --- paired conflict vs agreement
    pv = D.pivot_table(index="pair_id", columns="set", values=pcol)
    meta = D[D.set == "conflict"].set_index("pair_id")[["label", "speaker_id"]]
    pv = pv.join(meta).dropna()
    cc = ((pv.conflict >= t).astype(int) == pv.label).values
    ca = ((pv.agreement >= t).astype(int) == pv.label).values
    b01 = int((~cc & ca).sum()); b10 = int((cc & ~ca).sum())
    res["n_pairs"] = int(len(pv))
    res["paired_conflict_acc"] = float(cc.mean())
    res["paired_agreement_acc"] = float(ca.mean())
    res["paired_diff"] = float(cc.mean() - ca.mean())
    res["mcnemar_b_agree_only"] = b01
    res["mcnemar_b_conflict_only"] = b10
    res["mcnemar_p"] = float(binomtest(b10, b10 + b01, 0.5).pvalue) if (b10 + b01) else float("nan")

    rng = np.random.default_rng(RNG_SEED)
    spks = pv.speaker_id.unique()
    idx_by_spk = {s: np.where(pv.speaker_id.values == s)[0] for s in spks}
    boot = []
    for _ in range(NBOOT):
        s = rng.choice(spks, len(spks), replace=True)
        idx = np.concatenate([idx_by_spk[x] for x in s])
        boot.append(cc[idx].mean() - ca[idx].mean())
    res["paired_diff_ci"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    # --- threshold-free headline: pooled AUC(lex) - AUC(label), paired speaker boot
    yl, yw, pp = D.label.values, D.words_say_depressed.values, D[pcol].values
    sp = D.speaker_id.values
    a_lab, a_lex = auc(yl, pp), auc(yw, pp)
    res["pooled_auc_label"] = a_lab
    res["pooled_auc_lex"] = a_lex
    res["pooled_auc_lex_minus_label"] = a_lex - a_lab
    uspk = np.unique(sp)
    ib = {s: np.where(sp == s)[0] for s in uspk}
    rng2 = np.random.default_rng(RNG_SEED + 1)
    db, lb, wb = [], [], []
    for _ in range(NBOOT):
        s = rng2.choice(uspk, len(uspk), replace=True)
        idx = np.concatenate([ib[x] for x in s])
        al, aw = auc(yl[idx], pp[idx]), auc(yw[idx], pp[idx])
        if np.isnan(al) or np.isnan(aw):
            continue
        lb.append(al); wb.append(aw); db.append(aw - al)
    res["pooled_auc_label_ci"] = [float(np.percentile(lb, 2.5)), float(np.percentile(lb, 97.5))]
    res["pooled_auc_lex_ci"] = [float(np.percentile(wb, 2.5)), float(np.percentile(wb, 97.5))]
    res["pooled_diff_ci"] = [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]
    res["pooled_diff_n_boot"] = len(db)

    # --- speaker-level label shuffle on the conflict set (accuracy at t)
    sl = C.groupby("speaker_id").label.first()
    rng3 = np.random.default_rng(RNG_SEED + 2)
    null = []
    for _ in range(NPERM):
        perm = pd.Series(rng3.permutation(sl.values), index=sl.index)
        yv = C.speaker_id.map(perm).values
        null.append(float((C.pred.values == yv).mean()))
    res["shuffle_null_mean"] = float(np.mean(null))
    res["shuffle_null_ci"] = [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))]
    res["shuffle_p_conflict_acc"] = float((np.array(null) >= res["follow_label"]).mean())

    # --- shuffle null for the pooled AUC gap (speaker-level label permutation)
    slp = D.groupby("speaker_id").label.first()
    rng4 = np.random.default_rng(RNG_SEED + 3)
    nl = []
    for _ in range(NPERM):
        perm = pd.Series(rng4.permutation(slp.values), index=slp.index)
        yv = D.speaker_id.map(perm).values
        nl.append(auc(yv, pp))
    res["shuffle_null_pooled_auc_label"] = float(np.nanmean(nl))
    return res, D


def fmt_ci(c):
    return "[%+.3f,%+.3f]" % (c[0], c[1])


rows_all, blocks_all, detail = [], [], {}
for short, nice in MODELS:
    for pname, pcol in PROMPTS:
        r = analyse(short, nice, pcol, pname)
        if r is None:
            continue
        res, D = r
        detail[f"{short}|{pname}"] = res
        for s in ["conflict", "agreement", "pooled"]:
            b = res[f"{s}|thr_session"]
            blocks_all.append(dict(model=nice, prompt=pname, block=s, **b))
        rows_all.append(res)

json.dump(rows_all, open(f"{OUT}/px_multimodel_summary.json", "w"), indent=2, default=float)
B = pd.DataFrame(blocks_all)
B.to_csv(f"{OUT}/px_multimodel_blocks.csv", index=False)

# ---------------------------------------------------------------- printing
print("\n" + "=" * 108)
print("MULTI-MODEL DAIC WORDS-vs-VOICE PARADOX  (195 matched pairs, 297 clips, 74 speakers, 30 s window)")
print("=" * 108)

for pname, _ in PROMPTS:
    sub = [r for r in rows_all if r["prompt"] == pname]
    if not sub:
        continue
    print("\n### PROMPT = %s ###" % pname)
    print("\n-- A. block metrics at each model's own session threshold --")
    print("%-16s %-10s %4s %6s %8s %8s %10s %8s %7s" %
          ("model", "block", "n", "meanp", "acc", "bal_acc", "AUC_label", "AUC_lex", "thr"))
    for r in sub:
        for s in ["conflict", "agreement", "pooled"]:
            b = r[f"{s}|thr_session"]
            print("%-16s %-10s %4d %6.3f %8.3f %8.3f %10.3f %8.3f %7.3f" %
                  (r["model"], s, b["n"], b["mean_p"], b["acc"], b["bal_acc"],
                   b["auc_label"], b["auc_lex"], r["thr_session"]))

    print("\n-- B. threshold-free headline: pooled AUC against label vs against lexical direction --")
    print("%-16s %18s %18s %20s %10s" %
          ("model", "AUC_label [95%CI]", "AUC_lex [95%CI]", "lex-label [95%CI]", "shuf_lab"))
    for r in sub:
        print("%-16s %6.3f %11s %6.3f %11s %7.3f %12s %10.3f" %
              (r["model"], r["pooled_auc_label"], fmt_ci(r["pooled_auc_label_ci"]),
               r["pooled_auc_lex"], fmt_ci(r["pooled_auc_lex_ci"]),
               r["pooled_auc_lex_minus_label"], fmt_ci(r["pooled_diff_ci"]),
               r["shuffle_null_pooled_auc_label"]))

    print("\n-- C. paired conflict-minus-agreement accuracy (195 pairs, same speaker, dur matched) --")
    print("%-16s %8s %8s %9s %18s %11s %9s %9s" %
          ("model", "confl", "agree", "diff", "diff 95%CI", "McNemar p", "follow_w", "shuf_null"))
    for r in sub:
        print("%-16s %8.3f %8.3f %+9.3f %18s %11.2e %9.3f %9.3f" %
              (r["model"], r["paired_conflict_acc"], r["paired_agreement_acc"],
               r["paired_diff"], fmt_ci(r["paired_diff_ci"]), r["mcnemar_p"],
               r["follow_words"], r["shuffle_null_mean"]))

    print("\n-- D. does the model have ANY depression signal here? --")
    print("%-16s %22s %22s %14s" %
          ("model", "whole interview@30s AUC", "agreement-seg AUC", "conflict-seg AUC"))
    for r in sub:
        print("%-16s %22.3f %22.3f %14.3f" %
              (r["model"], r["interview30_auc"],
               r["agreement|thr_session"]["auc_label"], r["conflict|thr_session"]["auc_label"]))

    print("\n-- E. threshold robustness (accuracy on conflict set) --")
    print("%-16s %14s %14s %14s" % ("model", "thr_session", "thr 0.50", "thr_heldout201"))
    for r in sub:
        print("%-16s %14.3f %14.3f %14.3f" %
              (r["model"], r["conflict|thr_session"]["acc"], r["conflict|thr0.50"]["acc"],
               r["conflict|thr_heldout"]["acc"]))

print("\nNOTE: within the conflict subset AUC_lex == 1 - AUC_label and within the agreement")
print("subset AUC_lex == AUC_label, both by construction. Only the POOLED row compares two")
print("genuinely independent targets (label and lexical direction are exactly orthogonal there).")
print("\nwrote %s/px_multimodel_summary.json and px_multimodel_blocks.csv" % OUT)
print("JOB_DONE", flush=True)
