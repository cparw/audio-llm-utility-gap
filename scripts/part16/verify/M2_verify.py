"""
INDEPENDENT VERIFIER for PART16 / M2.
Written from scratch. Does not import, exec or copy the builder's script.
Goes back to the ORIGINAL sources:
  <local data dir>/Desktop/release/edaic_rerun/part14_segments_sentiment.csv   (RoBERTa scores)
  manifests/part14_manifest_new.csv          (Part 14 selection on disk)
and RE-RUNS distilbert-base-uncased-finetuned-sst-2-english itself rather than
trusting the builder's saved distilbert columns.
"""
import os, sys, json, hashlib
import numpy as np, pandas as pd

R   = "<local data dir>/Desktop/release/edaic_rerun"
V   = f"{R}/part16/verify"
SEG = f"{R}/part14_segments_sentiment.csv"
MAN = f"{R}/part14_manifest_new.csv"
MYSCORES = f"{V}/M2_verify_my_distilbert_scores.csv"
THR = 0.70

out = {}
def say(*a):
    print(*a); sys.stdout.flush()

# ----------------------------------------------------------------------------
# 0. my own distilbert run
# ----------------------------------------------------------------------------
def run_distilbert(texts):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    mid = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForSequenceClassification.from_pretrained(mid)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    mdl = mdl.to(dev).eval()
    say("  id2label:", mdl.config.id2label)
    pos_ix = [i for i, l in mdl.config.id2label.items() if str(l).upper().startswith("POS")][0]
    P = np.zeros(len(texts), dtype=np.float64)
    B = 32
    with torch.no_grad():
        for s in range(0, len(texts), B):
            batch = [str(t) for t in texts[s:s+B]]
            enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            logits = mdl(**enc).logits.float()
            pr = torch.softmax(logits, dim=-1)[:, pos_ix].cpu().numpy()
            P[s:s+B] = pr
            if (s // B) % 25 == 0:
                say(f"    {s}/{len(texts)}")
    return P

S = pd.read_csv(SEG)
S["span"] = S.end - S.start
say(f"segments csv rows={len(S)} speakers={S.pid.nunique()}")

if os.path.exists(MYSCORES):
    say("reusing my own cached distilbert scores")
    ms = pd.read_csv(MYSCORES)
    assert len(ms) == len(S)
    S["my_p_pos"] = ms["my_p_pos"].values
else:
    say("running distilbert sst-2 myself over %d texts ..." % len(S))
    S["my_p_pos"] = run_distilbert(S.text.tolist())
    pd.DataFrame({"row": np.arange(len(S)), "pid": S.pid.values,
                  "my_p_pos": S.my_p_pos.values}).to_csv(MYSCORES, index=False)

# ----------------------------------------------------------------------------
# 1. Part 14 rule, re-implemented from the builder's stated logic
# ----------------------------------------------------------------------------
E = S[(S.speech_s >= 5) & (S.nw >= 25) & (S.span >= 8)].copy()
out["n_eligible"] = int(len(E))
out["n_eligible_speakers"] = int(E.pid.nunique())
say(f"eligible rows={len(E)} speakers={E.pid.nunique()}")

def build(E, POS, NEG, tag):
    """returns (rule_arm series over E.index, selected_arm series, stats dict, manifest df)"""
    CONF = ((E.y == 1) & (E.sev >= 15) & POS) | ((E.y == 0) & (E.sev <= 4) & NEG)
    POOL = ((E.y == 1) & NEG) | ((E.y == 0) & POS)
    rule = pd.Series("none", index=E.index)
    rule[POOL.values] = "agreement"
    rule[CONF.values] = "conflict"          # conflict wins if somehow both (cannot happen)
    conf = E[CONF.values]
    pool = E[POOL.values]
    pairs = []
    unmatched = 0
    # group pool by speaker once; keep original row order inside the group
    pool_by_pid = {p: g for p, g in pool.groupby("pid", sort=False)}
    for i, r in conf.iterrows():
        g = pool_by_pid.get(r.pid)
        if g is None:
            unmatched += 1; continue
        g = g[g.index != i]
        if len(g) == 0:
            unmatched += 1; continue
        d = (g.span - r.span).abs().values
        j = g.index[int(np.argmin(d))]      # first minimum in row order == idxmin
        pairs.append((i, j))
    cidx = [a for a, b in pairs]; aidx = [b for a, b in pairs]
    man = pd.concat([
        E.loc[cidx].assign(set="conflict",  pair_id=range(len(cidx)),
                           seg_uid=["c%06d" % k for k in cidx]),
        E.loc[aidx].assign(set="agreement", pair_id=range(len(aidx)),
                           seg_uid=["a%06d" % k for k in aidx]),
    ])
    man["speaker_id"] = man.pid; man["label"] = man.y
    sel = pd.Series("none", index=E.index)
    sel[aidx] = "agreement"
    sel[cidx] = "conflict"
    st = dict(tag=tag, n_conflict_rule=int(CONF.sum()), n_pool_rule=int(POOL.sum()),
              n_pairs=len(pairs), unmatched=int(unmatched), n_rows=len(man),
              n_keys=int(man.groupby(["set", "seg_uid"]).ngroups),
              n_speakers=int(man.speaker_id.nunique()),
              depressed_positive=int((E.loc[cidx].y == 1).sum()),
              control_negative=int((E.loc[cidx].y == 0).sum()),
              uniq_conflict_seg=len(set(cidx)), uniq_agreement_seg=len(set(aidx)))
    return rule, sel, st, man

rb_rule, rb_sel, rb_st, rb_man = build(E, (E.clearly_positive == 1).values,
                                          (E.clearly_negative == 1).values, "roberta")
say("roberta arms: " + json.dumps(rb_st))
out["roberta_arms"] = rb_st

# does my roberta rebuild reproduce part14_manifest_new.csv exactly?
disk = pd.read_csv(MAN)
mine_keys = sorted(zip(rb_man["set"], rb_man.seg_uid))
disk_keys = sorted(zip(disk["set"], disk.seg_uid))
out["manifest_rows_disk"] = int(len(disk))
out["manifest_pairs_disk"] = int(disk.pair_id.nunique())
out["manifest_speakers_disk"] = int(disk.speaker_id.nunique())
out["manifest_keys_disk"] = int(disk.groupby(["set", "seg_uid"]).ngroups)
out["my_roberta_rebuild_matches_disk_manifest"] = bool(mine_keys == disk_keys and len(rb_man) == len(disk))
say("manifest on disk rows=%d pairs=%d speakers=%d keys=%d ; my rebuild identical=%s"
    % (len(disk), disk.pair_id.nunique(), disk.speaker_id.nunique(),
       disk.groupby(["set", "seg_uid"]).ngroups, out["my_roberta_rebuild_matches_disk_manifest"]))

def db_arms(E, thr):
    POS = (E.my_p_pos >= thr).values
    NEG = (E.my_p_pos <= 1.0 - thr).values
    return build(E, POS, NEG, "distilbert@%g" % thr)

db_rule, db_sel, db_st, db_man = db_arms(E, THR)
say("distilbert arms @0.70: " + json.dumps(db_st))
out["distilbert_arms"] = db_st

# ----------------------------------------------------------------------------
# 2. agreement / kappa with speaker bootstrap
# ----------------------------------------------------------------------------
def po_kappa(a, b, cats):
    """raw agreement and Cohen kappa, computed from the confusion matrix by hand"""
    n = len(a)
    if n == 0: return np.nan, np.nan, np.nan
    ia = np.searchsorted(cats, a); ib = np.searchsorted(cats, b)
    k = len(cats)
    M = np.zeros((k, k), dtype=np.float64)
    np.add.at(M, (ia, ib), 1.0)
    po = np.trace(M) / n
    pe = float((M.sum(1) / n) @ (M.sum(0) / n))
    kap = np.nan if abs(1.0 - pe) < 1e-15 else (po - pe) / (1.0 - pe)
    return po, pe, kap

def cell(name, pid, a, b, cats, draws=2000, seed=0):
    pid = np.asarray(pid); a = np.asarray(a); b = np.asarray(b)
    cats = np.array(sorted(cats))
    po, pe, kap = po_kappa(a, b, cats)
    # bootstrap: resample speakers with replacement, one draw feeds both quantities
    spk = np.unique(pid)
    idx_by_spk = [np.where(pid == s)[0] for s in spk]
    rng = np.random.default_rng(seed)
    bp, bk = [], []
    for _ in range(draws):
        pick = rng.integers(0, len(spk), len(spk))
        sel = np.concatenate([idx_by_spk[p] for p in pick])
        p_, e_, k_ = po_kappa(a[sel], b[sel], cats)
        if np.isfinite(p_): bp.append(p_)
        if np.isfinite(k_): bk.append(k_)
    bp = np.array(bp); bk = np.array(bk)
    r = dict(name=name, n=int(len(a)), n_speakers=int(len(spk)), po=float(po), pe=float(pe),
             kappa=float(kap),
             po_lo=float(np.percentile(bp, 2.5)), po_hi=float(np.percentile(bp, 97.5)),
             kappa_lo=float(np.percentile(bk, 2.5)), kappa_hi=float(np.percentile(bk, 97.5)),
             usable_po=int(len(bp)), usable_kappa=int(len(bk)))
    say("  %-26s po=%.4f [%.4f, %.4f]  kappa=%.4f [%.4f, %.4f]  pe=%.4f  n=%d spk=%d  usable %d/%d"
        % (name, r["po"], r["po_lo"], r["po_hi"], r["kappa"], r["kappa_lo"], r["kappa_hi"],
           r["pe"], r["n"], r["n_speakers"], r["usable_kappa"], draws))
    # confusion matrix for the record
    ia = np.searchsorted(cats, a); ib = np.searchsorted(cats, b)
    M = np.zeros((len(cats), len(cats)), dtype=int); np.add.at(M, (ia, ib), 1)
    r["cats"] = [str(c) for c in cats]; r["confusion"] = M.tolist()
    return r

def three_way(p_pos, p_neg_like, thr, kind):
    """label: negative / neutral / positive"""
    if kind == "roberta":
        lab = np.where(p_pos >= thr, "positive",
              np.where(p_neg_like >= thr, "negative", "neutral"))
    else:
        lab = np.where(p_pos >= thr, "positive",
              np.where(p_pos <= 1 - thr, "negative", "neutral"))
    return lab

say("\n--- sentiment label, all segments")
lab_rb_all = three_way(S.p_pos.values, S.p_neg.values, THR, "roberta")
lab_db_all = three_way(S.my_p_pos.values, None, THR, "distilbert")
res = {}
res["label_all"] = cell("label_all", S.pid.values, lab_rb_all, lab_db_all,
                        ["negative", "neutral", "positive"])
say("--- sentiment label, eligible segments")
res["label_elig"] = cell("label_elig", E.pid.values,
                         three_way(E.p_pos.values, E.p_neg.values, THR, "roberta"),
                         three_way(E.my_p_pos.values, None, THR, "distilbert"),
                         ["negative", "neutral", "positive"])
say("--- rule arm, eligible")
res["arm_rule"] = cell("arm_rule", E.pid.values, rb_rule.values, db_rule.values,
                       ["agreement", "conflict", "none"])
say("--- selected arm, eligible")
res["arm_selected"] = cell("arm_selected", E.pid.values, rb_sel.values, db_sel.values,
                           ["agreement", "conflict", "none"])

# ----------------------------------------------------------------------------
# 3. count-matched threshold
# ----------------------------------------------------------------------------
say("\n--- threshold sweep (target: roberta conflict_rule=%d)" % rb_st["n_conflict_rule"])
sweep = []
for thr in [0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]:
    _, _, st, _ = db_arms(E, thr)
    sweep.append(dict(thr=thr, **{k: st[k] for k in
                 ["n_conflict_rule", "n_pool_rule", "n_pairs", "unmatched", "n_speakers"]}))
    say("   thr=%-7g conflict=%-5d pairs=%-5d spk=%d" % (thr, st["n_conflict_rule"], st["n_pairs"], st["n_speakers"]))
out["sweep"] = sweep
best = min(sweep, key=lambda d: abs(d["n_conflict_rule"] - rb_st["n_conflict_rule"]))
out["my_closest_threshold"] = best["thr"]
say("   my closest threshold = %g (conflict=%d vs roberta %d)" % (best["thr"], best["n_conflict_rule"], rb_st["n_conflict_rule"]))

MT = 0.995   # the threshold THEY reported; recompute at it regardless of my own pick
m_rule, m_sel, m_st, _ = db_arms(E, MT)
say("--- rule arm @%.3f" % MT)
res["arm_rule_matched"] = cell("arm_rule_matched", E.pid.values, rb_rule.values, m_rule.values,
                               ["agreement", "conflict", "none"])
say("--- selected arm @%.3f" % MT)
res["arm_selected_matched"] = cell("arm_selected_matched", E.pid.values, rb_sel.values, m_sel.values,
                                   ["agreement", "conflict", "none"])
out["matched_arms_stats"] = m_st

# ----------------------------------------------------------------------------
# 4. overlap of the two selections + survival of the 483 conflict segments
# ----------------------------------------------------------------------------
rb_keys = set(zip(rb_man["set"], rb_man.seg_uid)); db_keys = set(zip(db_man["set"], db_man.seg_uid))
rb_seg = set(rb_man.seg_uid.str[1:]); db_seg = set(db_man.seg_uid.str[1:])
out["overlap_same_arm_seg_uid"] = len(rb_keys & db_keys)
out["overlap_seg_any_arm"] = len(rb_seg & db_seg)
conf_idx = sorted(set(rb_man[rb_man["set"] == "conflict"].index))
surv = pd.Series(db_sel.loc[conf_idx].values).value_counts().to_dict()
out["roberta_conflict_483_under_distilbert_selected"] = {k: int(v) for k, v in surv.items()}
out["conflict_survival_frac"] = float(surv.get("conflict", 0) / len(conf_idx))
say("\nroberta's %d conflict segments under distilbert SELECTED arm: %s  (conflict share %.4f)"
    % (len(conf_idx), surv, out["conflict_survival_frac"]))
conf_rule_surv = pd.Series(db_rule.loc[conf_idx].values).value_counts().to_dict()
out["roberta_conflict_483_under_distilbert_rule"] = {k: int(v) for k, v in conf_rule_surv.items()}

# ----------------------------------------------------------------------------
# 5. compare my own distilbert scores with the builder's saved ones
# ----------------------------------------------------------------------------
BSENT = f"{R}/part16/M2_distilbert_sentiment.csv"
if os.path.exists(BSENT):
    B = pd.read_csv(BSENT)
    out["builder_sentiment_csv_rows"] = int(len(B))
    if "id" in B.columns and len(B) == len(E):
        theirs = B["distilbert_score"].values
        mine = E.my_p_pos.values
        out["score_max_abs_diff"] = float(np.max(np.abs(theirs - mine)))
        out["score_corr"] = float(np.corrcoef(theirs, mine)[0, 1])
        out["score_n_flips_at_0.70"] = int(((theirs >= .70) != (mine >= .70)).sum())
        out["score_n_flips_at_0.30"] = int(((theirs <= .30) != (mine <= .30)).sum())
        say("\nmy distilbert vs builder's saved: max|d|=%.2e corr=%.8f flips@0.70=%d flips@0.30=%d"
            % (out["score_max_abs_diff"], out["score_corr"],
               out["score_n_flips_at_0.70"], out["score_n_flips_at_0.30"]))
BMAN = f"{R}/part16/M2_distilbert_manifest.csv"
if os.path.exists(BMAN):
    BM = pd.read_csv(BMAN)
    out["builder_manifest_csv_rows"] = int(len(BM))
    out["builder_manifest_pairs"] = int(BM.pair_id.nunique())
    out["builder_manifest_speakers"] = int(BM.speaker_id.nunique())
    my_bkeys = sorted(zip(db_man["set"], db_man.seg_uid)); th = sorted(zip(BM["set"], BM.seg_uid))
    out["my_distilbert_manifest_matches_builder"] = bool(my_bkeys == th)
    say("builder distilbert manifest rows=%d pairs=%d spk=%d ; identical to mine=%s"
        % (len(BM), BM.pair_id.nunique(), BM.speaker_id.nunique(),
           out["my_distilbert_manifest_matches_builder"]))

out["cells"] = res
json.dump(out, open(f"{V}/M2_verify_out.json", "w"), indent=1, default=str)
say("\nwrote %s/M2_verify_out.json" % V)
