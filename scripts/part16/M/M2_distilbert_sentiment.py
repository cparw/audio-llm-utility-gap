"""
PART 16 / M2 - second sentiment scorer on the Part 14 selection.

Swaps cardiffnlp/twitter-roberta-base-sentiment-latest for
distilbert-base-uncased-finetuned-sst-2-english on the SAME segment texts,
keeps every other Part 14 rule identical, rebuilds the conflict/agreement
arms, and reports arm overlap + Cohen's kappa against the RoBERTa arms.

Rule replicated verbatim from the Part 14 builder (part14_select.py):
  span         = end - start
  eligibility  : speech_s >= 5 & nw >= 25 & span >= 8
  POS          : clearly_positive == 1   (p_pos >= 0.70)
  NEG          : clearly_negative == 1   (p_neg >= 0.70)
  CONFLICT     : (y==1 & sev>=15 & POS) | (y==0 & sev<=4 & NEG)
  AGREE pool   : (y==1 & NEG) | (y==0 & POS)
  matching     : for each conflict segment, among the SAME speaker's pool
                 segments (excluding itself), take the one with the smallest
                 |span difference| (pandas idxmin -> first min in row order).
                 Pool members are NOT consumed; a segment may serve several
                 pairs. Conflict segments with no candidate are dropped.
  seg_uid      : 'c%06d' / 'a%06d' % row-index into the segment csv.
"""
import os, sys, json, time, hashlib, subprocess
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

R    = "<local data dir>/Desktop/release/edaic_rerun"
OUT  = R + "/part16"
SEG  = R + "/part14_segments_sentiment.csv"
MAN  = R + "/part14_manifest_new.csv"
SRC  = "<local data dir>/Desktop/af2_run/speech-health/results/health/paradox/daic_segments_scored.csv"
DISC = R + "/DISCREPANCIES.md"
MID  = "distilbert-base-uncased-finetuned-sst-2-english"
THR  = 0.70
DRAWS = 2000
CMD  = "/usr/local/bin/python3 " + OUT + "/scripts/M2_distilbert_sentiment.py"

def sha1mb(p):
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read(1024 * 1024)).hexdigest()
    except Exception as e:
        return "UNREADABLE:" + str(e)

def disc(sev, msg):
    with open(DISC, "a") as f:
        f.write(f"\n- [{sev}] PART16 M2 ({time.strftime('%Y-%m-%d %H:%M')}): {msg}\n")

def f4(x):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"

# ----------------------------------------------------------------- load
for p in (SEG, MAN, SRC):
    print(("FOUND  " if os.path.exists(p) else "MISSING") + "  " + p, flush=True)

S = pd.read_csv(SEG)
S["span"] = S.end - S.start
MANIFEST = pd.read_csv(MAN)
print(f"segments {len(S)}  speakers {S.pid.nunique()}", flush=True)
print(f"manifest rows {len(MANIFEST)}  pairs {MANIFEST.pair_id.nunique()}  "
      f"speakers {MANIFEST.speaker_id.nunique()}", flush=True)

# ------------------------------------------------- the Part 14 selection
def select(S, pos, neg, tag):
    """Part 14 rule with POS/NEG boolean masks supplied by the scorer."""
    E = S[(S.speech_s >= 5) & (S.nw >= 25) & (S.span >= 8)].copy()
    P = pos.reindex(E.index); N = neg.reindex(E.index)
    CONF = ((E.y == 1) & (E.sev >= 15) & P) | ((E.y == 0) & (E.sev <= 4) & N)
    AGRp = ((E.y == 1) & N) | ((E.y == 0) & P)
    conf = E[CONF]
    pool = E[AGRp]
    pairs = []
    for i, r in conf.iterrows():
        cand = pool[(pool.pid == r.pid) & (~pool.index.isin([i]))]
        if len(cand):
            pairs.append((i, (cand.span - r.span).abs().idxmin()))
        else:
            pairs.append((i, None))
    Pp = pd.DataFrame(pairs, columns=["conf_idx", "ctrl_idx"])
    unmatched = int(Pp.ctrl_idx.isna().sum())
    Pp = Pp[Pp.ctrl_idx.notna()].copy(); Pp["ctrl_idx"] = Pp.ctrl_idx.astype(int)
    cm = E.loc[Pp.conf_idx].copy(); am = E.loc[Pp.ctrl_idx].copy()
    keep = pd.concat([
        cm.assign(set="conflict",  pair_id=range(len(cm)), seg_uid=["c%06d" % k for k in cm.index]),
        am.assign(set="agreement", pair_id=range(len(am)), seg_uid=["a%06d" % k for k in am.index])])
    keep["speaker_id"] = keep.pid; keep["label"] = keep.y
    info = dict(tag=tag, n_eligible=len(E), n_eligible_spk=int(E.pid.nunique()),
                n_conflict_rule=int(CONF.sum()), n_pool_rule=int(AGRp.sum()),
                n_pairs=len(Pp), unmatched=unmatched, n_rows=len(keep),
                n_clips=int(keep.groupby(["set", "seg_uid"]).ngroups),
                n_speakers=int(keep.speaker_id.nunique()),
                depressed_positive=int((cm.y == 1).sum()),
                control_negative=int((cm.y == 0).sum()),
                uniq_conflict_seg=int(cm.index.nunique()),
                uniq_agreement_seg=int(am.index.nunique()))
    # per-segment rule arm over the eligible pool, and selected arm
    rule_arm = pd.Series("none", index=E.index)
    rule_arm[AGRp.values] = "agreement"
    rule_arm[CONF.values] = "conflict"          # conflict wins if somehow both
    sel_arm = pd.Series("none", index=E.index)
    sel_arm.loc[sorted(set(am.index))] = "agreement"
    sel_arm.loc[sorted(set(cm.index))] = "conflict"
    return keep, info, rule_arm, sel_arm, E

# --- replication check: rerun the rule on the SAVED RoBERTa columns -----
rob_pos = S.clearly_positive == 1
rob_neg = S.clearly_negative == 1
keep_r, info_r, rulearm_r, selarm_r, E = select(S, rob_pos, rob_neg, "roberta")
print("\n=== REPLICATION OF PART 14 UNDER RoBERTa ===", flush=True)
print(json.dumps(info_r, indent=1), flush=True)
rep_keys = set(zip(keep_r["set"], keep_r.seg_uid))
man_keys = set(zip(MANIFEST["set"], MANIFEST.seg_uid))
repl_exact = (rep_keys == man_keys) and (len(keep_r) == len(MANIFEST))
print(f"replication matches part14_manifest_new.csv exactly: {repl_exact} "
      f"(rebuilt rows {len(keep_r)} vs manifest {len(MANIFEST)}; "
      f"key overlap {len(rep_keys & man_keys)})", flush=True)
if not repl_exact:
    disc("HIGH", f"re-running the Part 14 rule on the saved RoBERTa columns did not reproduce "
                 f"part14_manifest_new.csv exactly: rebuilt {len(keep_r)} rows vs {len(MANIFEST)}, "
                 f"key overlap {len(rep_keys & man_keys)}. M2 distilbert arms still built with the same code.")

# ------------------------------------------------ score with distilbert
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MID)
mod = AutoModelForSequenceClassification.from_pretrained(MID).to(dev).eval()
id2 = {int(k): v for k, v in mod.config.id2label.items()}
print(f"\ndistilbert id2label {id2}  device {dev}", flush=True)
lab = {v.lower(): k for k, v in id2.items()}
iPOS, iNEG = lab["positive"], lab["negative"]

txt = S.text.fillna("").astype(str).tolist()
p_pos_db = np.zeros(len(txt), dtype=np.float64)
B, t0 = 64, time.time()
for i in range(0, len(txt), B):
    b = [t if t.strip() else "." for t in txt[i:i + B]]
    enc = tok(b, return_tensors="pt", truncation=True, max_length=512, padding=True).to(dev)
    with torch.no_grad():
        pr = torch.softmax(mod(**enc).logits.float().cpu(), -1).numpy()
    p_pos_db[i:i + len(b)] = pr[:, iPOS]
    if (i // B) % 20 == 0:
        print(f"  {i + len(b)}/{len(txt)}  {time.time() - t0:.0f}s", flush=True)
S["db_p_pos"] = p_pos_db
S["db_p_neg"] = 1.0 - p_pos_db
S["db_clearly_positive"] = (S.db_p_pos >= THR).astype(int)
S["db_clearly_negative"] = (S.db_p_neg >= THR).astype(int)
print(f"\ndistilbert clearly positive {int(S.db_clearly_positive.sum())}  "
      f"clearly negative {int(S.db_clearly_negative.sum())}  "
      f"neither {int(len(S) - S.db_clearly_positive.sum() - S.db_clearly_negative.sum())}  "
      f"({(time.time() - t0) / 60:.2f} min)", flush=True)

keep_d, info_d, rulearm_d, selarm_d, _ = select(S, S.db_clearly_positive == 1,
                                                S.db_clearly_negative == 1, "distilbert")
print("\n=== NEW ARMS UNDER DISTILBERT SST-2 ===", flush=True)
print(json.dumps(info_d, indent=1), flush=True)

# ---------------------------------------------- agreement / kappa by hand
def kappa(a, b, cats):
    n = len(a)
    idx = {c: i for i, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)), float)
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    po = np.trace(M) / n
    pe = float(M.sum(1) @ M.sum(0)) / (n * n)
    k = float("nan") if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)
    return po, pe, k, M

def three(pos, neg):
    out = np.where(pos.values == 1, "positive", np.where(neg.values == 1, "negative", "neutral"))
    return pd.Series(out, index=pos.index)

lab_r_all = three(S.clearly_positive, S.clearly_negative)
lab_d_all = three(S.db_clearly_positive, S.db_clearly_negative)

CELLS = []   # (id, what, series_r, series_d, cats, speaker series)
CELLS.append(("M2.label_all",   "sentiment label (pos/neu/neg at 0.70), all segments",
              lab_r_all, lab_d_all, ["negative", "neutral", "positive"], S.pid))
CELLS.append(("M2.label_elig",  "sentiment label (pos/neu/neg at 0.70), eligible segments",
              lab_r_all.loc[E.index], lab_d_all.loc[E.index],
              ["negative", "neutral", "positive"], E.pid))
CELLS.append(("M2.arm_rule",    "rule arm (conflict/agreement/none), eligible segments",
              rulearm_r, rulearm_d, ["agreement", "conflict", "none"], E.pid))
CELLS.append(("M2.arm_selected", "selected arm (conflict/agreement/none), eligible segments",
              selarm_r, selarm_d, ["agreement", "conflict", "none"], E.pid))

def boot(rv, dv, spk, cats, stat, draws=DRAWS, seed=0):
    rng = np.random.default_rng(seed)
    spk = np.asarray(spk); rv = np.asarray(rv); dv = np.asarray(dv)
    us = np.unique(spk)
    where = {s: np.where(spk == s)[0] for s in us}
    vals, bad = [], 0
    for _ in range(draws):
        pick = rng.choice(us, size=len(us), replace=True)
        idx = np.concatenate([where[s] for s in pick])
        po, pe, k, _ = kappa(rv[idx], dv[idx], cats)
        v = po if stat == "po" else k
        if v is None or not np.isfinite(v):
            bad += 1
        else:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan"), 0, draws
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals), bad

report, rows_tsv = [], []
report.append("=" * 100)
report.append("PART16 / M2  distilbert-base-uncased-finetuned-sst-2-english vs "
              "cardiffnlp/twitter-roberta-base-sentiment-latest")
report.append("=" * 100)

for cid, what, sr, sd, cats, spk in CELLS:
    po, pe, k, M = kappa(sr.values, sd.values, cats)
    n = len(sr); nspk = int(pd.Series(spk).nunique())
    lo_po, hi_po, use_po, bad_po = boot(sr.values, sd.values, spk.values, cats, "po")
    lo_k,  hi_k,  use_k,  bad_k  = boot(sr.values, sd.values, spk.values, cats, "k")
    report.append("")
    report.append(f"--- {cid}  {what}")
    report.append(f"    n={n}  n_speakers={nspk}")
    report.append(f"    agreement po={po:.4f} [{lo_po:.4f}, {hi_po:.4f}]  usable draws {use_po}/{DRAWS}")
    report.append(f"    pe={pe:.4f}   kappa={k:.4f} [{lo_k:.4f}, {hi_k:.4f}]  usable draws {use_k}/{DRAWS}")
    report.append(f"    confusion matrix (rows RoBERTa, cols DistilBERT), order {cats}:")
    hdr = "         " + "".join(f"{c:>12}" for c in cats) + f"{'rowsum':>12}"
    report.append(hdr)
    for i, c in enumerate(cats):
        report.append(f"    {c:<9}" + "".join(f"{int(M[i, j]):>12}" for j in range(len(cats)))
                      + f"{int(M[i].sum()):>12}")
    report.append("    " + "colsum   " + "".join(f"{int(M[:, j].sum()):>12}" for j in range(len(cats)))
                  + f"{int(M.sum()):>12}")
    rows_tsv.append((cid + ".agreement", what + " | raw agreement po", f"{po:.4f}",
                     f"{lo_po:.4f}", f"{hi_po:.4f}", n, nspk, OUT + "/M2_distilbert_sentiment.csv"))
    rows_tsv.append((cid + ".kappa", what + " | Cohen kappa", f"{k:.4f}",
                     f"{lo_k:.4f}", f"{hi_k:.4f}", n, nspk, OUT + "/M2_distilbert_sentiment.csv"))

report.append("")
report.append("--- arm counts")
for tag, info in (("roberta", info_r), ("distilbert", info_d)):
    report.append(f"    {tag:<11} pairs={info['n_pairs']:<5} clips={info['n_rows']:<5} "
                  f"speakers={info['n_speakers']:<5} unmatched={info['unmatched']:<4} "
                  f"conflict_rule={info['n_conflict_rule']:<5} pool_rule={info['n_pool_rule']:<5} "
                  f"dep_pos={info['depressed_positive']:<4} ctrl_neg={info['control_negative']}")
report.append(f"    part14 manifest on disk: rows={len(MANIFEST)} "
              f"pairs={MANIFEST.pair_id.nunique()} speakers={MANIFEST.speaker_id.nunique()}")
report.append(f"    replication of Part 14 from saved RoBERTa columns exact: {repl_exact}")

# seg_uid overlap of the two selections
rk, dk = set(zip(keep_r["set"], keep_r.seg_uid)), set(zip(keep_d["set"], keep_d.seg_uid))
rs, ds = set(keep_r.seg_uid.str[1:]), set(keep_d.seg_uid.str[1:])
report.append("")
report.append("--- selection overlap (RoBERTa arms vs DistilBERT arms)")
report.append(f"    same (arm, seg_uid): {len(rk & dk)}")
report.append(f"    segment in both selections regardless of arm: {len(rs & ds)}")
report.append(f"    distilbert-only segments: {len(ds - rs)}   roberta-only segments: {len(rs - ds)}")
report.append(f"    speakers: roberta {info_r['n_speakers']}, distilbert {info_d['n_speakers']}, "
              f"shared {len(set(keep_r.speaker_id) & set(keep_d.speaker_id))}")


# ---- matched-count threshold sweep: is the blow-up only confidence scale? ----
report.append("")
report.append("--- distilbert threshold sweep (does a higher cut reproduce the RoBERTa-sized set?)")
report.append(f"    {'thr':>6}{'clearly_pos':>13}{'clearly_neg':>13}{'conflict_rule':>15}{'pairs':>8}{'spk':>6}")
sweep_rows = []
TARGET = info_r["n_conflict_rule"]
best = None
for thr in [0.70, 0.80, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]:
    pos_t = (S.db_p_pos >= thr).astype(int)
    neg_t = (S.db_p_neg >= thr).astype(int)
    k_t, i_t, ra_t, sa_t, _ = select(S, pos_t == 1, neg_t == 1, f"db@{thr}")
    report.append(f"    {thr:>6}{int(pos_t.sum()):>13}{int(neg_t.sum()):>13}"
                  f"{i_t['n_conflict_rule']:>15}{i_t['n_pairs']:>8}{i_t['n_speakers']:>6}")
    sweep_rows.append((thr, i_t, ra_t, sa_t))
    if best is None or abs(i_t["n_conflict_rule"] - TARGET) < abs(best[1]["n_conflict_rule"] - TARGET):
        best = (thr, i_t, ra_t, sa_t)
bthr, binfo, bra, bsa = best
report.append(f"    closest to the RoBERTa conflict count ({TARGET}): thr={bthr} "
              f"-> conflict_rule={binfo['n_conflict_rule']} pairs={binfo['n_pairs']} "
              f"speakers={binfo['n_speakers']}")
for cid, what, sd2 in (("M2.arm_rule_matched", f"rule arm, distilbert at matched threshold {bthr}", bra),
                       ("M2.arm_selected_matched", f"selected arm, distilbert at matched threshold {bthr}", bsa)):
    sr2 = rulearm_r if "rule" in cid else selarm_r
    cats = ["agreement", "conflict", "none"]
    po2, pe2, k2, M2m = kappa(sr2.values, sd2.values, cats)
    lo_po2, hi_po2, u1, b1 = boot(sr2.values, sd2.values, E.pid.values, cats, "po")
    lo_k2, hi_k2, u2, b2 = boot(sr2.values, sd2.values, E.pid.values, cats, "k")
    report.append("")
    report.append(f"--- {cid}  {what}")
    report.append(f"    n={len(sr2)}  n_speakers={int(E.pid.nunique())}")
    report.append(f"    agreement po={po2:.4f} [{lo_po2:.4f}, {hi_po2:.4f}]  usable draws {u1}/{DRAWS}")
    report.append(f"    pe={pe2:.4f}   kappa={k2:.4f} [{lo_k2:.4f}, {hi_k2:.4f}]  usable draws {u2}/{DRAWS}")
    hdr = "         " + "".join(f"{c:>12}" for c in cats) + f"{'rowsum':>12}"
    report.append("    confusion matrix (rows RoBERTa, cols DistilBERT@%s), order %s:" % (bthr, cats))
    report.append(hdr)
    for i2, c2 in enumerate(cats):
        report.append(f"    {c2:<9}" + "".join(f"{int(M2m[i2, j2]):>12}" for j2 in range(3))
                      + f"{int(M2m[i2].sum()):>12}")
    rows_tsv.append((cid + ".agreement", what + " | raw agreement po", f"{po2:.4f}",
                     f"{lo_po2:.4f}", f"{hi_po2:.4f}", len(sr2), int(E.pid.nunique()),
                     OUT + "/M2_distilbert_sentiment.csv"))
    rows_tsv.append((cid + ".kappa", what + " | Cohen kappa", f"{k2:.4f}",
                     f"{lo_k2:.4f}", f"{hi_k2:.4f}", len(sr2), int(E.pid.nunique()),
                     OUT + "/M2_distilbert_sentiment.csv"))
side_sweep = [dict(threshold=t, **{k: v for k, v in i.items() if k != "tag"}) for t, i, _, _ in sweep_rows]
rows_tsv.append(("M2.matched_threshold",
                 "distilbert threshold whose conflict count is closest to RoBERTa's 488",
                 str(bthr), "", "", binfo["n_conflict_rule"], binfo["n_speakers"],
                 OUT + "/M2_REPORT.txt"))

for cid, val, n, nspk in (("M2.pairs", info_d["n_pairs"], info_d["n_rows"], info_d["n_speakers"]),
                          ("M2.clips", info_d["n_rows"], info_d["n_rows"], info_d["n_speakers"]),
                          ("M2.speakers", info_d["n_speakers"], info_d["n_rows"], info_d["n_speakers"])):
    rows_tsv.append((cid, "distilbert SST-2 rebuild of the Part 14 selection | " + cid.split(".")[1],
                     str(val), "", "", n, nspk, OUT + "/M2_distilbert_sentiment.csv"))
rows_tsv.append(("M2.same_arm_seg_uid", "segments with the same (arm, seg_uid) in both selections",
                 str(len(rk & dk)), "", "", len(rk | dk), info_d["n_speakers"],
                 OUT + "/M2_distilbert_sentiment.csv"))

# ------------------------------------------------------------- write out
per = pd.DataFrame({
    "id": E.index.astype(int),
    "pid": E.pid.values,
    "label_y": E.y.values,
    "sev": E.sev.values,
    "text_sha": [hashlib.sha256(str(t).encode("utf-8")).hexdigest() for t in E.text.fillna("")],
    "roberta_label": lab_r_all.loc[E.index].values,
    "roberta_score": E.p_pos.values,
    "roberta_p_neg": E.p_neg.values,
    "roberta_p_neu": E.p_neu.values,
    "distilbert_label": lab_d_all.loc[E.index].values,
    "distilbert_score": S.db_p_pos.loc[E.index].values,
    "distilbert_p_neg": S.db_p_neg.loc[E.index].values,
    "arm_roberta": rulearm_r.values,
    "arm_distilbert": rulearm_d.values,
    "arm_roberta_selected": selarm_r.values,
    "arm_distilbert_selected": selarm_d.values,
})
PC = OUT + "/M2_distilbert_sentiment.csv"
per.to_csv(PC, index=False)
MC = OUT + "/M2_distilbert_manifest.csv"
cols = [c for c in ["seg_uid", "set", "pair_id", "speaker_id", "label", "sev", "start", "end",
                    "span", "speech_s", "nw", "n_turns", "db_p_pos", "db_p_neg",
                    "db_clearly_positive", "db_clearly_negative", "text"] if c in keep_d.columns]
keep_d[cols].to_csv(MC, index=False)

side = dict(
    task="PART16 M2 second sentiment scorer on the Part 14 selection",
    generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    shell_command=CMD,
    model_id=MID, model_id2label=id2, device=dev, threshold=THR,
    prompt=None,
    label_mapping_judgement=(
        "SST-2 is binary (NEGATIVE/POSITIVE) while cardiffnlp is three-way "
        "(negative/neutral/positive). Mapping used, as instructed: "
        "distilbert_score = P(POSITIVE); distilbert negative score = 1 - P(POSITIVE). "
        "clearly_positive = P(POSITIVE) >= 0.70; clearly_negative = 1-P(POSITIVE) >= 0.70 "
        "i.e. P(POSITIVE) <= 0.30. Everything in 0.30 < P(POSITIVE) < 0.70 is treated as the "
        "analogue of cardiffnlp's 'neutral' arm, i.e. neither clearly positive nor clearly "
        "negative, so it enters neither the conflict rule nor the agreement pool. This is the "
        "judgement call: SST-2 has no neutral head, so its neutral band is induced purely by the "
        "0.70 / 0.30 confidence cut rather than by a trained neutral class. SST-2 is also far "
        "more confident than cardiffnlp (it is trained on movie reviews with no abstain class), "
        "so its induced neutral band is much narrower in practice."),
    part14_rule_restated=(
        "span=end-start; eligible = speech_s>=5 & nw>=25 & span>=8; "
        "POS = clearly_positive==1 (p_pos>=0.70); NEG = clearly_negative==1 (p_neg>=0.70); "
        "conflict = (y==1 & sev>=15 & POS) | (y==0 & sev<=4 & NEG); "
        "agreement pool = (y==1 & NEG) | (y==0 & POS); "
        "match = same speaker's pool segment with the smallest |span difference|, itself excluded, "
        "pool members not consumed, ties broken by first row order (pandas idxmin); "
        "conflict segments with no same-speaker candidate are dropped; "
        "seg_uid = 'c%06d'/'a%06d' %% segment-csv row index."),
    source_files={p: dict(path=p, exists=os.path.exists(p), sha256_first_1mb=sha1mb(p))
                  for p in [SEG, MAN, SRC]},
    part14_builder_scripts_read=[
        "<local data dir>/scratch/"
        "part14_select.py",
        "<local data dir>/scratch/"
        "part14_sentiment.py",
        "<local data dir>/Desktop/release/edaic_rerun/scripts_part14/part14_boot.py"],
    n=int(len(per)), n_segments_all=int(len(S)),
    n_speakers=int(E.pid.nunique()),
    seed=0, bootstrap_draws=DRAWS, bootstrap_unit="speaker (pid)", bootstrap_pct=[2.5, 97.5],
    roberta_arms=info_r, distilbert_arms=info_d,
    part14_manifest_on_disk=dict(rows=int(len(MANIFEST)), pairs=int(MANIFEST.pair_id.nunique()),
                                 speakers=int(MANIFEST.speaker_id.nunique())),
    replication_of_part14_exact=bool(repl_exact),
    auc_note="No AUC is produced by this task; no model scores are ranked here.",
    folds_note="No cross-validation is involved in this task, so no fold file was needed.",
    distilbert_threshold_sweep=side_sweep,
    matched_threshold=bthr,
    outputs=[PC, MC, OUT + "/rows/M2.tsv", OUT + "/M2_distilbert_sentiment.json",
             OUT + "/M2_REPORT.txt"],
)
with open(OUT + "/M2_distilbert_sentiment.json", "w") as f:
    json.dump(side, f, indent=1, default=str)

with open(OUT + "/rows/M2.tsv", "w") as f:
    f.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n")
    for r in rows_tsv:
        f.write("\t".join(str(x) for x in r) + "\n")

txt_report = "\n".join(report)
with open(OUT + "/M2_REPORT.txt", "w") as f:
    f.write(txt_report + "\n")
print("\n" + txt_report, flush=True)
print("\nwrote:\n  " + "\n  ".join(side["outputs"]), flush=True)
print("M2_DONE", flush=True)
