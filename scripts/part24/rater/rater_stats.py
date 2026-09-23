#!/usr/local/bin/python3
"""
PART 24 item 3. Rater agreement on the 80 item heard window rater sheet (E-DAIC).

Inputs
  sheet : the filled rater sheet. Two layouts are read:
          local  id,dataset,label,transcript,rater1,rater2   (rater_sheet_heard80.csv)
          live   id,text,model label,rater 1,rater 2         (Google Sheet export)
  key   : rater_key_heard80.csv  (id, label, automatic_decision = arm, pid,
          lex_direction, sent_heard, source_id)
  sent  : part9_heard_window_sentiment.csv (per segment RoBERTa probabilities on
          the heard window; used to check the key and the sheet text, and
          written into the per item csv)

Columns, as they are on disk
  id                  R001..R080, blind ids minted after the shuffle
  arm                 key.automatic_decision, 40 conflict and 40 agreement
  scorer direction    key.sent_heard = sign of cardiffnlp/twitter-roberta-base-
                      sentiment-latest on the heard window text the rater reads
                      (p_pos > p_neg -> positive). This is the primary scorer.
                      key.lex_direction = dq_lex lexicon direction that set the
                      arm (conflict = label 1 with positive words, or label 0 with
                      negative words). Reported as a second scorer row.
                      A positive text implies "not depressed", a negative text
                      implies "depressed".
  rater1, rater2      the two human columns

What a rater cell means (--question)
  away       Yes = the words point away from the diagnosis label shown on the
             row (the PART 4 question, "do the words point away from the
             label"), No = the words go with it. conflict/agreement typed
             literally are read the same way.
  depressed  Yes = the words suggest depression, No = they do not.
  auto       (default) away when the sheet shows the diagnosis label (local
             layout), depressed when it does not (live layout, which shows the
             model label instead). positive/negative typed literally are always
             read as the text direction.
  Every cell is turned into the text direction the rater reads (positive or
  negative) and compared with the scorer direction. Inter rater agreement and
  kappa are computed on the answers as asked (Yes/No for away, the direction
  otherwise).

Statistics, on all 80 items, on the 40 conflict items, and on the 40 agreement
items (the last is a reading check, not a paste line)
  rater k agreement with the scorer direction   proportion, sklearn not needed
  rater k Cohen's kappa with the scorer          sklearn cohen_kappa_score
  inter rater raw agreement                      proportion
  inter rater Cohen's kappa                      sklearn cohen_kappa_score
Intervals: 2000 draws, a fresh numpy default_rng(0) per cell,
  idx = rng.choice(N, size=N, replace=True) over items, 2.5 and 97.5
  percentiles. A second interval resamples speakers (pid) the same way and
  takes every item of each drawn speaker, since 80 items come from 48 speakers.
  A kappa draw that is undefined (both columns constant and identical) is
  dropped and counted, never replaced.

Outputs (in --out)
  rater_stats.csv, rater_stats_items.csv, rater_stats.json, rater_stats_paste.txt
  (with --tag T the names carry _T)

Usage
  /usr/local/bin/python3 rater_stats.py                    # local sheet, needs 80/80 filled
  /usr/local/bin/python3 rater_stats.py --sheet live.csv   # a dated copy of the live sheet
  /usr/local/bin/python3 rater_stats.py --allow-partial    # compute on the filled rows only
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import cohen_kappa_score

REL = "<local data dir>/release/edaic_rerun"
DEF_SHEET = os.path.join(REL, "rater_sheet_heard80.csv")
DEF_KEY = os.path.join(REL, "rater_key_heard80.csv")
DEF_SENT = os.path.join(REL, "part9_heard_window_sentiment.csv")
DEF_OUT = "<local data dir>/part24/rater"

N_BOOT = 2000
SEED = 0
N_ITEMS = 80

YES = {"yes", "y", "1", "true", "t"}
NO = {"no", "n", "0", "false", "f"}
POS = {"positive", "pos"}
NEG = {"negative", "neg"}
CONF = {"conflict", "away"}
AGR = {"agreement", "toward", "towards"}

SCORERS = [
    ("roberta_heard", "sent_heard",
     "cardiffnlp/twitter-roberta-base-sentiment-latest sign on the heard window"),
    ("lexicon_arm", "lex_direction",
     "dq_lex lexicon direction that set the conflict/agreement arm"),
]
SUBSETS = [("all80", None), ("conflict40", "conflict"), ("agreement40", "agreement")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_col(c):
    return str(c).strip().lower().replace(" ", "").replace("_", "")


def canon(v):
    """Cell -> 'yes','no','pos','neg','conflict','agreement', '' (empty) or 'UNREADABLE'."""
    if v is None:
        return ""
    s = str(v).strip().lower().rstrip(".!")
    if s in ("", "nan", "none"):
        return ""
    if s in YES:
        return "yes"
    if s in NO:
        return "no"
    if s in POS:
        return "pos"
    if s in NEG:
        return "neg"
    if s in CONF:
        return "conflict"
    if s in AGR:
        return "agreement"
    return "UNREADABLE"


def to_direction(tok, label, reading):
    """Rater token -> text direction 'positive'/'negative' the rater reads, or None."""
    if tok == "pos":
        return "positive"
    if tok == "neg":
        return "negative"
    if tok in ("conflict", "agreement"):
        tok = "yes" if tok == "conflict" else "no"
        reading = "away"
    if tok not in ("yes", "no"):
        return None
    if reading == "depressed":
        return "negative" if tok == "yes" else "positive"
    # away: Yes = words point away from the label
    toward = "negative" if int(label) == 1 else "positive"
    away = "positive" if int(label) == 1 else "negative"
    return away if tok == "yes" else toward


def to_away(tok, label, reading):
    """Rater token -> 'yes'/'no' on the away question, or None."""
    d = to_direction(tok, label, reading)
    if d is None:
        return None
    toward = "negative" if int(label) == 1 else "positive"
    return "no" if d == toward else "yes"


def kappa(a, b, labels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(cohen_kappa_score(a, b, labels=labels))


def boot(stat, n, groups=None):
    """Percentile bootstrap. groups=None resamples items; else resamples groups
    (sorted unique pid) and takes all items of each drawn group."""
    rng = np.random.default_rng(SEED)
    vals, undefined = [], 0
    if groups is None:
        for _ in range(N_BOOT):
            idx = rng.choice(n, size=n, replace=True)
            v = stat(idx)
            if np.isnan(v):
                undefined += 1
            else:
                vals.append(v)
    else:
        uniq = np.array(sorted(set(groups)))
        members = [np.flatnonzero(groups == g) for g in uniq]
        S = len(uniq)
        for _ in range(N_BOOT):
            sidx = rng.choice(S, size=S, replace=True)
            idx = np.concatenate([members[j] for j in sidx])
            v = stat(idx)
            if np.isnan(v):
                undefined += 1
            else:
                vals.append(v)
    if not vals:
        return float("nan"), float("nan"), 0, undefined
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals), undefined


def load(args):
    raw = pd.read_csv(args.sheet, dtype=str, keep_default_na=False)
    cols = {norm_col(c): c for c in raw.columns}
    for need in ("id", "rater1", "rater2"):
        if need not in cols:
            sys.exit("STOP: sheet %s has no %s column (columns: %s)" % (args.sheet, need, list(raw.columns)))
    text_col = cols.get("transcript") or cols.get("text")
    if text_col is None:
        sys.exit("STOP: sheet has neither a transcript nor a text column")
    layout = "local" if "label" in cols else "live"
    sheet = pd.DataFrame({
        "id": raw[cols["id"]].str.strip(),
        "sheet_text": raw[text_col],
        "rater1_raw": raw[cols["rater1"]],
        "rater2_raw": raw[cols["rater2"]],
    })
    if layout == "local":
        sheet["sheet_label"] = raw[cols["label"]].str.strip()

    key = pd.read_csv(args.key)
    sent = pd.read_csv(args.sent).drop_duplicates("seg_uid").set_index("seg_uid")
    key = key.join(sent[["p_neg_heard", "p_neu_heard", "p_pos_heard", "sent_heard", "lex_direction", "heard_text"]],
                   on="source_id", rsuffix="_p9")
    problems = []
    if key["p_neg_heard"].isna().any():
        problems.append("key rows missing from part9 sentiment csv: %d" % int(key["p_neg_heard"].isna().sum()))
    if not (key["sent_heard"] == key["sent_heard_p9"]).all():
        problems.append("key sent_heard differs from part9 on %d rows" % int((key["sent_heard"] != key["sent_heard_p9"]).sum()))
    if not (key["lex_direction"] == key["lex_direction_p9"]).all():
        problems.append("key lex_direction differs from part9")
    rs = np.where(key["p_pos_heard"] > key["p_neg_heard"], "positive", "negative")
    if not (rs == key["sent_heard"].values).all():
        problems.append("sent_heard is not the sign of p_pos > p_neg on %d rows" % int((rs != key["sent_heard"].values).sum()))
    arm = np.where(((key["label"] == 1) & (key["lex_direction"] == "positive")) |
                   ((key["label"] == 0) & (key["lex_direction"] == "negative")), "conflict", "agreement")
    if not (arm == key["automatic_decision"].values).all():
        problems.append("arm does not follow label x lex_direction on %d rows" % int((arm != key["automatic_decision"].values).sum()))

    if len(sheet) != N_ITEMS or sheet["id"].duplicated().any():
        problems.append("sheet has %d rows / duplicate ids %d" % (len(sheet), int(sheet["id"].duplicated().sum())))
    df = key.merge(sheet, on="id", how="outer", indicator=True, validate="one_to_one")
    if (df["_merge"] != "both").any():
        problems.append("ids not matched between sheet and key: %s" % df.loc[df["_merge"] != "both", "id"].tolist())
    df = df[df["_merge"] == "both"].drop(columns="_merge").sort_values("id").reset_index(drop=True)
    tmis = int((df["sheet_text"].str.strip() != df["heard_text"].astype(str).str.strip()).sum())
    if tmis:
        problems.append("sheet text differs from the heard window text on %d rows" % tmis)
    if layout == "local":
        lmis = int((df["sheet_label"].astype(int) != df["label"].astype(int)).sum())
        if lmis:
            problems.append("sheet label differs from key label on %d rows" % lmis)
    if problems:
        sys.exit("STOP, input check failed:\n  " + "\n  ".join(problems))
    return df, layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=DEF_SHEET)
    ap.add_argument("--key", default=DEF_KEY)
    ap.add_argument("--sent", default=DEF_SENT)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--question", choices=["auto", "away", "depressed"], default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    df, layout = load(args)
    for r in ("rater1", "rater2"):
        df[r + "_tok"] = df[r + "_raw"].map(canon)

    fill = {}
    for r in ("rater1", "rater2"):
        t = df[r + "_tok"]
        fill[r] = {"non_empty": int((t != "").sum()), "readable": int(((t != "") & (t != "UNREADABLE")).sum()),
                   "unreadable": int((t == "UNREADABLE").sum()), "n": len(df)}
    print("sheet  %s  (layout %s)" % (args.sheet, layout))
    for r in ("rater1", "rater2"):
        print("  %s non empty %d of %d, readable %d, unreadable %d"
              % (r, fill[r]["non_empty"], fill[r]["n"], fill[r]["readable"], fill[r]["unreadable"]))
    bad = df[(df["rater1_tok"] == "UNREADABLE") | (df["rater2_tok"] == "UNREADABLE")]
    if len(bad):
        print("UNREADABLE cells (fix them, nothing is guessed):")
        for _, r in bad.iterrows():
            print("  %s rater1=%r rater2=%r" % (r["id"], r["rater1_raw"], r["rater2_raw"]))
        sys.exit(2)
    full = all(fill[r]["readable"] == N_ITEMS for r in ("rater1", "rater2"))
    if not full and not args.allow_partial:
        print("NOT RUN: both rater columns need %d filled cells (use --allow-partial to override)" % N_ITEMS)
        sys.exit(3)

    toks = set(df["rater1_tok"]) | set(df["rater2_tok"])
    toks.discard("")
    if args.question != "auto":
        reading, why = args.question, "set by --question"
    elif toks and toks <= {"pos", "neg"}:
        reading, why = "valence", "every cell is positive/negative"
    elif layout == "local":
        reading, why = "away", "auto: the local sheet shows the diagnosis label, so Yes = words point away from it (PART 4 question)"
    else:
        reading, why = "depressed", "auto: the live sheet shows no diagnosis label, so Yes = the words suggest depression"
    print("reading: %s  (%s)" % (reading, why))

    def dirs(r, rd):
        return [to_direction(t, l, rd) for t, l in zip(df[r + "_tok"], df["label"])]

    for r in ("rater1", "rater2"):
        df[r + "_dir"] = dirs(r, reading if reading != "valence" else "depressed")
        df[r + "_away"] = [to_away(t, l, reading if reading != "valence" else "depressed")
                           for t, l in zip(df[r + "_tok"], df["label"])]
        for name, col, _ in SCORERS:
            df["%s_match_%s" % (r, name)] = np.where(df[r + "_dir"].isna(), np.nan,
                                                     (df[r + "_dir"] == df[col]).astype(float))
    inter_code = "away" if reading == "away" else "dir"
    inter_labels = ["no", "yes"] if inter_code == "away" else ["negative", "positive"]

    rows = []

    def add(subset, scorer, metric, value, k, n, pid, stat, lab_note=""):
        ilo, ihi, iu, iund = boot(stat, n)
        slo, shi, su, sund = boot(stat, n, groups=pid)
        rows.append({"subset": subset, "scorer": scorer, "metric": metric, "value": value, "k": k, "n": n,
                     "n_speakers": int(len(set(pid))), "ci_item_lo": ilo, "ci_item_hi": ihi,
                     "boot_item_used": iu, "boot_item_undefined": iund,
                     "ci_spk_lo": slo, "ci_spk_hi": shi, "boot_spk_used": su, "boot_spk_undefined": sund,
                     "reading": reading, "note": lab_note})

    for sname, arm in SUBSETS:
        sub = df if arm is None else df[df["automatic_decision"] == arm]
        for r in ("rater1", "rater2"):
            s = sub[sub[r + "_dir"].notna()]
            pid = s["pid"].values
            rd = s[r + "_dir"].values
            for name, col, _ in SCORERS:
                sc = s[col].values
                m = (rd == sc).astype(float)
                add(sname, name, r + "_agree_scorer", float(m.mean()) if len(m) else float("nan"),
                    int(m.sum()), len(m), pid, lambda idx, m=m: float(m[idx].mean()))
                add(sname, name, r + "_kappa_scorer", kappa(rd, sc, ["negative", "positive"]) if len(m) else float("nan"),
                    "", len(m), pid,
                    lambda idx, a=rd, b=sc: kappa(a[idx], b[idx], ["negative", "positive"]))
        both = sub[sub["rater1_dir"].notna() & sub["rater2_dir"].notna()]
        a = both["rater1_" + inter_code].values
        b = both["rater2_" + inter_code].values
        pid = both["pid"].values
        eq = (a == b).astype(float)
        add(sname, "none", "interrater_agree", float(eq.mean()) if len(eq) else float("nan"), int(eq.sum()), len(eq),
            pid, lambda idx, eq=eq: float(eq[idx].mean()), "on %s answers" % inter_code)
        add(sname, "none", "interrater_kappa", kappa(a, b, inter_labels) if len(eq) else float("nan"), "", len(eq),
            pid, lambda idx, a=a, b=b: kappa(a[idx], b[idx], inter_labels), "on %s answers" % inter_code)

    res = pd.DataFrame(rows)

    # reading check: on the 40 agreement items the scorer and the arm agree, so the
    # right reading should fit them well. Compare with the other Yes/No reading.
    warn = []
    if reading in ("away", "depressed"):
        other = "depressed" if reading == "away" else "away"
        ag = df[df["automatic_decision"] == "agreement"]
        for r in ("rater1", "rater2"):
            s = ag[ag[r + "_tok"].isin(["yes", "no"])]
            if len(s) == 0:
                continue
            this = np.mean([to_direction(t, l, reading) == d for t, l, d in zip(s[r + "_tok"], s["label"], s["sent_heard"])])
            alt = np.mean([to_direction(t, l, other) == d for t, l, d in zip(s[r + "_tok"], s["label"], s["sent_heard"])])
            line = "%s on the 40 agreement items: %.4f under '%s', %.4f under '%s'" % (r, this, reading, alt, other)
            print("reading check  " + line)
            if alt > this + 0.25:
                warn.append("WARNING " + line + ". The raters may have answered the other question; rerun with --question " + other)
    for w in warn:
        print(w)

    suf = ("_" + args.tag) if args.tag else ""
    os.makedirs(args.out, exist_ok=True)
    p_csv = os.path.join(args.out, "rater_stats%s.csv" % suf)
    p_items = os.path.join(args.out, "rater_stats_items%s.csv" % suf)
    p_json = os.path.join(args.out, "rater_stats%s.json" % suf)
    p_paste = os.path.join(args.out, "rater_stats_paste%s.txt" % suf)
    res.to_csv(p_csv, index=False, float_format="%.6f")
    items = df[["id", "pid", "source_id", "automatic_decision", "label", "p_neg_heard", "p_neu_heard", "p_pos_heard",
                "sent_heard", "lex_direction", "rater1_raw", "rater2_raw", "rater1_tok", "rater2_tok",
                "rater1_dir", "rater2_dir", "rater1_away", "rater2_away",
                "rater1_match_roberta_heard", "rater2_match_roberta_heard",
                "rater1_match_lexicon_arm", "rater2_match_lexicon_arm"]].rename(columns={"automatic_decision": "arm"})
    items.to_csv(p_items, index=False)

    def g(subset, scorer, metric):
        r = res[(res.subset == subset) & (res.scorer == scorer) & (res.metric == metric)].iloc[0]
        return r

    def f(x):
        return "nan" if x != x else "%.4f" % x

    def f2(x):
        return "nan" if x != x else "%.2f" % x

    lines = []
    if not full:
        lines.append("PARTIAL FILL (--allow-partial): the n in each line is the filled count, not 80 or 40. Not for the paper.")
    for subset, label in (("all80", "All 80 items"), ("conflict40", "The 40 conflict items")):
        for name, col, desc in SCORERS:
            r1 = g(subset, name, "rater1_agree_scorer")
            r2 = g(subset, name, "rater2_agree_scorer")
            who = "the sentiment scorer (RoBERTa, heard window)" if name == "roberta_heard" else "the lexicon direction that set the arm"
            lines.append("%s, %s: rater 1 agrees with the scorer direction on %s (%d of %d, 95%% item bootstrap %s to %s), "
                         "rater 2 on %s (%d of %d, %s to %s)."
                         % (label, who, f(r1.value), r1.k, r1.n, f2(r1.ci_item_lo), f2(r1.ci_item_hi),
                            f(r2.value), r2.k, r2.n, f2(r2.ci_item_lo), f2(r2.ci_item_hi)))
        ia = g(subset, "none", "interrater_agree")
        ik = g(subset, "none", "interrater_kappa")
        lines.append("%s: the two raters agree on %s (%d of %d, %s to %s), Cohen's kappa %s (95%% item bootstrap %s to %s)."
                     % (label, f(ia.value), ia.k, ia.n, f2(ia.ci_item_lo), f2(ia.ci_item_hi),
                        f(ik.value), f2(ik.ci_item_lo), f2(ik.ci_item_hi)))
    lines += warn
    txt = "\n".join(lines)
    print()
    print("PASTE LINES (reading: %s)" % reading)
    print(txt)
    with open(p_paste, "w") as fh:
        fh.write("reading: %s (%s)\n" % (reading, why) + txt + "\n")

    snap = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
    side = {
        "part": "24 item 3, rater agreement",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": " ".join([sys.executable] + sys.argv),
        "inputs": {p: {"path": p, "sha256": sha256(p)} for p in (args.sheet, args.key, args.sent)},
        "sheet_layout": layout,
        "reading": reading, "reading_reason": why,
        "reading_check_warnings": warn,
        "fill": fill,
        "scorers": {n: {"column": c, "what": d} for n, c, d in SCORERS},
        "scorer_model": "cardiffnlp/twitter-roberta-base-sentiment-latest; sign rule p_pos > p_neg, neutral ignored "
                        "(PART8_9_validity.txt 9b). Snapshot not written by the PART 9 run; the HF cache refs/main is "
                        + snap + ", downloaded 2026-09-22 18:40 local, 4 min before part9_heard_window_sentiment.csv",
        "lexicon": "dq_lex via px_build.py lines 137-139 and 159-162 (PART8_9_validity.txt)",
        "seed": SEED, "gpu": "none (CPU, Mac)",
        "bootstrap": "2000 draws; fresh numpy default_rng(0) per cell; idx = rng.choice(N, size=N, replace=True) over "
                     "items; 2.5/97.5 percentiles; undefined kappa draws dropped and counted. Second interval: same "
                     "rule over sorted unique pid, all items of each drawn speaker.",
        "n": {s: int(len(df) if a is None else (df.automatic_decision == a).sum()) for s, a in SUBSETS},
        "n_speakers": {s: int((df if a is None else df[df.automatic_decision == a]).pid.nunique()) for s, a in SUBSETS},
        "interrater_on": inter_code,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__},
        "outputs": [p_csv, p_items, p_paste],
    }
    with open(p_json, "w") as fh:
        json.dump(side, fh, indent=1)
    print()
    for p in (p_csv, p_items, p_json, p_paste):
        print("wrote", p)


if __name__ == "__main__":
    main()
