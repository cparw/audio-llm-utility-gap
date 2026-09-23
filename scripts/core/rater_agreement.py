#!/usr/bin/env python3.10
"""
Rater agreement for the blind conflict-set rater sheet.

Question that was put to the raters, one row at a time (see
overnight/scripts/build_rater_sheet.py and score_rater_sheet.py):

    "do the words point away from the label?"
        Yes -> transcript contradicts the label -> automatic decision 'conflict'
        No  -> transcript goes with the label    -> automatic decision 'agreement'

So a rater cell of Yes/y is scored against 'conflict' and No/n against
'agreement'. Anything else, blank included, is treated as not filled.

Reported per dataset block:
    rater1 percent agreement with the automatic lexicon direction
    rater2 percent agreement with the automatic lexicon direction
    Cohen's kappa between rater1 and rater2 (sklearn.metrics.cohen_kappa_score)
    95 percent bootstrap interval on that kappa, 2000 draws, seed 0, rows resampled
    how many cells each rater filled and how many were left blank

The E-DAIC block is the 80 transcripts drawn as 40 pairs, one conflict arm and
one agreement arm per pair, so both arms are present and percent agreement is a
real agreement rate. A block that carries only one arm (the Pitt block is
single-arm by construction) can only produce a hit rate, and is labelled as such.

Usage
    python3.10 rater_agreement.py                          # default paths
    python3.10 rater_agreement.py filled_sheet.csv key.csv # explicit paths
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

OVERNIGHT = os.path.expanduser("~/Desktop/release/overnight")
DEF_SHEET = os.path.join(OVERNIGHT, "rater_sheet.csv")
DEF_KEY = os.path.join(OVERNIGHT, "rater_key.csv")

# spec: Yes/No case-insensitive, y/n accepted, anything else is blank
YES = {"yes", "y"}
NO = {"no", "n"}

LABELS = ["agreement", "conflict"]   # fixed order so the 2x2 never moves
N_BOOT = 2000
BOOT_SEED = 0
BLOCK_ORDER = ["E-DAIC", "Pitt"]

RULE = "=" * 78
THIN = "-" * 78


def norm_cell(v):
    """Rater cell -> 'conflict' / 'agreement' / None. None means not filled."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in YES:
        return "conflict"
    if s in NO:
        return "agreement"
    return None


def is_empty(v):
    """True when the cell is genuinely empty, as opposed to filled with junk."""
    if v is None:
        return True
    return str(v).strip().lower() in ("", "nan", "none")


def kappa(a, b):
    """Cohen's kappa over the fixed 2-category set. nan when undefined."""
    if len(a) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(cohen_kappa_score(a, b, labels=LABELS))


def bootstrap_kappa(a, b, n_draws=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap on kappa, resampling rows with replacement.

    Returns (lo, hi, n_used, n_undefined). A draw in which kappa is undefined
    (both raters constant and identical, so the chance term is 1) is dropped and
    counted, never silently replaced with a number.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    if n < 2:
        return (float("nan"), float("nan"), 0, 0)
    rng = np.random.default_rng(seed)
    vals = []
    undefined = 0
    for _ in range(n_draws):
        idx = rng.integers(0, n, n)
        k = kappa(a[idx], b[idx])
        if np.isnan(k):
            undefined += 1
        else:
            vals.append(k)
    if len(vals) == 0:
        return (float("nan"), float("nan"), 0, undefined)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi), len(vals), undefined)


def fill_report(df, rater, col):
    """Counts of filled / blank / unreadable cells for one rater in one block."""
    total = len(df)
    filled = int(df[rater].notna().sum())
    raw = df[col]
    empty = int(raw.map(is_empty).sum())
    junk = total - filled - empty
    line = "  %s filled %d of %d, blank %d" % (rater, filled, total, empty)
    if junk:
        line += ", unreadable %d (not Yes/No/y/n, counted as blank)" % junk
    print(line)
    return filled


def score_block(df, name):
    print()
    print(RULE)
    print("BLOCK: %s   rows %d" % (name, len(df)))
    print(RULE)

    arms = sorted(df["auto"].unique())
    single_arm = len(arms) < 2
    print("  automatic decision arms present: %s" % ", ".join(
        "%s=%d" % (a, int((df["auto"] == a).sum())) for a in arms))
    if single_arm:
        print("  CAVEAT: this block carries ONE arm only, it is single-arm by")
        print("          construction, so the percentages below are HIT RATES")
        print("          against a constant target, NOT agreement rates. A rater")
        print("          who answers %s to everything scores 100 percent." %
              ("Yes" if arms[0] == "conflict" else "No"))

    print(THIN)
    print("fill counts")
    n1 = fill_report(df, "rater1", "rater1_raw")
    n2 = fill_report(df, "rater2", "rater2_raw")

    print(THIN)
    word = "hit rate vs automatic lexicon direction" if single_arm else \
           "percent agreement with automatic lexicon direction"
    print(word)
    for rater in ("rater1", "rater2"):
        sub = df[df[rater].notna()]
        if len(sub) == 0:
            print("  %s : NOT ENOUGH DATA, no cell filled in this block" % rater)
            continue
        hits = int((sub[rater] == sub["auto"]).sum())
        print("  %s : %.1f%%  (%d of %d filled rows)"
              % (rater, 100.0 * hits / len(sub), hits, len(sub)))

    print(THIN)
    print("Cohen's kappa, rater1 vs rater2  (sklearn.metrics.cohen_kappa_score)")
    both = df[df["rater1"].notna() & df["rater2"].notna()]
    if len(both) == 0:
        print("  NOT ENOUGH DATA, no row in this block has both raters filled")
        return
    a = list(both["rater1"])
    b = list(both["rater2"])
    k = kappa(a, b)
    raw_ag = 100.0 * float(np.mean([x == y for x, y in zip(a, b)]))
    print("  rows with both filled : %d of %d" % (len(both), len(df)))
    print("  raw agreement         : %.1f%%" % raw_ag)
    if np.isnan(k):
        print("  kappa                 : UNDEFINED, both raters used a single")
        print("                          identical category so chance agreement is 1")
    else:
        print("  kappa                 : %.4f" % k)

    ct = pd.crosstab(both["rater1"], both["rater2"]).reindex(
        index=LABELS, columns=LABELS, fill_value=0)
    print("  2x2 (rows rater1, cols rater2):")
    for line in ct.to_string().split("\n"):
        print("    " + line)

    lo, hi, used, undef = bootstrap_kappa(a, b)
    print("  95%% bootstrap interval on kappa, %d draws, seed %d, rows resampled:"
          % (N_BOOT, BOOT_SEED))
    if used == 0:
        print("    NOT ENOUGH DATA, kappa was undefined in every draw")
    else:
        print("    [%.4f, %.4f]   from %d usable draws" % (lo, hi, used))
        if undef:
            print("    %d of %d draws dropped, kappa undefined in them" % (undef, N_BOOT))
    if name == "E-DAIC":
        print("  NOTE: E-DAIC rows come in 40 conflict/agreement pairs. This")
        print("        bootstrap resamples ROWS, so the two rows of a pair are")
        print("        treated as independent and the interval is mildly optimistic.")


def main(sheet_path, key_path):
    print(RULE)
    print("RATER AGREEMENT")
    print(RULE)
    print("sheet : %s" % sheet_path)
    print("key   : %s" % key_path)

    for p in (sheet_path, key_path):
        if not os.path.exists(p):
            print("NOT AVAILABLE: %s is missing" % p)
            return 1

    sheet = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False)
    for col in ("id", "rater1", "rater2"):
        if col not in sheet.columns:
            print("NOT AVAILABLE: sheet has no column %r" % col)
            return 1
    for col in ("id", "automatic_decision"):
        if col not in key.columns:
            print("NOT AVAILABLE: key has no column %r" % col)
            return 1

    df = sheet.merge(key[["id", "automatic_decision", "pair_id"]]
                     if "pair_id" in key.columns else key[["id", "automatic_decision"]],
                     on="id", how="inner", validate="one_to_one")
    print("rows  : sheet %d, key %d, merged on id %d" % (len(sheet), len(key), len(df)))
    if len(df) != len(sheet):
        print("WARNING: %d sheet rows did not match the key and are excluded"
              % (len(sheet) - len(df)))
    if len(df) == 0:
        print("NOT ENOUGH DATA: nothing merged")
        return 1

    df["rater1_raw"] = df["rater1"]
    df["rater2_raw"] = df["rater2"]
    df["rater1"] = df["rater1_raw"].map(norm_cell)
    df["rater2"] = df["rater2_raw"].map(norm_cell)
    df["auto"] = df["automatic_decision"].astype(str).str.strip().str.lower()

    tot1 = int(df["rater1"].notna().sum())
    tot2 = int(df["rater2"].notna().sum())
    print("filled: rater1 %d of %d, rater2 %d of %d" % (tot1, len(df), tot2, len(df)))
    if tot1 == 0 and tot2 == 0:
        print()
        print("NOT ENOUGH DATA: both rater columns are entirely empty.")
        print("Nothing to score. Fill rater1 / rater2 with Yes or No and re-run.")
        return 0
    for nm, t in (("rater1", tot1), ("rater2", tot2)):
        if t == 0:
            print("NOT ENOUGH DATA: %s is entirely empty, "
                  "its percentages and the kappa will be skipped." % nm)

    blocks = [b for b in BLOCK_ORDER if b in set(df.get("dataset", pd.Series(dtype=str)))]
    blocks += sorted(set(df["dataset"]) - set(blocks)) if "dataset" in df.columns else []
    if not blocks:
        blocks = ["ALL ROWS"]
        df["dataset"] = "ALL ROWS"
    for b in blocks:
        score_block(df[df["dataset"] == b].reset_index(drop=True), b)
    print()
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    s = args[0] if len(args) > 0 else DEF_SHEET
    k = args[1] if len(args) > 1 else DEF_KEY
    sys.exit(main(s, k))
