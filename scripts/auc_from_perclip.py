"""Recompute an AUC and its speaker bootstrap interval from a released per-clip score file.

The released per-clip files carry no clinical label. Join the label from your own licensed
copy of the corpus with --labels, a csv with one row per id and a 0/1 label column.

The interval follows the paper's rule (records/AUDIT.md, section c): 2000 draws of speakers with
replacement from the sorted list of speakers, numpy default_rng(0), draws with one class skipped,
2.5 and 97.5 percentiles. With --minus, the second score is scored on the same draws, so the
interval is for the paired difference.

Example, the E-DAIC zero-shot answer of Table 1 (0.8366 [0.7786, 0.8866]):

  python scripts/auc_from_perclip.py \
      --scores scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv \
      --id pid --speaker speaker --score p_yes_whole \
      --labels my_edaic_labels.csv --label-id pid --label-col label
"""
import argparse
import re

import numpy as np
import pandas as pd


def to_float(v):
    # some files store numbers as text like np.float64(0.5)
    if isinstance(v, str):
        m = re.fullmatch(r"\s*np\.float\d*\((.*)\)\s*", v)
        v = m.group(1) if m else v
    return float(v)


def auc_rank(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def speaker_bootstrap(y, spk, a, b=None, draws=2000, seed=0):
    uspk = np.array(sorted(set(spk)))
    rows = {s: np.flatnonzero(spk == s) for s in uspk}
    rng = np.random.default_rng(seed)
    va, vd = [], []
    for _ in range(draws):
        pick = rng.choice(len(uspk), size=len(uspk), replace=True)
        ii = np.concatenate([rows[uspk[p]] for p in pick])
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        x = auc_rank(yy, a[ii])
        va.append(x)
        if b is not None:
            vd.append(x - auc_rank(yy, b[ii]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return len(va), q(va), (q(vd) if b is not None else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="released per-clip score csv")
    ap.add_argument("--id", required=True, help="id column in the score file")
    ap.add_argument("--speaker", help="speaker column (default: the id column)")
    ap.add_argument("--score", required=True, help="score column, for example p_yes")
    ap.add_argument("--minus", help="second score column for a paired difference, score minus this")
    ap.add_argument("--labels", required=True, help="csv with your own labels")
    ap.add_argument("--label-id", default="id", help="id column in the label csv")
    ap.add_argument("--label-col", default="label", help="0/1 label column in the label csv")
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()

    df = pd.read_csv(a.scores, dtype={a.id: str})
    lab = pd.read_csv(a.labels, dtype={a.label_id: str})[[a.label_id, a.label_col]]
    lab = lab.rename(columns={a.label_id: a.id, a.label_col: "_label"})
    m = df.merge(lab, on=a.id, how="inner")
    if len(m) < len(df):
        print(f"note: {len(df) - len(m)} of {len(df)} rows have no label and are left out")
    y = m["_label"].astype(int).to_numpy()
    spk = m[a.speaker or a.id].astype(str).to_numpy()
    s1 = m[a.score].map(to_float).to_numpy()
    s2 = m[a.minus].map(to_float).to_numpy() if a.minus else None

    usable, ci, dci = speaker_bootstrap(y, spk, s1, s2, draws=a.draws)
    print(f"n clips {len(y)}, n speakers {len(set(spk))}, positives {int(y.sum())}, usable draws {usable}")
    print(f"AUC {a.score}: {auc_rank(y, s1):.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    if s2 is not None:
        d = auc_rank(y, s1) - auc_rank(y, s2)
        print(f"paired {a.score} minus {a.minus}: {d:+.4f} [{dci[0]:+.4f}, {dci[1]:+.4f}]")


if __name__ == "__main__":
    main()
