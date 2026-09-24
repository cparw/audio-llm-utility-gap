"""Recompute the E-DAIC whole interview zero-shot answer AUC (the 0.8366 line)
with sklearn roc_auc_score, plus the speaker bootstrap (2000 draws, rng(0),
unique speakers resampled with replacement, all their clips, 2.5/97.5 pct).
Independent code, read only."""
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

P = ("scores/part23/pull/"
     "p23-a-probes/p23a/out/A2_edaic_whole_zeroshot_perclip.csv")
df = pd.read_csv(P)


def num(v):
    m = re.search(r"[-+0-9.eE]+", str(v).replace("np.float64(", ""))
    return float(m.group(0))


for c in ("p_yes_whole", "p_yes_rule_whole"):
    df[c + "_f"] = df[c].map(num)

print("rows", len(df), "speakers", df.speaker.nunique(), "pos", int(df.label.sum()),
      "capped", int(df.capped.sum()))
y = df.label.values
for c in ("p_yes_whole_f", "p_yes_rule_whole_f"):
    auc = roc_auc_score(y, df[c].values)
    rng = np.random.default_rng(0)
    spk = df.speaker.unique()
    idx = {s: np.flatnonzero(df.speaker.values == s) for s in spk}
    vals = []
    for _ in range(2000):
        pick = rng.choice(spk, size=len(spk), replace=True)
        ii = np.concatenate([idx[s] for s in pick])
        yy = y[ii]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, df[c].values[ii]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"{c}: AUC {auc:.6f}  CI [{lo:.4f}, {hi:.4f}]  draws used {len(vals)}")
