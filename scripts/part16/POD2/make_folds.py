"""Write the 5 speaker folds once, so every run loads the same file.
GroupKFold(n_splits=5) grouped by speaker over the manifest's own row order --
the identical call used by sft_projector.py, which produced the paper's projector fine-tune.
usage: make_folds.py <manifest.csv> <out_folds.csv>"""
import csv, sys, json, numpy as np, sklearn
from sklearn.model_selection import GroupKFold
MAN, OUT = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(MAN)))
y = np.array([int(r["label"]) for r in rows]); spk = np.array([str(r["speaker"]) for r in rows])
fold = np.full(len(rows), -1, int)
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups=spk)):
    fold[te] = k
assert (fold >= 0).all()
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["row", "path", "speaker", "fold"])
    for i, r in enumerate(rows): w.writerow([i, r["path"], r["speaker"], fold[i]])
# speaker disjointness check
bad = [s for s in set(spk) if len(set(fold[spk == s])) > 1]
print(json.dumps({"sklearn": sklearn.__version__, "n": len(rows), "n_spk": int(len(set(spk))),
                  "fold_sizes": np.bincount(fold).tolist(),
                  "fold_pos": [int(y[fold == k].sum()) for k in range(5)],
                  "speakers_split_across_folds": bad}, indent=1))
