"""Check only (the run itself READS the saved fold file). sft_projector.py derives folds as
GroupKFold(n_splits=5).split(zeros(n), y, groups=speaker) over manifest row order; sft_full_oof.csv is written in
manifest row order, so re-deriving on that order shows the fold membership the 300 s-labelled run implies."""
import csv, numpy as np, sklearn
from sklearn.model_selection import GroupKFold
rows = list(csv.DictReader(open("/workspace/p23/sft_full_oof.csv")))
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
imp = {}
for f, (tr, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups=spk)):
    for i in te: imp[spk[i]] = f
saved = {r["speaker_id"]: int(r["fold"]) for r in csv.DictReader(open("/workspace/p23/edaic_groupkfold5_pod.csv"))}
agree = sum(imp[s] == saved[s] for s in saved)
print(f"sklearn {sklearn.__version__} numpy {np.__version__}: implied vs saved fold agree {agree}/{len(saved)}; implied sizes {np.bincount(list(imp.values())).tolist()} saved sizes {np.bincount(list(saved.values())).tolist()}")
