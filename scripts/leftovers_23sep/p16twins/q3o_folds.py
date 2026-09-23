import sys, hashlib, numpy as np, sklearn
from sklearn.model_selection import GroupKFold
m = np.load(sys.argv[1], allow_pickle=True); y = m["label"].astype(int); spk = m["spk"].astype(str); name = m["name"].astype(str)
f = [hashlib.md5(np.sort(te).tobytes()).hexdigest()[:8] for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups=spk)]
arm = np.array([n.split("_", 1)[0] for n in name]); fc = {}
for a in ("conflict", "agreement"):
    k = arm == a
    fc[a] = [hashlib.md5(np.sort(te).tobytes()).hexdigest()[:8] for tr, te in GroupKFold(n_splits=5).split(np.zeros(k.sum()), y[k], groups=spk[k])]
print("sklearn", sklearn.__version__, "Q3O Pitt all-cell folds", f, "arm folds", fc)
