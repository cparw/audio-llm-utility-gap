import sys, hashlib, numpy as np, sklearn
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
z = np.load(sys.argv[1], allow_pickle=True); y = z["label"].astype(int); spk = z["spk"].astype(str)
X = z["ans"][:, -1, :].astype(np.float64)
f = [hashlib.md5(np.sort(te).tobytes()).hexdigest()[:8] for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups=spk)]
print("sklearn", sklearn.__version__, "fold test-index md5", f)
# same folds for both envs: use speaker hash folds, then compare one fit
g = np.array([int(hashlib.md5(s.encode()).hexdigest(), 16) % 5 for s in spk])
tr = np.flatnonzero(g != 0)
sc = StandardScaler().fit(X[tr]); lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
print("fixed-fold fit |coef| %.6f n_iter %d" % (np.linalg.norm(lr.coef_), lr.n_iter_[0]))
