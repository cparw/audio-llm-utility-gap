import sys, numpy as np, sklearn, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
z = np.load(sys.argv[1], allow_pickle=True)
X = z["ans"][:, -1, :].astype(np.float64); y = z["label"].astype(int); spk = z["spk"].astype(str)
out = []
for tr, te in GroupKFold(n_splits=5).split(X, y, groups=spk):
    sc = StandardScaler().fit(X[tr]); lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
    out.append((int(lr.n_iter_[0]), round(float(np.linalg.norm(lr.coef_)), 4)))
print("sklearn", sklearn.__version__, "n_iter, |coef| per fold", out)
