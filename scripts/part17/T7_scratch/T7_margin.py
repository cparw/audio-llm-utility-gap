# Inner-CV layer scores at the two outer folds whose layer pick differs from the pod run.
import sys, csv, numpy as np
sys.path.insert(0, "<local data dir>/release/edaic_rerun/part17")
from T7_probe import P16, SHARDS, score_layer
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
sc = list(csv.DictReader(open(f"{P16}/full_zeroshot_scores.csv")))
clips = [r["clip"] for r in sc]; y = np.array([int(r["label"]) for r in sc]); spk = np.array([r["speaker"] for r in sc])
for key, rep, fold, pub_pick in [("llm", 1, 1, 18), ("ans", 0, 1, 28)]:
    X = np.stack([np.load(f"{SHARDS}/{c}.npz")[key] for c in clips])
    outer = list(GroupKFold(n_splits=5, shuffle=True, random_state=rep).split(X, y, groups=spk))
    tr, te = outer[fold]
    isp = list(GroupKFold(n_splits=4, shuffle=True, random_state=rep).split(X[tr], y[tr], groups=spk[tr]))
    s = np.array(Parallel(n_jobs=16)(delayed(score_layer)(X[tr], y[tr], isp, L) for L in range(X.shape[1])))
    o = np.argsort(-s, kind="stable")
    print(f"{key} rep {rep} fold {fold}: mac pick {int(np.argmax(s))}  pod pick {pub_pick}")
    print("   top5:", [(int(L), round(float(s[L]), 5)) for L in o[:5]])
    print(f"   inner AUC at pod pick L{pub_pick}: {s[pub_pick]:.5f}  gap to mac winner {s.max()-s[pub_pick]:.5f}  n_inner_train {len(tr)}")
