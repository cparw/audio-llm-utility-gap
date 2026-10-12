"""Words check on adresso: score the words with a text model (out of fold), split clips into
agreement (words agree with the label) and conflict (words contradict it), train a probe on
agreement only, test on conflict only, one clip per speaker so no speaker crosses over."""
import csv, os, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
D = os.path.expanduser("~/Desktop/Hard drive data for paper/adresso")
tr = {r["spk"]: r["text"] for r in csv.DictReader(open(f"{D}/adresso_transcripts.csv"))}
z = np.load(f"{D}/adresso_encoder_states.npz", allow_pickle=True)
F, y, spk = z["feats"], z["label"].astype(int), [str(s) for s in z["spk"]]
texts = [tr[s] for s in spk]
# 1. text model, out of fold word score
txt_oof = np.zeros(len(y))
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(texts, y):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True).fit([texts[i] for i in a])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0).fit(vec.transform([texts[i] for i in a]), y[a])
    txt_oof[b] = clf.decision_function(vec.transform([texts[i] for i in b]))
print(f"text model alone, out of fold AUC {roc_auc_score(y, txt_oof):.3f}", flush=True)
# 2. arms: conflict = words point the wrong way (bottom third of word score for patients, top third for controls)
lo, hi = np.quantile(txt_oof, 1/3), np.quantile(txt_oof, 2/3)
conflict = ((y==1) & (txt_oof <= lo)) | ((y==0) & (txt_oof >= hi))
agreement = ((y==1) & (txt_oof >= hi)) | ((y==0) & (txt_oof <= lo))
print(f"agreement {agreement.sum()} clips ({(y[agreement]==1).sum()} ad), conflict {conflict.sum()} clips ({(y[conflict]==1).sum()} ad)", flush=True)
# 3. probe trained on agreement, tested on conflict, every layer; plus within-agreement CV for reference
rows = []
for L in range(F.shape[1]):
    X = F[:, L, :].astype(np.float64)
    sc = StandardScaler().fit(X[agreement])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X[agreement]), y[agreement])
    conf_auc = roc_auc_score(y[conflict], clf.decision_function(sc.transform(X[conflict])))
    within = []
    ya, Xa = y[agreement], X[agreement]
    for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(Xa, ya):
        s2 = StandardScaler().fit(Xa[a]); c2 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(s2.transform(Xa[a]), ya[a])
        within.append(roc_auc_score(ya[b], c2.decision_function(s2.transform(Xa[b]))))
    rows.append((L, np.mean(within), conf_auc))
    print(f"layer {L:2d}  within agreement {np.mean(within):.3f}   on conflict {conf_auc:.3f}", flush=True)
with open(f"{D}/adresso_words_check.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["layer","within_agreement_auc","conflict_auc"])
    for r in rows: w.writerow([r[0], f"{r[1]:.3f}", f"{r[2]:.3f}"])
best = max(rows, key=lambda r: r[1])
print(f"SUMMARY: best within-agreement layer {best[0]} at {best[1]:.3f}, that layer on conflict {best[2]:.3f}; "
      f"max conflict over layers {max(r[2] for r in rows):.3f}", flush=True)
