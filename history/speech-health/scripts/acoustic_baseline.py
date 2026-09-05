"""Classical acoustic baseline: eGeMAPSv02 functionals + logistic regression, speaker-grouped 5-fold AUC.
Runs on every dataset whose audio is on this machine."""
import csv, os, glob, numpy as np, opensmile
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
D = os.path.expanduser("~/Desktop/Hard drive data for paper")
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
def feats(paths):
    X = []
    for i, p in enumerate(paths):
        X.append(smile.process_file(p).values[0])
        if (i+1) % 100 == 0: print(f"  {i+1}/{len(paths)}", flush=True)
    return np.array(X)
def cv_auc(X, y, g):
    a = []
    for tr, te in GroupKFold(5).split(X, y, groups=g):
        sc = StandardScaler().fit(X[tr]); c = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5).fit(sc.transform(X[tr]), y[tr])
        a.append(roc_auc_score(y[te], c.decision_function(sc.transform(X[te]))))
    return np.mean(a), np.std(a)
sets = {}
r = [x for x in csv.DictReader(open(f"{D}/DementiaBank/pitt_conflict_manifest.csv")) if x.get("segment_path")]
sets["pitt alzheimers (468)"] = ([f"{D}/DementiaBank/{x['segment_path']}" for x in r], [int(x["label"]) for x in r], [x["grp"]+x["spk"] for x in r])
r = list(csv.DictReader(open(f"{D}/adresso/adresso_manifest.csv")))
sets["adresso alzheimers (237)"] = ([f"{D}/{x['segment_path']}" for x in r], [int(x["label"]) for x in r], [x["spk"] for x in r])
r = list(csv.DictReader(open(f"{D}/adress2020/adress2020_manifest.csv")))
sets["adress2020 pooled (156)"] = ([f"{D}/{x['segment_path']}" for x in r], [int(x["label"]) for x in r], [x["spk"] for x in r])
wavs = {os.path.basename(p): p for p in glob.glob(f"{D}/KCL/extracted/**/*.wav", recursive=True)}
r = [x for x in csv.DictReader(open(f"{D}/KCL/kcl_local_scores.csv")) if x["task"]=="read" and x["file"] in wavs]
sets["kcl parkinsons read (37)"] = ([wavs[x["file"]] for x in r], [int(x["label"]) for x in r], [x["speaker"] for x in r])
out = open(f"{D}/acoustic_baseline.csv", "w", newline=""); w = csv.writer(out); w.writerow(["dataset","n","egemaps_logreg_auc","std"])
for name, (paths, y, g) in sets.items():
    print(f"{name}: extracting", flush=True)
    X = feats(paths); m, s = cv_auc(X, np.array(y), np.array(g))
    w.writerow([name, len(y), f"{m:.3f}", f"{s:.3f}"]); out.flush()
    print(f"ACOUSTIC BASELINE {name}: {m:.3f} +- {s:.3f}", flush=True)
