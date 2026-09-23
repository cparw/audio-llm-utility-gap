"""PART20 POD6 job A sensitivity: Pitt 468 TF-IDF probe on the raw CHAT 'text' column of pitt_conflict_manifest.csv
(the paper's transcript condition used the cleaned pitt_text.csv; this checks the choice does not matter).
Same model, same pitt_groupkfold5_mac folds, same bootstrap. usage: a_pitt_rawtext.py OUTDIR"""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p20_common import auc_rank, boot_auc, paste, sidecar
OUT = sys.argv[1]; PCM = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"; FF = "folds/pitt_groupkfold5_mac.csv"
pc = pd.read_csv(PCM); pc["clip"] = pc["segment_path"].map(os.path.basename)
fo = pd.read_csv(FF); m = fo.merge(pc[["clip", "label", "text", "set"]], left_on="clip_id", right_on="clip", how="inner"); assert len(m) == 468
y = m.label.values.astype(int); X = m.text.fillna("").astype(str).values; oof = np.zeros(len(m))
for k in range(5):
    tr = m.fold.values != k; te = ~tr; v = TfidfVectorizer(); A = v.fit_transform(X[tr])
    oof[te] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(A, y[tr]).predict_proba(v.transform(X[te]))[:, 1]
spk = m.speaker_id.astype(str).values; a = auc_rank(y, oof); lo, hi, nu = boot_auc(y, oof, spk)
per = f"{OUT}/A_tfidf_pitt_rawCHAT_oof.csv"
pd.DataFrame({"clip": m["clip"], "speaker": spk, "label": y, "fold": m.fold, "arm": m["set"], "p_tfidf": oof}).to_csv(per, index=False)
sidecar(per.replace(".csv", ".sidecar.json"), result="Pitt 468 TF-IDF LR on raw CHAT manifest text (sensitivity)", auc=round(a, 4), lo=round(lo, 4), hi=round(hi, 4),
        usable_draws=nu, n=468, n_speakers=int(len(set(spk))), seed="default_rng(0), 2000 draws", model_id="none", prompt="none",
        folds=FF, command="/usr/local/bin/python3 " + " ".join(sys.argv), sources=[PCM, FF])
open(per.replace("_oof.csv", ".RESULT"), "w").close()
paste("P20.POD6.A.tfidf.pitt_rawCHAT", "language saliency TF-IDF LR AUC Pitt 468, raw CHAT manifest text (sensitivity)", a, lo, hi, 468, len(set(spk)), per)
