"""PART20 POD6 job A: language saliency, measured.
Per dataset: TfidfVectorizer (words, sklearn defaults) fit INSIDE each training fold, then
LogisticRegression(C=1, class_weight='balanced', max_iter=5000); out-of-fold AUC (rank formula) on the saved
single-split _mac speaker folds (the split that reproduces the eGeMAPS baseline); 2000-draw speaker bootstrap.
usage: a_lang_saliency.py OUTDIR"""
import sys, os, json, time, platform, numpy as np, pandas as pd, sklearn, scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p20_common import auc_rank, boot_auc, paste, sidecar
OUT = sys.argv[1]
F = "folds"
L = "<local data dir>/paper work/paper1_local_runs"
ASR = "<local data dir>/release/part4_text/asr"
PCM = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
O25 = "<local data dir>/release/overnight2/text_new"
CMD = "/usr/local/bin/python3 " + " ".join(sys.argv)

def load(ds):
    if ds in ("pcgita", "neurovoz", "kcl"):
        src = f"{ASR}/asr_{ds}.csv"; d = pd.read_csv(src)
        d = pd.DataFrame({"clip": d.clip_path.map(os.path.basename), "label": d.label, "speaker": d.speaker_id.astype(str), "text": d.text})
    else:
        name = {"edaic30": "edaic30", "edaicfull": "edaicfull", "pitt": "pitt", "adresso": "adresso", "adress2020": "adress2020"}[ds]
        src = f"{L}/{name}_text.csv"; d = pd.read_csv(src)
        clip = d["id"].astype(str).map(os.path.basename)
        if ds != "pitt": clip = clip + ".wav"
        d = pd.DataFrame({"clip": clip, "label": d.label, "speaker": d.speaker.astype(str), "text": d.text})
    return d, src

FOLD = {"pcgita": "pcgita", "neurovoz": "neurovoz", "kcl": "kcl", "edaic30": "edaic", "edaicfull": "edaic",
        "pitt": "pitt", "adresso": "adresso", "adress2020": "adress2020"}
NAME = {"pcgita": "PC-GITA", "neurovoz": "NeuroVoz", "kcl": "MDVR-KCL", "edaic30": "E-DAIC first 30 s",
        "edaicfull": "E-DAIC full interview", "pitt": "Pitt 468", "adresso": "ADReSSo", "adress2020": "ADReSS-2020"}
O25F = {"pcgita": "pcgita", "neurovoz": "neurovoz", "kcl": "kcl", "edaic30": "edaic30", "edaicfull": "edaicfull",
        "pitt": "pitt", "adresso": "adresso", "adress2020": "adress2020"}
res = {}
for ds in ["pcgita", "neurovoz", "kcl", "edaic30", "edaicfull", "pitt", "adresso", "adress2020"]:
    t0 = time.time()
    d, src = load(ds)
    ff = f"{F}/{FOLD[ds]}_groupkfold5_mac.csv"; fo = pd.read_csv(ff)
    fo["speaker_id"] = fo.speaker_id.astype(str)
    m = d.merge(fo, left_on="clip", right_on="clip_id", how="inner")
    assert len(m) == len(d) == len(fo), (ds, len(m), len(d), len(fo))
    assert (m.speaker == m.speaker_id).all(), ds
    # cross-check clip set and labels against the paper's transcript-condition output (Qwen2.5-Omni)
    o = pd.read_csv(f"{O25}/o25_{O25F[ds]}_text.csv")
    okey = o["id"].astype(str).map(os.path.basename)
    if ds in ("edaicfull",): okey = o.speaker_id.astype(str) + ".wav"
    elif ds not in ("pitt", "pcgita", "neurovoz", "kcl"): okey = okey + ".wav"
    olab = dict(zip(okey, o.label)); lab_ok = int(sum(olab.get(c) == l for c, l in zip(m["clip"], m.label)))
    if ds == "pitt":
        pc = pd.read_csv(PCM); pc["b"] = pc["segment_path"].map(os.path.basename)
        tx = dict(zip(pc.b, pc.text)); same_text = int(sum(str(tx.get(c)) == str(t) for c, t in zip(m["clip"], m.text)))
    else:
        same_text = None
    y = m.label.values.astype(int); X = m.text.fillna("").astype(str).values; oof = np.zeros(len(m)); vocab = []
    for k in range(5):
        tr = m.fold.values != k; te = ~tr
        vec = TfidfVectorizer()  # words: default token_pattern, lowercase
        Xtr = vec.fit_transform(X[tr]); Xte = vec.transform(X[te]); vocab.append(len(vec.vocabulary_))
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    a = auc_rank(y, oof); lo, hi, nu = boot_auc(y, oof, m.speaker.values)
    nspk = m.speaker.nunique()
    pc_out = f"{OUT}/A_tfidf_{ds}_oof.csv"
    pd.DataFrame({"clip": m["clip"], "speaker": m.speaker, "label": y, "fold": m.fold, "p_tfidf": oof}).to_csv(pc_out, index=False)
    sidecar(pc_out.replace(".csv", ".sidecar.json"), result=f"TF-IDF word logistic regression, out of fold, {NAME[ds]}",
            auc=round(a, 4), lo=round(lo, 4), hi=round(hi, 4), usable_draws=nu, n=int(len(m)), n_speakers=int(nspk),
            n_positive=int(y.sum()), seed="bootstrap numpy.random.default_rng(0), 2000 draws, speakers with replacement",
            model="TfidfVectorizer() fit inside each training fold + LogisticRegression(C=1, class_weight='balanced', max_iter=5000)",
            model_id="none (no neural model)", prompt="none (no prompt; transcript text only)",
            folds=ff, fold_note="single-split _mac file: sklearn GroupKFold(5) as installed on the Mac; the split that reproduces the eGeMAPS baseline",
            transcript_source=src, vocab_size_per_fold=vocab,
            labels_match_paper_transcript_run=f"{lab_ok}/{len(m)} vs {O25}/o25_{O25F[ds]}_text.csv",
            pitt_text_equals_conflict_manifest=(f"{same_text}/{len(m)}" if same_text is not None else None),
            command=CMD, versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__),
            sources=[src, ff, f"{O25}/o25_{O25F[ds]}_text.csv"] + ([PCM] if ds == "pitt" else []))
    open(pc_out.replace("_oof.csv", ".RESULT"), "w").close()
    paste(f"P20.POD6.A.tfidf.{ds}", f"language saliency TF-IDF LR AUC {NAME[ds]}", a, lo, hi, len(m), nspk, pc_out)
    res[ds] = dict(auc=a, lo=lo, hi=hi, n=int(len(m)), n_spk=int(nspk), labels_ok=lab_ok, pitt_text_same=same_text,
                   secs=round(time.time() - t0, 1))
json.dump(res, open(f"{OUT}/A_summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))
