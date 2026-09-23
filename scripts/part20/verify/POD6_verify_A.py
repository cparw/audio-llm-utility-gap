#!/usr/bin/env python3
"""POD6 verifier, part A: TF-IDF language saliency, refitted end to end with own code.
usage: POD6_verify_A.py DATA_DIR OUT_DIR [dataset ...]
DATA_DIR holds the transcript files and the folds/ directory (copies of the release files).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from POD6_verify import tfidf_oof, cell

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
b = os.path.basename

CFG = {
    "pcgita":      ("asr_pcgita.csv",       "clip_path", "text", lambda v: b(v)),
    "neurovoz":    ("asr_neurovoz.csv",     "clip_path", "text", lambda v: b(v)),
    "kcl":         ("asr_kcl.csv",          "clip_path", "text", lambda v: b(v)),
    "edaic30":     ("edaic30_text.csv",     "id",        "text", lambda v: f"{v}.wav"),
    "edaicfull":   ("edaicfull_text.csv",   "id",        "text", lambda v: f"{v}.wav"),
    "pitt":        ("pitt_text.csv",        "id",        "text", lambda v: b(v)),
    "pitt_rawcha": ("pitt_conflict_manifest.csv", "segment_path", "text", lambda v: b(v)),
    "adresso":     ("adresso_text.csv",     "id",        "text", lambda v: f"{v}.wav"),
    "adress2020":  ("adress2020_text.csv",  "id",        "text", lambda v: f"{v}.wav"),
}
FOLD = {"pcgita": "pcgita", "neurovoz": "neurovoz", "kcl": "kcl", "edaic30": "edaic", "edaicfull": "edaic",
        "pitt": "pitt", "pitt_rawcha": "pitt", "adresso": "adresso", "adress2020": "adress2020"}

todo = sys.argv[3:] or list(CFG)
summary = {}
for ds in todo:
    fn, idc, tc, key = CFG[ds]
    t = pd.read_csv(os.path.join(D, fn))
    fo = pd.read_csv(os.path.join(D, "folds", f"{FOLD[ds]}_groupkfold5_mac.csv"), dtype={"speaker_id": str})
    t["clip_id"] = t[idc].map(key)
    assert t["clip_id"].is_unique
    t = t.rename(columns={"label": "label_t"})[["clip_id", tc, "label_t"]]
    m = fo.merge(t, on="clip_id", how="left", validate="1:1")
    miss = int(m[tc].isna().sum())
    y = m["label_t"].astype(int).to_numpy()
    spk = m["speaker_id"].astype(str).to_numpy()
    fold = m["fold"].to_numpy()
    oof = tfidf_oof(m[tc].to_numpy(), y, fold)
    r = cell(spk, y, oof)
    r["missing_text"] = miss; r["transcripts"] = fn; r["folds"] = f"{FOLD[ds]}_groupkfold5_mac.csv"
    if ds == "neurovoz":   # speaker ids as integers change np.unique order
        from POD6_verify import boot_ci
        lo, hi, nu = boot_ci(m["speaker_id"].astype(int).to_numpy(), y, oof)
        r["lo_intspk"], r["hi_intspk"] = lo, hi
    summary[ds] = r
    pd.DataFrame({"clip_id": m["clip_id"], "speaker_id": spk, "label": y, "fold": fold, "p_tfidf": oof}).to_csv(
        os.path.join(OUT, f"A_{ds}_oof_verify.csv"), index=False)
    print(f"{ds:12s} AUC {r['auc']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] n={r['n']} spk={r['n_spk']} "
          f"pos={r['n_pos']} neg={r['n_neg']} missing_text={miss}", flush=True)
json.dump(summary, open(os.path.join(OUT, "A_summary_verify.json"), "w"), indent=1)
