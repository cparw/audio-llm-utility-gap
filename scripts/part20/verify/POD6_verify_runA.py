#!/usr/bin/env python3
"""POD6 verifier, part A check: recompute each compute-side TF-IDF value from its
per-clip csv, check the fold file and n, and compare with the own end-to-end refit.
usage: POD6_verify_runA.py COMPUTE_DIR OWN_REFIT_DIR FOLDS_DIR ROWS_TSV [ds ...]
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from POD6_verify import auc_mw, auc_trap, boot_ci

CD, OWN, FD, TSV = sys.argv[1:5]
FOLD = {"pcgita": "pcgita", "neurovoz": "neurovoz", "kcl": "kcl", "edaic30": "edaic", "edaicfull": "edaic",
        "pitt": "pitt", "adresso": "adresso", "adress2020": "adress2020"}
NAME = {"pcgita": "PC-GITA", "neurovoz": "NeuroVoz", "kcl": "MDVR-KCL", "edaic30": "E-DAIC first 30 s",
        "edaicfull": "E-DAIC full interview", "pitt": "Pitt 468", "adresso": "ADReSSo", "adress2020": "ADReSS-2020"}
todo = sys.argv[5:] or list(FOLD)
out = []
for ds in todo:
    f = os.path.join(CD, f"A_tfidf_{ds}_oof.csv")
    sc = json.load(open(f.replace(".csv", ".sidecar.json")))
    c = pd.read_csv(f, dtype={"speaker": str})
    fo = pd.read_csv(os.path.join(FD, f"{FOLD[ds]}_groupkfold5_mac.csv"), dtype={"speaker_id": str})
    m = c.merge(fo, left_on="clip", right_on="clip_id", how="inner", validate="1:1")
    checks = dict(n_rows=len(c), n_fold_file=len(fo), n_joined=len(m),
                  fold_same=bool((m["fold_x"] == m["fold_y"]).all()),
                  spk_same=bool((m["speaker"] == m["speaker_id"]).all()),
                  fold_file_named=os.path.basename(sc.get("folds", "")))
    y = c["label"].astype(int).to_numpy(); s = c["p_tfidf"].to_numpy(float); spk = c["speaker"].to_numpy()
    a = auc_mw(y, s); t = auc_trap(y, s)
    lo, hi, nu = boot_ci(spk, y, s)
    own = pd.read_csv(os.path.join(OWN, f"A_{ds}_oof_verify.csv"))
    mo = c.merge(own, left_on="clip", right_on="clip_id")
    dmax = float(np.abs(mo["p_tfidf_x"] - mo["p_tfidf_y"]).max())
    own_auc = auc_mw(mo["label_x"].to_numpy(), mo["p_tfidf_y"].to_numpy())
    comp = (sc["auc"], sc["lo"], sc["hi"])
    ver = (a, lo, hi)
    agree = all(round(float(u), 4) == round(float(v), 4) for u, v in zip(comp, ver)) and abs(a - t) < 1e-9 \
        and checks["fold_same"] and checks["spk_same"] and checks["n_joined"] == checks["n_fold_file"] == len(c) \
        and sc["n"] == len(c) and sc["n_speakers"] == len(np.unique(spk)) and round(own_auc, 4) == round(a, 4)
    vs = "above" if lo > 0.5 else ("below" if hi < 0.5 else "spans")
    print(f"{ds:10s} comp {comp[0]:.4f} [{comp[1]:.4f},{comp[2]:.4f}]  ver {a:.4f} [{lo:.4f},{hi:.4f}] trap {t:.6f} "
          f"own-refit {own_auc:.4f} maxdiff {dmax:.2e} n={len(c)} spk={len(np.unique(spk))} pos={int(y.sum())} "
          f"checks={checks} src={os.path.basename(sc.get('transcript_source',''))} AGREE={agree}", flush=True)
    out.append(dict(id=f"A_tfidf_{ds}", what=f"TF-IDF transcript LR, out-of-fold AUC, {NAME[ds]} ({os.path.basename(sc.get('transcript_source',''))}, _mac folds)",
                    value_computed=f"{comp[0]:.4f}", value_verified=f"{a:.4f}", lo=f"{lo:.4f}", hi=f"{hi:.4f}",
                    n=len(c), n_spk=int(len(np.unique(spk))), file=f"scores/part20/POD6/A_language_saliency/A_tfidf_{ds}_oof.csv",
                    two_dp=f"{a:.2f} [{lo:.2f}, {hi:.2f}]", vs_half=vs, agree_4dp="yes" if agree else "NO",
                    _comp_lo=f"{comp[1]:.4f}", _comp_hi=f"{comp[2]:.4f}", _own=own_auc, _maxdiff=dmax))
json.dump(out, open(os.path.join(os.path.dirname(TSV), "..", "verify", "POD6_A_check.json"), "w"), indent=1, default=str)
