"""Conflict/agreement AUCs with PAIRED 2000-draw speaker bootstrap from a scored per-clip csv.
usage: analyze.py SCORED.csv ROWID LABEL"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_auc import auc_rank, boot_paired
F, RID, LAB = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = "<local data dir>/Desktop/release/edaic_rerun/part16/POD1"
ROWS = "reports/part16/rows/POD1.tsv"
d = pd.read_csv(F)
dst = f"{OUT}/{os.path.basename(F)}"
print(f"{LAB}\n  file {dst}\n  n={len(d)} speakers={d.speaker.nunique()} arms={d.arm.value_counts().to_dict()}")
print(f"  median answer_mass = {d.answer_mass.median():.4f}   pooled AUC = {auc_rank(d.label,d.p_yes):.4f}")
res = {}
for arm in ("conflict", "agreement"):
    s = d[d.arm == arm]
    res[arm] = dict(auc=auc_rank(s.label, s.p_yes), n=len(s), n_spk=int(s.speaker.nunique()),
                    n_pos=int(s.label.sum()))
ci = boot_paired(d.label.values, d.p_yes.values, d.speaker.values, d.arm.values, 2000, 0)
out = []
for arm in ("conflict", "agreement"):
    lo, hi = ci[arm]; res[arm]["lo"], res[arm]["hi"] = lo, hi
    print(f"  {arm.upper():10s} AUC {res[arm]['auc']:.4f} [{lo:.4f},{hi:.4f}]  n={res[arm]['n']} spk={res[arm]['n_spk']} pos={res[arm]['n_pos']}")
    out.append((f"{RID}_{arm}_auc", f"{LAB} {arm} arm AUC", res[arm]['auc'], lo, hi,
                res[arm]['n'], res[arm]['n_spk'], dst))
print(f"  usable paired draws {ci['usable']}/2000   GAP agreement-conflict = {res['agreement']['auc']-res['conflict']['auc']:.4f}")
res["usable_draws"] = ci["usable"]; res["n"] = len(d); res["n_speakers"] = int(d.speaker.nunique())
res["n_pairs"] = res["conflict"]["n"]; res["median_answer_mass"] = round(float(d.answer_mass.median()), 6)
res["pooled_auc"] = round(auc_rank(d.label, d.p_yes), 6)
res["prompt"] = d.prompt.iloc[0]
json.dump(res, open(f"{OUT}/{os.path.basename(F).replace('.csv','_analysis.json')}", "w"), indent=1, default=float)
with open(ROWS, "a") as fh:
    for r in out:
        fh.write("\t".join([r[0], r[1]] + [f"{v:.4f}" for v in r[2:5]] + [str(r[5]), str(r[6]), r[7]]) + "\n")
print(f"  rows appended 2\n")
