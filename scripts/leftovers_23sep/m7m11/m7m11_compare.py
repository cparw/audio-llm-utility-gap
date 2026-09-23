#!/usr/local/bin/python3
"""Compare the Mac results (m7_meanof5_gaps.csv, m11_group_intervals.csv) with the independent Linux pod verifier
(pull/lo-m7m11-2/out/m7m11_verify.csv) at 4 dp, and write the per-value table m7m11_values.tsv plus its sidecar."""
import json, hashlib, os
import numpy as np, pandas as pd

OUT = "<local data dir>/leftovers_23sep/m7m11"
m7 = pd.read_csv(OUT + "/m7_meanof5_gaps.csv")
m11 = pd.read_csv(OUT + "/m11_group_intervals.csv")
VP = "lo-m7m11-3"   # rerun after the >=20-word difference rule fix; the first run (lo-m7m11-2) agreed 60/60 before that fix
ver = pd.read_csv(OUT + f"/pull/{VP}/out/m7m11_verify.csv").set_index("id")
venv = json.load(open(OUT + f"/pull/{VP}/out/m7m11_verify_env.json"))

def r4(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) or x == "" else f"{float(x):.4f}"

out = []
for src, df in (("m7_meanof5_gaps.csv", m7), ("m11_group_intervals.csv", m11)):
    for r in df.itertuples():
        v = ver.loc[r.id] if r.id in ver.index else None
        mv, ml, mh = r4(r.value_full), r4(r.lo_full), r4(r.hi_full)
        vv, vl, vh = (r4(v["value"]), r4(v["lo"]), r4(v["hi"])) if v is not None else ("", "", "")
        agree = v is not None and mv == vv and ml == vl and mh == vh
        extra = ""
        if src.startswith("m11") and v is not None and isinstance(v["extra"], str) and v["extra"].startswith("all37"):
            a37 = f"all37 [{r4(r.lo_all37)}, {r4(r.hi_all37)}]"
            extra = f"mac {a37} | pod {v['extra']} | all37 agree {a37 == v['extra']}"
        out.append(dict(id=r.id, what=r.what, value=mv, lo=ml, hi=mh, n=r.n, n_spk=r.n_spk,
                        kind=getattr(r, "kind", getattr(r, "rule", "")), file=OUT + "/" + src,
                        source=r.source, verifier_value=vv, verifier_lo=vl, verifier_hi=vh,
                        agree_4dp=agree, verifier_extra=extra, closes=r.closes, note=getattr(r, "note", "") if isinstance(getattr(r, "note", ""), str) else ""))
o = pd.DataFrame(out)
o.to_csv(OUT + "/m7m11_values.tsv", sep="\t", index=False)
n_ag = int(o.agree_4dp.sum())
all37_ok = int(o.verifier_extra.str.contains("all37 agree True").sum()); all37_n = int(o.verifier_extra.str.contains("all37 agree").sum())
print(f"rows {len(o)}; agree at 4 dp {n_ag}/{len(o)}; M11 all-37 scheme values agree {all37_ok}/{all37_n}")
for r in o[~o.agree_4dp].itertuples():
    print("DISAGREE", r.id, r.value, r.lo, r.hi, "| verifier", r.verifier_value, r.verifier_lo, r.verifier_hi)
m7s = json.load(open(OUT + "/m7_meanof5_gaps.sidecar.json")); m11s = json.load(open(OUT + "/m11_group_intervals.sidecar.json"))
side = dict(table=OUT + "/m7m11_values.tsv", rows=len(o), agree_4dp=n_ag,
            m7_summary=dict(
                point="exact: Table 1 mean of five per-repeat AUCs (omni_final/omni_<ds>_nested_repeats.json) minus zero-shot AUC recomputed from the per-clip file",
                interval="APPROXIMATE: single-split paired percentile interval shifted by (mean of five minus single-split AUC); the exact per-draw mean-of-five interval needs per-repeat per-clip OOF, which was never saved, and the Table 1 states are gone",
                why_not_exact=m7s["why_not_exact"],
                validation_max_endpoint_error={k: v["max_endpoint_error"] for k, v in m7s["validation"].items()},
                validation_note="shifted single split vs exact mean of five on the same states and the Table 1 outer splits: NeuroVoz 0.0019, ADReSS-2020 0.0053; shifted single repeat vs exact: Pitt 0.0084, E-DAIC 300 s 0.0167",
                pcgita="0.3306 (tex prints 0.35; the 0.3454 in part16 M7 used the single split 0.9089)"),
            m11_summary=dict(cause=m11s["cause"], which_was_wrong=m11s["which_was_wrong"], rule=m11s["rule"], new_rows=m11s["new_rows"]),
            m11_all37_scheme_agree=f"{all37_ok}/{all37_n}",
            verifier=dict(script=OUT + "/scripts/m7m11_verify.py", pod="lo-m7m11-3 (weqnlwasxdgpgq), RunPod CPU, 32 vCPU; first run lo-m7m11-2 (z0ikr6rmvcs5tk) agreed 60/60 before the >=20-word difference rule fix", env=venv,
                          method="independent code: sklearn roc_auc_score, pandas groupby indices, Python sorted speaker list, np.quantile"),
            main=dict(m7=OUT + "/scripts/m7_meanof5_gaps.py", m11=OUT + "/scripts/m11_group_intervals.py", env="Mac, /usr/local/bin/python3, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3"),
            refit=dict(script=OUT + "/scripts/pod1_adress2020_rep5_refit.py", pod="lo-m7m11-1 (was0szcwexnom3), RunPod CPU, 32 vCPU, sklearn 1.9.1 numpy 2.1.2 scipy 1.18.1 Python 3.12.3",
                       output=OUT + "/pull/lo-m7m11-1/out/adress2020_pod4b_enc_rep5_oof.csv"),
            sha256={p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (OUT + "/m7_meanof5_gaps.csv", OUT + "/m11_group_intervals.csv",
                                                                                   OUT + f"/pull/{VP}/out/m7m11_verify.csv", OUT + "/m7m11_values.tsv")})
json.dump(side, open(OUT + "/m7m11_values.sidecar.json", "w"), indent=1)
