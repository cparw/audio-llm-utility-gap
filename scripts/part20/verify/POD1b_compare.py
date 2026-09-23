#!/usr/bin/env python3
"""POD1b: compare the verifier's own recomputation (POD1b_verify.py json) with the compute job's sidecar numbers.
Pure comparison, no compute-job code imported. Agreement = equal after rounding to 4 dp (value and both bounds).
usage: POD1b_compare.py VERIFY_JSON SIDECAR_JSON SCORE_KEY"""
import json, sys
V = json.load(open(sys.argv[1])); S = json.load(open(sys.argv[2])); key = sys.argv[3]
cells = {c["id"]: c for c in V["cells"]}
R = S["results"][key]
pairs = [("overall", "overall", "auc"), ("conflict", "conflict_poolarm", "auc"),
         ("agreement", "agreement_poolarm", "auc"), ("conflict_minus_agreement", "paired_conf_minus_agr", "diff")]
ok_all = True
for sk, vk, vf in pairs:
    r = R[sk]; c = cells[vk]
    agree = all(round(float(r[a]), 4) == round(float(c[b]), 4) for a, b in ((vf, "value"), ("lo", "lo"), ("hi", "hi")))
    agree = agree and r["n"] == c["n"] and r["n_spk"] == c["n_spk"]
    ok_all &= agree
    print(f"{key}\t{sk}\tcomputed {r[vf]:.6f} [{r['lo']:.6f}, {r['hi']:.6f}] n={r['n']} spk={r['n_spk']}\t"
          f"verified {c['value']:.6f} [{c['lo']:.6f}, {c['hi']:.6f}] n={c['n']} spk={c['n_spk']} deg={c['degenerate']}\t"
          f"maxdiff {max(abs(r[vf]-c['value']), abs(r['lo']-c['lo']), abs(r['hi']-c['hi'])):.2e}\t{'AGREE' if agree else 'DISAGREE'}")
if key == "p_yes" and "per_fold" in V:
    for f, e in V["per_fold"].items():
        d = S["per_fold_diagnostics"][f]
        agree = (round(d["auc_p_yes_within_fold"], 4) == round(e["auc_within_fold"], 4) and d["n"] == e["n"]
                 and d["n_answer_mass_below_0.5"] == e["n_answer_mass_below_half"]
                 and abs(d["answer_mass_min"] - e["answer_mass_min"]) < 1e-12
                 and abs(d["answer_mass_median"] - e["answer_mass_median"]) < 1e-12)
        ok_all &= agree
        print(f"fold {f}\tcomputed within {d['auc_p_yes_within_fold']:.6f} massmin {d['answer_mass_min']:.3g} below {d['n_answer_mass_below_0.5']}"
              f"\tverified {e['auc_within_fold']:.6f} massmin {e['answer_mass_min']:.3g} below {e['n_answer_mass_below_half']}\t{'AGREE' if agree else 'DISAGREE'}")
    am = S["scoring"]["answer_mass"]; va = V["answer_mass"]
    agree = am["n_below_0.5"] == va["n_below_half"] and abs(am["median"] - va["median"]) < 1e-12 and abs(am["min"] - va["min"]) < 1e-12
    ok_all &= agree
    print(f"answer_mass\tcomputed {am}\tverified {va}\t{'AGREE' if agree else 'DISAGREE'}")
print("ALL_AGREE" if ok_all else "SOME_DISAGREE")
