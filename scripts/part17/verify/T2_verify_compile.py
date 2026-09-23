"""Compile T2 verify JSONs against the main job's T2.tsv rows (value column = max abs diff)."""
import json, csv, os
D = "<local data dir>/release/edaic_rerun/part17"
V = f"{D}/verify"
rows = {r["id"]: r for r in csv.DictReader(open(f"{D}/rows/T2.tsv"), delimiter="\t")}
J = {s: json.load(open(f"{V}/T2_verify_{s}.json")) for s in
     ["mac_single", "pod_single", "pcgita_seeds", "pitt_mac_seeds", "pitt15_seeds", "edaicfull", "omni_pitt_repeats"]
     if os.path.exists(f"{V}/T2_verify_{s}.json")}
out = []


def add(rid, mine, note=""):
    orig = rows[rid]["value"] if rid in rows else "n/a"
    try:
        agree = round(float(orig), 4) == round(float(mine), 4)
    except Exception:
        agree = None
    out.append({"id": rid, "orig": orig, "mine": f"{mine:.3e}", "agree_4dp": agree, "note": note})


if "mac_single" in J:
    for ds, r in J["mac_single"].items():
        m = max(v["maxabs_refit_vs_saved"] for v in r["streams"].values())
        o = min(v["maxabs_other_tie_file"] for v in r["streams"].values())
        lm = all(v["layers_match"] for v in r["streams"].values())
        add(f"T2_{ds}_single_mac_q2a_probe2_{ds}", m, f"other-tie min {o:.3f}; layers re-derived {lm}; sk_eq {r['equals_sklearn_here']}")
        e = r["egemaps"]
        add(f"T2_{ds}_single_mac_egemaps_{ds}", e["maxabs_refit_vs_saved"], f"other-tie {e['maxabs_other_tie_file']:.3f}")
if "pod_single" in J:
    P = J["pod_single"]
    names = {"omni_final_pitt": "T2_pitt_single_pod_omni_final_pitt", "omni_final_edaic": "T2_edaic_single_pod_omni_final_edaic",
             "q2a_kcl_local": "T2_kcl_single_pod_q2a_probe2_kcl_local"}
    for ds in ["adresso", "adress2020", "pcgita", "neurovoz", "kcl"]:
        names[f"q3o_new_{ds}"] = f"T2_{ds}_single_pod_q3o_new_{ds}"
    for k, rid in names.items():
        r = P[k]
        m = max(v["maxabs_refit_vs_saved"] for v in r["streams"].values())
        o = min(v["maxabs_other_tie_file"] for v in r["streams"].values())
        lm = {s: v.get("layers_match") for s, v in r["streams"].items()}
        add(rid, m, f"other-tie min {o:.3f}; stable-rule eq {r['equals_my_stable_rule']}; layers {lm}")
    print("pod-written:", P["pod_written_pitt468"], {k: v for k, v in P["pod_written_sft1c"].items() if k != "structure_sample"})
for sec, rid in [("pcgita_seeds", "T2_pcgita_seed_pod_q3o_POD3_pcgita"), ("pitt15_seeds", "T2_pitt_seed_pod_pod3_q3o_POD3_pitt")]:
    if sec in J:
        m = max(v["maxabs_refit_vs_saved"] for s in J[sec]["seeds"].values() for v in s["streams"].values())
        o = min(v["maxabs_other_tie"] for s in J[sec]["seeds"].values() for v in s["streams"].values())
        lm = {sd: {k: v.get("layers_match") for k, v in s["streams"].items()} for sd, s in J[sec]["seeds"].items()}
        add(rid, m, f"other-tie min {o:.3f}; layers {lm}")
if "pitt_mac_seeds" in J:
    m = max(s["maxabs_refit_vs_saved"] for s in J["pitt_mac_seeds"]["seeds"].values())
    o = min(s["maxabs_other_tie"] for s in J["pitt_mac_seeds"]["seeds"].values())
    add("T2_pitt_seed_mac_omni_part10_pitt_nested5", m, f"other-tie min {o:.3f}")
if "edaicfull" in J:
    m = max(v["maxabs_mean_refit_vs_saved"] for v in J["edaicfull"]["streams"].values())
    o = min(v["maxabs_unshuffled_contrast"] for v in J["edaicfull"]["streams"].values())
    add("T2_edaic_full_shuffle_part16_EDAICFULL", m, f"unshuffled contrast min {o:.3f}; files eq sklearn shuffle "
        f"{all(f['equals_sklearn_shuffle_rs'] for f in J['edaicfull']['files'].values())}")
for r in out:
    print(r)
json.dump(out, open(f"{V}/T2_verify_compiled.json", "w"), indent=1)
