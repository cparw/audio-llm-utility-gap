import json, csv, os, datetime
SC = "<local data dir>/scratch/vp16"
V = "<local data dir>/leftovers_23sep/verify"
WK = V + "/p16twins_work"
mac = json.load(open(SC + "/mac_results.json")); m5 = json.load(open(SC + "/m5_mac_results.json"))
d4 = json.load(open(SC + "/pull/lo-vp16-4/out/v_dir.json")); d5 = json.load(open(SC + "/pull/lo-vp16-5/out/v_dir.json"))
st = {k: json.load(open(f"{SC}/pull/lo-vp16-1/out/v_{k}_default.json")) for k in
      ("pitt_enc", "pitt_llm", "pitt_ans", "pcgita_enc", "pcgita_llm", "pcgita_ans")}
stH = {k: json.load(open(f"{SC}/pull/lo-vp16-1/out/v_{k}_Haswell.json")) for k in st}
stS = {k: json.load(open(f"{SC}/pull/lo-vp16-1/out/v_{k}_SkylakeX.json")) for k in st}
claims = [  # id, value, lo, hi
("3b_all_cos","0.0051",None,None),("3b_conflict_cos","-0.0067",None,None),("3b_agreement_cos","0.0086",None,None),("SEED14_3b","0.0051",None,None),
("3b_conflict_auc_probe","0.6145","0.5021","0.7246"),("3b_agreement_auc_probe","0.9349","0.9034","0.9614"),
("3b_conflict_auc_wo_d","0.6196","0.5094","0.7283"),("3b_agreement_auc_wo_d","0.9341","0.9021","0.9604"),
("3a_pitt_enc_best_stage","0.8276","0.7765","0.8760"),("3a_pitt_llm_best_stage","0.8634","0.8179","0.9062"),("3a_pitt_ans_best_stage","0.8485","0.8069","0.8882"),
("3a_pcgita_enc_best_stage","0.9133","0.8519","0.9604"),("3a_pcgita_llm_best_stage","0.8960","0.8341","0.9463"),("3a_pcgita_ans_best_stage","0.8962","0.8327","0.9452"),
("M3_pitt_speaker","0.6771","0.6096","0.7494"),("M3_pitt_clip_minus_speaker","-0.0193","-0.0565","0.0172"),
("M3_pcgita_speaker","0.5948","0.4778","0.7089"),("M3_pcgita_clip_minus_speaker","-0.0313","-0.0820","0.0179"),
("M3_neurovoz_speaker","0.7948","0.7045","0.8733"),("M3_neurovoz_clip_minus_speaker","-0.0997","-0.1325","-0.0613"),
("M7fix#01 Pitt encoder probe mean of 5","0.7706","0.7157","0.8220"),("M7fix#02 Pitt paired probe minus zero-shot","+0.1128","0.0455","0.1768"),
("M7_pitt_probe_rep5","0.7917","0.7324","0.8480"),("M7_pitt_gap_rep5","0.1339","0.0641","0.2019"),
("POD4_4b_resolved","468",None,None),("POD4_4b_invwin","158",None,None),("POD4_4b_invshare","0.0200",None,None),("POD4_4b_parshare","0.5871",None,None),
("POD4_4b_pardur","17.61",None,None),("POD4_4b_under10s","68",None,None),("M9.FILES","307",None,None),("M9.SWEEPMIN","0.8662",None,None),
("M5_all_auc_without_d","0.8327","0.7857","0.8759"),("M5_all_auc_without_d_Xproj","0.8223","0.7743","0.8672"),
("M5_conflict_refit_auc_without_d","0.6869","0.5806","0.7872"),("M5_agreement_refit_auc_without_d","0.9346","0.9057","0.9603"),
("M5_conflict_restricted_auc_without_d","0.6326","0.5277","0.7336"),("M5_conflict_restricted_auc_without_d_Xproj","0.6180","0.5128","0.7213"),
("M5_agreement_restricted_auc_without_d","0.9070","0.8702","0.9383"),("M5_agreement_restricted_auc_without_d_Xproj","0.8988","0.8593","0.9326"),
("M5_paired_auc_without_d_conflict_minus_agreement","-0.2745","-0.3754","-0.1707"),
("M5_all_randfloor","0.0124","0.0005","0.0343"),("M5_conflict_refit_randfloor","0.0123","0.0005","0.0350"),("M5_agreement_refit_randfloor","0.0126","0.0005","0.0359"),
("MEDAIC_full_llm_nested_NOCI","0.6853",None,None),("MEDAIC_full_llm_minus_answer_POINT","-0.1432",None,None),
("MEDAIC_full_enc_nested_NOCI","0.5930",None,None),("MEDAIC_full_enc_minus_answer_POINT","-0.2355",None,None),
("MEDAIC_full_ans_nested_NOCI","0.7542",None,None),("MEDAIC_full_ans_minus_answer_POINT","-0.0743",None,None),
("MEDAIC_full_proj_nested_NOCI","0.5293",None,None),("MEDAIC_full_proj_minus_answer_POINT","-0.2992",None,None),
("P16.POD2.L5","4.13;4.07;4.05;4.03;4.00",None,None),("P16.POD2.L6","21.013",None,None)]

def rec(cid):
    """-> (value, [lo, hi] or None, source, how)"""
    if cid.startswith("3b_") or cid == "SEED14_3b":
        arm = "all" if cid in ("3b_all_cos", "SEED14_3b") else cid.split("_")[1]
        c = d4[arm]
        if cid.endswith("_cos") or cid == "SEED14_3b":
            return c["cosine"], None, "pod4", f"Linux refit, arm {arm}; pod5 rerun {d5[arm]['cosine']:.6f}"
        if "auc_probe" in cid: return c["auc_probe"], c["auc_probe_ci"], "pod4", "Linux refit inside the arm"
        return c["auc_wo_d_trainz"], c["auc_wo_d_trainz_ci"], "pod4", "Linux refit, held-out projection standardised with the training fold"
    if cid.startswith("3a_"):
        k = cid[3:].replace("_best_stage", ""); s = st[k]
        return s["best_mean"], s["best_ci"], "pod1", (f"best stage {s['best_stage']} of {s['stages']}; per repeat {[round(a,4) for a in s['best_per_repeat']]}; "
                f"AVX2 Haswell kernel gives {stH[k]['best_mean']:.4f} [{stH[k]['best_ci'][0]:.4f}, {stH[k]['best_ci'][1]:.4f}] stage {stH[k]['best_stage']}")
    if cid.endswith("randfloor"):
        k = {"M5_all_randfloor": "all", "M5_conflict_refit_randfloor": "conflict_refit", "M5_agreement_refit_randfloor": "agreement_refit"}[cid]
        return m5[k]["randfloor"], m5[k]["randfloor_ci"], "mac", "Mac refit, sklearn 1.7.2"
    r = mac[cid]
    return r["value"], r.get("ci"), "mac", ""

def dec(s): return len(s.split(".")[1]) if "." in s else 0
def fmt(v, nd):
    if isinstance(v, str): return v
    return f"{v:.{nd}f}" if nd else str(int(round(v)))
rows = []; nyes = 0
for cid, val, lo, hi in claims:
    v, ci, src, how = rec(cid)
    nd = dec(val)
    rv = fmt(v, nd)
    okv = (rv == val) if isinstance(v, str) else (round(float(rv), nd) == round(float(val), nd))
    claimed = val + (f" [{lo}, {hi}]" if lo is not None else "")
    recs = (("+" if val.startswith("+") and not rv.startswith("-") else "") + rv)
    okc = True
    if lo is not None:
        rl, rh = fmt(ci[0], dec(lo)), fmt(ci[1], dec(hi))
        okc = round(float(rl), 4) == round(float(lo), 4) and round(float(rh), 4) == round(float(hi), 4)
        recs += f" [{rl}, {rh}]"
    ok = okv and okc; nyes += ok
    rows.append(dict(id=cid, claimed=claimed, recomputed=recs, verified="yes" if ok else "no", src=src, how=how, full=v, full_ci=ci))
os.makedirs(V, exist_ok=True)
tsv = V + "/p16twins_VERIFIED.tsv"
with open(SC + "/p16twins_VERIFIED.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n"); w.writerow(["id", "claimed", "recomputed", "verified"])
    for r in rows: w.writerow([r["id"], r["claimed"], r["recomputed"], r["verified"]])
side = {
 "task": "independent verifier for p16twins", "date_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
 "tsv": tsv, "n_values": len(rows), "n_verified_yes": nyes,
 "rule": "verified = the recomputed value, and both CI bounds when the claim has them, print the same as the claim at the claim's precision (4 dp; 2 dp for pardur; 3 dp for L6; integers for counts)",
 "independence": "all code written for this check (scratch vp16_mac.py, vp16_m5.py, pod/v_stage.py, pod/v_dir.py, pod/v_lmhead.py; copies in p16twins_work/scripts). No task script was imported or copied. AUC by Mann-Whitney counting (searchsorted or pairwise), not scipy rankdata or sklearn roc_auc_score.",
 "bootstrap": "2000 draws, fresh numpy default_rng(0) per cell, idx = rng.choice(N, N, replace=True) over sorted unique speaker ids, 2.5/97.5 percentiles, paired = same idx; 3a CI = mean over the 5 repeats inside each draw",
 "environments": {
   "pod1 lo-vp16-1": "AMD EPYC 9654, 32 vCPU, Python 3.12.3, sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1; OpenBLAS default kernel SkylakeX (AVX512). All 3a streams, all stages x 5 repeats, run three times: default, forced SkylakeX, forced Haswell (AVX2)",
   "pod4 lo-vp16-4 and pod5 lo-vp16-5": "AMD EPYC 7702P (AVX2), same Python stack; 3b direction refit, identical results on both",
   "lm_head": d4.get("env", {}) and json.load(open(SC + "/pull/lo-vp16-4/out/v_lmhead_source.json")),
   "mac": "/usr/local/bin/python3 3.10.2, sklearn 1.7.2, numpy 2.2.6 (per-clip recomputes, M5 refit)"},
 "inputs": {"q3o_pitt_states": "<local data dir>/paper1_local_runs/part16_POD3_states/q3o_pitt_states.npz (uploaded, sha256 adc889e0... matched on the pod)",
            "q3o_pcgita_states": "<local data dir>/paper1_local_runs/part16_POD3_states/q3o_pcgita_states.npz (uploaded, sha256 172ab28d... matched)",
            "arms": "<local data dir>/DementiaBank/pitt_conflict_manifest.csv set column (146 conflict, 322 agreement)",
            "q2a_states_M5": "<local data dir>/paper1_local_runs/probe2/pitt_states.npz; Qwen2-Audio lm_head from the cached HF shard model-00005-of-00005.safetensors",
            "perclip": ["<local data dir>/release/omni_final/omni_{pitt,pcgita,neurovoz}_zeroshot_scores.csv",
                        "<local data dir>/release/overnight2/part10/pitt_enc_nested5_oof.npz",
                        "<local data dir>/release/edaic_rerun/part16/M5_readout_direction_q2a.csv",
                        "<local data dir>/release/edaic_rerun/part16/POD4/cha_overlap_468.csv and cut_report.json",
                        "<local data dir>/release/edaic_rerun/part16/M9_inventory.csv (608 files reopened)",
                        "<local data dir>/release/edaic_rerun/logs/podA/o25_full_rep.log and variants/o25_full_zeroshot_scores.csv",
                        "<local data dir>/release/edaic_rerun/part16/POD2/lora.log"]},
 "extra_checks": {
   "mac_sklearn_1.7.2_3b_probe": "the same probe on the Mac gives all 0.7864, conflict 0.5520, agreement 0.9314 with equal fold sizes, so the old Mac numbers are a fold-assignment artefact of sklearn 1.7.2",
   "M5_mac_refit": "the Mac refit reproduces the M5 per-clip probe and without-d columns exactly (max abs diff 0.0) for all, conflict and agreement",
   "kernel": {k: {"avx512": [round(st[k]["best_mean"], 6), st[k]["best_stage"]], "forced_skylakex": [round(stS[k]["best_mean"], 6), stS[k]["best_stage"]],
                  "forced_haswell_avx2": [round(stH[k]["best_mean"], 6), stH[k]["best_stage"], [round(x, 4) for x in stH[k]["best_ci"]]]} for k in st}},
 "rows": rows,
}
json.dump(side, open(SC + "/p16twins_VERIFIED.sidecar.json", "w"), indent=1, default=str)
print(nyes, "of", len(rows))
for r in rows: print(r["id"], "|", r["claimed"], "|", r["recomputed"], "|", r["verified"])
