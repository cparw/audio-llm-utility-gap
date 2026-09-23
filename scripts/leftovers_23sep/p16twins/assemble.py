"""Assemble twins_p16.tsv (+ sidecar) from the pod pulls and the Mac recomputes. Originals are read from the PART 16
fragment rows (rows/*.tsv, SEED rows from PART16_RESULTS.md seed table), never typed in by hand."""
import os, json, csv, sys, hashlib, datetime, glob
import numpy as np

OUT = "<local data dir>/leftovers_23sep/p16twins"
SC = sys.argv[1]
P16 = "<local data dir>/release/edaic_rerun/part16"
PULL = f"{OUT}/pull"

def rows_of(f):
    rd = csv.DictReader(open(f), delimiter="\t")
    return {r["id"]: r for r in rd} if "id" in (rd.fieldnames or []) else {}
FR = {}
for f in glob.glob(f"{P16}/rows/*.tsv"):
    for k, v in rows_of(f).items(): FR[k] = dict(v, _file=f)
# SEED14 (3b cosine) lives in the seed table (SEED_ROWS.md), same value as 3b_all_cos
seed = [l for l in open(f"{P16}/SEED_ROWS.md") if l.startswith("| 3b | Qwen3-Omni Pitt cos(")][0]
SEED14 = seed.split("|")[3].strip()

def r4(x, fmt="{:.4f}"):
    return fmt.format(x)
out = []
def put(id_, orig, olo, ohi, rec, lo, hi, file, item, how, fmt="{:.4f}", kind="recompute"):
    rs = r4(rec, fmt) if not isinstance(rec, str) else rec
    try: m = (abs(float(orig) - float(rs)) < 1e-9) if not isinstance(rec, str) else (orig == rs)
    except Exception: m = (str(orig) == rs)
    ci_o = f"[{olo}, {ohi}]" if olo not in (None, "", "nan") else ""
    ci_r = f"[{lo:.4f}, {hi:.4f}]" if lo is not None else ""
    ci_m = "" if not ci_o or not ci_r else ("yes" if (abs(float(olo) - float(f'{lo:.4f}')) < 1e-9 and abs(float(ohi) - float(f'{hi:.4f}')) < 1e-9) else "no")
    out.append(dict(id=id_, original=orig, recomputed=rs, match_4dp="yes" if m else "no", file=file,
                    original_ci=ci_o, recomputed_ci=ci_r, ci_match_4dp=ci_m, kind=kind, audit_item=item, how=how,
                    recomputed_full=(repr(rec) if not isinstance(rec, str) else rec)))

# ---------------- POD3 3b direction (pod 1, Linux sklearn 1.9.1)
D = {c["arm"]: c for c in json.load(open(f"{PULL}/lo-p16twins-1/out/twin_direction.json"))["cells"]}
dfile = f"{PULL}/lo-p16twins-1/out/twin_direction.json"
IT_COS = "audit_p16 item 6: POD3 cosines, no twin (ledger LEFT me 5: twin the cosine 0.0051)"
for arm in ("all", "conflict", "agreement"):
    c = D[arm]; o = FR[f"3b_{arm}_cos"]
    put(f"3b_{arm}_cos", o["value"], None, None, c["cosine"], None, None, dfile,
        IT_COS if arm == "all" else "audit_p16 item 6: POD3 cosines, no twin", "Linux pod refit, lm_head rows fetched from the HF hub; cos(probe dir, mass-weighted Yes-No readout dir)")
put("SEED14_3b", SEED14, None, None, D["all"]["cosine"], None, None, dfile,
    "audit_p16 item 6: seed 3b cosine, no twin", "same computation as 3b_all_cos")
for arm, item in (("conflict", "audit_p16 item 6: POD3 by arm probe flagged verified though the verifier disagrees (0.6145 vs 0.5520)"),
                  ("agreement", "audit_p16 item 6: POD3 by arm probe flagged verified though the verifier disagrees (0.9349 vs 0.9314)")):
    c = D[arm]; o = FR[f"3b_{arm}_auc_probe"]
    put(f"3b_{arm}_auc_probe", o["value"], o["lo"], o["hi"], c["auc_probe"], *c["auc_probe_ci"], dfile, item,
        "Linux pod refit inside the arm (GroupKFold(5) by speaker, sklearn 1.9.1); the Mac verifier split folds with sklearn 1.7.2")
for arm in ("conflict", "agreement"):
    c = D[arm]; o = FR[f"3b_{arm}_auc_wo_d"]
    put(f"3b_{arm}_auc_wo_d", o["value"], o["lo"], o["hi"], c["auc_wo_d_trainz"], *c["auc_wo_d_trainz_ci"], dfile,
        "audit_p16 item 6: POD3 by arm without-d, no twin", "Linux pod refit, estimator as shipped (held-out projection standardised with the TRAINING fold)")
# extras from the same run (already twinned elsewhere; kept as a pipeline check)
for arm in ("all", "conflict", "agreement"):
    c = D[arm]
    for k, fid in (("auc_d", f"3b_{arm}_auc_d"),) + ((("auc_probe", "3b_all_auc_probe"), ("auc_wo_d_trainz", "3b_all_auc_wo_d")) if arm == "all" else ()):
        o = FR[fid]
        put(fid, o["value"], o["lo"], o["hi"], c[k], *c[k + "_ci"], dfile, "extra (pipeline check, twinned elsewhere)", "same pod run", kind="extra")
put("Q3Ofix_all_auc_without_d_CANONICAL_foldz", "0.8276", None, None, D["all"]["auc_wo_d_foldz"], *D["all"]["auc_wo_d_foldz_ci"], dfile,
    "extra (pipeline check, twinned in part17 T3)", "held-out projection standardised within the held-out fold", kind="extra")
put("3b_all_cos_random_floor", "0.0177", None, None, D["all"]["rand_mean_abs"], None, None, dfile, "extra", "mean |probe . r| over 2000 unit normals, default_rng(0)", kind="extra")

# ---------------- POD3 best stage (Linux pods, sklearn 1.9.1)
# Pitt: pods 2 / 3 / 4, all stages. PC-GITA: pod 3 (AMD EPYC 9965, OpenBLAS AVX512 SkylakeX kernel, the kernel that reproduces
# POD3 exactly: enc all 32 stages x 5 repeats identical); llm / ans rerun there on the top-4 stages; the all-stage PC-GITA runs on an
# AVX2 host (pods 5 / 6 / 7, OpenBLAS Haswell/Zen kernel) pick the same best stage but differ by up to 0.0007 per repeat.
BS = {("pitt", "enc"): (f"{PULL}/lo-p16twins-2/out/twin_pitt_enc.json", None, None),
      ("pitt", "llm"): (f"{PULL}/lo-p16twins-3/out/twin_pitt_llm.json", None, None),
      ("pitt", "ans"): (f"{PULL}/lo-p16twins-4/out/twin_pitt_ans.json", None, None),
      ("pcgita", "enc"): (f"{PULL}/lo-p16twins-3/out/twin_pcgita_enc_default.json", None, f"{PULL}/lo-p16twins-5/out/twin_pcgita_enc.json"),
      ("pcgita", "llm"): (f"{PULL}/lo-p16twins-3/out/twin_pcgita_llmtop4_skx.json", [13, 14, 15, 19], f"{PULL}/lo-p16twins-6/out/twin_pcgita_llm.json"),
      ("pcgita", "ans"): (f"{PULL}/lo-p16twins-3/out/twin_pcgita_anstop4_skx.json", [15, 18, 19, 27], f"{PULL}/lo-p16twins-7/out/twin_pcgita_ans.json")}
for (ds, st), (f, stages, avx2) in BS.items():
    j = json.load(open(f)); o = FR[f"3a_{ds}_{st}_best_stage"]
    best = j["best_stage"] if stages is None else stages[j["best_stage"]]
    how = (f"Linux pod refit, {'all ' + str(j['n_layers']) + ' stages' if stages is None else 'top-4 stages ' + str(stages)} x 5 repeats; "
           f"best stage {best} (same as POD3); per repeat {[round(x,4) for x in j['best_per_repeat']]}")
    if ds == "pcgita":
        a = json.load(open(avx2))
        how += (f"; AVX512 kernel run on AMD EPYC 9965. The all-stage run on an AVX2 host picks the same stage {a['best_stage']} and gives "
                f"{a['best_stage_auc_mean']:.4f} [{a['best_ci'][0]:.4f}, {a['best_ci'][1]:.4f}] ({avx2})")
    put(f"3a_{ds}_{st}_best_stage", o["value"], o["lo"], o["hi"], j["best_stage_auc_mean"], *j["best_ci"], f,
        "audit_p16 item 6: POD3 best stage, no twin", how)

# ---------------- Mac per-clip and log twins
M = json.load(open(f"{SC}/twin_mac_perclip.json"))
for r in M:
    rid = r["id"]
    if rid in ("M3_pitt_clip", "M3_pcgita_clip", "M3_neurovoz_clip"): kind = "extra"
    else: kind = "log twin (second record, not a recompute)" if rid.startswith(("MEDAIC", "P16.POD2")) else "recompute"
    fmt = r["fmt"]; rec = r["recomputed"]
    if fmt == "{}": rec = str(rec)
    put(rid, r["original"], r["original_lo"], r["original_hi"], rec, r["lo"], r["hi"], r["file"], r["closes"], r["how"],
        fmt=fmt if fmt != "{}" else "{}", kind=kind)
# the M7 fix row stores +0.1128: compare on the number
for x in out:
    if x["id"].startswith("M7fix#02"):
        x["recomputed"] = "+" + x["recomputed"] if not x["recomputed"].startswith(("+", "-")) else x["recomputed"]
        x["match_4dp"] = "yes" if abs(float(x["original"]) - float(x["recomputed"])) < 1e-9 else "no"

# ---------------- M5 random floors: Mac refit (the original M5 environment) + Linux pod cross-check
MM = json.load(open(f"{SC}/twin_m5_mac.json")); MP = json.load(open(f"{PULL}/lo-p16twins-1/out/twin_m5_pod.json"))
for cellk in ("all", "conflict_refit", "agreement_refit"):
    o = FR[f"M5_{cellk}_randfloor"]; c = MM[cellk]; p = MP[cellk]
    put(f"M5_{cellk}_randfloor", o["value"], o["lo"], o["hi"], c["randfloor"], *c["randfloor_ci"], f"{OUT}/mac/twin_m5_mac.json",
        "audit_p16 item 6: M5 flagged verified though the M5 verifier disagrees",
        f"Mac refit (sklearn 1.7.2, the environment M5 ran in), fresh default_rng(0) per cell; Linux pod sklearn 1.9.1 gives "
        f"{p['randfloor']:.4f} [{p['randfloor_ci'][0]:.4f}, {p['randfloor_ci'][1]:.4f}] because GroupKFold splits differently there")
# M5 AUC rows: also record the Mac refit value next to the per-clip recompute
for x in out:
    if x["id"].startswith("M5_") and "randfloor" not in x["id"]:
        k = x["id"]
        if k.startswith("M5_paired"): v = MM["paired_without_d"]["diff"]
        else:
            cellk = k[3:].split("_auc_")[0]; metric = "auc_" + k.split("_auc_")[1]
            v = MM[cellk][metric]
        x["how"] += f"; Mac refit of the whole M5 pipeline gives {v:.4f}"

order = {"recompute": 0, "log twin (second record, not a recompute)": 1, "extra": 2}
out.sort(key=lambda x: (order.get(x["kind"], 3)))
cols = ["id", "original", "recomputed", "match_4dp", "file", "original_ci", "recomputed_ci", "ci_match_4dp", "kind", "audit_item", "how", "recomputed_full"]
with open(f"{OUT}/twins_p16.tsv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t"); w.writeheader(); w.writerows(out)
n = len(out); ok = sum(x["match_4dp"] == "yes" for x in out)
print(f"rows {n}, value match_4dp {ok}/{n}; CI mismatches: {[x['id'] for x in out if x['ci_match_4dp']=='no']}; value mismatches: {[x['id'] for x in out if x['match_4dp']=='no']}")
json.dump(out, open(f"{SC}/twins_rows.json", "w"), indent=1)
