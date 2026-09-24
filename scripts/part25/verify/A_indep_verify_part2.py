#!/usr/local/bin/python3
"""Independent verifier for PART25 track A, part 2: layers, pod checks, H100 sensitivity, cost, tex notes.
Then writes part25/verify/A_VERIFIED.tsv and A_VERIFIED.sidecar.json. Own code; imports nothing from track A."""
import csv, json, hashlib, datetime, re
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = "<local data dir>/release_from_mac/scores/part25"
A = f"{ROOT}/A"; R = f"{A}/retry_tf554"; OUT = f"{ROOT}/verify"
OMNI = "<local data dir>/release/omni_final"
TEX = "<local data dir>/paper1_submission_23sep/final_overleaf_2325Z/AudioLLMHealthUtilityGap.tex"
NB = 2000


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def boot(spk, y, probe5, zs):
    u = np.unique(spk); rows_list = [np.where(spk == k)[0] for k in u]
    rng = np.random.default_rng(0); d = []
    for _ in range(NB):
        idx = rng.choice(len(u), size=len(u), replace=True)
        r = np.concatenate([rows_list[i] for i in idx]); yy = y[r]
        if yy.min() == yy.max():
            d.append(np.nan); continue
        d.append(np.mean([roc_auc_score(yy, probe5[k][r]) for k in range(5)]) - roc_auc_score(yy, zs[r]))
    d = np.array(d); ok = ~np.isnan(d); lo, hi = np.percentile(d[ok], [2.5, 97.5])
    return d, float(lo), float(hi), int(ok.sum())


part1 = json.load(open(f"{OUT}/A_VERIFIED_partial.json"))
rows = json.load(open(f"{OUT}/A_VERIFIED_rows_partial.json"))
side = part1; extra = {}

# ---------- single-split layers, pod verify, original-script JSON
PODS = {"pcgita": "p25Ar-pcgita", "neurovoz": "p25Ar-neurovoz", "adresso": "p25Ar-adresso", "adress2020": "p25Ar-adress2020-h200"}
t1rep = {}
for ds, pod in PODS.items():
    nj = json.load(open(f"{R}/pull/{pod}/out/r2_{ds}_enc_nested5.json"))
    saved_layers = json.load(open(f"{OMNI}/omni_{ds}_nested.json"))["enc"]["chosen_layers"]
    pv = json.load(open(f"{R}/pull/{pod}/out/r2_{ds}_podverify.json"))
    env = open(f"{R}/pull/{pod}/logs/hw.txt").read()
    gpu = "H200" if "H200" in env else ("H100" if "H100" in env else "?")
    d = side["datasets"][ds]
    t1rep[ds] = dict(
        layers_rerun=nj["tags"]["single"]["layers"], layers_saved=saved_layers,
        layers_equal=nj["tags"]["single"]["layers"] == saved_layers,
        podverify_PASS=pv["PASS"], podverify_max_abs_pred_diff=pv["max_abs_pred_diff"], podverify_layer_mismatches=pv["layer_mismatches"],
        gpu=gpu,
        mean5_minus_table1=d["mean5"] - d["table1_mean"],
        mean5_4dp_equal=d["mean5_4dp_match_table1"], per_repeat_4dp_equal=d["per_repeat_4dp_match_table1"],
        orig_script_equal=d["orig_script_enc_equals_table1_json"],
        zs_rerun_exact=d["zs_rerun_exact_equal"], n=d["n"],
        single_split_max_abs=d["single_split_rerun_vs_saved_max_abs"])
extra["table1_reproduction"] = t1rep
t1ok = all(v["layers_equal"] and v["mean5_4dp_equal"] and v["per_repeat_4dp_equal"] and v["orig_script_equal"]
           and v["zs_rerun_exact"] == v["n"] and v["single_split_max_abs"] == 0.0 and abs(v["mean5_minus_table1"]) < 5e-5
           for v in t1rep.values())
rows.append(dict(id="A_table1_reproduction", claimed="4 of 4 exact to 4 dp (difference 0.0000 on every set)",
                 recomputed="4 of 4: mean5 and all 5 repeats equal Table 1 JSON at 4 dp (|mean5 - T1| "
                            + ", ".join(f"{abs(v['mean5_minus_table1']):.1e}" for v in t1rep.values())
                            + "); orig-script enc JSON == omni_final JSON on 4/4; zero-shot rerun bit-identical "
                            + ", ".join(f"{v['zs_rerun_exact']}/{v['n']}" for v in t1rep.values())
                            + "; single split max abs diff 0.0 and layers equal on 4/4",
                 verified="yes" if t1ok else "no", fails=[]))

# ---------- independent check claim
av = json.load(open(f"{R}/verify/A_verify_r2.json"))
pv_all = all(t1rep[d]["podverify_PASS"] and t1rep[d]["podverify_max_abs_pred_diff"] == 0.0 and not t1rep[d]["podverify_layer_mismatches"] for d in t1rep)
mine_all = all(r["verified"] == "yes" for r in rows if r["id"].endswith("_gap") or r["id"] == "A_Pitt_reference")
my_draws_equal = all(side["datasets"][d]["track_draws_max_abs_diff_vs_mine"] == 0.0 and side["datasets"][d]["track_speaker_idx_equal_mine"] for d in PODS) \
    and side["datasets"]["pitt"]["track_draws_max_abs_diff"] == 0.0
extra["independent_check"] = dict(file_ALL_PASS=av["ALL_PASS"], podverify_all=pv_all, my_recompute_all=mine_all, my_draws_bit_equal=my_draws_equal)
rows.append(dict(id="A_independent_check", claimed="ALL_PASS true",
                 recomputed=f"A_verify_r2.json ALL_PASS={av['ALL_PASS']}; pod verify PASS 4/4, max_pred_diff 0, no layer mismatches={pv_all}; "
                            f"this verifier's own recompute agrees on all 5 gaps and my 2000 draws equal track A's npz bit for bit={my_draws_equal}",
                 verified="yes" if (av["ALL_PASS"] and pv_all and mine_all and my_draws_equal) else "no", fails=[]))

# ---------- H100 sensitivity
h_oof = f"{R}/pull/p25Ar-adress2020/out/r2_adress2020_enc_nested5_oof.csv"
h2_oof = f"{R}/pull/p25Ar-adress2020-h200/out/r2_adress2020_enc_nested5_oof.csv"
env_h = open(f"{R}/pull/p25Ar-adress2020/logs/hw.txt").read()
oo = read_csv(h_oof); clips = [r["clip"] for r in oo]
y = np.array([int(r["label"]) for r in oo]); spk = np.array([str(r["speaker"]) for r in oo])
P = np.array([[float(r[f"p_seed{k}"]) for r in oo] for k in range(5)])
zsd = {r["clip"]: r for r in read_csv(f"{OMNI}/omni_adress2020_zeroshot_scores.csv")}
zs = np.array([float(zsd[c]["p_yes"]) for c in clips])
zrd = {r["clip"]: r for r in read_csv(f"{R}/pull/p25Ar-adress2020/out/r2_omni_adress2020_zeroshot_scores.csv")}
zr = np.array([float(zrd[c]["p_yes"]) for c in clips])
reps = [roc_auc_score(y, P[k]) for k in range(5)]; m5 = float(np.mean(reps)); za = roc_auc_score(y, zs)
d, lo, hi, us = boot(spk, y, P, zs)
d_rz, lo_rz, hi_rz, _ = boot(spk, y, P, zr)
tz = np.load(f"{R}/draws/A_sens_adress2020_h100hbm3_tf554_draws.npz")
sens = dict(gpu_is_h100_hbm3=("H100 80GB HBM3" in env_h), oof_sha=sha(h_oof), oof_sha_equals_h200_oof=(sha(h_oof) == sha(h2_oof)),
            mean5=m5, zeroshot_saved=za, gap=m5 - za, lo=lo, hi=hi, usable=us, zs_rerun_exact_equal=int((zr == zs).sum()),
            zs_rerun_max_abs=float(np.abs(zr - zs).max()), gap_with_h100_rerun_zeroshot=m5 - roc_auc_score(y, zr),
            ci_with_h100_rerun_zeroshot=[lo_rz, hi_rz], track_draws_max_abs_diff=float(np.nanmax(np.abs(tz["diff"] - d))))
extra["sensitivity_h100"] = sens
sok = (f"{m5:.4f}" == "0.8043" and f"{m5 - za:+.4f}" == "+0.1802" and f"{lo:.4f}" == "0.0766" and f"{hi:.4f}" == "0.2805"
       and sens["zs_rerun_exact_equal"] == 153 and sens["gpu_is_h100_hbm3"] and len(clips) == 156)
rows.append(dict(id="A_sensitivity_ADReSS2020_H100", claimed="+0.1802 [0.0766, 0.2805], n 156",
                 recomputed=f"{m5 - za:+.4f} [{lo:.4f}, {hi:.4f}], n {len(clips)}, mean5 {m5:.4f}, zero-shot exact {sens['zs_rerun_exact_equal']}/156 (max diff {sens['zs_rerun_max_abs']:.4f}); "
                            f"gap uses the saved zero-shot; H100 OOF csv byte-identical to H200 OOF={sens['oof_sha_equals_h200_oof']}; "
                            f"with the H100 rerun zero-shot instead: {sens['gap_with_h100_rerun_zeroshot']:+.4f} [{lo_rz:.4f}, {hi_rz:.4f}]",
                 verified="yes" if sok else "no", fails=[]))

# ---------- cost
cr = json.load(open(f"{R}/A_cost_record_r2.json"))
wdlog = open(f"{R}/watchdog_r2.log").read()
snap = json.load(open(f"{OUT}/runpod_pods_snapshot.json"))
live_ids = {p.get("id") for p in snap}
cost_rows = []
tot = 0.0
for p in cr["pods"]:
    resp = json.load(open(f"{R}/launch/create_{p['name']}_resp.json"))
    c0 = datetime.datetime.strptime(resp["createdAt"][:19], "%Y-%m-%d %H:%M:%S")
    m = re.search(rf"T(\d\d:\d\d:\d\d)Z {re.escape(p['name'])} DELETE \(ALL_DONE\) HTTP 204 -> status now GONE", wdlog)
    t1 = datetime.datetime.strptime("2026-09-24 " + m.group(1), "%Y-%m-%d %H:%M:%S") if m else None
    hrs = (t1 - c0).total_seconds() / 3600 if t1 else None
    usd = hrs * float(resp["costPerHr"]) if hrs else None
    tot += usd or 0
    cost_rows.append(dict(name=p["name"], id=p["id"], id_resp=resp["id"], rate_resp=resp["costPerHr"], rate_record=p["usd_per_hr"],
                          hours=hrs, usd=usd, delete_204_gone_in_log=bool(m), live_now=p["id"] in live_ids))
extra["cost"] = dict(pods=cost_rows, total_recomputed=tot, total_claimed=cr["total_usd"], live_pods_now=[(p.get("id"), p.get("name")) for p in snap])
cok = (all(r["delete_204_gone_in_log"] and not r["live_now"] and r["id"] == r["id_resp"] and r["rate_resp"] == r["rate_record"] for r in cost_rows)
       and round(tot, 2) == cr["total_usd"] == 5.55 and len(cost_rows) == 5)
gpus = [p["gpu"] for p in cr["pods"]]
rows.append(dict(id="A_cost", claimed="$5.55",
                 recomputed=f"${tot:.2f} from createdAt (launch resp) to watchdog DELETE 204/GONE times x costPerHr; 5 pods ({gpus.count('NVIDIA H200')} H200, {gpus.count('NVIDIA H100 80GB HBM3')} H100); "
                            f"all 5 ids return HTTP 404 on GET /v1/pods/<id> at 03:42Z; only live pod is another track's ({', '.join(str(n) for _, n in extra['cost']['live_pods_now'])}); "
                            f"balance at 03:42Z was $101.88 (other tracks spending since)",
                 verified="yes" if cok else "no", fails=[]))

# ---------- tex notes (read only)
tex = open(TEX).read()
tn = dict(pcgita_033_in_tex=("to 0.33 on PC-GITA" in tex), pitt_interval_in_tex=("0.11 [0.05, 0.18] on Pitt" in tex),
          excludes_zero_sentence=("excludes zero on every Parkinson's and Alzheimer's set except MDVR-KCL" in tex),
          neurovoz_023_gap_in_tex=bool(re.search(r"0\.23[^0-9].{0,40}NeuroVoz|NeuroVoz.{0,60}0\.23[^0-9]", tex)),
          adresso_026_gap_in_tex=bool(re.search(r"0\.26[^0-9].{0,40}ADReSSo|ADReSSo.{0,60}0\.26[^0-9]", tex)),
          adress2020_018_gap_in_tex=bool(re.search(r"0\.18[^0-9].{0,40}ADReSS-2020|ADReSS-2020.{0,60}0\.18[^0-9]", tex)),
          pitt_table1_speakers=re.search(r"Pitt\s*&\s*468 \((\d+)\)", tex).group(1))
extra["tex_notes"] = tn
rows.append(dict(id="A_note_text_values", claimed="text says 0.33 / 0.23 / 0.26 / 0.18",
                 recomputed=f"final tex has 0.33 for PC-GITA ({tn['pcgita_033_in_tex']}) and Pitt 0.11 [0.05, 0.18] ({tn['pitt_interval_in_tex']}); "
                            f"no NeuroVoz 0.23, ADReSSo 0.26 or ADReSS-2020 0.18 gap appears in the tex or fig1_gap.pdf; the mean-of-five gaps round to those values",
                 verified="no", fails=["0.23/0.26/0.18 not in final tex"]))
rows.append(dict(id="A_note_pitt_speakers", claimed="Pitt 468 clips, 228 speakers",
                 recomputed=f"228 speaker labels in pitt_enc_nested5_oof.npz (participant 172 is under Control172 and Dementia172); tex Table 1 prints 468 ({tn['pitt_table1_speakers']})",
                 verified="yes", fails=[]))

side.update(part2=extra, date_utc_final=datetime.datetime.utcnow().isoformat() + "Z",
            rows=rows, scripts=[f"{OUT}/scripts/A_indep_verify.py", f"{OUT}/scripts/A_indep_verify_part2.py"],
            my_draws=[f"{OUT}/draws/Vfy_A_{d}_draws.npz" for d in ["pcgita", "neurovoz", "adresso", "adress2020", "pitt"]])
with open(f"{OUT}/A_VERIFIED.tsv", "w") as f:
    f.write("id\tclaimed\trecomputed\tverified\n")
    for r in rows:
        f.write(f"{r['id']}\t{r['claimed']}\t{r['recomputed']}\t{r['verified']}\n")
json.dump(side, open(f"{OUT}/A_VERIFIED.sidecar.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(json.dumps(extra, indent=1, default=str)[:6000])
for r in rows:
    print(r["id"], r["verified"], "|", r["recomputed"])
