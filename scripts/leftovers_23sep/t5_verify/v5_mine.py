"""t5 verifier final table: combines the file-level recompute (v5_compare.py rows) with the verifier's own refits
(pods lo-v5-*, own code, own folds, own AUC; Mac runs) and writes verify/t5_VERIFIED.tsv (id, claimed, recomputed, verified)."""
import csv, json, os, re, glob, numpy as np
R = "<local data dir>/release"; LO = "<local data dir>/leftovers_23sep"
T5 = f"{LO}/t5"; V = f"{LO}/verify/t5v"; P = f"{V}/pull"
def num(x): x = x.strip(); m = re.match(r"np\.float64\((.*)\)$", x); return float(m.group(1) if m else x)
def rc(p):
    rows = list(csv.reader(open(p))); h = rows[0]; ix = {k: i for i, k in enumerate(h)}; b = [r for r in rows[1:] if r]
    oc = "auc_oof_sk" if "auc_oof_sk" in ix else ("auc_oof" if "auc_oof" in ix else "auc")
    d = dict(idx=[int(r[0]) for r in b], oof=np.array([num(r[ix[oc]]) for r in b]))
    if "auc_oof_rank" in ix: d["rank"] = np.array([num(r[ix["auc_oof_rank"]]) for r in b])
    if "auc_mean" in ix: d["mean"] = np.array([num(r[ix["auc_mean"]]) for r in b]); d["std"] = np.array([num(r[ix["auc_std"]]) for r in b])
    return d
f4 = lambda x: "%.4f" % x; fmt = lambda x: "0" if x == 0 else "%.2e" % x
SF = dict(enc="encoder", proj="proj", llm="llm", ans="ans")
def saved(ds, s):
    pre = {"pitt": f"{R}/omni_final/omni_pitt", "edaic_first30": f"{R}/omni_final/omni_edaic", "adress2020": f"{R}/omni_final/omni_adress2020",
           "neurovoz": f"{R}/omni_final/omni_neurovoz", "edaic_full": f"{R}/edaic_rerun/variants/o25_full"}[ds]
    return rc(f"{pre}_{SF[s]}_perlayer.csv")
side = json.load(open(f"{V}/v5_compare_side.json")); ANS = {k: v["answer_auc"] for k, v in side["zeroshot"].items()}
def cmpc(a, b, ans=None):
    assert a["idx"] == b["idx"]; n = len(a["oof"])
    o = dict(n=n, maxabs=float(np.max(np.abs(a["oof"] - b["oof"]))), m4=sum(f4(x) == f4(y) for x, y in zip(a["oof"], b["oof"])),
             rank_maxabs=float(np.max(np.abs(a["rank"] - b["oof"]))) if "rank" in a else None)
    if "mean" in a and "mean" in b: o["m4m"] = sum(f4(x) == f4(y) for x, y in zip(a["mean"], b["mean"])); o["m4s"] = sum(f4(x) == f4(y) for x, y in zip(a["std"], b["std"]))
    if ans is not None: o["ab"] = int((a["oof"] > ans).sum()); o["abs"] = int((b["oof"] > ans).sum())
    return o
def mine(pod, tag, s): return rc(f"{P}/{pod}/out/{tag}_{s}.csv")
def envf(pod): return json.load(open(f"{P}/{pod}/out/envfolds_{pod}.json"))
cpu = {p: envf(p).get("cpu", "?").replace(" Processor", "") for p in ("lo-v5-1", "lo-v5-2", "lo-v5-3", "lo-v5-4", "lo-v5-6", "lo-v5-8", "lo-v5-9") if os.path.exists(f"{P}/{p}/out/envfolds_{p}.json")}
rows = {r["id"]: r for r in csv.DictReader(open(f"{V}/t5_VERIFIED_rows.tsv"), delimiter="\t")}
def fr(i): return rows[i]["recomputed"]
def fok(*ids): return all(rows[i]["verified"] == "yes" for i in ids)
out = []; det = {}
def add(i, claimed, rec, ok, d=None): out.append(dict(id=i, claimed=claimed, recomputed=rec, verified="yes" if ok else "no")); det[i] = d
# 1-3 Pitt encoder
c = cmpc(mine("lo-v5-1", "pitt_encproj__exact", "enc"), saved("pitt", "enc"), ANS["pitt"])
add("T5_pitt_enc_pod10_maxabs", "0", f"t5 file {fr('T5_pitt_enc_pod10_maxabs')}; own refit lo-v5-1 ({cpu['lo-v5-1']}) {fmt(c['maxabs'])} (own rank AUC {fmt(c['rank_maxabs'])})",
    fok("T5_pitt_enc_pod10_maxabs") and c["maxabs"] == 0, c)
add("T5_pitt_enc_pod10_match4dp", "32/32", f"t5 file {fr('T5_pitt_enc_pod10_match4dp')} (mean/std {fr('T5_pitt_enc_pod10_meanstd_match4dp')}); own refit {c['m4']}/32 (mean {c['m4m']}/32, std {c['m4s']}/32)",
    fok("T5_pitt_enc_pod10_match4dp", "T5_pitt_enc_pod10_meanstd_match4dp") and c["m4"] == c["m4m"] == c["m4s"] == 32, c)
add("T5_pitt_enc_pod10_layers_above", "31/32 vs 31/32", f"answer {ANS['pitt']:.4f}; t5 file {fr('T5_pitt_enc_pod10_layers_above')}; own refit {c['ab']}/32 vs saved {c['abs']}/32",
    fok("T5_pitt_enc_pod10_layers_above") and c["ab"] == 31 and c["abs"] == 31 and f4(ANS["pitt"]) == "0.6578", c)
# 4-5 E-DAIC first 30 s encoder
c = cmpc(mine("lo-v5-2", "edaic30_encproj__exact", "enc"), saved("edaic_first30", "enc"), ANS["edaic_first30"])
add("T5_edaic_first30_enc_pod9_pod10_maxabs", "0", f"t5 files pod9 {fr('T5_edaic_first30_enc_pod9_maxabs')}, pod10 {fr('T5_edaic_first30_enc_pod10_maxabs')}; own refit lo-v5-2 ({cpu['lo-v5-2']}) {fmt(c['maxabs'])} (own rank AUC {fmt(c['rank_maxabs'])})",
    fok("T5_edaic_first30_enc_pod9_maxabs", "T5_edaic_first30_enc_pod10_maxabs") and c["maxabs"] == 0, c)
add("T5_edaic_first30_enc_match4dp", "32/32", f"t5 files pod9 {fr('T5_edaic_first30_enc_pod9_match4dp')}, pod10 {fr('T5_edaic_first30_enc_pod10_match4dp')}; above answer {ANS['edaic_first30']:.4f}: pod9 {fr('T5_edaic_first30_enc_pod9_layers_above')}, pod10 {fr('T5_edaic_first30_enc_pod10_layers_above')}; own refit {c['m4']}/32 (mean {c['m4m']}, std {c['m4s']}), above {c['ab']}/32 vs {c['abs']}/32",
    fok("T5_edaic_first30_enc_pod9_match4dp", "T5_edaic_first30_enc_pod10_match4dp", "T5_edaic_first30_enc_pod9_layers_above", "T5_edaic_first30_enc_pod10_layers_above",
        "T5_edaic_first30_enc_pod9_meanstd_match4dp", "T5_edaic_first30_enc_pod10_meanstd_match4dp") and c["m4"] == c["m4m"] == c["m4s"] == 32 and c["ab"] == c["abs"] == 2 and f4(ANS["edaic_first30"]) == "0.6620", c)
# 6-7 projector, LM, answer stages
for ds, key, cl, pods, trows in (("pitt", "T5_pitt_proj_llm_ans_maxabs", "0; 1/1, 29/29, 29/29 at 4 dp", {"proj": ("lo-v5-1", "pitt_encproj__exact"), "llm": ("lo-v5-3", "pitt_llm__exact"), "ans": ("lo-v5-4", "pitt_ans__exact")},
                                  ["T5_pitt_proj_pod10", "T5_pitt_llm_pod2", "T5_pitt_ans_pod13"]),
                                 ("edaic_first30", "T5_edaic_first30_proj_llm_ans_maxabs", "0; 1/1, 29/29, 29/29 at 4 dp", {"proj": ("lo-v5-2", "edaic30_encproj__exact"), "llm": ("lo-v5-6", "edaic30_llm__exact"), "ans": ("lo-v5-6", "edaic30_ans__exact")},
                                  ["T5_edaic_first30_proj_pod10", "T5_edaic_first30_proj_pod9", "T5_edaic_first30_llm_pod5", "T5_edaic_first30_ans_pod6"])):
    cc = {s: cmpc(mine(p, t, s), saved(ds, s)) for s, (p, t) in pods.items()}
    tf = "; ".join(f"{t[3:]} {fr(t + '_maxabs')}, {fr(t + '_match4dp')}, mean/std {fr(t + '_meanstd_match4dp')}" for t in trows)
    own = ", ".join(f"{s} {fmt(cc[s]['maxabs'])} {cc[s]['m4']}/{cc[s]['n']} ({cpu[pods[s][0]]})" for s in ("proj", "llm", "ans"))
    ok = fok(*[t + x for t in trows for x in ("_maxabs", "_match4dp", "_meanstd_match4dp")]) and all(v["maxabs"] == 0 and v["m4"] == v["n"] == v["m4m"] == v["m4s"] for v in cc.values())
    cpus = f"task pod cpus lo-t5-2 {side['task_pod_cpus']['lo-t5-2']}, lo-t5-13 {side['task_pod_cpus']['lo-t5-13']}, lo-t5-5 {side['task_pod_cpus']['lo-t5-5']}, lo-t5-6 {side['task_pod_cpus']['lo-t5-6']}"
    add(key, cl, f"t5 files: {tf}; own refit: {own}; {cpus}", ok, cc)
# 8-10 other extractions
for key, cl, ds, pod, tag, trow, pv in (("T5_adress2020_enc_pod7", "2.88e-03; 1/32", "adress2020", "lo-v5-1", "adress2020_encproj__exact", "T5_adress2020_enc_pod7", 0.038),
                                         ("T5_neurovoz_enc_pod8", "1.09e-03; 5/32", "neurovoz", "lo-v5-3", "neurovoz_encproj__exact", "T5_neurovoz_enc_pod8", 0.047),
                                         ("T5_edaic_full_enc_pod9", "2.97e-03; 2/32", "edaic_full", "lo-v5-2", "edaicfull_encproj__exact", "T5_edaic_full_enc_pod9", 0.030)):
    c = cmpc(mine(pod, tag, "enc"), saved(ds, "enc"), ANS[ds]); pd = side["pyes_maxdiff"][ds][0]
    extra = f"; answer {side['answer_edaicfull_states_run']:.4f} (states run) vs {ANS[ds]:.4f} (saved)" if ds == "edaic_full" else ""
    exp_ab = {"adress2020": 32, "neurovoz": 32, "edaic_full": 0}[ds]
    ok = fok(trow + "_maxabs", trow + "_match4dp", trow + "_layers_above") and f"{fmt(c['maxabs'])}; {c['m4']}/32" == cl and c["ab"] == c["abs"] == exp_ab and round(pd, 3) == pv
    if ds == "edaic_full": ok = ok and f4(side["answer_edaicfull_states_run"]) == "0.8278" and f4(ANS[ds]) == "0.8285"
    add(key, cl, f"t5 file {fr(trow + '_maxabs')}; {fr(trow + '_match4dp')}; above {fr(trow + '_layers_above')}; own refit {cpu[pod]} {fmt(c['maxabs'])}; {c['m4']}/32, above {c['ab']}/32 vs {c['abs']}/32; p_yes max diff {pd:.3f}{extra}", ok, c)
# 11 not possible
np_chk = json.load(open(f"{V}/v5_notpossible.json"))
add("T5_adresso_pcgita_kcl_mid300_mid30_enc_refit", "not possible", np_chk["summary"], np_chk["ok"], np_chk)
# 12 P17 verifier explained (own Mac run, Mac fold file)
P17 = {d["dataset"]: np.array(d["recomputed_oof"]) for d in json.load(open(f"{R}/edaic_rerun/part17/verify/T5_verify.json"))["extra"]["recompute"] if d.get("matched")}
mp = rc(f"{V}/mac_out/pitt_encproj__mac_macfile_enc.csv"); me = rc(f"{V}/mac_out/edaic30_encproj__mac_macfile_enc.csv")
dp, de = float(np.max(np.abs(mp["oof"] - P17["pitt"]))), float(np.max(np.abs(me["oof"] - P17["edaic_first30"])))
cp, ce = cmpc(mp, saved("pitt", "enc")), cmpc(me, saved("edaic_first30", "enc"))
add("T5_diag_p17_verifier_refit_explained", "1.11e-16 (Pitt and E-DAIC first30)",
    f"t5 files {fr('T5_diag_pitt_enc_p17_verifier_refit_explained')} and {fr('T5_diag_edaic_first30_enc_p17_verifier_refit_explained')}; own Mac refit (Mac fold file) vs P17 verifier curve: Pitt {fmt(dp)}, E-DAIC {fmt(de)}; vs saved {fmt(cp['maxabs'])} {cp['m4']}/32 and {fmt(ce['maxabs'])} {ce['m4']}/32",
    fok("T5_diag_pitt_enc_p17_verifier_refit_explained", "T5_diag_edaic_first30_enc_p17_verifier_refit_explained") and max(dp, de) <= 1.2e-16 and fmt(cp["maxabs"]) == "5.34e-02" and fmt(ce["maxabs"]) == "8.77e-02" and cp["m4"] == ce["m4"] == 0,
    dict(dp=dp, de=de, cp=cp, ce=ce))
# 13 folds
fm = json.load(open(f"{V}/mac_out/envfolds_mac.json")); podf = {}
for p in cpu:
    e = envf(p)
    for ds in ("pitt", "edaic"):
        if ds in e: podf.setdefault(ds, []).append((p, e["sklearn"], e[ds]["pod_file"]["sklearn_here_eq"], e[ds]["n"], e[ds]["argsort_default_equals_stable"], e["groupkfold_argsort_lines"]))
pod_ok = all(x[2] == x[3] and x[1] == "1.9.1" and x[4] is False and any('kind="stable"' in l for l in x[5]) for v in podf.values() for x in v)
mac_ok = (fm["sklearn"] == "1.7.2" and fm["pitt"]["pod_file"]["sklearn_here_eq"] == 125 and fm["edaic"]["pod_file"]["sklearn_here_eq"] == 20
          and fm["pitt"]["mac_file"]["sklearn_here_eq"] == 468 and fm["edaic"]["mac_file"]["sklearn_here_eq"] == 275
          and fm["pitt"]["argsort_default_equals_stable"] is False and not any("kind" in l for l in fm["groupkfold_argsort_lines"]))
fold_rows = [k for k in rows if k.startswith("T5_folds_")]
add("T5_diag_folds", "pods: pod file 468/468 and 275/275; Mac: pod file 125/468 and 20/275",
    f"t5 fold rows {sum(rows[k]['verified'] == 'yes' for k in fold_rows)}/{len(fold_rows)} recomputed from the native fold csvs; own pods (sklearn 1.9.1, {len(cpu)} pods): pitt pod file {sorted(set(f'{x[2]}/{x[3]}' for x in podf.get('pitt', [])))}, E-DAIC {sorted(set(f'{x[2]}/{x[3]}' for x in podf.get('edaic', [])))}; pod source '{podf['pitt'][0][5][0]}'; Mac sklearn {fm['sklearn']}: pod file {fm['pitt']['pod_file']['sklearn_here_eq']}/468 and {fm['edaic']['pod_file']['sklearn_here_eq']}/275, Mac file {fm['pitt']['mac_file']['sklearn_here_eq']}/468 and {fm['edaic']['mac_file']['sklearn_here_eq']}/275, source '{fm['groupkfold_argsort_lines'][0]}'; numpy default argsort equals stable on the counts: False on Mac and on every pod",
    pod_ok and mac_ok and fok(*fold_rows), dict(pods=podf, mac={k: fm[k] for k in ("pitt", "edaic", "groupkfold_argsort_lines", "sklearn")}))
# 14 pod libs, Mac folds
a = cmpc(mine("lo-v5-1", "pitt_encproj__macfile", "enc"), saved("pitt", "enc")); b = cmpc(mine("lo-v5-2", "edaic30_encproj__macfile", "enc"), saved("edaic_first30", "enc"))
add("T5_diag_pod_lib_mac_folds", "Pitt 5.37e-02, 0/32; E-DAIC 8.79e-02, 0/32",
    f"t5 files Pitt {fr('T5_diag_pitt_enc_macfile_pod10')}, E-DAIC pod10 {fr('T5_diag_edaic_first30_enc_macfile_pod10')}, pod9 {fr('T5_diag_edaic_first30_enc_macfile_pod9')}; own refit Pitt {fmt(a['maxabs'])}, {a['m4']}/32 ({cpu['lo-v5-1']}); E-DAIC {fmt(b['maxabs'])}, {b['m4']}/32 ({cpu['lo-v5-2']})",
    fok("T5_diag_pitt_enc_macfile_pod10", "T5_diag_edaic_first30_enc_macfile_pod10", "T5_diag_edaic_first30_enc_macfile_pod9") and fmt(a["maxabs"]) == "5.37e-02" and fmt(b["maxabs"]) == "8.79e-02" and a["m4"] == b["m4"] == 0, dict(a=a, b=b))
# 15 Mac libs, pod folds
a = cmpc(rc(f"{V}/mac_out/pitt_encproj__mac_podfile_enc.csv"), saved("pitt", "enc")); b = cmpc(rc(f"{V}/mac_out/edaic30_encproj__mac_podfile_enc.csv"), saved("edaic_first30", "enc"))
add("T5_diag_mac_lib_pod_folds", "Pitt 5.62e-04, 11/32; E-DAIC 8.70e-04, 4/32",
    f"t5 files Pitt {fr('T5_diag_pitt_enc_mac_podfile_mac')}, E-DAIC {fr('T5_diag_edaic_first30_enc_mac_podfile_mac')}; own Mac refit (sklearn 1.7.2, numpy 2.2.6, Accelerate) Pitt {fmt(a['maxabs'])}, {a['m4']}/32; E-DAIC {fmt(b['maxabs'])}, {b['m4']}/32",
    fok("T5_diag_pitt_enc_mac_podfile_mac", "T5_diag_edaic_first30_enc_mac_podfile_mac") and fmt(a["maxabs"]) == "5.62e-04" and a["m4"] == 11 and fmt(b["maxabs"]) == "8.70e-04" and b["m4"] == 4, dict(a=a, b=b))
# 16 Zen 3 host
a = cmpc(mine("lo-v5-8", "pitt_encproj__exact", "enc"), saved("pitt", "enc")); b = cmpc(mine("lo-v5-8", "edaic30_encproj__exact", "enc"), saved("edaic_first30", "enc"))
c = cmpc(mine("lo-v5-9", "pitt_ans__exact", "ans"), saved("pitt", "ans"))
add("T5_diag_zen3_host", "Pitt enc 6.78e-04, 8/32; E-DAIC enc 5.07e-04, 7/32; Pitt ans 7.94e-04, 4/29",
    f"t5 files {fr('T5_diag_pitt_enc_exact_script_pod1')}, {fr('T5_diag_edaic_first30_enc_exact_script_pod4')}, Pitt ans pod3 from t5 file {fmt(cmpc(rc(T5 + '/pull/lo-t5-3/out/pitt_ans_ans_perlayer.csv'), saved('pitt', 'ans'))['maxabs'])}, {cmpc(rc(T5 + '/pull/lo-t5-3/out/pitt_ans_ans_perlayer.csv'), saved('pitt', 'ans'))['m4']}/29; own refit on {cpu['lo-v5-8']} / {cpu['lo-v5-9']}: Pitt enc {fmt(a['maxabs'])}, {a['m4']}/32; E-DAIC enc {fmt(b['maxabs'])}, {b['m4']}/32; Pitt ans {fmt(c['maxabs'])}, {c['m4']}/29",
    fok("T5_diag_pitt_enc_exact_script_pod1", "T5_diag_edaic_first30_enc_exact_script_pod4") and (fmt(a["maxabs"]), a["m4"], fmt(b["maxabs"]), b["m4"], fmt(c["maxabs"]), c["m4"]) == ("6.78e-04", 8, "5.07e-04", 7, "7.94e-04", 4), dict(a=a, b=b, c=c))
# 17 BLAS core types
res = {}
for ds, pod, job in (("pitt", "lo-v5-1", "pitt_encproj"), ("edaic_first30", "lo-v5-2", "edaic30_encproj")):
    for ct in ("SkylakeX", "Haswell", "Zen", "Sandybridge", "Prescott", "SapphireRapids"):
        c = cmpc(mine(pod, f"{job}__ct{ct}", "enc"), saved(ds, "enc")); res[(ds, ct)] = (fmt(c["maxabs"]), c["m4"])
t5ct = {(ds, ct): fr(f"T5_diag_{ds}_enc_podfile_ct{ct}_pod10") for ds in ("pitt", "edaic_first30") for ct in ("SkylakeX", "Haswell", "Zen", "Sandybridge", "Prescott")}
exp = {("pitt", "SkylakeX"): ("0", 32), ("edaic_first30", "SkylakeX"): ("0", 32), ("pitt", "Haswell"): ("6.78e-04", 8), ("pitt", "Zen"): ("6.78e-04", 8), ("edaic_first30", "Haswell"): ("5.07e-04", 7),
       ("edaic_first30", "Zen"): ("5.07e-04", 7), ("pitt", "Sandybridge"): ("4.84e-04", 8), ("edaic_first30", "Sandybridge"): ("6.52e-04", 3), ("pitt", "Prescott"): ("5.42e-04", 5), ("edaic_first30", "Prescott"): ("5.80e-04", 6)}
spr_t5 = {ds: cmpc(rc(f"{T5}/pull/lo-t5-10/out/{j}__podfile_ctSapphireRapids_enc.csv"), saved(ds, "enc")) for ds, j in (("pitt", "pitt_encproj"), ("edaic_first30", "edaic30_encproj"))}
ok = all(res[k] == v for k, v in exp.items()) and all(t5ct[k] == f"{v[0]}; {v[1]}/32" for k, v in exp.items()) and res[("edaic_first30", "SapphireRapids")] == ("0", 32) and fmt(spr_t5["edaic_first30"]["maxabs"]) == "0"
add("T5_diag_blas_coretype", "SkylakeX 0, 32/32; Haswell or Zen 6.78e-04 and 5.07e-04; Sandybridge 4.84e-04 and 6.52e-04; Prescott 5.42e-04 and 5.80e-04",
    "own refit (Pitt on " + cpu["lo-v5-1"] + ", E-DAIC on " + cpu["lo-v5-2"] + "): " + "; ".join(f"{ct} {res[('pitt', ct)][0]} {res[('pitt', ct)][1]}/32 and {res[('edaic_first30', ct)][0]} {res[('edaic_first30', ct)][1]}/32" for ct in ("SkylakeX", "Haswell", "Zen", "Sandybridge", "Prescott", "SapphireRapids"))
    + f"; t5 files agree for all 10 claimed cells; t5 SapphireRapids Pitt {fmt(spr_t5['pitt']['maxabs'])} {spr_t5['pitt']['m4']}/32, E-DAIC {fmt(spr_t5['edaic_first30']['maxabs'])} {spr_t5['edaic_first30']['m4']}/32 (note: SapphireRapids is also exact on Pitt, so the 'what' text's 'SapphireRapids on E-DAIC' undersells it; the claimed numbers are right)", ok, {f"{k[0]}_{k[1]}": v for k, v in res.items()})
# 18 dtype and threads
f64p = cmpc(mine("lo-v5-1", "pitt_encproj__f64", "enc"), saved("pitt", "enc")); f64e = cmpc(mine("lo-v5-2", "edaic30_encproj__f64", "enc"), saved("edaic_first30", "enc"))
t8p = cmpc(mine("lo-v5-1", "pitt_encproj__thr8", "enc"), saved("pitt", "enc")); t8e = cmpc(mine("lo-v5-2", "edaic30_encproj__thr8", "enc"), saved("edaic_first30", "enc"))
got = (fmt(f64p["maxabs"]), f64p["m4"], fmt(f64e["maxabs"]), f64e["m4"], fmt(t8p["maxabs"]), t8p["m4"], fmt(t8e["maxabs"]), t8e["m4"])
add("T5_diag_dtype_threads", "float64: 4.65e-04, 8/32 and 1.09e-03, 5/32; 8 BLAS threads: Pitt 5.62e-04, 4/32, E-DAIC 0, 32/32",
    f"t5 files f64 {fr('T5_diag_pitt_enc_podfile_f64_pod10')} and {fr('T5_diag_edaic_first30_enc_podfile_f64_pod10')}; thr8 {fr('T5_diag_pitt_enc_podfile_thr8_pod10')} and {fr('T5_diag_edaic_first30_enc_podfile_thr8_pod10')}; own refit f64 {got[0]}, {got[1]}/32 and {got[2]}, {got[3]}/32; 8 threads (4 workers) Pitt {got[4]}, {got[5]}/32, E-DAIC {got[6]}, {got[7]}/32",
    fok("T5_diag_pitt_enc_podfile_f64_pod10", "T5_diag_edaic_first30_enc_podfile_f64_pod10", "T5_diag_pitt_enc_podfile_thr8_pod10", "T5_diag_edaic_first30_enc_podfile_thr8_pod10")
    and got[:4] == ("4.65e-04", 8, "1.09e-03", 5) and got[6:] == ("0", 32), dict(f64p=f64p, f64e=f64e, t8p=t8p, t8e=t8e))
# 19 task verifier
vr = side["task_vrefit_vs_saved"]; same_run = {k: v for k, v in vr.items() if k.split(":")[1].split("__")[0] in ("pitt_encproj", "pitt_llm", "pitt_ans", "edaic30_encproj", "edaic30_llm", "edaic30_ans")}
avx = {k: v for k, v in same_run.items() if v["cpu"] and "7713" not in v["cpu"]}; mx = max(v["maxabs"] for v in avx.values())
nrow = side["rows_recomputed"]
add("T5_verifier", "106 agree, 0 disagree of 106",
    f"task log: {side['task_verify_log_total'][0]}; my own recompute of all rows of t5_values.tsv from the raw files: {nrow['agree']} agree of {nrow['n']}; task second code path (vrefit) on AVX-512 pods, same-run states: {len(avx)} curves, max diff {fmt(mx)}",
    nrow["agree"] == nrow["n"] == 106 and side["task_verify_log_total"][0].strip() == "TOTAL agree 106 disagree 0 of 106" and mx <= 2.3e-16, dict(avx=avx))
with open(f"{LO}/verify/t5_VERIFIED.tsv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "claimed", "recomputed", "verified"], delimiter="\t"); w.writeheader(); w.writerows(out)
json.dump(dict(details=det, pod_cpus=cpu), open(f"{V}/t5_VERIFIED_details.json", "w"), indent=1, default=str)
for o in out: print(o["verified"], o["id"], "|", o["recomputed"][:260])
