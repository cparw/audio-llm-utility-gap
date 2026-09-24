"""t5 independent verifier: recompute every t5 value from the raw per-layer files, with own file lookup, own AUC,
own 4 dp rule (two values formatted %.4f are the same string), no pandas.  Also scores the verifier's own refits
(pods lo-v5-*, Mac) against the saved curves.  Writes t5_VERIFIED.tsv, t5_VERIFIED_rows.tsv, t5_VERIFIED_sidecar.json."""
import csv, json, os, re, glob, hashlib, sys, numpy as np
R = "<local data dir>/release"; G = "<local data dir>/release_from_mac"
LO = f"{G}/scores/leftovers_23sep"; T5 = f"{LO}/t5"; V = f"{LO}/verify/t5v"; OUTD = f"{LO}/verify"
def fsha(p):
    h = hashlib.sha256(); f = open(p, "rb"); h.update(f.read()); f.close(); return h.hexdigest()
def read_curve(p):
    rows = list(csv.reader(open(p))); hdr = rows[0]; body = [r for r in rows[1:] if r]
    ix = {h: i for i, h in enumerate(hdr)}
    oofcol = "auc_oof" if "auc_oof" in ix else ("auc_oof_sk" if "auc_oof_sk" in ix else "auc")
    d = dict(idx=[int(r[0]) for r in body], oof=np.array([float(r[ix[oofcol]]) for r in body]))
    if "auc_mean" in ix: d["mean"] = np.array([float(r[ix["auc_mean"]]) for r in body]); d["std"] = np.array([float(r[ix["auc_std"]]) for r in body])
    if "auc_oof_rank" in ix: d["oof_rank"] = np.array([float(r[ix["auc_oof_rank"]]) for r in body])
    o = np.argsort(d["idx"]); return {k: (np.array(v)[o] if k != "idx" else list(np.array(v)[o])) for k, v in d.items()}
def auc(t, s):
    t = np.asarray(t, int); s = np.asarray(s, float); o = np.argsort(s, kind="mergesort"); ss = s[o]; r = np.empty(len(s)); i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    p = t.sum(); n = len(t) - p; return float((r[t == 1].sum() - p * (p + 1) / 2) / (p * n))
def zs(p):
    rows = list(csv.DictReader(open(p))); return rows, auc([int(r["label"]) for r in rows], [float(r["p_yes"]) for r in rows])
f4 = lambda x: "%.4f" % x
def cmp(ref, sav, ans=None, col="oof"):
    a, b = ref[col], sav["oof"]; assert len(a) == len(b) and ref["idx"] == sav["idx"]
    out = dict(n=len(a), maxabs=float(np.max(np.abs(a - b))), m4=sum(f4(x) == f4(y) for x, y in zip(a, b)))
    if "mean" in ref and "mean" in sav:
        out["m4_mean"] = sum(f4(x) == f4(y) for x, y in zip(ref["mean"], sav["mean"])); out["m4_std"] = sum(f4(x) == f4(y) for x, y in zip(ref["std"], sav["std"]))
    if ans is not None: out["above_ref"] = int((a > ans).sum()); out["above_saved"] = int((b > ans).sum())
    return out
fmt = lambda x: "0" if x == 0 else "%.2e" % x
STREAM_F = dict(enc="encoder", proj="proj", llm="llm", ans="ans")
SAVED = {}
for ds, pre in (("pitt", f"{R}/omni_final/omni_pitt"), ("edaic_first30", f"{R}/omni_final/omni_edaic"), ("adress2020", f"{R}/omni_final/omni_adress2020"),
                ("neurovoz", f"{R}/omni_final/omni_neurovoz"), ("edaic_full", f"{R}/edaic_rerun/variants/o25_full")):
    for s, f in STREAM_F.items(): SAVED[(ds, s)] = f"{pre}_{f}_perlayer.csv"
ZS = {"pitt": f"{R}/omni_final/omni_pitt_zeroshot_scores.csv", "edaic_first30": f"{R}/omni_final/omni_edaic_zeroshot_scores.csv",
      "adress2020": f"{R}/omni_final/omni_adress2020_zeroshot_scores.csv", "neurovoz": f"{R}/omni_final/omni_neurovoz_zeroshot_scores.csv",
      "edaic_full": f"{R}/edaic_rerun/variants/o25_full_zeroshot_scores.csv"}
ANS = {ds: zs(p)[1] for ds, p in ZS.items()}
side = dict(saved={f"{k[0]}_{k[1]}": dict(file=v, sha256=fsha(v)) for k, v in SAVED.items()},
            zeroshot={k: dict(file=v, sha256=fsha(v), answer_auc=ANS[k]) for k, v in ZS.items()})
side["answer_edaicfull_states_run"] = zs(f"{R}/edaic_rerun/part16/EDAICFULL/full_zeroshot_scores.csv")[1]
fig = read_curve(f"{R}/overnight/curves/omni/pitt_encoder.csv"); side["pitt_figure_file_vs_omni_final_maxabs"] = float(np.max(np.abs(fig["oof"] - read_curve(SAVED[("pitt", "enc")])["oof"])))
# p_yes of the states used for the other-extraction refits vs the saved zero-shot runs (own lookup by clip name)
def pyes_diff(npz_p, zs_p, order_csv=None):
    zz = {r["clip"]: float(r["p_yes"]) for r in csv.DictReader(open(zs_p))}
    if order_csv: st = {r["clip"]: float(r["p_yes"]) for r in csv.DictReader(open(order_csv))}
    else:
        z = np.load(npz_p); st = dict(zip([str(x) for x in z["name"]], z["p_yes"].astype(float)))
    return float(max(abs(st[k] - zz[k]) for k in zz)), len(zz), sum(k in st for k in zz)
side["pyes_maxdiff"] = dict(
    pitt=pyes_diff(f"{R}/overnight2/part10/o25_pitt_states.npz", ZS["pitt"]), edaic_first30=pyes_diff(f"{R}/overnight2/part10/o25_edaic_states.npz", ZS["edaic_first30"]),
    adress2020=pyes_diff(f"{G}/edaic_rerun/part16/POD4B/states/p16_adress2020_orig_states.npz", ZS["adress2020"]),
    neurovoz=pyes_diff(f"{G}/scores/part20/POD6/B_neurovoz_full_check/omni_nvfull_states.npz", ZS["neurovoz"]),
    edaic_full=pyes_diff(None, ZS["edaic_full"], f"{R}/edaic_rerun/part16/EDAICFULL/full_zeroshot_scores.csv"))
JOB = dict(pitt="pitt", edaic_first30="edaic30", adress2020="adress2020", neurovoz="neurovoz", edaic_full="edaicfull")
def jobname(ds, s): return f"{JOB[ds]}_{'encproj' if s in ('enc', 'proj') else s}"
def env_cpu(pod):
    p = f"{T5}/pull/{pod}/out/env_{pod}.json"
    if not os.path.exists(p): return None
    e = json.load(open(p)); c = e.get("cpu") or e.get("lscpu_model") or ""
    if not c:
        for k, v in e.items():
            if "cpu" in k.lower() and isinstance(v, str) and "EPYC" in v: c = v; break
    m = re.search(r"AMD EPYC \S+", json.dumps(e)); return m.group(0) if m else c
# ---------------- recompute every row of t5_values.tsv
rows = list(csv.DictReader(open(f"{T5}/t5_values.tsv"), delimiter="\t")); res = []
P17 = {d["dataset"]: np.array(d["recomputed_oof"]) for d in json.load(open(f"{R}/edaic_rerun/part17/verify/T5_verify.json"))["extra"]["recompute"] if d.get("matched")}
def fold_counts(native_csv, ds):
    nat = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(native_csv))}; out = []
    for t in ("pod", "mac"):
        f = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(f"{R}/folds/{ds}_groupkfold5_{t}.csv"))}
        out.append(f"{sum(nat[k] == f.get(k) for k in nat)}/{len(nat)}")
    return f"pod file {out[0]}; Mac file {out[1]}"
DSN = [("edaic_first30", "edaic_first30"), ("edaic_full", "edaic_full"), ("adress2020", "adress2020"), ("neurovoz", "neurovoz"), ("pitt", "pitt")]
for r in rows:
    i = r["id"]; claimed = r["value"]; mine = None
    try:
        m = re.match(r"T5_(\w+?)_(enc|proj|llm|ans)_pod(\d+)_(maxabs|match4dp|meanstd_match4dp|layers_above)$", i)
        if m:
            ds, s, n, what = m.groups(); ref = read_curve(f"{T5}/pull/lo-t5-{n}/out/{jobname(ds, s)}_{STREAM_F[s]}_perlayer.csv")
            c = cmp(ref, read_curve(SAVED[(ds, s)]), ANS[ds]); N = c["n"]
            mine = {"maxabs": fmt(c["maxabs"]), "match4dp": f"{c['m4']}/{N}", "meanstd_match4dp": f"{c['m4_mean']}/{N} and {c['m4_std']}/{N}",
                    "layers_above": f"{c['above_ref']}/{N} vs {c['above_saved']}/{N}"}[what]
        elif i.endswith("_enc_refit"):
            mine = "see not_possible check"
        elif i.startswith("T5_folds_"):
            m = re.match(r"T5_folds_(\w+?)_(encproj|llm|ans)_(pod\d+|mac)$", i); job, st, where = m.groups(); ds = "pitt" if job == "pitt" else "edaic"
            nat = f"{T5}/mac_diag/{job}_{st}_folds_native.csv" if where == "mac" else f"{T5}/pull/lo-t5-{where[3:]}/out/{job}_{st}_folds_native.csv"
            mine = fold_counts(nat, ds)
        elif i.endswith("_p17_verifier_refit_explained"):
            ds = "pitt" if "_pitt_" in i else "edaic_first30"; job = jobname(ds, "enc")
            mac = read_curve(f"{T5}/mac_diag/{job}__mac_macfile_enc.csv"); mine = fmt(float(np.max(np.abs(mac["oof"] - P17[ds]))))
        elif i.startswith("T5_diag_"):
            m = re.match(r"T5_diag_(\w+?)_(enc|ans)_(.+)$", i); ds, s, rest = m.groups(); job = jobname(ds, s)
            if rest.endswith("_mac") and rest.startswith("mac_"):
                f = f"{T5}/mac_diag/{job}__mac_{rest.split('_')[1]}_{s}.csv"
            else:
                tag, pod = rest.rsplit("_pod", 1); d = f"{T5}/pull/lo-t5-{pod}/out"
                f = {"exact_script": f"{d}/{job}_{STREAM_F[s]}_perlayer.csv", "vrefit": f"{d}/{job}__vrefit_{s}.csv", "macfile": f"{d}/{job}__macfile_{s}.csv"}.get(tag, f"{d}/{job}__{tag}_{s}.csv")
            c = cmp(read_curve(f), read_curve(SAVED[(ds, s)])); mine = f"{fmt(c['maxabs'])}; {c['m4']}/{c['n']}"
    except Exception as e:
        mine = f"ERROR {e!r}"
    ok = (mine == claimed) or (mine == "see not_possible check" and claimed == "not possible")
    res.append(dict(id=i, claimed=claimed, recomputed=mine, verified="yes" if ok else "no"))
with open(f"{V}/t5_VERIFIED_rows.tsv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "claimed", "recomputed", "verified"], delimiter="\t"); w.writeheader(); w.writerows(res)
nyes = sum(r["verified"] == "yes" for r in res); print(f"ROWS: {nyes} agree of {len(res)}"); [print("  DISAGREE", r) for r in res if r["verified"] != "yes"]
side["rows_recomputed"] = dict(n=len(res), agree=nyes, file=f"{V}/t5_VERIFIED_rows.tsv")
side["task_pod_cpus"] = {f"lo-t5-{n}": env_cpu(f"lo-t5-{n}") for n in range(1, 14)}
# task verifier log and its vrefit (second code path) max diffs on AVX-512 pods
lg = open(f"{T5}/verify/t5_verify.log").read(); side["task_verify_log_total"] = re.findall(r"TOTAL.*", lg)
vr = {}
for f in sorted(glob.glob(f"{T5}/pull/lo-t5-*/out/*__vrefit_*.csv")):
    pod = f.split("/pull/")[1].split("/")[0]; b = os.path.basename(f); job = b.split("__")[0]; s = b.rsplit("_", 1)[1][:-4]
    ds = {v: k for k, v in JOB.items()}[job.rsplit("_", 1)[0]]
    vr[f"{pod}:{b}"] = dict(maxabs=cmp(read_curve(f), read_curve(SAVED[(ds, s)]))["maxabs"], cpu=side["task_pod_cpus"].get(pod))
side["task_vrefit_vs_saved"] = vr
json.dump(side, open(f"{V}/v5_compare_side.json", "w"), indent=1, default=str)
print(json.dumps({k: side[k] for k in ("pitt_figure_file_vs_omni_final_maxabs", "pyes_maxdiff", "answer_edaicfull_states_run", "task_verify_log_total", "task_pod_cpus")}, default=str))
print("answers", {k: round(v, 4) for k, v in ANS.items()})
print("vrefit maxabs by cpu:", sorted({(v["cpu"], fmt(v["maxabs"])) for v in vr.values()}, key=str))
