#!/usr/bin/env python3
"""T5 leftovers (23 Sep): compare every pod refit of the Qwen2.5-Omni per-layer probe curves with the saved curves.
Reads the pulled pod folders (t5/pull/<pod>/out), the saved curves (omni_final, edaic_rerun/variants; read only),
and the Mac diagnostic run (t5/mac_diag). Writes t5_runs.csv, t5_perlayer.csv, t5_values.tsv, t5_sidecar.json."""
import os, sys, json, csv, glob, hashlib, datetime, re
import numpy as np, pandas as pd
from scipy.stats import rankdata

T5 = "<local data dir>/leftovers_23sep/t5"
R = "<local data dir>/release"
OF = f"{R}/omni_final"; VAR = f"{R}/edaic_rerun/variants"
AUDIT = "STATUS_LEDGER section 5 DONE_UNVERIFIED item 1 and section 1 spot check 8 (P17 T5 curve values); audit_p17 item 5 caveat"
FN = {"enc": "encoder_perlayer.csv", "proj": "proj_perlayer.csv", "llm": "llm_perlayer.csv", "ans": "ans_perlayer.csv"}
SAVED_PREFIX = {"pitt": f"{OF}/omni_pitt", "edaic30": f"{OF}/omni_edaic", "adress2020": f"{OF}/omni_adress2020",
                "neurovoz": f"{OF}/omni_neurovoz", "edaicfull": f"{VAR}/o25_full"}
ZS = {"pitt": f"{OF}/omni_pitt_zeroshot_scores.csv", "edaic30": f"{OF}/omni_edaic_zeroshot_scores.csv",
      "adress2020": f"{OF}/omni_adress2020_zeroshot_scores.csv", "neurovoz": f"{OF}/omni_neurovoz_zeroshot_scores.csv",
      "edaicfull": f"{VAR}/o25_full_zeroshot_scores.csv"}
T5KEY = {"pitt": "pitt", "edaic30": "edaic_first30", "adress2020": "adress2020", "neurovoz": "neurovoz", "edaicfull": "edaic_full"}
STATES_CLASS = json.load(open(f"{T5}/inputs/inputs_manifest.json"))["sources"]
NO_STATES = {"adresso": "only a 161-clip P16 POD4B subset exists (curve uses 237 clips)", "pcgita": "no Qwen2.5-Omni PC-GITA states on disk",
             "kcl": "no Qwen2.5-Omni MDVR-KCL states on disk", "edaic_mid300": "podC states were never pulled; none on disk",
             "edaic_mid30": "podB states were never pulled; none on disk"}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

def rank_auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); n1 = int(y.sum()); n0 = len(y) - n1
    r = rankdata(s); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def f4(x): return f"{x:.4f}"

def compare(ref, sv):
    out = {}
    for col in ("auc_oof", "auc_mean", "auc_std"):
        a = ref[col].values.astype(float); b = sv[col].values.astype(float)
        out[f"maxabs_{col}"] = float(np.max(np.abs(a - b)))
        out[f"match4_{col}"] = int(sum(f4(x) == f4(y) for x, y in zip(a, b)))
    out["n_layers"] = int(len(sv)); return out

ANSWER = {ds: rank_auc(pd.read_csv(p)["label"], pd.read_csv(p)["p_yes"]) for ds, p in ZS.items()}

# ---------------- collect every run
envs = {}
for e in glob.glob(f"{T5}/pull/*/out/env_*.json") + glob.glob(f"{T5}/mac_diag/env_mac.json"):
    d = json.load(open(e)); pod = os.path.basename(e)[4:-5]
    model = next((l.split(":", 1)[1].strip() for l in d.get("lscpu", []) if "Model name" in l), "Apple M4 Max" if pod == "mac" else "?")
    arch = sorted({t.get("architecture") for t in d.get("threadpool_info", []) if t.get("architecture")})
    envs[pod] = dict(cpu=model, openblas_arch="/".join(arch) if arch else ("Accelerate (numpy and scipy build config)" if pod == "mac" else "?"),
                     avx512f="avx512f" in (d.get("cpu_flags_simd") or []), numpy=d["numpy"], scipy=d["scipy"], sklearn=d["sklearn"],
                     python=d["python"].split()[0], env_file=e)
runs = []
def add(pod, job, key, variant, path):
    ds = job.split("_")[0]
    svp = f"{SAVED_PREFIX[ds]}_{FN[key]}"
    ref = pd.read_csv(path); sv = pd.read_csv(svp)
    if "layer" not in ref.columns: ref = ref.rename(columns={ref.columns[0]: "layer"})
    assert len(ref) == len(sv), (path, svp)
    c = compare(ref, sv)
    row = dict(pod=pod, job=job, dataset=ds, t5_key=T5KEY[ds], stream=key, variant=variant, states_class=STATES_CLASS[ds]["states_class"],
               refit_file=path, saved_file=svp, **c)
    row.update({k: envs.get(pod, {}).get(k) for k in ("cpu", "openblas_arch", "avx512f", "numpy", "scipy", "sklearn")})
    if key == "enc":
        row["answer_auc"] = ANSWER[ds]
        row["layers_above_saved"] = int((sv["auc_oof"].values > ANSWER[ds]).sum())
        row["layers_above_refit"] = int((ref["auc_oof"].values > ANSWER[ds]).sum())
    runs.append(row)
for pod_dir in sorted(glob.glob(f"{T5}/pull/*/out")):
    pod = pod_dir.split("/")[-2]
    for f in sorted(os.listdir(pod_dir)):
        m = re.match(r"^([a-z0-9]+_(?:encproj|llm|ans))_(encoder|proj|llm|ans)_perlayer\.csv$", f)
        if m:
            key = {"encoder": "enc"}.get(m.group(2), m.group(2)); add(pod, m.group(1), key, "exact_script", f"{pod_dir}/{f}"); continue
        m = re.match(r"^([a-z0-9]+_(?:encproj|llm|ans))__(.+)_(enc|proj|llm|ans)\.csv$", f)
        if m: add(pod, m.group(1), m.group(3), m.group(2), f"{pod_dir}/{f}")
for f in sorted(glob.glob(f"{T5}/mac_diag/*__mac_*_*.csv")):
    m = re.match(r"^([a-z0-9]+_(?:encproj|llm|ans))__(mac_[a-z]+file)_(enc|proj|llm|ans)\.csv$", os.path.basename(f))
    if m: add("mac", m.group(1), m.group(3), m.group(2), f)
RUNS = pd.DataFrame(runs)
RUNS.to_csv(f"{T5}/t5_runs.csv", index=False)

# ---------------- primary pod refit per curve: the exact script on an AVX-512 host (OpenBLAS SkylakeX kernels)
prim = RUNS[(RUNS.variant == "exact_script") & (RUNS.avx512f == True)]
per_layer = []
for (ds, key), g in prim.groupby(["dataset", "stream"]):
    r0 = g.iloc[0]; ref = pd.read_csv(r0.refit_file); sv = pd.read_csv(r0.saved_file)
    for i in range(len(sv)):
        per_layer.append(dict(dataset=ds, t5_key=T5KEY[ds], stream=key, layer=i, pod=r0.pod, saved_auc_oof=sv.auc_oof.iloc[i],
                              refit_auc_oof=ref.auc_oof.iloc[i], abs_diff=abs(ref.auc_oof.iloc[i] - sv.auc_oof.iloc[i]),
                              match_4dp=f4(ref.auc_oof.iloc[i]) == f4(sv.auc_oof.iloc[i]),
                              saved_auc_mean=sv.auc_mean.iloc[i], refit_auc_mean=ref.auc_mean.iloc[i],
                              saved_auc_std=sv.auc_std.iloc[i], refit_auc_std=ref.auc_std.iloc[i]))
pd.DataFrame(per_layer).to_csv(f"{T5}/t5_perlayer.csv", index=False)

# ---------------- per-value table
vals = []
def V(id_, what, value, n, file, closes, note=""):
    vals.append(dict(id=id_, what=what, value=value, n=n, file=file, closes=closes, note=note))
def fmt(x): return f"{x:.2e}" if x != 0 else "0"
for ds in ("pitt", "edaic30", "adress2020", "neurovoz", "edaicfull"):
    sc = STATES_CLASS[ds]["states_class"]
    for key in ("enc", "proj", "llm", "ans"):
        g = prim[(prim.dataset == ds) & (prim.stream == key)]
        if not len(g): continue
        for _, r in g.iterrows():
            tag = f"T5_{T5KEY[ds]}_{key}_{r.pod.replace('lo-t5-', 'pod')}"
            note = ("states from the same extraction run as the saved curve (p_yes equal to 1e-16)" if sc == "same_run" else
                    f"states from a different extraction (max |p_yes diff| {STATES_CLASS[ds]['max_abs_p_yes_diff_vs_zeroshot']:.4f} vs the saved run), so a 4 dp match is not expected")
            cl = AUDIT if sc == "same_run" else "partly: shows the size of the gap only; does not close (the saved curve's own states are not on disk)"
            V(f"{tag}_maxabs", f"{T5KEY[ds]} {key} curve: max |refit auc_oof - saved auc_oof| over {r.n_layers} layers, pod refit (exact curves_from_states.py, {r.cpu}, OpenBLAS {r.openblas_arch}, sklearn {r.sklearn} numpy {r.numpy} scipy {r.scipy})",
              fmt(r.maxabs_auc_oof), r.n_layers, r.refit_file, cl, note)
            V(f"{tag}_match4dp", f"{T5KEY[ds]} {key} curve: layers whose refit auc_oof equals the saved auc_oof at 4 dp",
              f"{r.match4_auc_oof}/{r.n_layers}", r.n_layers, r.refit_file, cl, note)
            V(f"{tag}_meanstd_match4dp", f"{T5KEY[ds]} {key} curve: layers matching at 4 dp on auc_mean and on auc_std (fold mean and fold SD)",
              f"{r.match4_auc_mean}/{r.n_layers} and {r.match4_auc_std}/{r.n_layers}", r.n_layers, r.refit_file, cl, note)
            if key == "enc":
                V(f"{tag}_layers_above", f"{T5KEY[ds]} encoder layers above the zero-shot answer {r.answer_auc:.4f}: refit curve vs saved curve",
                  f"{int(r.layers_above_refit)}/32 vs {int(r.layers_above_saved)}/32", 32, r.refit_file, cl, note)
for k, why in NO_STATES.items():
    V(f"T5_{k}_enc_refit", f"{k} encoder curve refit", "not possible", 32, "", "does not close: " + why, why)

# diagnostics: why the Mac refit failed
def one(pod_pred, job, key, variant):
    g = RUNS[(RUNS.job == job) & (RUNS.stream == key) & (RUNS.variant == variant) & RUNS.pod.map(pod_pred)]
    return g
avx = lambda p: envs.get(p, {}).get("avx512f") is True
zen3 = lambda p: p != "mac" and envs.get(p, {}).get("avx512f") is False
for job, lab in (("pitt_encproj", "pitt"), ("edaic30_encproj", "edaic_first30")):
    for pred, variant, what in ((avx, "macfile", "pod refit with the Mac fold order (sklearn 1.7.2 GroupKFold) on an AVX-512 pod"),
                                (lambda p: p == "mac", "mac_macfile", "Mac refit, Mac fold order (reproduces the P17 verifier's refit)"),
                                (lambda p: p == "mac", "mac_podfile", "Mac refit (sklearn 1.7.2, numpy 2.2.6, Accelerate) with the pod fold order"),
                                (zen3, "exact_script", "exact script on a Zen 3 pod without AVX-512 (OpenBLAS Haswell/Zen kernels)"),
                                (avx, "podfile_f64", "AVX-512 pod, pod folds, features cast to float64"),
                                (avx, "podfile_thr8", "AVX-512 pod, pod folds, 8 BLAS/OpenMP threads instead of 1"),
                                (avx, "podfile_ctHaswell", "AVX-512 pod forced to OpenBLAS Haswell kernels"),
                                (avx, "podfile_ctZen", "AVX-512 pod forced to OpenBLAS Zen kernels"),
                                (avx, "podfile_ctSandybridge", "AVX-512 pod forced to OpenBLAS Sandybridge kernels"),
                                (avx, "podfile_ctPrescott", "AVX-512 pod forced to OpenBLAS Prescott (generic) kernels"),
                                (avx, "podfile_ctSkylakeX", "AVX-512 pod forced to OpenBLAS SkylakeX kernels"),
                                (avx, "vrefit", "independent verifier refit code (fold file, rank AUC) on the same AVX-512 pod")):
        g = one(pred, job, "enc", variant)
        for _, r in g.iterrows():
            V(f"T5_diag_{lab}_enc_{variant}_{r.pod.replace('lo-t5-', 'pod')}", f"{lab} encoder curve, {what}: max |diff| vs saved; layers matching at 4 dp",
              f"{fmt(r.maxabs_auc_oof)}; {r.match4_auc_oof}/32", 32, r.refit_file, "diagnosis for " + AUDIT, f"{r.cpu}, OpenBLAS {r.openblas_arch}")
# fold checks
for f in sorted(glob.glob(f"{T5}/pull/*/out/*_fold_check.json")) + sorted(glob.glob(f"{T5}/mac_diag/*_fold_check.json")):
    d = json.load(open(f)); pod = "mac" if "mac_diag" in f else f.split("/")[-3]
    if d["job"].split("_")[0] not in ("pitt", "edaic30"): continue
    V(f"T5_folds_{d['job']}_{pod.replace('lo-t5-', 'pod')}", f"{d['job']}: GroupKFold(5) as run on {pod} ({envs.get(pod, {}).get('sklearn')}) vs released fold files, clips with the same fold",
      f"pod file {d['vs_pod']['fold_equal']}/{d['n']}; Mac file {d['vs_mac']['fold_equal']}/{d['n']}", d["n"], f, "diagnosis for " + AUDIT,
      f"numpy default argsort equals stable argsort on the speaker clip counts here: {d['argsort_default_equals_stable']}")
VT = pd.DataFrame(vals); VT.to_csv(f"{T5}/t5_values.tsv", sep="\t", index=False)

# the P17 verifier's own Mac refit, for the record
V17 = json.load(open(f"{R}/edaic_rerun/part17/verify/T5_verify.json"))
rc = {r["dataset"]: r for r in V17["extra"]["recompute"]}
mac_vs_p17 = {}
for job, ds in (("pitt_encproj", "pitt"), ("edaic30_encproj", "edaic_first30")):
    p = f"{T5}/mac_diag/{job}__mac_macfile_enc.csv"
    if os.path.exists(p) and ds in rc and rc[ds].get("recomputed_oof"):
        mac_vs_p17[ds] = float(np.max(np.abs(pd.read_csv(p).auc_oof.values - np.array(rc[ds]["recomputed_oof"]))))
        V(f"T5_diag_{ds}_enc_p17_verifier_refit_explained", f"{ds} encoder: max |Mac refit with the Mac fold order - the curve the P17 verifier recomputed on the Mac (T5_verify.json extra.recompute)|",
          fmt(mac_vs_p17[ds]), 32, p, "diagnosis for " + AUDIT, "the P17 refit used sklearn 1.7.2 GroupKFold, whose tie order differs from the pods'")
VT = pd.DataFrame(vals); VT.to_csv(f"{T5}/t5_values.tsv", sep="\t", index=False)

pods = [l.split() for l in open(f"{T5}/pods.txt") if l.strip()]
side = dict(task="leftovers_23sep t5: refit of the Qwen2.5-Omni per-layer probe curves (T5 omitted-dataset curves) on Linux CPU pods",
            date=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), command=" ".join(["/usr/local/bin/python3"] + sys.argv),
            closes=AUDIT, script_that_made_the_saved_curves=f"{R}/scripts/curves_from_states.py",
            script_sha256=sha(f"{R}/scripts/curves_from_states.py"),
            script_note="byte identical to the heredoc run on the H200 pod gv9jvm57b21mcz on 18 Sep (4 copies in release/c_conversation.txt lines 240964, 249629, 258264, 266899) and to release_public/scripts/core/curves_from_states.py; the pod ran it via run_all_omni.sh with OMP_NUM_THREADS=1",
            saved_curves={f"{ds}_{k}": dict(file=f"{SAVED_PREFIX[ds]}_{FN[k]}", sha256=sha(f"{SAVED_PREFIX[ds]}_{FN[k]}"))
                          for ds in SAVED_PREFIX for k in FN if os.path.exists(f"{SAVED_PREFIX[ds]}_{FN[k]}")},
            states=STATES_CLASS, inputs_manifest=f"{T5}/inputs/inputs_manifest.json",
            fold_files={f: sha(f) for f in sorted(glob.glob(f"{R}/folds/*_groupkfold5_pod.csv") + glob.glob(f"{R}/folds/*_groupkfold5_mac.csv"))
                        if os.path.basename(f).split("_")[0] in ("pitt", "edaic", "adress2020", "neurovoz")},
            answers_recomputed={T5KEY[k]: v for k, v in ANSWER.items()}, pods=pods, pod_env=envs,
            p17_verifier_mac_refit_reproduced_by_mac_macfile_run_maxabs=mac_vs_p17,
            outputs=[f"{T5}/{x}" for x in ("t5_values.tsv", "t5_runs.csv", "t5_perlayer.csv", "t5_sidecar.json")],
            rules="4 dp match: the two values formatted with 4 decimals are the same string. Max |diff| over layers on auc_oof (pooled out-of-fold AUC, the column Figure 1 plots).",
            groupkfold_source={"sklearn 1.9.1 on the pods (inspect.getsource, GroupKFold._iter_test_indices)": 'indices = np.argsort(n_samples_per_group, kind="stable")[::-1]',
                               "sklearn 1.7.2 on the Mac (same call)": "indices = np.argsort(n_samples_per_group)[::-1]",
                               "numpy default argsort equals stable argsort on the speaker clip counts": "False on the Mac and False on every pod, so the tie order comes from the sklearn version, not from numpy or the CPU"},
            findings=["With the pod fold order and one BLAS thread, the unchanged curves_from_states.py reproduces every saved Pitt and E-DAIC first-30 s curve (encoder, projector, LM, answer) with max |diff| 0 on AVX-512 AMD EPYC hosts where OpenBLAS picks SkylakeX kernels (Zen 4 9654, 9754, 9655P; Zen 5 9965, 9575F).",
                      "The P17 Mac refit (0/32 at 4 dp) used sklearn 1.7.2, whose GroupKFold tie order differs from sklearn 1.9.1; the Mac-order refit reproduces the P17 verifier's curve to 1.1e-16 and gives the audit's 0.0534 and 0.0877.",
                      "With the right folds the remaining differences come from BLAS arithmetic: Zen 3 (OpenBLAS Haswell kernels) 6.8e-4 and 5.1e-4, Mac Accelerate 5.6e-4 and 8.7e-4, forcing Haswell/Zen/Sandybridge/Prescott kernels on an AVX-512 host 4.8e-4 to 6.8e-4, float64 features 4.7e-4 and 1.1e-3, 8 BLAS threads 5.6e-4 on Pitt (0 on E-DAIC).",
                      "ADReSS-2020, NeuroVoz and E-DAIC full have only states from a different extraction on disk (p_yes differs by up to 0.038, 0.047, 0.030); their refits differ by 2.9e-3, 1.1e-3, 3.0e-3, and the layer counts above the answer are unchanged (32, 32, 0).",
                      "ADReSSo, PC-GITA, MDVR-KCL, E-DAIC middle 300 s and middle 30 s have no matching states on disk, so their curves cannot be refitted."],
            run_notes=["lo-t5-11 and lo-t5-12 were deleted at 22:17 UTC before use: the Mac upload link (about 2 MB/s shared with other jobs) made separate verifier pods slow, so the independent refit code also ran on each main pod.",
                       "lo-t5-1, lo-t5-3 and lo-t5-4 landed on Zen 3 hosts without AVX-512; their runs are kept as the host-sensitivity rows. The AVX-512 reruns (lo-t5-9, lo-t5-10, lo-t5-13) got their states pod to pod (sha256 checked).",
                       "On lo-t5-4 and lo-t5-7 the driver file was replaced while it ran; bash then stopped with a syntax error after the variants. The same-pod verifier refit and ALL_DONE were run by hand; every result file was already written.",
                       "Pod cost: 13 pods, about 186 pod-minutes at $0.96/h (lo-t5-13 $1.12/h), about $3.05 in total, from create times and watchdog delete times."])
json.dump(side, open(f"{T5}/t5_sidecar.json", "w"), indent=1, default=str)
print(RUNS[RUNS.variant.isin(["exact_script", "macfile", "mac_macfile", "mac_podfile", "vrefit"])][["pod", "job", "stream", "variant", "cpu", "openblas_arch", "maxabs_auc_oof", "match4_auc_oof", "n_layers"]].to_string())
print(len(VT), "values written;", "mac vs P17 verifier refit:", mac_vs_p17)
