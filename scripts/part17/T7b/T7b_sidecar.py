"""Writes the T7b sidecar (sources, sha256, versions, commands, gate, diagnostics) and the rows fragment."""
import hashlib, json, os, shutil
import numpy as np

T = "<local data dir>/release/edaic_rerun/part17/T7b"
P17 = "<local data dir>/release/edaic_rerun/part17"
P16 = "<local data dir>/release/edaic_rerun/part16/EDAICFULL"
TGZ = "<local data dir>/paper work/paper1_local_runs/part16_EDAICFULL_states/edaicfull_shards.tgz"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def auc_rank(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float); sv = s[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j+1] == sv[i]: j += 1
        r[o[i:j+1]] = (i + j) / 2.0 + 1.0; i = j + 1
    return float((r[y == 1].sum() - n1*(n1+1)/2.0) / (n1*n0))


gate = json.load(open(f"{T}/T7b_gate.json"))
run = json.load(open(f"{T}/T7b_boot_run.json"))
pod_env = json.load(open(f"{T}/pod_env.json"))
pub = json.load(open(f"{P16}/edaic_full_nested_repeats.json"))

diag = {}
for tag, what in [("diag_f64", "same pod, same sklearn 1.9.1, input cast to float64 before the probe (one-line change)"),
                  ("diag_omp1", "same pod, unmodified float32 script, OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1")]:
    j = json.load(open(f"{T}/{tag}/edaic_full_nested_repeats.json"))
    diag[tag] = dict(what=what, npz=f"{T}/{tag}/edaic_full_oof_repeats.npz", npz_sha256=sha256(f"{T}/{tag}/edaic_full_oof_repeats.npz"),
                     script=f"{T}/{tag}/probe_boot_{tag}.py", log=f"{T}/logs/{tag}.log")
    for k in ("llm", "enc", "ans", "proj"):
        mism = [(i // 5, i % 5, a, b) for i, (a, b) in enumerate(zip(j[k]["layers_picked"], pub[k]["layers_picked"])) if a != b]
        diag[tag][k] = dict(per_repeat=j[k]["per_repeat"], mean_of_repeat_aucs=j[k]["mean_of_repeat_aucs"],
                            per_repeat_match_published=j[k]["per_repeat"] == pub[k]["per_repeat"],
                            layer_mismatches_rep_fold_this_published=mism)
    diag[tag]["reproduces_published"] = all(diag[tag][k]["per_repeat_match_published"] and not diag[tag][k]["layer_mismatches_rep_fold_this_published"]
                                            for k in ("llm", "enc", "ans", "proj"))

sources = {
    "shard_tarball": TGZ, "probe_script_original": f"{P16}/probe_boot.py", "zeroshot_scores_rows_order": f"{P16}/full_zeroshot_scores.csv",
    "published_nested_repeats_json": f"{P16}/edaic_full_nested_repeats.json", "published_perclip_csv_answer_p_yes": f"{P16}/edaic_full_perclip.csv",
    "published_perclip_json": f"{P16}/edaic_full_perclip.json", "published_probe_log": f"{P16}/probe2.log",
    "mac_attempt_probe": f"{P17}/T7_probe.py", "mac_attempt_boot": f"{P17}/T7_boot.py", "mac_attempt_sidecar": f"{P17}/T7_edaic300_meanof5.json",
}
outputs = {
    "per_repeat_oof_npz": f"{T}/pod_rerun_out/edaic_full_oof_repeats.npz",
    "rerun_nested_repeats_json": f"{T}/pod_rerun_out/edaic_full_nested_repeats.json",
    "rerun_perclip_csv_mean_oof": f"{T}/pod_rerun_out/edaic_full_perclip.csv",
    "rerun_gap_csv_notused_convention": f"{T}/pod_rerun_out/edaic_full_gap.csv",
    "perclip_csv_per_repeat": f"{T}/T7b_edaic300_meanof5_perclip.csv",
    "summary_csv": f"{T}/T7b_edaic300_meanof5.csv", "gate_json": f"{T}/T7b_gate.json", "boot_run_json": f"{T}/T7b_boot_run.json",
    "pod_env_json": f"{T}/pod_env.json", "fragment": f"{P17}/rows/T7b.tsv",
    "probe_log": f"{T}/logs/runA_default.log", "boot_log": f"{T}/logs/T7b_boot.log",
    "boot_script": f"{T}/T7b_boot.py", "sidecar_script": f"{T}/T7b_sidecar.py",
}
os.makedirs(f"{P17}/rows", exist_ok=True)
shutil.copyfile(f"{T}/T7b.tsv", f"{P17}/rows/T7b.tsv")

z = np.load(outputs["per_repeat_oof_npz"])
per = {k: [auc_rank(z["label"], v) for v in z[f"{k}_oof_repeats"]] for k in ("llm", "enc", "ans", "proj")}

side = dict(
    task="PART17 T7b: E-DAIC 300 s window probe, mean of five per-repeat AUCs (paper convention), 2000-draw speaker interval, paired vs zero-shot answer 0.8278",
    date="2026-09-23",
    REPRODUCTION_GATE=("PASSED. The unmodified part16 probe_boot.py rerun on Linux x86_64 with sklearn 1.9.1 / numpy 2.1.2 / scipy 1.18.1 reproduces every "
                       "published per-repeat AUC and all 25 layer picks for enc, proj, llm, ans. Its edaic_full_nested_repeats.json and edaic_full_perclip.csv "
                       "are byte-identical to the published files (sha256 817e9d58... and 7566ae55...), and its log is identical to probe2.log from the stage lines on."),
    gate=dict(passed=gate["passed"], checks=gate["checks"],
              per_stream={k: {kk: gate[k][kk] for kk in ("per_repeat_4dp_rerun", "per_repeat_4dp_published", "per_repeat_match_4dp", "layers_match",
                                                          "mean_of_repeat_aucs_full_precision", "mean_of_4dp_repeat_aucs_rounded", "mean_of_repeat_aucs_published",
                                                          "max_abs_mean_oof_vs_published_perclip", "reproduced")} for k in ("llm", "enc", "ans", "proj")}),
    per_repeat_auc_full_precision=per,
    results=dict(answer=dict(auc=run["answer_auc"], ci=run["answer_ci"]), used_mean_of_five=run["results"], not_used_auc_of_mean_oof=run["notused"]),
    paste_lines=run["paste_lines"], side_by_side=run["side_by_side"],
    bootstrap=dict(draws=2000, seed="numpy.random.default_rng(0), fresh per cell",
                   unit="speaker (one clip per speaker, 275 speakers), resampled with replacement; draw code identical to probe_boot.py boot()",
                   mean_of_five="inside every draw the five per-repeat AUCs are recomputed on the drawn speakers and averaged",
                   paired="one speaker draw per replicate; mean of five recomputed and answer AUC recomputed inside the same draw; statistic = difference",
                   percentiles=[2.5, 97.5], auc="rank AUC (ties averaged), same function as probe_boot.py",
                   check_not_used=("the AUC-of-averaged-OOF cells computed by this bootstrap reproduce probe_boot.py's published intervals exactly "
                                   "(LM 0.7088 [0.6232, 0.7874], paired -0.1190 [-0.1995, -0.0478]; enc 0.5916 [0.5110, 0.6706], paired -0.2361 [-0.3272, -0.1394])"),
                   check_platform="T7b_boot.py rerun on the Mac (/usr/local/bin/python3) on the synced npz gives byte-identical summary csv, fragment, per-clip csv and gate json"),
    pod=dict(runpod_id="eyawl43rjukfkj", ssh="root@205.196.17.122 -p 12029", env=pod_env,
             blas_threads=("OpenBLAS 0.3.27 (scipy-openblas64, SkylakeX kernels) 64 threads in the main process; joblib reports cpu_count 14 on this pod, "
                           "so the 16 loky workers doing the inner layer scoring run with 1 BLAS thread each (checked with threadpoolctl inside a worker)"),
             not_terminated="pod left running for the verifier; /workspace/T7B_DONE touched"),
    diagnostics=dict(note=("Not needed for the gate (default environment reproduced). Run to pin the cause of the Mac mismatch. "
                           "float64 input on the SAME pod and sklearn does NOT reproduce and flips exactly the two near-tie layer picks the Mac runs flipped "
                           "(LM repeat 1 fold 1: 18 -> 14; answer-state repeat 0 fold 1: 28 -> 19). So the float32 fit path is the cause. "
                           "Single-thread BLAS keeps all 100 layer picks and the encoder per-repeat AUCs, but moves some LM / answer-state / projector per-repeat AUCs "
                           "in the 4th decimal (LM mean still 0.7001, enc 0.5884): the published numbers are tied to multi-threaded BLAS in the final outer fit."),
                     runs=diag),
    commands=[
        "scp probe_boot.py full_zeroshot_scores.csv edaic_full_nested_repeats.json edaic_full_perclip.csv edaic_full_perclip.json probe2.log edaicfull_shards.tgz -> pod:/workspace/ref/",
        "pod: python3 -m pip install --break-system-packages -q scikit-learn numpy scipy pandas",
        "pod: cd /workspace/edaicfull/out && tar -xzf /workspace/ref/edaicfull_shards.tgz && cp /workspace/ref/full_zeroshot_scores.csv . && cp /workspace/ref/probe_boot.py /workspace/edaicfull/",
        "pod: cd /workspace/edaicfull && python3 -W ignore probe_boot.py > /workspace/t7b/runA_default.log   (unmodified script, default threads, 74 s wall)",
        ("pod: cd /workspace/t7b && python3 T7b_boot.py --npz /workspace/edaicfull/out/edaic_full_oof_repeats.npz --rerun_json /workspace/edaicfull/out/edaic_full_nested_repeats.json "
         "--ref_json /workspace/ref/edaic_full_nested_repeats.json --ref_perclip /workspace/ref/edaic_full_perclip.csv --out_dir /workspace/t7b/out "
         f"--file_label {T}/T7b_edaic300_meanof5_perclip.csv"),
        "pod diagnostics: /workspace/diag_f64/probe_boot_diag_f64.py (D path + .astype(np.float64)); OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 /workspace/diag_omp1/probe_boot_diag_omp1.py (D path only)",
        "scp pod outputs -> part17/T7b/ ; Mac: /usr/local/bin/python3 T7b_boot.py (same args, Mac paths, out to scratch) -> byte-identical ; /usr/local/bin/python3 T7b_sidecar.py",
    ],
    n=275, n_speakers=275, n_positive=66,
    window="E-DAIC 300 s window (same window as part16 EDAICFULL; see T7 sidecar for the truncation note)",
    sources_sha256={k: dict(path=p, sha256=sha256(p)) for k, p in sources.items() if os.path.exists(p)},
    outputs_sha256={k: dict(path=p, sha256=sha256(p)) for k, p in outputs.items() if os.path.exists(p) and k != "sidecar_script"},
)
json.dump(side, open(f"{T}/T7b_edaic300_meanof5.json", "w"), indent=1)
print("wrote", f"{T}/T7b_edaic300_meanof5.json", f"{P17}/rows/T7b.tsv")
for k in ("llm", "enc"):
    print(k, diag["diag_f64"][k]["per_repeat"], diag["diag_f64"][k]["layer_mismatches_rep_fold_this_published"],
          diag["diag_omp1"][k]["per_repeat"], diag["diag_omp1"][k]["layer_mismatches_rep_fold_this_published"])
print({t: diag[t]["reproduces_published"] for t in diag})
