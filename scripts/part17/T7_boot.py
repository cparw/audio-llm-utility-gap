"""T7 step 2: speaker bootstrap for the E-DAIC 300 s window probe, mean-of-five convention.

REPRODUCTION GATE FAILED (see T7_scratch/T7_gate_*.json): no Mac run reproduces the pod's
per-repeat AUCs at 4 dp. So NOTHING here is an interval for the published 0.7001 / 0.5884.
Rows written here:
  ZS      zero-shot answer AUC (p_yes_answer from the published per-clip file)
  NOTUSED AUC of the averaged OOF, recomputed from the PUBLISHED per-clip file
          (0.7088 / 0.5916 / ...), single and paired. Checks this bootstrap against probe_boot.py.
  PROXY_D Mac vectors with the POD's recorded layer picks (closest reconstruction).
  PROXY_A Mac vectors with layers re-selected on this Mac (fully independent rerun).
Bootstrap: 2000 draws, fresh numpy.random.default_rng(0) per cell, speakers resampled with
replacement (same draw code as probe_boot.py), percentile 2.5/97.5. PAIRED: one draw per
replicate, all five per-repeat AUCs and the answer AUC recomputed inside it.
"""
import csv, json, hashlib, os, sys, numpy as np
sys.path.insert(0, "<local data dir>/release/edaic_rerun/part17")
from T7_probe import auc_rank, P16, P17, SCR

NB = 2000
STREAMS = [("llm", "LM probe"), ("enc", "encoder probe"), ("ans", "answer-state probe"), ("proj", "projector probe")]
WIN = "E-DAIC 300 s window (audio truncated at 300 s by Qwen2_5OmniProcessor; 217/275 windows cut)"

perclip = list(csv.DictReader(open(f"{P16}/edaic_full_perclip.csv")))
pid = [r["pid"] for r in perclip]
y = np.array([int(r["label"]) for r in perclip])
pz = np.array([float(r["p_yes_answer"]) for r in perclip])
pub_mean_oof = {k: np.array([float(r[f"oof_{k}"]) for r in perclip]) for k, _ in STREAMS}
spk = np.array(pid)  # one clip per speaker; speaker id == pid (checked in full_zeroshot_scores.csv)
pub = json.load(open(f"{P16}/edaic_full_nested_repeats.json"))

V = {}
for tag, fn in [("D", "T7_oof_repeats_D_mac_podpicks.npz"), ("A", "T7_oof_repeats_A.npz")]:
    z = np.load(f"{SCR}/{fn}")
    assert [c.replace(".wav", "") for c in z["clip"]] == pid and np.array_equal(z["label"], y)
    V[tag] = {k: z[f"{k}_oof_repeats"] for k, _ in STREAMS}
    V[tag]["_file"] = f"{SCR}/{fn}"

uspk = np.array(sorted(set(spk)))
idx_of = {s: i for i, s in enumerate(spk)}


def boot(vecs, ref=None):
    """vecs: list of score vectors; statistic = mean of their AUCs. ref: answer vector for paired diff."""
    rng = np.random.default_rng(0)
    va, vd = [], []
    for _ in range(NB):
        pick = rng.choice(len(uspk), size=len(uspk), replace=True)
        ii = np.array([idx_of[uspk[p]] for p in pick])
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy): continue
        a = float(np.mean([auc_rank(yy, v[ii]) for v in vecs])); va.append(a)
        if ref is not None: vd.append(a - auc_rank(yy, ref[ii]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    out = dict(usable=len(va), lo=q(va)[0], hi=q(va)[1])
    if ref is not None: out.update(dlo=q(vd)[0], dhi=q(vd)[1])
    return out


rows = []
def add(rid, what, value, lo, hi, src, ez=""):
    r = dict(id=rid, what=what, value=round(value, 4), lo="" if lo is None else round(lo, 4),
             hi="" if hi is None else round(hi, 4), n=len(y), n_spk=len(uspk), excludes_zero=ez,
             usable_draws=NB, file=src)
    rows.append(r)
    print(f"{rid:40s} {value:.4f} [{r['lo']}, {r['hi']}] {('excl0 ' + ez) if ez else ''}  n={len(y)} n_spk={len(uspk)}  {src}", flush=True)
    return r


PC = f"{P16}/edaic_full_perclip.csv"
a_zs = auc_rank(y, pz); b = boot([pz])
add("T7_zeroshot_answer", f"Qwen2.5-Omni zero-shot answer AUC, {WIN}", a_zs, b["lo"], b["hi"], PC)

# published values without an interval (gate failed)
for k, lab in STREAMS:
    add(f"T7_published_{k}_meanof5", f"{lab}, mean of five per-repeat AUCs (pod run, published {pub[k]['per_repeat']}); "
        f"NO INTERVAL: per-repeat vectors not saved and not reproducible on Mac, {WIN}",
        pub[k]["mean_of_repeat_aucs"], None, None, f"{P16}/edaic_full_nested_repeats.json")

# not-used convention, from the published per-clip file
for k, lab in STREAMS:
    v = pub_mean_oof[k]; b = boot([v], pz)
    add(f"T7_notused_{k}_aucofmeanoof", f"NOT USED: {lab} AUC of OOF averaged over 5 repeats, {WIN}", auc_rank(y, v), b["lo"], b["hi"], PC)
    ez = "yes" if (b["dlo"] > 0 or b["dhi"] < 0) else "no"
    add(f"T7_notused_{k}_aucofmeanoof_minus_answer", f"NOT USED: {lab} (AUC of averaged OOF) minus zero-shot answer, paired, {WIN}",
        auc_rank(y, v) - a_zs, b["dlo"], b["dhi"], PC, ez)

# proxies
for tag, desc in [("D", "PROXY, NOT the paper's value: Mac rerun with the pod's recorded layer picks"),
                  ("A", "PROXY, NOT the paper's value: Mac rerun, layers re-selected on Mac")]:
    for k, lab in STREAMS:
        vecs = list(V[tag][k]); per = [auc_rank(y, v) for v in vecs]; m = float(np.mean(per))
        b = boot(vecs, pz)
        add(f"T7_PROXY{tag}_{k}_meanof5", f"{desc}; {lab} mean of five per-repeat AUCs "
            f"{[round(x, 4) for x in per]} (published {pub[k]['per_repeat']} mean {pub[k]['mean_of_repeat_aucs']}), {WIN}",
            m, b["lo"], b["hi"], V[tag]["_file"])
        ez = "yes" if (b["dlo"] > 0 or b["dhi"] < 0) else "no"
        add(f"T7_PROXY{tag}_{k}_meanof5_minus_answer", f"{desc}; {lab} mean of five per-repeat AUCs minus zero-shot answer 0.8278, paired, {WIN}",
            m - a_zs, b["dlo"], b["dhi"], V[tag]["_file"], ez)

# ---------- outputs ----------
with open(f"{P17}/T7_edaic300_meanof5.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "what", "value", "lo", "hi", "n", "n_spk", "excludes_zero", "usable_draws", "file"])
    w.writeheader(); [w.writerow(r) for r in rows]
os.makedirs(f"{P17}/rows", exist_ok=True)
with open(f"{P17}/rows/T7.tsv", "w", newline="") as fh:
    fh.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n")
    for r in rows:
        fh.write("\t".join(str(r[c]) for c in ["id", "what", "value", "lo", "hi", "n", "n_spk", "file"]) + "\n")
for tag, fn in [("D", "T7_edaic300_meanof5_perclip.csv"), ("A", "T7_edaic300_meanof5_perclip_macselected.csv")]:
    with open(f"{P17}/{fn}", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = [f"oof_{k}_r{r}" for k in ("enc", "llm", "ans", "proj") for r in range(5)]
        w.writerow(["pid", "label", "p_yes_answer"] + cols)
        for i in range(len(y)):
            w.writerow([pid[i], y[i], repr(float(pz[i]))] + [repr(float(V[tag][k][r][i])) for k in ("enc", "llm", "ans", "proj") for r in range(5)])


def sha1mb(p):
    with open(p, "rb") as fh: return hashlib.sha256(fh.read(1 << 20)).hexdigest()


srcs = [f"{P16}/full_zeroshot_scores.csv", PC, f"{P16}/edaic_full_nested_repeats.json", f"{P16}/probe_boot.py",
        f"{P16}/edaic_full_perclip.json", f"{P16}/probe2.log",
        "<local data dir>/paper work/paper1_local_runs/part16_EDAICFULL_states/edaicfull_shards.tgz",
        V["D"]["_file"], V["A"]["_file"],
        f"{SCR}/T7_oof_repeats_B_miniconda_sk180_np244_accelerate.npz", f"{SCR}/T7_oof_repeats_C_tfarm_sk161_np1264_openblas.npz",
        f"{P17}/T7_probe.py", f"{P17}/T7_forced_picks.py", f"{P17}/T7_boot.py"]
gates = {t: json.load(open(f"{SCR}/T7_gate_{t}.json")) for t in
         ["A", "B_miniconda_sk180_np244_accelerate", "C_tfarm_sk161_np1264_openblas", "D_mac_podpicks"]}
side = dict(
    task="PART17 T7: E-DAIC 300 s window probe, mean of five per-repeat AUCs, paired vs zero-shot answer 0.8278",
    date="2026-09-23",
    REPRODUCTION_GATE="FAILED. No Mac variant reproduces the pod per-repeat AUCs at 4 dp. Intervals here are NOT intervals for 0.7001 / 0.5884.",
    gate_variants=dict(
        A="/usr/local/bin/python3 3.10, sklearn 1.7.2, numpy 2.2.6 (Accelerate), scipy 1.15.3; layers re-selected",
        B_miniconda_sk180_np244_accelerate="/opt/miniconda3/bin/python3, sklearn 1.8.0, numpy 2.4.4 (Accelerate), scipy 1.17.1; layers re-selected",
        C_tfarm_sk161_np1264_openblas="miniforge3/envs/tf_arm python 3.9.21, sklearn 1.6.1, numpy 1.26.4 (OpenBLAS), scipy 1.13.1; layers re-selected",
        D_mac_podpicks="as A, but each outer fold uses the layer the pod picked (edaic_full_nested_repeats.json layers_picked)"),
    pod_env="POD2 dtz7yg0r57zikq H200, image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (x86_64), python 3.12.3, sklearn 1.9.1 (from part16/POD2/lora_pitt_oof.sidecar.json); pod numpy/BLAS version not recorded",
    gate_results=gates,
    cause=("Fold assignment is NOT the cause: GroupKFold(shuffle=True) takes the rng.permutation branch, never the argsort-of-group-sizes branch "
           "(checked in sklearn 1.7.2 and 1.8.0 local source and the 1.9.1 tag on GitHub), so the stable tie-break emulation does not apply. "
           "Folds are identical across A/B/C (5x55 per repeat), and 24/25 LM and 25/25 encoder layer picks match the pod. "
           "Cause is LogisticRegression lbfgs numerics (stops at tol=1e-4 on 220x3584 standardized data): with the pod's own layer picks (D) "
           "per-repeat AUCs still differ by up to 0.0007 (LM), 0.0004 (enc). Two inner layer choices are near-ties that flip: "
           "LM repeat 1 outer fold 1, Mac layer 14 inner AUC 0.80183 vs pod layer 18 0.80137 (gap 0.00046); "
           "answer-state repeat 0 outer fold 1, Mac layer 19 0.82740 vs pod layer 28 0.82631 (gap 0.00109). All three Mac stacks flip the same way."),
    pod_saved_per_repeat_vectors=("probe_boot.py wrote /workspace/edaicfull/out/edaic_full_oof_repeats.npz on the pod "
                                  "(np.savez_compressed with <stream>_oof_repeats). It is not in edaicfull_shards.tgz and was not found on this Mac or the G-Drive."),
    truncation=("Every value inherits the 300 s truncation: Qwen2_5OmniProcessor (WhisperFeatureExtractor chunk_length=300 s) cuts audio at 300 s before the "
                "audio tower; 217 of 275 windows truncated. Describe as the 300 s window, not the full interview."),
    n=int(len(y)), n_speakers=int(len(uspk)), n_positive=int(y.sum()),
    bootstrap=dict(draws=NB, seed="numpy.random.default_rng(0), fresh per cell", unit="speaker (== clip, one clip per speaker), resampled with replacement",
                   draw_code="identical to probe_boot.py boot(): rng.choice(len(sorted unique speakers), size=n, replace=True)",
                   paired="one speaker draw per replicate; all five per-repeat AUCs averaged and the answer AUC recomputed inside it",
                   percentiles=[2.5, 97.5], auc="rank formula, ties averaged, recomputed from per-clip scores"),
    source_sha256_first1mb={p: sha1mb(p) for p in srcs if os.path.exists(p)},
    outputs=dict(summary=f"{P17}/T7_edaic300_meanof5.csv", perclip_podpicks_D=f"{P17}/T7_edaic300_meanof5_perclip.csv",
                 perclip_macselected_A=f"{P17}/T7_edaic300_meanof5_perclip_macselected.csv", fragment=f"{P17}/rows/T7.tsv"),
    exact_command=("cd part17/T7_scratch && tar -xzf '<local data dir>/paper work/paper1_local_runs/part16_EDAICFULL_states/edaicfull_shards.tgz' ; "
                   "cd part17 && /usr/local/bin/python3 T7_probe.py proj && /usr/local/bin/python3 -W ignore T7_probe.py enc llm ans  "
                   "(variant A; run before the T7_TAG switch existed, outputs renamed T7_oof_repeats.npz -> T7_oof_repeats_A.npz, T7_gate.json -> T7_gate_A.json) ; "
                   "T7_TAG=B_miniconda_sk180_np244_accelerate /opt/miniconda3/bin/python3 -W ignore T7_probe.py enc proj llm ans ; "
                   "T7_TAG=C_tfarm_sk161_np1264_openblas <local data dir>/miniforge3/envs/tf_arm/bin/python -W ignore T7_probe.py enc proj llm ans ; "
                   "/usr/local/bin/python3 -W ignore T7_forced_picks.py ; /usr/local/bin/python3 T7_scratch/T7_margin.py ; /usr/local/bin/python3 T7_boot.py"),
)
json.dump(side, open(f"{P17}/T7_edaic300_meanof5.json", "w"), indent=1)
print("wrote", f"{P17}/T7_edaic300_meanof5.csv", f"{P17}/T7_edaic300_meanof5.json", f"{P17}/rows/T7.tsv")
