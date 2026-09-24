import json, hashlib, glob, os, datetime
LO = "<local data dir>/leftovers_23sep"; V = f"{LO}/verify/t5v"
def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
man = {}
for m in glob.glob(f"{V}/inputs/manifest_*.json"): man.update(json.load(open(m)))
side = json.load(open(f"{V}/v5_compare_side.json")); det = json.load(open(f"{V}/t5_VERIFIED_details.json")); np_ = json.load(open(f"{V}/v5_notpossible.json"))
created = {}
for n in range(1, 10):
    d = json.load(open(f"{V}/pods_json/create_{n}.json")); created[f"lo-v5-{n}"] = dict(id=d["id"], costPerHr=d["costPerHr"], created=d.get("createdAt"))
out = dict(task="leftovers_23sep verify t5 (independent verifier)", date=datetime.datetime.utcnow().isoformat() + "Z",
  outputs=[f"{LO}/verify/t5_VERIFIED.tsv", f"{V}/t5_VERIFIED_rows.tsv", f"{LO}/verify/t5_VERIFIED_sidecar.json"],
  method=["File level: every one of the 106 rows of t5/t5_values.tsv recomputed from the raw per-layer files the task pulled (own csv reader, own saved-curve lookup, own 4 dp string rule, answers from the zero-shot score files with own midrank AUC); 106 of 106 agree (t5v/t5_VERIFIED_rows.tsv).",
          "Own refit: states subsets built by my own prep from the ORIGINAL sources (not the t5 inputs), own refit code (own GroupKFold reimplementation with stable tie order, checked equal to sklearn 1.9.1 GroupKFold and to folds/*_pod.csv clip for clip; per layer StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced'); own midrank AUC, plus sklearn roc_auc_score on the pooled vector for the bit-level check), 1 BLAS thread, on 7 RunPod CPU pods with sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1, Python 3.12.3.",
          "Mac runs: same code with the Mac libraries (sklearn 1.7.2, numpy 2.2.6, scipy 1.15.3, Accelerate) for the Mac-lib and P17-verifier diagnostics.",
          "Not-possible check: scanned every npz over 300 kB under Desktop/release and G-Drive paper work for arrays with an encoder stream and clip names, compared clip sets and p_yes with the 5 saved zero-shot runs."],
  scripts={os.path.basename(p): sha(p) for p in sorted(glob.glob(f"{V}/scripts/*"))},
  inputs=man, saved_curves=side["saved"], zeroshot=side["zeroshot"], pyes_maxdiff=side["pyes_maxdiff"],
  answer_edaicfull_states_run=side["answer_edaicfull_states_run"], pitt_figure_file_vs_omni_final_maxabs=side["pitt_figure_file_vs_omni_final_maxabs"],
  verifier_pods=dict(created=created, cpus=det["pod_cpus"], note="lo-v5-5 and lo-v5-7 landed on AMD EPYC 7702P (Zen 2, no AVX-512) and were deleted unused at 22:59Z; lo-v5-8 and lo-v5-9 (EPYC 7713, Zen 3) ran the Zen 3 host check; the watchdog (watchdog_v5.sh, one test loop, then nohup and disown) pulled and deleted the other 7 between 23:09Z and 23:14Z and exited; all 9 ids return 404 on REST",
                     cost_estimate_usd=1.72),
  not_possible=np_, task_vrefit_vs_saved=side["task_vrefit_vs_saved"], details=det["details"],
  caveats=["T5_diag_blas_coretype: the claimed numbers are right; its description says only SkylakeX 'and SapphireRapids on E-DAIC' reproduce exactly, but SapphireRapids is also exact on Pitt (0, 32/32) in the task's own pod10 file and in my refit.",
           "T5 LM/answer and proj rows were checked on different AVX-512 hosts than the task (9654/9754 here vs 9754/9575F/9965 there); all exact, which supports the SkylakeX-kernel explanation.",
           "Nothing written under omni_final, variants or folds; tex untouched; nothing pushed."])
json.dump(out, open(f"{LO}/verify/t5_VERIFIED_sidecar.json", "w"), indent=1, default=str); print("sidecar ok", len(out["scripts"]), "scripts")
