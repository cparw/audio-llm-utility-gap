"""PART 26 Kimi-Audio E-DAIC, try 2: add the run record (pods, costs, pod-side check, Mac checks) to the sidecar written by
p26_kimi_edaic_gap.py. It reads only files on disk and changes no number the gap script wrote.
usage: /usr/local/bin/python3 p26_kimi_edaic_finalize.py"""
import os, json, hashlib, datetime
C = "scores/part26/Kimi-Audio_E-DAIC"
SC = f"{C}/p26_kimi_edaic_enc_perclip.sidecar.json"
CPU = f"{C}/pull/p26-kimi-edaic-cpu2"
GPU = f"{C}/pull/p26-kimi-edaic-gpu"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

S = json.load(open(SC))
V = json.load(open(f"{C}/verify/p26_kimi_edaic_verify.json"))
ER = json.load(open(f"{C}/verify/p26_kimi_edaic_encoder_recompute.json"))
PV = json.load(open(f"{CPU}/out/p26_kimi_edaic_podverify.json"))
NJ = json.load(open(f"{CPU}/out/p26_kimi_edaic_enc_nested5.json"))
ZJ = json.load(open(f"{GPU}/out/p26_kimi_edaic_zeroshot.json"))
S["run"] = {
 "try": "PART 26 try 2. Try 1 extracted the states on the GPU pod and pulled them, but its CPU pod never received the states and was idle-killed by the watchdog; try 2 reused the pulled states (sha256 checked) and ran the probe on a new CPU pod.",
 "gpu_pod_extraction": {"name": "p26-kimi-edaic-gpu", "id": "m42ljldnq92qmw", "gpu": "NVIDIA H100 80GB HBM3", "cost_per_hr": 3.49,
                        "created_utc": "2026-09-24 01:05:48", "deleted_utc": "2026-09-24 01:26:59 (watchdog, ALL_DONE, HTTP 204)",
                        "hours": 0.353, "cost_usd": 1.23},
 "cpu_pods_try1": [{"id": "p6afmxz5ilpzr1", "flavor": "cpu3c", "cost_per_hr": 0.96, "created_utc": "01:15:28", "deleted_utc": "01:22:07",
                    "note": "no public ip, deleted, ran nothing"},
                   {"id": "1b33rynkogtm3y", "flavor": "cpu3c", "cost_per_hr": 0.96, "created_utc": "01:22:22", "deleted_utc": "01:39:40",
                    "note": "pip done, states never uploaded, watchdog idle kill"}],
 "cpu_pod_probe": {"name": "p26-kimi-edaic-cpu2", "id": "6zo60zbviepd7a", "flavor": "cpu3c", "vcpu": 32, "cost_per_hr": 0.96,
                   "created_utc": "2026-09-24 02:43:00", "deleted_utc": "2026-09-24 " + open(f"{C}/wd/p26-kimi-edaic-cpu2.gone").read().strip().split()[-1] + " (watchdog, ALL_DONE, HTTP 204, then REST 404)", "hours": 0.088, "cost_usd": 0.08},
 "extraction": {"script": "podfiles/gpu/kimi_enc_probe.py = release podD2_final/kimi_probe.py plus NEW forward hooks on the Kimi continuous-feature Whisper encoder (32 layers), unchanged otherwise",
                "script_sha256": sha(f"{GPU}/kimi_enc_probe.py"), "command": "python kimi_enc_probe.py data/mf_edaic.csv out/p26_kimi_edaic mdd",
                "window": ZJ["window_seconds"], "dtype": ZJ["dtype"], "torch": ZJ["torch"], "transformers": ZJ["transformers"], "librosa": ZJ["librosa"],
                "kimi_snapshot": ZJ["snapshot"].split("/")[-1], "enc_frames": [ZJ["enc_frames_min"], ZJ["enc_frames_max"]],
                "audio": "E-DAIC_audio.tar.gz on the pod, tarball sha256 83ea8873... equals drive_data/checksums.txt; all 275 dcaps_proc wavs equal the Mac files by sha256; x[:16000*30] cut in the script",
                "states_sha256": sha(f"{GPU}/out/p26_kimi_edaic_states.npz"), "encstates_sha256": sha(f"{GPU}/out/p26_kimi_edaic_encstates.npz")},
 "probe": {"script": "part25/A/podfiles/p25a_nested5.py (md5 615ced0380cca2636ed7cc30c71810e6), unchanged",
           "command": NJ["command"], "versions": NJ["versions"], "n_jobs": NJ["n_jobs"], "seconds": NJ["seconds"],
           "blas_threads": "OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 per fit"},
 "podverify": {k: PV[k] for k in PV if k in ("PASS", "max_abs_pred_diff", "max_abs_inner_score_diff", "layer_mismatches", "mean5_rank_from_oof", "abs_mean5_diff")},
 "mac_encoder_recompute": {"pass": ER["pass"], "what": "Hugging Face WhisperModel fp32 eager on the Mac, weights from the whisper-large-v3 folder of the same Kimi snapshot, 4 clips",
                           "min_cosine": min(v["min_cosine"] for v in ER["clips"].values()), "max_rel_l2": max(v["max_rel_l2"] for v in ER["clips"].values()),
                           "same_layer_match_all": all(v["best_match_is_same_layer"] for v in ER["clips"].values())},
 "mac_independent_check": {"ALL_CHECKS_PASS": V["ALL_CHECKS_PASS"], "file": f"{C}/verify/p26_kimi_edaic_verify.json"},
}
S["verified"] = bool(V["ALL_CHECKS_PASS"] and PV.get("PASS") and ER["pass"])
S["finalized_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
json.dump(S, open(SC, "w"), indent=1)
print("FINALIZED verified", S["verified"])
