"""PART 22 pod rows + sidecars. usage: python3 rows22.py provisional|final
Reads the synced pod results; writes sidecar json per result and rows (id, what, value, lo, hi, n, n_spk, file).
"""
import csv, json, os, sys, hashlib, shutil
BASE = "scores/part22"
SYNC = f"{BASE}/pod/sync"
MODE = sys.argv[1]
def sha1mb(p):
    with open(p, "rb") as f: return hashlib.sha256(f.read(1 << 20)).hexdigest()
P = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
MID = "Qwen/Qwen2.5-Omni-7B"
REFF = "scores/part16/EDAICFULL/edaic_full_perclip.csv"
WINF = "manifests/edaic_windows_full.csv"
rows = []
def row(i, what, v, n, nspk, f, lo="", hi="", fmt="{:.6f}"):
    rows.append(dict(id=i, what=what, value=(fmt.format(v) if isinstance(v, float) else str(v)), lo=lo, hi=hi, n=n, n_spk=nspk, file=f))

def sidecar(res_path, dst_name, srcs, n, cmd, libs, extra=None):
    shutil.copy(res_path, f"{BASE}/{dst_name}")
    sc = dict(result=f"scores/part22/{dst_name}", sources={s: sha1mb(s) for s in srcs if os.path.exists(s)}, n=n, seed="n/a (deterministic forward pass; no resampling)",
              model_id=MID, prompt_verbatim=P, exact_command=cmd, libs=libs, dtype="bfloat16", pod="RunPod 8flz9sdjujokel H200")
    if extra: sc.update(extra)
    json.dump(sc, open(f"{BASE}/{dst_name.rsplit('.', 1)[0]}.sidecar.json", "w"), indent=1)

meta_libs = None
# gate
g = f"{SYNC}/gate/gate.json"
if os.path.exists(g):
    G = json.load(open(g))
    s2libs = json.load(open(f"{SYNC}/step2_truncation_proof.json"))["libs"] if os.path.exists(f"{SYNC}/step2_truncation_proof.json") else None
    sidecar(g, "p22_validation_gate.json", [g, f"{SYNC}/gate/gate_zeroshot_scores.csv", f"{SYNC}/extract_full.py", REFF, WINF], len(G["rows"]),
            G["command"], s2libs)
    row("P22_gate_maxdiff", "validation gate: max |p_yes pod - reference| over 10 windows, unmodified extract_full.py", G["max_abs_diff"], 10, 10, "scores/part22/p22_validation_gate.json", fmt="{:.2e}")
# step 2
s2 = f"{SYNC}/step2_truncation_proof.json"
if os.path.exists(s2):
    S = json.load(open(s2)); meta_libs = S["libs"]
    sidecar(s2, "p22_step2_truncation_proof.json", [s2, f"{SYNC}/p22_run.py", REFF, WINF] + [f"{SYNC}/../sync/windows_full.csv"], len(S["results"]), S["command"], S["libs"],
            dict(yes_ids=S["yes_ids"], no_ids=S["no_ids"], rule_yes_ids=S.get("rule_yes_ids"), rule_no_ids=S.get("rule_no_ids"),
                 processor_call_verbatim=S["processor_call_verbatim"], cut_rule=S["cut_rule"]))
    for r in S["results"]:
        f = "scores/part22/p22_step2_truncation_proof.json"; pid = r["pid"]
        row(f"P22_s2_{pid}_tok_full", f"pid {pid} ({r['win_dur_s']} s window) audio tokens, default processor, full window", r["full"]["n_audio_tok"], 1, 1, f)
        row(f"P22_s2_{pid}_tok_cut300", f"pid {pid} audio tokens, default processor, first 300 s", r["cut300"]["n_audio_tok"], 1, 1, f)
        row(f"P22_s2_{pid}_pyes_full", f"pid {pid} p_yes, full window", r["full"]["p_yes"], 1, 1, f)
        row(f"P22_s2_{pid}_pyes_cut300", f"pid {pid} p_yes, first 300 s", r["cut300"]["p_yes"], 1, 1, f)
        row(f"P22_s2_{pid}_verdict", f"pid {pid} verdict", r["verdict"], 1, 1, f)
    row("P22_s2_overall", "step 2 overall verdict (3 longest windows)", S["overall_verdict"], 3, 3, "scores/part22/p22_step2_truncation_proof.json")
# step 3 probe
s3 = f"{SYNC}/step3_probe.json"
if os.path.exists(s3):
    Q = json.load(open(s3))
    sidecar(s3, "p22_step3_probe.json", [s3, f"{SYNC}/p22_run.py", WINF], 1, Q["command"], Q["libs"],
            dict(lifted_call_verbatim=Q["lifted_call_verbatim"]))
    f = "scores/part22/p22_step3_probe.json"
    row("P22_s3_probe_tok_default", f"pid {Q['pid']} 900 s window audio tokens, default processor", Q["default"]["n_audio_tok"], 1, 1, f)
    row("P22_s3_probe_tok_lifted", f"pid {Q['pid']} 900 s window audio tokens, truncation=False", Q["lifted"]["n_audio_tok"], 1, 1, f)
    row("P22_s3_probe_model_ran", "model forward ran on the 22,500-token input", str(Q["model_ran"]), 1, 1, f)
    if Q["model_ran"]:
        row("P22_s3_probe_pyes_lifted", f"pid {Q['pid']} p_yes, truncation=False", Q["lifted"]["p_yes"], 1, 1, f)
        row("P22_s3_probe_peak_gib", f"pid {Q['pid']} peak GPU memory GiB, truncation=False", Q["lifted"]["peak_gpu_gib"], 1, 1, f, fmt="{:.2f}")
# step 3 all
s3a = f"{BASE}/pod/rows_step3.json"
if MODE == "final" and os.path.exists(s3a):
    for r in json.load(open(s3a)): rows.append(r)
FIELDS = ["id", "what", "value", "lo", "hi", "n", "n_spk", "file"]
os.makedirs(f"{BASE}/rows", exist_ok=True)
# rows/P22.tsv belongs to the verifier (verified FINAL rows, its own schema); this script never writes it.
dst = f"{BASE}/rows/P22_provisional.tsv"
keep = []
with open(dst, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t"); w.writeheader()
    for r in keep + rows: w.writerow({k: r.get(k, "") for k in FIELDS})
print("wrote", dst, len(rows), "rows (+", len(keep), "kept)")
