"""PART 23 A diagnostic: why do whole-window p_yes differ from PART 22 on clips whose input is identical?
Runs the UNMODIFIED part16 extract_full.py (sha256 b9262a04...) on the 10 PART 22 gate windows on THIS pod (H100 NVL)
and compares to the PART 22 gate on H200 (which matched the published 300 s reference exactly, max diff 0).
Also compares, on the whole-window run, the clips shorter than 300 s (input identical to the 300 s window)."""
import csv, json, hashlib, time, numpy as np
D = "/workspace/p23a"; OUT = f"{D}/out"
mine = {r["clip"]: r for r in csv.DictReader(open(f"{D}/gate/gate_zeroshot_scores.csv"))}
ref = {r["clip"]: r for r in csv.DictReader(open(f"{D}/ref/gate_zeroshot_scores.csv"))}
rows = []
for c, r in ref.items():
    m = mine[c]
    rows.append(dict(clip=c, n_audio_tok=m["n_audio_tok"], p_yes_h100nvl_this_pod=float(m["p_yes"]), p_yes_h200_part22_gate=float(r["p_yes"]),
                     abs_diff=abs(float(m["p_yes"]) - float(r["p_yes"])), mass_h100nvl=float(m["mass"]), mass_h200=float(r["mass"])))
with open(f"{OUT}/A2d_gpu_gate_diagnostic.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
EX = {r["pid"]: r for r in csv.DictReader(open(f"{D}/A_extract_perclip.csv"))}
T7 = {r["pid"]: r for r in csv.DictReader(open(f"{D}/ref/T7b_edaic300_meanof5_perclip.csv"))}
short = [p for p in EX if float(EX[p]["win_dur_s"]) < 300]
ds = np.array([abs(float(EX[p]["p_yes"]) - float(T7[p]["p_yes_answer"])) for p in short])
d = np.array([r["abs_diff"] for r in rows])
sh = lambda p: hashlib.sha256(open(p, "rb").read(1 << 20)).hexdigest()
res = dict(gate_n=len(rows), gate_max_abs_diff=float(d.max()), gate_n_exact=int((d == 0).sum()), gate_n_le_1e4=int((d <= 1e-4).sum()),
           short_clips_lt_300s_n=len(short), short_clips_max_abs_diff_vs_300s_reference=float(ds.max()),
           short_clips_n_le_1e4=int((ds <= 1e-4).sum()),
           gpu_this_pod=json.load(open(f"{D}/A_extract_meta.json"))["libs"]["gpu"], gpu_part22="NVIDIA H200 (lifted_all275.meta.json)",
           script="extract_full.py copied unmodified from edaic_rerun/part16/EDAICFULL (sha256 b9262a04547b719a98593718a9f5016d3d355c0c43df5f5935a579fabc8dc949)",
           command="cd /workspace/p23a/gate && HF_HOME=/workspace/hf WINDOW_S=0 python3 extract_full.py /workspace/p23a/ref/mf_gate10_p23a.csv /workspace/p23a/gate/gate mdd",
           sources={p: sh(p) for p in [f"{D}/gate/gate_zeroshot_scores.csv", f"{D}/ref/gate_zeroshot_scores.csv", f"{D}/gate/extract_full.py", f"{D}/A_extract_perclip.csv"]},
           written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(res, open(f"{OUT}/A2d_gpu_gate_diagnostic.sidecar.json", "w"), indent=1)
for r in rows: print(f"{r['clip']:8s} tok {r['n_audio_tok']:>5s} H100NVL {r['p_yes_h100nvl_this_pod']:.6f} H200 {r['p_yes_h200_part22_gate']:.6f} d {r['abs_diff']:.2e}")
print("GATE", json.dumps({k: v for k, v in res.items() if k != "sources"}))
open(f"{OUT}/A2d_gpu_gate_diagnostic.RESULT", "w").write("done\n"); open(f"{D}/A2d_gpu_gate_diagnostic.RESULT", "w").write("done\n")
