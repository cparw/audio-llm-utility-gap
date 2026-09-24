"""Cross-check which input the part10 AF3 Pitt scores came from, using per-clip p_yes on disk.
A = part10 (the Table 2 file)          overnight2/part10/af3_pitt.csv
B = POD4c val10 (known first 30 s)     POD4c/work/af3_val10.csv   (WINDOW_S=30, transformers 5.17.0)
C = Sep 16 full-segment run            paper1_local_runs/runpod_results/pitt468_af3.csv (whole array, no slicing)"""
import csv, os
import numpy as np
A = {os.path.basename(r["clip_path"]): float(r["p_yes"]) for r in csv.DictReader(open("<local data dir>/release/overnight2/part10/af3_pitt.csv"))}
Bv = list(csv.DictReader(open("scores/part20/POD4c/work/af3_val10.csv")))
C = {os.path.basename(r["clip_path"]): float(r["p_yes"]) for r in csv.DictReader(open("<local data dir>/paper1_local_runs/runpod_results/pitt468_af3.csv"))}
dur = {r["clip"]: float(r["dur_s"]) for r in csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pitt468_durations.csv")))}
print("val10 clips: clip, dur_s, scored_s, |part10 - val10|, |fullseg - val10|")
dA, dC = [], []
for r in Bv:
    k = r["orig_clip_id"]; b = float(r["p_yes"])
    dA.append(abs(A[k] - b)); dC.append(abs(C[k] - b))
    print(f"  {k} {float(r['dur_s']):.2f} {float(r['scored_s']):.1f} {abs(A[k]-b):.2e} {abs(C[k]-b):.4f}")
print(f"val10: max |part10 - val10| = {max(dA):.2e}; max |fullseg - val10| = {max(dC):.4f}; median |fullseg - val10| = {np.median(dC):.4f}")
ks = sorted(A); d = np.array([dur[k] for k in ks]); diff = np.array([abs(A[k] - C[k]) for k in ks])
for lo, hi in ((30, 30.5), (30.5, 33.3), (33.3, 40), (40, 56)):
    m = (d > lo) & (d <= hi)
    print(f"part10 vs fullseg, segments {lo} to {hi} s: n={m.sum()} median |dp_yes|={np.median(diff[m]):.4f} share over 0.01={np.mean(diff[m] > 0.01):.2f}")
print(f"all 468: median |dp_yes| {np.median(diff):.4f}, rank corr with duration {np.corrcoef(np.argsort(np.argsort(d)), np.argsort(np.argsort(diff)))[0,1]:.3f}")
