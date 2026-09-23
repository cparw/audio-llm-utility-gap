"""Range over seeds 1,2,3 and over seeds 1,2,3 plus the original run (omni_final/omnisft_pitt_oof.csv, READ ONLY).
AUCs recomputed from each per-clip csv (rank formula); never read from json."""
import os, sys, csv, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats20b import auc
MAC = "<local data dir>/release/scores/part20/POD1b"
F = {f"seed{k}": f"{MAC}/std_ft_seed{k}_pitt_oof.csv" for k in (1, 2, 3)}
F["original"] = "<local data dir>/release/omni_final/omnisft_pitt_oof.csv"
V = {}
for nm, f in F.items():
    R = list(csv.DictReader(open(f))); y = np.array([int(r["label"]) for r in R]); s = np.array([float(r["p_yes"]) for r in R])
    arm = np.array([r["set"] for r in R])
    V[nm] = {"overall": auc(y, s), "conflict": auc(y[arm == "conflict"], s[arm == "conflict"]),
             "agreement": auc(y[arm == "agreement"], s[arm == "agreement"])}
out = {"values": V, "files": F}
for cell in ("overall", "conflict", "agreement"):
    s3 = [V[f"seed{k}"][cell] for k in (1, 2, 3)]; s4 = s3 + [V["original"][cell]]
    out[cell] = {"seeds_1_2_3": [min(s3), max(s3)], "seeds_1_2_3_plus_original": [min(s4), max(s4)],
                 "mean_1_2_3": float(np.mean(s3)), "sd_1_2_3": float(np.std(s3, ddof=1)),
                 "mean_4": float(np.mean(s4)), "sd_4": float(np.std(s4, ddof=1))}
    print(f"RANGE {cell}: seeds 1,2,3 {min(s3):.4f} to {max(s3):.4f} ({min(s3):.2f} to {max(s3):.2f}); "
          f"with original {min(s4):.4f} to {max(s4):.4f} ({min(s4):.2f} to {max(s4):.2f}); "
          f"mean of 3 {np.mean(s3):.4f} sd {np.std(s3, ddof=1):.4f}; mean of 4 {np.mean(s4):.4f} sd {np.std(s4, ddof=1):.4f}")
json.dump(out, open(f"{MAC}/std_ft_seed_range_pitt.json", "w"), indent=1)
print(json.dumps(V))
