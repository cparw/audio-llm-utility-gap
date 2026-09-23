"""PART20 POD6 job B lower-priority check: the same pod pipeline on the full 1270 NeuroVoz set, so the subset probe
can be compared with a full-set value from the SAME machine and code (not only with the paper's 0.9213 / 0.9160).
Also checks the pod single split equals folds/neurovoz_groupkfold5_pod.csv and seeds equal the saved _seed files,
and the matched-subset minus full-set difference is NOT a paired test (the subset is part of the full set);
it is reported only as a point difference.
usage: b_full_check.py OOF.csv NESTED.json ZEROSHOT_THIS_POD.csv OUTDIR"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p20_common import auc_rank, boot_auc, paste, sidecar
OOF, NJ, ZP, OUT = sys.argv[1:5]
F = "folds"; ZS = "<local data dir>/release/omni_final/omni_neurovoz_zeroshot_scores.csv"
d = pd.read_csv(OOF); y = d.label.values.astype(int); spk = d.speaker.astype(str).values
fchk = {}
for t, f in [("single", f"{F}/neurovoz_groupkfold5_pod.csv")] + [(f"seed{s}", f"{F}/neurovoz_groupkfold5_pod_seed{s}_UNVERIFIED.csv") for s in range(5)]:
    s = pd.read_csv(f).set_index("clip_id").loc[d["clip"], "fold"].values; fchk[t] = float((s == d[f"fold_{t}"].values).mean())
print("FOLD CHECK vs saved release/folds files:", fchk, flush=True)
P = np.column_stack([d[f"p_seed{s}"].values for s in range(5)]); per = [auc_rank(y, P[:, s]) for s in range(5)]; m5 = float(np.mean(per))
rng = np.random.default_rng(0); u = np.unique(spk); idx = {p: np.where(spk == p)[0] for p in u}; v = []
for _ in range(2000):
    ii = np.concatenate([idx[p] for p in rng.choice(u, size=len(u), replace=True)])
    if len(set(y[ii])) < 2: continue
    v.append(np.mean([auc_rank(y[ii], P[ii, s]) for s in range(5)]))
lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
a1 = auc_rank(y, d.p_single.values); lo1, hi1, _ = boot_auc(y, d.p_single.values, spk)
zp = pd.read_csv(ZP); zr = pd.read_csv(ZS); mm = zp.merge(zr, on="clip", suffixes=("_pod6", "_paper")); df = (mm.p_yes_pod6 - mm.p_yes_paper).abs()
zchk = dict(n=int(len(mm)), within_1e3=int((df <= 1e-3).sum()), median_abs_diff=float(df.median()), max_abs_diff=float(df.max()),
            auc_pod6=auc_rank(mm.label_pod6.values, mm.p_yes_pod6.values), auc_paper=auc_rank(mm.label_paper.values, mm.p_yes_paper.values))
print("ZERO-SHOT CHECK full 1270:", zchk, flush=True)
outcsv = f"{OUT}/B_encprobe_nested5_nvfull_samepod_oof.csv"; d.to_csv(outcsv, index=False)
nj = json.load(open(NJ))
sidecar(outcsv.replace(".csv", ".sidecar.json"), result="Qwen2.5-Omni encoder nested probe, full NeuroVoz 1270, SAME pod and code as the matched subset (check)",
        mean5=round(m5, 4), lo=round(lo, 4), hi=round(hi, 4), per_repeat=[round(a, 4) for a in per], single_split=dict(auc=round(a1, 4), lo=round(lo1, 4), hi=round(hi1, 4)),
        paper=dict(mean5=0.9213, per_repeat=[0.9275, 0.9154, 0.928, 0.9124, 0.9232], single_split=0.9160), fold_check_vs_saved=fchk,
        chosen_layers={k: nj[k]["layers"] for k in nj if k != "mean5"}, zeroshot_check=zchk, n=int(len(d)), n_speakers=int(len(u)),
        model_id="Qwen/Qwen2.5-Omni-7B", prompt=str(zp.prompt.iloc[0]), seed="default_rng(0), 2000 draws",
        command="/usr/local/bin/python3 " + " ".join(sys.argv), sources=[OOF, NJ, ZP, ZS])
open(f"{OUT}/B_encprobe_nested5_nvfull_samepod.RESULT", "w").close()
paste("P20.POD6.B.nvfull_samepod.encprobe.mean5", "NeuroVoz FULL 1270 Qwen2.5-Omni encoder nested probe mean of five, same pod as subset (paper 0.9213)", m5, lo, hi, len(d), len(u), outcsv)
paste("P20.POD6.B.nvfull_samepod.encprobe.single", "NeuroVoz FULL 1270 Qwen2.5-Omni encoder nested probe single split, same pod as subset (paper 0.9160)", a1, lo1, hi1, len(d), len(u), outcsv)
json.dump(dict(mean5=m5, lo=lo, hi=hi, per=per, single=[a1, lo1, hi1], fold_check=fchk, zeroshot_check=zchk), open(f"{OUT}/B_full_check_summary.json", "w"), indent=1)
