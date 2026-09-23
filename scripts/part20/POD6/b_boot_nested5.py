"""PART20 POD6 job B: interval for the five-repeat encoder probe on the NeuroVoz matched subset.
Point = mean of the five per-repeat out-of-fold AUCs. Interval: 2000 draws, default_rng(0), speakers with replacement;
per draw all five per-repeat AUCs are recomputed on the drawn clips and averaged. Also the single split.
Pipeline check: this pod's zero-shot p_yes on the subset vs the paper's omni_final per-clip file.
usage: b_boot_nested5.py OOF.csv NESTED.json ZEROSHOT_THIS_POD.csv OUTDIR"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p20_common import auc_rank, boot_auc, paste, sidecar
OOF, NJ, ZP, OUT = sys.argv[1:5]
R = "<local data dir>/release"; ZS = f"{R}/omni_final/omni_neurovoz_zeroshot_scores.csv"
d = pd.read_csv(OOF); y = d.label.values.astype(int); spk = d.speaker.astype(str).values
P = np.column_stack([d[f"p_seed{s}"].values for s in range(5)])
per = [auc_rank(y, P[:, s]) for s in range(5)]; m5 = float(np.mean(per))
rng = np.random.default_rng(0); u = np.unique(spk); idx = {p: np.where(spk == p)[0] for p in u}; v = []
for _ in range(2000):
    ii = np.concatenate([idx[p] for p in rng.choice(u, size=len(u), replace=True)])
    if len(set(y[ii])) < 2: continue
    v.append(np.mean([auc_rank(y[ii], P[ii, s]) for s in range(5)]))
lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
a1 = auc_rank(y, d.p_single.values); lo1, hi1, _ = boot_auc(y, d.p_single.values, spk)
# pipeline check against the paper per-clip zero-shot file
zp = pd.read_csv(ZP); zr = pd.read_csv(ZS); mm = zp.merge(zr, on="clip", suffixes=("_pod6", "_paper"))
diff = (mm.p_yes_pod6 - mm.p_yes_paper).abs()
chk = dict(n_matched=int(len(mm)), max_abs_diff=float(diff.max()), median_abs_diff=float(diff.median()),
           within_1e3=int((diff <= 1e-3).sum()), pearson=float(np.corrcoef(mm.p_yes_pod6, mm.p_yes_paper)[0, 1]),
           auc_pod6=auc_rank(mm.label_pod6.values, mm.p_yes_pod6.values), auc_paper=auc_rank(mm.label_paper.values, mm.p_yes_paper.values),
           prompt_same=bool((mm.prompt_pod6 == mm.prompt_paper).all()))
print("PIPELINE CHECK (this pod zero-shot vs omni_final per clip):", chk, flush=True)
nj = json.load(open(NJ))
outcsv = f"{OUT}/B_encprobe_nested5_nvmatched_oof.csv"; d.to_csv(outcsv, index=False)
sidecar(outcsv.replace(".csv", ".sidecar.json"), result="Qwen2.5-Omni ENCODER nested probe, five renumbered splits, NeuroVoz age and sex matched subset",
        mean5=round(m5, 4), lo=round(lo, 4), hi=round(hi, 4), per_repeat=[round(a, 4) for a in per], usable_draws=len(v),
        single_split=dict(auc=round(a1, 4), lo=round(lo1, 4), hi=round(hi1, 4)), chosen_layers={k: nj[k]["layers"] for k in nj if k != "mean5"},
        paper_full_set=dict(mean5=0.9213, single_split=0.9160), n=int(len(d)), n_speakers=int(len(u)),
        folds="NEW for this subset: renumber speakers with default_rng(seed).permutation(unique ids), seeds 0..4, GroupKFold(5) POD tie rule; files in B_neurovoz_matched/folds/",
        model_id="Qwen/Qwen2.5-Omni-7B (thinker.audio_tower, 32 encoder layers, mean over frames), bf16, 30 s window",
        prompt=str(zp.prompt.iloc[0]), pipeline_check_vs_paper_zeroshot=chk,
        probe="StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced'); inner GroupKFold(4) layer choice",
        seed="bootstrap default_rng(0), 2000 draws, speakers with replacement; all five repeats recomputed per draw",
        command="/usr/local/bin/python3 " + " ".join(sys.argv), sources=[OOF, NJ, ZP, ZS])
open(f"{OUT}/B_encprobe_nested5_nvmatched.RESULT", "w").close()
paste("P20.POD6.B.nvmatched.encprobe.mean5", "NeuroVoz matched subset Qwen2.5-Omni encoder nested probe, mean of five repeats (full-set paper 0.9213)", m5, lo, hi, len(d), len(u), outcsv)
paste("P20.POD6.B.nvmatched.encprobe.single", "NeuroVoz matched subset Qwen2.5-Omni encoder nested probe, single split (full-set paper 0.9160)", a1, lo1, hi1, len(d), len(u), outcsv)
json.dump(dict(mean5=m5, lo=lo, hi=hi, per=per, single=[a1, lo1, hi1], check=chk), open(f"{OUT}/B_encprobe_summary.json", "w"), indent=1)
