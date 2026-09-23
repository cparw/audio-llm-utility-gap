"""PART20 POD6 job B extra: zero-shot answer AUC (paper per-clip file) on the 363 NeuroVoz clips NOT in the matched subset,
and the age/sex make-up of those speakers, to show where the full-set 0.6951 comes from. Not asked; reported as context.
usage: b_complement.py OUTDIR"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p20_common import auc_rank, boot_auc, paste, sidecar
OUT = sys.argv[1]; ZS = "<local data dir>/release/omni_final/omni_neurovoz_zeroshot_scores.csv"
MD = "<local data dir>/paper work/paper1_local_runs/drive_data/spanish_neurovoz/zenodo_upload/metadata"
z = pd.read_csv(ZS); sub = set(pd.read_csv(f"{OUT}/nv_matched_clips.csv")["clip"]); c = z[~z["clip"].isin(sub)].copy()
y = c.label.values.astype(int); s = c.p_yes.values; sp = c.speaker.astype(str).values
a = auc_rank(y, s); lo, hi, nu = boot_auc(y, s, sp)
meta = pd.concat([pd.read_csv(f"{MD}/data_pd.csv"), pd.read_csv(f"{MD}/data_hc.csv")]).groupby("ID").agg(age=("Age", "max"), sex=("Sex", "max"), grp=("Group", "first")).reset_index()
meta["ID"] = meta.ID.astype(str); cm = meta[meta.ID.isin(set(sp))]
desc = {g: dict(n_spk=int((cm.grp == g).sum()), age_mean=float(cm[cm.grp == g].age.mean()), male_frac=float(cm[cm.grp == g].sex.mean())) for g in ("PD", "HC")}
per = f"{OUT}/B_zeroshot_nv_complement363.csv"; c.to_csv(per, index=False)
sidecar(per.replace(".csv", ".sidecar.json"), result="Qwen2.5-Omni zero-shot answer AUC on the 363 NeuroVoz clips outside the matched subset (paper per-clip file)",
        auc=round(a, 4), lo=round(lo, 4), hi=round(hi, 4), usable_draws=nu, n=int(len(c)), n_speakers=int(len(set(sp))), speakers_by_group=desc,
        model_id="Qwen/Qwen2.5-Omni-7B", prompt=str(c.prompt.iloc[0]), seed="default_rng(0), 2000 draws", command="/usr/local/bin/python3 " + " ".join(sys.argv),
        sources=[ZS, f"{OUT}/nv_matched_clips.csv", f"{MD}/data_pd.csv", f"{MD}/data_hc.csv"])
open(per.replace(".csv", ".RESULT"), "w").close()
print(desc)
paste("P20.POD6.B.nvcomplement.zeroshot", "NeuroVoz 363 clips OUTSIDE the matched subset, Qwen2.5-Omni zero-shot answer AUC (paper file; context)", a, lo, hi, len(c), len(set(sp)), per)
