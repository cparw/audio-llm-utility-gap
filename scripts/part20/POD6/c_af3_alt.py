"""PART20 POD6 job C: collect landed Pitt results; add the AF3 alternative copy overnight2/part10/af3_pitt.csv
(conflict, agreement, PAIRED conflict minus agreement). Method check first: the same code must reproduce the M8 row
for the part3 AF3 copy. Arm from pitt_conflict_manifest.csv column 'set', key basename(segment_path).
PAIRED: one speaker draw per replicate over all 468 clips (228 speaker labels); both arms scored on that draw.
usage: c_af3_alt.py OUTDIR"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p20_common import auc_rank, boot_auc, paste, sidecar, vs_half
OUT = sys.argv[1]
PCM = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
R = "<local data dir>/release"
M8 = f"{R}/edaic_rerun/part16/M8_conflict_cells.csv"
pc = pd.read_csv(PCM); arm = dict(zip(pc["segment_path"].map(os.path.basename), pc["set"]))

def cells(path):
    d = pd.read_csv(path); d["b"] = d["clip_path"].map(os.path.basename); d["arm"] = d["b"].map(arm)
    assert d["arm"].notna().all() and len(d) == 468, path
    y = d.label.values.astype(int); s = d.p_yes.values.astype(float); spk = d.speaker_id.astype(str).values; a = d["arm"].values
    out = {}
    for nm in ("conflict", "agreement"):
        m = a == nm; v = auc_rank(y[m], s[m]); lo, hi, nu = boot_auc(y[m], s[m], spk[m])
        out[nm] = dict(auc=v, lo=lo, hi=hi, n=int(m.sum()), n_spk=int(len(set(spk[m]))), usable=nu)
    rng = np.random.default_rng(0); u = np.unique(spk); idx = {p: np.where(spk == p)[0] for p in u}; dv = []
    for _ in range(2000):
        ii = np.concatenate([idx[p] for p in rng.choice(u, size=len(u), replace=True)])
        c = ii[a[ii] == "conflict"]; g = ii[a[ii] == "agreement"]
        if len(set(y[c])) < 2 or len(set(y[g])) < 2: continue
        dv.append(auc_rank(y[c], s[c]) - auc_rank(y[g], s[g]))
    dv = np.array(dv)
    out["conflict_minus_agreement"] = dict(auc=out["conflict"]["auc"] - out["agreement"]["auc"], lo=float(np.percentile(dv, 2.5)),
                                           hi=float(np.percentile(dv, 97.5)), n=468, n_spk=int(len(u)), usable=int(len(dv)))
    return d, out

m8 = pd.read_csv(M8); ref = m8[(m8.model == "Audio Flamingo 3") & (m8.set == "B_pitt")].set_index("arm")
_, chk = cells(f"{R}/overnight2/part3/af3_pitt.csv")
diffs = {k: [round(chk[k]["auc"] - ref.loc[k, "auc"], 4), round(chk[k]["lo"] - ref.loc[k, "lo"], 4), round(chk[k]["hi"] - ref.loc[k, "hi"], 4)] for k in chk}
print("METHOD CHECK vs M8 AF3 part3 row (value, lo, hi differences):", diffs, flush=True)
src = f"{R}/overnight2/part10/af3_pitt.csv"
d, res = cells(src)
per = f"{OUT}/C_af3_pitt_part10_arms.csv"
d[["speaker_id", "b", "label", "p_yes", "answer_mass", "arm"]].rename(columns={"b": "clip"}).to_csv(per, index=False)
js = json.load(open(src.replace(".csv", ".json"))) if os.path.exists(src.replace(".csv", ".json")) else {}
sidecar(per.replace(".csv", ".sidecar.json"), result="Audio Flamingo 3 Pitt 468 zero-shot answer AUC by arm, alternative copy overnight2/part10",
        cells=res, method_check_vs_M8_part3=diffs, model_id=js.get("model", js.get("checkpoint", "see source json")),
        prompt=str(d.prompt.iloc[0]), prompt_unique=int(d.prompt.nunique()), source_json=js,
        seed="bootstrap numpy.random.default_rng(0), 2000 draws, speakers with replacement; paired = one speaker draw per replicate",
        n=468, n_speakers=int(d.speaker_id.nunique()), command="/usr/local/bin/python3 " + " ".join(sys.argv),
        sources=[src, PCM, M8, f"{R}/overnight2/part3/af3_pitt.csv"])
for k, v in res.items():
    paste(f"P20.POD6.C.af3part10.pitt.{k}", f"Pitt AF3 zero-shot answer AUC {k} (alt copy overnight2/part10/af3_pitt.csv)",
          v["auc"], v["lo"], v["hi"], v["n"], v["n_spk"], per, diff=(k == "conflict_minus_agreement"))
open(per.replace("_arms.csv", ".RESULT"), "w").close()
json.dump(dict(check=diffs, part10=res), open(f"{OUT}/C_af3_part10_summary.json", "w"), indent=1)
