import numpy as np, pandas as pd, os
R="<local data dir>/release"; G="<local data dir>/paper work/paper1_local_runs"
pairs = [
 (f"{G}/probe2/edaic_states.npz", f"{G}/probe2/edaic_zeroshot_scores.csv"),
 (f"{G}/probe2/edaic_states.npz", f"{G}/probe2/pod_runs/edaic_zeroshot_scores.csv"),
 (f"{G}/probe2/neurovoz_states.npz", f"{G}/probe2/neurovoz_zeroshot_scores.csv"),
 (f"{G}/probe2/neurovoz_states.npz", f"{G}/probe2/pod_runs/neurovoz_zeroshot_scores.csv"),
 (f"{R}/overnight2/part10/o25_pitt_states.npz", f"{R}/omni_final/omni_pitt_zeroshot_scores.csv"),
 (f"{R}/overnight2/part10/o25_edaic_states.npz", f"{R}/omni_final/omni_edaic_zeroshot_scores.csv"),
] + [(f"{R}/overnight2/q3o/q3o_{d}_states.npz", f"{R}/overnight2/q3o_new/q3o_{d}_zeroshot_scores.csv") for d in ["adress2020","adresso","kcl","neurovoz","pcgita"]] + [
 (f"{R}/edaic_rerun/part16/pod_sync/q3o_pitt_states.npz", f"{R}/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv"),
 (f"{G}/part16_POD3_states/q3o_pitt_states.npz", f"{R}/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv"),
 (f"{G}/part16_POD3_states/q3o_pcgita_states.npz", f"{R}/overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv"),
 (f"{G}/part16_POD3_states/q3o_pcgita_states.npz", f"{R}/overnight2/q3o/q3o_pcgita_zeroshot_scores.csv"),
 (f"{R}/edaic_rerun/part16/pod_sync/q3o_pitt_states.npz", f"{R}/overnight2/part10/q3o_pitt_zeroshot_scores.csv"),
]
for a,b in pairs:
    z=np.load(a,allow_pickle=True); s=pd.read_csv(b)
    col = "clip" if "clip" in s.columns else s.columns[0]
    nm = z["name"].astype(str); key = s[col].astype(str).map(os.path.basename).values
    same_order = (key==nm).all() if len(key)==len(nm) else False
    m = dict(zip(key, s["p_yes"].values))
    d = np.array([abs(m[n]-p) if n in m else np.nan for n,p in zip(nm, z["p_yes"])])
    print(os.path.basename(a), "|", b.split('release/')[-1] if 'release/' in b else b.split('runs/')[-1], "| order", same_order, "| max|dp_yes|", np.nanmax(d), "| spk csv ex", s[[c for c in s.columns if 'speaker' in c][0]].astype(str).values[:1] if any('speaker' in c for c in s.columns) else '')
