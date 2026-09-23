import numpy as np, pandas as pd, glob, os, hashlib, json
G="<local data dir>/paper work/paper1_local_runs"; R="<local data dir>/release"
def sig(labels):
    u, c = np.unique(np.asarray(labels).astype(str), return_counts=True)
    return hashlib.sha1(("|".join(f"{a}:{b}" for a,b in zip(u,c))).encode()).hexdigest()[:10], len(u)
src = {}
for d in ["pitt","adresso","adress2020","pcgita","neurovoz","kcl","edaic"]:
    L = []
    L.append(("q2a probe2 states", np.load(f"{G}/probe2/{d}_states.npz",allow_pickle=True)["spk"]))
    L.append(("egemaps groups", np.load(f"{R}/overnight/egemaps_{d}.npz",allow_pickle=True)["groups"]))
    L.append(("omni_final enc oof csv", pd.read_csv(f"{R}/omni_final/omni_{d}_enc_nested_oof.csv",dtype={"speaker":str})["speaker"].values))
    L.append(("q3o_new enc oof csv", pd.read_csv(f"{R}/overnight2/q3o_new/q3o_{d}_enc_nested_oof.csv",dtype={"speaker":str})["speaker"].values))
    kd = f"{R}/overnight2/kimi/kimi_{d}_states.npz" if d in ("pitt","edaic","pcgita") else f"{R}/overnight2/kimi_new/kimi_{d}_states.npz"
    L.append(("kimi states", np.load(kd,allow_pickle=True)["spk"]))
    if d in ("pitt","edaic"): L.append(("o25 part10 states", np.load(f"{R}/overnight2/part10/o25_{d}_states.npz",allow_pickle=True)["spk"]))
    else: L.append(("q3o states", np.load(f"{R}/overnight2/q3o/q3o_{d}_states.npz",allow_pickle=True)["spk"]))
    if d in ("pitt","pcgita"): L.append(("q3o POD3 states", np.load(f"{G}/part16_POD3_states/q3o_{d}_states.npz",allow_pickle=True)["spk"]))
    for f in [f"{R}/omni_final/omnisft_{d}_oof.csv"]:
        L.append(("omnisft oof csv", pd.read_csv(f,dtype={"speaker":str})["speaker"].values))
    if d in ("pitt","edaic"):
        L.append(("q3o part10 zeroshot csv", pd.read_csv(f"{R}/overnight2/part10/q3o_{d}_zeroshot_scores.csv",dtype={"speaker":str})["speaker"].values))
    if d in ("edaic","neurovoz"): L.append(("q2a pod_runs enc csv", pd.read_csv(f"{G}/probe2/pod_runs/{d}_enc_nested_oof.csv",dtype={"speaker":str})["speaker"].values))
    if d=="pitt": L.append(("q2a pod_runs pitt enc csv", pd.read_csv(f"{G}/probe2/pod_runs/pitt_enc_nested_oof.csv",dtype={"speaker":str})["speaker"].values))
    print("==", d)
    for name, lab in L:
        s, n = sig(lab); print(f"   {name:28s} n={len(lab)} groups={n} sig={s} ex={np.asarray(lab).astype(str)[:2].tolist()}")
