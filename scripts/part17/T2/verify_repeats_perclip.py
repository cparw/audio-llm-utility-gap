"""PART17 T2: verify the 5 repeated GroupKFold(5) partitions (speaker ids permuted with default_rng(seed),
seed 0..4, as nested_repeats_all.py / POD3 perlayer_nested.py do) against saved PER-CLIP repeat OOFs.
Rules tried on the permuted integer codes: mac (default argsort here) and stable (kind='stable', reversed)."""
import sys, json, os, time, hashlib
import numpy as np, pandas as pd
from eng import *
G = "<local data dir>/paper work/paper1_local_runs"; R = "<local data dir>/release"; P3 = f"{R}/edaic_rerun/part16/POD3"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repeats_perclip_results.jsonl")
def sha1mb(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read(1 << 20)); return h.hexdigest()
SUF = {"enc": "encoder", "proj": "proj", "llm": "llm", "ans": "ans"}
RUNS = []
for d in ["pitt", "pcgita"]:
    summ = json.load(open(f"{P3}/q3o_{d}_perlayer_summary.json"))
    for key, v in summ["streams"].items():
        RUNS.append(dict(run=f"q3o_POD3_{d}", model="Qwen3-Omni", dataset=d, stream=key, states=f"{G}/part16_POD3_states/q3o_{d}_states.npz",
                         oof=f"{P3}/q3o_{d}_{SUF[key]}_nested_oof.csv", layers=v["nested_chosen_layers"], summary=f"{P3}/q3o_{d}_perlayer_summary.json",
                         saved_auc=v["nested_per_repeat"], script=f"{P3}/perlayer_nested.py"))
RUNS.append(dict(run="omni_part10_pitt_nested5", model="Qwen2.5-Omni", dataset="pitt", stream="enc", states=f"{R}/overnight2/part10/o25_pitt_states.npz",
                 oof=f"{R}/overnight2/part10/pitt_enc_nested5_oof.npz", layers=None, summary=None, saved_auc=None, script=f"{R}/scripts/nested_repeats_all.py (assumed; no script saved beside the npz)"))
only = set(sys.argv[1:])
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        j = json.loads(line); done.add((j["run"], j["stream"]))
for run in RUNS:
    if (only and run["run"] not in only) or (run["run"], run["stream"]) in done: continue
    t0 = time.time()
    z = np.load(run["states"], allow_pickle=True); y = z["label"].astype(int); spk = z["spk"].astype(str); nm = z["name"].astype(str)
    X = z[run["stream"]].astype(np.float32); nl = X.shape[1]
    if run["oof"].endswith(".npz"):
        q = np.load(run["oof"], allow_pickle=True); saved = q["oof"]
        align = dict(names=bool((q["name"].astype(str) == nm).all()), spk=bool((q["spk"].astype(str) == spk).all()), label=bool((q["label"] == y).all()))
    else:
        s = pd.read_csv(run["oof"]); cols = [c for c in s.columns if c.startswith("p_probe_rep")]; saved = s[cols].values.T
        align = dict(names=bool((s["clip"].astype(str).values == nm).all()), spk=bool((s["speaker"].astype(str).values == spk).all()), label=bool((s["label"].values == y).all()))
    reps = saved.shape[0]
    rec = dict(run=run["run"], model=run["model"], dataset=run["dataset"], stream=run["stream"], states=run["states"], oof_file=run["oof"],
               summary=run["summary"], script=run["script"], n=int(len(y)), n_spk=int(len(np.unique(spk))), align=align, repeats=reps,
               sha_states=sha1mb(run["states"]), sha_oof=sha1mb(run["oof"]), per_repeat=[])
    for sd in range(reps):
        g = perm_codes(spk, sd); lay = run["layers"][sd] if run["layers"] else None
        r = dict(seed=sd, saved_auc_rank=rank_auc(y, saved[sd]), saved_layers=lay, cands={})
        for kind in ("mac", "stable"):
            f = gkf_folds(g, kind)
            ch = lay if lay is not None else inner_select(X, y, g, f, kind)
            o = outer_at_layers(X, y, f, ch); d = np.abs(o - saved[sd])
            r["cands"][kind] = dict(maxabs=float(d.max()), medabs=float(np.median(d)), auc=rank_auc(y, o), layers=ch)
        best = min(r["cands"], key=lambda k: r["cands"][k]["maxabs"]); r["best_rule"] = best
        f = gkf_folds(g, best)
        if lay is not None:
            ch = inner_select(X, y, g, f, best); o = outer_at_layers(X, y, f, ch)
            r["rederived_layers"] = ch; r["layers_match"] = (ch == lay); r["full_maxabs"] = float(np.abs(o - saved[sd]).max()); r["full_auc"] = rank_auc(y, o)
        else:
            r["rederived_layers"] = r["cands"][best]["layers"]; r["layers_match"] = None; r["full_maxabs"] = r["cands"][best]["maxabs"]; r["full_auc"] = r["cands"][best]["auc"]
        r["fold"] = f.tolist()
        rec["per_repeat"].append(r)
        print(f"{run['run']:26s} {run['stream']:4s} seed {sd} best={best:6s} mac={r['cands']['mac']['maxabs']:.3e} stable={r['cands']['stable']['maxabs']:.3e} "
              f"layers saved={lay} re={r['rederived_layers']} full={r['full_maxabs']:.3e} auc saved={r['saved_auc_rank']:.4f} re={r['full_auc']:.4f} {time.time()-t0:.0f}s", flush=True)
    rec["clip"] = nm.tolist(); rec["spk"] = spk.tolist()
    with open(OUT, "a") as fh: fh.write(json.dumps(rec) + "\n")
print("REPEATS PERCLIP DONE", flush=True)
