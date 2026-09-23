"""PART17 T2: verify single-split GroupKFold(5) partitions by refitting saved nested / plain probes.
For each run x stream: refit the outer folds at the saved chosen layers under two tie rules (mac = numpy
default argsort on this Mac, stable = argsort kind='stable' then reversed as GroupKFold does), compare to
the saved per-clip OOF; then rerun the inner GroupKFold(4) layer choice under the matching rule and
compare the chosen layers. Writes one json line per run x stream to single_results.jsonl."""
import sys, json, time, os, hashlib
import numpy as np, pandas as pd
from eng import *
from sklearn.pipeline import Pipeline
G = "<local data dir>/paper work/paper1_local_runs"; R = "<local data dir>/release"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "single_results.jsonl")
P2 = f"{G}/probe2"
RUNS = []
for d in ["pitt", "adresso", "adress2020", "kcl", "pcgita", "edaic", "neurovoz"]:
    RUNS.append(dict(run=f"q2a_probe2_{d}", model="Qwen2-Audio", dataset=d, states=f"{P2}/{d}_states.npz",
                     json=f"{P2}/{d}_nested.json", oof=f"{P2}/{d}_%s_nested_oof.csv", streams=["enc", "llm", "ans"],
                     script=f"{P2}/nested_probe_local.py"))
RUNS.append(dict(run="q2a_probe2_kcl_local", model="Qwen2-Audio", dataset="kcl", states=f"{G}/kcl_probe_recut/kcl_states.npz",
                 json=f"{P2}/kcl_local_nested.json", oof=f"{P2}/kcl_local_%s_nested_oof.csv", streams=["enc", "llm"], script=f"{P2}/nested_probe_local.py"))
for d in ["edaic", "neurovoz"]:
    RUNS.append(dict(run=f"q2a_probe2_podruns_{d}", model="Qwen2-Audio", dataset=d, states=f"{P2}/{d}_states.npz",
                     json=f"{P2}/pod_runs/{d}_nested.json", oof=f"{P2}/pod_runs/{d}_%s_nested_oof.csv", streams=["enc", "llm", "ans"],
                     script=f"{P2}/nested_probe_local.py", note="states on disk are the Mac re-extraction; the pod's own states are not on disk"))
RUNS.append(dict(run="q2a_probe2_podruns_pitt", model="Qwen2-Audio", dataset="pitt", states=f"{P2}/pitt_states.npz",
                 json=None, oof=f"{P2}/pod_runs/pitt_%s_nested_oof.csv", streams=["enc"], script=f"{P2}/nested_probe_local.py",
                 note="no nested json saved; layers from inner selection; states on disk are the Mac phase-2 re-extraction"))
for d in ["pitt", "edaic"]:
    RUNS.append(dict(run=f"omni_final_{d}", model="Qwen2.5-Omni", dataset=d, states=f"{R}/overnight2/part10/o25_{d}_states.npz",
                     json=f"{R}/omni_final/omni_{d}_nested.json", oof=f"{R}/omni_final/omni_{d}_%s_nested_oof.csv",
                     streams=["enc", "proj", "llm", "ans"], script=f"{R}/scripts/nested_probe_local.py",
                     note="states = part10 re-extraction; p_yes identical to omni_final zero-shot to 1e-16"))
for d in ["adress2020", "adresso", "kcl", "neurovoz", "pcgita"]:
    RUNS.append(dict(run=f"q3o_new_{d}", model="Qwen3-Omni", dataset=d, states=f"{R}/overnight2/q3o/q3o_{d}_states.npz",
                     json=f"{R}/overnight2/q3o_new/q3o_{d}_nested.json", oof=f"{R}/overnight2/q3o_new/q3o_{d}_%s_nested_oof.csv",
                     streams=["enc", "proj", "llm", "ans"], script=f"{R}/scripts/nested_probe_local.py"))

def sha1mb(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read(1 << 20)); return h.hexdigest()

def cmp(a, b):
    d = np.abs(a - b)
    return dict(maxabs=float(d.max()), medabs=float(np.median(d)), pearson=float(np.corrcoef(a, b)[0, 1]))

only = set(sys.argv[1:])
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        j = json.loads(line); done.add((j["run"], j["stream"]))
for run in RUNS:
    if only and run["run"] not in only: continue
    t0 = time.time()
    z = np.load(run["states"], allow_pickle=True)
    y = z["label"].astype(int); spk = z["spk"].astype(str); nm = z["name"].astype(str)
    J = json.load(open(run["json"])) if run["json"] else {}
    for key in run["streams"]:
        if (run["run"], key) in done: continue
        oofp = run["oof"] % key
        s = pd.read_csv(oofp)
        rec = dict(run=run["run"], model=run["model"], dataset=run["dataset"], stream=key, states=run["states"], oof_file=oofp,
                   json=run["json"], script=run["script"], note=run.get("note", ""), n=int(len(y)), n_spk=int(len(np.unique(spk))),
                   sha_states=sha1mb(run["states"]), sha_oof=sha1mb(oofp), sha_json=sha1mb(run["json"]) if run["json"] else None)
        rec["align_names"] = bool((s["clip"].astype(str).values == nm).all())
        rec["align_spk"] = bool((s["speaker"].astype(str).values == spk).all())
        rec["align_label"] = bool((s["label"].values == y).all())
        saved = s["p_probe"].values.astype(float)
        rec["saved_auc_rank"] = rank_auc(y, saved)
        X = z[key].astype(np.float32)
        lay = J[key]["chosen_layers"] if key in J else None
        rec["saved_layers"] = lay
        cands = {}
        for kind in ("mac", "stable"):
            f = gkf_folds(spk, kind)
            if lay is not None:
                o = outer_at_layers(X, y, f, lay)
                cands[kind] = dict(cmp(o, saved), auc=rank_auc(y, o))
        if lay is not None:
            best = min(cands, key=lambda k: cands[k]["maxabs"])
        else:
            # no saved layers: run inner selection under both rules
            for kind in ("mac", "stable"):
                f = gkf_folds(spk, kind); ch = inner_select(X, y, spk, f, kind); o = outer_at_layers(X, y, f, ch)
                cands[kind] = dict(cmp(o, saved), auc=rank_auc(y, o), chosen=ch)
            best = min(cands, key=lambda k: cands[k]["maxabs"])
        rec["candidates"] = cands; rec["best_rule"] = best
        f = gkf_folds(spk, best)
        ch = cands[best].get("chosen") or inner_select(X, y, spk, f, best)
        rec["rederived_layers"] = ch
        rec["layers_match"] = (lay == ch) if lay is not None else None
        o = outer_at_layers(X, y, f, ch)
        rec["full_rederive"] = dict(cmp(o, saved), auc=rank_auc(y, o))
        rec["fold"] = f.tolist(); rec["clip"] = nm.tolist(); rec["spk"] = spk.tolist()
        rec["secs"] = round(time.time() - t0, 1)
        with open(OUT, "a") as fh: fh.write(json.dumps(rec) + "\n")
        print(f"{run['run']:28s} {key:4s} best={best:6s} mac={cands['mac']['maxabs']:.3e} stable={cands['stable']['maxabs']:.3e} "
              f"layers saved={lay} rederived={ch} full={rec['full_rederive']['maxabs']:.3e} auc saved={rec['saved_auc_rank']:.4f} "
              f"re={rec['full_rederive']['auc']:.4f} {rec['secs']}s", flush=True)
    del z
print("SINGLE DONE", flush=True)
