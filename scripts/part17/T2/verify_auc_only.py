"""PART17 T2: runs whose only saved record is an AUC (no per-clip OOF).
(a) single-split per-layer curves (curves_from_states.py): auc_oof per layer, saved at full precision.
(b) five-repeat nested probes (nested_repeats_all.py): per_repeat AUC saved at 4 dp.
Both tie rules are tried. This is AUC-level evidence only; it cannot confirm a partition clip by clip."""
import sys, json, os, time, hashlib
import numpy as np, pandas as pd
from eng import *
G = "<local data dir>/paper work/paper1_local_runs"; R = "<local data dir>/release"; P2 = f"{G}/probe2"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auc_only_results.jsonl")
def sha1mb(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read(1 << 20)); return h.hexdigest()
SUF = {"enc": "encoder", "proj": "proj", "llm": "llm", "ans": "ans"}
D7 = ["pitt", "adresso", "adress2020", "pcgita", "neurovoz", "kcl", "edaic"]
CURVES, REPS = [], []
for d in D7:
    for k in ("enc", "llm", "ans"):
        CURVES.append(dict(run=f"q2a_podA_curves_{d}", model="Qwen2-Audio", dataset=d, stream=k, states=f"{P2}/{d}_states.npz", file=f"{R}/omni_final/q2a_{d}_{SUF[k]}_perlayer.csv"))
    REPS.append(dict(run=f"q2a_podA_repeats_{d}", model="Qwen2-Audio", dataset=d, states=f"{P2}/{d}_states.npz", file=f"{R}/omni_final/q2a_{d}_nested_repeats.json", streams=["enc", "llm", "ans"]))
for d in ["pitt", "edaic"]:
    for k in ("enc", "proj", "llm", "ans"):
        CURVES.append(dict(run=f"omni_podA_curves_{d}", model="Qwen2.5-Omni", dataset=d, stream=k, states=f"{R}/overnight2/part10/o25_{d}_states.npz", file=f"{R}/omni_final/omni_{d}_{SUF[k]}_perlayer.csv"))
    REPS.append(dict(run=f"omni_podA_repeats_{d}", model="Qwen2.5-Omni", dataset=d, states=f"{R}/overnight2/part10/o25_{d}_states.npz", file=f"{R}/omni_final/omni_{d}_nested_repeats.json", streams=["enc", "proj", "llm", "ans"]))
for d in ["adress2020", "adresso", "kcl", "neurovoz", "pcgita"]:
    REPS.append(dict(run=f"q3o_podD2_repeats_{d}", model="Qwen3-Omni", dataset=d, states=f"{R}/overnight2/q3o/q3o_{d}_states.npz", file=f"{R}/overnight2/q3o_new/q3o_{d}_nested_repeats.json", streams=["enc", "proj", "llm", "ans"]))
KIMI = {"pitt": "part3", "edaic": "part3", "pcgita": "part3", "adress2020": "kimi_new", "adresso": "kimi_new", "kcl": "kimi_new", "neurovoz": "kimi_new"}
for d, src in KIMI.items():
    st = f"{R}/overnight2/kimi/kimi_{d}_states.npz" if src == "part3" else f"{R}/overnight2/kimi_new/kimi_{d}_states.npz"
    REPS.append(dict(run=f"kimi_repeats_{d}", model="Kimi-Audio", dataset=d, states=st, file=f"{R}/overnight2/{src}/kimi_{d}_nested_repeats.json", streams=["llm", "ans"]))
    if src == "part3":
        for k in ("llm", "ans"):
            CURVES.append(dict(run=f"kimi_curves_{d}", model="Kimi-Audio", dataset=d, stream=k, states=st, file=f"{R}/overnight2/part3/kimi_{d}_{k}_perlayer.csv"))
for m in ["q2a", "o25", "q3o"]:
    REPS.append(dict(run=f"{m}_repeats_edaic_conflict390", model={"q2a": "Qwen2-Audio", "o25": "Qwen2.5-Omni", "q3o": "Qwen3-Omni"}[m], dataset="edaic_conflict390",
                     states=f"{R}/edaic_conflict_new/{m}_edaic_conflict_states.npz", file=f"{R}/edaic_conflict_new/{m}_edaic_conflict_nested_repeats.json", streams=["enc", "proj", "llm", "ans"]))
which = sys.argv[1] if len(sys.argv) > 1 else "all"
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        j = json.loads(line); done.add((j["kind"], j["run"], j["stream"]))
cache = {}
def load(p):
    if p not in cache:
        cache.clear(); z = np.load(p, allow_pickle=True)
        cache[p] = dict(z=z, y=z["label"].astype(int), spk=z["spk"].astype(str))
    return cache[p]
if which in ("all", "curves"):
    for c in CURVES:
        if ("curve", c["run"], c["stream"]) in done or not os.path.exists(c["file"]): continue
        t0 = time.time(); L = load(c["states"]); y, spk = L["y"], L["spk"]; X = L["z"][c["stream"]].astype(np.float32)
        s = pd.read_csv(c["file"]); saved = s["auc_oof"].values.astype(float)
        rec = dict(kind="curve", run=c["run"], model=c["model"], dataset=c["dataset"], stream=c["stream"], states=c["states"], file=c["file"],
                   sha_states=sha1mb(c["states"]), sha_file=sha1mb(c["file"]), n=int(len(y)), n_spk=int(len(np.unique(spk))), n_layers=int(X.shape[1]), cands={})
        for kind in ("mac", "stable"):
            f = gkf_folds(spk, kind); oo = layer_oofs(X, y, f)
            mine = np.array([rank_auc(y, oo[l]) for l in range(X.shape[1])])
            d = np.abs(mine - saved)
            rec["cands"][kind] = dict(maxabs_auc=float(d.max()), n_layers_equal_4dp=int(sum(f"{a:.4f}" == f"{b:.4f}" for a, b in zip(mine, saved))), mine=mine.tolist())
        rec["best_rule"] = min(rec["cands"], key=lambda k: rec["cands"][k]["maxabs_auc"])
        with open(OUT, "a") as fh: fh.write(json.dumps(rec) + "\n")
        print(f"CURVE {c['run']:28s} {c['stream']:4s} " + " ".join(f"{k}: max|dAUC| {v['maxabs_auc']:.2e} eq4dp {v['n_layers_equal_4dp']}/{X.shape[1]}" for k, v in rec["cands"].items()) + f" {time.time()-t0:.0f}s", flush=True)
if which in ("all", "reps"):
    PRIMARY = lambda r: "llm" if r["model"] == "Kimi-Audio" else "enc"   # the Table 1 probe stream of each model
    ORDER = [(r, k) for r in REPS for k in r["streams"] if k == PRIMARY(r)] + [(r, k) for r in REPS for k in r["streams"] if k != PRIMARY(r)]
    for r, key in ORDER:
        if not os.path.exists(r["file"]): continue
        J = json.load(open(r["file"]))
        if True:
            if ("reps", r["run"], key) in done or key not in J: continue
            t0 = time.time(); L = load(r["states"]); y, spk = L["y"], L["spk"]; X = L["z"][key].astype(np.float32); nl = X.shape[1]
            saved = J[key]["per_repeat"]; nrep = len(saved)
            rec = dict(kind="reps", run=r["run"], model=r["model"], dataset=r["dataset"], stream=key, states=r["states"], file=r["file"],
                       sha_states=sha1mb(r["states"]), sha_file=sha1mb(r["file"]), n=int(len(y)), n_spk=int(len(np.unique(spk))), saved_per_repeat=saved, cands={})
            def run_rule(kind, seeds):
                out = []
                for sd in seeds:
                    g = perm_codes(spk, sd); f = gkf_folds(g, kind); ch = inner_select(X, y, g, f, kind); o = outer_at_layers(X, y, f, ch)
                    out.append(dict(seed=sd, auc=rank_auc(y, o), auc4=round(rank_auc(y, o), 4), layers=ch, fold=f.tolist()))
                return out
            st = run_rule("stable", range(nrep))
            rec["cands"]["stable"] = st
            rec["stable_match_4dp"] = [f"{a['auc']:.4f}" == f"{b:.4f}" for a, b in zip(st, saved)]
            mac_seeds = [0]   # contrast only; the stable rule is judged on |dAUC|, see README
            mc = run_rule("mac", mac_seeds)
            rec["cands"]["mac"] = mc
            rec["mac_match_4dp"] = [f"{a['auc']:.4f}" == f"{saved[a['seed']]:.4f}" for a in mc]
            with open(OUT, "a") as fh: fh.write(json.dumps(rec) + "\n")
            print(f"REPS {r['run']:32s} {key:4s} saved {saved} stable {[a['auc4'] for a in st]} match {sum(rec['stable_match_4dp'])}/{nrep} | "
                  f"mac(seeds {list(mac_seeds)}) {[a['auc4'] for a in mc]} match {sum(rec['mac_match_4dp'])}/{len(mc)} {time.time()-t0:.0f}s", flush=True)
print("AUC ONLY DONE", flush=True)
