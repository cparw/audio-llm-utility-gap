"""PART17 T2: write release/folds/*.csv (clip_id, speaker_id, fold) for every partition the verification
confirmed, plus folds/README.md, the fragment rows part17/rows/T2.tsv and the sidecar part17/T2_folds.json.
Reads only the verification outputs in this folder and the source files they name. ZERO writes under omni_final."""
import os, sys, json, csv, hashlib, datetime, platform
import numpy as np, pandas as pd, sklearn, scipy
from gkf import gkf_folds
from eng import perm_codes

HERE = os.path.dirname(os.path.abspath(__file__))
G = "<local data dir>/paper work/paper1_local_runs"; R = "<local data dir>/release"
OUTD = f"{R}/folds"; P17 = f"{R}/edaic_rerun/part17"; ROWS = f"{P17}/rows/T2.tsv"; SIDE = f"{P17}/T2_folds.json"
for p in (OUTD, ROWS, SIDE): assert "omni_final" not in p
os.makedirs(OUTD, exist_ok=True)
D7 = ["pitt", "adresso", "adress2020", "pcgita", "neurovoz", "kcl", "edaic"]
NICE = {"pitt": "Pitt", "adresso": "ADReSSo", "adress2020": "ADReSS-2020", "pcgita": "PC-GITA", "neurovoz": "NeuroVoz",
        "kcl": "MDVR-KCL", "edaic": "E-DAIC (first 30 s)"}

def sha1mb(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read(1 << 20)); return h.hexdigest()
def jl(name):
    p = f"{HERE}/{name}"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
SINGLE = jl("single_results.jsonl"); EGE = jl("egemaps_results.jsonl"); REPC = jl("repeats_perclip_results.jsonl")
AUCO = jl("auc_only_results.jsonl"); NOISE = {(r["run"], r["stream"]): r for r in jl("noise_results.jsonl")}
EDF = json.load(open(f"{HERE}/edaicfull_results.json"))
SOURCES = {}
def src(p):
    if p and os.path.exists(p) and p not in SOURCES: SOURCES[p] = sha1mb(p)
    return p

FILES = []   # dict(file, dataset, family, rule, seed, clips, spk, fold, status, evidence=[...], runs_used=[...])
ROWS_OUT = []

def write_fold_csv(name, clips, spk, fold):
    path = f"{OUTD}/{name}"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip_id", "speaker_id", "fold"])
        for c, s, f in zip(clips, spk, fold): w.writerow([c, s, int(f)])
    return path

def same_partition(clipsA, foldA, clipsB, foldB):
    """identical fold label per clip (clip sets must match)."""
    a = dict(zip(clipsA, map(int, foldA))); b = dict(zip(clipsB, map(int, foldB)))
    return set(a) == set(b) and all(a[c] == b[c] for c in a)

def fold_of_172(clips, spk, fold):
    out = {}
    for c, s, f in zip(clips, spk, fold):
        if s.endswith("172"): out.setdefault(s, set()).add(int(f))
    return {k: sorted(v) for k, v in out.items()}

# ------------------------------------------------------------------ canonical clip order / ids per dataset
BASE = {}
for d in D7:
    z = np.load(src(f"{G}/probe2/{d}_states.npz"), allow_pickle=True)
    BASE[d] = dict(clips=z["name"].astype(str).tolist(), spk=z["spk"].astype(str).tolist())

def single_runs(ds, rule):
    return [r for r in SINGLE if r["dataset"] == ds and r["best_rule"] == rule]

def summarize_runs(runs):
    """group stream records by run -> evidence dicts"""
    ev = {}
    for r in runs:
        e = ev.setdefault(r["run"], dict(run=r["run"], model=r["model"], states=r["states"], json=r.get("json"), script=r.get("script"),
                                         note=r.get("note", ""), streams=[], maxabs=0.0, layers_all_match=True, oof_files=[], alt_min=1.0,
                                         auc_pairs=[]))
        st = r["stream"]; best = r["best_rule"]; alt = "mac" if best == "stable" else "stable"
        mx = r["candidates"][best]["maxabs"]
        e["streams"].append(dict(stream=st, maxabs_at_saved_layers=mx, maxabs_alt_rule=r["candidates"][alt]["maxabs"],
                                 saved_layers=r.get("saved_layers"), rederived_layers=r.get("rederived_layers"), layers_match=r.get("layers_match"),
                                 saved_auc=r.get("saved_auc_rank"), refit_auc=r["candidates"][best].get("auc"),
                                 noise_floor_float64=NOISE.get((r["run"], st), {}).get("mac_vs_mac_float64")))
        e["maxabs"] = max(e["maxabs"], mx); e["alt_min"] = min(e["alt_min"], r["candidates"][alt]["maxabs"])
        if r.get("layers_match") is False: e["layers_all_match"] = False
        e["oof_files"].append(r["oof_file"]); src(r["oof_file"]); src(r["states"]); src(r.get("json")); src(r.get("script") if r.get("script", "").startswith("/") else None)
    return list(ev.values())

# ------------------------------------------------------------------ family 1: single split, Mac rule
for d in D7:
    clips, spk = BASE[d]["clips"], BASE[d]["spk"]; fold = gkf_folds(np.array(spk), "mac")
    ev = summarize_runs(single_runs(d, "mac"))
    for r in [r for r in SINGLE if r["dataset"] == d and r["best_rule"] == "mac"]:
        assert same_partition(clips, fold, r["clip"], r["fold"]), (d, r["run"])
    eg = [r for r in EGE if r["dataset"] == d][0]
    assert eg["best_rule"] == "mac" and same_partition(clips, fold, eg["clip"], eg["fold"]), d
    src(eg["states"]); src(eg["oof_file"]); src(eg["script"])
    ev.append(dict(run=eg["run"], model=eg["model"], states=eg["states"], script=eg["script"], note="", streams=[dict(stream="egemaps",
              maxabs_at_saved_layers=eg["candidates"]["mac"]["maxabs"], maxabs_alt_rule=eg["candidates"]["stable"]["maxabs"],
              saved_auc=eg["saved_auc_rank"], refit_auc=eg["candidates"]["mac"]["auc"])], maxabs=eg["candidates"]["mac"]["maxabs"],
              layers_all_match=True, oof_files=[eg["oof_file"]], alt_min=eg["candidates"]["stable"]["maxabs"]))
    FILES.append(dict(file=f"{d}_groupkfold5_mac.csv", dataset=d, family="single_mac", rule="mac", seed=None, clips=clips, spk=spk, fold=fold,
                      status="CONFIRMED", evidence=ev))

# ------------------------------------------------------------------ family 2: single split, pod rule
EXTERNAL = {  # fold files written on pods by sklearn GroupKFold itself
    "pitt": (f"{R}/edaic_rerun/part16/POD2/pitt468_folds.csv", "part16 POD2 make_folds.py (pod), same call as sft_projector.py"),
    "edaic": (f"{R}/edaic_rerun/part16/POD1/sft1c_folds.json", "part16 POD1 sft1c fold list (pod)")}
for d in D7:
    clips, spk = BASE[d]["clips"], BASE[d]["spk"]; fold = gkf_folds(np.array(spk), "stable")
    runs = single_runs(d, "stable")
    for r in runs:
        assert same_partition(clips, fold, r["clip"], r["fold"]), (d, r["run"])
    ev = summarize_runs(runs)
    conf = [e for e in ev if not e["run"].startswith("q2a_probe2_podruns")]
    ext = None
    if d in EXTERNAL:
        p, what = EXTERNAL[d]; src(p)
        if p.endswith(".csv"):
            q = pd.read_csv(p, dtype={"speaker": str}); m = dict(zip(q["speaker"], q["fold"]))
        else:
            q = json.load(open(p)); m = {s: int(k) for k, v in q.items() for s in v}
        agree = int(sum(m[s] == f for s, f in zip(spk, fold))); ext = dict(file=p, what=what, rows_agree=agree, rows=len(spk))
    curves = [a for a in AUCO if a["kind"] == "curve" and a["dataset"] == d]
    for a in curves: src(a["file"]); src(a["states"])
    FILES.append(dict(file=f"{d}_groupkfold5_pod.csv", dataset=d, family="single_pod", rule="stable", seed=None, clips=clips, spk=spk, fold=fold,
                      status="CONFIRMED" if conf else "UNVERIFIED", evidence=ev, external=ext, curves=curves))

# ------------------------------------------------------------------ family 3: five repeats, pod rule (nested_repeats_all.py)
def auc_reps(d, rule="stable"):
    return [a for a in AUCO if a["kind"] == "reps" and a["dataset"] == d]
for d in D7:
    clips, spk = BASE[d]["clips"], BASE[d]["spk"]; sp = np.array(spk)
    reps = auc_reps(d)
    for a in reps: src(a["file"]); src(a["states"])
    perclip = [r for r in REPC if r["dataset"] == d and r["per_repeat"][0]["best_rule"] == "stable" and
               set(r["spk"]) == set(spk)]
    for s in range(5):
        fold = gkf_folds(perm_codes(sp, s), "stable")
        pcev = []
        for r in perclip:
            rr = [x for x in r["per_repeat"] if x["seed"] == s]
            if not rr: continue
            rr = rr[0]; assert same_partition(clips, fold, r["clip"], rr["fold"]), (d, r["run"], s)
            pcev.append(dict(run=r["run"], model=r["model"], stream=r["stream"], states=r["states"], oof_file=r["oof_file"],
                             maxabs_at_saved_layers=rr["cands"]["stable"]["maxabs"], maxabs_alt_rule=rr["cands"]["mac"]["maxabs"],
                             saved_layers=rr["saved_layers"], rederived_layers=rr["rederived_layers"], layers_match=rr["layers_match"],
                             saved_auc=rr["saved_auc_rank"], refit_auc=rr["cands"]["stable"]["auc"]))
            src(r["states"]); src(r["oof_file"]); src(r.get("summary"))
        aev = []
        for a in reps:
            st = [x for x in a["cands"]["stable"] if x["seed"] == s]
            if not st: continue
            st = st[0]   # same speaker-label multiset as BASE[d] (labelsets.py), so the same per-speaker split
            mc = [x for x in a["cands"]["mac"] if x["seed"] == s]
            aev.append(dict(run=a["run"], model=a["model"], stream=a["stream"], file=a["file"], saved_auc4=a["saved_per_repeat"][s],
                            refit_auc_stable=st["auc"], refit_auc_mac=(mc[0]["auc"] if mc else None)))
        status = "CONFIRMED" if pcev else "UNVERIFIED"
        name = f"{d}_groupkfold5_pod_seed{s}.csv" if status == "CONFIRMED" else f"{d}_groupkfold5_pod_seed{s}_UNVERIFIED.csv"
        FILES.append(dict(file=name, dataset=d, family="seed_pod", rule="stable", seed=s, clips=clips, spk=spk, fold=fold, status=status,
                          perclip=pcev, aucev=aev))

# ------------------------------------------------------------------ family 4: five repeats, Mac rule (Omni Pitt encoder, part10 nested5)
for r in [r for r in REPC if r["run"] == "omni_part10_pitt_nested5"]:
    for rr in r["per_repeat"]:
        s = rr["seed"]; assert rr["best_rule"] == "mac"
        clips, spk = BASE["pitt"]["clips"], BASE["pitt"]["spk"]; fold = gkf_folds(perm_codes(np.array(spk), s), "mac")
        assert same_partition(clips, fold, r["clip"], rr["fold"])
        src(r["states"]); src(r["oof_file"])
        FILES.append(dict(file=f"pitt_groupkfold5_mac_seed{s}.csv", dataset="pitt", family="seed_mac", rule="mac", seed=s, clips=clips, spk=spk,
                          fold=fold, status="CONFIRMED", perclip=[dict(run=r["run"], model=r["model"], stream=r["stream"], states=r["states"],
                          oof_file=r["oof_file"], maxabs_at_saved_layers=rr["cands"]["mac"]["maxabs"], maxabs_alt_rule=rr["cands"]["stable"]["maxabs"],
                          saved_layers=None, rederived_layers=rr["rederived_layers"], layers_match=None, saved_auc=rr["saved_auc_rank"],
                          refit_auc=rr["cands"]["mac"]["auc"])], aucev=[]))

# ------------------------------------------------------------------ family 5: Qwen3-Omni POD3 Pitt (unpadded ids Control15)
pod3 = [r for r in REPC if r["run"] == "q3o_POD3_pitt"]
if pod3:
    clips, spk = pod3[0]["clip"], pod3[0]["spk"]
    for s in range(5):
        fold = gkf_folds(perm_codes(np.array(spk), s), "stable"); pcev = []
        for r in pod3:
            rr = [x for x in r["per_repeat"] if x["seed"] == s]
            if not rr: continue
            rr = rr[0]; assert rr["best_rule"] == "stable" and same_partition(clips, fold, r["clip"], rr["fold"])
            pcev.append(dict(run=r["run"], model=r["model"], stream=r["stream"], states=r["states"], oof_file=r["oof_file"],
                             maxabs_at_saved_layers=rr["cands"]["stable"]["maxabs"], maxabs_alt_rule=rr["cands"]["mac"]["maxabs"],
                             saved_layers=rr["saved_layers"], rederived_layers=rr["rederived_layers"], layers_match=rr["layers_match"],
                             saved_auc=rr["saved_auc_rank"], refit_auc=rr["cands"]["stable"]["auc"]))
            src(r["states"]); src(r["oof_file"]); src(r.get("summary"))
        FILES.append(dict(file=f"pitt_groupkfold5_pod_Control15ids_seed{s}.csv", dataset="pitt", family="seed_pod_pod3", rule="stable", seed=s,
                          clips=clips, spk=spk, fold=fold, status="CONFIRMED", perclip=pcev, aucev=[]))

# ------------------------------------------------------------------ family 6: E-DAIC full window, shuffle=True
src(EDF["states_tgz"]); src(EDF["perclip"]); src(EDF["json"]); src(EDF["script"])
edf_ok = all(EDF["streams"][k]["layers_match"] for k in ("enc", "proj"))
for rep in range(5):
    FILES.append(dict(file=f"edaic_full_groupkfold5_shuffle_rs{rep}.csv", dataset="edaic_full", family="shuffle", rule=f"shuffle random_state={rep}",
                      seed=rep, clips=EDF["clip"], spk=EDF["spk"], fold=EDF["folds"][str(rep)] if str(rep) in EDF["folds"] else EDF["folds"][rep],
                      status="CONFIRMED" if edf_ok else "UNVERIFIED", edf=EDF["streams"]))

# family 7 (E-DAIC 390-row conflict manifest) deliberately not released: not the Part 14 set, AUC-only evidence, see README

# ------------------------------------------------------------------ write csvs
for f in FILES:
    f["path"] = write_fold_csv(f["file"], f["clips"], f["spk"], f["fold"])
    f["n"] = len(f["clips"]); f["n_spk"] = len(set(f["spk"])); f["fold_sizes"] = np.bincount(np.array(f["fold"], int), minlength=5).tolist()
    f["spk_split_across_folds"] = sorted(s for s in set(f["spk"]) if len({int(x) for x, t in zip(f["fold"], f["spk"]) if t == s}) > 1)
    assert not f["spk_split_across_folds"], f["file"]
    if f["dataset"] == "pitt": f["p172"] = fold_of_172(f["clips"], f["spk"], f["fold"])
json.dump([{k: v for k, v in f.items() if k not in ("clips", "spk", "fold")} for f in FILES], open(f"{HERE}/files_manifest.json", "w"), indent=1, default=str)
print(len(FILES), "fold files written to", OUTD)
