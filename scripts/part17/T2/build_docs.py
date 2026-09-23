"""PART17 T2: folds/README.md, part17/rows/T2.tsv, part17/T2_folds.json from files_manifest.json + the verification outputs.
ZERO writes under omni_final."""
import os, sys, json, hashlib, datetime, platform
import numpy as np, sklearn, scipy, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = "<local data dir>/release"; OUTD = f"{R}/folds"; P17 = f"{R}/edaic_rerun/part17"
ROWS = f"{P17}/rows/T2.tsv"; SIDE = f"{P17}/T2_folds.json"; SCRIPTS_DIR = f"{P17}/T2"
for p in (OUTD, ROWS, SIDE, SCRIPTS_DIR): assert "omni_final" not in p
M = json.load(open(f"{HERE}/files_manifest.json"))
NOISE = [json.loads(l) for l in open(f"{HERE}/noise_results.jsonl")]
AUCO = [json.loads(l) for l in open(f"{HERE}/auc_only_results.jsonl")] if os.path.exists(f"{HERE}/auc_only_results.jsonl") else []
NICE = {"pitt": "Pitt", "adresso": "ADReSSo", "adress2020": "ADReSS-2020", "pcgita": "PC-GITA", "neurovoz": "NeuroVoz", "kcl": "MDVR-KCL",
        "edaic": "E-DAIC first 30 s", "edaic_full": "E-DAIC full window", "edaic_conflict390": "E-DAIC conflict manifest, 390 rows"}
def g(x, n=2): return "n/a" if x is None else f"{x:.{n}e}"
def sha1mb(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read(1 << 20)); return h.hexdigest()

# ------------------------------------------------------------------ numbers used in the prose
pod_runs = [e for f in M if f["family"] == "single_pod" for e in f["evidence"] if not e["run"].startswith("q2a_probe2_podruns")]
pod_max = max([e["maxabs"] for e in pod_runs] + [p["maxabs_at_saved_layers"] for f in M if f["family"] in ("seed_pod", "seed_pod_pod3") for p in (f.get("perclip") or [])])
alt_min = min(min(e["alt_min"] for f in M if f["family"] in ("single_mac", "single_pod") for e in f["evidence"]),
              min(p["maxabs_alt_rule"] for f in M if f.get("perclip") for p in f["perclip"]))
f64_max = max(n["mac_vs_mac_float64"] for n in NOISE)

# ------------------------------------------------------------------ README
L = []
A = L.append
A("# Speaker fold assignments")
A("")
A("Every probe in the paper is scored out of fold with scikit-learn `GroupKFold(n_splits=5)` grouped by speaker. "
  "The files here give the fold of every clip for each split that was actually run, so a probe number can be rebuilt "
  "without re-deriving the folds. Each file was checked by refitting the saved probe with it and comparing the result "
  "with the saved out-of-fold predictions.")
A("")
A("## Columns")
A("")
A("- `clip_id`: the clip exactly as the run's states file names it.")
A("- `speaker_id`: the group label exactly as the run passed it to GroupKFold.")
A("- `fold`: 0 to 4, in the order GroupKFold yields its test folds.")
A("")
A("## Why one dataset has several files")
A("")
A("With shuffling off, GroupKFold sorts speakers by clip count with `numpy.argsort` and hands them out, largest first, "
  "to the fold that has the fewest clips so far. When speakers have the same clip count, the order argsort leaves them in "
  "decides the split. On ADReSSo, ADReSS-2020, MDVR-KCL and E-DAIC every speaker has exactly one clip, so that order decides "
  "the whole split; on Pitt, PC-GITA and NeuroVoz it decides most of it. That order was different on the Mac used for some "
  "runs (numpy 2.2.6, arm64) and on the Linux GPU pods used for the others. The same code, clips and speaker labels therefore "
  "gave two different splits, and both are released:")
A("")
A("- `_mac`: GroupKFold exactly as installed on the Mac (numpy default argsort).")
A("- `_pod`: the same algorithm with `numpy.argsort(counts, kind=\"stable\")`, reversed as GroupKFold reverses it. This reproduces every "
  "pod run that could be checked. It describes the result; the pods' numpy build itself was not inspected.")
A("- `_seed0` to `_seed4`: `nested_repeats_all.py` (and part16 `perlayer_nested.py`) first renumber the speakers with "
  "`numpy.random.default_rng(seed).permutation(numpy.unique(ids))` and then run GroupKFold on the new numbers, once per seed. "
  "The Table 1 probe values are the mean over these five splits. The same tie rule applies after renumbering.")
A("- `_shuffle_rs0` to `_shuffle_rs4`: the E-DAIC full-window probe used `GroupKFold(5, shuffle=True, random_state=r)`, which "
  "does not depend on argsort ties.")
A("")
A("## How the files were checked")
A("")
A("For each run whose hidden states are on disk, the probe was refitted with the file's folds (StandardScaler, then "
  "LogisticRegression with max_iter 2000 and balanced class weights, as in the run's script). For nested probes each outer fold "
  "was refitted at the layer the run saved, the predictions were compared clip by clip, and the inner GroupKFold(4) layer choice "
  "was then rerun with the same tie rule to check it picks the same layers.")
A("")
A(f"Runs made on the Mac match exactly (largest difference 1.1e-16). Runs made on pods cannot match to 1e-6 on the Mac: the "
  f"logistic regression stops at a gradient tolerance, and different numeric libraries stop at slightly different points. On "
  f"identical states and folds the largest pod-versus-Mac difference is {pod_max:.3f}; casting the same features to float64 on the "
  f"Mac alone moves the same predictions by up to {f64_max:.3f}. Every checked pod run also re-derives the saved layer choices "
  f"(exceptions listed below) and agrees on AUC. The other tie rule is off by at least {alt_min:.2f} on every run checked, so the "
  f"two rules are never confused.")
A("")
A("`CONFIRMED` means reproduced clip by clip as above. A name ending in `_UNVERIFIED` means no per-clip out-of-fold file was saved "
  "for that split, so it could not be checked clip by clip; the paragraph says what it does reproduce.")
A("")
A("Pitt: participant 172 appears under two speaker labels, `Control172` and `Dementia172`, because the conflict manifest lists "
  "that person in both diagnosis groups. The labels are kept exactly as the runs used them, so in some splits this one person "
  "sits in two folds; each Pitt paragraph says where. The Qwen3-Omni part16 POD3 Pitt run wrote the Pitt ids without zero padding "
  "(`Control15` instead of `Control015`), which sorts differently and gives different splits, so it has its own files.")
A("")
A("## Files")
A("")

MODEL_OF = lambda run: ("Qwen2-Audio" if run.startswith("q2a") else "Qwen2.5-Omni" if run.startswith(("omni", "o25")) else
                         "Qwen3-Omni" if run.startswith("q3o") else "Kimi-Audio" if run.startswith("kimi") else run)
def run_line(e):
    st = "/".join(s["stream"] for s in e["streams"])
    if e["run"].startswith("egemaps"):
        return f"`{e['run']}` (eGeMAPS baseline): max abs diff {g(e['maxabs'])}"
    lay = "saved layers re-derived" if e["layers_all_match"] else "saved layers re-derived except " + "; ".join(
        f"{s['stream']} {s['saved_layers']} vs {s['rederived_layers']}" for s in e["streams"] if s.get("layers_match") is False)
    extra = " (this pod_runs file is a byte copy of the Mac output probe2/pitt_enc_nested_oof.csv, not a pod result)" if e["run"] == "q2a_probe2_podruns_pitt" else ""
    return f"`{e['run']}` ({e['model']}, {st}): max abs diff {g(e['maxabs'])}, {lay}{extra}"
def fsz(f): return "/".join(str(x) for x in f["fold_sizes"])
def p172(f):
    p = f.get("p172")
    return (" Participant 172: " + ", ".join(f"`{k}` fold {v[0] if len(v) == 1 else v}" for k, v in sorted(p.items())) + ".") if p else ""
def curves_line(cv):
    if not cv: return ""
    runs = sorted(set(MODEL_OF(c["run"]) for c in cv)); st = max(c["cands"]["stable"]["maxabs_auc"] for c in cv)
    lo = min(c["cands"]["mac"]["maxabs_auc"] for c in cv); hi = max(c["cands"]["mac"]["maxabs_auc"] for c in cv)
    return (f" {len(cv)} per-layer AUC curves saved by pod runs ({', '.join(runs)}) agree with this split within {st:.4f} AUC at every layer; "
            f"the Mac order misses them by {lo:.4f} to {hi:.4f}.")
def auc_line(aev):
    if not aev: return ""
    d = max(abs(a["refit_auc_stable"] - a["saved_auc4"]) for a in aev)
    dm = [abs(a["refit_auc_mac"] - a["saved_auc4"]) for a in aev if a["refit_auc_mac"] is not None]
    runs = sorted(set(f"{MODEL_OF(a['run'])} {a['stream']}" for a in aev))
    return (f" Saved per-repeat AUCs ({len(aev)}: {', '.join(runs)}) are reproduced within {d:.4f}"
            + ((f"; the Mac order misses by {min(dm):.4f}" if min(dm) == max(dm) else f"; the Mac order misses by {min(dm):.4f} to {max(dm):.4f}") if dm else "") + ".")

A("### Single split, Mac tie order")
A("")
for f in [f for f in M if f["family"] == "single_mac"]:
    A(f"**`{f['file']}`** ({NICE[f['dataset']]}; {f['n']} clips, {f['n_spk']} speaker labels, fold sizes {fsz(f)}). {f['status']}. "
      f"sklearn GroupKFold(5) as installed on the Mac. Reproduces " + "; ".join(run_line(e) for e in f["evidence"]) + "." + p172(f))
    A("")
A("### Single split, pod tie order")
A("")
for f in [f for f in M if f["family"] == "single_pod"]:
    conf = [e for e in f["evidence"] if not e["run"].startswith("q2a_probe2_podruns")]
    notc = [e for e in f["evidence"] if e["run"].startswith("q2a_probe2_podruns")]
    t = (f"**`{f['file']}`** ({NICE[f['dataset']]}; {f['n']} clips, {f['n_spk']} speaker labels, fold sizes {fsz(f)}). {f['status']}. "
         f"GroupKFold(5) with the stable tie order. Reproduces " + "; ".join(run_line(e) for e in conf) + ".")
    if f.get("external"):
        x = f["external"]; t += f" The fold file `{os.path.basename(x['file'])}` written on a pod by GroupKFold itself ({x['what']}) agrees on {x['rows_agree']} of {x['rows']} clips."
    t += curves_line(f.get("curves"))
    for e in notc:
        t += (f" Not confirmed: `{e['run']}` ({'/'.join(s['stream'] for s in e['streams'])}) used states the pod extracted itself, which are not on disk; "
              f"on the Mac states it sits closest to this split (max abs diff {g(e['maxabs'])}, Mac order {g(e['alt_min'])}).")
    if f["dataset"] not in ("pitt", "edaic"):
        t += " The Qwen2.5-Omni single-split probe on this dataset ran on the same pod with the same labels, but its states are not on disk, so for that run this split is inferred, not refitted."
    A(t + p172(f)); A("")
for fam, title, intro in [
    ("seed_pod", "Five renumbered splits, pod tie order (the Table 1 probe values)",
     "Speakers renumbered with `default_rng(seed)`, then GroupKFold(5) with the stable tie order. These are the splits behind the five-repeat "
     "probe means in Table 1 for every model and dataset except the Qwen2.5-Omni Pitt encoder value (next section). A per-clip out-of-fold file was saved only for the part16 POD3 Qwen3-Omni PC-GITA "
     "repeats, so only the PC-GITA files are confirmed clip by clip; the others reproduce the saved per-repeat AUCs only and carry `_UNVERIFIED`. "
     "For every run and stream checked, the pod split is closer to the saved AUC than the Mac split (the Mac split was run on seed 0, and on all "
     "five seeds for the Qwen2-Audio Pitt encoder). The Qwen3-Omni Pitt and E-DAIC cells of Table 1 come from runs whose states are not on disk, "
     "so for those two cells these splits are inferred, not checked."),
    ("seed_mac", "Five renumbered splits, Mac tie order (Qwen2.5-Omni Pitt encoder probe, 0.7706)",
     "Speakers renumbered with `default_rng(seed)`, then sklearn GroupKFold(5) as installed on the Mac. These reproduce "
     "`overnight2/part10/pitt_enc_nested5_oof.npz` exactly, the file behind the Table 1 Qwen2.5-Omni Pitt encoder value 0.7706. Its layer choices "
     "were not saved, so they were re-derived with the inner GroupKFold(4) and the per-clip scores then match exactly. The older value 0.7761 "
     "(`omni_final/omni_pitt_nested_repeats.json`) is the same five-repeat probe on the same states, run on a pod, so on the pod files of the "
     "previous section."),
    ("seed_pod_pod3", "Five renumbered splits, pod tie order, Pitt ids without zero padding (part16 POD3 Qwen3-Omni)",
     "Same as the pod files above but on the ids `Control15`, `Dementia15` and so on, as written in the part16 POD3 Qwen3-Omni Pitt states. "
     "Only the projector stream has one repeat, so seeds 1 to 4 are checked on the other three streams."),
    ("shuffle", "E-DAIC full window, shuffled splits (part16 EDAICFULL)",
     "`GroupKFold(5, shuffle=True, random_state=r)` as in `part16/EDAICFULL/probe_boot.py`; no tie order is involved. The saved per-clip file holds only "
     "the mean over the five splits, so the five files are checked together: refitting all five at the saved layers and averaging reproduces the saved "
     "mean out-of-fold scores.")]:
    FF = [f for f in M if f["family"] == fam]
    if not FF: continue
    A(f"### {title}"); A(""); A(intro); A("")
    for f in FF:
        t = f"- **`{f['file']}`** ({NICE[f['dataset']]}, {'random_state' if fam == 'shuffle' else 'seed'} {f['seed']}). {f['status']}."
        if fam == "shuffle":
            t += " " + "; ".join(f"{k} max abs diff {g(v['maxabs_oof_mean'])}" + ("" if v["layers_match"] else " (inner layer choice differs in some folds, near ties)")
                                 for k, v in f["edf"].items()) + "."
        if f.get("perclip"):
            byrun = {}
            for p in f["perclip"]: byrun.setdefault(p["run"], []).append(p)
            for run, ps in byrun.items():
                mism = [p for p in ps if p.get("layers_match") is False]
                t += (f" Per clip vs `{run}` ({'/'.join(p['stream'] for p in ps)}): max abs diff {g(max(p['maxabs_at_saved_layers'] for p in ps))}"
                      + (", saved layers re-derived" if all(p.get("layers_match") for p in ps) else
                         ("" if all(p.get("layers_match") is None for p in ps) else ", saved layers re-derived except " + "; ".join(
                             f"{p['stream']} {p['saved_layers']} vs {p['rederived_layers']}" for p in mism)))
                      + f"; the other tie order misses by {min(p['maxabs_alt_rule'] for p in ps):.2f} or more.")
        t += auc_line(f.get("aucev"))
        if f["status"] == "UNVERIFIED" and not f.get("perclip"): t += " No per-clip out-of-fold file exists for this split."
        A(t + p172(f))
    A("")
A("## Which runs used which split")
A("")
A("| run | where | split |")
A("|---|---|---|")
A("| Qwen2-Audio single-split nested probes, `probe2/<ds>_{enc,llm,ans}_nested_oof.csv` (all seven datasets) | Mac | `_mac` |")
A("| eGeMAPS baseline, `overnight/perclip/egemaps_<ds>_oof.csv` | Mac | `_mac` |")
A("| Qwen2.5-Omni Pitt encoder probe 0.7706, `part10/pitt_enc_nested5_oof.npz` | Mac | `pitt_groupkfold5_mac_seed0..4` |")
A("| Qwen2.5-Omni single-split nested probes, `omni_final/omni_<ds>_*_nested_oof.csv` | pod | `_pod` |")
A("| Qwen3-Omni single-split nested probes, `overnight2/q3o_new/q3o_<ds>_*_nested_oof.csv` | pod | `_pod` |")
A("| Qwen2-Audio MDVR-KCL `kcl_local` nested probe | pod | `kcl_groupkfold5_pod` |")
A("| per-layer AUC curves written on pods (Qwen2-Audio, Qwen2.5-Omni, Kimi-Audio) | pod | `_pod` |")
A("| five-repeat nested probes behind Table 1 (Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, Kimi-Audio), all cells except the one above | pod | `_pod_seed0..4` |")
A("| Qwen3-Omni part16 POD3 Pitt and PC-GITA repeats | pod | `pitt_groupkfold5_pod_Control15ids_seed0..4`, `pcgita_groupkfold5_pod_seed0..4` |")
A("| E-DAIC full window, part16 EDAICFULL | pod | `edaic_full_groupkfold5_shuffle_rs0..4` |")
A("")
A("## Not covered")
A("")
A("- Projector fine-tunes (`sft_projector.py`) use the same GroupKFold call but saved no fold record, and a fine-tune cannot be refitted "
  "to check one. The pod-written `pitt468_folds.csv` from part16 POD2, made with the same call, equals `pitt_groupkfold5_pod.csv`.")
A("- Qwen2.5-Omni states are on disk only for Pitt and E-DAIC, and Qwen3-Omni states only for the five other datasets plus the "
  "part16 POD3 Pitt and PC-GITA re-extractions. Splits of runs without states are named above by inference only.")
A("- The older E-DAIC conflict manifest (390 rows, `edaic_conflict_new/`) has five-repeat probes but no per-clip out-of-fold file, and its saved AUCs "
  "sit near chance where the layer choice is unstable; its splits are not released here. The Part 14 conflict set has no nested probe.")
A("- Runs made by others on other machines are not included.")
A("")
open(f"{OUTD}/README.md", "w").write("\n".join(L))
assert "\u2014" not in "\n".join(L) and "\u2013" not in "\n".join(L)

# ------------------------------------------------------------------ fragment rows: one per dataset x split family x run
from collections import OrderedDict
G_ = OrderedDict()
def add(key, value, tag, streams, f, fname_pat):
    e = G_.setdefault(key, dict(value=0.0, tags=set(), streams=set(), n=f["n"], n_spk=f["n_spk"], files=set(), pat=fname_pat, dataset=f["dataset"]))
    e["value"] = max(e["value"], value); e["tags"].add(tag); e["streams"].update(streams); e["files"].add(f["path"])
FAMPAT = {"single_mac": "{d}_groupkfold5_mac.csv", "single_pod": "{d}_groupkfold5_pod.csv", "seed_pod": "{d}_groupkfold5_pod_seed0..4[_UNVERIFIED].csv",
          "seed_mac": "{d}_groupkfold5_mac_seed0..4.csv", "seed_pod_pod3": "{d}_groupkfold5_pod_Control15ids_seed0..4.csv", "shuffle": "edaic_full_groupkfold5_shuffle_rs0..4.csv"}
for f in M:
    fam = f["family"]; pat = FAMPAT[fam].format(d=f["dataset"])
    if fam == "seed_pod": pat = pat.replace("[_UNVERIFIED]", "_UNVERIFIED" if f["status"] == "UNVERIFIED" else "")
    if fam in ("single_mac", "single_pod"):
        for e in f["evidence"]:
            if e["run"].startswith("q2a_probe2_podruns") and fam == "single_pod": tag = "NOT CONFIRMED (pod-extracted states not on disk)"
            elif e["maxabs"] < 1e-6: tag = "CONFIRMED exact"
            else: tag = "CONFIRMED cross-platform" + ("" if e["layers_all_match"] else ", one near-tie layer flip")
            add((f["dataset"], fam, e["run"], "perclip"), e["maxabs"], tag, [s_["stream"] for s_ in e["streams"]], f, pat)
        for c in f.get("curves") or []:
            add((f["dataset"], fam, c["run"], "curve"), c["cands"]["stable"]["maxabs_auc"], "AUC-ONLY curve, NOT CONFIRMED per clip", [c["stream"]], f, pat)
    elif fam in ("seed_pod", "seed_mac", "seed_pod_pod3"):
        for p_ in f.get("perclip") or []:
            tag = "CONFIRMED exact" if p_["maxabs_at_saved_layers"] < 1e-6 else "CONFIRMED cross-platform" + (", near-tie layer flip in some folds" if p_.get("layers_match") is False else "")
            add((f["dataset"], fam, p_["run"], "perclip"), p_["maxabs_at_saved_layers"], tag, [p_["stream"]], f, pat)
        for a_ in f.get("aucev") or []:
            add((f["dataset"], fam, a_["run"], "auc"), abs(a_["refit_auc_stable"] - a_["saved_auc4"]), "AUC-ONLY per-repeat AUC, NOT CONFIRMED per clip", [a_["stream"]], f, pat)
    elif fam == "shuffle":
        for k, v in f["edf"].items():
            add((f["dataset"], fam, "part16_EDAICFULL", "perclip"), v["maxabs_oof_mean"],
                "CONFIRMED cross-platform" + ("" if v["layers_match"] else ", near-tie layer flips in llm/ans"), [k], f, pat)
rows = []
for (d, fam, run, kind), e in G_.items():
    tags = sorted(e["tags"]); tag = max(tags, key=len) if all(t.startswith("CONFIRMED cross-platform") for t in tags) else "; ".join(tags)
    unit = {"perclip": "max abs diff per clip vs saved OOF", "curve": "max |dAUC| vs saved per-layer AUC curve", "auc": "max |dAUC| vs saved per-repeat AUC"}[kind]
    rid = f"T2_{d}_{fam}_{run}" + ("" if kind == "perclip" else f"_{kind}")
    files = sorted(e["files"]); ffield = files[0] if len(files) == 1 else os.path.dirname(files[0]) + "/" + e["pat"]
    rows.append([rid, f"{d} {e['pat']} vs {run} ({'/'.join(sorted(e['streams']))}), {unit}: {tag}", f"{e['value']:.3e}", "", "", str(e["n"]), str(e["n_spk"]), ffield])
with open(ROWS, "w") as fh:
    fh.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n")
    for r in rows: fh.write("\t".join(r) + "\n")
print(len(rows), "rows")

# ------------------------------------------------------------------ sidecar
import shutil, glob
os.makedirs(SCRIPTS_DIR, exist_ok=True)
keep = ["gkf.py", "eng.py", "verify_single.py", "verify_egemaps.py", "verify_repeats_perclip.py", "verify_auc_only.py", "verify_edaicfull.py",
        "noise.py", "conv.py", "labelsets.py", "pyes_check.py", "build_folds.py", "build_docs.py", "verify_written.py", "verify_written_edaicfull.py", "verify_written.log", "conv.log",
        "single_results.jsonl", "egemaps_results.jsonl", "repeats_perclip_results.jsonl", "auc_only_results.jsonl", "edaicfull_results.json",
        "noise_results.jsonl", "conv_results.jsonl", "files_manifest.json", "written_check.jsonl",
        "verify_single.log", "verify_repeats_perclip.log", "verify_auc_only.log", "verify_edaicfull.log", "noise.log"]
copied = []
for k in keep:
    p = f"{HERE}/{k}"
    if os.path.exists(p): shutil.copy2(p, f"{SCRIPTS_DIR}/{k}"); copied.append(f"{SCRIPTS_DIR}/{k}")
SRC = {}
for f in M:
    for e in f.get("evidence") or []:
        for k in ("states", "json", "script"):
            p = e.get(k)
            if p and p.startswith("/") and os.path.exists(p): SRC[p] = None
        for p in e.get("oof_files", []): SRC[p] = None
    for p in (f.get("perclip") or []):
        SRC[p["states"]] = None; SRC[p["oof_file"]] = None
    for a in (f.get("aucev") or []): SRC[a["file"]] = None
    for c in (f.get("curves") or []): SRC[c["file"]] = None; SRC[c["states"]] = None
    if f.get("external"): SRC[f["external"]["file"]] = None
for a in AUCO:
    if a["dataset"] != "edaic_conflict390": SRC[a["states"]] = None; SRC[a["file"]] = None
EDF = json.load(open(f"{HERE}/edaicfull_results.json"))
for k in ("states_tgz", "perclip", "json", "script"): SRC[EDF[k]] = None
SRC = {p: sha1mb(p) for p in sorted(SRC) if p and os.path.exists(p)}
WC = [json.loads(l) for l in open(f"{HERE}/written_check.jsonl")] if os.path.exists(f"{HERE}/written_check.jsonl") else []
side = dict(
    result="T2_folds", part="PART17 task 2", date=datetime.datetime.now().isoformat(timespec="seconds"),
    command=[f"cd {SCRIPTS_DIR} && /usr/local/bin/python3 verify_single.py", "/usr/local/bin/python3 verify_egemaps.py",
             "/usr/local/bin/python3 verify_repeats_perclip.py", "/usr/local/bin/python3 verify_edaicfull.py (needs edaicfull_shards.tgz extracted to ./edaicfull/)",
             "/usr/local/bin/python3 noise.py", "/usr/local/bin/python3 conv.py", "/usr/local/bin/python3 verify_auc_only.py all",
             "/usr/local/bin/python3 build_folds.py", "/usr/local/bin/python3 verify_written.py", "/usr/local/bin/python3 build_docs.py"],
    env=dict(python=platform.python_version(), machine=platform.machine(), numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, pandas=pd.__version__),
    tie_rules=dict(mac="sklearn GroupKFold(5) as installed here: np.argsort(counts)[::-1] with numpy's default kind",
                   stable="same algorithm with np.argsort(counts, kind='stable')[::-1]; reproduces the pod runs",
                   seeds="speaker ids renumbered by numpy.random.default_rng(seed).permutation(np.unique(ids)), seed 0..4, then GroupKFold(5)",
                   shuffle="GroupKFold(5, shuffle=True, random_state=r), r 0..4 (E-DAIC full window only)"),
    seed="no bootstrap in this task; fold seeds as in tie_rules",
    sources=SRC,
    outputs=sorted(f"{OUTD}/{f['file']}" for f in M) + [f"{OUTD}/README.md", ROWS, SIDE] + copied,
    files=[dict(file=f"{OUTD}/{f['file']}", dataset=f["dataset"], family=f["family"], rule=f["rule"], seed=f["seed"], status=f["status"], n=f["n"],
                n_speakers=f["n_spk"], fold_sizes=f["fold_sizes"], pitt_172=f.get("p172")) for f in M],
    written_file_check=WC,
    note="clip_id and speaker_id are exactly as each run used them; Pitt participant 172 is two labels (Control172, Dementia172) as used.")
json.dump(side, open(SIDE, "w"), indent=1)
print("sidecar", SIDE, len(SRC), "sources")
