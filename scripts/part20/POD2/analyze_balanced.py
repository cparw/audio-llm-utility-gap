"""PART20 POD2 analysis: merge fold checkpoints -> balanced_lora_pitt_oof.csv, AUC (scipy rankdata, ties averaged),
speaker bootstrap (2000, fresh default_rng(0) per cell, percentile 2.5/97.5), paired differences vs the earlier LoRA
(one speaker draw per replicate). Optional POD1 comparison if its per-clip file is given.
usage: analyze_balanced.py <dir with fold jsons> [pod1_perclip_csv]"""
import sys, os, json, csv, hashlib, datetime, numpy as np
from scipy.stats import rankdata
D = sys.argv[1]; POD1 = sys.argv[2] if len(sys.argv) > 2 else None
R = "<local data dir>/release"
MAN = f"{R}/edaic_rerun/part16/POD2/mf_pitt468_pod.csv"
FOLDS = f"{R}/folds/pitt_groupkfold5_pod.csv"
EARLIER = f"{R}/edaic_rerun/part16/POD2/lora_pitt_oof.csv"
PROJ = f"{R}/omni_final/omnisft_pitt_oof.csv"   # READ ONLY
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
MID = "Qwen/Qwen2.5-Omni-7B"
ROWS = f"{R}/scores/part20/rows/POD2_provisional.tsv"
def sha1mb(p): return hashlib.sha256(open(p, "rb").read(1 << 20)).hexdigest()
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(s); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def boot(y, s, spk, n=2000):
    rng = np.random.default_rng(0); u = np.unique(spk); idx = {g: np.where(spk == g)[0] for g in u}; out = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(u, len(u), replace=True)]); a = auc(y[ii], s[ii])
        if not np.isnan(a): out.append(a)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)
def boot_pair(y, sa, sb, spk, n=2000):
    rng = np.random.default_rng(0); u = np.unique(spk); idx = {g: np.where(spk == g)[0] for g in u}; out = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(u, len(u), replace=True)])
        a = auc(y[ii], sa[ii]); b = auc(y[ii], sb[ii])
        if not (np.isnan(a) or np.isnan(b)): out.append(a - b)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)
def boot_arms(y, s, spk, arm, n=2000):  # conflict minus agreement, one speaker draw per replicate
    rng = np.random.default_rng(0); u = np.unique(spk); idx = {g: np.where(spk == g)[0] for g in u}; out = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(u, len(u), replace=True)])
        ia = ii[arm[ii] == "conflict"]; ib = ii[arm[ii] == "agreement"]
        a = auc(y[ia], s[ia]); b = auc(y[ib], s[ib])
        if not (np.isnan(a) or np.isnan(b)): out.append(a - b)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)

def boot_gapchange(y, sa, sb, spk, arm, n=2000):  # (conf-agr)_a minus (conf-agr)_b, one speaker draw per replicate
    rng = np.random.default_rng(0); u = np.unique(spk); idx = {g: np.where(spk == g)[0] for g in u}; out = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(u, len(u), replace=True)])
        ia = ii[arm[ii] == "conflict"]; ib = ii[arm[ii] == "agreement"]
        v = (auc(y[ia], sa[ia]) - auc(y[ib], sa[ib])) - (auc(y[ia], sb[ia]) - auc(y[ib], sb[ib]))
        if not np.isnan(v): out.append(v)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)
rows = list(csv.DictReader(open(MAN))); fr = list(csv.DictReader(open(FOLDS)))
assert len(rows) == len(fr) == 468
for a, b in zip(rows, fr): assert os.path.basename(a["path"]) == b["clip_id"] and a["speaker"] == b["speaker_id"]
proj = list(csv.DictReader(open(PROJ)))
assert set(r["prompt"] for r in proj) == {P}, "prompt differs from the paper run's per-clip prompt column"
assert [os.path.basename(r["path"]) for r in proj] == [os.path.basename(r["path"]) for r in rows]
N = len(rows); p2 = np.full(N, np.nan); pm = np.full(N, np.nan); ms = np.full(N, np.nan); fj = {}
for k in range(5):
    j = json.load(open(f"{D}/balanced_lora_pitt_fold{k}.json")); fj[k] = j
    for r in j["results"]: p2[r["row"]] = r["p_yes_2logit"]; pm[r["row"]] = r["p_yes"]; ms[r["row"]] = r["answer_mass"]
assert not np.isnan(p2).any() and not np.isnan(pm).any()
fold = np.array([int(r["fold"]) for r in fr])
for k in range(5): assert sorted(r["row"] for r in fj[k]["results"]) == np.where(fold == k)[0].tolist()
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows]); arm = np.array([r["set"] for r in rows])
name = [os.path.basename(r["path"]) for r in rows]
OUT = f"{D}/balanced_lora_pitt_oof.csv"
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["name", "spk", "label", "arm", "fold", "p_yes", "p_yes_2logit", "answer_mass", "prompt"])
    for i in range(N): w.writerow([name[i], spk[i], y[i], arm[i], fold[i], repr(float(pm[i])), repr(float(p2[i])), repr(float(ms[i])), P])
er = list(csv.DictReader(open(EARLIER))); assert [r["name"] for r in er] == name
e2 = np.array([float(r["p_yes"]) for r in er]); em = np.array([float(r["p_yes_multivariant"]) for r in er])
assert (np.array([int(r["label"]) for r in er]) == y).all() and (np.array([r["arm"] for r in er]) == arm).all()
cells = {"overall": np.ones(N, bool), "conflict": arm == "conflict", "agreement": arm == "agreement"}
res = {"balanced": {}, "earlier": {}, "diff_balanced_minus_earlier": {}, "conflict_minus_agreement": {}}
for sc, bs, es in (("2logit", p2, e2), ("multivariant", pm, em)):
    for c, m in cells.items():
        a = auc(y[m], bs[m]); lo, hi, k = boot(y[m], bs[m], spk[m])
        res["balanced"][f"{c}_{sc}"] = dict(v=a, lo=lo, hi=hi, n=int(m.sum()), n_spk=int(len(set(spk[m]))), draws=k)
        a2 = auc(y[m], es[m]); lo2, hi2, k2 = boot(y[m], es[m], spk[m])
        res["earlier"][f"{c}_{sc}"] = dict(v=a2, lo=lo2, hi=hi2, n=int(m.sum()), n_spk=int(len(set(spk[m]))), draws=k2)
        dl, dh, dk = boot_pair(y[m], bs[m], es[m], spk[m])
        res["diff_balanced_minus_earlier"][f"{c}_{sc}"] = dict(v=a - a2, lo=dl, hi=dh, n=int(m.sum()), n_spk=int(len(set(spk[m]))), draws=dk)
    cl, ch, ck = boot_arms(y, bs, spk, arm)
    ec = auc(y[arm == "conflict"], es[arm == "conflict"]) - auc(y[arm == "agreement"], es[arm == "agreement"])
    gl, gh, gk = boot_gapchange(y, bs, es, spk, arm)
    res.setdefault("gap_change_balanced_minus_earlier", {})[sc] = dict(v=(res["balanced"][f"conflict_{sc}"]["v"] - res["balanced"][f"agreement_{sc}"]["v"]) - ec, lo=gl, hi=gh, n=N, n_spk=int(len(set(spk))), draws=gk)
    res["conflict_minus_agreement"][sc] = dict(v=res["balanced"][f"conflict_{sc}"]["v"] - res["balanced"][f"agreement_{sc}"]["v"], lo=cl, hi=ch, n=N, n_spk=int(len(set(spk))), draws=ck)
if POD1 and os.path.exists(POD1):
    p1 = list(csv.DictReader(open(POD1))); key = {os.path.basename(r.get("name") or r.get("path") or r.get("clip_id")): r for r in p1}
    col = "p_yes"; p1s = np.array([float(key[n][col]) for n in name])
    res["pod1"] = {}; res["diff_balancedlora_minus_pod1"] = {}
    for c, m in cells.items():
        a = auc(y[m], p1s[m]); lo, hi, k = boot(y[m], p1s[m], spk[m])
        res["pod1"][c] = dict(v=a, lo=lo, hi=hi, n=int(m.sum()), n_spk=int(len(set(spk[m]))), draws=k, file=POD1, col=col)
        for sc, bs in (("2logit", p2), ("multivariant", pm)):
            dl, dh, dk = boot_pair(y[m], bs[m], p1s[m], spk[m])
            res["diff_balancedlora_minus_pod1"][f"{c}_{sc}"] = dict(v=auc(y[m], bs[m]) - a, lo=dl, hi=dh, n=int(m.sum()), n_spk=int(len(set(spk[m]))), draws=dk)
res["answer_mass"] = dict(median=float(np.median(ms)), min=float(ms.min()), max=float(ms.max()))
params = json.load(open(f"{D}/balanced_lora_pitt_params.json"))
srcs = [MAN, FOLDS, EARLIER, PROJ, "<local data dir>/release/edaic_rerun/part16/POD2/lora_pitt.py"]
side = {"result": "PART20 POD2 balanced LoRA, Pitt 468, Qwen2.5-Omni, out of fold", "model_id": MID, "prompt_verbatim": P,
        "prompt_check": f"equals every row of the per-clip prompt column of {PROJ}",
        "n": N, "n_speakers": int(len(set(spk))), "n_conflict": int((arm == 'conflict').sum()), "n_agreement": int((arm == 'agreement').sum()),
        "seed": {"bootstrap": "numpy.random.default_rng(0) fresh per cell, 2000 draws, speakers resampled", "lora_init_torch_seed": 0, "fold_torch_seed": "1000+k", "shuffle": "numpy RandomState(k*10+ep), as the earlier LoRA"},
        "per_clip_csv": OUT, "columns": {"p_yes": "rule 3: sum of single-token Yes/yes/YES (bare + leading space) over Yes+No variants at the first answer position",
        "p_yes_2logit": "softmax over [' Yes',' No'] only; this is the column the earlier LoRA 0.8250 and the projector 0.7673 were computed on", "answer_mass": "P(Yes variants)+P(No variants), full vocab softmax"},
        "training": {"lora": "r=8 alpha=16 dropout=0.05, q/k/v/o_proj of all 28 LM layers (112 modules); audio encoder and projector frozen", "loss": "per-clip weight x CE over [' Yes',' No'] logits at the answer position",
                     "optimizer": "AdamW lr 1e-4", "epochs": 3, "batch": 1, "dtype": "bfloat16 model, float32 LoRA params", "window_s": 30,
                     "weighting": "inside each training fold, per label y: w=(N_y/2)/N_{y,a}, rescaled to mean 1 over the training fold (rescale factor is exactly 1)",
                     "weight_tables": {k: fj[k]["weight_table"] for k in range(5)}, "epoch_loss": {k: fj[k]["epoch_loss"] for k in range(5)},
                     "minutes_per_fold": {k: fj[k]["minutes"] for k in range(5)}, "steps_per_fold": {k: fj[k]["steps"] for k in range(5)}, "peak_gpu_gb": {k: fj[k]["peak_gpu_gb"] for k in range(5)}},
        "parameter_counts": params, "results": res,
        "sources_sha256_first_1MB": {p: sha1mb(p) for p in srcs},
        "audio": {"mac_source": "<local data dir>/DementiaBank/segments", "pod_path": "/workspace/data/pitt468", "check": "full-file sha256 of all 468 pod wavs equals the Mac originals (382 sent as FLAC and rebuilt with the original header bytes, 86 sent as wav); 516863704 bytes"},
        "command_train": {k: fj[k]["command"] for k in range(5)}, "command_analysis": " ".join(["/usr/local/bin/python3"] + sys.argv),
        "pod": {"id": "7ckjx1jr7jm7yh", "gpu": "H100 NVL 94GB", "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", "torch": "2.8.0+cu128", "transformers": "5.17.0", "peft": "0.21.0"},
        "date_utc": datetime.datetime.utcnow().isoformat()}
json.dump(side, open(f"{D}/balanced_lora_pitt_oof.sidecar.json", "w"), indent=1)
# paired-difference per-clip file
DF = f"{D}/balanced_minus_earlier_lora_pitt_perclip.csv"
with open(DF, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["name", "spk", "label", "arm", "fold", "balanced_p_yes", "balanced_p_yes_2logit", "earlier_p_yes_multivariant", "earlier_p_yes_2logit"])
    for i in range(N): w.writerow([name[i], spk[i], y[i], arm[i], fold[i], repr(float(pm[i])), repr(float(p2[i])), repr(float(em[i])), repr(float(e2[i]))])
json.dump({"result": "paired AUC difference, balanced LoRA minus earlier LoRA (part16 POD2), Pitt 468", "per_clip_csv": DF, "n": N, "n_speakers": int(len(set(spk))),
           "seed": "default_rng(0) fresh per cell, one speaker draw per replicate scores both models", "model_id": MID, "prompt_verbatim": P,
           "sources": {OUT: sha1mb(OUT), EARLIER: sha1mb(EARLIER)}, "results": res["diff_balanced_minus_earlier"], "command": side["command_analysis"]},
          open(f"{D}/balanced_minus_earlier_lora_pitt.sidecar.json", "w"), indent=1)
from decimal import Decimal, ROUND_HALF_UP
def two(v): return str(Decimal(f"{v:.4f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))  # half-up on the 4 dp value, as the paper writes 0.8250 -> 0.83
lines = []; tsv = []
def emit(idn, what, d, f, diff=False):
    if diff: vs = "interval excludes zero" if (d["lo"] > 0 or d["hi"] < 0) else "interval includes zero"
    else: vs = "above 0.5" if d["lo"] > 0.5 else ("below 0.5" if d["hi"] < 0.5 else "interval includes 0.5")
    lines.append(f"{idn} | {what}: {d['v']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}] n={d['n']} n_spk={d['n_spk']} file={f} | {two(d['v'])} [{two(d['lo'])}, {two(d['hi'])}] {vs}")
    tsv.append([idn, what, f"{d['v']:.4f}", f"{d['lo']:.4f}", f"{d['hi']:.4f}", d["n"], d["n_spk"], f, f"{two(d['v'])} [{two(d['lo'])}, {two(d['hi'])}]", vs])
for sc in ("2logit", "multivariant"):
    for c in cells:
        emit(f"POD2.bal.{c}.{sc}", f"balanced LoRA {c} AUC ({sc})", res["balanced"][f"{c}_{sc}"], OUT)
        emit(f"POD2.earlier.{c}.{sc}", f"earlier LoRA {c} AUC ({sc}, recomputed)", res["earlier"][f"{c}_{sc}"], EARLIER)
        emit(f"POD2.diff.{c}.{sc}", f"balanced minus earlier LoRA {c} ({sc}, paired)", res["diff_balanced_minus_earlier"][f"{c}_{sc}"], DF, True)
    emit(f"POD2.bal.cma.{sc}", f"balanced LoRA conflict minus agreement ({sc}, paired)", res["conflict_minus_agreement"][sc], OUT, True)
    emit(f"POD2.gapchange.{sc}", f"gap change: (conflict-agreement) balanced minus (conflict-agreement) earlier LoRA ({sc}, paired)", res["gap_change_balanced_minus_earlier"][sc], DF, True)
if "pod1" in res:
    for c in cells:
        emit(f"POD2.pod1.{c}", f"POD1 balanced projector {c} AUC (p_yes)", res["pod1"][c], POD1)
        for sc in ("2logit", "multivariant"):
            emit(f"POD2.bal_minus_pod1.{c}.{sc}", f"balanced LoRA ({sc}) minus POD1 balanced projector {c} (paired)", res["diff_balancedlora_minus_pod1"][f"{c}_{sc}"], OUT, True)
print("\n".join(lines))
print("PARAMS lora_trainable_params", params["lora_trainable_params"], f"({params['lora_pct_of_full_model']}% of full model, {params['lora_pct_of_thinker']}% of thinker)")
print("ANSWER_MASS", res["answer_mass"])
json.dump({"lines": lines, "tsv": tsv}, open(f"{D}/_emit.json", "w"))
