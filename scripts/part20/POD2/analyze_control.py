"""PART20 POD2 same-pod uniform-weight LoRA control: merge fold checkpoints -> uniform_lora_pitt_control_oof.csv;
AUC (scipy rankdata) + speaker bootstrap (2000, fresh default_rng(0) per cell); paired balanced-minus-control and
control-minus-earlier (one speaker draw per replicate); gap change balanced vs control.
usage: analyze_control.py"""
import sys, os, json, csv, hashlib, datetime, numpy as np
from decimal import Decimal, ROUND_HALF_UP
from scipy.stats import rankdata
R = "<local data dir>/release"; D = f"{R}/scores/part20/POD2"; C = f"{D}/control"
MAN = f"{R}/edaic_rerun/part16/POD2/mf_pitt468_pod.csv"; FOLDS = f"{R}/folds/pitt_groupkfold5_pod.csv"
EARLIER = f"{R}/edaic_rerun/part16/POD2/lora_pitt_oof.csv"; BAL = f"{D}/balanced_lora_pitt_oof.csv"
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
def sha1mb(p): return hashlib.sha256(open(p, "rb").read(1 << 20)).hexdigest()
def auc(y, s):
    y = np.asarray(y); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(s); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def draws(spk):
    rng = np.random.default_rng(0); u = np.unique(spk); idx = {g: np.where(spk == g)[0] for g in u}
    for _ in range(2000): yield np.concatenate([idx[g] for g in rng.choice(u, len(u), replace=True)])
def ci(v): v = [x for x in v if not np.isnan(x)]; return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)
rows = list(csv.DictReader(open(MAN))); fr = list(csv.DictReader(open(FOLDS)))
for a, b in zip(rows, fr): assert os.path.basename(a["path"]) == b["clip_id"] and a["speaker"] == b["speaker_id"]
N = 468; p2 = np.full(N, np.nan); pm = np.full(N, np.nan); ms = np.full(N, np.nan); fj = {}
fold = np.array([int(r["fold"]) for r in fr])
for k in range(5):
    j = json.load(open(f"{C}/uniform_lora_pitt_fold{k}.json")); fj[k] = j
    assert sorted(r["row"] for r in j["results"]) == np.where(fold == k)[0].tolist()
    for r in j["results"]: p2[r["row"]] = r["p_yes_2logit"]; pm[r["row"]] = r["p_yes"]; ms[r["row"]] = r["answer_mass"]
assert not np.isnan(p2).any()
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows]); arm = np.array([r["set"] for r in rows])
name = [os.path.basename(r["path"]) for r in rows]
OUT = f"{C}/uniform_lora_pitt_control_oof.csv"
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["name", "spk", "label", "arm", "fold", "p_yes", "p_yes_2logit", "answer_mass", "prompt"])
    for i in range(N): w.writerow([name[i], spk[i], y[i], arm[i], fold[i], repr(float(pm[i])), repr(float(p2[i])), repr(float(ms[i])), P])
b = list(csv.DictReader(open(BAL))); assert [r["name"] for r in b] == name
b2 = np.array([float(r["p_yes_2logit"]) for r in b]); bm = np.array([float(r["p_yes"]) for r in b])
e = list(csv.DictReader(open(EARLIER))); assert [r["name"] for r in e] == name
e2 = np.array([float(r["p_yes"]) for r in e]); em = np.array([float(r["p_yes_multivariant"]) for r in e])
cells = {"overall": np.ones(N, bool), "conflict": arm == "conflict", "agreement": arm == "agreement"}
res = {}
def cell_auc(s, ii, m): jj = ii[m[ii]]; return auc(y[jj], s[jj])
for sc, cs, bs, es in (("2logit", p2, b2, e2), ("multivariant", pm, bm, em)):
    for c, m in cells.items():
        mi = np.where(m)[0]; sp = spk[m]
        v = auc(y[m], cs[m]); lo, hi, k = ci([auc(y[mi][ii], cs[mi][ii]) for ii in draws(sp)])
        res[f"control.{c}.{sc}"] = dict(v=v, lo=lo, hi=hi, n=int(m.sum()), n_spk=int(len(set(sp))), draws=k, file=OUT)
        d = auc(y[m], bs[m]) - v; lo, hi, k = ci([auc(y[mi][ii], bs[mi][ii]) - auc(y[mi][ii], cs[mi][ii]) for ii in draws(sp)])
        res[f"bal_minus_control.{c}.{sc}"] = dict(v=d, lo=lo, hi=hi, n=int(m.sum()), n_spk=int(len(set(sp))), draws=k, file=f"{C}/balanced_vs_control_lora_pitt_perclip.csv", diff=1)
        d = v - auc(y[m], es[m]); lo, hi, k = ci([auc(y[mi][ii], cs[mi][ii]) - auc(y[mi][ii], es[mi][ii]) for ii in draws(sp)])
        res[f"control_minus_earlier.{c}.{sc}"] = dict(v=d, lo=lo, hi=hi, n=int(m.sum()), n_spk=int(len(set(sp))), draws=k, file=f"{C}/balanced_vs_control_lora_pitt_perclip.csv", diff=1)
    gap = lambda s, ii: cell_auc(s, ii, arm == "conflict") - cell_auc(s, ii, arm == "agreement")
    allidx = np.arange(N)
    v = gap(cs, allidx); lo, hi, k = ci([gap(cs, ii) for ii in draws(spk)])
    res[f"control.cma.{sc}"] = dict(v=v, lo=lo, hi=hi, n=N, n_spk=228, draws=k, file=OUT, diff=1)
    v = gap(bs, allidx) - gap(cs, allidx); lo, hi, k = ci([gap(bs, ii) - gap(cs, ii) for ii in draws(spk)])
    res[f"gapchange_bal_minus_control.{sc}"] = dict(v=v, lo=lo, hi=hi, n=N, n_spk=228, draws=k, file=f"{C}/balanced_vs_control_lora_pitt_perclip.csv", diff=1)
# per-fold diagnostics
diag = {}
for nm, s in (("balanced", b2), ("control", p2), ("earlier", e2)):
    diag[nm] = {"per_fold_auc_2logit": [round(auc(y[fold == k], s[fold == k]), 4) for k in range(5)],
                "per_fold_mean_p_yes_2logit": [round(float(s[fold == k].mean()), 4) for k in range(5)]}
    diag[nm]["mean_within_fold_auc"] = round(float(np.mean(diag[nm]["per_fold_auc_2logit"])), 4)
with open(f"{C}/balanced_vs_control_lora_pitt_perclip.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["name", "spk", "label", "arm", "fold", "balanced_p_yes", "balanced_p_yes_2logit", "control_p_yes", "control_p_yes_2logit", "earlier_p_yes_multivariant", "earlier_p_yes_2logit"])
    for i in range(N): w.writerow([name[i], spk[i], y[i], arm[i], fold[i], repr(float(bm[i])), repr(float(b2[i])), repr(float(pm[i])), repr(float(p2[i])), repr(float(em[i])), repr(float(e2[i]))])
side = {"result": "PART20 POD2 same-pod uniform-weight LoRA control (w=1) for the balanced LoRA, Pitt 468, Qwen2.5-Omni, out of fold",
        "why": "separates the effect of the balancing weights from run-to-run variation of the LoRA recipe (the earlier LoRA ran on another pod with unseeded init and dropout)",
        "model_id": "Qwen/Qwen2.5-Omni-7B", "prompt_verbatim": P, "n": N, "n_speakers": 228,
        "identical_to_balanced_except": "loss weight w=1 for every training clip (script uniform_lora_pitt_control.py = balanced_lora_pitt.py plus one line)",
        "seed": {"bootstrap": "default_rng(0) fresh per cell, 2000 draws, speakers resampled; paired = one speaker draw per replicate", "lora_init_torch_seed": 0, "fold_torch_seed": "1000+k"},
        "per_clip_csv": OUT, "paired_per_clip_csv": f"{C}/balanced_vs_control_lora_pitt_perclip.csv", "results": res, "per_fold_diagnostics": diag,
        "epoch_loss": {k: fj[k]["epoch_loss"] for k in range(5)}, "minutes_per_fold": {k: fj[k]["minutes"] for k in range(5)},
        "answer_mass": {"median": float(np.median(ms)), "min": float(ms.min())},
        "sources_sha256_first_1MB": {p: sha1mb(p) for p in [MAN, FOLDS, EARLIER, BAL, f"{D}/balanced_lora_pitt.py"] + [f"{C}/uniform_lora_pitt_fold{k}.json" for k in range(5)]},
        "command_train": {k: fj[k]["command"] for k in range(5)}, "command_analysis": "/usr/local/bin/python3 " + os.path.abspath(__file__),
        "date_utc": datetime.datetime.utcnow().isoformat()}
json.dump(side, open(f"{C}/uniform_lora_pitt_control_oof.sidecar.json", "w"), indent=1)
json.dump({k: v for k, v in side.items() if k != "per_fold_diagnostics"} | {"result": "paired differences balanced LoRA minus same-pod uniform control, and control minus earlier LoRA"},
          open(f"{C}/balanced_vs_control_lora_pitt.sidecar.json", "w"), indent=1)
def two(v): return str(Decimal(f"{v:.4f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
tsv = []
for key, d in res.items():
    if d.get("diff"): vs = "interval excludes zero" if (d["lo"] > 0 or d["hi"] < 0) else "interval includes zero"
    else: vs = "above 0.5" if d["lo"] > 0.5 else ("below 0.5" if d["hi"] < 0.5 else "interval includes 0.5")
    print(f"POD2.{key}: {d['v']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}] n={d['n']} n_spk={d['n_spk']} file={d['file']} | {two(d['v'])} [{two(d['lo'])}, {two(d['hi'])}] {vs}")
    tsv.append([f"POD2.{key}", key, f"{d['v']:.4f}", f"{d['lo']:.4f}", f"{d['hi']:.4f}", d["n"], d["n_spk"], d["file"], f"{two(d['v'])} [{two(d['lo'])}, {two(d['hi'])}]", vs])
print("DIAG", json.dumps(diag))
json.dump(tsv, open(f"{C}/_tsv.json", "w"))
