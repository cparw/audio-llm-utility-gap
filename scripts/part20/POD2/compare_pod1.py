"""PART20 POD2: paired AUC difference, balanced LoRA (this pod) minus POD1 balanced projector fine-tune, Pitt 468,
like for like (two-logit vs two-logit; rule 3 vs rule 3). One speaker draw per replicate, 2000 draws, fresh default_rng(0) per cell."""
import os, csv, json, hashlib, datetime, numpy as np
from decimal import Decimal, ROUND_HALF_UP
from scipy.stats import rankdata
R = "scores/part20"
BAL = f"{R}/POD2/balanced_lora_pitt_oof.csv"; P1 = f"{R}/POD1/balanced_ft_pitt_oof.csv"
OUT = f"{R}/POD2/balanced_lora_minus_pod1_projector_pitt_perclip.csv"
def sha1mb(p): return hashlib.sha256(open(p, "rb").read(1 << 20)).hexdigest()
def auc(y, s):
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum()); r = rankdata(s); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def two(v): return str(Decimal(f"{v:.4f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
b = list(csv.DictReader(open(BAL))); p = {r["clip_id"]: r for r in csv.DictReader(open(P1))}
assert len(p) == 468 and all(r["name"] in p for r in b)
P = b[0]["prompt"]; assert all(r["prompt"] == P for r in p.values())
y = np.array([int(r["label"]) for r in b]); spk = np.array([r["spk"] for r in b]); arm = np.array([r["arm"] for r in b])
assert all(int(p[r["name"]]["label"]) == int(r["label"]) and p[r["name"]]["speaker"] == r["spk"] and p[r["name"]]["set"] == r["arm"] and int(p[r["name"]]["fold"]) == int(r["fold"]) for r in b)
S = {"2logit": (np.array([float(r["p_yes_2logit"]) for r in b]), np.array([float(p[r["name"]]["p_yes"]) for r in b])),
     "multivariant": (np.array([float(r["p_yes"]) for r in b]), np.array([float(p[r["name"]]["p_yes_multi"]) for r in b]))}
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["name", "spk", "label", "arm", "fold", "lora_p_yes", "lora_p_yes_2logit", "proj_p_yes_multi", "proj_p_yes_2logit"])
    for i, r in enumerate(b): w.writerow([r["name"], r["spk"], r["label"], r["arm"], r["fold"], r["p_yes"], r["p_yes_2logit"], p[r["name"]]["p_yes_multi"], p[r["name"]]["p_yes"]])
res = {}; lines = []; tsv = []
cells = {"overall": np.ones(468, bool), "conflict": arm == "conflict", "agreement": arm == "agreement"}
for sc, (sl, sp) in S.items():
    for c, m in cells.items():
        yy, a1, a2, ss = y[m], sl[m], sp[m], spk[m]
        u = np.unique(ss); idx = {g: np.where(ss == g)[0] for g in u}; rng = np.random.default_rng(0); d = []; dp = []
        for _ in range(2000):
            ii = np.concatenate([idx[g] for g in rng.choice(u, len(u), replace=True)])
            d.append(auc(yy[ii], a1[ii]) - auc(yy[ii], a2[ii]))
        v = auc(yy, a1) - auc(yy, a2); lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        rng = np.random.default_rng(0); pv = []
        for _ in range(2000):
            ii = np.concatenate([idx[g] for g in rng.choice(u, len(u), replace=True)]); pv.append(auc(yy[ii], a2[ii]))
        res[f"{c}.{sc}"] = dict(lora=auc(yy, a1), proj=auc(yy, a2), proj_ci=[float(np.percentile(pv, 2.5)), float(np.percentile(pv, 97.5))], diff=v, lo=lo, hi=hi, n=int(m.sum()), n_spk=int(len(u)))
        vs = "interval excludes zero" if (lo > 0 or hi < 0) else "interval includes zero"
        lines.append(f"POD2.bal_lora_minus_pod1_proj.{c}.{sc} | balanced LoRA {auc(yy, a1):.4f} minus POD1 balanced projector {auc(yy, a2):.4f} ({c}, {sc}, paired): {v:.4f} [{lo:.4f}, {hi:.4f}] n={int(m.sum())} n_spk={len(u)} file={OUT} | {two(v)} [{two(lo)}, {two(hi)}] {vs}")
        tsv.append([f"POD2.bal_lora_minus_pod1_proj.{c}.{sc}", f"PAIRED balanced LoRA minus POD1 balanced projector AUC {c} ({sc})", f"{v:.4f}", f"{lo:.4f}", f"{hi:.4f}", int(m.sum()), len(u), OUT, f"{two(v)} [{two(lo)}, {two(hi)}]", vs])
json.dump({"result": "paired AUC difference, balanced LoRA (PART20 POD2) minus balanced projector fine-tune (PART20 POD1), Pitt 468, Qwen2.5-Omni, out of fold, same folds and same per-clip weighting",
           "per_clip_csv": OUT, "sources_absolute": [BAL, P1], "sha256_first_1MB": {BAL: sha1mb(BAL), P1: sha1mb(P1)}, "n": 468, "n_speakers": 228,
           "seed": "default_rng(0) fresh per cell, 2000 draws, one speaker draw per replicate scores both models", "model_id": "Qwen/Qwen2.5-Omni-7B", "prompt_verbatim": P,
           "like_for_like": "2logit: LoRA p_yes_2logit vs POD1 p_yes (both softmax over [' Yes',' No']); multivariant: LoRA p_yes vs POD1 p_yes_multi (both rule 3)",
           "note": "POD1 answer_mass min is 4.0e-05 (its file); the LoRA answer_mass min is 0.9970", "results": res,
           "command": "/usr/local/bin/python3 " + os.path.abspath(__file__), "date_utc": datetime.datetime.utcnow().isoformat()},
          open(OUT.replace("_perclip.csv", ".sidecar.json"), "w"), indent=1)
print("\n".join(lines)); json.dump(tsv, open(f"{R}/POD2/_tsv_pod1.json", "w"))
