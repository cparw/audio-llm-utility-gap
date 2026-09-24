#!/usr/bin/env python3
"""Age and sex matched evaluation of the existing Pitt out-of-fold probe and the model's answer.
Matching rule as in age_match_pitt.py (same sex, nearest age within 3 years, dementia arm restricted
to Probable/Possible AD), applied to the 228 speakers behind the 468 scored segments. Only the
evaluation is restricted: folds and layer choice were fit on all speakers."""
import csv, os, numpy as np
P = os.path.expanduser("~/Desktop/Hard drive data for paper/DementiaBank")
OUT = os.path.expanduser("~/Desktop/Hard drive data for paper/speech-health-repo/results/health/dementiabank/age_matched_eval.csv")
def auc(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(1, len(p)+1)
    n1, n0 = y.sum(), (1-y).sum(); return (rk[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
sc = list(csv.DictReader(open(f"{P}/ad_scores_qwen2audio.csv")))
mf = [r for r in csv.DictReader(open(f"{P}/pitt_conflict_manifest.csv")) if r.get("segment_path")]
assert all(a["spk"] == b["spk"] and a["grp"] == b["grp"] for a, b in zip(sc, mf))
oof = np.load(f"{P}/ad_probe_oof.npy"); assert len(oof) == len(sc) == 468
y = np.array([int(r["label"]) for r in sc]); pm = np.array([float(r["p_main"]) for r in sc])
age = np.array([float(r["age"]) for r in sc]); st = np.array([r["set"] for r in mf])
spk = {}
for r in sc: spk.setdefault((r["grp"], r["spk"]), r)
ctrl = [r for r in spk.values() if int(r["label"]) == 0]
ad = [r for r in spk.values() if int(r["label"]) == 1 and r["dx"] in ("ProbableAD", "PossibleAD")]
pairs, used = [], set()
for sex in ("male", "female"):
    A = sorted([r for r in ad if r["sex"] == sex], key=lambda r: float(r["age"]))
    C = sorted([r for r in ctrl if r["sex"] == sex], key=lambda r: float(r["age"]))
    for a in A:
        best, bd = None, 99
        for c in C:
            if c["spk"] in used: continue
            d = abs(float(a["age"]) - float(c["age"]))
            if d < bd: bd, best = d, c
        if best is not None and bd <= 3: pairs.append((a, best)); used.add(best["spk"])
keep = {(a["grp"], a["spk"]) for a, _ in pairs} | {(c["grp"], c["spk"]) for _, c in pairs}
m = np.array([(r["grp"], r["spk"]) in keep for r in sc])
rows = [("pairs", len(pairs)), ("segments", int(m.sum())),
        ("mean age AD", round(np.mean([float(a["age"]) for a, _ in pairs]), 1)),
        ("mean age control", round(np.mean([float(c["age"]) for _, c in pairs]), 1)),
        ("all 468: age alone AUC", round(auc(y, age), 3)),
        ("all 468: speaker mean age control", round(np.mean([float(r["age"]) for r in ctrl]), 1)),
        ("all 468: speaker mean age dementia (all dx)", round(np.mean([float(r["age"]) for r in spk.values() if int(r["label"]) == 1]), 1)),
        ("matched: probe AUC", round(auc(y[m], oof[m]), 3)), ("matched: answer AUC", round(auc(y[m], pm[m]), 3)),
        ("matched: age alone AUC", round(auc(y[m], age[m]), 3))]
for s in ("agreement", "conflict"):
    mm = m & (st == s)
    rows += [(f"matched {s} n", int(mm.sum())), (f"matched {s}: probe AUC", round(auc(y[mm], oof[mm]), 3)), (f"matched {s}: answer AUC", round(auc(y[mm], pm[mm]), 3))]
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["quantity", "value"]); w.writerows(rows)
for r in rows: print(f"{r[0]}: {r[1]}")
