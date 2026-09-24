"""PART26 Qwen3-Omni-30B-A3B Pitt: independent check of the pod rerun and of p26_gap.py, written separately (no shared code).
 1 the per-clip csv matches the states (clip order, labels, speakers) and the zero-shot file (p_yes, labels, speaker bijection);
 2 the fold columns equal the release fold files pitt_groupkfold5_pod_Control15ids_seed0..4 read again by clip, and the
   outer folds are speaker-disjoint;
 3 the per-repeat AUCs and the answer AUC are recomputed by brute-force pairwise counting (ties count one half);
 4 the bootstrap is rebuilt from the spec with its own loop (sorted speaker list, default_rng(0), choice(n, n, replace=True),
   pairwise-count AUC) and compared draw by draw with the saved npz, and the percentiles with the gap json;
 5 the pod environment: pinned versions, avx512f, the pod-side podverify PASS, pulled files equal the pod sha256 list;
 6 reference: the gap from the saved POD3 OOF reproduces the PART25 C number; the Mac rerun of the same script on Mac
   libraries (local scratch folder) is compared for information only.
Writes verify/p26_verify.json.   usage: /usr/local/bin/python3 p26_verify.py [MAC_RERUN_OOF_CSV]"""
import os, sys, csv, json, hashlib, numpy as np

R = "scores/part26/Qwen3-Omni-30B-A3B_Pitt"
CELL = "Qwen3-Omni-30B-A3B_Pitt"
POD = f"{R}/pull/p26-q3o-pitt"
res = {"checks": {}}
def check(name, ok, **kw):
    res["checks"][name] = {"ok": bool(ok), **kw}; print(("PASS " if ok else "FAIL ") + name, kw if kw else "", flush=True)

def auc_pairs(yv, s):
    yv = np.asarray(yv); s = np.asarray(s, float)
    a = s[yv == 1][:, None]; b = s[yv == 0][None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.shape[0] * b.shape[1]))

def sha(p):
    h = hashlib.sha256(); f = open(p, "rb")
    while True:
        b = f.read(1 << 20)
        if not b: break
        h.update(b)
    return h.hexdigest()

G = json.load(open(f"{R}/{CELL}_gap.json"))
rows = list(csv.DictReader(open(f"{R}/per_clip/{CELL}_perclip.csv")))
names = [r["clip"] for r in rows]; lab = np.array([int(r["label"]) for r in rows]); sp = [r["speaker"] for r in rows]
S = np.load("<local data dir>/release/edaic_rerun/part16/pod_sync/q3o_pitt_states.npz", allow_pickle=True)
check("1a per-clip rows equal the states (order, labels, speakers)", names == [str(x) for x in S["name"]] and
      list(lab) == [int(x) for x in S["label"]] and sp == [str(x) for x in S["spk"]], n=len(rows))
zrows = {r["clip"]: r for r in csv.DictReader(open("<local data dir>/release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv"))}
zp = np.array([float(zrows[c]["p_yes"]) for c in names])
check("1b p_yes_zeroshot column equals the q3o_new zero-shot file", all(r["p_yes_zeroshot"] == zrows[r["clip"]]["p_yes"] for r in rows)
      and all(int(zrows[c]["label"]) == l for c, l in zip(names, lab)) and len(zrows) == len(rows))
m1, m2 = {}, {}
for a, b in zip(sp, [zrows[c]["speaker"] for c in names]): m1.setdefault(a, set()).add(b); m2.setdefault(b, set()).add(a)
check("1c speaker ids one to one with the zero-shot file", all(len(v) == 1 for v in m1.values()) and all(len(v) == 1 for v in m2.values()),
      n_spk=len(m1))
# 2 folds
fold_ok = True; disjoint = True
for s in range(5):
    fm = {}
    with open(f"<local data dir>/release/folds/pitt_groupkfold5_pod_Control15ids_seed{s}.csv") as fh:
        for r in csv.DictReader(fh): fm[r["clip_id"]] = (r["fold"], r["speaker_id"])
    fold_ok &= all(r[f"fold_seed{s}"] == fm[r["clip"]][0] and r["speaker"] == fm[r["clip"]][1] for r in rows) and len(fm) == len(rows)
    per_spk = {}
    for r in rows: per_spk.setdefault(r["speaker"], set()).add(r[f"fold_seed{s}"])
    disjoint &= all(len(v) == 1 for v in per_spk.values())
check("2 fold columns equal the release Control15-id fold files; outer folds speaker-disjoint", fold_ok and disjoint)
# 3 AUCs
P = np.array([[float(r[f"p_probe_seed{s}"]) for r in rows] for s in range(5)])
per = [auc_pairs(lab, P[s]) for s in range(5)]; m5 = float(np.mean(per)); za = auc_pairs(lab, zp)
PJ = json.load(open(f"{POD}/out/p26_q3o_pitt_enc_nested5.json"))
check("3a per-repeat AUCs by pairwise counting equal the pod json and the gap json",
      max(abs(a - b) for a, b in zip(per, PJ["per_repeat_auc"])) < 1e-12 and max(abs(a - b) for a, b in zip(per, G["probe"]["per_repeat_auc"])) < 1e-12,
      per_repeat=[round(a, 6) for a in per], mean5=round(m5, 6))
check("3b answer AUC equals master_lookup 0.7619", round(za, 4) == 0.7619 and abs(za - G["answer"]["auc"]) < 1e-12, answer=round(za, 6))
check("3c gap point", abs((m5 - za) - G["gap"]["point"]) < 1e-12, gap=round(m5 - za, 6))
# 4 bootstrap from the spec
U = sorted(set(sp)); where = {u: [i for i, x in enumerate(sp) if x == u] for u in U}
rng = np.random.default_rng(0); D = np.full(2000, np.nan)
for b in range(2000):
    pick = rng.choice(len(U), size=len(U), replace=True)
    ii = np.array([i for k in pick for i in where[U[k]]]); yy = lab[ii]
    if yy.min() == yy.max(): continue
    D[b] = np.mean([auc_pairs(yy, P[s][ii]) for s in range(5)]) - auc_pairs(yy, zp[ii])
Z = np.load(f"{R}/draws/{CELL}_draws.npz", allow_pickle=True)
ok = ~np.isnan(D); lo, hi = np.percentile(D[ok], 2.5), np.percentile(D[ok], 97.5)
dd = float(np.nanmax(np.abs(D - Z["diff"])))
check("4 bootstrap rebuilt from the spec equals the saved draws and interval",
      dd < 1e-9 and round(lo, 4) == round(G["gap"]["lo"], 4) and round(hi, 4) == round(G["gap"]["hi"], 4) and int(ok.sum()) == G["gap"]["usable_draws"],
      max_abs_draw_diff=dd, lo=round(float(lo), 6), hi=round(float(hi), 6), usable=int(ok.sum()))
# 5 pod environment and pull integrity
V = json.load(open(f"{POD}/out/p26_q3o_pitt_podverify.json"))
drv = open(f"{POD}/logs/DRIVER.log").read()
ver = PJ["versions"]
check("5a pod versions pinned (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1), avx512f present, one BLAS thread",
      (ver["sklearn"], ver["numpy"], ver["scipy"]) == ("1.9.1", "2.1.2", "1.18.1") and "avx512f present" in drv and "OPENBLAS_NUM_THREADS=1" in open(f"{POD}/driver26.sh").read(),
      versions=ver)
check("5b pod-side podverify PASS (refit at saved layers, inner layer choice re-derived for every outer fold, rank AUC)",
      V.get("PASS") is True and not V["layer_mismatches"] and V["max_abs_pred_diff"] < 1e-6,
      max_abs_pred_diff=V["max_abs_pred_diff"], max_abs_inner_score_diff=V["max_abs_inner_score_diff"])
bad = []
for line in open(f"{POD}/logs/out_sha256.txt"):
    h, p = line.split()
    if sha(f"{POD}/out/{p}") != h: bad.append(p)
check("5c pulled out/ files equal the pod sha256 list", not bad, bad=bad)
check("5d pod read the staged states (sha256) and the five Control15-id fold files",
      PJ["states"]["sha256"] == sha(f"{R}/stage/q3o_pitt_encstates.npz") and
      all(PJ["tags"][f"seed{s}"]["fold_file"] == f"pitt_groupkfold5_pod_Control15ids_seed{s}.csv" and
          PJ["tags"][f"seed{s}"]["fold_file_sha256"] == sha(f"<local data dir>/release/folds/pitt_groupkfold5_pod_Control15ids_seed{s}.csv") for s in range(5)))
# 6 references (information)
P3 = np.array([[float(r[f"p_probe_pod3_rep{s}"]) for r in rows] for s in range(5)])
per3 = [auc_pairs(lab, P3[s]) for s in range(5)]
res["reference"] = {"table1_pod3_per_repeat": per3, "table1_pod3_mean5": float(np.mean(per3)),
                    "rerun_minus_table1": m5 - float(np.mean(per3)), "max_abs_clip_diff_vs_pod3": float(np.max(np.abs(P - P3))),
                    "pod3_gap_reproduces_part25C": G["reference_pod3_oof_gap"]["reproduces_part25C_4dp"]}
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    M = {r["clip"]: r for r in csv.DictReader(open(sys.argv[1]))}
    PM = np.array([[float(M[c][f"p_seed{s}"]) for c in names] for s in range(5)])
    res["reference"]["mac_rerun"] = {"file": sys.argv[1], "per_repeat": [auc_pairs(lab, PM[s]) for s in range(5)],
                                     "max_abs_clip_diff_vs_pod": float(np.max(np.abs(PM - P)))}
print("REFERENCE", json.dumps(res["reference"]), flush=True)
res["ALL_PASS"] = all(v["ok"] for v in res["checks"].values())
res["result"] = {"probe_mean5": m5, "per_repeat": per, "answer": za, "gap": m5 - za, "lo": float(lo), "hi": float(hi)}
os.makedirs(f"{R}/verify", exist_ok=True)
json.dump(res, open(f"{R}/verify/p26_verify.json", "w"), indent=1)
print("VERIFY", "ALL PASS" if res["ALL_PASS"] else "FAILED", json.dumps(res["result"]), flush=True)
