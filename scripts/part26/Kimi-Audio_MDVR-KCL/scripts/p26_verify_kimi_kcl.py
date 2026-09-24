"""PART26 independent Mac check of the Kimi-Audio MDVR-KCL cell. Written separately from p26_gap_kimi_kcl.py (no shared code).
 1 pulled CPU pod outputs: sha256 equal the pod's own out_sha256.txt; the GPU pod states likewise;
 2 the states the CPU pod probed are byte equal to the GPU pod states (sha256 in the pod json and of the pulled in/ copy);
 3 OOF csv clip order, speakers, labels equal the states; fold columns equal the release fold files read again here by clip;
 4 per-repeat AUCs recomputed as exact fractions (Mann-Whitney, ties 1/2) from the OOF csv, against the pod json;
 5 zero-shot AUC recomputed exactly from the saved file; equals the master 0.7113 at 4 dp; clip set, labels, speakers equal;
 6 per-clip csv columns equal the pod OOF csv and the zero-shot file, string for string;
 7 bootstrap rebuilt with an own loop (rank AUC via scipy rankdata, sorted speaker list) from a fresh default_rng(0);
   draws and interval compared with the saved npz and sidecar;
 8 pod-side podverify result read (refit at saved layers, inner layer choice re-derived);
 9 Mac refit at the saved layers (StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced')): AUC and max
   prediction difference recorded (different BLAS, so not expected to be bit equal);
10 Mac re-derivation of the inner layer choice for every outer fold of every repeat with an own group fill.
Writes verify/p26_verify_kimi_kcl.json."""
import os, csv, json, hashlib, numpy as np
from fractions import Fraction
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

R = "scores/part26/Kimi-Audio_MDVR-KCL"
POD = f"{R}/pull/p26-kimi-kcl-cpu"; GPOD = f"{R}/pull/p26-kimi-kcl-gpu"
ZS = "<local data dir>/release/overnight2/kimi_new/kimi_kcl_zeroshot_scores.csv"
FD = "<local data dir>/release/folds"
res = {"checks": {}}
def ck(name, ok, **kw):
    res["checks"][name] = dict(ok=bool(ok), **kw); print(("PASS " if ok else "FAIL ") + name, kw if kw else "", flush=True)
def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
# 1
for tag, base in (("cpu", POD), ("gpu", GPOD)):
    bad = []; cnt = 0
    for line in open(f"{base}/out_sha256.txt"):
        h, f = line.split(); cnt += 1
        if sha(f"{base}/out/{f}") != h: bad.append(f)
    ck(f"pulled_{tag}_out_sha256_match_pod", cnt > 0 and not bad, n_files=cnt, mismatched=bad)
# 2
J = json.load(open(f"{POD}/out/p26_kimi_kcl_enc_nested5.json"))
gs = sha(f"{GPOD}/out/p26_kimi_kcl_states.npz")
ck("probed_states_equal_gpu_states", J["states"]["sha256"] == gs, pod_json_sha=J["states"]["sha256"], gpu_sha=gs)
s = np.load(f"{GPOD}/out/p26_kimi_kcl_states.npz", allow_pickle=True)
names = [str(x) for x in s["name"]]; spk = [str(x) for x in s["spk"]]; lab = [int(x) for x in s["label"]]
# 3
rows = list(csv.DictReader(open(f"{POD}/out/p26_kimi_kcl_enc_nested5_oof.csv")))
ck("oof_clip_order_equals_states", [r["clip"] for r in rows] == names and [r["speaker"] for r in rows] == spk and [int(r["label"]) for r in rows] == lab, n=len(rows))
fok = {}
for t, fn in [(f"seed{i}", f"kcl_groupkfold5_pod_seed{i}_UNVERIFIED.csv") for i in range(5)] + [("single", "kcl_groupkfold5_pod.csv")]:
    fm = {r["clip_id"]: (r["speaker_id"], r["fold"]) for r in csv.DictReader(open(f"{FD}/{fn}"))}
    fok[t] = all(fm[r["clip"]] == (r["speaker"], r[f"fold_{t}"]) for r in rows) and len(fm) == len(rows)
ck("oof_folds_equal_release_fold_files", all(fok.values()), per_tag=fok)
# 4
def exact_auc(y, p):
    y = np.asarray(y); p = np.asarray(p, float); pos = p[y == 1]; neg = p[y == 0]; c = Fraction(0)
    for v in pos: c += int((v > neg).sum()) + Fraction(int((v == neg).sum()), 2)
    return c / (len(pos) * len(neg))
ex = [exact_auc(lab, [float(r[f"p_seed{i}"]) for r in rows]) for i in range(5)]
exm = sum(ex) / 5
ck("per_repeat_auc_exact_equals_pod_json", all(abs(float(e) - j) < 1e-12 for e, j in zip(ex, J["per_repeat_auc"])) and abs(float(exm) - J["mean5"]) < 1e-12,
   exact=[str(e) for e in ex], mean5_exact=str(exm), mean5=float(exm), per_repeat=[float(e) for e in ex])
ck("single_split_auc_exact", abs(float(exact_auc(lab, [float(r["p_single"]) for r in rows])) - J["single_split_auc"]) < 1e-12)
# 5
zrows = list(csv.DictReader(open(ZS))); zmap = {r["clip"]: r for r in zrows}
zy = [int(zmap[c]["label"]) for c in names]; zsp = [zmap[c]["speaker"] for c in names]
zauc = exact_auc(lab, [float(zmap[c]["p_yes"]) for c in names])
ck("zeroshot_clips_labels_speakers_equal", set(zmap) == set(names) and zy == lab and zsp == spk and len(zrows) == len(names))
ck("zeroshot_auc_exact_equals_0.7113_4dp", round(float(zauc), 4) == 0.7113, exact=str(zauc), value=float(zauc))
# 6
S = json.load(open(f"{R}/per_clip/p26_kimi_kcl_perclip.sidecar.json"))
pc = list(csv.DictReader(open(f"{R}/per_clip/p26_kimi_kcl_perclip.csv")))
okpc = len(pc) == len(rows) and all(
    p["clip"] == r["clip"] and p["speaker"] == r["speaker"] and p["label"] == r["label"] and p["p_probe_single"] == r["p_single"] and p["fold_single"] == r["fold_single"]
    and all(p[f"p_probe_seed{i}"] == r[f"p_seed{i}"] and p[f"fold_seed{i}"] == r[f"fold_seed{i}"] for i in range(5))
    and float(p["p_yes_zeroshot"]) == float(zmap[p["clip"]]["p_yes"]) for p, r in zip(pc, rows))
ck("perclip_csv_equals_pod_oof_and_zeroshot", okpc)
# 7
def rank_auc(yy, sc):
    r = rankdata(sc); n1 = int((yy == 1).sum()); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
Y = np.array(lab); Pm = np.array([[float(r[f"p_seed{i}"]) for r in rows] for i in range(5)]); Z = np.array([float(zmap[c]["p_yes"]) for c in names])
SP = np.array(spk); U = sorted(set(spk)); byspk = {u: [i for i in range(len(SP)) if SP[i] == u] for u in U}
rng = np.random.default_rng(0); D = []
for b in range(2000):
    pick = rng.choice(len(U), size=len(U), replace=True)
    ii = np.array([i for k in pick for i in byspk[U[k]]]); yy = Y[ii]
    if yy.min() == yy.max(): D.append(np.nan); continue
    D.append(np.mean([rank_auc(yy, Pm[r][ii]) for r in range(5)]) - rank_auc(yy, Z[ii]))
D = np.array(D); ok = ~np.isnan(D)
lo, hi = np.percentile(D[ok], 2.5), np.percentile(D[ok], 97.5)
dr = np.load(f"{R}/draws/p26_kimi_kcl_draws.npz", allow_pickle=True)
same_nan = np.array_equal(np.isnan(dr["diff"]), ~ok)
mx = float(np.nanmax(np.abs(dr["diff"] - D)))
ck("bootstrap_rebuilt_equals_saved_draws", same_nan and mx < 1e-12 and int(ok.sum()) == S["gap"]["usable_draws"] and dr["diff"].shape == (2000,),
   max_abs_draw_diff=mx, usable=int(ok.sum()))
ck("bootstrap_interval_equals_sidecar", abs(lo - S["gap"]["ci_2p5_97p5"][0]) < 1e-12 and abs(hi - S["gap"]["ci_2p5_97p5"][1]) < 1e-12, lo=float(lo), hi=float(hi))
ck("gap_point_equals_sidecar", abs(float(exm - zauc) - S["gap"]["point"]) < 1e-12, gap=float(exm - zauc))
# 8
PV = json.load(open(f"{POD}/out/p26_kimi_kcl_podverify.json"))
ck("podverify_pass", PV.get("PASS") is True, max_abs_pred_diff=PV.get("max_abs_pred_diff"), layer_mismatches=PV.get("layer_mismatches"),
   abs_mean5_diff=PV.get("abs_mean5_diff"))
# 9
X = s["enc"].astype(np.float32); mac = {}
for t in [f"seed{i}" for i in range(5)] + ["single"]:
    fo = np.array([int(r[f"fold_{t}"]) for r in rows]); refit = np.zeros(len(rows)); saved = np.array([float(r[f"p_{t}"]) for r in rows])
    for k in range(5):
        tr, te = np.where(fo != k)[0], np.where(fo == k)[0]; L = J["tags"][t]["layers"][k]
        sc = StandardScaler().fit(X[tr][:, L]); m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr][:, L]), Y[tr])
        refit[te] = m.predict_proba(sc.transform(X[te][:, L]))[:, 1]
    mac[t] = {"auc_mac_refit": float(exact_auc(lab, refit)), "auc_pod": J["tags"][t]["auc"], "max_abs_pred_diff": float(np.max(np.abs(refit - saved)))}
ck("mac_refit_at_saved_layers_same_auc", all(abs(v["auc_mac_refit"] - v["auc_pod"]) < 1e-9 for v in mac.values()), per_tag=mac)
# 10
def fill(groups, k):
    u = sorted(set(groups)); cnt = {g: 0 for g in u}
    for g in groups: cnt[g] += 1
    order = [u[i] for i in np.argsort([cnt[g] for g in u], kind="stable")[::-1]]
    load = [0] * k; fg = {}
    for g in order:
        f = int(np.argmin(load)); fg[g] = f; load[f] += cnt[g]
    return np.array([fg[g] for g in groups])
lm = []; worst = 0.0
for t in [f"seed{i}" for i in range(5)] + ["single"]:
    if t == "single": g = np.array(spk)
    else:
        rr = np.random.default_rng(int(t[4:])); uu = np.unique(np.array(spk)); perm = {q: i for i, q in enumerate(rr.permutation(uu))}
        g = np.array([perm[q] for q in spk])
    fo = np.array([int(r[f"fold_{t}"]) for r in rows])
    for k in range(5):
        tr = np.where(fo != k)[0]; f4 = fill(list(g[tr]), 4); sc = []
        for l in range(X.shape[1]):
            a = 0.0
            for j in range(4):
                itr, ite = tr[f4 != j], tr[f4 == j]
                ss = StandardScaler().fit(X[itr, l]); m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(ss.transform(X[itr, l]), Y[itr])
                p = m.predict_proba(ss.transform(X[ite, l]))[:, 1]
                a += rank_auc(Y[ite], p) if len(set(Y[ite])) > 1 else 0.5
            sc.append(a / 4)
        worst = max(worst, float(np.max(np.abs(np.array(sc) - np.array(J["tags"][t]["inner"][k])))))
        b = int(np.argmax(sc))
        if b != J["tags"][t]["layers"][k]:
            lm.append({"tag": t, "fold": k, "saved": J["tags"][t]["layers"][k], "mac": b,
                       "saved_inner_at_saved": J["tags"][t]["inner"][k][J["tags"][t]["layers"][k]], "saved_inner_at_mac": J["tags"][t]["inner"][k][b]})
        mac_best = b; saved_L = J["tags"][t]["layers"][k]
        if abs(sc[saved_L] - max(sc)) < 1e-9 and b != saved_L:
            lm[-1]["mac_scores_tie"] = True
ck("mac_inner_layer_choice_equals_pod_up_to_exact_ties", all(m.get("mac_scores_tie") or m["saved_inner_at_saved"] == m["saved_inner_at_mac"] for m in lm),
   mismatches=lm, max_abs_inner_score_diff=worst)
# 11 the pod podverify flagged one mismatch; it must be an exact tie in the saved inner scores (np.argmax takes the first index)
tie = []
for m in PV.get("layer_mismatches", []):
    inn = J["tags"][m["tag"]]["inner"][m["fold"]]
    tie.append({**m, "inner_saved": inn[m["saved"]], "inner_rederived": inn[m["rederived"]], "inner_max": max(inn),
                "exact_tie": inn[m["saved"]] == inn[m["rederived"]] == max(inn) and m["saved"] < m["rederived"]})
ck("podverify_mismatches_are_exact_ties_first_index_kept", all(x["exact_tie"] for x in tie), ties=tie)
# 12 sensitivity: refit the tied folds at the other tied layer; the probe and gap if the tie had gone the other way
sens = []
for x in tie:
    t, k = x["tag"], x["fold"]
    fo = np.array([int(r[f"fold_{t}"]) for r in rows]); tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
    alt = np.array([float(r[f"p_{t}"]) for r in rows])
    ss = StandardScaler().fit(X[tr][:, x["rederived"]]); m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(ss.transform(X[tr][:, x["rederived"]]), Y[tr])
    alt[te] = m.predict_proba(ss.transform(X[te][:, x["rederived"]]))[:, 1]
    a_alt = float(exact_auc(lab, alt)); i = int(t[4:]) if t != "single" else None
    m5_alt = float((sum(ex) - (ex[i] if i is not None else 0)) / 5 + (a_alt / 5 if i is not None else 0)) if i is not None else float(exm)
    sens.append({"tag": t, "fold": k, "alt_layer": x["rederived"], "repeat_auc_saved": float(ex[i]) if i is not None else None,
                 "repeat_auc_alt": a_alt, "mean5_alt": m5_alt, "gap_alt": m5_alt - float(zauc)})
res["tie_sensitivity"] = sens; print("tie sensitivity", sens, flush=True)
res["all_pass"] = all(v["ok"] for v in res["checks"].values())
json.dump(res, open(f"{R}/verify/p26_verify_kimi_kcl.json", "w"), indent=1, default=str)
print("ALL CHECKS PASS" if res["all_pass"] else "SOME CHECKS FAIL")
