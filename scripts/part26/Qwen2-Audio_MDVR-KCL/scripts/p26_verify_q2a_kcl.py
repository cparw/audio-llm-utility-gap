"""PART26 independent Mac check of the Qwen2-Audio MDVR-KCL cell. Written separately from p26_gap_q2a_kcl.py (no shared code).
 1 pulled pod outputs: sha256 equal the pod's own out_sha256.txt;
 2 staged states equal the source probe2/kcl_states.npz (enc, label, spk, p_yes, name), and the states p_yes equal the zero-shot file;
 3 OOF csv fold columns equal the release fold files read again here by clip;
 4 per-repeat AUCs recomputed as exact fractions (Mann-Whitney, ties 1/2) from the OOF csv, against the pod json and the saved Table 1 json;
 5 zero-shot AUC recomputed exactly; clip set, labels and speakers equal;
 6 per-clip csv columns equal the pod OOF csv and the zero-shot file, string for string;
 7 bootstrap rebuilt with an own loop (rank AUC via scipy rankdata) from the same rng; draws and interval compared with the saved npz and sidecar;
 8 the podverify layer mismatch: the exact-fraction tie log from the pod is read and checked;
 9 Mac refit at the saved layers (StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced')): AUC and max prediction difference recorded (a different BLAS, so not expected to be bit equal).
Writes verify/p26_verify_q2a_kcl.json."""
import os, csv, json, hashlib, numpy as np
from fractions import Fraction
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

R = "scores/part26/Qwen2-Audio_MDVR-KCL"
POD = f"{R}/pull/p26-q2a-kcl"
SRC = "<local data dir>/paper1_local_runs/probe2/kcl_states.npz"
ZS = "<local data dir>/paper1_local_runs/probe2/kcl_zeroshot_scores.csv"
T1 = "<local data dir>/release/omni_final/q2a_kcl_nested_repeats.json"
FD = "<local data dir>/release/folds"
res = {"checks": {}}
def ck(name, ok, **kw):
    res["checks"][name] = dict(ok=bool(ok), **kw); print(("PASS " if ok else "FAIL ") + name, kw if kw else "", flush=True)
def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
# 1
bad = []; cnt = 0
for line in open(f"{POD}/out_sha256.txt"):
    h, f = line.split(); cnt += 1
    if sha(f"{POD}/out/{f}") != h: bad.append(f)
ck("pulled_out_sha256_match_pod", cnt > 0 and not bad, n_files=cnt, mismatched=bad)
# 2
a = np.load(f"{POD}/in/q2a_kcl_encstates.npz", allow_pickle=True) if os.path.exists(f"{POD}/in/q2a_kcl_encstates.npz") else np.load(f"{R}/stage/in/q2a_kcl_encstates.npz", allow_pickle=True)
s = np.load(SRC, allow_pickle=True)
ck("staged_states_equal_source", all(np.array_equal(a[k], s[k]) for k in ("enc", "label", "spk", "p_yes", "name")), shape=list(a["enc"].shape))
names = [str(x) for x in s["name"]]; spk = [str(x) for x in s["spk"]]; lab = [int(x) for x in s["label"]]
zrows = list(csv.DictReader(open(ZS))); zmap = {r["clip"]: r for r in zrows}
ck("states_pyes_equal_zeroshot", max(abs(float(zmap[c]["p_yes"]) - float(p)) for c, p in zip(names, s["p_yes"])) < 1e-12)
# 3
rows = list(csv.DictReader(open(f"{POD}/out/p26_q2a_kcl_enc_nested5_oof.csv")))
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
J = json.load(open(f"{POD}/out/p26_q2a_kcl_enc_nested5.json")); T = json.load(open(T1))["enc"]
ex = [exact_auc(lab, [float(r[f"p_seed{i}"]) for r in rows]) for i in range(5)]
exm = sum(ex) / 5
ck("per_repeat_auc_exact_equals_pod_json", all(abs(float(e) - j) < 1e-12 for e, j in zip(ex, J["per_repeat_auc"])) and abs(float(exm) - J["mean5"]) < 1e-12,
   exact=[str(e) for e in ex], mean5_exact=str(exm), mean5=float(exm))
ck("per_repeat_auc_equals_table1_4dp", [round(float(e), 4) for e in ex] == T["per_repeat"] and round(float(exm), 4) == T["mean"],
   rerun=[round(float(e), 4) for e in ex], table1=T["per_repeat"], table1_mean=T["mean"])
ck("single_split_auc_exact", abs(float(exact_auc(lab, [float(r["p_single"]) for r in rows])) - J["single_split_auc"]) < 1e-12)
# 5
zy = [int(zmap[c]["label"]) for c in names]; zsp = [zmap[c]["speaker"] for c in names]
zauc = exact_auc(lab, [float(zmap[c]["p_yes"]) for c in names])
ck("zeroshot_clips_labels_speakers_equal", set(zmap) == set(names) and zy == lab and zsp == spk and len(zrows) == len(names))
ck("zeroshot_auc_exact_equals_0.5923_4dp", round(float(zauc), 4) == 0.5923, exact=str(zauc), value=float(zauc))
# 6
S = json.load(open(f"{R}/Q2A_kcl_perclip.sidecar.json"))
pc = list(csv.DictReader(open(f"{R}/Q2A_kcl_perclip.csv")))
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
rng = np.random.default_rng(0); D = []; M = []; ZB = []
for b in range(2000):
    pick = rng.choice(len(U), size=len(U), replace=True)
    ii = np.array([i for k in pick for i in byspk[U[k]]]); yy = Y[ii]
    if yy.min() == yy.max(): D.append(np.nan); M.append(np.nan); ZB.append(np.nan); continue
    m = np.mean([rank_auc(yy, Pm[r][ii]) for r in range(5)]); zb = rank_auc(yy, Z[ii])
    D.append(m - zb); M.append(m); ZB.append(zb)
D = np.array(D); ok = ~np.isnan(D)
lo, hi = np.percentile(D[ok], 2.5), np.percentile(D[ok], 97.5)
dr = np.load(f"{R}/Q2A_kcl_draws.npz", allow_pickle=True)
same_nan = np.array_equal(np.isnan(dr["diff"]), ~ok)
mx = float(np.nanmax(np.abs(dr["diff"] - D)))
ck("bootstrap_rebuilt_equals_saved_draws", same_nan and mx < 1e-12 and int(ok.sum()) == S["gap"]["usable_draws"], max_abs_draw_diff=mx, usable=int(ok.sum()))
ck("bootstrap_interval_equals_sidecar", abs(lo - S["gap"]["ci_2p5_97p5"][0]) < 1e-12 and abs(hi - S["gap"]["ci_2p5_97p5"][1]) < 1e-12, lo=float(lo), hi=float(hi))
ck("gap_point_equals_sidecar", abs(float(exm - zauc) - S["gap"]["point"]) < 1e-12, gap=float(exm - zauc))
# 8
PV = json.load(open(f"{POD}/out/p26_q2a_kcl_podverify.json"))
tl = open(f"{POD}/logs/tiecheck_seed2_fold1.log").read() if os.path.exists(f"{POD}/logs/tiecheck_seed2_fold1.log") else ""
inner = J["tags"]["seed2"]["inner"][1]
ck("podverify_only_mismatch_is_exact_tie", PV["max_abs_pred_diff"] == 0.0 and PV["abs_mean5_diff"] < 1e-9 and PV["layer_mismatches"] == [{"tag": "seed2", "fold": 1, "saved": 14, "rederived": 6}]
   and "EXACT_TIE_14_vs_6 True" in tl and abs(inner[14] - inner[6]) < 1e-12 and max(inner) == inner[14],
   podverify_PASS_flag=PV["PASS"], inner_14=inner[14], inner_6=inner[6], tie_log=tl.strip().splitlines())
ck("table1_saved_run_also_chose_layer14_seed2_fold1", J["tags"]["seed2"]["layers"] == [21, 14, 4, 11, 26], layers=J["tags"]["seed2"]["layers"])
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
res["all_pass_except_podverify_flag"] = all(v["ok"] for v in res["checks"].values())
json.dump(res, open(f"{R}/verify/p26_verify_q2a_kcl.json", "w"), indent=1, default=str)
print("ALL CHECKS PASS" if res["all_pass_except_podverify_flag"] else "SOME CHECKS FAIL")
