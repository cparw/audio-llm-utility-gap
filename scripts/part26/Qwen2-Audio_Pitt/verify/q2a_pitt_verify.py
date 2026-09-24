"""PART26 Qwen2-Audio Pitt independent check, written separately from q2a_pitt_gap.py (no shared code).
 1 per-clip csv rows vs the original states file (clip, label, speaker), the zero-shot file (p_yes, label, speaker),
   the release fold files (fold per clip, speaker id) and the pod OOF csv (probe scores, string-equal);
 2 per-repeat AUCs, mean of five, answer AUC and gap recomputed with a Mann-Whitney rank formula;
 3 the 2000-draw speaker bootstrap rebuilt from a fresh default_rng(0) with its own speaker-to-rows map and rank AUC,
   compared draw by draw with the saved npz and with the sidecar interval;
 4 Mac refit of every outer fold at the pod's saved layer (cross-platform sanity, not exact);
 5 the master_lookup five-repeat values and the PART25 C Linux refit for comparison; 6 the pod no longer exists."""
import csv, json, os, subprocess, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
C = "scores/part26/Qwen2-Audio_Pitt"
SRC = "<local data dir>/paper1_local_runs/probe2/pitt_states.npz"
ZS = "<local data dir>/paper1_local_runs/probe2/pitt_zeroshot_scores.csv"
FD = "<local data dir>/release/folds"
MJ = "<local data dir>/paper1_local_runs/omni_final/q2a_pitt_nested_repeats.json"
side = json.load(open(f"{C}/per_clip/q2a_pitt_perclip.sidecar.json"))
rows = list(csv.DictReader(open(f"{C}/per_clip/q2a_pitt_perclip.csv")))
pod = {r["clip"]: r for r in csv.DictReader(open(f"{C}/pull/p26-q2a-pitt/out/p26_q2a_pitt_enc_nested5_oof.csv"))}
podj = json.load(open(f"{C}/pull/p26-q2a-pitt/out/p26_q2a_pitt_enc_nested5.json"))
out = {"checks": {}}; ck = out["checks"]
z = np.load(SRC, allow_pickle=True)
sn = [str(x) for x in z["name"]]; sl = [int(x) for x in z["label"]]; ss = [str(x) for x in z["spk"]]
ck["clip_order_equals_states"] = [r["clip"] for r in rows] == sn
ck["labels_equal_states"] = [int(r["label"]) for r in rows] == sl
ck["speakers_equal_states"] = [r["speaker"] for r in rows] == ss
zr = {r["clip"]: r for r in csv.DictReader(open(ZS))}
ck["zeroshot_rows_equal"] = len(zr) == len(rows) and all(float(zr[r["clip"]]["p_yes"]) == float(r["p_yes_zeroshot"]) and
    int(zr[r["clip"]]["label"]) == int(r["label"]) and zr[r["clip"]]["speaker"] == r["speaker"] for r in rows)
fok = True
for s in range(5):
    fm = {r["clip_id"]: r for r in csv.DictReader(open(f"{FD}/pitt_groupkfold5_pod_seed{s}_UNVERIFIED.csv"))}
    fok &= len(fm) == len(rows) and all(int(fm[r["clip"]]["fold"]) == int(r[f"fold_seed{s}"]) and fm[r["clip"]]["speaker_id"] == r["speaker"] for r in rows)
fm = {r["clip_id"]: r for r in csv.DictReader(open(f"{FD}/pitt_groupkfold5_pod.csv"))}
fok &= all(int(fm[r["clip"]]["fold"]) == int(r["fold_single"]) for r in rows)
ck["folds_equal_release_fold_files"] = bool(fok)
ck["probe_scores_string_equal_pod_oof"] = len(pod) == len(rows) and all(
    all(pod[r["clip"]][f"p_seed{s}"] == r[f"p_probe_seed{s}"] for s in range(5)) and pod[r["clip"]]["p_single"] == r["p_probe_single"] for r in rows)
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_probe_seed{s}"]) for r in rows] for s in range(5)]); zs = np.array([float(r["p_yes_zeroshot"]) for r in rows])
def mw(yy, sc):
    rk = rankdata(sc); a = int(yy.sum()); b = len(yy) - a
    return (rk[yy == 1].sum() - a * (a + 1) / 2) / (a * b)
per = [mw(y, P[s]) for s in range(5)]; m5 = float(np.mean(per)); za = mw(y, zs)
out["recomputed"] = {"per_repeat": per, "mean5": m5, "answer_auc": za, "gap": m5 - za}
ck["per_repeat_equal_sidecar"] = max(abs(a - b) for a, b in zip(per, side["probe"]["per_repeat_auc"])) < 1e-12
ck["answer_equal_sidecar"] = abs(za - side["zeroshot"]["auc"]) < 1e-12
ck["gap_equal_sidecar"] = abs((m5 - za) - side["gap"]["point"]) < 1e-12
# bootstrap rebuilt
keys = sorted(set(spk.tolist())); where = {k: [] for k in keys}
for i, s in enumerate(spk): where[s].append(i)
g = np.random.default_rng(0); D = np.load(side["files"]["draws_npz"], allow_pickle=True)
dif = np.empty(2000); mm = np.empty(2000); zz = np.empty(2000); idx_all = np.empty((2000, len(keys)), int)
for b in range(2000):
    pick = g.choice(len(keys), size=len(keys), replace=True); idx_all[b] = pick
    ii = np.array([j for p in pick for j in where[keys[p]]]); yy = y[ii]
    mm[b] = np.mean([mw(yy, P[s][ii]) for s in range(5)]); zz[b] = mw(yy, zs[ii]); dif[b] = mm[b] - zz[b]
ck["bootstrap_speaker_order_equals_saved"] = [str(x) for x in D["speakers"]] == keys
ck["bootstrap_idx_equal_saved"] = bool(np.array_equal(idx_all, D["speaker_idx"]))
out["bootstrap_max_abs_diff_vs_saved"] = {"diff": float(np.max(np.abs(dif - D["diff"]))), "mean5": float(np.max(np.abs(mm - D["mean5"]))),
                                          "zeroshot": float(np.max(np.abs(zz - D["zeroshot"])))}
lo, hi = float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))
out["interval_recomputed"] = [lo, hi]
ck["bootstrap_draws_equal_saved"] = max(out["bootstrap_max_abs_diff_vs_saved"].values()) < 1e-12
ck["interval_equal_sidecar"] = abs(lo - side["gap"]["ci_2.5"]) < 1e-12 and abs(hi - side["gap"]["ci_97.5"]) < 1e-12
ck["interval_4dp_equal_sidecar"] = [round(lo, 4), round(hi, 4)] == side["gap"]["ci_4dp"]
# Mac refit at the saved layers
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X = z["enc"]; refit = {}
for s in range(5):
    L = podj["tags"][f"seed{s}"]["layers"]; f = np.array([int(r[f"fold_seed{s}"]) for r in rows]); p = np.zeros(len(y))
    for k in range(5):
        tr, te = f != k, f == k; sc = StandardScaler().fit(X[tr][:, L[k]])
        p[te] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr][:, L[k]]), y[tr]).predict_proba(sc.transform(X[te][:, L[k]]))[:, 1]
    refit[f"seed{s}"] = {"auc_mac": mw(y, p), "auc_pod": per[s], "abs_auc_diff": abs(mw(y, p) - per[s]), "max_abs_pred_diff": float(np.max(np.abs(p - P[s])))}
out["mac_refit_at_saved_layers"] = refit
ck["mac_refit_auc_within_0.005"] = all(v["abs_auc_diff"] < 0.005 for v in refit.values())
M = json.load(open(MJ))["enc"]
out["master"] = {"per_repeat": M["per_repeat"], "mean": M["mean"], "abs_diff_4dp": [round(abs(round(a, 4) - b), 4) for a, b in zip(per, M["per_repeat"])],
                 "mean5_minus_master": m5 - M["mean"]}
ck["per_repeat_within_0.001_of_master"] = all(abs(a - b) <= 0.001 for a, b in zip(per, M["per_repeat"]))
ck["answer_equals_master_0.6173"] = round(za, 4) == 0.6173
st = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-H", "Authorization: Bearer " + open(os.path.expanduser("~/.runpod_key")).read().strip(),
                     "https://rest.runpod.io/v1/pods/dow5f8gprrxhi2"], capture_output=True, text=True).stdout
out["pod_rest_status_http"] = st; ck["pod_deleted"] = st == "404"
out["PASS"] = all(bool(v) for v in ck.values())
cv = lambda o: bool(o) if isinstance(o, np.bool_) else float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o)
json.dump(out, open(f"{C}/verify/q2a_pitt_verify.json", "w"), indent=1, default=cv)
print(json.dumps(out, indent=1, default=cv))
