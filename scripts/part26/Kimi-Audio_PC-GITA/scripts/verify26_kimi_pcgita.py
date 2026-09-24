"""PART26 Kimi-Audio PC-GITA: independent Mac verification, written separately from the pod probe and the gap script
(no shared code). Checks:
 1 the fold columns of the pod OOF csv equal the release fold files, read again here by clip;
 2 every per-repeat AUC recomputed with the rank (Mann-Whitney) formula equals the pod JSON;
 3 Mac refit of all 25 outer folds (5 repeats x 5 folds) at the pod's chosen layers (StandardScaler then
   LogisticRegression(max_iter=2000, class_weight='balanced'), the model spec), predictions vs the OOF columns;
 4 Mac re-derivation of the inner layer choice for every outer fold of repeat seed0: own greedy group fill (stable tie order
   over np.unique speaker order, renumbered by default_rng(0) as in the estimator), 32 layers x 4 inner folds, mean inner AUC;
   chosen layer and the 32 inner scores vs the pod JSON;
 5 own speaker bootstrap (2000 draws, fresh default_rng(0), unique speakers in sorted order, rng.choice with replacement,
   all clips of each drawn speaker, rank AUC) of mean-of-five minus zero-shot, vs the sidecar interval and the saved draws;
 6 the per-clip csv: probe columns equal the OOF csv, p_yes_zeroshot equals the canonical zero-shot file;
 7 encoder recompute on the Mac for 4 clips with a different code path: Hugging Face WhisperModel (eager attention, float32,
   CPU) from the whisper-large-v3 folder of the same Kimi-Audio-7B-Instruct snapshot (downloaded by the PART26 E-DAIC cell,
   read only), Hugging Face WhisperFeatureExtractor (128 mel), librosa 16 kHz, first 30 s; layer output mean over the frames
   Kimi keeps vs the pod enc (cosine, relative L2, best-matching layer index).
Cross-version note (added before the first run): the Mac has sklearn 1.7.2 / numpy 2.2.6 on arm64, the pod has the pinned
sklearn 1.9.1 / numpy 2.1.2 / scipy 1.18.1 on x86_64. The Qwen2-Audio PC-GITA cell saw a Mac refit prediction difference of
0.037 for the same reason, so 3 and 4 pass on AUC agreement and layer choice (near ties allowed), with the raw differences
recorded. The exact same-version check is the pod-side p25a_podverify.py (pinned versions).
usage: verify26_kimi_pcgita.py"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import sys, csv, json, datetime, platform, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from joblib import Parallel, delayed
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
C = "scores/part26/Kimi-Audio_PC-GITA"
POD = f"{C}/pull/p26-kimi-pcgita-cpu/out"
OOF = f"{POD}/p26_kimi_pcgita_enc_nested5_oof.csv"; PJ = f"{POD}/p26_kimi_pcgita_enc_nested5.json"
ST = f"{C}/stage/kimi_pcgita_encstates.npz"; GPUST = f"{C}/pull/p26-kimi-pcgita-gpu/out/kimi_pcgita_encstates.npz"
ZS = "<local data dir>/release/overnight2/part3/kimi_pcgita_zeroshot_scores.csv"
FD = "<local data dir>/release/folds"
SIDE = f"{C}/Kimi-Audio_PC-GITA_perclip.sidecar.json"; PCC = f"{C}/Kimi-Audio_PC-GITA_perclip.csv"; DR = f"{C}/draws/Kimi-Audio_PC-GITA_draws.npz"
WHISPER = "<local data dir>/release_from_mac/scores/part26/Kimi-Audio_E-DAIC/verify/hf_whisper/whisper-large-v3"
MACW = "<local data dir>/paper1_local_runs/clips_local/spanish_dataset"
rec = {"created_utc": datetime.datetime.utcnow().isoformat() + "Z", "script": os.path.abspath(__file__)}
def mw(yy, s):
    r = rankdata(s); n1 = int((yy == 1).sum()); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
z = np.load(ST, allow_pickle=True); X = z["enc"].astype(np.float32); Y = z["label"].astype(int)
SPK = z["spk"].astype(str); NM = z["name"].astype(str)
rows = list(csv.DictReader(open(OOF))); J = json.load(open(PJ))
assert [r["clip"] for r in rows] == list(NM)
TAGS = [f"seed{s}" for s in range(5)]
Pm = np.array([[float(r[f"p_{t}"]) for r in rows] for t in TAGS])
# 1 fold columns
fok = {}
for t in TAGS + ["single"]:
    fn = f"{FD}/pcgita_groupkfold5_pod.csv" if t == "single" else f"{FD}/pcgita_groupkfold5_pod_{t}.csv"
    f = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(fn))}
    fok[t] = all(int(r[f"fold_{t}"]) == f[r["clip"]] for r in rows)
rec["1_fold_columns_equal_release_files"] = fok
# 2 AUCs
per = [mw(Y, Pm[k]) for k in range(5)]
rec["2_rank_auc_per_repeat"] = per; rec["2_max_abs_vs_pod_json"] = float(np.max(np.abs(np.array(per) - np.array(J["per_repeat_auc"]))))
rec["2_mean5"] = float(np.mean(per))
# 3 refit at saved layers
def fit(tr, te, l):
    s = StandardScaler().fit(X[tr, l])
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(s.transform(X[tr, l]), Y[tr]).predict_proba(s.transform(X[te, l]))[:, 1]
mx = 0.0; jobs = []
for k, t in enumerate(TAGS):
    fo = np.array([int(r[f"fold_{t}"]) for r in rows]); lay = J["tags"][t]["layers"]
    for f in range(5): jobs.append((k, np.where(fo != f)[0], np.where(fo == f)[0], lay[f]))
outs = Parallel(n_jobs=12)(delayed(fit)(tr, te, l) for (_, tr, te, l) in jobs)
ref = np.zeros_like(Pm)
for (k, tr, te, l), p in zip(jobs, outs): ref[k, te] = p
rec["3_mac_refit_max_abs_pred_diff"] = float(np.abs(ref - Pm).max())
rec["3_mac_refit_auc"] = [mw(Y, ref[k]) for k in range(5)]
rec["3_mac_refit_auc_max_abs_vs_pod"] = float(np.max(np.abs(np.array(rec["3_mac_refit_auc"]) - np.array(per))))
rec["3_mac_refit_mean5_abs_vs_pod"] = float(abs(np.mean(rec["3_mac_refit_auc"]) - np.mean(per)))
# 4 inner layer re-derivation, seed0
def fill(groups, k):
    keys = sorted(set(groups)); cnt = {g: 0 for g in keys}
    for g in groups: cnt[g] += 1
    arr = np.array([cnt[g] for g in keys]); order = np.argsort(arr, kind="stable")[::-1]; load = np.zeros(k); fg = {}
    for gi in order:
        j = int(np.argmin(load)); fg[keys[gi]] = j; load[j] += arr[gi]
    return np.array([fg[g] for g in groups])
rng = np.random.default_rng(0); u = sorted(set(SPK)); perm = {s: i for i, s in enumerate(rng.permutation(np.array(u)))}
G0 = np.array([perm[s] for s in SPK])
fo0 = np.array([int(r["fold_seed0"]) for r in rows])
rec["4_seed0_own_fill_equals_saved_outer"] = bool(np.array_equal(fill(list(G0), 5), fo0))
def inner_score(tr, l):
    f4 = fill(list(G0[tr]), 4); s = 0.0
    for j in range(4):
        a, b = tr[f4 != j], tr[f4 == j]
        sc = StandardScaler().fit(X[a, l]); p = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[a, l]), Y[a]).predict_proba(sc.transform(X[b, l]))[:, 1]
        s += mw(Y[b], p) if len(set(Y[b])) > 1 else 0.5
    return s / 4
inn = {}
for f in range(5):
    tr = np.where(fo0 != f)[0]
    sc = Parallel(n_jobs=14)(delayed(inner_score)(tr, l) for l in range(32))
    inn[f] = {"layer": int(np.argmax(sc)), "pod_layer": J["tags"]["seed0"]["layers"][f],
              "max_abs_inner_diff": float(np.max(np.abs(np.array(sc) - np.array(J["tags"]["seed0"]["inner"][f]))))}
    print("inner seed0 fold", f, inn[f], flush=True)
rec["4_seed0_inner"] = inn
rec["4_seed0_layers_match"] = all(v["layer"] == v["pod_layer"] for v in inn.values())
for f, v in inn.items():
    pin = J["tags"]["seed0"]["inner"][f]; v["pod_inner_at_mac_layer_minus_pod_best"] = float(pin[v["layer"]] - max(pin))
rec["4_seed0_layers_match_or_near_tie"] = all(v["layer"] == v["pod_layer"] or (-v["pod_inner_at_mac_layer_minus_pod_best"] <= 2 * v["max_abs_inner_diff"]) for v in inn.values())
# 5 own bootstrap
zmap = {r["clip"]: float(r["p_yes"]) for r in csv.DictReader(open(ZS))}; ZP = np.array([zmap[n] for n in NM])
us = np.unique(SPK); members = {i: np.where(SPK == s)[0] for i, s in enumerate(us)}
r2 = np.random.default_rng(0); D = []
for b in range(2000):
    pick = r2.choice(len(us), size=len(us), replace=True); ii = np.concatenate([members[i] for i in pick])
    if len(set(Y[ii])) < 2: D.append(np.nan); continue
    D.append(np.mean([mw(Y[ii], Pm[k][ii]) for k in range(5)]) - mw(Y[ii], ZP[ii]))
D = np.array(D); ok = ~np.isnan(D)
lo, hi = float(np.percentile(D[ok], 2.5)), float(np.percentile(D[ok], 97.5))
S = json.load(open(SIDE)); dr = np.load(DR)
rec["5_point_gap"] = float(np.mean(per) - mw(Y, ZP)); rec["5_zeroshot_auc"] = mw(Y, ZP)
rec["5_ci"] = [lo, hi]; rec["5_ci_vs_sidecar_max_abs"] = float(max(abs(lo - S["gap_ci"][0]), abs(hi - S["gap_ci"][1])))
rec["5_point_vs_sidecar_abs"] = abs(rec["5_point_gap"] - S["gap_point"])
rec["5_draws_max_abs_vs_saved"] = float(np.nanmax(np.abs(D - dr["diff"]))); rec["5_usable"] = int(ok.sum())
# 6 per-clip csv
pc = list(csv.DictReader(open(PCC)))
rec["6_perclip_rows"] = len(pc)
rec["6_probe_columns_equal_oof"] = all(pc[i][f"p_probe_{t}"] == rows[i][f"p_{t}"] for i in range(len(rows)) for t in TAGS)
rec["6_zeroshot_equal_canonical"] = all(float(pc[i]["p_yes_zeroshot"]) == zmap[pc[i]["clip"]] for i in range(len(pc)))
# 7 encoder recompute
try:
    import torch, librosa, transformers
    from transformers import WhisperModel, WhisperFeatureExtractor
    g = np.load(GPUST, allow_pickle=True); gn = list(g["name"].astype(str)); kf = g["kept_frames"]; dur = g["dur_file_s"]
    mac = {}
    for line in open(f"{C}/stage/mf_pcgita_mac.csv").read().splitlines()[1:]:
        p = line.rsplit(",", 2)[0].strip('"'); mac[os.path.basename(p)] = p
    order = np.argsort(dur); picks = [gn[order[0]], gn[order[len(order) // 3]], gn[order[2 * len(order) // 3]], gn[order[-1]]]
    fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000)
    enc = WhisperModel.from_pretrained(WHISPER, torch_dtype=torch.float32, attn_implementation="eager").encoder.eval()
    grab = {}
    hs = [l.register_forward_hook((lambda i: (lambda m, a, o: grab.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0].detach())))(i)) for i, l in enumerate(enc.layers)]
    res = {}
    for c in picks:
        i0 = gn.index(c); k = int(kf[i0])
        x = librosa.load(mac[c], sr=16000)[0][:480000]
        with torch.no_grad(): enc(fe(x, sampling_rate=16000, return_tensors="pt").input_features)
        H = np.stack([grab[i][:k].mean(0).numpy() for i in range(32)]).astype(np.float64); P = g["enc"][i0].astype(np.float64)
        HA = np.stack([grab[i].mean(0).numpy() for i in range(32)]).astype(np.float64); PA = g["enc_allframes"][i0].astype(np.float64)
        cos = [float(H[i] @ P[i] / (np.linalg.norm(H[i]) * np.linalg.norm(P[i]))) for i in range(32)]
        rel = [float(np.linalg.norm(H[i] - P[i]) / np.linalg.norm(P[i])) for i in range(32)]
        best = [int(np.argmin([np.linalg.norm(H[i] - P[j]) for j in range(32)])) for i in range(32)]
        cosA = [float(HA[i] @ PA[i] / (np.linalg.norm(HA[i]) * np.linalg.norm(PA[i]))) for i in range(32)]
        res[c] = {"dur_file_s": float(dur[i0]), "kept_frames": k, "min_cosine": min(cos), "median_cosine": float(np.median(cos)),
                  "max_rel_l2": max(rel), "median_rel_l2": float(np.median(rel)), "best_match_is_same_layer": best == list(range(32)),
                  "allframes_min_cosine": min(cosA)}
        print("recompute", c, res[c], flush=True)
    rec["7_encoder_recompute"] = {"weights": WHISPER, "transformers": transformers.__version__, "torch": torch.__version__, "clips": res,
                                  "pass": all(v["best_match_is_same_layer"] and v["min_cosine"] > 0.99 for v in res.values())}
except Exception as e:
    rec["7_encoder_recompute"] = {"error": repr(e), "pass": False}
rec["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()}
rec["PASS"] = bool(all(fok.values()) and rec["2_max_abs_vs_pod_json"] < 1e-9 and rec["3_mac_refit_auc_max_abs_vs_pod"] < 2e-3 and rec["3_mac_refit_mean5_abs_vs_pod"] < 5e-4
                   and rec["4_seed0_layers_match_or_near_tie"] and rec["4_seed0_own_fill_equals_saved_outer"]
                   and rec["5_ci_vs_sidecar_max_abs"] < 1e-9 and rec["5_point_vs_sidecar_abs"] < 1e-12 and rec["5_draws_max_abs_vs_saved"] < 1e-9
                   and rec["6_perclip_rows"] == 1100 and rec["6_probe_columns_equal_oof"] and rec["6_zeroshot_equal_canonical"]
                   and rec["7_encoder_recompute"]["pass"])
json.dump(rec, open(f"{C}/verify/verify26_kimi_pcgita.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in rec.items() if k not in ("4_seed0_inner", "7_encoder_recompute")}, indent=1, default=str))
print("VERIFY PASS" if rec["PASS"] else "VERIFY FAIL")
