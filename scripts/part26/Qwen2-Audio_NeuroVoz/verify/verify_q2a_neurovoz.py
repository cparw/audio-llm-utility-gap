"""PART26 Qwen2-Audio NeuroVoz independent check, written separately from p26_gap_q2a_neurovoz.py (no shared code).
 1 per-clip csv equals the pulled pod OOF csv value by value, and its zero-shot column equals the zero-shot file;
 2 fold columns equal the release fold files read again by clip; clip set, labels, speakers equal the staged states;
 3 per-repeat AUCs recomputed with the rank (Mann-Whitney) formula; mean of five; zero-shot AUC; compared with the sidecar;
 4 the bootstrap redone with its own loop (pandas groupby for speaker rows, rank AUC), compared draw by draw with the npz,
   and the 2.5/97.5 percentiles compared with the sidecar;
 5 cross-checks: master_lookup per-repeat values (omni_final json), the PART25 C Linux refit OOF (rep5_refit.py, sklearn
   GroupKFold folds) clip by clip, the pod-side podverify PASS, the staged npz sha256 equal to the uploaded one.
usage: verify_q2a_neurovoz.py PULLDIR"""
import sys, csv, json, hashlib, numpy as np, pandas as pd
from scipy.stats import rankdata
R = "scores/part26/Qwen2-Audio_NeuroVoz"; PULL = sys.argv[1]
F = "<local data dir>/release/folds"
ZS = "<local data dir>/paper1_local_runs/probe2/neurovoz_zeroshot_scores.csv"
ML = "<local data dir>/paper1_local_runs/omni_final/q2a_neurovoz_nested_repeats.json"
C25 = "<local data dir>/release_from_mac/scores/part25/C/pull/p25C-3/out/q2a_neurovoz_enc_rep5_oof.csv"
out = {}; fails = []
def chk(name, ok, **kw):
    out[name] = dict(ok=bool(ok), **kw)
    if not ok: fails.append(name)
def rauc(y, s):
    r = rankdata(s); n1 = int((y == 1).sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
pcsv = pd.read_csv(f"{R}/per_clip/p26_q2a_neurovoz_perclip.csv", dtype={"speaker": str})
pod = pd.read_csv(f"{PULL}/out/p26_q2a_neurovoz_enc_nested5_oof.csv", dtype={"speaker": str})
side = json.load(open(f"{R}/per_clip/p26_q2a_neurovoz_perclip.sidecar.json"))
zs = pd.read_csv(ZS, dtype={"speaker": str}).set_index("clip")
# 1
same = all(np.array_equal(pcsv[f"p_probe_seed{s}"].values, pod[f"p_seed{s}"].values) for s in range(5)) and \
       np.array_equal(pcsv["p_probe_single"].values, pod["p_single"].values) and list(pcsv["clip"]) == list(pod["clip"])
chk("perclip_equals_pod_oof", same)
chk("perclip_zeroshot_equals_file", np.array_equal(pcsv.p_yes_zeroshot.values, zs.loc[pcsv["clip"], "p_yes"].values)
    and np.array_equal(pcsv.label.values, zs.loc[pcsv["clip"], "label"].values) and list(pcsv.speaker) == list(zs.loc[pcsv["clip"], "speaker"]))
chk("no_missing_scores", not pcsv[[f"p_probe_seed{s}" for s in range(5)]].isna().any().any(), n=len(pcsv))
# 2
z = np.load(f"{R}/stage/q2a_neurovoz_encstates.npz", allow_pickle=True)
st = pd.DataFrame({"clip": z["name"].astype(str), "spk": z["spk"].astype(str), "label": z["label"].astype(int)}).set_index("clip")
chk("clips_labels_speakers_equal_states", set(st.index) == set(pcsv["clip"]) and len(pcsv) == 1270 and
    np.array_equal(st.loc[pcsv["clip"], "label"].values, pcsv.label.values) and list(st.loc[pcsv["clip"], "spk"]) == list(pcsv.speaker))
ff = {}
for s in range(5):
    fd = pd.read_csv(f"{F}/neurovoz_groupkfold5_pod_seed{s}_UNVERIFIED.csv", dtype={"speaker_id": str}).set_index("clip_id")
    ff[s] = bool(np.array_equal(fd.loc[pcsv["clip"], "fold"].values, pcsv[f"fold_seed{s}"].values)) and list(fd.loc[pcsv["clip"], "speaker_id"]) == list(pcsv.speaker)
chk("fold_columns_equal_release_fold_files", all(ff.values()), per_seed=ff)
# 3
y = pcsv.label.values.astype(int)
per = [rauc(y, pcsv[f"p_probe_seed{s}"].values) for s in range(5)]; m5 = float(np.mean(per)); za = rauc(y, pcsv.p_yes_zeroshot.values)
chk("per_repeat_auc_rank_equals_sidecar", max(abs(a - b) for a, b in zip(per, side["probe_per_repeat"])) < 1e-12, rank=per)
chk("mean5_and_answer_equal_sidecar", abs(m5 - side["probe_mean5"]) < 1e-12 and abs(za - side["zeroshot_auc"]) < 1e-12, mean5=m5, answer=za)
chk("answer_auc_4dp_is_0.5519", round(za, 4) == 0.5519, answer=za)
# 4
sp = pcsv.speaker.values; uniq = np.array(sorted(set(sp)))
grp = pcsv.groupby("speaker").indices
rows = [np.asarray(grp[u]) for u in uniq]
rng = np.random.default_rng(0); D = np.full(2000, np.nan); M = np.full(2000, np.nan); Zb = np.full(2000, np.nan)
Pm = np.stack([pcsv[f"p_probe_seed{s}"].values for s in range(5)]); zv = pcsv.p_yes_zeroshot.values
for b in range(2000):
    pick = rng.choice(len(uniq), size=len(uniq), replace=True)
    ii = np.concatenate([rows[k] for k in pick]); yy = y[ii]
    if yy.min() == yy.max(): continue
    M[b] = np.mean([rauc(yy, Pm[r, ii]) for r in range(5)]); Zb[b] = rauc(yy, zv[ii]); D[b] = M[b] - Zb[b]
dz = np.load(f"{R}/draws/p26_q2a_neurovoz_draws.npz", allow_pickle=True)
ok = ~np.isnan(D)
lo, hi = float(np.percentile(D[ok], 2.5)), float(np.percentile(D[ok], 97.5))
chk("bootstrap_draws_equal_npz", np.allclose(D, dz["diff"], atol=1e-12, equal_nan=True) and np.allclose(M, dz["mean5"], atol=1e-12, equal_nan=True),
    max_abs=float(np.nanmax(np.abs(D - dz["diff"]))), usable=int(ok.sum()))
chk("interval_equals_sidecar", abs(lo - side["gap_ci"][0]) < 1e-12 and abs(hi - side["gap_ci"][1]) < 1e-12, lo=lo, hi=hi)
chk("point_gap_equals_sidecar", abs((m5 - za) - side["gap_point"]) < 1e-12, gap=m5 - za)
# 5
T = json.load(open(ML))["enc"]
chk("master_per_repeat_4dp", [round(a, 4) for a in per] == T["per_repeat"] and round(m5, 4) == T["mean"], master=T["per_repeat"], mine=[round(a, 4) for a in per])
c = pd.read_csv(C25, dtype={"speaker": str}).set_index("clip").loc[pcsv["clip"]]
dmax = max(float(np.max(np.abs(c[f"p_seed{s}"].values - pcsv[f"p_probe_seed{s}"].values))) for s in range(5))
fold_same = all(np.array_equal(c[f"fold_seed{s}"].values, pcsv[f"fold_seed{s}"].values) for s in range(5))
chk("part25C_refit_clip_by_clip", dmax < 1e-6, max_abs_score_diff=dmax, folds_identical=fold_same,
    part25C_per_repeat=[rauc(y, c[f"p_seed{s}"].values) for s in range(5)])
pv = json.load(open(f"{PULL}/out/p26_q2a_neurovoz_podverify.json"))
chk("podverify_PASS", pv.get("PASS") is True, max_abs_pred_diff=pv.get("max_abs_pred_diff"), layer_mismatches=len(pv.get("layer_mismatches", [])))
up = [l.split() for l in open(f"{PULL}/in/upload.sha256")]
hsh = hashlib.sha256(open(f"{R}/stage/q2a_neurovoz_encstates.npz", "rb").read()).hexdigest()
chk("staged_states_sha_equals_uploaded", any(p == "in/q2a_neurovoz_encstates.npz" and h == hsh for h, p in up))
J = json.load(open(f"{PULL}/out/p26_q2a_neurovoz_enc_nested5.json"))
chk("pod_versions_pinned", (J["versions"]["sklearn"], J["versions"]["numpy"], J["versions"]["scipy"]) == ("1.9.1", "2.1.2", "1.18.1"), versions=J["versions"])
out["FAILED"] = fails; out["PASS"] = not fails
json.dump(out, open(f"{R}/verify/verify_q2a_neurovoz.json", "w"), indent=1)
for k, v in out.items():
    if isinstance(v, dict): print(("OK  " if v["ok"] else "FAIL"), k, {a: b for a, b in v.items() if a != "ok"})
print("VERIFY PASS" if not fails else f"VERIFY FAIL {fails}")
