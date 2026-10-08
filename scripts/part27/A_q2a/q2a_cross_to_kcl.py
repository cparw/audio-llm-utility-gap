"""PART 27 A. Qwen2-Audio encoder, cross-dataset PD probe to MDVR-KCL, re-run on the saved local states.

Method copied from speech-health-repo/scripts/probing/cross_matrix.py:
  layer   = argmax over the 33 encoder layers of the mean fold AUC from GroupKFold(5) by speaker on the SOURCE only
            (folds where train or test has one class are skipped, nanargmax, first layer wins ties)
  model   = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
  fit     = all source clips at that layer, then predict_proba on every KCL clip at the same layer
  AUC     = sklearn roc_auc_score
The target (KCL) is never used to pick the layer, so the saved method is already nested with respect to KCL.
The "nested" variant below is the same rule written out on its own; it is kept as a separate column so the check is explicit.
"""
import os, json, csv, hashlib, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import sklearn

P = "<local data dir>/paper1_local_runs"
OUT = "<local data dir>/release_from_mac/scores/part27/A_q2a"
SAVED = {"pcgita": 0.860, "neurovoz": 0.744}
t0 = time.time()

def load(ds, sub="probe2"):
    d = np.load(f"{P}/{sub}/{ds}_states.npz", allow_pickle=True)
    return dict(X=d["enc"].astype(np.float32), y=d["label"].astype(int), spk=d["spk"].astype(str),
                name=d["name"].astype(str), p_yes=d["p_yes"].astype(float))

kcl = load("kcl")
kcl2 = load("kcl", "kcl_probe_recut")
same_recut = bool((kcl["name"] == kcl2["name"]).all() and np.array_equal(kcl["X"], kcl2["X"]))
print("kcl probe2 == kcl_probe_recut:", same_recut, flush=True)

# zero-shot answer per clip, from the csv, aligned by clip name
zs = {r["clip"]: r for r in csv.DictReader(open(f"{P}/probe2/kcl_zeroshot_scores.csv"))}
zs_p = np.array([float(zs[n]["p_yes"]) for n in kcl["name"]])
zs_lab = np.array([int(zs[n]["label"]) for n in kcl["name"]])
zs_spk = np.array([zs[n]["speaker"] for n in kcl["name"]])
assert (zs_lab == kcl["y"]).all() and (zs_spk == kcl["spk"]).all()
assert np.allclose(zs_p, kcl["p_yes"])
auc_zs = roc_auc_score(kcl["y"], zs_p)
print(f"KCL zero-shot AUC {auc_zs:.4f}", flush=True)

def mk():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))

def cv_perlayer(X, y, spk):  # verbatim logic of cross_matrix.py cv_perlayer
    out = []
    ns = min(5, len(set(spk)))
    for li in range(X.shape[1]):
        XL = X[:, li, :]; fold = []
        for tr, te in GroupKFold(ns).split(XL, y, spk):
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2: continue
            clf = mk(); clf.fit(XL[tr], y[tr])
            try: fold.append(roc_auc_score(y[te], clf.predict_proba(XL[te])[:, 1]))
            except Exception: pass
        out.append(np.mean(fold) if fold else np.nan)
    return out

res = {}; scores = {}
for src in ["pcgita", "neurovoz"]:
    S = load(src)
    print(f"{src}: {len(S['y'])} clips, {len(set(S['spk']))} spk, PD {int(S['y'].sum())}", flush=True)
    within = cv_perlayer(S["X"], S["y"], S["spk"])
    bl = int(np.nanargmax(within))
    # every layer's KCL AUC, for context only (never used for selection)
    kcl_by_layer = []
    for li in range(S["X"].shape[1]):
        clf = mk(); clf.fit(S["X"][:, li, :], S["y"])
        p = clf.predict_proba(kcl["X"][:, li, :])[:, 1]
        kcl_by_layer.append(roc_auc_score(kcl["y"], p))
        if li == bl: p_rep = p
    # nested variant: same source-only rule, recomputed independently
    within2 = cv_perlayer(S["X"], S["y"], S["spk"])
    bl2 = int(np.nanargmax(within2))
    clf = mk(); clf.fit(S["X"][:, bl2, :], S["y"]); p_nest = clf.predict_proba(kcl["X"][:, bl2, :])[:, 1]
    res[src] = dict(n_src=int(len(S["y"])), n_src_spk=int(len(set(S["spk"]))), layer_script=bl,
                    within_cv_auc=float(within[bl]), kcl_auc_script=float(roc_auc_score(kcl["y"], p_rep)),
                    layer_nested=bl2, kcl_auc_nested=float(roc_auc_score(kcl["y"], p_nest)),
                    saved_value=SAVED[src], within_by_layer=[float(a) for a in within],
                    kcl_auc_by_layer=[float(a) for a in kcl_by_layer],
                    kcl_best_layer_oracle=int(np.argmax(kcl_by_layer)), kcl_auc_oracle=float(np.max(kcl_by_layer)))
    scores[f"{src}_script"] = p_rep; scores[f"{src}_nested"] = p_nest
    print(f"  script layer {bl} within {within[bl]:.3f} -> KCL {res[src]['kcl_auc_script']:.4f} (saved {SAVED[src]:.3f}); "
          f"nested layer {bl2} -> KCL {res[src]['kcl_auc_nested']:.4f}; oracle layer {res[src]['kcl_best_layer_oracle']} "
          f"{res[src]['kcl_auc_oracle']:.4f}  {time.time()-t0:.0f}s", flush=True)

# speaker bootstrap, one set of 2000 draws from a fresh default_rng(0), shared by every quantity (paired)
y = kcl["y"]; spk = kcl["spk"]; uspk = np.unique(spk)
rows_of = {s: np.where(spk == s)[0] for s in uspk}
rng = np.random.default_rng(0)
draws = [np.concatenate([rows_of[s] for s in rng.choice(uspk, size=len(uspk), replace=True)]) for _ in range(2000)]
def boot(fn):
    v = []
    for ix in draws:
        if len(set(y[ix])) < 2: v.append(np.nan); continue
        v.append(fn(ix))
    v = np.array(v); return v
def ci(v): return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
n_skip = int(sum(len(set(y[ix])) < 2 for ix in draws))

zs_boot = boot(lambda ix: roc_auc_score(y[ix], zs_p[ix]))
table = [dict(direction="zero-shot", variant="answer", layer="", kcl_auc=auc_zs, auc_ci=ci(zs_boot))]
for src in ["pcgita", "neurovoz"]:
    for var in ["script", "nested"]:
        p = scores[f"{src}_{var}"]
        a = roc_auc_score(y, p)
        b = boot(lambda ix: roc_auc_score(y[ix], p[ix]))
        d = boot(lambda ix: roc_auc_score(y[ix], p[ix]) - roc_auc_score(y[ix], zs_p[ix]))
        table.append(dict(direction=f"{src}->kcl", variant=var, layer=res[src][f"layer_{var}"], kcl_auc=float(a),
                          auc_ci=ci(b), diff_vs_zeroshot=float(a - auc_zs), diff_ci=ci(d),
                          frac_draws_diff_le_0=float(np.nanmean(d <= 0))))
for t in table: print(t, flush=True)

# per-clip scores
with open(f"{OUT}/q2a_kcl_perclip_scores.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clip", "speaker", "label", "pcgita_to_kcl_script", "pcgita_to_kcl_nested",
                "neurovoz_to_kcl_script", "neurovoz_to_kcl_nested", "zeroshot_p_yes"])
    for i in range(len(y)):
        w.writerow([kcl["name"][i], spk[i], int(y[i]), repr(float(scores["pcgita_script"][i])),
                    repr(float(scores["pcgita_nested"][i])), repr(float(scores["neurovoz_script"][i])),
                    repr(float(scores["neurovoz_nested"][i])), repr(float(zs_p[i]))])

with open(f"{OUT}/q2a_cross_to_kcl_summary.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["direction", "variant", "layer", "kcl_auc", "auc_lo", "auc_hi", "diff_vs_zeroshot", "diff_lo", "diff_hi",
                "frac_draws_diff_le_0"])
    for t in table:
        w.writerow([t["direction"], t["variant"], t["layer"], "%.4f" % t["kcl_auc"], "%.4f" % t["auc_ci"][0],
                    "%.4f" % t["auc_ci"][1], ("%.4f" % t["diff_vs_zeroshot"]) if "diff_ci" in t else "",
                    ("%.4f" % t["diff_ci"][0]) if "diff_ci" in t else "", ("%.4f" % t["diff_ci"][1]) if "diff_ci" in t else "",
                    ("%.4f" % t["frac_draws_diff_le_0"]) if "diff_ci" in t else ""])

def sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
side = dict(
    script=os.path.abspath(__file__), python=sys.version.split()[0], sklearn=sklearn.__version__, numpy=np.__version__,
    method_source="speech-health-repo/scripts/probing/cross_matrix.py (read task, train-best layer by source GroupKFold(5) by speaker)",
    saved_table="speech-health-repo/results/health/probing_qwen/cross_matrix_read.csv (pcgita->kcl 0.860, neurovoz->kcl 0.744)",
    inputs={k: f"{P}/probe2/{k}_states.npz" for k in ["pcgita", "neurovoz", "kcl"]},
    zeroshot_file=f"{P}/probe2/kcl_zeroshot_scores.csv",
    kcl_probe2_equals_kcl_probe_recut=same_recut,
    kcl_clip_set="MDVR-KCL ReadText, 37 speakers (16 PD, 21 HC), one clip each, first 30 s of the recording",
    bootstrap=dict(n=2000, rng="numpy default_rng(0), fresh, one set of draws shared by every row (paired)",
                   unit="KCL speaker, rng.choice(unique speakers, 37, replace=True)", percentiles=[2.5, 97.5],
                   draws_skipped_one_class=n_skip),
    per_source=res, table=table, zeroshot_auc=auc_zs,
    sha256={"perclip": sha(f"{OUT}/q2a_kcl_perclip_scores.csv"), "summary": sha(f"{OUT}/q2a_cross_to_kcl_summary.csv")},
    runtime_s=round(time.time() - t0, 1))
json.dump(side, open(f"{OUT}/q2a_cross_to_kcl.sidecar.json", "w"), indent=1)
print("DONE", round(time.time() - t0), "s", flush=True)
