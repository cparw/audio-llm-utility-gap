"""PART26 Kimi-Audio MDVR-KCL: check the re-extracted encoder states before any probe runs (Mac, read only on the sources).
 1 pulled out/ files match the pod's own sha256 list;
 2 clip set and order, labels and speakers equal the saved zero-shot file (the answer) and all six fold files;
 3 enc is (37, 32, 1280), every value finite, no two layers identical, no constant columns across clips at every layer;
 4 every clip kept 1500 encoder frames (30 s) and the final layer_norm output equals Kimi's continuous feature (ln_vs_feat);
 5 reproduction: the re-run p_yes against the saved zero-shot p_yes, and the re-run LM states against the saved
   overnight2/kimi_new/kimi_kcl_states.npz (same forward pass, so these show the same audio, window and weights);
 6 states "states ok" definition of the inventory.
Then copies the states npz, byte for byte, to stage/cpu/in/kimi_kcl_encstates.npz and writes the CPU upload sha list.
Writes verify/p26_states_check_kimi_kcl.json."""
import os, csv, json, hashlib, shutil, numpy as np
from sklearn.metrics import roc_auc_score
R = "scores/part26/Kimi-Audio_MDVR-KCL"
POD = f"{R}/pull/p26-kimi-kcl-gpu"
ST = f"{POD}/out/p26_kimi_kcl_states.npz"
ZS = "<local data dir>/release/overnight2/kimi_new/kimi_kcl_zeroshot_scores.csv"
OLD = "<local data dir>/release/overnight2/kimi_new/kimi_kcl_states.npz"
FD = "<local data dir>/release/folds"
res = {"checks": {}}
def ck(name, ok, **kw):
    res["checks"][name] = dict(ok=bool(ok), **kw); print(("PASS " if ok else "FAIL ") + name, kw if kw else "", flush=True)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
bad = []; cnt = 0
for line in open(f"{POD}/out_sha256.txt"):
    h, f = line.split(); cnt += 1
    if sha(f"{POD}/out/{f}") != h: bad.append(f)
ck("pulled_out_sha256_match_pod", cnt > 0 and not bad, n_files=cnt, mismatched=bad)
z = np.load(ST, allow_pickle=True)
names = [str(x) for x in z["name"]]; spk = [str(x) for x in z["spk"]]; lab = [int(x) for x in z["label"]]
zr = list(csv.DictReader(open(ZS)))
ck("clips_labels_speakers_equal_zeroshot_file_in_order", names == [r["clip"] for r in zr] and spk == [r["speaker"] for r in zr]
   and lab == [int(r["label"]) for r in zr], n=len(names), n_pos=sum(lab), n_spk=len(set(spk)))
fok = {}
for fn in ["kcl_groupkfold5_pod.csv"] + [f"kcl_groupkfold5_pod_seed{i}_UNVERIFIED.csv" for i in range(5)]:
    fm = {r["clip_id"]: r["speaker_id"] for r in csv.DictReader(open(f"{FD}/{fn}"))}
    fok[fn] = set(fm) == set(names) and len(fm) == len(names) and all(fm[c] == s for c, s in zip(names, spk))
ck("clips_speakers_equal_all_fold_files", all(fok.values()), per_file=fok)
E = z["enc"]
ck("enc_shape_37x32x1280", tuple(E.shape) == (37, 32, 1280), shape=list(E.shape), dtype=str(E.dtype))
ck("enc_all_finite", bool(np.isfinite(E).all()))
ck("enc_layers_distinct", all(not np.array_equal(E[:, i], E[:, j]) for i in range(32) for j in range(i + 1, 32)))
nconst = [int((E[:, l].std(axis=0) == 0).sum()) for l in range(32)]
ck("enc_no_constant_columns", max(nconst) == 0, constant_columns_per_layer_max=max(nconst))
fr = z["enc_frames"]; lf = z["ln_vs_feat_maxabs"]
ck("enc_frames_1500_every_clip", bool((fr == 1500).all()), frames_min=int(fr.min()), frames_max=int(fr.max()))
ck("final_layernorm_equals_kimi_continuous_feature", float(lf.max()) == 0.0, max_abs=float(lf.max()))
zp = np.array([float(r["p_yes"]) for r in zr]); rp = z["p_yes"].astype(float)
d = np.abs(zp - rp); a_saved = float(roc_auc_score(lab, zp)); a_rerun = float(roc_auc_score(lab, rp))
ck("rerun_p_yes_reproduces_saved_zeroshot", float(d.max()) < 0.05 and abs(a_rerun - a_saved) < 0.02,
   max_abs=float(d.max()), median_abs=float(np.median(d)), auc_saved=a_saved, auc_rerun=a_rerun,
   pearson=float(np.corrcoef(zp, rp)[0, 1]))
o = np.load(OLD, allow_pickle=True)
same_order = [str(x) for x in o["name"]] == names
Lr, Lo = z["llm"].astype(np.float64), o["llm"].astype(np.float64)
rel = float(np.abs(Lr - Lo).max() / np.abs(Lo).max())
cors = [float(np.corrcoef(Lr[:, l].ravel(), Lo[:, l].ravel())[0, 1]) for l in range(Lo.shape[1])]
ck("rerun_llm_states_reproduce_saved", same_order and Lr.shape == Lo.shape and min(cors) > 0.99,
   shape=list(Lr.shape), max_abs=float(np.abs(Lr - Lo).max()), max_abs_over_max=rel, min_layer_corr=min(cors))
res["states_ok"] = all(v["ok"] for v in res["checks"].values())
res["states_file"] = ST; res["states_sha256"] = sha(ST)
res["keys"] = {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}
if res["states_ok"]:
    dst = f"{R}/stage/cpu/in/kimi_kcl_encstates.npz"; shutil.copyfile(ST, dst)
    assert sha(dst) == res["states_sha256"]
    C = f"{R}/stage/cpu"; files = []
    for d0, _, fs in os.walk(C):
        for f in fs:
            p = os.path.relpath(os.path.join(d0, f), C)
            if p != "upload.sha256" and not f.startswith("._"): files.append(p)
    with open(f"{C}/upload.sha256", "w") as fh:
        for p in sorted(files): fh.write(f"{sha(os.path.join(C, p))}  {p}\n")
    res["cpu_stage_files"] = sorted(files)
json.dump(res, open(f"{R}/verify/p26_states_check_kimi_kcl.json", "w"), indent=1, default=str)
print("STATES OK" if res["states_ok"] else "STATES NOT OK")
