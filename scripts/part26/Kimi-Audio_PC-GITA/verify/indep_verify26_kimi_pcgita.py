"""Independent verifier, PART 26, Kimi-Audio on PC-GITA.
Written from scratch. Reads only. Recomputes per-repeat AUCs, mean of five,
zero-shot answer AUC, gap, and a 2000-draw speaker bootstrap from the per-clip
OOF file and the canonical zero-shot file. Checks folds against saved fold files.
"""
import csv, json, hashlib, sys
import numpy as np

D = "scores/part26/Kimi-Audio_PC-GITA"
PERCLIP = f"{D}/Kimi-Audio_PC-GITA_perclip.csv"
PODOOF = f"{D}/pull/p26-kimi-pcgita-cpu/out/p26_kimi_pcgita_enc_nested5_oof.csv"
DRAWS = f"{D}/draws/Kimi-Audio_PC-GITA_draws.npz"
ZS = "<local data dir>/release/overnight2/part3/kimi_pcgita_zeroshot_scores.csv"
FOLDS = "<local data dir>/release/folds/pcgita_groupkfold5_pod_seed{}.csv"
FOLD_SINGLE = "<local data dir>/release/folds/pcgita_groupkfold5_pod.csv"

CLAIM = dict(probe=0.8768, per_repeat=[0.9028, 0.8478, 0.8616, 0.8797, 0.8918],
             gap=0.2621, lo=0.1810, hi=0.3364)

out = {}
fails = []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def my_auc(y, s):
    """Mann-Whitney AUC with average ranks for ties. Own code."""
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def check(name, ok, detail=""):
    out.setdefault("checks", {})[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)


# ---------- load ----------
pc = read(PERCLIP)
zs = read(ZS)
pod = read(PODOOF)
clips = [r["clip"] for r in pc]
check("perclip_n1100_unique", len(pc) == 1100 and len(set(clips)) == 1100, f"n={len(pc)} unique={len(set(clips))}")

zmap = {r["clip"]: r for r in zs}
check("zs_n1100_unique", len(zs) == 1100 and len(zmap) == 1100, f"n={len(zs)}")
check("zs_same_clip_set", set(zmap) == set(clips))
out["zs_sha256"] = sha(ZS)

# labels / speakers line up with zero-shot file
lab_bad = sum(int(r["label"]) != int(zmap[r["clip"]]["label"]) for r in pc)
spk_bad = sum(r["speaker"] != zmap[r["clip"]]["speaker"] for r in pc)
pyes_diff = max(abs(float(r["p_yes_zeroshot"]) - float(zmap[r["clip"]]["p_yes"])) for r in pc)
check("zs_label_match", lab_bad == 0, f"mismatches={lab_bad}")
check("zs_speaker_match", spk_bad == 0, f"mismatches={spk_bad}")
check("zs_pyes_copied_exact", pyes_diff == 0.0, f"max_abs={pyes_diff}")
# speaker is the clip prefix
pref_bad = sum(not r["clip"].startswith(r["speaker"]) for r in pc)
check("speaker_is_clip_prefix", pref_bad == 0, f"bad={pref_bad}")

# pod OOF equals perclip probe columns
podmap = {r["clip"]: r for r in pod}
check("pod_oof_same_clips", set(podmap) == set(clips) and len(pod) == 1100)
podd = 0.0; podfold = 0; podlab = 0
for r in pc:
    q = podmap[r["clip"]]
    for k in range(5):
        podd = max(podd, abs(float(r[f"p_probe_seed{k}"]) - float(q[f"p_seed{k}"])))
        podfold += int(r[f"fold_seed{k}"]) != int(q[f"fold_seed{k}"])
    podd = max(podd, abs(float(r["p_probe_single_split"]) - float(q["p_single"])))
    podfold += int(r["fold_single_split"]) != int(q["fold_single"])
    podlab += (int(r["label"]) != int(q["label"])) + (r["speaker"] != q["speaker"])
check("pod_oof_equals_perclip", podd == 0.0 and podfold == 0 and podlab == 0,
      f"max_pred_diff={podd} fold_mism={podfold} lab_spk_mism={podlab}")

# ---------- folds vs saved fold files ----------
fold_info = {}
for k in list(range(5)) + ["single"]:
    fp = FOLD_SINGLE if k == "single" else FOLDS.format(k)
    fr = read(fp)
    fm = {r["clip_id"]: r for r in fr}
    col = "fold_single_split" if k == "single" else f"fold_seed{k}"
    same_set = set(fm) == set(clips) and len(fr) == 1100
    fmis = sum(int(r[col]) != int(fm[r["clip"]]["fold"]) for r in pc)
    smis = sum(r["speaker"] != fm[r["clip"]]["speaker_id"] for r in pc)
    # grouped: each speaker in exactly one fold
    spk_f = {}
    for r in fr:
        spk_f.setdefault(r["speaker_id"], set()).add(r["fold"])
    grouped = all(len(v) == 1 for v in spk_f.values())
    nfold = len({r["fold"] for r in fr})
    fold_info[str(k)] = dict(file=fp, sha256=sha(fp), same_clip_set=same_set, fold_mismatch=fmis,
                             speaker_mismatch=smis, speaker_grouped=grouped, n_folds=nfold)
    check(f"folds_{k}", same_set and fmis == 0 and smis == 0 and grouped and nfold == 5,
          json.dumps(fold_info[str(k)]))
out["folds"] = fold_info
# repeats differ from each other (not the same split five times)
fcols = np.array([[int(r[f"fold_seed{k}"]) for k in range(5)] for r in pc])
out["n_distinct_fold_columns"] = len({tuple(fcols[:, k]) for k in range(5)})

# ---------- point estimates ----------
y = np.array([int(r["label"]) for r in pc])
P = np.array([[float(r[f"p_probe_seed{k}"]) for k in range(5)] for r in pc])
zsc = np.array([float(zmap[r["clip"]]["p_yes"]) for r in pc])
psingle = np.array([float(r["p_probe_single_split"]) for r in pc])
spk = np.array([r["speaker"] for r in pc])
check("finite_scores", np.isfinite(P).all() and np.isfinite(zsc).all())

per = [my_auc(y, P[:, k]) for k in range(5)]
mean5 = float(np.mean(per))
zauc = my_auc(y, zsc)
gap = mean5 - zauc
single = my_auc(y, psingle)
from sklearn.metrics import roc_auc_score
sk_per = [roc_auc_score(y, P[:, k]) for k in range(5)]
sk_z = roc_auc_score(y, zsc)
check("own_auc_equals_sklearn", max(abs(a - b) for a, b in zip(per + [zauc], sk_per + [sk_z])) < 1e-12)
out["n"] = int(len(y)); out["n_pos"] = int(y.sum()); out["n_spk"] = int(len(np.unique(spk)))
# each speaker has a single label
lab_by_spk = {}
for s_, l_ in zip(spk, y):
    lab_by_spk.setdefault(s_, set()).add(int(l_))
check("speaker_single_label", all(len(v) == 1 for v in lab_by_spk.values()))

out["per_repeat"] = per
out["mean5"] = mean5
out["zeroshot_auc"] = zauc
out["gap"] = gap
out["single_split_auc"] = single

r4 = lambda v: round(float(v) + 0.0, 4)
check("claim_per_repeat_4dp", [r4(v) for v in per] == CLAIM["per_repeat"], str([r4(v) for v in per]))
check("claim_mean5_4dp", r4(mean5) == CLAIM["probe"], str(r4(mean5)))
check("claim_gap_4dp", r4(gap) == CLAIM["gap"], str(r4(gap)))
check("answer_auc_0.6147", r4(zauc) == 0.6147, str(r4(zauc)))
check("single_split_0.8685", r4(single) == 0.8685, str(r4(single)))

# ---------- speaker bootstrap ----------
uspk = np.unique(spk)
idx_by = {s_: np.where(spk == s_)[0] for s_ in uspk}
rng = np.random.default_rng(0)
B = 2000
diff = np.full(B, np.nan); m5 = np.full(B, np.nan); zb = np.full(B, np.nan)
prb = np.full((B, 5), np.nan)
sidx = np.zeros((B, len(uspk)), dtype=np.int64)
for b in range(B):
    pick = rng.choice(len(uspk), size=len(uspk), replace=True)
    sidx[b] = pick
    ii = np.concatenate([idx_by[uspk[j]] for j in pick])
    yb = y[ii]
    if yb.min() == yb.max():
        continue
    pr = [my_auc(yb, P[ii, k]) for k in range(5)]
    prb[b] = pr
    m5[b] = np.mean(pr)
    zb[b] = my_auc(yb, zsc[ii])
    diff[b] = m5[b] - zb[b]
ok = np.isfinite(diff)
lo, hi = np.percentile(diff[ok], [2.5, 97.5])
out["usable_draws"] = int(ok.sum())
out["gap_ci"] = [float(lo), float(hi)]
check("claim_lo_4dp", r4(lo) == CLAIM["lo"], str(r4(lo)))
check("claim_hi_4dp", r4(hi) == CLAIM["hi"], str(r4(hi)))
check("usable_2000", int(ok.sum()) == 2000)

# ---------- compare to saved draws ----------
z = np.load(DRAWS, allow_pickle=False)
out["draws_sha256"] = sha(DRAWS)
out["draws_keys"] = list(z.files)
sv_spk = z["speakers"]
check("draws_speakers_equal_np_unique", sv_spk.shape == uspk.shape and (sv_spk == uspk).all())
check("draws_speaker_idx_equal", z["speaker_idx"].shape == sidx.shape and (z["speaker_idx"].astype(np.int64) == sidx).all())
dd = float(np.nanmax(np.abs(z["diff"] - diff)))
dm = float(np.nanmax(np.abs(z["mean5"] - m5)))
dz = float(np.nanmax(np.abs(z["zeroshot"] - zb)))
dp = float(np.nanmax(np.abs(z["per_repeat"] - prb)))
out["saved_draws_maxabs"] = dict(diff=dd, mean5=dm, zeroshot=dz, per_repeat=dp)
check("saved_draws_match_1e-10", max(dd, dm, dz, dp) < 1e-10, json.dumps(out["saved_draws_maxabs"]))
slo, shi = np.percentile(z["diff"], [2.5, 97.5])
out["saved_draws_ci"] = [float(slo), float(shi)]
check("saved_draws_ci_4dp", r4(slo) == CLAIM["lo"] and r4(shi) == CLAIM["hi"])
for k in z.files:
    if z[k].ndim == 0:
        out.setdefault("draws_scalars", {})[k] = str(z[k])

# ---------- sidecar agreement ----------
sc = json.load(open(f"{D}/Kimi-Audio_PC-GITA_perclip.sidecar.json"))
check("sidecar_per_repeat_1e-12", max(abs(a - b) for a, b in zip(sc["probe_per_repeat"], per)) < 1e-12)
check("sidecar_gap_ci_1e-12", abs(sc["gap_point"] - gap) < 1e-12 and abs(sc["gap_ci"][0] - lo) < 1e-12 and abs(sc["gap_ci"][1] - hi) < 1e-12)
check("sidecar_zs_file", sc["zeroshot_file"] == ZS and sc["sha256"]["zeroshot"] == out["zs_sha256"])
check("sidecar_perclip_sha", sc["sha256"]["perclip_csv"] == sha(PERCLIP))
check("sidecar_draws_sha", sc["sha256"]["draws_npz"] == out["draws_sha256"])
for k in range(5):
    check(f"sidecar_fold_sha_seed{k}", sc["pod"]["fold_files"][f"seed{k}"]["fold_file_sha256"] == fold_info[str(k)]["sha256"])

out["fails"] = fails
out["verified"] = len(fails) == 0
print(json.dumps(out, indent=1, default=float))
