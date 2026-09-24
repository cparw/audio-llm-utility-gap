"""Independent verifier, PART26 cell Qwen3-Omni-30B-A3B x NeuroVoz.
Written from scratch. Does not import or copy gap_cell.py, verify_cell.py or A_gaps.py.
Inputs (read only): part26 per-clip csv, raw pod OOF csv, zero-shot per-clip file, release fold files, staged states
(labels/speakers/names/p_yes only), saved draws npz.
AUC: own Mann-Whitney rank formula (average ranks for ties), cross-checked against sklearn.
Bootstrap: own loop, fresh numpy default_rng(0), unique speakers in np.unique order of the speaker strings,
idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of every drawn speaker; per draw
gap = mean of five per-repeat AUCs minus zero-shot AUC on the same rows; 2.5 / 97.5 percentiles (numpy default)."""
import csv, json, sys, hashlib, datetime, platform, numpy as np
from scipy.stats import rankdata

CELL = "scores/part26/Qwen3-Omni-30B-A3B_NeuroVoz"
PC = f"{CELL}/per_clip/Qwen3-Omni-30B-A3B_NeuroVoz_perclip.csv"
RAW = f"{CELL}/pull/p26-q3o-neurovoz/out/p26_q3o_neurovoz_enc_nested5_oof.csv"
PODJ = f"{CELL}/pull/p26-q3o-neurovoz/out/p26_q3o_neurovoz_enc_nested5.json"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_neurovoz_zeroshot_scores.csv"
FOLDS = "<local data dir>/release/folds"
DRAWS = f"{CELL}/draws/Qwen3-Omni-30B-A3B_NeuroVoz_draws.npz"
STATES = f"{CELL}/stage/q3o_neurovoz_encstates.npz"
OUT = sys.argv[1]

CLAIM = dict(probe="0.9372", answer="0.6393", gap="+0.2980", lo="0.2360", hi="0.3578",
             per_repeat=["0.9329", "0.9370", "0.9414", "0.9371", "0.9377"],
             layers={"seed0": [31, 29, 31, 31, 19], "seed1": [29, 31, 31, 28, 29], "seed2": [17, 26, 17, 14, 21],
                     "seed3": [30, 27, 26, 17, 21], "seed4": [21, 20, 29, 26, 25]})
checks = []
def chk(name, ok, detail=""):
    checks.append({"check": name, "pass": bool(ok), "detail": detail}); print(("PASS " if ok else "FAIL ") + name, detail, flush=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def auc(y, s):
    y = np.asarray(y); r = rankdata(s)  # average ranks
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

f4 = lambda v: f"{v:.4f}"
def rows(p): return list(csv.DictReader(open(p, newline="")))

# ---------- load per-clip file ----------
R = rows(PC); n = len(R)
clips = [r["clip"] for r in R]; spk = np.array([r["speaker"] for r in R]); y = np.array([int(r["label"]) for r in R])
P = np.array([[float(r[f"p_probe_seed{s}"]) for r in R] for s in range(5)])
FO = np.array([[int(r[f"fold_seed{s}"]) for r in R] for s in range(5)])
zs_pc = np.array([float(r["p_yes_zeroshot"]) for r in R])
chk("perclip n=1270, unique clips", n == 1270 and len(set(clips)) == n, f"n={n}")
chk("perclip 107 speakers, 621 positives", len(np.unique(spk)) == 107 and int(y.sum()) == 621,
    f"spk={len(np.unique(spk))} pos={int(y.sum())}")
chk("perclip no missing / nonfinite OOF scores", np.isfinite(P).all() and np.isfinite(zs_pc).all())
chk("labels constant within speaker", all(len(set(y[spk == s])) == 1 for s in np.unique(spk)))
chk("label matches clip prefix HC/PD", all((c.startswith("PD_") and l == 1) or (c.startswith("HC_") and l == 0) for c, l in zip(clips, y)),
    f"prefixes={sorted(set(c.split('_')[0] for c in clips))}")

# ---------- raw pod OOF equals per-clip ----------
W = rows(RAW)
chk("raw pod OOF same clip order", [r["clip"] for r in W] == clips)
chk("raw pod OOF speakers/labels equal", all(r["speaker"] == s and int(r["label"]) == l for r, s, l in zip(W, spk, y)))
dP = max(abs(float(r[f"p_seed{s}"]) - P[s][i]) for s in range(5) for i, r in enumerate(W))
chk("raw pod OOF scores equal per-clip scores exactly", dP == 0.0, f"max abs diff {dP}")
chk("raw pod OOF fold columns equal per-clip fold columns",
    all(int(r[f"fold_seed{s}"]) == FO[s][i] for s in range(5) for i, r in enumerate(W)))

# ---------- zero-shot file ----------
Z = {r["clip"]: r for r in rows(ZS)}
chk("zero-shot clip set equals per-clip clip set", set(Z) == set(clips) and len(Z) == n, f"zs n={len(Z)}")
chk("zero-shot speakers and labels line up", all(Z[c]["speaker"] == s and int(Z[c]["label"]) == l for c, s, l in zip(clips, spk, y)))
zs = np.array([float(Z[c]["p_yes"]) for c in clips])
dz = float(np.max(np.abs(zs - zs_pc)))
chk("zero-shot p_yes equals per-clip p_yes_zeroshot", dz == 0.0, f"max abs diff {dz}")
chk("zero-shot file is the one listed in part25 C_build_cells.csv (sha256)",
    sha(ZS) == "c5bc6852aad557adfe861b8ad8925ab53a0f94b7dccc2a16a9c92567b8594a14")

# ---------- states (labels / speakers / names / p_yes only) ----------
S = np.load(STATES, allow_pickle=False)
sn = S["name"].astype(str); ss = S["spk"].astype(str); sl = S["label"].astype(int); sp = S["p_yes"]
chk("states clip order, speakers, labels equal per-clip", list(sn) == clips and np.array_equal(ss, spk) and np.array_equal(sl, y))
chk("states p_yes equals zero-shot file", float(np.max(np.abs(sp - zs))) == 0.0)
chk("states have 32 encoder layers", S["enc"].shape[1] == 32, str(S["enc"].shape))

# ---------- fold files ----------
def gkf_stable(groups, k):
    """own greedy GroupKFold fill: groups sorted by clip count descending (stable argsort reversed), each to lightest fold"""
    u, inv = np.unique(groups, return_inverse=True); cnt = np.bincount(inv)
    order = np.argsort(cnt, kind="stable")[::-1]; load = np.zeros(k); fg = np.empty(len(u), int)
    for gi in order:
        j = int(np.argmin(load)); fg[gi] = j; load[j] += cnt[gi]
    return fg[inv]
fold_info = {}
for s in range(5):
    fp = f"{FOLDS}/neurovoz_groupkfold5_pod_seed{s}_UNVERIFIED.csv"
    F = {r["clip_id"]: r for r in rows(fp)}
    same_set = set(F) == set(clips) and len(F) == n
    spk_ok = same_set and all(F[c]["speaker_id"] == sp_ for c, sp_ in zip(clips, spk))
    ff = np.array([int(F[c]["fold"]) for c in clips]) if same_set else None
    eq = same_set and np.array_equal(ff, FO[s])
    grp = all(len(set(FO[s][spk == u])) == 1 for u in np.unique(spk))
    rng = np.random.default_rng(s); u = np.unique(spk); perm = {a: i for i, a in enumerate(rng.permutation(u))}
    rebuilt = gkf_stable(np.array([perm[a] for a in spk]), 5)
    reb = np.array_equal(rebuilt, FO[s])
    sizes = np.bincount(FO[s]).tolist()
    both_classes = all(len(set(y[FO[s] == k])) == 2 for k in range(5))
    chk(f"seed{s}: fold file clips+speakers line up", spk_ok)
    chk(f"seed{s}: used folds equal saved fold file", eq)
    chk(f"seed{s}: every speaker in one fold only", grp)
    chk(f"seed{s}: folds rebuilt from default_rng({s}) renumbering + stable greedy fill equal the used folds", reb)
    fold_info[f"seed{s}"] = {"file": fp, "sha256": sha(fp), "sizes": sizes, "both_classes_in_each_test_fold": both_classes}
distinct = len({tuple(FO[s]) for s in range(5)}) == 5
chk("the five repeat splits are all different", distinct)
fs = {r["clip_id"]: int(r["fold"]) for r in rows(f"{FOLDS}/neurovoz_groupkfold5_pod.csv")}
chk("single-split column equals neurovoz_groupkfold5_pod.csv", all(fs[c] == int(r["fold_single"]) for c, r in zip(clips, R)))

# ---------- layers claimed vs pod json ----------
J = json.load(open(PODJ))
chk("claimed layers equal pod json layers", all(J["tags"][t]["layers"] == CLAIM["layers"][t] for t in CLAIM["layers"]))
chk("layer in range 0..31 and chosen per outer fold (5 per seed)", all(len(v) == 5 and all(0 <= l < 32 for l in v) for v in CLAIM["layers"].values()))
# the chosen layer must be the argmax of the saved inner scores of that outer fold
am = all(int(np.argmax(J["tags"][t]["inner"][k])) == CLAIM["layers"][t][k] for t in CLAIM["layers"] for k in range(5))
chk("each chosen layer is argmax of its saved inner-fold AUC vector", am)

# ---------- AUCs ----------
per = [auc(y, P[s]) for s in range(5)]
from sklearn.metrics import roc_auc_score
per_sk = [roc_auc_score(y, P[s]) for s in range(5)]
m5 = float(np.mean(per)); za = auc(y, zs); gap = m5 - za
chk("own rank AUC equals sklearn AUC (1e-12)", max(abs(a - b) for a, b in zip(per, per_sk)) < 1e-12 and abs(za - roc_auc_score(y, zs)) < 1e-12)
chk("per-repeat AUCs match claim at 4 dp", [f4(a) for a in per] == CLAIM["per_repeat"], str([f4(a) for a in per]))
chk("per-repeat AUCs match pod json", max(abs(a - b) for a, b in zip(per, J["per_repeat_auc"])) < 1e-12)
chk("probe mean of five matches claim at 4 dp", f4(m5) == CLAIM["probe"], f4(m5))
chk("answer AUC matches claim at 4 dp", f4(za) == CLAIM["answer"], f4(za))
chk("gap matches claim at 4 dp", f"{gap:+.4f}" == CLAIM["gap"], f"{gap:+.6f}")

# ---------- own bootstrap ----------
def boot(order_key=None):
    u = np.unique(spk)
    if order_key is not None: u = np.array(sorted(u, key=order_key))
    members = [np.flatnonzero(spk == a) for a in u]; k = len(u)
    rng = np.random.default_rng(0)
    D = np.empty(2000); M = np.empty(2000); Zb = np.empty(2000); PR = np.empty((2000, 5)); IDX = np.empty((2000, k), int)
    for b in range(2000):
        idx = rng.choice(k, size=k, replace=True); IDX[b] = idx
        ii = np.concatenate([members[i] for i in idx]); yy = y[ii]
        assert 0 < yy.sum() < len(yy)
        PR[b] = [auc(yy, P[s][ii]) for s in range(5)]; Zb[b] = auc(yy, zs[ii]); M[b] = PR[b].mean(); D[b] = M[b] - Zb[b]
    return dict(u=u, D=D, M=M, Z=Zb, PR=PR, IDX=IDX, lo=float(np.percentile(D, 2.5)), hi=float(np.percentile(D, 97.5)))
B = boot()
chk("bootstrap lo matches claim at 4 dp", f4(B["lo"]) == CLAIM["lo"], f"{B['lo']:.6f}")
chk("bootstrap hi matches claim at 4 dp", f4(B["hi"]) == CLAIM["hi"], f"{B['hi']:.6f}")
chk("interval above zero", B["lo"] > 0)
N = np.load(DRAWS, allow_pickle=False)
chk("saved draws: speaker order equals mine", np.array_equal(N["speakers"].astype(str), B["u"]))
chk("saved draws: speaker indices equal mine (all 2000 draws)", np.array_equal(N["speaker_idx"], B["IDX"]))
dd = float(np.max(np.abs(N["diff"] - B["D"]))); dm = float(np.max(np.abs(N["mean5"] - B["M"])))
dzb = float(np.max(np.abs(N["zeroshot"] - B["Z"]))); dpr = float(np.max(np.abs(N["per_repeat"] - B["PR"])))
chk("saved draws: gap, mean5, zero-shot, per-repeat equal mine (1e-12)", max(dd, dm, dzb, dpr) < 1e-12,
    f"diff {dd:.2e} mean5 {dm:.2e} zs {dzb:.2e} per_repeat {dpr:.2e}")
Bn = boot(order_key=int)   # sensitivity only: numeric speaker order
out = {"cell": "Qwen3-Omni-30B-A3B_NeuroVoz", "verifier": "independent, own code", "date_utc": datetime.datetime.utcnow().isoformat() + "Z",
       "n": n, "n_speakers": int(len(np.unique(spk))), "n_pos": int(y.sum()),
       "per_repeat_auc": per, "mean5": m5, "answer_auc": za, "gap": gap, "gap_ci": [B["lo"], B["hi"]],
       "mean5_ci": [float(np.percentile(B["M"], 2.5)), float(np.percentile(B["M"], 97.5))],
       "answer_ci": [float(np.percentile(B["Z"], 2.5)), float(np.percentile(B["Z"], 97.5))],
       "sensitivity_numeric_speaker_order_gap_ci": [Bn["lo"], Bn["hi"]],
       "folds": fold_info, "claim": CLAIM,
       "n_checks": len(checks), "failed": [c["check"] for c in checks if not c["pass"]], "checks": checks,
       "inputs_sha256": {k: sha(v) for k, v in {"perclip": PC, "raw_oof": RAW, "zeroshot": ZS, "draws": DRAWS, "pod_json": PODJ}.items()},
       "versions": {"python": platform.python_version(), "numpy": np.__version__}}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("per_repeat_auc", "mean5", "answer_auc", "gap", "gap_ci", "sensitivity_numeric_speaker_order_gap_ci", "n_checks", "failed")}, indent=1))
