"""Independent verifier, PART26 cell Qwen2-Audio x NeuroVoz.
Written from scratch. Reads only: the per-clip OOF csv, the zero-shot per-clip csv, the saved release fold files,
the pod OOF csv, the staged states (name/spk/label only), the master repeats json, the Omni PART25 A per-clip file
(for its fold columns), and the claimed draws npz (only to compare against). Own rank AUC, own bootstrap."""
import csv, json, sys, hashlib, numpy as np

P = "scores/part26/Qwen2-Audio_NeuroVoz"
PERCLIP = P + "/per_clip/p26_q2a_neurovoz_perclip.csv"
DRAWS = P + "/draws/p26_q2a_neurovoz_draws.npz"
PODOOF = P + "/pull/p26-q2a-neurovoz/out/p26_q2a_neurovoz_enc_nested5_oof.csv"
PODJSON = P + "/pull/p26-q2a-neurovoz/out/p26_q2a_neurovoz_enc_nested5.json"
STATES = P + "/stage/q2a_neurovoz_encstates.npz"
ZS = "<local data dir>/paper1_local_runs/probe2/neurovoz_zeroshot_scores.csv"
FOLDDIR = "<local data dir>/release/folds"
MASTER = "<local data dir>/paper1_local_runs/omni_final/q2a_neurovoz_nested_repeats.json"
OMNI_A = "<local data dir>/release_from_mac/scores/part25/A/per_clip/A_neurovoz_perclip.csv"
CBUILD = "<local data dir>/release_from_mac/scores/part25/C/C_build_cells.csv"
CLAIM = dict(probe=0.9324, answer=0.5519, gap=0.3805, lo=0.3278, hi=0.4325,
             per_repeat=[0.9310, 0.9296, 0.9283, 0.9386, 0.9346],
             layers={"seed0": [24, 25, 18, 24, 20], "seed1": [20, 20, 30, 22, 25], "seed2": [17, 21, 25, 19, 30],
                     "seed3": [20, 25, 20, 17, 20], "seed4": [20, 30, 24, 21, 28]})
SEEDS = [f"seed{s}" for s in range(5)]
out = {"checks": {}, "values": {}}
def chk(name, ok, detail=None):
    out["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  {detail}"), flush=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def readcsv(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))

def my_auc(y, s):
    """Mann-Whitney AUC with average ranks for ties, own ranking code."""
    y = np.asarray(y); s = np.asarray(s, float); n = len(s)
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    r = np.empty(n, float); i = 0
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = int((y == 1).sum()); n0 = n - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def fast_auc(y, s):
    """Vectorised average-rank AUC (used inside the bootstrap), checked against my_auc below."""
    s = np.asarray(s, float); n = len(s)
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]; ends = np.r_[starts[1:], n]
    avg = (starts + ends - 1) / 2.0 + 1.0
    r = np.empty(n); r[o] = np.repeat(avg, ends - starts)
    n1 = int((y == 1).sum()); n0 = n - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

# ---------- load per-clip file ----------
pc = readcsv(PERCLIP)
clips = [r["clip"] for r in pc]; spk = np.array([r["speaker"] for r in pc]); y = np.array([int(r["label"]) for r in pc])
Pm = np.array([[float(r[f"p_probe_{t}"]) for t in SEEDS] for r in pc])
Fm = np.array([[int(r[f"fold_{t}"]) for t in SEEDS] for r in pc])
zs_pc = np.array([float(r["p_yes_zeroshot"]) for r in pc])
n = len(pc)
out["values"]["n"] = n; out["values"]["n_pos"] = int(y.sum()); out["values"]["n_speakers"] = int(len(set(spk)))
chk("perclip_rows_1270_unique_clips", n == 1270 and len(set(clips)) == n, f"n={n} unique={len(set(clips))}")
chk("no_nan_scores", np.isfinite(Pm).all() and np.isfinite(zs_pc).all() and ((Pm >= 0) & (Pm <= 1)).all())
chk("labels_binary_and_counts", set(y.tolist()) == {0, 1} and y.sum() == 621, f"n_pos={int(y.sum())}")
# label constant within speaker, and label matches clip prefix HC/PD
lab_by_spk = {}
for s_, l_ in zip(spk, y): lab_by_spk.setdefault(s_, set()).add(int(l_))
chk("label_constant_within_speaker", all(len(v) == 1 for v in lab_by_spk.values()))
pref = np.array([1 if c.startswith("PD") else 0 if c.startswith("HC") else -1 for c in clips])
chk("label_equals_clip_prefix_PD_HC", np.array_equal(pref, y), f"mismatch={int((pref != y).sum())}")
# speaker id equals last 4 digits of the file name
spk_from_name = np.array([str(int(c.rsplit("_", 1)[1].split(".")[0])) for c in clips])
chk("speaker_equals_clip_name_suffix", np.array_equal(spk_from_name, spk), f"mismatch={int((spk_from_name != spk).sum())}")

# ---------- states alignment (names / speakers / labels only) ----------
z = np.load(STATES, allow_pickle=True)
chk("perclip_order_equals_states_order",
    list(z["name"].astype(str)) == clips and np.array_equal(z["spk"].astype(str), spk) and np.array_equal(z["label"].astype(int), y))

# ---------- per-clip equals pod OOF (bit for bit) ----------
po = readcsv(PODOOF)
same = [r["clip"] for r in po] == clips and [r["speaker"] for r in po] == list(spk) and [int(r["label"]) for r in po] == list(y)
dmax = max(abs(float(po[i][f"p_{t}"]) - Pm[i, k]) for i in range(n) for k, t in enumerate(SEEDS))
fsame = all(int(po[i][f"fold_{t}"]) == Fm[i, k] for i in range(n) for k, t in enumerate(SEEDS))
chk("perclip_equals_pod_oof", same and dmax == 0.0 and fsame, f"max_abs={dmax}")
out["values"]["sha256_perclip"] = sha(PERCLIP); out["values"]["sha256_pod_oof"] = sha(PODOOF)

# ---------- zero-shot file ----------
zr = readcsv(ZS); zmap = {r["clip"]: r for r in zr}
chk("zeroshot_file_same_clip_set", len(zr) == n and set(zmap) == set(clips), f"zs_rows={len(zr)}")
chk("zeroshot_speaker_label_align", all(zmap[c]["speaker"] == s_ and int(zmap[c]["label"]) == l_ for c, s_, l_ in zip(clips, spk, y)))
zs = np.array([float(zmap[c]["p_yes"]) for c in clips])
chk("zeroshot_column_equals_file", np.max(np.abs(zs - zs_pc)) == 0.0, f"max_abs={np.max(np.abs(zs - zs_pc))}")
zsha = sha(ZS); out["values"]["sha256_zeroshot"] = zsha
cb = [r for r in readcsv(CBUILD) if r["model"] == "Qwen2-Audio" and r["dataset"] == "NeuroVoz"]
chk("zeroshot_path_and_sha_equal_C_build_cells", len(cb) == 1 and cb[0]["zs_file"] == ZS and cb[0]["zs_file_sha256"] == zsha)

# ---------- folds ----------
for k, t in enumerate(SEEDS):
    ff = f"{FOLDDIR}/neurovoz_groupkfold5_pod_{t}_UNVERIFIED.csv"
    fr = {r["clip_id"]: r for r in readcsv(ff)}
    ok = set(fr) == set(clips) and len(fr) == n
    ok = ok and all(fr[c]["speaker_id"] == s_ for c, s_ in zip(clips, spk))
    fsaved = np.array([int(fr[c]["fold"]) for c in clips])
    eq = np.array_equal(fsaved, Fm[:, k])
    # group integrity: each speaker in exactly one fold, 5 folds, both classes in each test fold
    integ = all(len(set(Fm[spk == s_, k])) == 1 for s_ in set(spk)) and set(Fm[:, k]) == {0, 1, 2, 3, 4}
    # own rebuild of the documented rule: rng(seed).permutation(unique spk) renumbering, greedy largest-first fill
    rng = np.random.default_rng(k); u = np.unique(spk); perm = rng.permutation(u)
    newid = {s_: i for i, s_ in enumerate(perm)}; g = np.array([newid[s_] for s_ in spk])
    cnt = np.bincount(g, minlength=len(u))
    order = sorted(range(len(u)), key=lambda gi: (-cnt[gi], -gi))  # desc count, ties: higher id first
    load = [0] * 5; fg = {}
    for gi in order:
        f = min(range(5), key=lambda j: (load[j], j)); fg[gi] = f; load[f] += cnt[gi]
    rebuilt = np.array([fg[gi] for gi in g])
    chk(f"folds_{t}_equal_release_fold_file", ok and eq and integ, f"{ff.rsplit('/',1)[1]} sha={sha(ff)[:12]} diff={int((fsaved != Fm[:, k]).sum())}")
    chk(f"folds_{t}_equal_own_rebuild_of_rule", np.array_equal(rebuilt, fsaved), f"diff={int((rebuilt != fsaved).sum())}")
# Omni Table 1 (PART25 A) fold columns
oa = {r["clip"]: r for r in readcsv(OMNI_A)}
chk("folds_equal_omni_partA_fold_columns", set(oa) == set(clips) and
    all(int(oa[c][f"fold_{t}"]) == Fm[i, k] for i, c in enumerate(clips) for k, t in enumerate(SEEDS)))
# layers claimed vs pod json
pj = json.load(open(PODJSON))
chk("layers_equal_pod_json", all(pj["tags"][t]["layers"] == CLAIM["layers"][t] for t in SEEDS),
    {t: pj["tags"][t]["layers"] for t in SEEDS})
chk("pod_json_versions_pinned", pj["versions"]["sklearn"] == "1.9.1" and pj["versions"]["numpy"] == "2.1.2" and
    pj["versions"]["scipy"] == "1.18.1" and "Linux" in pj["versions"]["platform"], pj["versions"])

# ---------- AUCs ----------
per = [my_auc(y, Pm[:, k]) for k in range(5)]; mean5 = float(np.mean(per)); ans = my_auc(y, zs)
out["values"].update(per_repeat=per, mean5=mean5, answer=ans, gap_point=mean5 - ans)
print("per_repeat", [f"{a:.6f}" for a in per], "mean5", f"{mean5:.6f}", "answer", f"{ans:.6f}", "gap", f"{mean5-ans:+.6f}")
chk("per_repeat_4dp_match_claim", [round(a, 4) for a in per] == CLAIM["per_repeat"], [round(a, 4) for a in per])
chk("mean5_4dp_match_claim", round(mean5, 4) == CLAIM["probe"], round(mean5, 4))
chk("answer_4dp_match_claim", round(ans, 4) == CLAIM["answer"], round(ans, 4))
chk("gap_4dp_match_claim", round(mean5 - ans, 4) == CLAIM["gap"], round(mean5 - ans, 4))
m = json.load(open(MASTER))["enc"]
chk("master_json_per_repeat_and_mean", m["per_repeat"] == [round(a, 4) for a in per] and m["mean"] == round(mean5, 4), m)
# cross-check my AUC against sklearn once
from sklearn.metrics import roc_auc_score
chk("own_auc_equals_sklearn", max(abs(my_auc(y, Pm[:, k]) - roc_auc_score(y, Pm[:, k])) for k in range(5)) < 1e-12 and
    abs(ans - roc_auc_score(y, zs)) < 1e-12)

# ---------- speaker bootstrap, own code ----------
rng = np.random.default_rng(0)
U = np.unique(spk)                                # string order, as states spk is <U3
rows_of = {s_: np.flatnonzero(spk == s_) for s_ in U}
D = np.empty(2000); M5 = np.empty(2000); A0 = np.empty(2000); PR = np.empty((2000, 5)); IDX = np.empty((2000, len(U)), int)
usable = 0
for b in range(2000):
    idx = rng.choice(len(U), size=len(U), replace=True); IDX[b] = idx
    rows = np.concatenate([rows_of[U[i]] for i in idx]); yb = y[rows]
    if yb.min() == yb.max():
        D[b] = M5[b] = A0[b] = np.nan; PR[b] = np.nan; continue
    usable += 1
    PR[b] = [fast_auc(yb, Pm[rows, k]) for k in range(5)]
    M5[b] = PR[b].mean(); A0[b] = fast_auc(yb, zs[rows]); D[b] = M5[b] - A0[b]
# spot check fast_auc vs my_auc on a few draws
rng2 = np.random.default_rng(0); sc = 0.0
for b in range(5):
    idx = rng2.choice(len(U), size=len(U), replace=True); rows = np.concatenate([rows_of[U[i]] for i in idx])
    sc = max(sc, abs(my_auc(y[rows], zs[rows]) - A0[b]), abs(my_auc(y[rows], Pm[rows, 2]) - PR[b, 2]))
chk("fast_auc_equals_slow_auc_on_draws", sc < 1e-12, sc)
Dg = D[np.isfinite(D)]
lo, hi = np.percentile(Dg, [2.5, 97.5])
out["values"].update(usable_draws=usable, gap_lo=float(lo), gap_hi=float(hi),
                     mean5_ci=[float(x) for x in np.percentile(M5[np.isfinite(M5)], [2.5, 97.5])],
                     answer_ci=[float(x) for x in np.percentile(A0[np.isfinite(A0)], [2.5, 97.5])],
                     frac_draws_gap_le_0=float((Dg <= 0).mean()))
print(f"bootstrap usable {usable}  gap [{lo:+.6f}, {hi:+.6f}]")
chk("usable_draws_2000", usable == 2000, usable)
chk("lo_4dp_match_claim", round(lo, 4) == CLAIM["lo"], round(lo, 4))
chk("hi_4dp_match_claim", round(hi, 4) == CLAIM["hi"], round(hi, 4))
chk("interval_above_zero", lo > 0, lo)
# compare against the claimed npz, draw by draw
d = np.load(DRAWS, allow_pickle=True)
chk("npz_speakers_equal_my_order", list(d["speakers"].astype(str)) == list(U))
chk("npz_speaker_idx_equal_my_idx", np.array_equal(d["speaker_idx"], IDX))
dd = float(np.nanmax(np.abs(d["diff"] - D))); dp = float(np.nanmax(np.abs(d["per_repeat"] - PR)))
chk("npz_draws_equal_mine", dd < 1e-12 and dp < 1e-12 and np.nanmax(np.abs(d["zeroshot"] - A0)) < 1e-12,
    f"max_abs_diff_draw={dd:.2e} per_repeat={dp:.2e}")
chk("npz_point_diff_equal_mine", abs(float(d["point_diff"]) - (mean5 - ans)) < 1e-12, float(d["point_diff"]))
out["values"]["sha256_draws"] = sha(DRAWS)
# sensitivity (not a gate): integer speaker order
Ui = np.array(sorted(U, key=int)); rng = np.random.default_rng(0); Di = []
for b in range(2000):
    idx = rng.choice(len(Ui), size=len(Ui), replace=True); rows = np.concatenate([rows_of[Ui[i]] for i in idx]); yb = y[rows]
    Di.append(np.mean([fast_auc(yb, Pm[rows, k]) for k in range(5)]) - fast_auc(yb, zs[rows]))
out["values"]["sensitivity_int_speaker_order_ci"] = [float(x) for x in np.percentile(Di, [2.5, 97.5])]
print("sensitivity, integer speaker order:", [round(x, 4) for x in out["values"]["sensitivity_int_speaker_order_ci"]])

allpass = all(v["pass"] for v in out["checks"].values())
out["ALL_PASS"] = allpass
out["n_checks"] = len(out["checks"]); out["n_pass"] = sum(v["pass"] for v in out["checks"].values())
print("ALL_PASS", allpass, out["n_pass"], "/", out["n_checks"])
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
