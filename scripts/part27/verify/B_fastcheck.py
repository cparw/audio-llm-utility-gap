# Independent fast check of PART 27 B (E-DAIC 400 Hz low-pass, Qwen2.5-Omni, 275 whole-interview clips).
import csv, re, numpy as np
R = "<local data dir>/release_from_mac/scores"
MP = f"{R}/part27/B/mainpull"; ST = f"{R}/part27/B/stage"
P23 = f"{R}/part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv"
OUT = f"{R}/part27/verify/B_fastcheck.tsv"
K = ["enc", "proj", "llm", "ans"]

def auc(y, s):  # hand-written Mann-Whitney AUC, ties = 0.5
    p = s[y == 1]; n = s[y == 0]
    return ((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n))

# ---- unfiltered reference, pid-sorted
ref = list(csv.DictReader(open(P23)))
ref.sort(key=lambda r: int(r["pid"]))
pids = [r["pid"] for r in ref]
assert len(pids) == 275 and len(set(pids)) == 275
assert all(r["speaker"] == r["pid"] for r in ref)
y = np.array([int(r["label"]) for r in ref])
f = lambda v: float(re.sub(r"^np\.float64\((.*)\)$", r"\1", v))
zs_unf = np.array([f(r["p_yes_whole"]) for r in ref])
unf = {k: [np.array([float(r[f"oof_{k}_r{i}"]) for r in ref]) for i in range(5)] for k in K}
fold_ref = [np.array([int(r[f"fold_rs{i}"]) for r in ref]) for i in range(5)]

# ---- label cross-check vs pod ref file
lab2 = {r["pid"]: int(r["label"]) for r in csv.DictReader(open(f"{ST}/p23a/ref/edaic_lifted_perclip.csv"))}
assert all(lab2[p] == yy for p, yy in zip(pids, y))

# ---- low-pass zero-shot, paired by pid
zl = {r["pid"]: float(r["p_yes"]) for r in csv.DictReader(open(f"{MP}/A_extract_perclip_lp400.csv"))}
assert set(zl) == set(pids) and len(zl) == 275
zs_lp = np.array([zl[p] for p in pids])

# ---- probe clip order: p23a_probe.py uses windows_full.csv row order
win = [r["pid"] for r in csv.DictReader(open(f"{ST}/edaicfull/windows_full.csv"))]
order_is_sorted = win == pids
print("windows_full order == pid-sorted order:", order_is_sorted)
perm = [win.index(p) for p in pids]  # maps sorted position -> npz position
lp = {k: [] for k in K}
fold_ok = True; npz_auc_ok = True
for k in K:
    for i in range(5):
        z = np.load(f"{MP}/probe_lp400/{k}_rep{i}.npz")
        assert z["oof"].shape == (275,)
        o = z["oof"][perm]; lp[k].append(o)
        fold_ok &= bool(np.array_equal(z["fold"][perm], fold_ref[i]))
        npz_auc_ok &= abs(auc(y, o) - float(z["auc"])) < 1e-12
print("npz fold arrays match PART 23 fold_rs columns after pairing:", fold_ok)
print("my AUC equals stored npz auc for all 20 repeats:", npz_auc_ok)
fr0 = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(f"{ST}/p23a/folds/edaic_full_groupkfold5_shuffle_rs0.csv"))}
print("fold file rs0 matches PART 23 fold_rs0:", all(fr0[p + ".wav"] == fo for p, fo in zip(pids, fold_ref[0])))

# ---- bootstrap
rng = np.random.default_rng(0)
B = 2000
IDX = [rng.choice(275, 275, replace=True) for _ in range(B)]
def m5(lst, ix=None):
    return np.mean([auc(y if ix is None else y[ix], s if ix is None else s[ix]) for s in lst])
stats = {}
def add(name, fn):
    pt = fn(None); bs = np.array([fn(ix) for ix in IDX])
    stats[name] = (pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5))
add("zs_lp", lambda ix: m5([zs_lp], ix))
add("zs_unf", lambda ix: m5([zs_unf], ix))
add("zs_lp_minus_unf", lambda ix: m5([zs_lp], ix) - m5([zs_unf], ix))
for k in K:
    add(f"{k}_lp", lambda ix, k=k: m5(lp[k], ix))
    add(f"{k}_chg_vs_unf", lambda ix, k=k: m5(lp[k], ix) - m5(unf[k], ix))
    add(f"{k}_minus_lp_answer", lambda ix, k=k: m5(lp[k], ix) - m5([zs_lp], ix))

claims = {
 "zs_lp": (0.7604, 0.6934, 0.8234), "zs_unf": (0.8366,), "zs_lp_minus_unf": (-0.0762, -0.1416, -0.0144),
 "enc_lp": (0.6017, 0.5361, 0.6697), "enc_chg_vs_unf": (0.0036, -0.0616, 0.0670), "enc_minus_lp_answer": (-0.1587, -0.2410, -0.0690),
 "proj_lp": (0.6393, 0.5731, 0.7081), "proj_chg_vs_unf": (0.0527, -0.0206, 0.1216), "proj_minus_lp_answer": (-0.1211, -0.2054, -0.0340),
 "llm_lp": (0.6633, 0.6027, 0.7269), "llm_chg_vs_unf": (-0.0804, -0.1588, -0.0015), "llm_minus_lp_answer": (-0.0971, -0.1637, -0.0224),
 "ans_lp": (0.7177, 0.6516, 0.7802), "ans_chg_vs_unf": (-0.1182, -0.1802, -0.0586), "ans_minus_lp_answer": (-0.0427, -0.1001, 0.0163),
}
rows = []
for name, cl in claims.items():
    for suf, c, v in zip(["point", "ci_lo", "ci_hi"], cl, stats[name]):
        ok = f"{v:.4f}" == f"{c:.4f}"
        rows.append((f"{name}_{suf}", f"{c:.4f}", f"{v:.4f}", "yes" if ok else "no"))
rows.append(("clip_order_windows_full_eq_pid_sorted", "True", str(order_is_sorted), "yes" if order_is_sorted else "no"))
rows.append(("npz_fold_eq_p23_fold_after_pairing", "True", str(fold_ok), "yes" if fold_ok else "no"))
rows.append(("my_auc_eq_npz_stored_auc_20reps", "True", str(npz_auc_ok), "yes" if npz_auc_ok else "no"))
with open(OUT, "w") as fh:
    fh.write("id\tclaimed\trecomputed\tverified\n")
    for r in rows: fh.write("\t".join(r) + "\n")
for r in rows: print("\t".join(r))
print("all yes:", all(r[3] == "yes" for r in rows))
