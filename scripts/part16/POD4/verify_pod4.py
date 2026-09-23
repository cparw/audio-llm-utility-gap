#!/usr/bin/env python3
"""Independent recomputation of every POD4 AUC.

The production path used the tie-averaged RANK formula. This verifier instead
counts concordant pairs directly (explicit Mann-Whitney U: every positive vs
every negative, ties scored 0.5). Different algorithm, same definition, so
agreement to 4 decimals is a real check rather than the same code run twice.
Bootstrap CIs are re-drawn with the SAME seed (default_rng(0)) and the same
speaker-resampling rule.
"""
import csv, json, sys, os
import numpy as np

D = "<local data dir>/Desktop/release/edaic_rerun/part16/POD4"

def auc_pairwise(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    # chunk to keep memory sane
    tot = 0.0
    for i in range(0, len(pos), 512):
        blk = pos[i:i + 512][:, None]
        tot += (blk > neg[None, :]).sum() + 0.5 * (blk == neg[None, :]).sum()
    return tot / (len(pos) * len(neg))

def boot(spk, y, s, nb=2000, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(spk)
    idx = {p: np.where(spk == p)[0] for p in u}; v = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        if len(np.unique(y[ii])) < 2: continue
        v.append(auc_pairwise(y[ii], s[ii]))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

def load(f, col, key="clip"):
    rows = list(csv.DictReader(open(f)))
    return (np.array([r["speaker"] for r in rows]),
            np.array([int(r["label"]) for r in rows]),
            np.array([float(r[col]) for r in rows]),
            np.array([r[key] for r in rows]))

CASES = [
 ("POD4_4b_pub_zs",  f"{D}/omni_pitt_zeroshot_scores.csv", "p_yes",   0.6578, None),
 ("POD4_4b_pub_enc", f"{D}/omni_pitt_enc_nested_oof.csv",  "p_probe", 0.7969, None),
 ("POD4_4b_A_zs",  f"{D}/p16_pitt_orig_zeroshot_scores.csv",     "p_yes",   0.6497, None),
 ("POD4_4b_A_enc", f"{D}/p16_pitt_orig_enc_nested_oof.csv",      "p_probe", 0.7949, None),
 ("POD4_4b_B_zs",  f"{D}/p16_pitt_paronly_zeroshot_scores.csv",  "p_yes",   0.7230, None),
 ("POD4_4b_B_enc", f"{D}/p16_pitt_paronly_enc_nested_oof.csv",   "p_probe", 0.7497, None),
 ("POD4_4b_C_zs",  f"{D}/p16_pitt_durmatch_zeroshot_scores.csv", "p_yes",   0.6145, None),
 ("POD4_4b_C_enc", f"{D}/p16_pitt_durmatch_enc_nested_oof.csv",  "p_probe", 0.7980, None),
]
# arm restriction
sets = {}
for r in csv.DictReader(open(f"{D}/mf_orig.csv")):
    sets[os.path.basename(r["path"])] = r["set"]

out = []
print(f"{'id':22s} {'reported':>9s} {'verify':>9s} {'match':>6s}   CI reported vs verify")
for cid, f, col, rep, _ in CASES:
    if not os.path.exists(f):
        print(f"{cid:22s} FILE MISSING {f}"); continue
    spk, y, s, clip = load(f, col)
    a = auc_pairwise(y, s)
    lo, hi, nu = boot(spk, y, s)
    ok = abs(round(a, 4) - rep) < 1e-9
    print(f"{cid:22s} {rep:9.4f} {a:9.4f} {'OK' if ok else 'DIFF':>6s}   [{lo:.4f}, {hi:.4f}] draws={nu}")
    out.append(dict(id=cid, reported=rep, verify=round(float(a), 4), match=bool(ok),
                    lo=round(lo, 4), hi=round(hi, 4), usable_draws=nu, file=f))
# arm splits
for cid, f, col, rep, arm in [
  ("POD4_4b_B_zs_conf",  f"{D}/p16_pitt_paronly_zeroshot_scores.csv", "p_yes",   0.5408, "conflict"),
  ("POD4_4b_B_enc_conf", f"{D}/p16_pitt_paronly_enc_nested_oof.csv",  "p_probe", 0.6295, "conflict"),
  ("POD4_4b_B_zs_agr",   f"{D}/p16_pitt_paronly_zeroshot_scores.csv", "p_yes",   0.8007, "agreement"),
  ("POD4_4b_B_enc_agr",  f"{D}/p16_pitt_paronly_enc_nested_oof.csv",  "p_probe", 0.8037, "agreement"),
  ("POD4_4b_A_zs_conf",  f"{D}/p16_pitt_orig_zeroshot_scores.csv",    "p_yes",   0.5298, "conflict"),
  ("POD4_4b_A_enc_conf", f"{D}/p16_pitt_orig_enc_nested_oof.csv",     "p_probe", 0.6161, "conflict"),
  ("POD4_4b_A_zs_agr",   f"{D}/p16_pitt_orig_zeroshot_scores.csv",    "p_yes",   0.6952, "agreement"),
  ("POD4_4b_A_enc_agr",  f"{D}/p16_pitt_orig_enc_nested_oof.csv",     "p_probe", 0.8674, "agreement"),
]:
    spk, y, s, clip = load(f, col)
    m = np.array([sets.get(c) == arm for c in clip])
    a = auc_pairwise(y[m], s[m]); lo, hi, nu = boot(spk[m], y[m], s[m])
    ok = abs(round(a, 4) - rep) < 1e-9
    print(f"{cid:22s} {rep:9.4f} {a:9.4f} {'OK' if ok else 'DIFF':>6s}   [{lo:.4f}, {hi:.4f}] n={m.sum()}")
    out.append(dict(id=cid, reported=rep, verify=round(float(a), 4), match=bool(ok),
                    lo=round(lo, 4), hi=round(hi, 4), usable_draws=nu, n=int(m.sum()), file=f))

# paired deltas, redrawn with the same seed
def paired(fa, fb, col, rep, cid):
    sa, ya, ssa, ca = load(fa, col); sb, yb, ssb, cb = load(fb, col)
    da = dict(zip(ca, zip(sa, ya, ssa))); db = dict(zip(cb, zip(sb, yb, ssb)))
    keys = sorted(set(da) & set(db))
    spk = np.array([da[k][0] for k in keys]); y = np.array([da[k][1] for k in keys])
    s1 = np.array([da[k][2] for k in keys]); s2 = np.array([db[k][2] for k in keys])
    d = auc_pairwise(y, s2) - auc_pairwise(y, s1)
    rng = np.random.default_rng(0); u = np.unique(spk)
    idx = {p: np.where(spk == p)[0] for p in u}; v = []
    for _ in range(2000):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        if len(np.unique(y[ii])) < 2: continue
        v.append(auc_pairwise(y[ii], s2[ii]) - auc_pairwise(y[ii], s1[ii]))
    v = np.array(v); lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
    ok = abs(round(d, 4) - rep) < 1e-9
    print(f"{cid:22s} {rep:+9.4f} {d:+9.4f} {'OK' if ok else 'DIFF':>6s}   [{lo:+.4f}, {hi:+.4f}] draws={len(v)}")
    out.append(dict(id=cid, reported=rep, verify=round(float(d), 4), match=bool(ok),
                    lo=round(float(lo), 4), hi=round(float(hi), 4), usable_draws=len(v)))

paired(f"{D}/p16_pitt_orig_zeroshot_scores.csv",     f"{D}/p16_pitt_paronly_zeroshot_scores.csv", "p_yes",   0.0732, "POD4_4b_pd_zs_AB")
paired(f"{D}/p16_pitt_durmatch_zeroshot_scores.csv", f"{D}/p16_pitt_paronly_zeroshot_scores.csv", "p_yes",   0.1084, "POD4_4b_pd_zs_CB")
paired(f"{D}/p16_pitt_orig_enc_nested_oof.csv",      f"{D}/p16_pitt_paronly_enc_nested_oof.csv",  "p_probe", -0.0452, "POD4_4b_pd_enc_AB")
paired(f"{D}/p16_pitt_durmatch_enc_nested_oof.csv",  f"{D}/p16_pitt_paronly_enc_nested_oof.csv",  "p_probe", -0.0483, "POD4_4b_pd_enc_CB")
paired(f"{D}/p16_pitt_orig_zeroshot_scores.csv",     f"{D}/p16_pitt_durmatch_zeroshot_scores.csv","p_yes",   -0.0352, "POD4_4b_pd_zs_AC")
paired(f"{D}/p16_pitt_orig_enc_nested_oof.csv",      f"{D}/p16_pitt_durmatch_enc_nested_oof.csv", "p_probe", 0.0032, "POD4_4b_pd_enc_AC")

bad = [o for o in out if not o["match"]]
print(f"\nVERIFY: {len(out)-len(bad)}/{len(out)} agree to 4 decimals; {len(bad)} disagree")
for b in bad: print("  DISAGREE", b["id"], b["reported"], b["verify"])
json.dump(out, open(f"{D}/VERIFY_POD4.json", "w"), indent=1)
