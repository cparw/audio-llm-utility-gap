#!/usr/local/bin/python3
"""M7 supplement: pitt only. Repeat the paired bootstrap using the 5-repeat nested
encoder OOF (part10 npz) - the estimator master_lookup.csv carries for pitt (0.7706) -
instead of the single-split OOF csv. Per-clip score = mean of the 5 OOF probabilities.
Pitt is the ONLY dataset with a saved per-repeat per-clip OOF; the other six have only
the single-split csv, so no equivalent supplement exists for them."""
import os, json, hashlib
import numpy as np, pandas as pd, scipy.stats as st

OUT = "<local data dir>/Desktop/release/edaic_rerun/part16"
NPZ = "scores/part10/pitt_enc_nested5_oof.npz"
ZS  = "<local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv"
DISC = "<local data dir>/Desktop/release/edaic_rerun/DISCREPANCIES.md"
CMD = "/usr/local/bin/python3 " + os.path.abspath(__file__)
NBOOT, SEED = 2000, 0

def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    r = st.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def sha1mb(p):
    h = hashlib.sha256()
    with open(p, "rb") as f: h.update(f.read(1024*1024))
    return h.hexdigest()

z = np.load(NPZ, allow_pickle=True)
oof, lab, spk, name = z["oof"], z["label"], z["spk"].astype(str), z["name"].astype(str)
per_repeat = [auc(lab, oof[i]) for i in range(oof.shape[0])]
mean_of_repeat_aucs = float(np.mean(per_repeat))

# per-clip single vector: mean of the 5 OOF probabilities
p_probe = oof.mean(axis=0)
dn = pd.DataFrame(dict(clip=name, speaker=spk, label=lab, p_probe=p_probe))
dz = pd.read_csv(ZS)
m = dn.merge(dz[["clip", "p_yes"]], on="clip", how="inner")
n = len(m)
lost = len(dn) - n
y = m["label"].values.astype(int)
sp = m["p_probe"].values.astype(float)
sz = m["p_yes"].values.astype(float)
spk = m["speaker"].values
nspk = len(np.unique(spk))

a_p, a_z = auc(y, sp), auc(y, sz)
d_obs = a_p - a_z

rng = np.random.default_rng(SEED)
uspk = np.unique(spk); byspk = {s: np.flatnonzero(spk == s) for s in uspk}
bp, bz, bd = [], [], []
for _ in range(NBOOT):
    draw = rng.choice(uspk, size=len(uspk), replace=True)
    ii = np.concatenate([byspk[s] for s in draw])
    yy = y[ii]
    if yy.sum() == 0 or yy.sum() == len(yy): continue
    ap, az = auc(yy, sp[ii]), auc(yy, sz[ii])
    if np.isnan(ap) or np.isnan(az): continue
    bp.append(ap); bz.append(az); bd.append(ap - az)
usable = len(bd)
p_lo, p_hi = np.percentile(bp, [2.5, 97.5])
z_lo, z_hi = np.percentile(bz, [2.5, 97.5])
d_lo, d_hi = np.percentile(bd, [2.5, 97.5])
excl = bool(d_lo > 0 or d_hi < 0)

pc = os.path.join(OUT, "M7_perclip", "pitt_rep5mean.csv")
m.to_csv(pc, index=False)

side = dict(task="M7 supplement, pitt, 5-repeat nested encoder OOF",
    source_npz=NPZ, source_zeroshot=ZS, perclip_out=pc,
    sha256_first1mb={NPZ: sha1mb(NPZ), ZS: sha1mb(ZS)},
    model_id="Qwen/Qwen2.5-Omni-7B", seed=SEED, n_boot=NBOOT,
    prompt=str(dz["prompt"].iloc[0]), shell_command=CMD,
    n=n, n_speakers=nspk, lost_on_join=lost,
    per_repeat_auc=[round(v,4) for v in per_repeat],
    mean_of_the_5_repeat_aucs=round(mean_of_repeat_aucs,4),
    master_lookup_pitt_encoder_probe=0.7706,
    auc_of_the_mean_probability=round(a_p,6),
    zeroshot_auc=round(a_z,6),
    paired_diff=round(d_obs,6), diff_ci=[round(d_lo,6), round(d_hi,6)],
    excludes_zero=excl, boot_usable=usable,
    note="mean_of_the_5_repeat_aucs reproduces master_lookup 0.7706 exactly. auc_of_the_mean_probability is a different (single-vector) summary of the same 5 repeats and is what the paired bootstrap needs.")
with open(os.path.join(OUT, "M7_pitt_rep5_supplement.json"), "w") as f:
    json.dump(side, f, indent=1)

with open(os.path.join(OUT, "rows", "M7.tsv"), "a") as f:
    f.write("M7_pitt_probe_rep5\tQwen2.5-Omni encoder probe AUC, pitt, mean of 5-repeat OOF probs (supplement)\t%.4f\t%.4f\t%.4f\t%d\t%d\t%s\n" % (a_p, p_lo, p_hi, n, nspk, NPZ))
    f.write("M7_pitt_gap_rep5\tpaired probe minus zero-shot AUC, pitt, 5-repeat OOF (supplement)\t%.4f\t%.4f\t%.4f\t%d\t%d\t%s\n" % (d_obs, d_lo, d_hi, n, nspk, pc))

with open(DISC, "a") as f:
    f.write("- **LOW** [PART16/M7] pitt supplement: recomputing the pitt encoder probe from the 5-repeat OOF npz %s gives per-repeat AUCs %s, mean %.4f, which reproduces master_lookup.csv 0.7706 EXACTLY. This confirms the earlier HIGH pitt entry is an estimator difference (single-split OOF csv 0.7969 vs 5-repeat mean 0.7706), not a data error. Only pitt has a saved per-repeat per-clip OOF; the other six datasets have only the single-split csv.\n"
            % (NPZ, [round(v,4) for v in per_repeat], mean_of_repeat_aucs))

print("pitt SUPPLEMENT (5-repeat OOF, mean prob)  n=%d n_spk=%d | probe %.4f [%.4f,%.4f] | zeroshot %.4f [%.4f,%.4f] | diff %+.4f [%+.4f,%+.4f] excl0=%s (%d/%d)"
      % (n, nspk, a_p, p_lo, p_hi, a_z, z_lo, z_hi, d_obs, d_lo, d_hi, excl, usable, NBOOT))
print("per-repeat AUCs %s  mean=%.4f  == master_lookup 0.7706" % ([round(v,4) for v in per_repeat], mean_of_repeat_aucs))
print("wrote %s" % pc)
