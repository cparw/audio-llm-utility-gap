"""PART25 A retry sensitivity: ADReSS-2020 extracted under transformers 5.5.4 on an H100 80GB HBM3 (not the original H200).
153 of 156 zero-shot scores are bit-identical to the saved file, 3 are not. Same probe, same bootstrap as A_gaps_r2.py.
Writes A_sensitivity_r2.tsv, A_sensitivity_r2.sidecar.json, draws/A_sens_adress2020_h100hbm3_tf554_draws.npz."""
import csv, json, numpy as np, hashlib
from sklearn.metrics import roc_auc_score
R = "<local data dir>/release_from_mac/scores/part25/A/retry_tf554"; OF = "<local data dir>/release/omni_final"
pod = f"{R}/pull/p25Ar-adress2020"; oofp = f"{pod}/out/r2_adress2020_enc_nested5_oof.csv"
rows = list(csv.DictReader(open(oofp))); names = [r["clip"] for r in rows]
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
P = np.array([[float(r[f"p_seed{s}"]) for r in rows] for s in range(5)])
zm = {r["clip"]: float(r["p_yes"]) for r in csv.DictReader(open(f"{OF}/omni_adress2020_zeroshot_scores.csv"))}; zs = np.array([zm[c] for c in names])
u = np.unique(spk); ix = [np.where(spk == s)[0] for s in u]; rng = np.random.default_rng(0); d = np.full(2000, np.nan); m5b = d.copy(); zb = d.copy()
for b in range(2000):
    idx = rng.choice(len(u), size=len(u), replace=True); ii = np.concatenate([ix[i] for i in idx]); yy = y[ii]
    if yy.min() == yy.max(): continue
    m5b[b] = np.mean([roc_auc_score(yy, P[r][ii]) for r in range(5)]); zb[b] = roc_auc_score(yy, zs[ii]); d[b] = m5b[b] - zb[b]
ok = ~np.isnan(d); per = [float(roc_auc_score(y, P[r])) for r in range(5)]; m5 = float(np.mean(per)); za = float(roc_auc_score(y, zs))
lo, hi = float(np.percentile(d[ok], 2.5)), float(np.percentile(d[ok], 97.5))
np.savez_compressed(f"{R}/draws/A_sens_adress2020_h100hbm3_tf554_draws.npz", diff=d, mean5=m5b, zeroshot=zb, point_diff=np.array(m5 - za))
zc = json.load(open(f"{pod}/out/r2_adress2020_zs_compare.json"))
with open(f"{R}/A_sensitivity_r2.tsv", "w") as fh:
    fh.write("id\textraction\tn\tprobe_mean5_table1\tprobe_mean5\tminus_table1\tper_repeat\tzeroshot_auc\tgap\tgap_lo\tgap_hi\tzeroshot_p_yes_exact_equal\toof_file\n")
    fh.write(f"adress2020_h100hbm3_tf554\tPART25 A retry, H100 80GB HBM3, transformers 5.5.4\t{len(y)}\t0.8043\t{m5:.4f}\t{m5-0.8043:+.4f}\t{' '.join(f'{a:.4f}' for a in per)}\t{za:.4f}\t{m5-za:+.4f}\t{lo:.4f}\t{hi:.4f}\t{zc['p_yes_exact_equal']}/{zc['n']}\t{oofp}\n")
json.dump({"what": __doc__, "per_repeat": per, "mean5": m5, "zeroshot": za, "gap": m5 - za, "ci": [lo, hi], "usable": int(ok.sum()),
           "zs_compare": zc, "oof_csv": oofp, "oof_sha256": hashlib.sha256(open(oofp, "rb").read()).hexdigest(),
           "pod_env": [l.strip() for l in open(f"{pod}/logs/pip.log") if "VERSIONS" in l]}, open(f"{R}/A_sensitivity_r2.sidecar.json", "w"), indent=1)
print(f"H100 HBM3 tf5.5.4 adress2020 mean5 {m5:.4f} per {[round(a,4) for a in per]} gap {m5-za:+.4f} [{lo:.4f}, {hi:.4f}] zs exact {zc['p_yes_exact_equal']}/{zc['n']}")
