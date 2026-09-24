"""accel_24sep othermodels: independent check of part25/C/C_other_models.tsv.
Independent code. Reads only. Writes only into this folder.
AUC = sklearn roc_auc_score. Paired speaker bootstrap: 2000 draws, fresh numpy default_rng(0) per cell,
idx = rng.choice(n_spk, n_spk, replace=True) over sorted unique probe-file speakers, all clips of each drawn speaker,
probe term (mean of per-repeat AUCs, or the single AUC) and answer term in the same draw, 2.5/97.5 percentiles.
Draws with one class only are dropped and counted."""
import os, sys, glob, json, hashlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool

OUT = "checks/accel_24sep/othermodels"
TSV = "scores/part25/C/C_other_models.tsv"
REL = "<local data dir>/release"
PLR = "<local data dir>/paper1_local_runs"
P25 = "scores/part25/C"
DS = {"PC-GITA": "pcgita", "NeuroVoz": "neurovoz", "MDVR-KCL": "kcl", "E-DAIC": "edaic", "Pitt": "pitt",
      "ADReSSo": "adresso", "ADReSS-2020": "adress2020"}
MK = {"Qwen2-Audio": "q2a", "Qwen3-Omni-30B-A3B": "q3o", "Kimi-Audio": "kimi"}
NB = 2000


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def candidates(mk, ds):
    c = []
    if mk == "q2a":
        c += [f"{PLR}/probe2/{ds}_enc_nested_oof.csv", f"{PLR}/probe2/pod_runs/{ds}_enc_nested_oof.csv"]
        if ds == "kcl": c.append(f"{PLR}/probe2/kcl_local_enc_nested_oof.csv")
        if ds == "pcgita": c.append(f"{PLR}/probe2/pcgita_ash_enc_nested_oof.csv")
        c += sorted(glob.glob(f"{P25}/pull*/p25C-*/out/q2a_{ds}_enc_rep5_oof.csv"))
    elif mk == "q3o":
        c += [f"{REL}/overnight2/q3o_new/q3o_{ds}_enc_nested_oof.csv", f"{REL}/podD2_final/out/q3o_{ds}_enc_nested_oof.csv",
              f"{REL}/edaic_rerun/part16/POD3/q3o_{ds}_encoder_nested_oof.csv",
              f"{REL}/edaic_rerun/part16/POD3/out/q3o_{ds}_encoder_nested_oof.csv"]
        c += sorted(glob.glob(f"{P25}/pull*/p25C-*/out/q3o_{ds}_enc_rep5_oof.csv"))
    else:
        c += sorted(glob.glob(f"{P25}/pull*/p25C-*/out/kimi_{ds}_llm_rep5_oof.csv"))
    return [p for p in c if os.path.isfile(p)]


def pcols(df):
    return [k for k in df.columns if (k.startswith("p_probe") or k.startswith("p_seed"))]


def boot(args):
    key, y, P, z, spk = args
    u = np.unique(spk); by = [np.flatnonzero(spk == s) for s in u]
    rng = np.random.default_rng(0)
    dd = np.full(NB, np.nan); pp = np.full(NB, np.nan); qq = np.full(NB, np.nan)
    for b in range(NB):
        idx = rng.choice(len(u), size=len(u), replace=True)
        r = np.concatenate([by[i] for i in idx]); yb = y[r]
        if yb.min() == yb.max(): continue
        pa = np.mean([roc_auc_score(yb, P[r, j]) for j in range(P.shape[1])])
        za = roc_auc_score(yb, z[r])
        pp[b], qq[b], dd[b] = pa, za, pa - za
    ok = ~np.isnan(dd)
    lo, hi = np.percentile(dd[ok], [2.5, 97.5])
    np.savez_compressed(f"{OUT}/draws/{key}.npz", diff=dd, probe=pp, zs=qq)
    return key, float(lo), float(hi), int(ok.sum())


def main():
    os.makedirs(f"{OUT}/draws", exist_ok=True)
    T = pd.read_csv(TSV, sep="\t")
    rows, jobs = [], []
    for _, t in T.iterrows():
        mk, ds = MK[t["model"]], DS[t["dataset"]]
        zsrc = t["zeroshot_perclip_source"]
        Z = pd.read_csv(zsrc)
        zs_auc = roc_auc_score(Z["label"].astype(int), Z["p_yes"].astype(float))
        msrc = str(t["probe_master_source"]).split(" (")[0]
        if msrc.endswith(".csv"): msrc = f"{REL}/overnight2/q3o_new/q3o_{ds}_nested_repeats.json"
        stream = "llm" if mk == "kimi" else "enc"
        mj = json.load(open(msrc)); mrep = mj[stream]["per_repeat"]; mmean = mj[stream]["mean"]
        base = dict(model=t["model"], dataset=t["dataset"], condition=t["condition"], master_json=msrc, tsv_probe_master=t["probe_auc_master"],
                    json_probe_mean=mmean, json_per_repeat=json.dumps(mrep), json_mean_matches_tsv="yes" if round(mmean, 4) == round(t["probe_auc_master"], 4) else "NO",
                    tsv_zs_master=t["zeroshot_auc_master"], zs_recomputed=round(zs_auc, 4),
                    zs_matches="yes" if round(zs_auc, 4) == round(t["zeroshot_auc_master"], 4) else "NO",
                    zs_n=len(Z), zs_file=zsrc, tsv_diff_lo=t["diff_lo"], tsv_diff_hi=t["diff_hi"], tsv_point=t["diff_point_from_perclip"],
                    tsv_probe_source=t["probe_perclip_source"])
        cands = candidates(mk, ds)
        if not cands:
            r = dict(base); r.update(probe_file="NONE FOUND"); rows.append(r); continue
        for pf in cands:
            D = pd.read_csv(pf); pc = pcols(D)
            M = D.merge(Z[["clip", "label", "p_yes"]], on="clip", how="inner", suffixes=("", "_zs"))
            lab_ok = bool((M["label"].astype(int) == M["label_zs"].astype(int)).all())
            y = M["label"].astype(int).to_numpy(); P = M[pc].astype(float).to_numpy(); z = M["p_yes"].astype(float).to_numpy()
            spk = M["speaker"].astype(str).to_numpy()
            per = [roc_auc_score(y, P[:, j]) for j in range(P.shape[1])]
            full_per = [roc_auc_score(D["label"].astype(int), D[c].astype(float)) for c in pc]
            zs_m = roc_auc_score(y, z)
            kind = "rep5" if len(pc) == 5 else "single"
            match = ""
            if kind == "rep5":
                match = "yes" if [round(v, 4) for v in full_per] == [round(v, 4) for v in mrep] else "NO"
            key = f"{mk}_{ds}_{len(rows):02d}"
            r = dict(base); r.update(probe_file=pf, probe_md5=md5(pf), kind=kind, n_probe=len(D), n_joined=len(M),
                                     n_spk=len(np.unique(spk)), labels_agree="yes" if lab_ok else "NO",
                                     per_repeat_recomputed=json.dumps([round(v, 4) for v in full_per]),
                                     probe_recomputed=round(float(np.mean(full_per)), 4),
                                     per_repeat_matches_json=match,
                                     probe_mean_matches_master=("yes" if round(float(np.mean(full_per)), 4) == round(mmean, 4) else "NO"),
                                     point_diff=round(float(np.mean(per)) - zs_m, 4), key=key)
            rows.append(r); jobs.append((key, y, P, z, spk))
    with Pool(max(2, (os.cpu_count() or 4) - 1)) as pool:
        res = {k: (lo, hi, n) for k, lo, hi, n in pool.map(boot, jobs)}
    for r in rows:
        if r.get("key") in res:
            lo, hi, n = res[r["key"]]
            r.update(diff_lo=round(lo, 4), diff_hi=round(hi, 4), boot_usable=n, excludes_zero="yes" if (lo > 0 or hi < 0) else "no",
                     probe_above_answer="yes" if r["point_diff"] > 0 else "no")
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/om_cells.tsv", sep="\t", index=False)
    print(out[["model", "dataset", "kind", "probe_file", "n_joined", "n_spk", "labels_agree", "probe_recomputed", "per_repeat_matches_json",
               "probe_mean_matches_master", "zs_recomputed", "zs_matches", "point_diff", "diff_lo", "diff_hi", "boot_usable"]].to_string())


if __name__ == "__main__":
    main()
