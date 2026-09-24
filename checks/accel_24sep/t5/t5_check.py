#!/usr/local/bin/python3
"""Independent check of part25/T5/T5_counts.tsv. Read only on inputs; writes only in this folder."""
import csv, glob, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = os.path.dirname(os.path.abspath(__file__))
T5 = "scores/part25/T5/T5_counts.tsv"
LO = "scores/leftovers_23sep/t5"
PULL = LO + "/pull"
OF = "<local data dir>/release/omni_final"
ER = "<local data dir>/release/edaic_rerun"

# zero shot per clip files, one per T5 row (independently chosen, then compared with the T5 zeroshot_file column)
ZS = {
    "neurovoz": (OF + "/omni_neurovoz_zeroshot_scores.csv", "speaker", "p_yes"),
    "kcl": (OF + "/omni_kcl_zeroshot_scores.csv", "speaker", "p_yes"),
    "adresso": (OF + "/omni_adresso_zeroshot_scores.csv", "speaker", "p_yes"),
    "adress2020": (OF + "/omni_adress2020_zeroshot_scores.csv", "speaker", "p_yes"),
    "edaic_full": (ER + "/variants/o25_full_zeroshot_scores.csv", "speaker", "p_yes"),
    "edaic_full_statesrun": (ER + "/part16/EDAICFULL/full_zeroshot_scores.csv", "speaker", "p_yes"),
    "edaic_full_lifted": ("<local data dir>/release/scores/part22/edaic_lifted_perclip.csv", "pid", "p_yes_lifted"),
    "edaic_first30": (OF + "/omni_edaic_zeroshot_scores.csv", "speaker", "p_yes"),
    "edaic_mid300": (ER + "/variants/o25_mid300_zeroshot_scores.csv", "speaker", "p_yes"),
    "edaic_mid30": (ER + "/variants/o25_mid30_zeroshot_scores.csv", "speaker", "p_yes"),
    "pitt": (OF + "/omni_pitt_zeroshot_scores.csv", "speaker", "p_yes"),
    "pcgita": (OF + "/omni_pcgita_zeroshot_scores.csv", "speaker", "p_yes"),
}
SAVED = {
    "neurovoz": OF + "/omni_neurovoz_encoder_perlayer.csv",
    "kcl": OF + "/omni_kcl_encoder_perlayer.csv",
    "adresso": OF + "/omni_adresso_encoder_perlayer.csv",
    "adress2020": OF + "/omni_adress2020_encoder_perlayer.csv",
    "edaic_full": ER + "/variants/o25_full_encoder_perlayer.csv",
    "edaic_full_statesrun": ER + "/variants/o25_full_encoder_perlayer.csv",
    "edaic_full_lifted": ER + "/variants/o25_full_encoder_perlayer.csv",
    "edaic_first30": OF + "/omni_edaic_encoder_perlayer.csv",
    "edaic_mid300": ER + "/variants/o25_mid300_encoder_perlayer.csv",
    "edaic_mid30": ER + "/variants/o25_mid30_encoder_perlayer.csv",
    "pitt": OF + "/omni_pitt_encoder_perlayer.csv",
    "pcgita": OF + "/omni_pcgita_encoder_perlayer.csv",
}
# refit: (t5_perlayer dataset key, primary pod, job prefix, all pods that ran this job)
REFIT = {
    "neurovoz": ("neurovoz", "lo-t5-8", "neurovoz_encproj", ["lo-t5-8"]),
    "adress2020": ("adress2020", "lo-t5-7", "adress2020_encproj", ["lo-t5-7"]),
    "edaic_full": ("edaicfull", "lo-t5-9", "edaicfull_encproj", ["lo-t5-9"]),
    "edaic_full_statesrun": ("edaicfull", "lo-t5-9", "edaicfull_encproj", ["lo-t5-9"]),
    "edaic_full_lifted": ("edaicfull", "lo-t5-9", "edaicfull_encproj", ["lo-t5-9"]),
    "edaic_first30": ("edaic30", "lo-t5-10", "edaic30_encproj", ["lo-t5-10", "lo-t5-9", "lo-t5-4"]),
    "pitt": ("pitt", "lo-t5-10", "pitt_encproj", ["lo-t5-10", "lo-t5-1"]),
}


def boot(df, spk_col, score_col):
    y = df["label"].to_numpy().astype(int)
    s = df[score_col].to_numpy().astype(float)
    spk = df[spk_col].astype(str).to_numpy()
    auc = roc_auc_score(y, s)
    uniq = np.unique(spk)
    idx_of = {u: np.flatnonzero(spk == u) for u in uniq}
    rng = np.random.default_rng(0)
    vals = []
    skipped = 0
    for _ in range(2000):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        ii = np.concatenate([idx_of[uniq[k]] for k in pick])
        yy = y[ii]
        if yy.min() == yy.max():
            skipped += 1
            continue
        vals.append(roc_auc_score(yy, s[ii]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(auc=auc, lo=lo, hi=hi, n=len(df), n_spk=len(uniq), n_pos=int(y.sum()), usable=len(vals), skipped=skipped)


def counts(curve, ans, hi):
    c = np.asarray(curve, float)
    return dict(
        above_ans=int((c > ans).sum()), above_hi=int((c > hi).sum()),
        not_ans=[int(i) for i in np.flatnonzero(~(c > ans))], not_hi=[int(i) for i in np.flatnonzero(~(c > hi))],
        mn=float(c.min()), mn_layer=int(c.argmin()), mx=float(c.max()), n=len(c))


t5 = pd.read_csv(T5, sep="\t", dtype=str, keep_default_na=False)
pl = pd.read_csv(LO + "/t5_perlayer.csv")
res = {}
rows = []
for _, r in t5.iterrows():
    key = r["dataset"]
    zf, spk_col, sc = ZS[key]
    assert zf == r["zeroshot_file"], (key, zf, r["zeroshot_file"])
    assert SAVED[key] == r["saved_curve"], (key, SAVED[key], r["saved_curve"])
    df = pd.read_csv(zf)
    b = boot(df, spk_col, sc)
    out = dict(dataset=key, **b)
    out["t5_auc"] = float(r["answer_auc"]); out["t5_lo"] = float(r["answer_lo"]); out["t5_hi"] = float(r["answer_hi"])
    sv = pd.read_csv(SAVED[key]).sort_values("layer")
    cs = counts(sv["auc_oof"], b["auc"], b["hi"])
    out.update({"saved_" + k: v for k, v in cs.items()})
    out["t5_saved_above_answer"] = r["saved_above_answer"]; out["t5_saved_above_hi"] = r["saved_above_hi"]
    out["t5_refit"] = r["refit"]
    if key in REFIT:
        dkey, pod, job, pods = REFIT[key]
        sub = pl[(pl.dataset == dkey) & (pl.stream == "enc")].sort_values("layer")
        assert list(sub.layer) == list(range(32)), key
        assert set(sub.pod) == {pod}, (key, set(sub.pod))
        c_pl = counts(sub["refit_auc_oof"], b["auc"], b["hi"])
        out.update({"refit_" + k: v for k, v in c_pl.items()})
        # the main pull file named in T5
        pf = f"{PULL}/{pod}/out/{job}_encoder_perlayer.csv"
        assert pf == r["refit_file"], (pf, r["refit_file"])
        pc = pd.read_csv(pf).sort_values("layer")
        out["pull_vs_perlayer_maxabs"] = float(np.abs(pc["auc_oof"].to_numpy() - sub["refit_auc_oof"].to_numpy()).max())
        # recompute auc_oof from the saved OOF score arrays
        z = np.load(f"{PULL}/{pod}/out/{job}__podfile_oof.npz", allow_pickle=True)
        rec = np.array([roc_auc_score(z["label"].astype(int), z["enc"][:, L]) for L in range(z["enc"].shape[1])])
        out["oof_recompute_maxabs"] = float(np.abs(rec - pc["auc_oof"].to_numpy()).max())
        c_rec = counts(rec, b["auc"], b["hi"])
        out["recomputed_above_ans"] = c_rec["above_ans"]; out["recomputed_above_hi"] = c_rec["above_hi"]
        # labels in the OOF file agree with the zero-shot labels, by clip name
        zl = dict(zip(df[spk_col if key == "edaic_full_lifted" else "clip"].astype(str), df["label"].astype(int)))
        names = [str(n) for n in z["name"]]
        if key == "edaic_full_lifted":
            names = [n.replace(".wav", "") for n in names]
        miss = [n for n in names if n not in zl]
        out["oof_names_missing_in_zs"] = len(miss)
        out["oof_label_mismatch"] = int(sum(1 for n, l in zip(names, z["label"]) if n in zl and zl[n] != int(l)))
        # every variant refit file for this job across all pods
        var = []
        for p in pods:
            for f in sorted(glob.glob(f"{PULL}/{p}/out/{job}__*_enc.csv")) + [f"{PULL}/{p}/out/{job}_encoder_perlayer.csv"]:
                if not os.path.exists(f):
                    continue
                v = pd.read_csv(f).sort_values("layer")
                cv = counts(v["auc_oof"], b["auc"], b["hi"])
                var.append(dict(file=f.replace(PULL + "/", ""), above_ans=cv["above_ans"], above_hi=cv["above_hi"],
                                not_ans=cv["not_ans"], mn=cv["mn"], mn_layer=cv["mn_layer"]))
        out["variants"] = var
        out["variant_above_ans_range"] = [min(v["above_ans"] for v in var), max(v["above_ans"] for v in var)]
        out["variant_above_hi_range"] = [min(v["above_hi"] for v in var), max(v["above_hi"] for v in var)]
        out["t5_refit_above_answer"] = r["refit_above_answer"]; out["t5_refit_above_hi"] = r["refit_above_hi"]
        out["t5_layers_not_above_answer"] = r["layers_not_above_answer"]; out["t5_layers_not_above_hi"] = r["layers_not_above_hi"]
        out["t5_refit_min"] = r["refit_min"]; out["t5_refit_min_layer"] = r["refit_min_layer"]
    res[key] = out

# refit coverage: which datasets have any refit at all
cov = {
    "t5_perlayer_datasets": sorted(pl.dataset.unique().tolist()),
    "pull_encoder_files": sorted(os.path.relpath(f, PULL) for f in glob.glob(PULL + "/*/out/*encoder_perlayer.csv")),
    "pull_any_file_mentions": {k: sorted(os.path.relpath(f, PULL) for f in glob.glob(PULL + "/**/*", recursive=True)
                                         if k in os.path.basename(f).lower()) for k in ["kcl", "adresso", "pcgita", "gita", "mid300", "mid30"]},
    "assign": open(LO + "/assign.txt").read(),
}
json.dump(dict(results=res, coverage=cov), open(OUT + "/t5_check.json", "w"), indent=1, default=str)

# flat tsv
flat = []
for k, o in res.items():
    flat.append(dict(dataset=k, auc=o["auc"], lo=o["lo"], hi=o["hi"], usable=o["usable"], t5_auc=o["t5_auc"], t5_lo=o["t5_lo"], t5_hi=o["t5_hi"],
                     refit_above_ans=o.get("refit_above_ans", ""), refit_above_hi=o.get("refit_above_hi", ""),
                     t5_refit_above_ans=o.get("t5_refit_above_answer", ""), t5_refit_above_hi=o.get("t5_refit_above_hi", ""),
                     refit_not_ans=";".join(map(str, o.get("refit_not_ans", []))), refit_not_hi=";".join(map(str, o.get("refit_not_hi", []))),
                     refit_min=o.get("refit_mn", ""), refit_min_layer=o.get("refit_mn_layer", ""),
                     recomputed_above_ans=o.get("recomputed_above_ans", ""), recomputed_above_hi=o.get("recomputed_above_hi", ""),
                     oof_recompute_maxabs=o.get("oof_recompute_maxabs", ""), pull_vs_perlayer_maxabs=o.get("pull_vs_perlayer_maxabs", ""),
                     variant_above_ans_range=o.get("variant_above_ans_range", ""), variant_above_hi_range=o.get("variant_above_hi_range", ""),
                     n_variants=len(o.get("variants", [])), oof_label_mismatch=o.get("oof_label_mismatch", ""), oof_names_missing=o.get("oof_names_missing_in_zs", ""),
                     saved_above_ans=o["saved_above_ans"], saved_above_hi=o["saved_above_hi"], t5_saved_above_ans=o["t5_saved_above_answer"], t5_saved_above_hi=o["t5_saved_above_hi"],
                     saved_not_hi=";".join(map(str, o["saved_not_hi"])), saved_min=o["saved_mn"], saved_min_layer=o["saved_mn_layer"], t5_refit=o["t5_refit"]))
pd.DataFrame(flat).to_csv(OUT + "/t5_check.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(pd.DataFrame(flat).T.to_string())
print(json.dumps(cov, indent=1))
for k, o in res.items():
    if "variants" in o:
        print(k)
        for v in o["variants"]:
            print("   ", v["file"], v["above_ans"], v["above_hi"], v["not_ans"], round(v["mn"], 6), v["mn_layer"])
