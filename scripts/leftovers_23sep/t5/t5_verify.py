#!/usr/bin/env python3
"""Independent verifier for t5_values.tsv. Written apart from t5_analyze.py: no pandas, its own file lookup,
its own AUC (pairwise count, ties 0.5), its own 4 dp rule ('%.4f' strings). Each value is recomputed from the
raw pulled pod files and the saved curves, and compared with the claimed value. For every pod refit value it also
recomputes the same number from the independent refit code (vrefit files) as a second route."""
import csv, json, os, glob, sys, math
T5 = "<local data dir>/leftovers_23sep/t5"
R = "<local data dir>/release"
SAVED = {"pitt": R + "/omni_final/omni_pitt_{}_perlayer.csv", "edaic_first30": R + "/omni_final/omni_edaic_{}_perlayer.csv",
         "adress2020": R + "/omni_final/omni_adress2020_{}_perlayer.csv", "neurovoz": R + "/omni_final/omni_neurovoz_{}_perlayer.csv",
         "edaic_full": R + "/edaic_rerun/variants/o25_full_{}_perlayer.csv"}
ZS = {"pitt": R + "/omni_final/omni_pitt_zeroshot_scores.csv", "edaic_first30": R + "/omni_final/omni_edaic_zeroshot_scores.csv",
      "adress2020": R + "/omni_final/omni_adress2020_zeroshot_scores.csv", "neurovoz": R + "/omni_final/omni_neurovoz_zeroshot_scores.csv",
      "edaic_full": R + "/edaic_rerun/variants/o25_full_zeroshot_scores.csv"}
JOBDS = {"pitt": "pitt", "edaic_first30": "edaic30", "adress2020": "adress2020", "neurovoz": "neurovoz", "edaic_full": "edaicfull"}
FNAME = {"enc": "encoder", "proj": "proj", "llm": "llm", "ans": "ans"}
JOBSTREAM = {"enc": "encproj", "proj": "encproj", "llm": "llm", "ans": "ans"}

def col(path, name):
    with open(path) as fh:
        rd = csv.DictReader(fh); return [float(r[name]) for r in rd]
def auc_pairs(lab, sc):
    pos = [s for l, s in zip(lab, sc) if l == 1]; neg = [s for l, s in zip(lab, sc) if l == 0]
    t = 0.0
    for a in pos:
        for b in neg: t += 1.0 if a > b else (0.5 if a == b else 0.0)
    return t / (len(pos) * len(neg))
def answer(ds):
    with open(ZS[ds]) as fh:
        rows = list(csv.DictReader(fh))
    return auc_pairs([int(r["label"]) for r in rows], [float(r["p_yes"]) for r in rows])
def s4(x): return "%.4f" % x
def e2(x): return "0" if x == 0 else "%.2e" % x
def cmp(refp, savp, name="auc_oof"):
    a = col(refp, name); b = col(savp, name); assert len(a) == len(b)
    return max(abs(x - y) for x, y in zip(a, b)), sum(s4(x) == s4(y) for x, y in zip(a, b)), len(a)

claims = list(csv.DictReader(open(f"{T5}/t5_values.tsv"), delimiter="\t"))
log = []; agree = disagree = 0
def P(s): print(s); log.append(s)
ANS = {}
for c in claims:
    cid, claimed = c["id"], c["value"]; mine = None; route2 = ""
    parts = cid.split("_")
    try:
        if cid.startswith("T5_diag_") and cid.endswith("_p17_verifier_refit_explained"):
            ds = "pitt" if "_pitt_" in cid else "edaic_first30"; job = ("pitt" if ds == "pitt" else "edaic30") + "_encproj"
            v = json.load(open(R + "/edaic_rerun/part17/verify/T5_verify.json"))
            rec = [r for r in v["extra"]["recompute"] if r["dataset"] == ds][0]["recomputed_oof"]
            a = col(f"{T5}/mac_diag/{job}__mac_macfile_enc.csv", "auc_oof")
            mine = e2(max(abs(x - y) for x, y in zip(a, rec)))
        elif cid.startswith("T5_diag_"):
            rest = cid[len("T5_diag_"):]
            ds = "pitt" if rest.startswith("pitt_") else "edaic_first30"
            rest = rest[len(ds) + 1:]            # enc_<variant>_<pod>
            assert rest.startswith("enc_"); rest = rest[4:]
            variant, pod = rest.rsplit("_", 1)
            job = JOBDS[ds] + "_encproj"
            if pod == "mac":
                refp = f"{T5}/mac_diag/{job}__{variant}_enc.csv"
            else:
                pdir = f"{T5}/pull/lo-t5-{pod[3:]}/out"
                refp = f"{pdir}/{job}_encoder_perlayer.csv" if variant == "exact_script" else f"{pdir}/{job}__{variant}_enc.csv"
            d, m, n = cmp(refp, SAVED[ds].format("encoder"))
            mine = f"{e2(d)}; {m}/{n}"
        elif cid.startswith("T5_folds_"):
            rest = cid[len("T5_folds_"):]; job, pod = rest.rsplit("_", 1)
            fdir = f"{T5}/mac_diag" if pod == "mac" else f"{T5}/pull/lo-t5-{pod[3:]}/out"
            nat = {r["clip_id"]: r["fold"] for r in csv.DictReader(open(f"{fdir}/{job}_folds_native.csv"))}
            fd = "pitt" if job.startswith("pitt") else "edaic"
            outs = []
            for tag in ("pod", "mac"):
                ref = {r["clip_id"]: r["fold"] for r in csv.DictReader(open(f"{R}/folds/{fd}_groupkfold5_{tag}.csv"))}
                outs.append(sum(1 for k in nat if ref.get(k) == nat[k]))
            mine = f"pod file {outs[0]}/{len(nat)}; Mac file {outs[1]}/{len(nat)}"
        elif cid.endswith("_enc_refit"):
            k = cid[3:-len("_enc_refit")]
            man = json.load(open(f"{T5}/inputs/inputs_manifest.json"))["sources"]
            hits = [p for p in glob.glob(R + "/**/*.npz", recursive=True) + glob.glob("<local data dir>/release_from_mac/**/*.npz", recursive=True)
                    if any(t in os.path.basename(p).lower() for t in ({"adresso": ["adresso"], "pcgita": ["pcgita"], "kcl": ["kcl"],
                           "edaic_mid300": ["mid300"], "edaic_mid30": ["mid30"]}[k])) and any(t in os.path.basename(p).lower() for t in ("omni", "o25", "p16_"))
                    and "states" in os.path.basename(p)]
            ok = k not in man
            if k == "adresso":
                import numpy as np
                ns = [len(np.load(p, allow_pickle=True)["name"]) for p in hits]
                ok = ok and all(n != 237 for n in ns); note = f"candidate states {len(hits)} with clip counts {ns}"
            else:
                ok = ok and len(hits) == 0; note = f"candidate states found: {len(hits)}"
            mine = "not possible" if ok else "states exist"; route2 = note
        else:
            # T5_<t5key>_<stream>_<pod>_<metric>
            key = None
            for t in SAVED:
                if cid.startswith(f"T5_{t}_"): key = t
            rest = cid[len(f"T5_{key}_"):]
            stream, pod, metric = rest.split("_", 2)
            job = f"{JOBDS[key]}_{JOBSTREAM[stream]}"; pdir = f"{T5}/pull/lo-t5-{pod[3:]}/out"
            refp = f"{pdir}/{job}_{FNAME[stream]}_perlayer.csv"; savp = SAVED[key].format(FNAME[stream]); vrp = f"{pdir}/{job}__vrefit_{stream}.csv"
            if metric == "maxabs":
                d, m, n = cmp(refp, savp); mine = e2(d)
                d2, m2, n2 = cmp(vrp, savp); route2 = f"vrefit route {e2(d2)}"
            elif metric == "match4dp":
                d, m, n = cmp(refp, savp); mine = f"{m}/{n}"
                d2, m2, n2 = cmp(vrp, savp); route2 = f"vrefit route {m2}/{n2}"
            elif metric == "meanstd_match4dp":
                _, m1, n = cmp(refp, savp, "auc_mean"); _, m2, _ = cmp(refp, savp, "auc_std"); mine = f"{m1}/{n} and {m2}/{n}"
                _, v1, _ = cmp(vrp, savp, "auc_mean"); _, v2, _ = cmp(vrp, savp, "auc_std"); route2 = f"vrefit route {v1}/{n} and {v2}/{n}"
            elif metric == "layers_above":
                if key not in ANS: ANS[key] = answer(key)
                a = ANS[key]; ra = sum(x > a for x in col(refp, "auc_oof")); sa = sum(x > a for x in col(savp, "auc_oof"))
                mine = f"{ra}/32 vs {sa}/32"; route2 = f"answer {a:.6f}; vrefit route {sum(x > a for x in col(vrp, 'auc_oof'))}/32"
    except Exception as ex:
        mine = f"ERROR {type(ex).__name__}: {ex}"
    ok = (mine == claimed)
    agree += ok; disagree += (not ok)
    P(f"{'AGREE   ' if ok else 'DISAGREE'} {cid:70s} claimed [{claimed}] mine [{mine}] {route2}")
P(f"TOTAL agree {agree} disagree {disagree} of {len(claims)}")
open(f"{T5}/verify/t5_verify.log", "w").write("\n".join(log) + "\n")
