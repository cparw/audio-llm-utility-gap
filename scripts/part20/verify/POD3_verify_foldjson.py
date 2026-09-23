"""PART20 POD3 verifier: rebuild the out-of-fold table from the raw per-fold outputs ft966_fold{k}.json (own code),
check each fold scored exactly its fold-file rows and trained on all the others, and compare the compute csv
row by row (all score columns) with the raw fold outputs.
usage: POD3_verify_foldjson.py OUTDIR FOLDS_CSV MANIFEST [COMPUTE_CSV]"""
import csv, json, sys, os, glob, numpy as np
D, FOLDS, MAN = sys.argv[1:4]; CC = sys.argv[4] if len(sys.argv) > 4 else None
fr = list(csv.DictReader(open(FOLDS))); mr = list(csv.DictReader(open(MAN)))
ff = np.array([int(r["fold"]) for r in fr]); n = len(fr)
out = {"folds_present": []}; tab = np.full((n, 3), np.nan); owner = np.full(n, -1)
for k in range(5):
    f = f"{D}/ft966_fold{k}.json"
    if not os.path.exists(f): continue
    j = json.load(open(f)); out["folds_present"].append(k)
    idx = np.array(sorted(int(i) for i in j["res"]))
    exp = np.where(ff == k)[0]
    out[f"fold{k}"] = {"n_test": j["n_test"], "n_train": j["n_train"], "test_eq_foldfile": bool(np.array_equal(idx, exp)),
                       "n_train_eq": j["n_train"] == n - len(exp), "epochs": len(j["epoch_losses"]),
                       "losses": [round(x, 4) for x in j["epoch_losses"]], "minutes": j.get("minutes"),
                       "sft_ids": j.get("sft_ids"), "yes_ids": j.get("yes_ids"), "no_ids": j.get("no_ids")}
    for i, v in j["res"].items():
        i = int(i)
        if owner[i] != -1: out.setdefault("double_scored", []).append(i)
        owner[i] = k; tab[i] = v
out["rows_scored"] = int((owner >= 0).sum())
# repeated agreement segments: identical scores?
seg = [r["seg_uid"] for r in mr]; sets = [r["set"] for r in mr]
grp = {}
for i in range(n):
    if sets[i] == "agreement" and owner[i] >= 0: grp.setdefault(seg[i], []).append(i)
spread = [float(np.nanmax(tab[ix, c]) - np.nanmin(tab[ix, c])) for ix in grp.values() if len(ix) > 1 for c in range(3)]
out["max_repeat_spread_any_col"] = max(spread) if spread else None
if CC and os.path.exists(CC):
    cr = list(csv.DictReader(open(CC))); out["compute_csv"] = CC; out["compute_csv_cols"] = list(cr[0].keys()); out["compute_csv_n"] = len(cr)
    cols = {}
    for c in cr[0].keys():
        try: cols[c] = np.array([float(r[c]) for r in cr])
        except ValueError: pass
    # match each raw column to a csv column
    for ci, nm in enumerate(["p_yes_paper", "answer_mass", "p_yes_sft"]):
        best = None
        for c, v in cols.items():
            if len(v) == n:
                d = float(np.nanmax(np.abs(v - tab[:, ci])))
                if best is None or d < best[1]: best = (c, d)
        out[f"csv_col_for_{nm}"] = best
    key_c = [os.path.basename(r.get("path", r.get("id", r.get("clip", "")))).replace(".wav", "") for r in cr]
    key_m = [os.path.basename(r["path"]).replace(".wav", "") for r in mr]
    out["csv_rows_positional_eq_manifest"] = key_c == key_m
    if "fold" in cr[0]:
        out["csv_fold_eq_foldfile"] = [int(r["fold"]) for r in cr] == ff.tolist()
print(json.dumps(out, indent=1, default=str))
