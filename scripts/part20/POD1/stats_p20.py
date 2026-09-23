"""Part 20 POD1 assembly + stats (Mac side). Reads the pod fold jsons, writes the per-clip csv + sidecar,
computes rank AUC (scipy rankdata, ties averaged) with speaker bootstrap (2000 draws, fresh default_rng(0) per cell,
speakers of that cell resampled with replacement, percentile 2.5/97.5) and PAIRED balanced-minus-standard differences
(one speaker draw per replicate, both score vectors scored on it).
usage: stats_p20.py FOLD_JSON_DIR TAG OUT_CSV [STD_CSV]"""
import sys, os, csv, json, hashlib, datetime, numpy as np
from scipy.stats import rankdata
FD, TAG, OUT = sys.argv[1:4]
STD = sys.argv[4] if len(sys.argv) > 4 else "<local data dir>/release/omni_final/omnisft_pitt_oof.csv"
S = os.path.dirname(os.path.abspath(__file__))
MAN = f"{S}/mf_pitt468_p20.csv"
FOLDS = "folds/pitt_groupkfold5_pod.csv"
CM = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
rows = list(csv.DictReader(open(MAN)))
fold_of = {r["clip_id"]: int(r["fold"]) for r in csv.DictReader(open(FOLDS))}
arm_cm = {os.path.basename(r["segment_path"]): r["set"] for r in csv.DictReader(open(CM))}
n = len(rows); sc = {k: np.full(n, np.nan) for k in ("p_yes", "p_yes_multi", "answer_mass")}
fjs = []
for k in range(5):
    j = json.load(open(f"{FD}/{TAG}_fold{k}.json")); fjs.append(j)
    for s in j["scores"]:
        for key in sc: sc[key][s["row"]] = s[key]
assert not np.isnan(sc["p_yes"]).any(), "missing scores"
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
arm = np.array([r["set"] for r in rows]); cid = [r["clip_id"] for r in rows]
assert all(arm_cm[c] == a for c, a in zip(cid, arm))
std = {os.path.basename(r["path"]): r for r in csv.DictReader(open(STD))}
assert all(std[c]["speaker"] == s and int(std[c]["label"]) == l and std[c]["set"] == a for c, s, l, a in zip(cid, spk, y, arm))
assert all(std[c]["prompt"] == P for c in cid)
s_std = np.array([float(std[c]["p_yes"]) for c in cid])
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "clip_id", "speaker", "label", "set", "fold", "p_yes", "p_yes_multi", "answer_mass", "std_p_yes", "prompt"])
    for i, r in enumerate(rows):
        w.writerow([r["path"], cid[i], spk[i], int(y[i]), arm[i], fold_of[cid[i]], repr(float(sc["p_yes"][i])), repr(float(sc["p_yes_multi"][i])),
                    repr(float(sc["answer_mass"][i])), repr(float(s_std[i])), P])
def auc(yy, ss):
    n1 = int((yy == 1).sum()); n0 = len(yy) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(ss); return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def boot(mask, score_list, B=2000):
    """one fresh rng(0) per cell; returns per-score percentile CIs and paired diff CI (score0 - score1)"""
    idx_all = np.where(mask)[0]; u = np.unique(spk[idx_all]); g = {s: idx_all[spk[idx_all] == s] for s in u}
    rng = np.random.default_rng(0); A = []
    for _ in range(B):
        ii = np.concatenate([g[s] for s in rng.choice(u, size=len(u), replace=True)])
        A.append([auc(y[ii], sv[ii]) for sv in score_list])
    return np.array(A), len(u)
def ci(v):
    v = v[~np.isnan(v)]; return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)
cells = {"all": np.ones(n, bool), "conflict": arm == "conflict", "agreement": arm == "agreement"}
res = {}
for name, m in cells.items():
    b, a_s, mm = auc(y[m], sc["p_yes"][m]), auc(y[m], s_std[m]), auc(y[m], sc["p_yes_multi"][m])
    A, nspk = boot(m, [sc["p_yes"], s_std])
    lo_b, hi_b, nb = ci(A[:, 0]); lo_s, hi_s, _ = ci(A[:, 1]); lo_d, hi_d, nd = ci(A[:, 0] - A[:, 1])
    Am, _ = boot(m, [sc["p_yes_multi"]]); lo_m, hi_m, _ = ci(Am[:, 0])
    res[name] = {"n": int(m.sum()), "n_speakers": int(nspk), "balanced": b, "balanced_ci": [lo_b, hi_b],
                 "standard": a_s, "standard_ci": [lo_s, hi_s], "diff_bal_minus_std": b - a_s, "diff_ci": [lo_d, hi_d],
                 "usable_draws": nb, "usable_diff_draws": nd, "balanced_multivariant": mm, "balanced_multivariant_ci": [lo_m, hi_m]}
# extra: conflict-minus-agreement gap inside the balanced run, paired (one draw over all speakers)
u = np.unique(spk); g = {s: np.where(spk == s)[0] for s in u}; rng = np.random.default_rng(0); G = []
for _ in range(2000):
    ii = np.concatenate([g[s] for s in rng.choice(u, size=len(u), replace=True)])
    ic, ia = ii[arm[ii] == "conflict"], ii[arm[ii] == "agreement"]
    G.append([auc(y[ic], sc["p_yes"][ic]) - auc(y[ia], sc["p_yes"][ia]), auc(y[ic], s_std[ic]) - auc(y[ia], s_std[ia])])
G = np.array(G)
gap = {"balanced_gap": res["conflict"]["balanced"] - res["agreement"]["balanced"], "balanced_gap_ci": list(ci(G[:, 0])[:2]),
       "standard_gap": res["conflict"]["standard"] - res["agreement"]["standard"], "standard_gap_ci": list(ci(G[:, 1])[:2]),
       "gap_change": (res["conflict"]["balanced"] - res["agreement"]["balanced"]) - (res["conflict"]["standard"] - res["agreement"]["standard"]),
       "gap_change_ci": list(ci(G[:, 0] - G[:, 1])[:2])}
def sha1m(p):
    with open(p, "rb") as fh: return hashlib.sha256(fh.read(1 << 20)).hexdigest()
srcs = {"manifest": MAN, "folds": FOLDS, "conflict_manifest": CM, "standard_per_clip": STD,
        "training_script": f"{S}/balanced_ft_pitt.py", "stats_script": os.path.abspath(__file__),
        "recipe_reference": "<local data dir>/scratch/p15/sft_projector.py"}
srcs.update({f"fold{k}_json": f"{FD}/{TAG}_fold{k}.json" for k in range(5)})
side = {"result": os.environ.get("RESULT_TITLE", "PART20 POD1 balanced projector fine-tune, Pitt 468, Qwen2.5-Omni, out of fold"),
        "score_columns_note": os.environ.get("RESULT_NOTE", "p_yes = this run; std_p_yes = the reference run (STD_CSV); results keys: balanced = this run, standard = reference"),
        "model_id": "Qwen/Qwen2.5-Omni-7B", "prompt_verbatim": P, "n": n, "n_speakers": int(len(set(spk))),
        "n_conflict": int((arm == "conflict").sum()), "n_agreement": int((arm == "agreement").sum()), "seed_bootstrap": 0,
        "recipe": "sft_projector.py: projector (thinker.audio_tower.proj) only, float32 forward, encoder + LM frozen bf16, AdamW lr 1e-4, "
                  "3 epochs, one clip per update, order RandomState(fold*10+ep).permutation(train_idx), CE over [' Yes',' No'] at logits[0,-1]",
        "only_change": "per-clip loss weight: within each training fold and label y, w = (N_y/2)/N_{y,arm}, rescaled to mean 1 over the fold",
        "weight_tables": {f"fold{j['fold']}": j["weight_table"] for j in fjs},
        "losses": {f"fold{j['fold']}": j["losses"] for j in fjs},
        "p_yes_definition": "PRIMARY p_yes = softmax over the two logits [' Yes',' No'] at the first answer position (the recipe's own scoring, "
                            "identical to the standard file's p_yes, so the paired difference is like for like). p_yes_multi = rule-3 "
                            "multivariant (single-token Yes/yes/YES vs No/no/NO, bare + leading space); answer_mass = that total from full softmax.",
        "answer_mass_median": float(np.median(sc["answer_mass"])), "answer_mass_min": float(np.min(sc["answer_mass"])),
        "stats": "AUC scipy rankdata ties averaged; bootstrap 2000, default_rng(0) fresh per cell, speakers of that cell resampled with "
                 "replacement, percentile 2.5/97.5; paired diff = one speaker draw per replicate, both runs scored on it",
        "results": res, "extra_conflict_minus_agreement_gap": gap,
        "sources_absolute": srcs, "sha256_first_1MB": {k: sha1m(v) for k, v in srcs.items()},
        "command_pod": "python3 balanced_ft_pitt.py mf_pitt468_p20.csv pitt_groupkfold5_pod.csv <outdir> balanced balanced <foldlist>",
        "command_stats": " ".join(["python3"] + sys.argv), "per_clip_csv": os.path.abspath(OUT),
        "date": datetime.datetime.utcnow().isoformat() + "Z"}
json.dump(side, open(OUT.replace(".csv", ".sidecar.json"), "w"), indent=1)
print(json.dumps({"results": res, "gap": gap}, indent=1))
