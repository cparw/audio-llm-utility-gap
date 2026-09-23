"""PART20 POD3 verifier, fold file check (own code). Rebuilds GroupKFold(5) by speaker two ways
(own stable-argsort re-implementation; sklearn as installed on this machine, same call as sft_projector.py)
and compares with the saved fold file and the manifest fold column; checks labels/speakers vs reference."""
import csv, sys, json, numpy as np
from sklearn.model_selection import GroupKFold
import sklearn
FOLDS, MAN, REF = sys.argv[1:4]
fr = list(csv.DictReader(open(FOLDS))); mr = list(csv.DictReader(open(MAN))); ref = list(csv.DictReader(open(REF)))
out = {"n_fold_rows": len(fr), "n_man_rows": len(mr), "n_ref_rows": len(ref), "sklearn": sklearn.__version__, "numpy": np.__version__}
fspk = np.array([r["speaker_id"] for r in fr]); ff = np.array([int(r["fold"]) for r in fr])
mspk = np.array([r["speaker"] for r in mr]); mf = np.array([int(r["fold"]) for r in mr]); my = np.array([int(r["label"]) for r in mr])
out["fold_file_clip_eq_ref_seg_uid_positional"] = [r["clip_id"] for r in fr] == [r["seg_uid"] for r in ref]
out["fold_file_spk_eq_ref_positional"] = fspk.tolist() == [r["speaker_id"] for r in ref]
out["man_path_key_eq_ref_positional"] = [r["path"].split("/")[-1][:-4] for r in mr] == [f"{r['set']}_{r['speaker_id']}_{r['seg_uid']}" for r in ref]
out["man_label_eq_ref"] = my.tolist() == [int(r["label"]) for r in ref]
out["man_set_eq_ref"] = [r["set"] for r in mr] == [r["set"] for r in ref]
out["man_pair_eq_ref"] = [r["pair_id"] for r in mr] == [r["pair_id"] for r in ref]
out["man_fold_eq_fold_file"] = bool((mf == ff).all() and (mspk == fspk).all())
def stable(groups):
    ug, gi = np.unique(groups, return_inverse=True); cnt = np.bincount(gi)
    order = np.argsort(cnt, kind="stable")[::-1]; load = np.zeros(5); g2f = np.zeros(len(ug), int)
    for g in order:
        f = int(np.argmin(load)); load[f] += cnt[g]; g2f[g] = f
    return g2f[gi]
out["mismatch_vs_own_stable"] = int((stable(fspk) != ff).sum())
sk = np.full(len(mr), -1)
for k, (_, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(mr)), my, groups=mspk)): sk[te] = k
out["mismatch_vs_sklearn_here"] = int((sk != ff).sum())
# mac-order for contrast (numpy default argsort)
def mac(groups):
    ug, gi = np.unique(groups, return_inverse=True); cnt = np.bincount(gi)
    order = np.argsort(cnt)[::-1]; load = np.zeros(5); g2f = np.zeros(len(ug), int)
    for g in order:
        f = int(np.argmin(load)); load[f] += cnt[g]; g2f[g] = f
    return g2f[gi]
out["mismatch_vs_default_argsort_here"] = int((mac(fspk) != ff).sum())
s2f = {}; bad = 0
for s, f in zip(fspk, ff):
    if s in s2f and s2f[s] != f: bad += 1
    s2f.setdefault(s, f)
out["speaker_split_across_folds_rows"] = bad; out["n_speakers"] = len(s2f)
out["fold_rows"] = [int((ff == k).sum()) for k in range(5)]
out["fold_speakers"] = [int(len(set(fspk[ff == k]))) for k in range(5)]
sets = np.array([r["set"] for r in mr])
out["fold_conflict_rows"] = [int(((ff == k) & (sets == "conflict")).sum()) for k in range(5)]
out["fold_pos_rows"] = [int(((ff == k) & (my == 1)).sum()) for k in range(5)]
print(json.dumps(out, indent=1, default=str))
