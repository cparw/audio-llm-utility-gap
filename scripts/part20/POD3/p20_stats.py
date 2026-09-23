"""Merge the 5 fold results into p14_ft966_oof.csv, compute AUCs (rank formula, scipy rankdata, ties averaged) and
speaker bootstraps (2000 draws, fresh default_rng(0) per cell, speakers with replacement, percentile 2.5/97.5;
paired = one speaker draw per replicate), write the sidecar and print paste lines."""
import csv, json, os, sys, hashlib, datetime, numpy as np
from scipy.stats import rankdata
O = "/workspace/scores/part20/POD3"; MAN = "/workspace/p20/p14_ft966_manifest.csv"
P = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
rows = list(csv.DictReader(open(MAN)))
res = {}; fold_meta = {}
for k in range(5):
    d = json.load(open(f"{O}/ft966_fold{k}.json")); fold_meta[k] = {kk: v for kk, v in d.items() if kk != "res"}
    for i, v in d["res"].items(): res[int(i)] = v
assert sorted(res) == list(range(len(rows))), "missing rows"
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows]); arm = np.array([r["set"] for r in rows])
uid = np.array([r["seg_uid"] for r in rows]); fold = np.array([int(r["fold"]) for r in rows])
re = {}; re_meta = {}
for k in range(5):
    d = json.load(open(f"{O}/ft966_fold{k}_reeval.json")); re_meta[k] = {kk: v for kk, v in d.items() if kk != "res"}
    for i, v in d["res"].items(): re[int(i)] = v
p6 = np.array([re[i][2] for i in range(len(rows))]); m6 = np.array([re[i][3] for i in range(len(rows))]); prr = np.array([re[i][0] for i in range(len(rows))])
p = np.array([res[i][0] for i in range(len(rows))]); mass = np.array([res[i][1] for i in range(len(rows))]); psft = np.array([res[i][2] for i in range(len(rows))])
with open(f"{O}/p14_ft966_oof.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["row", "seg_uid", "clip", "speaker", "arm", "pair_id", "label", "fold", "p_yes", "answer_mass", "p_yes_sft", "p_yes_rule3_literal", "answer_mass_rule3_literal", "prompt"])
    for i, r in enumerate(rows):
        w.writerow([i, r["seg_uid"], os.path.basename(r["path"]), r["speaker"], r["set"], r["pair_id"], r["label"], r["fold"], f"{p[i]:.10f}", f"{mass[i]:.10f}", f"{psft[i]:.10f}", f"{p6[i]:.10f}", f"{m6[i]:.10f}", P])
def auc(yy, ss):
    yy = np.asarray(yy); n1 = (yy == 1).sum(); n0 = (yy == 0).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(ss); return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
# agreement distinct: first occurrence of each seg_uid; check duplicates scored identically
ag = np.where(arm == "agreement")[0]; first = {}
for i in ag: first.setdefault(uid[i], i)
dist = np.array(sorted(first.values()))
dup_maxdiff = max(abs(p[i] - p[first[uid[i]]]) for i in ag)
con = np.where(arm == "conflict")[0]
sets = {"conflict": con, "agreement483": ag, "agreement311": dist}
uspk = np.unique(spk)
def boot(fn, draws=2000):
    rng = np.random.default_rng(0); by = {u: np.where(spk == u)[0] for u in uspk}; vals = []
    for _ in range(draws):
        pick = rng.choice(uspk, size=len(uspk), replace=True)
        ii = np.concatenate([by[u] for u in pick]); v = fn(ii)
        if np.all(np.isfinite(v)): vals.append(v)
    vals = np.array(vals); return np.percentile(vals, 2.5, axis=0), np.percentile(vals, 97.5, axis=0), len(vals)
out = {}
for nm, idx in sets.items():
    mask = np.zeros(len(rows), bool); mask[idx] = True
    a = auc(y[idx], p[idx])
    lo, hi, u = boot(lambda ii: auc(y[ii][mask[ii]], p[ii][mask[ii]]))
    out[nm] = dict(auc=a, lo=float(lo), hi=float(hi), n=int(len(idx)), n_spk=int(len(set(spk[idx]))), n_pos=int(y[idx].sum()), usable=u,
                   auc_p_yes_sft=auc(y[idx], psft[idx]), auc_rule3_literal=auc(y[idx], p6[idx]), auc_reeval_paper=auc(y[idx], prr[idx]))
mc = np.zeros(len(rows), bool); mc[con] = True
for nm, idx in (("diff_conflict_minus_agreement483", ag), ("diff_conflict_minus_agreement311", dist)):
    ma = np.zeros(len(rows), bool); ma[idx] = True
    d = auc(y[con], p[con]) - auc(y[idx], p[idx])
    lo, hi, u = boot(lambda ii: auc(y[ii][mc[ii]], p[ii][mc[ii]]) - auc(y[ii][ma[ii]], p[ii][ma[ii]]))
    out[nm] = dict(auc=d, lo=float(lo), hi=float(hi), n=int(len(con) + len(idx)), n_spk=int(len(set(spk[np.concatenate([con, idx])]))), usable=u)
def sha1mb(f): return hashlib.sha256(open(f, "rb").read(1 << 20)).hexdigest()
srcs = {"manifest_pod": MAN, "fold_file_pod": "/workspace/p20/part14_groupkfold5_pod.csv", "script": "/workspace/p20/p20_ft966.py",
        "stats_script": "/workspace/p20/p20_stats.py", "cut_script": "/workspace/p20/p20_cut.py", "cut_list": "/workspace/p20/p20_cut_list.csv", "eval_script": "/workspace/p20/p20_eval.py"}
side = {"result": "Qwen2.5-Omni projector fine-tuned on the 966 Part 14 E-DAIC clips (483 conflict + 483 agreement rows), out of fold by speaker",
        "model_id": "Qwen/Qwen2.5-Omni-7B", "prompt": P, "n": len(rows), "n_speakers": int(len(uspk)), "seed_bootstrap": 0, "draws": 2000,
        "sources_pod": srcs, "sha256_first_1MB": {k: sha1mb(v) for k, v in srcs.items() if os.path.exists(v)},
        "sources_mac": {"manifest": "<local data dir>/release/edaic_rerun/part14_manifest_new.csv",
                        "clips": "<local data dir>/release/edaic_rerun/part14_clips/ (re-cut on pod from E-DAIC tarballs, every clip sha256-identical to these)",
                        "fold_file": "folds/part20/POD3/part14_groupkfold5_pod.csv (NEW, POD tie rule)",
                        "training_manifest": "manifests/part20/POD3/p14_ft966_manifest.csv"},
        "command": "bash /workspace/p20/p20_launch.sh  (per fold: python3 -u p20_ft966.py /workspace/p20/p14_ft966_manifest.csv /workspace/scores/part20/POD3 FOLD); then python3 p20_stats.py",
        "recipe": "sft_projector.py unchanged: thinker.audio_tower.proj only (fp32), rest frozen, AdamW lr 1e-4, 3 epochs, 1 clip/update, CE over [' Yes',' No'] logits, RandomState(fold*10+ep) order, no class weighting, bf16, 30 s window",
        "training_note": "trained on all 966 rows as specified; agreement rows reuse 311 distinct segments up to 6 times each, so repeated segments are seen more than once per epoch",
        "p_yes": "P(Yes)/(P(Yes)+P(No)) at first answer position, single-token Yes/No variants (paper list); p_yes_sft = softmax over [' Yes',' No'] kept for reference",
        "agreement_duplicate_max_abs_diff": float(dup_maxdiff), "median_answer_mass": float(np.median(mass)), "min_answer_mass": float(mass.min()),
        "fold_meta": fold_meta, "reeval_meta": re_meta, "reeval_max_abs_diff_paper_p_yes": float(np.abs(prr - p).max()),
        "rule3_literal_note": "' YES' (14080) and ' NO' (5664) are single tokens; the paper scripts omit them. p_yes (primary) uses the paper list; p_yes_rule3_literal adds them, re-scored from the saved final fold checkpoints", "results": out, "date": datetime.date.today().isoformat()}
json.dump(side, open(f"{O}/p14_ft966_oof.sidecar.json", "w"), indent=1)
F = "scores/part20/POD3/p14_ft966_oof.csv"
lab = {"conflict": "ft966 conflict AUC", "agreement483": "ft966 agreement AUC (483 rows)", "agreement311": "ft966 agreement AUC (311 distinct)",
       "diff_conflict_minus_agreement483": "ft966 paired conflict minus agreement (483 rows)", "diff_conflict_minus_agreement311": "ft966 paired conflict minus agreement (311 distinct)"}
for k, v in out.items():
    if k.startswith("diff"):
        tail = "interval excludes zero" if (v["lo"] > 0 or v["hi"] < 0) else "interval includes zero"
    else:
        tail = "above 0.5" if v["auc"] > 0.5 else "below 0.5"
    print(f"PASTE\t{lab[k]}\t{v['auc']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}] n={v['n']} n_spk={v['n_spk']} file={F}\t|\t{v['auc']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}] {tail}")
print("reeval max|diff| paper p_yes", float(np.abs(prr - p).max()), "rule3-literal AUCs", {k: round(v.get('auc_rule3_literal', np.nan), 4) for k, v in out.items() if not k.startswith('diff')}, "max|p6-p|", float(np.abs(p6 - p).max()))
print("dup_maxdiff", dup_maxdiff, "median mass", np.median(mass), "min mass", mass.min(), "p_yes_sft AUCs", {k: round(v.get('auc_p_yes_sft', np.nan), 4) for k, v in out.items() if not k.startswith('diff')})
