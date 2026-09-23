"""T7 variant D: same folds and probe as T7_probe.py, but the layer for each outer fold is TAKEN
from the pod run's recorded layers_picked (part16/EDAICFULL/edaic_full_nested_repeats.json)
instead of re-selected on this Mac. Isolates the outer-fit numerics from the inner near-tie flips.
Writes T7_scratch/T7_oof_repeats_D_mac_podpicks.npz and prints per-repeat AUCs vs published."""
import csv, json, numpy as np, sys
sys.path.insert(0, "<local data dir>/release/edaic_rerun/part17")
from T7_probe import P16, SCR, SHARDS, REPEATS, auc_rank, probe
from sklearn.model_selection import GroupKFold
sc = list(csv.DictReader(open(f"{P16}/full_zeroshot_scores.csv")))
clips = [r["clip"] for r in sc]; y = np.array([int(r["label"]) for r in sc]); spk = np.array([r["speaker"] for r in sc])
pub = json.load(open(f"{P16}/edaic_full_nested_repeats.json"))
perclip = list(csv.DictReader(open(f"{P16}/edaic_full_perclip.csv")))
saved = dict(label=y, speaker=spk, clip=np.array(clips)); gate = {}
for key in ("enc", "proj", "llm", "ans"):
    X = np.stack([np.load(f"{SHARDS}/{c}.npz")[key] for c in clips])
    picks = pub[key]["layers_picked"]; k = 0; oofs = []; aucs = []
    for rep in range(REPEATS):
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=rep).split(X, y, groups=spk):
            oof[te] = probe(X[:, picks[k], :], y, tr, te); k += 1
        oofs.append(oof); aucs.append(auc_rank(y, oof))
    oofs = np.array(oofs); mine4 = [round(a, 4) for a in aucs]
    pubcol = np.array([float(r[f"oof_{key}"]) for r in perclip])
    gate[key] = dict(per_repeat_mine=mine4, per_repeat_published=pub[key]["per_repeat"],
                     per_repeat_match=mine4 == pub[key]["per_repeat"],
                     max_abs_per_repeat_diff=float(np.max(np.abs(np.array(aucs) - np.array(pub[key]["per_repeat"])))),
                     mean_of5_fullprec=float(np.mean(aucs)), mean_of5_published=pub[key]["mean_of_repeat_aucs"],
                     auc_of_mean_oof=auc_rank(y, oofs.mean(0)), auc_of_mean_oof_published=pub[key]["auc_of_mean_oof"],
                     max_abs_diff_mean_oof_vs_published_perclip=float(np.max(np.abs(oofs.mean(0) - pubcol))),
                     pearson_mean_oof_vs_published=float(np.corrcoef(oofs.mean(0), pubcol)[0, 1]))
    saved[f"{key}_oof_repeats"] = oofs; saved[f"{key}_layers_picked"] = np.array(picks); saved[f"{key}_per_repeat_auc"] = np.array(aucs)
    g = gate[key]
    print(f"D {key}: mine {mine4}  pub {pub[key]['per_repeat']}  match {g['per_repeat_match']}  "
          f"max|d| {g['max_abs_per_repeat_diff']:.4f}  mean5 {g['mean_of5_fullprec']:.4f} vs {g['mean_of5_published']}  "
          f"AUC(mean OOF) {g['auc_of_mean_oof']:.4f} vs {g['auc_of_mean_oof_published']}  r {g['pearson_mean_oof_vs_published']:.5f}", flush=True)
np.savez_compressed(f"{SCR}/T7_oof_repeats_D_mac_podpicks.npz", **saved)
json.dump(gate, open(f"{SCR}/T7_gate_D_mac_podpicks.json", "w"), indent=1)
