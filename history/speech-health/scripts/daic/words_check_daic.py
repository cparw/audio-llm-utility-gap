"""The daic version of the alzheimers words check.

Train a probe only on segments where the words agree with the diagnosis,
test it only on segments where the words contradict it, no shared speakers.
If the probe drops on the contradiction set, it was reading words.

Needs hidden states for the scored daic segments. Run it where the daic
states live (the ptsd per layer run already extracted them):

    python words_check_daic.py --npz daic_states.npz --scored results/health/paradox/daic_segments_scored.csv

The npz must hold:
  feats: float array [n_segments, n_layers, dim]  (encoder or llm states)
  ids:   string array [n_segments], formatted "pid_start_end" with start/end
         in seconds matching the scored csv (e.g. "300_14.3_92.9")

Output: one csv, per layer AUC on the within set and on the conflict set.
"""
import argparse, csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--scored", required=True)
ap.add_argument("--out", default="daic_words_check.csv")
ap.add_argument("--q", type=float, default=1 / 3, help="tercile cut on the text model score")
args = ap.parse_args()

rows = list(csv.DictReader(open(args.scored)))
key = {f"{r['pid']}_{r['start']}_{r['end']}": i for i, r in enumerate(rows)}
z = np.load(args.npz, allow_pickle=True)
feats, ids = z["feats"], [str(s) for s in z["ids"]]
idx = [key[s] for s in ids if s in key]
missing = len(ids) - len(idx)
if missing:
    print(f"warning: {missing} of {len(ids)} ids not found in the scored csv")
feats = feats[[i for i, s in enumerate(ids) if s in key]]
sub = [rows[i] for i in idx]
y = np.array([int(r["y"]) for r in sub])
pid = np.array([r["pid"] for r in sub])
txt = np.array([float(r["oof"]) for r in sub])  # the text model's word score

lo, hi = np.quantile(txt, args.q), np.quantile(txt, 1 - args.q)
conflict = ((y == 1) & (txt <= lo)) | ((y == 0) & (txt >= hi))
agreement = ((y == 1) & (txt >= hi)) | ((y == 0) & (txt <= lo))
test_pids = set(pid[conflict])
train = agreement & ~np.isin(pid, list(test_pids))
print(f"agreement train {train.sum()} (speakers {len(set(pid[train]))}), "
      f"conflict test {conflict.sum()} (speakers {len(test_pids)}), no overlap")
if train.sum() < 30 or conflict.sum() < 20 or len(set(y[train])) < 2 or len(set(y[conflict])) < 2:
    raise SystemExit("not enough usable segments after the split, loosen --q")

with open(args.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "within_agreement_auc", "conflict_auc"])
    for L in range(feats.shape[1]):
        X = StandardScaler().fit_transform(feats[:, L, :].astype(np.float64))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[train], y[train])
        s = clf.decision_function(X)
        within = roc_auc_score(y[train], s[train])  # fit quality, optimistic on purpose
        conf = roc_auc_score(y[conflict], s[conflict])
        w.writerow([L, f"{within:.3f}", f"{conf:.3f}"])
        print(f"layer {L:2d}  fits its own training set {within:.3f}  holds on conflict {conf:.3f}")
print(f"wrote {args.out}")
