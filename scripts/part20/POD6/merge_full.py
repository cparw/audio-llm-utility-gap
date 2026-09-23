"""Merge the 907-clip subset states and the 363 other clips into the full 1270 NeuroVoz set, in the paper file order.
usage: merge_full.py SUBSET_states.npz REST_states.npz FULL_MANIFEST.csv OUT_states.npz"""
import sys, csv, os, numpy as np
a = np.load(sys.argv[1], allow_pickle=True); b = np.load(sys.argv[2], allow_pickle=True)
order = [os.path.basename(r["path"]) for r in csv.DictReader(open(sys.argv[3]))]
names = list(a["name"]) + list(b["name"]); pos = {n: i for i, n in enumerate(names)}
assert len(pos) == len(names) == len(order) == 1270 and set(order) == set(names)
ix = np.array([pos[n] for n in order])
out = {k: np.concatenate([a[k], b[k]])[ix] for k in ("enc", "proj", "llm", "ans", "label", "spk", "p_yes", "name")}
np.savez_compressed(sys.argv[4], **out); print("merged", {k: v.shape for k, v in out.items()})
