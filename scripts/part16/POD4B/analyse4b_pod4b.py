#!/usr/bin/env python3
"""POD4B analysis spec: AUC + speaker-bootstrap CI per arm, PAIRED arm contrasts,
and the PAIRED probe-minus-zero-shot gap under each arm. Feeds boot.py.

usage: analyse4b_pod4b.py CORPUS OUTSPEC
"""
import csv, json, os, sys
CORPUS, SPEC = sys.argv[1], sys.argv[2]
OUT = "/workspace/p16pod4b/out"
PUBD = "/workspace/p16pod4b/pub"

# restrict every cell to the clips that survived the cut, so all arms and the
# published run are scored on exactly the same clip set.
kept = [r["clip"] for r in csv.DictReader(open(f"/workspace/p16pod4b/mf_{CORPUS}_paronly.csv"))]
print(f"{CORPUS}: {len(kept)} clips common to all arms")

cells, paired = [], []
arms = {"A_orig": "orig", "B_paronly": "paronly", "C_durmatch": "durmatch"}
for lab, arm in arms.items():
    cells.append(dict(id=f"4b_{CORPUS}_{lab}_zs", what=f"{CORPUS} {lab} Omni zero-shot AUC",
                      file=f"{OUT}/p16_{CORPUS}_{arm}_zeroshot_scores.csv", col="p_yes", restrict=kept))
    cells.append(dict(id=f"4b_{CORPUS}_{lab}_enc", what=f"{CORPUS} {lab} Omni encoder nested probe AUC",
                      file=f"{OUT}/p16_{CORPUS}_{arm}_enc_nested_oof.csv", col="p_probe", restrict=kept))
# the published run, on the same clip set, as the anchor
cells.append(dict(id=f"4b_{CORPUS}_pub_zs", what=f"{CORPUS} published Omni zero-shot AUC, same clip set",
                  file=f"{PUBD}/omni_{CORPUS}_zeroshot_scores.csv", col="p_yes", restrict=kept))
cells.append(dict(id=f"4b_{CORPUS}_pub_enc", what=f"{CORPUS} published Omni encoder probe AUC, same clip set",
                  file=f"{PUBD}/omni_{CORPUS}_enc_nested_oof.csv", col="p_probe", restrict=kept))

# paired arm contrasts, both streams
for st in ("zs", "enc"):
    nm = "zero-shot" if st == "zs" else "encoder probe"
    paired.append(dict(id=f"4b_{CORPUS}_paired_{st}_A_to_B",
                       what=f"{CORPUS} PAIRED {nm}: original cut -> interviewer removed",
                       a=f"4b_{CORPUS}_A_orig_{st}", b=f"4b_{CORPUS}_B_paronly_{st}", restrict=kept))
    paired.append(dict(id=f"4b_{CORPUS}_paired_{st}_A_to_C",
                       what=f"{CORPUS} PAIRED {nm}: original cut -> duration-matched, interviewer KEPT",
                       a=f"4b_{CORPUS}_A_orig_{st}", b=f"4b_{CORPUS}_C_durmatch_{st}", restrict=kept))
    paired.append(dict(id=f"4b_{CORPUS}_paired_{st}_C_to_B",
                       what=f"{CORPUS} PAIRED {nm}: duration-matched WITH interviewer -> interviewer removed",
                       a=f"4b_{CORPUS}_C_durmatch_{st}", b=f"4b_{CORPUS}_B_paronly_{st}", restrict=kept))

# the gap, probe minus zero-shot, under each arm -- paired inside one speaker draw
for lab in arms:
    paired.append(dict(id=f"4b_{CORPUS}_gap_{lab}",
                       what=f"{CORPUS} {lab} GAP = encoder probe AUC minus zero-shot AUC",
                       a=f"4b_{CORPUS}_{lab}_zs", b=f"4b_{CORPUS}_{lab}_enc", restrict=kept))
paired.append(dict(id=f"4b_{CORPUS}_gap_pub",
                   what=f"{CORPUS} published GAP = encoder probe AUC minus zero-shot AUC, same clip set",
                   a=f"4b_{CORPUS}_pub_zs", b=f"4b_{CORPUS}_pub_enc", restrict=kept))

json.dump(dict(cells=cells, paired=paired), open(SPEC, "w"), indent=1)
print(f"spec written {SPEC}: cells {len(cells)} paired {len(paired)}")
