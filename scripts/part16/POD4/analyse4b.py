#!/usr/bin/env python3
"""4b analysis: AUC + speaker-bootstrap CI per arm, PAIRED arm contrasts,
and the conflict/agreement split. Builds the boot spec then calls boot.py."""
import csv, json, os, subprocess, sys
OUT = "/workspace/p16pod4/out"
PUB = "/workspace/p16pod4/pub"
MF  = "/workspace/p16pod4/pitt/mf_orig.csv"

# conflict / agreement membership, keyed by clip basename
sets = {}
for r in csv.DictReader(open(MF)):
    sets[os.path.basename(r["path"])] = r["set"]
conf = [k for k, v in sets.items() if v == "conflict"]
agre = [k for k, v in sets.items() if v == "agreement"]
print(f"conflict {len(conf)}  agreement {len(agre)}")

cells, paired = [], []
arms = {"A_orig": "orig", "B_paronly": "paronly", "C_durmatch": "durmatch"}
for lab, arm in arms.items():
    cells.append(dict(id=f"4b_{lab}_zs",  what=f"Pitt {lab} Omni zero-shot AUC",
                      file=f"{OUT}/p16_pitt_{arm}_zeroshot_scores.csv", col="p_yes"))
    cells.append(dict(id=f"4b_{lab}_enc", what=f"Pitt {lab} Omni encoder nested probe AUC",
                      file=f"{OUT}/p16_pitt_{arm}_enc_nested_oof.csv", col="p_probe"))
cells.append(dict(id="4b_pub_zs",  what="Pitt published Omni zero-shot AUC",
                  file=f"{PUB}/omni_pitt_zeroshot_scores.csv", col="p_yes"))
cells.append(dict(id="4b_pub_enc", what="Pitt published Omni encoder probe AUC",
                  file=f"{PUB}/omni_pitt_enc_nested_oof.csv", col="p_probe"))
# arm split on the interviewer-free cut and on the original
for lab, arm in (("B_paronly", "paronly"), ("A_orig", "orig")):
    for nm, keys in (("conflict", conf), ("agreement", agre)):
        cells.append(dict(id=f"4b_{lab}_zs_{nm}", what=f"Pitt {lab} zero-shot AUC, {nm} arm",
                          file=f"{OUT}/p16_pitt_{arm}_zeroshot_scores.csv", col="p_yes", restrict=keys))
        cells.append(dict(id=f"4b_{lab}_enc_{nm}", what=f"Pitt {lab} encoder probe AUC, {nm} arm",
                          file=f"{OUT}/p16_pitt_{arm}_enc_nested_oof.csv", col="p_probe", restrict=keys))
# paired contrasts
paired.append(dict(id="4b_paired_zs_A_to_B",  what="PAIRED zero-shot: original cut -> interviewer removed",
                   a="4b_A_orig_zs", b="4b_B_paronly_zs"))
paired.append(dict(id="4b_paired_zs_C_to_B",  what="PAIRED zero-shot: duration-matched WITH interviewer -> interviewer removed",
                   a="4b_C_durmatch_zs", b="4b_B_paronly_zs"))
paired.append(dict(id="4b_paired_enc_A_to_B", what="PAIRED encoder probe: original cut -> interviewer removed",
                   a="4b_A_orig_enc", b="4b_B_paronly_enc"))
paired.append(dict(id="4b_paired_enc_C_to_B", what="PAIRED encoder probe: duration-matched WITH interviewer -> interviewer removed",
                   a="4b_C_durmatch_enc", b="4b_B_paronly_enc"))
paired.append(dict(id="4b_paired_zs_A_to_C",  what="PAIRED zero-shot: original 30s -> duration-matched 17.6s (both WITH interviewer)",
                   a="4b_A_orig_zs", b="4b_C_durmatch_zs"))
paired.append(dict(id="4b_paired_enc_A_to_C", what="PAIRED encoder probe: original 30s -> duration-matched 17.6s (both WITH interviewer)",
                   a="4b_A_orig_enc", b="4b_C_durmatch_enc"))

json.dump(dict(cells=cells, paired=paired), open(f"{OUT}/spec4b.json", "w"), indent=1)
print("spec written, cells", len(cells), "paired", len(paired))
