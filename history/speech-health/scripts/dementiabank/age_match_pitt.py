#!/usr/bin/env python3
"""Age and sex matched Pitt subset, ADReSS style.
Cookie task only, dementia arm restricted to Probable/Possible AD.
Greedy nearest-age matching within sex, 1:1, caliper 3 years."""
import csv
from collections import defaultdict
import statistics as st

rows = list(csv.DictReader(open("pitt_manifest.csv")))
cookie = [r for r in rows if r["task"]=="cookie" and str(r["age"]).isdigit()]
spk = {}
for r in cookie:
    key=(r["group"], r["speaker_id"])
    if key not in spk: spk[key]=r
ctrl=[r for r in spk.values() if r["group"]=="Control"]
ad  =[r for r in spk.values() if r["group"]=="Dementia" and r["diagnosis"] in ("ProbableAD","PossibleAD")]
print(f"eligible speakers: control {len(ctrl)}, probable/possible AD {len(ad)}")

pairs=[]
used=set()
for sex in ("male","female"):
    A=sorted([r for r in ad if r["sex"]==sex], key=lambda r:int(r["age"]))
    C=sorted([r for r in ctrl if r["sex"]==sex], key=lambda r:int(r["age"]))
    for a in A:
        best=None; bd=99
        for c in C:
            if c["speaker_id"] in used: continue
            d=abs(int(a["age"])-int(c["age"]))
            if d<bd: bd, best = d, c
        if best is not None and bd<=3:
            pairs.append((a,best,bd)); used.add(best["speaker_id"])
ma=[int(a["age"]) for a,_,_ in pairs]; mc=[int(c["age"]) for _,c,_ in pairs]
sexes=[a["sex"] for a,_,_ in pairs]
print(f"matched pairs: {len(pairs)}  (male {sexes.count('male')}, female {sexes.count('female')})")
print(f"  age AD {st.mean(ma):.1f} vs control {st.mean(mc):.1f}  gap {st.mean(ma)-st.mean(mc):+.1f}")
keep=set()
for a,c,_ in pairs:
    keep.add(("Dementia",a["speaker_id"])); keep.add(("Control",c["speaker_id"]))
out=[r for r in cookie if (r["group"],r["speaker_id"]) in keep]
with open("pitt_agematched_manifest.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print(f"recordings in matched manifest: {len(out)}")
