#!/usr/bin/env python3
"""Manifest for the Pitt corpus from the CHAT transcript headers.
One row per recording: speaker, session, task, diagnosis, age, sex, MMSE,
expected media path, whether the audio is on disk yet, and how much of the
transcript is the investigator speaking."""
import os, re, csv, glob

T = "transcripts"; A = "Pitt"
rows = []
for cha in sorted(glob.glob(f"{T}/*/*/*.cha")):
    parts = cha.split(os.sep)          # transcripts/Group/task/NNN-V.cha
    group_dir, task, base = parts[1], parts[2], os.path.basename(cha)[:-4]
    spk, sess = base.split("-", 1)
    txt = open(cha, encoding="utf-8", errors="ignore").read()
    m = re.search(r"@ID:\s*eng\|Pitt\|PAR\|(\d*);?[\d.]*\|(\w*)\|(\w*)\|[^|]*\|Participant\|(\d*)\|", txt)
    age, sex, dx, mmse = (m.groups() if m else ("","","",""))
    # utterance share: investigator vs participant, by count and by timed duration
    par_ms = inv_ms = 0
    for who, line in re.findall(r"\*(PAR|INV):\t.*?(\d+)_(\d+)\s*$", txt, re.M):
        pass  # placeholder, replaced below
    for mm in re.finditer(r"\*(PAR|INV):.*?(\d+)_(\d+)\s*$", txt, re.M):
        d = int(mm.group(3)) - int(mm.group(2))
        if mm.group(1) == "PAR": par_ms += d
        else: inv_ms += d
    tot = par_ms + inv_ms
    media = f"{A}/{group_dir}/{task}/{base}.mp3"
    rows.append(dict(
        filepath=media, exists=int(os.path.exists(media)),
        speaker_id=spk, session=sess, task=task,
        group=group_dir, diagnosis=dx, label=1 if group_dir=="Dementia" else 0,
        age=age, sex=sex, mmse=mmse,
        par_sec=round(par_ms/1000,1), inv_sec=round(inv_ms/1000,1),
        inv_share=round(inv_ms/tot,3) if tot else "",
    ))
with open("pitt_manifest.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

from collections import Counter
print(f"rows: {len(rows)}")
print("by group/task:", dict(Counter((r['group'],r['task']) for r in rows)))
print("diagnoses:", dict(Counter(r['diagnosis'] for r in rows)))
spk = {}
for r in rows: spk.setdefault((r['group'],r['speaker_id']), r)
print(f"unique speakers: {len(spk)}  by group: {dict(Counter(g for g,_ in spk))}")
ages = [int(r['age']) for r in spk.values() if str(r['age']).isdigit()]
for g in ('Control','Dementia'):
    a=[int(r['age']) for (gg,_),r in spk.items() if gg==g and str(r['age']).isdigit()]
    mm=[int(r['mmse']) for (gg,_),r in spk.items() if gg==g and str(r['mmse']).isdigit()]
    import statistics as st
    print(f"{g}: n={len(a)}, age mean {st.mean(a):.1f} range {min(a)}-{max(a)}, mmse mean {st.mean(mm):.1f}" if a else g)
