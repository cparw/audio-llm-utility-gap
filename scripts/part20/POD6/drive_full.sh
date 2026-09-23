#!/bin/bash
# PART20 POD6 lower-priority check: same pipeline on the full 1270 NeuroVoz set (extract the 363 non-subset clips, merge, probe).
cd /workspace/p20
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 WINDOW_S=30
O=/workspace/scores/part20/POD6/B_neurovoz_full_check; mkdir -p $O
python3 - <<PY
import csv
sub={r["path"] for r in csv.DictReader(open("/workspace/p20/mf_nv_matched.csv"))}
rows=list(csv.DictReader(open("/workspace/p20/mf_nv_full1270.csv")))
w=csv.DictWriter(open("/workspace/p20/mf_nv_rest363.csv","w",newline=""),fieldnames=["path","label","speaker"]); w.writeheader()
n=0
for r in rows:
    if r["path"] not in sub: w.writerow(r); n+=1
print("rest rows",n)
PY
[ -f $O/omni_nvrest_states.npz ] || python3 extract_probe_layers.py /workspace/p20/mf_nv_rest363.csv $O/omni_nvrest pd "Qwen/Qwen2.5-Omni-7B" || { echo EXTRACT_FAILED; exit 1; }
python3 merge_full.py /workspace/scores/part20/POD6/B_neurovoz_matched/omni_nvmatched_states.npz $O/omni_nvrest_states.npz /workspace/p20/mf_nv_full1270.csv $O/omni_nvfull_states.npz || { echo MERGE_FAILED; exit 1; }
python3 b_nested5.py $O/omni_nvfull_states.npz $O/omni_nvfull enc || { echo PROBE_FAILED; exit 1; }
echo "FULL DONE $(date -u +%H:%M:%S)"
