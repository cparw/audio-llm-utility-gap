#!/bin/bash
# PART20 POD6 job B driver: extract Omni encoder states for the NeuroVoz matched subset, then nested five-repeat probe.
cd /workspace/p20
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 WINDOW_S=30
O=/workspace/scores/part20/POD6/B_neurovoz_matched
mkdir -p $O
echo "EXTRACT START $(date -u +%H:%M:%S)"
if [ ! -f $O/omni_nvmatched_states.npz ]; then
  python3 extract_probe_layers.py /workspace/p20/mf_nv_matched.csv $O/omni_nvmatched pd "Qwen/Qwen2.5-Omni-7B" || { echo EXTRACT_FAILED; exit 1; }
fi
echo "EXTRACT DONE $(date -u +%H:%M:%S)"
python3 b_nested5.py $O/omni_nvmatched_states.npz $O/omni_nvmatched enc || { echo PROBE_FAILED; exit 1; }
echo "B DONE $(date -u +%H:%M:%S)"
