#!/bin/bash
cd /workspace
export HF_HOME=/workspace/hf PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
unset HF_HUB_ENABLE_HF_TRANSFER
D=/workspace/scores/part20/POD4d
echo "=== ORIG START $(date -u +%H:%M:%S) ===" >> /workspace/logs/DRIVER_POD4d.log
/opt/kimienv/bin/python $D/scripts/kimi_zs.py $D/mf_orig468.csv $D/kimi_pitt_orig468_pod_scores.csv > /workspace/logs/main_orig.log 2>&1
echo "=== ORIG exit $? $(date -u +%H:%M:%S) ===" >> /workspace/logs/DRIVER_POD4d.log
