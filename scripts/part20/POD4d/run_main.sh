#!/bin/bash
cd /workspace
export HF_HOME=/workspace/hf PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
unset HF_HUB_ENABLE_HF_TRANSFER
D=/workspace/scores/part20/POD4d
echo "=== MAIN START $(date -u +%H:%M:%S) ===" >> /workspace/logs/DRIVER_POD4d.log
/opt/kimienv/bin/python $D/scripts/kimi_zs.py $D/mf_noinv468.csv $D/kimi_pitt_noinv468_scores.csv > /workspace/logs/main_noinv.log 2>&1
echo "=== MAIN noinv exit $? $(date -u +%H:%M:%S) ===" >> /workspace/logs/DRIVER_POD4d.log
