#!/bin/bash
cd /workspace
export HF_HOME=/workspace/hf PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
unset HF_HUB_ENABLE_HF_TRANSFER
D=/workspace/scores/part20/POD4d
/opt/kimienv/bin/python $D/scripts/kimi_diag.py $D/mf_val10.csv $D/diag_val10.npy > /workspace/logs/diag.log 2>&1
echo "=== DIAG exit $? $(date -u +%H:%M:%S) ===" >> /workspace/logs/DRIVER_POD4d.log
