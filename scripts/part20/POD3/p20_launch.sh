#!/bin/bash
# three GPU workers, folds run in parallel (each process = one fold, resumable from its per-epoch checkpoint)
cd /workspace/p20
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
O=/workspace/scores/part20/POD3; M=/workspace/p20/p14_ft966_manifest.csv
mkdir -p $O /workspace/logs
( for f in 0 3; do python3 -u p20_ft966.py $M $O $f >> /workspace/logs/ft966_w0.log 2>&1; done; echo W0_END >> /workspace/logs/ft966_w0.log ) &
( for f in 1 4; do python3 -u p20_ft966.py $M $O $f >> /workspace/logs/ft966_w1.log 2>&1; done; echo W1_END >> /workspace/logs/ft966_w1.log ) &
( for f in 2; do python3 -u p20_ft966.py $M $O $f >> /workspace/logs/ft966_w2.log 2>&1; done; echo W2_END >> /workspace/logs/ft966_w2.log ) &
wait
