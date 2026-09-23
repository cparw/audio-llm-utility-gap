#!/bin/bash
# waits for fold results, re-scores from saved final checkpoints, then merges + bootstraps
cd /workspace/p20; export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
O=/workspace/scores/part20/POD3
until [ -f $O/ft966_fold0.json ] && [ -f $O/ft966_fold1.json ] && [ -f $O/ft966_fold2.json ]; do sleep 10; done
python3 -u p20_eval.py 0 1 2 >> /workspace/logs/ft966_eval.log 2>&1
until [ -f $O/ft966_fold3.json ] && [ -f $O/ft966_fold4.json ]; do sleep 10; done
python3 -u p20_eval.py 3 4 >> /workspace/logs/ft966_eval.log 2>&1
python3 -u p20_stats.py > /workspace/logs/ft966_stats.log 2>&1 && echo STATS_OK >> /workspace/logs/ft966_stats.log
