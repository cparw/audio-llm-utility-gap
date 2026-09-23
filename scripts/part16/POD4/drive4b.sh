#!/bin/bash
# POD4 4b: Pitt with the interviewer removed.
set -x
cd /workspace/p16pod4
export HF_HOME=/workspace/hf
export PYTHONUNBUFFERED=1

echo "=== CUT ==="
python3 cut_arms.py 2>&1 || exit 1

for ARM in orig paronly durmatch; do
  echo "=== EXTRACT $ARM ==="
  WINDOW_S=30 python3 extract_probe_layers.py \
     /workspace/p16pod4/pitt/mf_${ARM}.csv \
     /workspace/p16pod4/out/p16_pitt_${ARM} \
     ad "Qwen/Qwen2.5-Omni-7B" 2>&1
  echo "=== PROBE $ARM ==="
  python3 nested_oof.py \
     /workspace/p16pod4/out/p16_pitt_${ARM}_states.npz \
     /workspace/p16pod4/out/p16_pitt_${ARM} enc 2>&1
done
echo "ALL4B DONE"
