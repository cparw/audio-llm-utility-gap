#!/bin/bash
# POD4B: one corpus, three arms. Zero-shot + encoder nested probe per arm.
set -x
CORPUS=$1
cd /workspace/p16pod4b
export HF_HOME=/workspace/hf
export PYTHONUNBUFFERED=1

echo "=== CUT $CORPUS ==="
python3 cut_arms4b.py $CORPUS 2>&1 || exit 1

for ARM in orig paronly durmatch; do
  echo "=== EXTRACT $CORPUS $ARM ==="
  WINDOW_S=30 python3 extract_probe_layers.py \
     /workspace/p16pod4b/mf_${CORPUS}_${ARM}.csv \
     /workspace/p16pod4b/out/p16_${CORPUS}_${ARM} \
     ad "Qwen/Qwen2.5-Omni-7B" 2>&1 || exit 1
  echo "=== PROBE $CORPUS $ARM ==="
  python3 nested_oof.py \
     /workspace/p16pod4b/out/p16_${CORPUS}_${ARM}_states.npz \
     /workspace/p16pod4b/out/p16_${CORPUS}_${ARM} enc 2>&1 || exit 1
done
echo "ALL4B DONE $CORPUS"
