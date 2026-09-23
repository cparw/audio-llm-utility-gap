#!/bin/bash
# POD4 4a: channel normalisation. $1 = tag (nv|pcgita), $2 = manifest path
set -x
cd /workspace/p16pod4
export HF_HOME=/workspace/hf
export PYTHONUNBUFFERED=1
TAG=$1; MF=$2
D=/workspace/p16pod4/${TAG}

echo "=== CHANNEL FEATURES + NORMALISE $TAG ==="
python3 chan4a.py "$MF" "$D" "$TAG" 2>&1 || exit 1

echo "=== EXTRACT BEFORE $TAG ==="
WINDOW_S=30 python3 extract_probe_layers.py "$MF" \
   /workspace/p16pod4/out/p16_${TAG}_before pd "Qwen/Qwen2.5-Omni-7B" 2>&1
echo "=== PROBE BEFORE $TAG ==="
python3 nested_oof.py /workspace/p16pod4/out/p16_${TAG}_before_states.npz \
   /workspace/p16pod4/out/p16_${TAG}_before enc 2>&1

echo "=== EXTRACT AFTER $TAG ==="
WINDOW_S=30 python3 extract_probe_layers.py "$D/mf_${TAG}_norm.csv" \
   /workspace/p16pod4/out/p16_${TAG}_after pd "Qwen/Qwen2.5-Omni-7B" 2>&1
echo "=== PROBE AFTER $TAG ==="
python3 nested_oof.py /workspace/p16pod4/out/p16_${TAG}_after_states.npz \
   /workspace/p16pod4/out/p16_${TAG}_after enc 2>&1

echo "ALL4A_${TAG} DONE"
