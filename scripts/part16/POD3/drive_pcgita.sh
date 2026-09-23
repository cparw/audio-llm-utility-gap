#!/bin/bash
set -x
export HF_HOME=/workspace/hf_cache
cd /workspace
mkdir -p /workspace/data/pcgita
tar xf /workspace/pcgita.tar -C /workspace/data/pcgita
echo "PCGITA_WAVS $(ls /workspace/data/pcgita/*.wav | wc -l)"
WINDOW_S=30 python3 -u extract_probe_layers.py /workspace/mf_pcgita.csv /workspace/out/q3o_pcgita pd "Qwen/Qwen3-Omni-30B-A3B-Instruct"
echo "EXTRACT_RC $?"
python3 -u perlayer_nested.py /workspace/out/q3o_pcgita_states.npz /workspace/out/q3o_pcgita 5
echo "PERLAYER_RC $?"
echo "PCGITA_ALL_DONE"
