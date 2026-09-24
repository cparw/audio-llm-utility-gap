#!/bin/bash
exec >> /workspace/logs/KIMI_SETUP.log 2>&1
set -x
cd /workspace
export HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONNOUSERSITE=1
echo "=== kimi setup start $(date -u +%H:%M) ==="
[ -d /workspace/kimi_src ] || git clone --recursive --depth 1 https://github.com/MoonshotAI/Kimi-Audio /workspace/kimi_src
python3 -m venv /workspace/kimienv
source /workspace/kimienv/bin/activate
pip -q install --upgrade pip
pip -q install "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128
python -c "import torch,torchaudio;print('torch',torch.__version__,'torchaudio',torchaudio.__version__,'cuda',torch.cuda.is_available())"
pip -q install "transformers==4.51.3" librosa soundfile scikit-learn numpy pandas huggingface_hub hf_transfer sentencepiece accelerate tiktoken blobfile loguru
B=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1
pip -q install "$B/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
python -c "import flash_attn;print('flash_attn',flash_attn.__version__,'OK')"
[ -f /workspace/kimi_src/requirements.txt ] && pip -q install -r /workspace/kimi_src/requirements.txt 2>&1 | tail -2
python - <<'PY'
import re
p="/workspace/kimi_src/kimia_infer/api/kimia.py"
s=open(p).read(); o=s
s=re.sub(r"^(from .*detokenizer.* import .*)$", r"# LAZY: \1", s, flags=re.M)
s=re.sub(r"^(import .*detokenizer.*)$", r"# LAZY: \1", s, flags=re.M)
if s!=o: open(p,"w").write(s); print("patched kimia.py lazy detokenizer")
else: print("no module-level detokenizer import found")
PY
python - <<'PY'
from huggingface_hub import snapshot_download
print("model at", snapshot_download("moonshotai/Kimi-Audio-7B-Instruct"))
PY
python -c "import sys;sys.path.insert(0,'/workspace/kimi_src');from kimia_infer.api.kimia import KimiAudio;print('KIMI_IMPORT_OK')"
echo "=== kimi setup done $(date -u +%H:%M) ==="
touch /workspace/KIMI_SETUP_DONE
