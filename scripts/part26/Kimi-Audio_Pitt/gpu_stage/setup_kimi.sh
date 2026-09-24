#!/bin/bash
exec >> /workspace/logs/KIMI_SETUP.log 2>&1
set -x
cd /workspace
export HF_HOME=/workspace/hf PYTHONNOUSERSITE=1
unset HF_HUB_ENABLE_HF_TRANSFER
echo "=== kimi setup start $(date -u +%H:%M:%S) ==="
python3 -m venv /opt/kimienv
source /opt/kimienv/bin/activate
pip install --upgrade pip
pip install huggingface_hub hf_xet
python -c "import huggingface_hub;print('hfhub',huggingface_hub.__version__)"
# model download in parallel with the rest of setup
nohup python - > /workspace/logs/KIMI_DL.log 2>&1 <<'PY' &
import time; t=time.time()
from huggingface_hub import snapshot_download
print("model at", snapshot_download("moonshotai/Kimi-Audio-7B-Instruct"), f"{time.time()-t:.0f}s", flush=True)
open("/workspace/KIMI_DL_DONE","w").write("ok")
PY
[ -d /workspace/kimi_src ] || git clone --recursive --depth 1 https://github.com/MoonshotAI/Kimi-Audio /workspace/kimi_src
( cd /workspace/kimi_src && git log -1 --format=%H && git submodule status --recursive )
pip install "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128
python -c "import torch,torchaudio;print('torch',torch.__version__,'torchaudio',torchaudio.__version__,'cuda',torch.cuda.is_available())"
pip install "transformers==4.51.3" librosa soundfile scikit-learn scipy numpy pandas hf_transfer sentencepiece accelerate tiktoken blobfile loguru
B=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1
pip install "$B/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
python -c "import flash_attn;print('flash_attn',flash_attn.__version__,'OK')"
# requirements.txt NOT installed: in the working run (edaic_conflict_new/kimi) pip -r failed atomically on a flash_attn source build, so none of it was installed there.
python - <<'PY'
import re
p="/workspace/kimi_src/kimia_infer/api/kimia.py"
s=open(p).read(); o=s
s=re.sub(r"^(from .*detokenizer.* import .*)$", r"# LAZY: \1", s, flags=re.M)
s=re.sub(r"^(import .*detokenizer.*)$", r"# LAZY: \1", s, flags=re.M)
if s!=o: open(p,"w").write(s); print("patched kimia.py lazy detokenizer")
else: print("no module-level detokenizer import found")
PY
python -c "import sys;sys.path.insert(0,'/workspace/kimi_src');from kimia_infer.api.kimia import KimiAudio;print('KIMI_IMPORT_OK')"
pip freeze > /workspace/logs/kimienv_freeze.txt
python -c "import torch,transformers,flash_attn,numpy,scipy,sklearn;print('VERS torch',torch.__version__,'tf',transformers.__version__,'fa',flash_attn.__version__,'np',numpy.__version__,'scipy',scipy.__version__,'sk',sklearn.__version__)"
echo "=== kimi pip done $(date -u +%H:%M:%S) ==="
touch /workspace/KIMI_PIP_DONE
