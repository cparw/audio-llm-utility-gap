#!/bin/bash
# PART25 A retry: the ORIGINAL omni-final pod environment (18 Sep, pod gv9jvm57b21mcz): image
# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, H200, transformers==5.5.4 (podA_logs/setup_out.log). The prior PART25 A
# rerun used transformers 5.17.0 and matched no zero-shot score exactly. Fit libraries pinned to the pod-matching versions.
cd /workspace; mkdir -p logs out data
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=1
PIP="python3 -m pip install --break-system-packages -q"
{ $PIP "transformers==5.5.4" accelerate librosa soundfile "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" joblib pandas hf_transfer \
  && python3 -c "import transformers,torch,sklearn,numpy,scipy,librosa,joblib,platform;print('VERSIONS tf',transformers.__version__,'torch',torch.__version__,'sk',sklearn.__version__,'np',numpy.__version__,'sp',scipy.__version__,'librosa',librosa.__version__,'joblib',joblib.__version__,'py',platform.python_version(),platform.platform(),'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" \
  && touch /workspace/PIP_DONE; } > logs/pip.log 2>&1
python3 -m pip freeze > logs/pip_freeze.txt 2>&1
{ nvidia-smi; lscpu; nproc; } > logs/hw.txt 2>&1
python3 -c "from huggingface_hub import snapshot_download; p=snapshot_download('Qwen/Qwen2.5-Omni-7B'); print('DL_OK',p)" > logs/hf.log 2>&1 && touch /workspace/HF_DONE
echo "SETUP_END $(date -u +%H:%M:%S) PIP=$([ -f PIP_DONE ] && echo ok) HF=$([ -f HF_DONE ] && echo ok)" >> logs/DRIVER.log
