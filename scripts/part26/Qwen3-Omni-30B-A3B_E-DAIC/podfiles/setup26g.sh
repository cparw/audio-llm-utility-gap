#!/bin/bash
# PART 26 Qwen3-Omni E-DAIC GPU pod setup: the podD2 library set (transformers 5.5.4, the version behind q3o_new and the
# lost podD2 E-DAIC states), then the Qwen3-Omni-30B-A3B-Instruct download. Markers PIP_DONE, HF_DONE.
cd /workspace; mkdir -p logs out data
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
PIP="python3 -m pip install --break-system-packages -q"
{ $PIP "transformers==5.5.4" accelerate librosa soundfile scikit-learn pandas joblib huggingface_hub \
  && python3 -c "import transformers,torch,sklearn,numpy,scipy,librosa,platform;print('VERSIONS tf',transformers.__version__,'torch',torch.__version__,'sk',sklearn.__version__,'np',numpy.__version__,'sp',scipy.__version__,'librosa',librosa.__version__,'py',platform.python_version(),platform.platform(),'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" \
  && touch /workspace/PIP_DONE; } > logs/pip.log 2>&1
python3 -c "from huggingface_hub import snapshot_download; p=snapshot_download('Qwen/Qwen3-Omni-30B-A3B-Instruct'); print('DL_OK',p)" > logs/hf.log 2>&1 && touch /workspace/HF_DONE
echo "SETUP_END $(date -u +%H:%M:%S) PIP=$([ -f PIP_DONE ] && echo ok) HF=$([ -f HF_DONE ] && echo ok)" >> logs/DRIVER.log
