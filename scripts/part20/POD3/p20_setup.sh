#!/bin/bash
mkdir -p /workspace/scores/part20/POD3 /workspace/logs /workspace/p20
exec >> /workspace/logs/p20_setup.log 2>&1
set -x
cd /workspace
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
PIP="python3 -m pip install --break-system-packages -q"
$PIP "transformers>=5.0" accelerate librosa soundfile scikit-learn scipy pandas numpy hf_transfer huggingface_hub sentencepiece
echo "PIP_EXIT $?"
python3 -c "import transformers,torch,sklearn,numpy,librosa;print('OK tf',transformers.__version__,'sk',sklearn.__version__,'np',numpy.__version__,'cuda',torch.cuda.is_available())" || exit 1
touch /workspace/p20/PIP_DONE
python3 -c "from huggingface_hub import snapshot_download; p=snapshot_download('Qwen/Qwen2.5-Omni-7B'); print('DL', p)"
echo "DL_EXIT $?"
touch /workspace/p20/DL_DONE
