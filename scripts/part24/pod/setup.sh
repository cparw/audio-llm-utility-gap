#!/bin/bash
# per-pod setup: HF model download first, then E-DAIC fetch, then pip
mkdir -p /workspace/logs /workspace/edaicfull /root/tgz
cd /workspace
export HF_HOME=/workspace/hf
echo "SETUP_START $(date -u +%H:%M:%S)" >> /workspace/logs/DRIVER.log
python3 -m pip install --break-system-packages -q huggingface_hub hf_xet > /workspace/logs/pip0.log 2>&1
( for a in 1 2 3 4 5; do
    python3 -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-Omni-7B'))" >> /workspace/logs/hf_dl.log 2>&1 && { echo "HF_DONE $(date -u +%H:%M:%S)" >> /workspace/logs/DRIVER.log; touch /workspace/HF_DONE; break; }
    echo "HF retry $a" >> /workspace/logs/hf_dl.log; sleep 10; done ) &
( python3 /workspace/fetch_edaic.py ${NW:-6} > /workspace/logs/fetch.log 2>&1; echo "FETCH_EXIT $? $(date -u +%H:%M:%S)" >> /workspace/logs/DRIVER.log; touch /workspace/FETCH_DONE ) &
python3 -m pip install --break-system-packages -q "transformers==5.17.0" accelerate librosa soundfile scikit-learn pandas numpy scipy > /workspace/logs/pip.log 2>&1
python3 -c "import transformers,torch,sklearn,librosa,numpy,scipy,soundfile,accelerate,huggingface_hub as h;print('PIP_OK tf',transformers.__version__,'torch',torch.__version__,'sk',sklearn.__version__,'librosa',librosa.__version__,'np',numpy.__version__,'scipy',scipy.__version__,'sf',soundfile.__version__,'acc',accelerate.__version__,'hub',h.__version__,'cuda',torch.cuda.is_available())" >> /workspace/logs/DRIVER.log 2>&1
touch /workspace/PIP_DONE
wait
