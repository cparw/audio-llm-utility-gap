export HF_HOME=/workspace/hf
python3 -m pip install --break-system-packages -q "transformers==5.17.0" accelerate librosa soundfile scikit-learn pandas numpy scipy huggingface_hub hf_transfer > pip.log 2>&1
echo PIPDONE >> pip.log
python3 -c "import sklearn,numpy,transformers,torch,scipy,pandas;print(sklearn.__version__,numpy.__version__,transformers.__version__,torch.__version__,scipy.__version__,pandas.__version__)" >> pip.log 2>&1
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "from huggingface_hub import snapshot_download; p=snapshot_download(\"Qwen/Qwen2.5-Omni-7B\"); print(p)" > dl.log 2>&1
echo DLDONE >> dl.log
