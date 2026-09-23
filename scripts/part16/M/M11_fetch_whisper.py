import os, time
os.environ["HF_HOME"] = "<local data dir>/DementiaBank/hf_cache"
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download("openai/whisper-large-v3",
                      allow_patterns=["*.json","*.txt","model.safetensors","preprocessor_config.json","*.model"])
print("PATH", p)
print("MIN %.1f" % ((time.time()-t0)/60))
for f in sorted(os.listdir(p)):
    print(f, os.path.getsize(os.path.join(p,f)))
