import time; t=time.time()
from huggingface_hub import snapshot_download
p = snapshot_download("moonshotai/Kimi-Audio-7B-Instruct", ignore_patterns=["audio_detokenizer/*.pt"])
print("model at", p, f"{time.time()-t:.0f}s", flush=True)
open("/workspace/KIMI_DL_DONE","w").write(p)
