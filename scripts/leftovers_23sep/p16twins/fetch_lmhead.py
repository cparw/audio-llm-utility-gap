"""Fetch the Qwen3-Omni-30B-A3B-Instruct thinker lm_head from the Hugging Face hub with HTTP range reads
(only the header of the shard and the bytes of that one tensor), convert bf16 -> float32 exactly, save npy."""
import json, struct, sys, time, hashlib
import numpy as np, requests
from huggingface_hub import hf_hub_download, hf_hub_url, HfApi
MID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
t0 = time.time()
info = HfApi().model_info(MID)
rev = info.sha
idx = json.load(open(hf_hub_download(MID, "model.safetensors.index.json", revision=rev)))
keys = [k for k in idx["weight_map"] if "lm_head" in k]
print("revision", rev, "lm_head keys", keys, flush=True)
key = "thinker.lm_head.weight" if "thinker.lm_head.weight" in keys else keys[0]
shard = idx["weight_map"][key]
url = hf_hub_url(MID, shard, revision=rev)
s = requests.Session()
h = s.get(url, headers={"Range": "bytes=0-7"}, allow_redirects=True, timeout=60)
hlen = struct.unpack("<Q", h.content[:8])[0]
hdr = json.loads(s.get(url, headers={"Range": f"bytes=8-{8+hlen-1}"}, allow_redirects=True, timeout=60).content)
meta = hdr[key]
a, b = meta["data_offsets"]
start = 8 + hlen + a; end = 8 + hlen + b - 1
print("shard", shard, "dtype", meta["dtype"], "shape", meta["shape"], "bytes", b - a, flush=True)
buf = bytearray()
CH = 64 << 20
pos = start
while pos <= end:
    e = min(end, pos + CH - 1)
    for attempt in range(5):
        try:
            r = s.get(url, headers={"Range": f"bytes={pos}-{e}"}, allow_redirects=True, timeout=300)
            assert r.status_code == 206 and len(r.content) == e - pos + 1, (r.status_code, len(r.content))
            break
        except Exception as ex:
            print("retry", attempt, ex, flush=True); time.sleep(3)
    buf += r.content; pos = e + 1
assert meta["dtype"] == "BF16", meta["dtype"]
u16 = np.frombuffer(bytes(buf), dtype="<u2").reshape(meta["shape"])
W = (u16.astype(np.uint32) << 16).view(np.float32)
np.save("/workspace/in/q3o_lm_head.npy", W)
json.dump({"model": MID, "revision": rev, "key": key, "shard": shard, "dtype": meta["dtype"], "shape": meta["shape"],
           "sha256_raw_bf16": hashlib.sha256(bytes(buf)).hexdigest(), "seconds": round(time.time() - t0, 1)},
          open("/workspace/out/lm_head_source.json", "w"), indent=1)
print("saved", W.shape, W.dtype, round(time.time() - t0, 1), "s", flush=True)
