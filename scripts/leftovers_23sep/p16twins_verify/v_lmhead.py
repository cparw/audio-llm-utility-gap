"""Verifier: read thinker.lm_head.weight of Qwen3-Omni-30B-A3B-Instruct straight from the HF hub with HTTP range
requests (safetensors header -> byte offsets), convert BF16 -> float32 exactly, save /workspace/in/lmh.npy."""
import json, struct, time, hashlib, urllib.request, sys
import numpy as np
REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
t0 = time.time()
api = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/revision/main"))
rev = api["sha"]
base = f"https://huggingface.co/{REPO}/resolve/{rev}/"
idx = json.load(urllib.request.urlopen(base + "model.safetensors.index.json"))
key = "thinker.lm_head.weight"
shard = idx["weight_map"][key]
def rng_get(a, b):
    req = urllib.request.Request(base + shard, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req).read()
hlen = struct.unpack("<Q", rng_get(0, 7))[0]
hdr = json.loads(rng_get(8, 8 + hlen - 1))
e = hdr[key]; s0, s1 = e["data_offsets"]; start = 8 + hlen
print("rev", rev, "shard", shard, e["dtype"], e["shape"], "bytes", s1 - s0, flush=True)
# parallel range reads in 16 pieces
from concurrent.futures import ThreadPoolExecutor
n = s1 - s0; k = 16; step = (n + k - 1) // k
pieces = [(start + s0 + i * step, min(start + s0 + (i + 1) * step, start + s1) - 1) for i in range(k)]
with ThreadPoolExecutor(k) as ex:
    parts = list(ex.map(lambda ab: rng_get(*ab), pieces))
raw = b"".join(parts)
assert len(raw) == n, (len(raw), n)
sha = hashlib.sha256(raw).hexdigest()
u16 = np.frombuffer(raw, dtype="<u2").reshape(e["shape"])
W = (u16.astype(np.uint32) << 16).view(np.float32)
np.save("/workspace/in/lmh.npy", W)
json.dump({"repo": REPO, "revision": rev, "shard": shard, "key": key, "dtype": e["dtype"], "shape": e["shape"],
           "sha256_raw_bf16": sha, "seconds": round(time.time() - t0, 1)},
          open("/workspace/out/v_lmhead_source.json", "w"), indent=1)
print("saved", W.shape, W.dtype, "sha", sha, round(time.time() - t0, 1), "s", flush=True)
