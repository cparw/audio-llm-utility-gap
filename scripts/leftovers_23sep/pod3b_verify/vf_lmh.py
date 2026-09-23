# Verifier lm_head loader (own code): HTTP range reads of the safetensors header and of the one tensor,
# parallel chunks, sha256 of the raw bytes, bf16 -> float32 by bit shift. Caches float32 npy + meta json.
import os, json, struct, hashlib, urllib.request, numpy as np
from concurrent.futures import ThreadPoolExecutor
REPO="Qwen/Qwen3-Omni-30B-A3B-Instruct"; REV="26291f793822fb6be9555850f06dfe95f2d7e695"
SHARD="model-00013-of-00015.safetensors"; TNAME="thinker.lm_head.weight"
URL=f"https://huggingface.co/{REPO}/resolve/{REV}/{SHARD}"
def _get(a,b,tries=6):
    for t in range(tries):
        try:
            req=urllib.request.Request(URL,headers={"Range":f"bytes={a}-{b}","User-Agent":"vf-pod3b/1.0"})
            with urllib.request.urlopen(req,timeout=120) as r:
                data=r.read()
            if len(data)==b-a+1: return data
        except Exception as e:
            err=e
    raise RuntimeError(f"range {a}-{b} failed")
def load(cache="/workspace/lmh"):
    os.makedirs(cache,exist_ok=True)
    npy=f"{cache}/W_bf16_as_f32.npy"; meta=f"{cache}/meta.json"
    if os.path.exists(npy) and os.path.exists(meta):
        return np.load(npy,mmap_mode="r"), json.load(open(meta))
    L=struct.unpack("<Q",_get(0,7))[0]
    hdr=json.loads(_get(8,8+L-1)); t=hdr[TNAME]; s,e=t["data_offsets"]; base=8+L
    n=e-s; step=16*1024*1024; parts=[(base+s+i, base+s+min(i+step,n)-1) for i in range(0,n,step)]
    with ThreadPoolExecutor(16) as ex: chunks=list(ex.map(lambda p:_get(*p),parts))
    raw=b"".join(chunks); assert len(raw)==n
    sha=hashlib.sha256(raw).hexdigest()
    u=np.frombuffer(raw,dtype="<u2").astype(np.uint32)<<16
    W=u.view(np.float32).reshape(t["shape"])
    np.save(npy,W)
    m={"repo":REPO,"rev":REV,"shard":SHARD,"tensor":TNAME,"dtype":t["dtype"],"shape":t["shape"],"data_offsets":[s,e],
       "header_len":L,"sha256_bf16_bytes":sha,"nbytes":n}
    json.dump(m,open(meta,"w"),indent=1)
    return np.load(npy,mmap_mode="r"), m
