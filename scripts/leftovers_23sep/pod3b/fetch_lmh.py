"""Fetch ONLY the thinker.lm_head.weight tensor of Qwen/Qwen3-Omni-30B-A3B-Instruct (pinned commit)
by HTTP range reads of shard model-00013-of-00015.safetensors. Parses the safetensors header itself,
converts BF16 -> float32 exactly (bits << 16), saves /workspace/lmh/q3o_lm_head.npy (full matrix, not pulled)
and out/lmh_fetch.json (offsets, sha256 of the raw tensor bytes)."""
import json, struct, hashlib, subprocess, numpy as np, os, time
REPO="Qwen/Qwen3-Omni-30B-A3B-Instruct"; REV="26291f793822fb6be9555850f06dfe95f2d7e695"
SHARD="model-00013-of-00015.safetensors"; KEY="thinker.lm_head.weight"
U=f"https://huggingface.co/{REPO}/resolve/{REV}/{SHARD}"
t0=time.time()
def rng(a,b,path):
    subprocess.run(["curl","-sSL","--retry","5","-r",f"{a}-{b}",U,"-o",path],check=True)
rng(0,7,"/workspace/lmh/h8"); n=struct.unpack("<Q",open("/workspace/lmh/h8","rb").read())[0]
rng(8,8+n-1,"/workspace/lmh/hdr"); hdr=json.loads(open("/workspace/lmh/hdr","rb").read())
e=hdr[KEY]; assert e["dtype"]=="BF16"; shp=e["shape"]; a,b=e["data_offsets"]
base=8+n; rng(base+a,base+b-1,"/workspace/lmh/lm_head.bf16")
raw=open("/workspace/lmh/lm_head.bf16","rb").read(); assert len(raw)==b-a==shp[0]*shp[1]*2
sha=hashlib.sha256(raw).hexdigest()
u16=np.frombuffer(raw,dtype="<u2").reshape(shp)
W=(u16.astype(np.uint32)<<16).view(np.float32)
np.save("/workspace/lmh/q3o_lm_head.npy",W)
info={"repo":REPO,"revision":REV,"shard":SHARD,"tensor":KEY,"dtype":"BF16","shape":shp,"header_len":n,
      "data_offsets":[a,b],"abs_byte_range":[base+a,base+b-1],"sha256_raw_bf16_bytes":sha,
      "float32_npy":"/workspace/lmh/q3o_lm_head.npy (bf16 bits shifted left 16, exact)","seconds":round(time.time()-t0,1)}
json.dump(info,open("/workspace/out/lmh_fetch.json","w"),indent=1); print(json.dumps(info))
