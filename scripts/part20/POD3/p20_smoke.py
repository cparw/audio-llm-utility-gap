import os, glob, time, torch, librosa, numpy as np, copy
t0=time.time()
from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
MID="Qwen/Qwen2.5-Omni-7B"; DT=torch.bfloat16; dev="cuda"
P="Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
proc=Proc.from_pretrained(MID); full=Model.from_pretrained(MID, dtype=DT); net=full.thinker
for a in ("talker","token2wav"):
    if hasattr(full,a): delattr(full,a)
if hasattr(net,"visual"): del net.visual
net=net.to(dev); proj=net.audio_tower.proj; tok=proc.tokenizer
print("proj", proj, "load s", round(time.time()-t0), "mem GB", torch.cuda.memory_allocated()/1e9, flush=True)
_orig=proj.forward
def _f(x): return _orig(x.float()).to(DT)
proj.forward=_f
for p in net.parameters(): p.requires_grad=False
for p in proj.parameters(): p.data=p.data.float(); p.requires_grad=True
opt=torch.optim.AdamW([p for p in proj.parameters()], lr=1e-4)
YES1=tok(" Yes",add_special_tokens=False).input_ids[0]; NO1=tok(" No",add_special_tokens=False).input_ids[0]
fs=sorted(glob.glob("/workspace/p20/part14_clips/*.wav"))[:6]
net.train(); torch.cuda.reset_peak_memory_stats()
for k,f in enumerate(fs):
    x,_=librosa.load(f,sr=16000); x=x[:16000*30]
    conv=[{"role":"user","content":[{"type":"audio","audio":x},{"type":"text","text":P}]}]
    text=proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp=proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True); inp.pop("use_audio_in_video",None)
    if k==0: print({kk:(tuple(v.shape),v.dtype) for kk,v in inp.items() if torch.is_tensor(v)}, flush=True)
    inp={kk:(v.to(dev) if hasattr(v,"to") else v) for kk,v in inp.items()}
    for kk,v in list(inp.items()):
        if torch.is_tensor(v) and v.dtype==torch.float32: inp[kk]=v.to(DT)
    torch.cuda.synchronize(); t1=time.time()
    out=net(**inp); two=out.logits[0,-1,[YES1,NO1]].float()
    loss=torch.nn.functional.cross_entropy(two.unsqueeze(0), torch.tensor([1],device=dev)); loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); print(k, "step s", round(time.time()-t1,3), "loss", float(loss), "grad ok", "peak GB", round(torch.cuda.max_memory_allocated()/1e9,2), flush=True)
