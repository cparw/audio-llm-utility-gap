import os, time, torch, numpy as np
from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
MID="Qwen/Qwen2.5-Omni-7B"; DT=torch.bfloat16; dev="cuda"
P="Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
t0=time.time(); proc=Proc.from_pretrained(MID); full=Model.from_pretrained(MID, dtype=DT); net=full.thinker
for a in ("talker","token2wav"):
    if hasattr(full,a): delattr(full,a)
if hasattr(net,"visual"): del net.visual
net=net.to(dev); print("load s", round(time.time()-t0), "text model", type(net.model).__name__, "attn", net.config._attn_implementation, "fe", proc.feature_extractor.n_samples, flush=True)
proj=net.audio_tower.proj; _o=proj.forward; proj.forward=lambda x: _o(x.float()).to(DT)
for p in net.parameters(): p.requires_grad=False
for p in proj.parameters(): p.data=p.data.float(); p.requires_grad=True
AUD=net.config.audio_token_id if hasattr(net.config,"audio_token_id") else net.config.audio_token_index
tok=proc.tokenizer; YES=tok(" Yes",add_special_tokens=False).input_ids[0]; NO=tok(" No",add_special_tokens=False).input_ids[0]
def inp(x, trunc):
    conv=[{"role":"user","content":[{"type":"audio","audio":x},{"type":"text","text":P}]}]
    text=proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    kw={} if trunc else {"truncation":False}
    i=proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, **kw)
    i={k:(v.to(dev) if hasattr(v,"to") else v) for k,v in i.items()}; i.pop("use_audio_in_video",None); return i
x=(np.random.RandomState(0).randn(16000*900)*0.05).astype(np.float32)
for trunc in (True, False):
    i=inp(x,trunc); print("900s trunc" if trunc else "900s truncation=False", "audio tok", int((i["input_ids"][0]==AUD).sum()), "seq", i["input_ids"].shape[1], "feat", tuple(i["input_features"].shape), i["input_features"].dtype, flush=True)
# fp32 features vs bf16-cast features (the 40730 script cast; p15 does not): compare logits on 30 s
x30=x[:16000*30]; net.eval()
with torch.no_grad():
    a=inp(x30,True); la=net(**a).logits[0,-1].float()
    b=dict(a); b["input_features"]=b["input_features"].to(DT); lb=net(**b).logits[0,-1].float()
print("fp32 vs bf16 features: max|dlogit|", float((la-lb).abs().max()), "p_yes", float(torch.softmax(la[[YES,NO]],-1)[0]), float(torch.softmax(lb[[YES,NO]],-1)[0]), flush=True)
net.train()
for gc in (False, True):
    if gc: net.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        for rep in range(2):
            ts=time.time(); i=inp(x,False); out=net(**i); two=out.logits[0,-1,[YES,NO]].float()
            loss=torch.nn.functional.cross_entropy(two.unsqueeze(0), torch.tensor([1],device=dev)); loss.backward(); torch.cuda.synchronize()
            g=float(proj.weight.grad.abs().sum()); proj.weight.grad=None; proj.bias.grad=None; del out,i
            print(f"gc {gc} rep {rep} fwd+bwd 22.5k tok {time.time()-ts:.2f}s peak alloc {torch.cuda.max_memory_allocated()/2**30:.1f} GB reserved {torch.cuda.max_memory_reserved()/2**30:.1f} GB gradsum {g:.4g}", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print("gc", gc, "OOM", str(e)[:200], flush=True)
        for p in proj.parameters(): p.grad=None
print("SMOKE_DONE", round(time.time()-t0), flush=True)
