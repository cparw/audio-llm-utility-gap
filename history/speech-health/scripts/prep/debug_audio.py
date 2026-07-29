import csv, torch, librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
M="Qwen/Qwen2-Audio-7B-Instruct"
proc=AutoProcessor.from_pretrained(M)
model=Qwen2AudioForConditionalGeneration.from_pretrained(M,torch_dtype=torch.float16,device_map="cuda").eval()
atid=model.config.audio_token_index
rows=list(csv.DictReader(open("/project2/msoleyma_946/speech_health/results_chaitanya/manifests/kcl.csv")))
rows=[r for r in rows if r["task_type"]=="read"][:2]
for r in rows:
    y,_=librosa.load(r["filepath"],sr=16000,mono=True)
    conv=[{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":"Does this voice show signs of Parkinson? Answer Yes or No."}]}]
    text=proc.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
    print("== clip", r["speaker_id"], "dur", round(len(y)/16000,1),"s ==")
    print("  template has <|AUDIO|>:", "AUDIO" in text.upper(), "| has audio_bos:", "audio_bos" in text)
    inp=proc(text=text,audio=[y],sampling_rate=16000,return_tensors="pt")
    print("  proc keys:", list(inp.keys()))
    ii=inp["input_ids"]; print("  n_audio_tokens:", int((ii==atid).sum()), "seq_len:", ii.shape[-1])
    if "input_features" in inp: print("  input_features shape:", tuple(inp["input_features"].shape))
    inp={k:v.to("cuda") for k,v in inp.items()}
    with torch.no_grad(): lg=model(**inp).logits[0,-1].float()
    p=torch.softmax(lg,-1); tv,ti=torch.topk(p,3)
    print("  top3 tokens:", [(proc.tokenizer.decode([int(i)]), round(float(v),3)) for v,i in zip(tv,ti)])
print("--- first 300 chars of template ---"); print(text[:300])
