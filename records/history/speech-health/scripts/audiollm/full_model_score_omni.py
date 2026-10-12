import os, csv, argparse, json, random
import numpy as np, torch, librosa
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from sklearn.metrics import roc_auc_score
MODEL="Qwen/Qwen2.5-Omni-7B"
SYS_TXT=("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
         "capable of perceiving auditory and visual inputs, as well as generating text and speech.")
TEXTS = {
 "parkinsons": {
  "en":{"neutral":"The weather today is fairly ordinary and the room is at a normal temperature.",
        "sick":"My hands have been shaking and I have been having trouble moving. It keeps getting harder.",
        "healthy":"I feel physically great. I move easily, my hands are steady, and I exercise every day."},
  "it":{"neutral":"Il tempo oggi e abbastanza normale e la stanza e a temperatura normale.",
        "sick":"Le mani mi tremano e faccio fatica a muovermi. Sta diventando sempre piu difficile.",
        "healthy":"Mi sento benissimo fisicamente. Mi muovo con facilita, le mani sono ferme, mi alleno ogni giorno."},
  "es":{"neutral":"El clima de hoy es bastante normal y la sala esta a temperatura normal.",
        "sick":"Me tiemblan las manos y me cuesta moverme. Cada vez es mas dificil.",
        "healthy":"Me siento muy bien fisicamente. Me muevo con facilidad, mis manos estan firmes, hago ejercicio."}},
 "depression": {
  "en":{"neutral":"The weather today is fairly ordinary and the room is at a normal temperature.",
        "sick":"I feel hopeless and empty. Nothing brings me joy, I am exhausted, and I can barely get out of bed.",
        "healthy":"I feel wonderful and full of energy. I am enjoying life and looking forward to every day."},
  "es":{"neutral":"El clima de hoy es bastante normal y la sala esta a temperatura normal.",
        "sick":"Me siento sin esperanza y vacio. Nada me da alegria, estoy agotado y apenas puedo levantarme.",
        "healthy":"Me siento maravilloso y lleno de energia. Disfruto la vida y espero cada dia con ganas."}},
}
DIAG={"parkinsons":"does this speaker show signs of Parkinson's disease",
      "depression":"does this speaker show signs of depression"}
SYMPTOM={"parkinsons":"does this speaker have reduced articulation, monotone pitch, and slowed or breathy speech",
         "depression":"does this speaker have flat affect, low energy, and a monotone or slowed voice"}

def load_model():
    proc=Qwen2_5OmniProcessor.from_pretrained(MODEL)
    model=Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.float16,device_map="cuda").eval()
    try:
        model.disable_talker(); print("talker disabled",flush=True)
    except Exception as e:
        print("disable_talker skipped",repr(e)[:80],flush=True)
    tok=proc.tokenizer
    def wids(ws):
        s=set()
        for w in ws:
            for v in (w," "+w):
                e=tok(v,add_special_tokens=False).input_ids
                if e: s.add(e[0])
        return sorted(s)
    yes,no=wids(["Yes","yes","YES"]),wids(["No","no","NO"])
    print("yes_ids",yes,"no_ids",no,flush=True)
    return proc,model,yes,no

def score(proc,model,yes,no,audio,prompt):
    conv=[{"role":"system","content":[{"type":"text","text":SYS_TXT}]},
          {"role":"user","content":[{"type":"audio","audio":audio},{"type":"text","text":prompt}]}]
    text=proc.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
    inp=proc(text=text,audio=[audio],return_tensors="pt",padding=True,use_audio_in_video=False)
    inp={k:v.to(model.device) for k,v in inp.items()}
    with torch.no_grad():
        g=model.generate(**inp,max_new_tokens=1,do_sample=False,output_scores=True,
                         return_dict_in_generate=True,return_audio=False,use_audio_in_video=False)
    lg=g.scores[0][0].float()
    p=torch.softmax(lg,dim=-1); py=float(p[yes].sum()); pn=float(p[no].sum())
    return py/(py+pn+1e-9), py+pn

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True); ap.add_argument("--tasks",default="read")
    ap.add_argument("--lang",default="en"); ap.add_argument("--cond",default="parkinsons")
    ap.add_argument("--tag",required=True); ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args()
    R="/project2/msoleyma_946/speech_health/results_chaitanya"
    rows=[r for r in csv.DictReader(open(a.manifest)) if r["task_type"] in set(a.tasks.split(","))]
    if a.limit: rows=rows[:a.limit]
    assert all(r["task_type"]!="vowel" for r in rows), "vowel clips present in reliance set"
    proc,model,yes,no=load_model()
    diag=DIAG[a.cond]; symp=SYMPTOM[a.cond]; T=TEXTS[a.cond][a.lang]
    pA=f"Based only on how this person's voice sounds, {diag}? Answer with one word, Yes or No."
    pB=f"Based only on how this person's voice sounds, {symp}? Answer with one word, Yes or No."
    def pRel(t): return f"Here is what the person said: '{t}'. Based on this recording, {diag}? Answer with one word, Yes or No."
    idx=list(range(len(rows))); rng=random.Random(0); shuf=idx[:]; rng.shuffle(shuf)
    for i in range(len(shuf)):
        if shuf[i]==i: shuf[i]=(i+1)%len(shuf)
    y0,_=librosa.load(rows[0]["filepath"],sr=16000,mono=True)
    s,m=score(proc,model,yes,no,y0,pA); print(f"SMOKE p(yes)={s:.3f} mass={m:.3f}",flush=True)
    cache={}
    def geta(i):
        if i not in cache: cache[i]=librosa.load(rows[i]["filepath"],sr=16000,mono=True)[0]
        return cache[i]
    out=[]
    for i,r in enumerate(rows):
        try:
            y=geta(i); tr=r.get("transcript","").strip()
            sA,mA=score(proc,model,yes,no,y,pA)
            sB,mB=score(proc,model,yes,no,y,pB)
            sN,_=score(proc,model,yes,no,y,pRel(T["neutral"]))
            sS,_=score(proc,model,yes,no,y,pRel(T["sick"]))
            sH,_=score(proc,model,yes,no,y,pRel(T["healthy"]))
            sTrue=score(proc,model,yes,no,y,pRel(tr))[0] if tr else float("nan")
            sShuf,_=score(proc,model,yes,no,geta(shuf[i]),pRel(T["sick"]))
            out.append(dict(speaker=r["speaker_id"],label=int(r["label"]),behav_diag=sA,behav_symptom=sB,
                behav_mass=mA,rel_neutral=sN,rel_sick=sS,rel_healthy=sH,rel_true=sTrue,rel_sick_shuffled=sShuf,
                lex_sens=abs(sS-sH),flip=int((sS>=0.5)!=(sH>=0.5))))
            if (i+1)%25==0: print(f"  {i+1}/{len(rows)}",flush=True)
        except Exception as e: print("FAIL",r["filepath"],repr(e),flush=True)
    if not out: print("NO OUTPUT"); return
    yl=np.array([o["label"] for o in out])
    def sa(k):
        try: return round(roc_auc_score(yl,[o[k] for o in out]),3)
        except Exception: return float("nan")
    from sklearn.metrics import roc_auc_score
    lex=np.array([o["lex_sens"] for o in out]); ac_auc=sa("rel_neutral")
    ac=max(0.0,(ac_auc-0.5)*2) if ac_auc==ac_auc else 0.0
    lex_frac=round(float(lex.mean())/(float(lex.mean())+ac+1e-9),3)
    summ=dict(tag=a.tag,model="Qwen2.5-Omni-7B",n=len(out),cond=a.cond,lang=a.lang,tasks=a.tasks,
        speakers=len(set(o["speaker"] for o in out)),
        behavioral_diag_AUC=sa("behav_diag"),behavioral_symptom_AUC=sa("behav_symptom"),
        acoustic_neutral_AUC=ac_auc,acoustic_true_AUC=sa("rel_true"),
        mean_lexical_sensitivity=round(float(lex.mean()),3),median_lexical_sensitivity=round(float(np.median(lex)),3),
        lexical_fraction=lex_frac,flip_rate=round(float(np.mean([o["flip"] for o in out])),3),
        sick_score_true_audio=round(float(np.mean([o["rel_sick"] for o in out])),3),
        sick_score_shuffled_audio=round(float(np.mean([o["rel_sick_shuffled"] for o in out])),3),
        mean_answer_mass=round(float(np.mean([o["behav_mass"] for o in out])),3),
        prompt_diag=pA,prompt_symptom=pB,sick_text=T["sick"],healthy_text=T["healthy"],neutral_text=T["neutral"])
    os.makedirs(f"{R}/reliance",exist_ok=True)
    with open(f"{R}/reliance/{a.tag}_clips.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    json.dump(summ,open(f"{R}/reliance/{a.tag}_summary.json","w"),indent=2)
    print("=== SUMMARY",a.tag,"===",flush=True); print(json.dumps(summ,indent=2),flush=True)

if __name__=="__main__": main()
