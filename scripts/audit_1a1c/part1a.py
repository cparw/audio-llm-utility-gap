import csv, os, json
import cha
S=os.path.dirname(os.path.abspath(__file__))
B="/Volumes/G-Drive Pro/paper work/Hard drive data for paper"
DB=B+"/DementiaBank"; A20=DB+"/challenges/ADReSS-2020"
DUR=json.load(open(S+"/durations.json"))
def d(p): return DUR.get(p)

def stats(path, w0, w1, file_end_s):
    u=cha.bound(cha.parse_cha_all(path), int(round((file_end_s or 0)*1000)))
    W0,W1=w0*1000.0,w1*1000.0
    out=dict(n_utt=len(u), n_untimed=sum(1 for x in u if not x["timed"]),
             n_untimed_inv=sum(1 for x in u if not x["timed"] and x["who"]=="INV"))
    def sel(who,mode):
        r=[]
        for x in u:
            if x["who"]!=who: continue
            if mode=="certain":   ok = x["lo"]<W1 and x["hi"]>W0 and (x["timed"] or (x["lo"]>=W0 and x["hi"]<=W1))
            elif mode=="possible":ok = x["lo"]<W1 and x["hi"]>W0
            elif mode=="inside":  ok = x["lo"]>=W0-1e-6 and x["hi"]<=W1+1e-6
            if ok: r.append(x)
        return r
    for who,tag in (("PAR","par"),("INV","inv")):
        out[tag+"_certain"]=sum(len(x["words"]) for x in sel(who,"certain"))
        out[tag+"_possible"]=sum(len(x["words"]) for x in sel(who,"possible"))
        out[tag+"_inside"]=sum(len(x["words"]) for x in sel(who,"inside"))
        iv=sorted((max(x["lo"],W0),min(x["hi"],W1)) for x in sel(who,"certain"))
        tot=0.0; ce=None; cs=None
        for a_,b_ in iv:
            if b_<=a_: continue
            if cs is None: cs,ce=a_,b_
            elif a_<=ce: ce=max(ce,b_)
            else: tot+=ce-cs; cs,ce=a_,b_
        if cs is not None: tot+=ce-cs
        out[tag+"_sec"]=tot/1000.0
    seq=[]
    for x in sorted(sel("PAR","certain")+sel("INV","certain"), key=lambda y:(y["lo"],y["hi"])):
        for wd in x["words"]: seq.append(x["who"]+":"+wd)
    out["first20"]=" ".join(seq[:20])
    tl=[x["end_ms"] for x in u if x["timed"]]
    out["tx_max_end"]=max(tl)/1000.0 if tl else 0.0
    fp=[x["start_ms"] for x in u if x["timed"] and x["who"]=="PAR"]
    out["first_par_s"]=min(fp)/1000.0 if fp else None
    return out

pitt=[]
for r in csv.DictReader(open(DB+"/pitt_conflict_manifest.csv")):
    seg=DB+"/segments/"+os.path.basename(r["segment_path"])
    wv=f'{DB}/Pitt/{r["grp"]}/cookie/0wav/{r["session"]}.wav'
    mp=f'{DB}/Pitt/{r["grp"]}/cookie/{r["session"]}.mp3'
    src=wv if os.path.exists(wv) else mp
    chap=f'{DB}/transcripts/{r["grp"]}/cookie/{r["session"]}.cha'
    w0=int(r["start_ms"])/1000.0; w1=int(r["end_ms"])/1000.0
    s=stats(chap,w0,w1,d(src))
    pitt.append(dict(spk=r["spk"],session=r["session"],grp=r["grp"],label=r["label"],set=r["set"],
        seg=seg,seg_dur=d(seg),src=src,src_dur=d(src),w0=w0,w1=w1,winlen=w1-w0,cha=chap,**s))

adso=[]
for r in csv.DictReader(open(B+"/adresso/adresso_manifest_patient.csv")):
    seg=B+"/adresso/segments_patient/"+r["spk"]+".wav"
    src=DB+"/challenges/ADReSS-M/ADReSS-M-train_x/train/"+r["spk"]+".mp3"
    w0=float(r["start_s"]); sd=d(src); w1=min(w0+30.0, sd or w0+30.0)
    adso.append(dict(spk=r["spk"],label=r["label"],seg=seg,seg_dur=d(seg),src=src,src_dur=sd,w0=w0,w1=w1,winlen=w1-w0))

a20=[]
for r in csv.DictReader(open(B+"/adress2020/adress2020_manifest_patient.csv")):
    seg=B+"/adress2020/segments_patient/"+r["spk"]+".wav"
    if r["split"]=="train":
        sub="cd" if r["label"]=="1" else "cc"
        src=f"{A20}/ADReSS-IS2020-train_x/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/{sub}/{r['spk']}.wav"
        chap=f"{A20}/ADReSS-IS2020-train_x/ADReSS-IS2020-data/train/transcription/{sub}/{r['spk']}.cha"
    else:
        src=f"{A20}/ADReSS-IS2020-test_x/ADReSS-IS2020-data/test/Full_wave_enhanced_audio/{r['spk']}.wav"
        chap=f"{A20}/ADReSS-IS2020-test_x/ADReSS-IS2020-data/test/transcription/{r['spk']}.cha"
    w0=float(r["start_s"]); sd=d(src); w1=min(w0+30.0, sd or w0+30.0)
    s=stats(chap,w0,w1,sd)
    a20.append(dict(spk=r["spk"],split=r["split"],label=r["label"],seg=seg,seg_dur=d(seg),src=src,src_dur=sd,
        w0=w0,w1=w1,winlen=w1-w0,cha=chap,**s))

json.dump(dict(pitt=pitt,adso=adso,a20=a20),open(S+"/part1a.json","w"))
print("pitt",len(pitt),"adso",len(adso),"a20",len(a20))
print("pitt untimed INV utts total:",sum(r["n_untimed_inv"] for r in pitt))
print("pitt inv_certain>0",sum(1 for r in pitt if r["inv_certain"]>0),
      " inv_possible>0",sum(1 for r in pitt if r["inv_possible"]>0))
print("a20  inv_certain>0",sum(1 for r in a20 if r["inv_certain"]>0),
      " untimed utts",sum(r["n_untimed"] for r in a20))
print("pitt par_certain<15",sum(1 for r in pitt if r["par_certain"]<15),
      " a20 par_certain<15",sum(1 for r in a20 if r["par_certain"]<15))
import statistics as st
print("pitt inv words: max",max(r["inv_certain"] for r in pitt),"median",st.median([r["inv_certain"] for r in pitt]))
