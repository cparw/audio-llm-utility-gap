#!/usr/bin/env python3
"""The Alzheimer's conflict set, from the Pitt cookie transcripts.

For depression the words point away from the diagnosis when a depressed person
says something positive. For Alzheimer's the words themselves are symptomatic,
so conflict means: a dementia patient whose SPEECH CONTENT looks healthy
(fluent, lexically rich, no retracing) or a control whose content looks
impaired. "How healthy the words look" is scored by a text-only model over
CHAT fluency markers, fitted out-of-fold so no speaker scores itself.

Segments: consecutive participant utterances merged to ~30 s windows using the
CHAT timestamps, same recipe as the daic build.
"""
import csv, glob, os, re
import numpy as np

TARGET_S = 30.0
rows_out = []

def parse_cha(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"@ID:\s*eng\|Pitt\|PAR\|(\d*);?[\d.]*\|(\w*)\|(\w*)\|[^|]*\|Participant\|(\d*)\|", txt)
    age, sex, dx, mmse = (m.groups() if m else ("","","",""))
    utts = []
    for mm in re.finditer(r"\*PAR:\t(.*?)(?:\x15|\s)(\d+)_(\d+)\x15?\s*$", txt, re.M):
        utts.append((mm.group(1), int(mm.group(2)), int(mm.group(3))))
    if not utts:  # timestamps sometimes lack the NAK char
        for mm in re.finditer(r"\*PAR:\t(.*?)\s(\d+)_(\d+)\s*$", txt, re.M):
            utts.append((mm.group(1), int(mm.group(2)), int(mm.group(3))))
    return dict(age=age, sex=sex, dx=dx, mmse=mmse), utts

def feats(text, dur_s):
    words = re.findall(r"[a-zA-Z']+", re.sub(r"\[[^\]]*\]|&-?\w+|<[^>]*>", " ", text))
    nw = len(words)
    uniq = len(set(w.lower() for w in words))
    return dict(
        nw=nw,
        ttr=uniq/max(nw,1),
        retrace=len(re.findall(r"\[/+\]", text)),
        fillers=len(re.findall(r"&-?(?:uh|um|er|hm|mhm)", text)),
        unintel=len(re.findall(r"\bxxx\b", text)),
        errors=len(re.findall(r"\[\*", text)),
        pauses=len(re.findall(r"\(\.+\)", text)),
        rate=nw/max(dur_s,0.5),
    )

segs = []
for cha in sorted(glob.glob("transcripts/*/cookie/*.cha")):
    grp = cha.split(os.sep)[1]
    base = os.path.basename(cha)[:-4]
    spk = base.split("-")[0]
    meta, utts = parse_cha(cha)
    if not utts: continue
    cur, t0 = [], utts[0][1]
    for text, a, b in utts:
        cur.append((text, a, b))
        if (b - t0)/1000.0 >= TARGET_S:
            merged = " ".join(t for t,_,_ in cur)
            dur = (b - t0)/1000.0
            f = feats(merged, dur)
            if f["nw"] >= 20:
                segs.append(dict(spk=spk, grp=grp, label=1 if grp=="Dementia" else 0,
                    session=base, start_ms=t0, end_ms=b, dur=round(dur,1),
                    dx=meta["dx"], mmse=meta["mmse"], age=meta["age"], sex=meta["sex"],
                    text=merged[:400], **f))
            cur, t0 = [], b
print(f"segments: {len(segs)} from {len(set((s['grp'],s['spk']) for s in segs))} speakers")
print(f"  by group: dementia {sum(s['label'] for s in segs)}, control {sum(1-s['label'] for s in segs)}")

# text-only "how impaired do the words look", out-of-fold logistic over 5 speaker folds
# CONTENT markers only. rate and nw are excluded on purpose: they are realised
# acoustically as speaking speed, so selecting on them builds a rate shortcut
# into the conflict arm (caught by the validity battery at AUC 0.92).
FEATS = ["ttr","retrace","fillers","unintel","errors","pauses"]
X = np.array([[s[f] for f in FEATS] for s in segs], float)
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
y = np.array([s["label"] for s in segs], float)
spkid = np.array([s["grp"]+s["spk"] for s in segs])
uspk = np.unique(spkid)
rng = np.random.default_rng(7); rng.shuffle(uspk)
folds = np.array_split(uspk, 5)
oof = np.zeros(len(y))
for fo in folds:
    te = np.isin(spkid, fo); tr = ~te
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(400):
        p = 1/(1+np.exp(-(X[tr]@w + b)))
        g = X[tr].T@(p - y[tr])/tr.sum(); gb = (p - y[tr]).mean()
        w -= 0.5*g + 0.001*w; b -= 0.5*gb
    oof[te] = 1/(1+np.exp(-(X[te]@w + b)))

def auc(y, s):
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    n1=(y==1).sum(); n0=(y==0).sum()
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
print(f"  text-only OOF AUC (words vs diagnosis): {auc(y, oof):.3f}")

# conflict = words point the wrong way; use per-arm tertiles
lo, hi = np.percentile(oof, [33, 67])
conf, agree = [], []
for s, p in zip(segs, oof):
    s = dict(s); s["p_text"] = round(float(p), 4)
    if   s["label"]==1 and p <= lo: s["set"]="conflict";  conf.append(s)
    elif s["label"]==0 and p >= hi: s["set"]="conflict";  conf.append(s)
    elif s["label"]==1 and p >= hi: s["set"]="agreement"; agree.append(s)
    elif s["label"]==0 and p <= lo: s["set"]="agreement"; agree.append(s)
print(f"  conflict: {len(conf)} ({sum(s['label'] for s in conf)} dementia-fluent, "
      f"{sum(1-s['label'] for s in conf)} control-impaired)")
print(f"  agreement: {len(agree)}")
print(f"  conflict speakers: {len(set(s['grp']+s['spk'] for s in conf))}")

keep = conf + agree
with open("pitt_conflict_manifest.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(keep[0].keys())); w.writeheader(); w.writerows(keep)
print("wrote pitt_conflict_manifest.csv")
