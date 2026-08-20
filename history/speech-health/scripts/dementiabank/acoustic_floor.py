#!/usr/bin/env python3
"""Recording-properties floor on the Pitt cookie task.
Decodes each mp3 with ffmpeg, computes level / silence / spectral measures,
then reports what each property ALONE gives against the label."""
import csv, glob, os, subprocess, sys
import numpy as np

def decode(path, sr=16000):
    r = subprocess.run(["ffmpeg","-v","quiet","-i",path,"-f","s16le","-ac","1",
                        "-ar",str(sr),"-"], capture_output=True, timeout=120)
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32)/32768.0
    return x, sr

rows=[]
files = sorted(glob.glob("Pitt/*/cookie/*.mp3"))
for i,f in enumerate(files):
    try:
        x, sr = decode(f)
        if len(x) < sr: continue
        rms = float(np.sqrt(np.mean(x**2)))
        # frame-level energy for silence fraction and noise floor
        fl = sr//10
        nfr = len(x)//fl
        fe = np.sqrt(np.mean(x[:nfr*fl].reshape(nfr,fl)**2, axis=1))
        fe_db = 20*np.log10(np.maximum(fe,1e-8))
        sil_frac = float(np.mean(fe_db < fe_db.max()-30))
        noise_floor = float(np.percentile(fe_db,5))
        # spectral centroid + rolloff on a mid slice
        mid = x[len(x)//4:len(x)//4+sr*10]
        sp = np.abs(np.fft.rfft(mid))**2
        fr = np.fft.rfftfreq(len(mid), 1/sr)
        centroid = float((fr*sp).sum()/sp.sum())
        cum = np.cumsum(sp)
        rolloff = float(fr[np.searchsorted(cum, 0.85*cum[-1])])
        rows.append(dict(filepath=f, group=f.split(os.sep)[1],
            dur=round(len(x)/sr,1), rms=rms, sil_frac=round(sil_frac,4),
            noise_floor=round(noise_floor,2), centroid=round(centroid,1),
            rolloff=round(rolloff,1)))
    except Exception as e:
        print(f"fail {f}: {e}", flush=True)
    if (i+1)%100==0: print(f"progress {i+1}/{len(files)}", flush=True)

with open("pitt_acoustic_floor.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def auc(y,s):
    y=np.asarray(y,float); s=np.asarray(s,float)
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    n1=(y==1).sum(); n0=(y==0).sum()
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

y=[1 if r["group"]=="Dementia" else 0 for r in rows]
print(f"\nFLOOR on {len(rows)} cookie clips:")
for k in ("dur","rms","sil_frac","noise_floor","centroid","rolloff"):
    s=[r[k] for r in rows]
    a=auc(y,s)
    print(f"  {k:12s} alone: AUC {max(a,1-a):.3f}  ({'dementia higher' if a>0.5 else 'control higher'})")
# all six together, simple logistic via numpy (no sklearn): use rank-combined score
X=np.array([[r[k] for k in ("dur","rms","sil_frac","noise_floor","centroid","rolloff")] for r in rows])
X=(X-X.mean(0))/X.std(0)
yv=np.array(y,float)
# closed-form-ish: fisher direction
mu1=X[yv==1].mean(0); mu0=X[yv==0].mean(0)
Sw=np.cov(X[yv==1].T)+np.cov(X[yv==0].T)
w=np.linalg.solve(Sw+1e-6*np.eye(6), mu1-mu0)
print(f"  all six (fisher direction): AUC {auc(yv, X@w):.3f}")
