import os,glob,subprocess,json,numpy as np
from concurrent.futures import ThreadPoolExecutor
SR=8000
P="/Volumes/G-Drive Pro/paper1_local_runs"
B="/Volumes/G-Drive Pro/paper work/Hard drive data for paper"
def dec(p):
    o=subprocess.run(["ffmpeg","-v","error","-i",p,"-ac","1","-ar",str(SR),"-f","f32le","-"],
                     capture_output=True).stdout
    return np.frombuffer(o,dtype=np.float32)
srcs={os.path.basename(x):x for x in glob.glob(B+"/KCL/extracted/26-29_09_2017_KCL/ReadText/*/*.wav")}
def job(f):
    clip=dec(P+"/kcl_read30b/"+f); src=dec(srcs[f])
    q=clip[:int(2.0*SR)]
    n=len(src); m=len(q)
    if n<m: return (f,None,None)
    # normalized cross correlation via FFT
    L=1<<int(np.ceil(np.log2(n+m)))
    F=np.fft.rfft(src,L)*np.conj(np.fft.rfft(q,L))
    cc=np.fft.irfft(F,L)[:n-m+1]
    cs=np.concatenate([[0],np.cumsum(src.astype(np.float64)**2)])
    e=np.sqrt(np.maximum(cs[m:n+1]-cs[:n-m+1],1e-12))
    score=cc/ (e*np.sqrt(np.sum(q.astype(np.float64)**2)+1e-12))
    i=int(np.argmax(score))
    return (f,round(i/SR,4),round(float(score[i]),4))
files=sorted(os.listdir(P+"/kcl_read30b"))
files=[f for f in files if f.endswith(".wav")]
with ThreadPoolExecutor(max_workers=6) as ex: res=list(ex.map(job,files))
json.dump({f:(o,s) for f,o,s in res},open("kcl_offsets.json","w"))
for f,o,s in res: print(f,o,s)
