import os,json,subprocess,random,numpy as np
from concurrent.futures import ThreadPoolExecutor
SR=8000
D=json.load(open('part1a.json'))
def dec(p):
    o=subprocess.run(["ffmpeg","-v","error","-i",p,"-ac","1","-ar",str(SR),"-f","f32le","-"],capture_output=True).stdout
    return np.frombuffer(o,dtype=np.float32)
def off(clip_p,src_p):
    clip=dec(clip_p); src=dec(src_p)
    q=clip[:int(2.0*SR)]
    n=len(src); m=len(q)
    if n<m or m<100: return (None,None)
    L=1<<int(np.ceil(np.log2(n+m)))
    F=np.fft.rfft(src,L)*np.conj(np.fft.rfft(q,L))
    cc=np.fft.irfft(F,L)[:n-m+1]
    cs=np.concatenate([[0],np.cumsum(src.astype(np.float64)**2)])
    e=np.sqrt(np.maximum(cs[m:n+1]-cs[:n-m+1],1e-12))
    sc=cc/(e*np.sqrt(np.sum(q.astype(np.float64)**2)+1e-12))
    i=int(np.argmax(sc)); return (round(i/SR,4),round(float(sc[i]),4))
out={}
random.seed(0)
tasks=[('a20',r) for r in D['a20']]+[('adso',r) for r in random.sample(D['adso'],60)]
def job(t):
    k,r=t; o,s=off(r['seg'],r['src']); return (k,r['spk'],r['w0'],o,s)
with ThreadPoolExecutor(max_workers=6) as ex: res=list(ex.map(job,tasks))
json.dump(res,open('ad_offsets.json','w'))
for k in ['a20','adso']:
    rr=[x for x in res if x[0]==k and x[3] is not None]
    ok=sum(1 for x in rr if abs(x[2]-x[3])<=0.2)
    print(k,'checked',len(rr),'manifest start_s matches audio within 0.2s:',ok,
          'mean|delta|',round(float(np.mean([abs(x[2]-x[3]) for x in rr])),4),
          'min corr',round(min(x[4] for x in rr),4))
