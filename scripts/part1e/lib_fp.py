import subprocess, numpy as np, hashlib, os
SR=16000
def decode(path, dur=None, offset=None):
    cmd=["ffmpeg","-v","error","-nostdin"]
    if offset: cmd+=["-ss",str(offset)]
    cmd+=["-i",path,"-ac","1","-ar",str(SR),"-f","s16le","-acodec","pcm_s16le"]
    if dur: cmd+=["-t",str(dur)]
    cmd+=["-"]
    p=subprocess.run(cmd,capture_output=True)
    if p.returncode!=0: raise RuntimeError(path+" :: "+p.stderr.decode()[:200])
    return np.frombuffer(p.stdout,dtype="<i2").astype(np.float32)/32768.0

def _framesig(x,win=400,hop=160):
    if len(x)<win: x=np.pad(x,(0,win-len(x)))
    n=1+(len(x)-win)//hop
    idx=np.arange(win)[None,:]+hop*np.arange(n)[:,None]
    return x[idx]

_MEL=None
def _melfb(nfft=512,nmel=40,fmin=20.0,fmax=8000.0,sr=SR):
    global _MEL
    if _MEL is not None: return _MEL
    def h2m(f): return 2595.0*np.log10(1.0+f/700.0)
    def m2h(m): return 700.0*(10.0**(m/2595.0)-1.0)
    pts=m2h(np.linspace(h2m(fmin),h2m(fmax),nmel+2))
    bins=np.floor((nfft+1)*pts/sr).astype(int)
    fb=np.zeros((nmel,nfft//2+1),dtype=np.float32)
    for i in range(nmel):
        l,c,r=bins[i],bins[i+1],bins[i+2]
        if c==l: c=l+1
        if r==c: r=c+1
        r=min(r,nfft//2)
        if c>=fb.shape[1]: break
        fb[i,l:c]=(np.arange(l,c)-l)/max(c-l,1)
        fb[i,c:r]=(r-np.arange(c,r))/max(r-c,1)
    _MEL=fb; return fb

_DCT=None
def _dct(nmel=40,ncep=13):
    global _DCT
    if _DCT is not None: return _DCT
    n=np.arange(nmel)
    D=np.cos(np.pi/nmel*(n[None,:]+0.5)*np.arange(ncep)[:,None]).astype(np.float32)
    D*= np.sqrt(2.0/nmel); D[0]*=1/np.sqrt(2.0)
    _DCT=D; return D

def mfcc(x,ncep=13,win=400,hop=160,nfft=512,preemph=0.97):
    if len(x)<win+hop: return np.zeros((1,ncep),dtype=np.float32)
    y=np.append(x[0],x[1:]-preemph*x[:-1])
    F=_framesig(y,win,hop)*np.hamming(win).astype(np.float32)
    S=np.abs(np.fft.rfft(F,nfft))**2/nfft
    M=S@_melfb(nfft).T
    L=np.log(np.maximum(M,1e-10))
    return (L@_dct(40,ncep).T).astype(np.float32)

def frame_db(x,win=400,hop=160):
    if len(x)<win: x=np.pad(x,(0,win-len(x)))
    F=_framesig(x,win,hop)
    r=np.sqrt(np.mean(F*F,axis=1)+1e-20)
    return 20.0*np.log10(r)

def md5file(p,bs=1<<22):
    h=hashlib.md5()
    with open(p,"rb") as f:
        while True:
            b=f.read(bs)
            if not b: break
            h.update(b)
    return h.hexdigest()
