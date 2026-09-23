import numpy as np, pandas as pd, librosa, soundfile as sf, io, zipfile, os, sys, warnings, time
warnings.filterwarnings('ignore')
MAN='/Users/chaitanyaparwatkar/Desktop/release/edaic_rerun/confounds/clip_manifest.csv'
OUT='/Users/chaitanyaparwatkar/Desktop/release/edaic_rerun/confounds/clip_features.csv'
FRAME=2048; HOP=512; EPS=1e-12
_zc={}
def get_zip(p):
    if p not in _zc: _zc[p]=zipfile.ZipFile(p)
    return _zc[p]

def feats(rec):
    ds,path,zp = rec['dataset'],rec['path'],rec['zip']
    try:
        if isinstance(zp,str) and zp:
            data=get_zip(zp).read(path); y,sr=sf.read(io.BytesIO(data),dtype='float32',always_2d=False)
        else:
            y,sr=sf.read(path,dtype='float32',always_2d=False)
        if y.ndim>1: y=y.mean(axis=1)
        y=np.asarray(y,dtype=np.float32)
        dur=len(y)/sr
        if len(y)<FRAME: return dict(rec,err='too_short')
        S=np.abs(librosa.stft(y,n_fft=FRAME,hop_length=HOP,center=True))
        rms=librosa.feature.rms(S=S,frame_length=FRAME,hop_length=HOP)[0]
        nfr=len(rms)
        order=np.argsort(rms)
        k=max(1,int(round(0.20*nfr)))
        qi=order[:k]           # quietest fifth
        li=order[-k:]          # loudest fifth
        rq=float(np.sqrt(np.mean(rms[qi]**2)))
        rl=float(np.sqrt(np.mean(rms[li]**2)))
        ro=float(np.sqrt(np.mean(rms**2)))
        Sq=S[:,qi]
        cen=float(np.mean(librosa.feature.spectral_centroid(S=Sq,sr=sr)[0]))
        rol=float(np.mean(librosa.feature.spectral_rolloff(S=Sq,sr=sr,roll_percent=0.85)[0]))
        freqs=librosa.fft_frequencies(sr=sr,n_fft=FRAME)
        mag=np.mean(Sq,axis=1)
        db=20*np.log10(mag+EPS)
        sel=freqs>=50.0
        tilt=float(np.polyfit(freqs[sel]/1000.0,db[sel],1)[0])   # dB per kHz
        return dict(rec,
            dur_s=dur, sample_rate=float(sr),
            loudness_db=20*np.log10(ro+EPS),
            noise_floor_db=20*np.log10(rq+EPS),
            snr_db=20*np.log10((rl+EPS)/(rq+EPS)),
            quiet_centroid_hz=cen, quiet_rolloff85_hz=rol, quiet_tilt_db_per_khz=tilt,
            n_frames=float(nfr), n_quiet_frames=float(k), err='')
    except Exception as e:
        return dict(rec,err=type(e).__name__+': '+str(e)[:120])

def main():
    m=pd.read_csv(MAN).fillna({'zip':''})
    recs=m.to_dict('records')
    from multiprocessing import Pool
    t0=time.time(); out=[]
    with Pool(10) as pool:
        for i,r in enumerate(pool.imap_unordered(feats,recs,chunksize=4)):
            out.append(r)
            if (i+1)%100==0:
                print(f'{i+1}/{len(recs)} {time.time()-t0:.0f}s',flush=True)
    d=pd.DataFrame(out)
    d.to_csv(OUT,index=False)
    print('done',len(d),'errors',(d.err.fillna('')!='').sum(),time.time()-t0)
    print(d[d.err.fillna('')!=''][['dataset','path','err']].head(20).to_string())

if __name__=='__main__': main()
