#!/usr/bin/env python3
"""Independent recomputation of M4 with the csv module and pure-python arithmetic.
No pandas for the point estimates. Bootstrap re-run with the same seed protocol."""
import csv, os, statistics, json
import numpy as np

R   = '<local data dir>/Desktop/release/edaic_rerun/'
OUT = R + 'part16/'
FILES = {
 'Qwen2.5-Omni':       (R+'part14/p14_o25_zeroshot_scores.csv',  'clip',      'p_yes','mass'),
 'Qwen2-Audio':        (R+'part14/p14_q2a_zeroshot_scores.csv',  'clip',      'p_yes','mass'),
 'Qwen3-Omni-30B-A3B': (R+'part14/p14_q3o_zeroshot_scores.csv',  'clip',      'p_yes','mass'),
 'AudioFlamingo2':     (R+'part14/p14_af2.csv',                  'clip_path', 'p_yes','answer_mass'),
 'AudioFlamingo3':     (R+'part14/p14_af3.csv',                  'clip_path', 'p_yes','answer_mass'),
 'Kimi-Audio':         (R+'part14/p14_kimi_zeroshot_scores.csv', 'clip',      'p_yes','mass'),
}

# arm + speaker taken from the CLIP NAME itself (independent of the manifest join)
def load(path, clipcol, pcol, mcol):
    recs=[]
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            base = os.path.basename(row[clipcol]).replace('.wav','')
            arm, spk, _uid = base.split('_', 2)
            recs.append((arm, int(spk), float(row[pcol]), float(row[mcol])))
    return recs

res={}
for name,(p,c,pc,mc) in FILES.items():
    recs=load(p,c,pc,mc)
    for arm in ('conflict','agreement'):
        sub=[r for r in recs if r[0]==arm]
        n=len(sub)
        yes=sum(1 for r in sub if r[2] > 0.5)
        rate=yes/n
        med=statistics.median([r[3] for r in sub])
        nspk=len({r[1] for r in sub})
        # bootstrap, same protocol/seed
        rng=np.random.default_rng(0)
        spks=sorted({r[1] for r in sub})
        byspk={s:[1.0 if r[2]>0.5 else 0.0 for r in sub if r[1]==s] for s in spks}
        spk_arr=np.array(spks)
        draws=[]
        for _ in range(2000):
            d=rng.choice(spk_arr, size=len(spk_arr), replace=True)
            v=[]
            for s in d: v.extend(byspk[int(s)])
            draws.append(sum(v)/len(v))
        lo,hi=np.percentile(np.array(draws),[2.5,97.5])
        res[(name,arm)]=(n,nspk,yes,rate,float(lo),float(hi),float(med))
    # paired diff
    rng=np.random.default_rng(0)
    spks=sorted({r[1] for r in recs})
    byspk={s:{a:[1.0 if r[2]>0.5 else 0.0 for r in recs if r[1]==s and r[0]==a] for a in ('conflict','agreement')} for s in spks}
    spk_arr=np.array(spks)
    dd=[]
    for _ in range(2000):
        d=rng.choice(spk_arr, size=len(spk_arr), replace=True)
        cv=[];av=[]
        for s in d:
            cv.extend(byspk[int(s)]['conflict']); av.extend(byspk[int(s)]['agreement'])
        if not cv or not av: continue
        dd.append(sum(cv)/len(cv) - sum(av)/len(av))
    c_rate=sum(1 for r in recs if r[0]=='conflict'  and r[2]>0.5)/sum(1 for r in recs if r[0]=='conflict')
    a_rate=sum(1 for r in recs if r[0]=='agreement' and r[2]>0.5)/sum(1 for r in recs if r[0]=='agreement')
    lo,hi=np.percentile(np.array(dd),[2.5,97.5])
    res[(name,'DIFF')]=(len(recs),len(spks),None,c_rate-a_rate,float(lo),float(hi),None)

# compare with primary
import pandas as pd
prim=pd.read_csv(OUT+'M4_yes_rates.csv'); pdiff=pd.read_csv(OUT+'M4_yes_rate_diffs.csv')
print(f"{'model':20s} {'arm':9s} {'primary':>9s} {'verify':>9s} {'lo_p':>8s} {'lo_v':>8s} {'hi_p':>8s} {'hi_v':>8s} {'mass_p':>8s} {'mass_v':>8s}  AGREE")
ok=True
for _,r in prim.iterrows():
    n,nspk,yes,rate,lo,hi,med = res[(r['model'],r['arm'])]
    a = (round(rate,4)==r['yes_rate'] and round(lo,4)==r['lo'] and round(hi,4)==r['hi']
         and round(med,4)==r['median_answer_mass'] and n==r['n'] and nspk==r['n_spk'] and yes==r['yes_count'])
    ok &= a
    print(f"{r['model']:20s} {r['arm']:9s} {r['yes_rate']:9.4f} {rate:9.4f} {r['lo']:8.4f} {lo:8.4f} {r['hi']:8.4f} {hi:8.4f} {r['median_answer_mass']:8.4f} {med:8.4f}  {'YES' if a else 'NO'}")
print()
for _,r in pdiff.iterrows():
    n,nspk,_,d,lo,hi,_ = res[(r['model'],'DIFF')]
    a = (round(d,4)==r['diff'] and round(lo,4)==r['lo'] and round(hi,4)==r['hi'])
    ok &= a
    print(f"{r['model']:20s} {'DIFF':9s} {r['diff']:9.4f} {d:9.4f} {r['lo']:8.4f} {lo:8.4f} {r['hi']:8.4f} {hi:8.4f} {'':8s} {'':8s}  {'YES' if a else 'NO'}")
print("\nALL VALUES AGREE TO 4 DECIMALS:", ok)
json.dump({'all_agree_4dp':bool(ok)}, open(OUT+'M4_verify.json','w'), indent=2)
