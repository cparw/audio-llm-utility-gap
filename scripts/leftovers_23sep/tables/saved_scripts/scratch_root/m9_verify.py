# independent recompute: pure-python csv + statistics, no pandas/numpy
import csv, statistics, sys
rows=list(csv.DictReader(open('<local data dir>/release/edaic_rerun/part16/M9_answer_mass.csv')))
MASS=("answer_mass","mass","denom","p_sum","psum")
bad=0; ok=0; checked=0
for r in rows:
    if not r["median_mass"]: continue
    f=r["file"]
    try: rd=list(csv.DictReader(open(f, errors="replace")))
    except Exception as e: print("READFAIL",f,e); bad+=1; continue
    if not rd: continue
    cols={k.lower():k for k in rd[0]}
    mc=next((cols[k] for k in MASS if k in cols), None)
    if mc is None: print("NOCOL",f); bad+=1; continue
    v=[]
    for x in rd:
        try:
            y=float(x[mc])
            if y==y: v.append(y)
        except Exception: pass
    if not v: continue
    v.sort()
    med=statistics.median(v)
    n=len(v)
    p10=v[max(0,int(round(0.1*(n-1))))]
    s01=sum(1 for y in v if y<0.01)/n
    s15=sum(1 for y in v if y<0.15)/n
    checked+=1
    d_med=abs(med-float(r["median_mass"]))
    d_n = n-int(r["n"])
    d01=abs(s01-float(r["share_below_0.01"])); d15=abs(s15-float(r["share_below_0.15"]))
    if round(med,4)!=round(float(r["median_mass"]),4) or d_n!=0 or round(s01,4)!=round(float(r["share_below_0.01"]),4) or round(s15,4)!=round(float(r["share_below_0.15"]),4):
        print(f"MISMATCH {r['model']}/{r['dataset']}/{r['stream']}  med {med:.6f} vs {float(r['median_mass']):.6f}  n {n} vs {r['n']}  col {mc} vs {r['mass_column_used']}  {f}")
        bad+=1
    else: ok+=1
print(f"\nVERIFY: {checked} cells recomputed independently, {ok} agree to 4 decimals on median/n/share<0.01/share<0.15, {bad} disagree")
