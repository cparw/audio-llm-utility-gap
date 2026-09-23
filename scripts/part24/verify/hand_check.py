# Independent hand recomputation of rater stats. Own parsing, own mapping, kappa from the 2x2 confusion matrix.
import csv, sys, math
import numpy as np
REL="<local data dir>/release/edaic_rerun"
sheet_p, stats_p, reading = sys.argv[1], sys.argv[2], sys.argv[3]
key={r["id"]:r for r in csv.DictReader(open(REL+"/rater_key_heard80.csv"))}
p9={}
for r in csv.DictReader(open(REL+"/part9_heard_window_sentiment.csv")):
    p9.setdefault(r["seg_uid"],[]).append(r)
def ans(v):
    s=v.strip().lower().rstrip(".")
    if s in ("yes","y"): return 1
    if s in ("no","n"): return 0
    raise ValueError(repr(v))
R=list(csv.DictReader(open(sheet_p,newline="")))
c1=[c for c in R[0] if c.replace(" ","").lower()=="rater1"][0]; c2=[c for c in R[0] if c.replace(" ","").lower()=="rater2"][0]
items=[]
for r in R:
    k=key[r["id"]]; segs=p9[k["source_id"]]
    # duplicates in part9 must agree on the fields used
    assert len({(s["heard_text"],s["lex_direction"],float(s["p_pos_heard"])>float(s["p_neg_heard"])) for s in segs})==1 and max(abs(float(s["p_pos_heard"])-float(segs[0]["p_pos_heard"]))+abs(float(s["p_neg_heard"])-float(segs[0]["p_neg_heard"])) for s in segs)<1e-6, k["id"]
    s=segs[0]
    sc = 1 if float(s["p_pos_heard"])>float(s["p_neg_heard"]) else 0   # 1 = positive text
    lab=int(k["label"])
    def textpos(a):
        if reading=="away":
            # label 1: Yes (away) = positive text; label 0: Yes (away) = negative text
            return a if lab==1 else 1-a
        return 1-a  # depressed: Yes = negative text
    a1,a2=ans(r[c1]),ans(r[c2])
    items.append(dict(id=r["id"],pid=k["pid"],arm=k["automatic_decision"],sc=sc,lex=1 if s["lex_direction"]=="positive" else 0,
                      d1=textpos(a1),d2=textpos(a2),a1=a1,a2=a2))
def kappa_cm(x,y):
    n=len(x); cm=[[0,0],[0,0]]
    for p,q in zip(x,y): cm[p][q]+=1
    po=(cm[0][0]+cm[1][1])/n
    r0=cm[0][0]+cm[0][1]; r1=cm[1][0]+cm[1][1]; k0=cm[0][0]+cm[1][0]; k1=cm[0][1]+cm[1][1]
    pe=(r0*k0+r1*k1)/(n*n)
    return float("nan") if pe==1 else (po-pe)/(1-pe)
def ci(fn,n,pids=None):
    g=np.random.default_rng(0); vals=[]
    if pids is not None:
        u=sorted(set(pids)); mem={p:[i for i in range(n) if pids[i]==p] for p in u}
    for _ in range(2000):
        if pids is None: idx=g.choice(n,size=n,replace=True)
        else:
            pk=g.choice(len(u),size=len(u),replace=True); idx=[i for j in pk for i in mem[u[j]]]
        v=fn(idx)
        if not math.isnan(v): vals.append(v)
    return np.percentile(vals,2.5),np.percentile(vals,97.5)
mine={}
for sub in ("all80","conflict40","agreement40"):
    S=[d for d in items if sub=="all80" or d["arm"]==sub[:-2]]; n=len(S); pids=[d["pid"] for d in S]
    for t in ("1","2"):
        for scn,scf in (("roberta_heard","sc"),("lexicon_arm","lex")):
            x=[d["d"+t] for d in S]; y=[d[scf] for d in S]
            m=[1.0 if p==q else 0.0 for p,q in zip(x,y)]
            f=lambda i,m=m: sum(m[j] for j in i)/len(i)
            mine[(sub,scn,"rater%s_agree_scorer"%t)]=(sum(m)/n,)+ci(f,n)+ci(f,n,pids)
            fk=lambda i,x=x,y=y: kappa_cm([x[j] for j in i],[y[j] for j in i])
            mine[(sub,scn,"rater%s_kappa_scorer"%t)]=(kappa_cm(x,y),)+ci(fk,n)+ci(fk,n,pids)
    x=[d["a1"] for d in S]; y=[d["a2"] for d in S]
    m=[1.0 if p==q else 0.0 for p,q in zip(x,y)]
    f=lambda i,m=m: sum(m[j] for j in i)/len(i)
    fk=lambda i,x=x,y=y: kappa_cm([x[j] for j in i],[y[j] for j in i])
    mine[(sub,"none","interrater_agree")]=(sum(m)/n,)+ci(f,n)+ci(f,n,pids)
    mine[(sub,"none","interrater_kappa")]=(kappa_cm(x,y),)+ci(fk,n)+ci(fk,n,pids)
got={(r["subset"],r["scorer"],r["metric"]):r for r in csv.DictReader(open(stats_p))}
bad=0; nchk=0
for k,v in mine.items():
    r=got.pop(k)
    fv=[float(r[c]) for c in ("value","ci_item_lo","ci_item_hi","ci_spk_lo","ci_spk_hi")]
    ok=all(round(a,4)==round(b,4) or abs(a-b)<5e-5 for a,b in zip(fv,v)); nchk+=5; bad+=0 if ok else 1
    tag="yes" if ok else "NO "
    if k[0]!="agreement40" or not ok:
        print("%s %-11s %-13s %-20s script %s  hand %s"%(tag,k[0],k[1],k[2],"%.4f [%.4f %.4f | %.4f %.4f]"%tuple(fv),"%.4f [%.4f %.4f | %.4f %.4f]"%tuple(v)))
print("rows left in script file not recomputed:",list(got))
print("SUMMARY %d rows, %d values, %d rows mismatched"%(len(mine),nchk,bad))
