# Build synthetic filled rater sheets from read only copies of the real sheets. Real sheets are never written.
import csv, sys, random, glob
REL="<local data dir>/release/edaic_rerun"
OUT=sys.argv[1]; LIVE=sys.argv[2]
key={r["id"]:r for r in csv.DictReader(open(REL+"/rater_key_heard80.csv"))}
p9={}
for r in csv.DictReader(open(REL+"/part9_heard_window_sentiment.csv")):
    p9.setdefault(r["seg_uid"],[]).append(r)
def scorer(i):
    s=p9[key[i]["source_id"]][0]
    return "positive" if float(s["p_pos_heard"])>float(s["p_neg_heard"]) else "negative"
def away_answer(direction,label):
    # Yes = words point away from shown label. label1 goes with negative words.
    goes_with="negative" if label==1 else "positive"
    return "No" if direction==goes_with else "Yes"
def dep_answer(direction):
    return "Yes" if direction=="negative" else "No"
flip=lambda a:"No" if a=="Yes" else "Yes"
# local layout, set A: rater1 flips 20 percent of scorer-consistent answers, rater2 30 percent, own seeds
rows=list(csv.reader(open(REL+"/rater_sheet_heard80.csv",newline="")))
h=rows[0]; i1=h.index("rater1"); i2=h.index("rater2")
for tag,p1,p2,s1,s2,spell in (("A",0.2,0.3,11,22,False),("B",0.45,0.1,33,44,True)):
    r1=random.Random(s1); r2=random.Random(s2)
    out=[h]
    for r in rows[1:]:
        r=list(r); lab=int(r[h.index("label")]); d=scorer(r[0])
        a=away_answer(d,lab)
        x=flip(a) if r1.random()<p1 else a
        y=flip(a) if r2.random()<p2 else a
        if spell:  # mixed spellings the script must read
            x={"Yes":r1.choice(["yes","Y"," YES "]),"No":r1.choice(["no","N","No."])}[x]
        r[i1]=x; r[i2]=y; out.append(r)
    csv.writer(open(OUT+"/synth_local_%s.csv"%tag,"w",newline="")).writerows(out)
# local layout, set C: rater1 exactly the scorer, rater2 exactly the lexicon direction
out=[h]
for r in rows[1:]:
    r=list(r); lab=int(r[h.index("label")])
    r[i1]=away_answer(scorer(r[0]),lab); r[i2]=away_answer(key[r[0]]["lex_direction"],lab); out.append(r)
csv.writer(open(OUT+"/synth_local_C.csv","w",newline="")).writerows(out)
# live layout, set D: depressed reading, Yes = suggests depression
lrows=list(csv.reader(open(LIVE,newline="")))
h=lrows[0]; j1=h.index("rater 1"); j2=h.index("rater 2")
r1=random.Random(55); r2=random.Random(66)
out=[h]
for r in lrows[1:]:
    r=list(r); a=dep_answer(scorer(r[0]))
    r[j1]=flip(a) if r1.random()<0.25 else a
    r[j2]=flip(a) if r2.random()<0.35 else a
    out.append(r)
csv.writer(open(OUT+"/synth_live_D.csv","w",newline="")).writerows(out)
print("ok")
