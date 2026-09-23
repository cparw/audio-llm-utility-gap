import csv, collections, hashlib, os
MAS="<local data dir>/release/master/master_lookup.csv"
ORIG="<local data dir>/scratch/master_lookup_ORIG.csv"
new=open(MAS,"rb").read(); orig=open(ORIG,"rb").read()
print("BYTE-PREFIX CHECK")
print("  original file bytes      :", len(orig), "md5", hashlib.md5(orig).hexdigest())
print("  new file starts with it  :", new.startswith(orig))
print("  prefix md5 of new file   :", hashlib.md5(new[:len(orig)]).hexdigest())
print("  bytes appended           :", len(new)-len(orig))
ro=list(csv.reader(open(ORIG,newline="",encoding="utf-8")))
rn=list(csv.reader(open(MAS,newline="",encoding="utf-8")))
print("  original data rows       :", len(ro)-1)
print("  original rows identical  :", ro==rn[:len(ro)])
print()
print("ROW COUNTS")
print("  before :", len(ro)-1)
print("  after  :", len(rn)-1)
print("  added  :", len(rn)-len(ro))
print()
print("ROWS ADDED PER DATASET")
added=rn[len(ro):]
c=collections.Counter(r[1] for r in added)
for ds in ["edaic_mid30","edaic_mid300","edaic_full","pitt_all","edaic_conflict_v2"]:
    print(f"  {ds:20s} {c[ds]:3d}")
extra={k:v for k,v in c.items() if k not in ("edaic_mid30","edaic_mid300","edaic_full","pitt_all","edaic_conflict_v2")}
print("  other datasets touched   :", extra if extra else "none")
print()
print("DUPLICATE (model,dataset,stream) KEY CHECK over the whole file")
keys=collections.Counter((r[0],r[1],r[2]) for r in rn[1:])
dups=[k for k,v in keys.items() if v>1]
print("  total keys :", len(keys), "| duplicates :", len(dups))
for k in dups: print("   DUP", k)
print()
print("MISSING SOURCE FILES among added rows")
miss=[r[6].split(" [")[0] for r in added if not os.path.exists(r[6].split(" [")[0])]
print("  ", miss if miss else "none, all 43 source paths exist on disk")
print()
print("ADDED ROWS")
for r in added:
    print(f"  {r[0]:20s} {r[1]:18s} {r[2]:20s} {r[3]:>7s} n={r[4]:>4s}")
