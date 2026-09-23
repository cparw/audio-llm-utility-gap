import csv, hashlib, sys, os
bad=[]; n=0
for r in csv.DictReader(open(sys.argv[1])):
    n+=1
    if not os.path.exists(r["path"]): bad.append((r["clip"],"missing")); continue
    h=hashlib.sha256(open(r["path"],"rb").read()).hexdigest()
    if h!=r["sha256"]: bad.append((r["clip"],h))
print(f"SHA_CHECK {sys.argv[1]} n={n} bad={len(bad)}", bad[:5])
