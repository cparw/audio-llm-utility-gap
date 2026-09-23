import hashlib, csv, os, sys, time, subprocess
W = "/workspace/pod4"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
mac = {l.split()[1]: l.split()[0] for l in open(f"{W}/ref/pitt468_mac_sha256.txt")}
while not os.path.exists(f"{W}/ORIG_OK"):
    d = "/workspace/data/pitt468"
    ok = [n for n in mac if os.path.exists(f"{d}/{n}") and sha(f"{d}/{n}") == mac[n]]
    print(time.strftime("%T"), "orig verified", len(ok), "/ 468", flush=True)
    if len(ok) == 468:
        open(f"{W}/ORIG_OK", "w").write("468/468 sha256 match <local data dir>/DementiaBank/segments\n"); break
    time.sleep(20)
