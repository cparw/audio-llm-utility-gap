#!/bin/bash
# pull POD4c outputs to the Mac and append any new provisional rows (by id) to rows/POD4c_provisional.tsv
M=scores/part20/POD4c
K="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=25"
mkdir -p $M
ssh $K -p 15717 root@<pod ip> 'cd /workspace/scores/part20/POD4c && tar -cf - --exclude=POD4c_provisional_rows.tsv .' | tar -xf - -C $M
/usr/local/bin/python3 - <<'EOF'
import json, glob, os
M = "scores/part20/POD4c"
T = "reports/part20/rows/POD4c_provisional.tsv"
have = set()
if os.path.exists(T):
    have = {l.split("\t")[0] for l in open(T).read().splitlines()[1:] if l.strip()}
else:
    open(T, "w").write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\ttwo_dp\tvs_half\n")
new = 0
for r in sorted(glob.glob(M + "/*.RESULT")):
    rid = os.path.basename(r)[:-7]; j = json.load(open(M + "/" + rid + ".json"))
    pid = "POD4c_" + rid
    if pid in have: continue
    v, (lo, hi) = j["value"], j["ci95"]
    two = f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
    with open(T, "a") as f:
        f.write(f"{pid}\t{j['what']}\t{v:.4f}\t{lo:.4f}\t{hi:.4f}\t{j['n']}\t{j['n_speakers']}\t{M}/{rid}.csv\t{two}\t{j['verdict']}\n")
    new += 1
print("synced; new rows", new, "total results", len(glob.glob(M + "/*.RESULT")))
EOF
