#!/bin/bash
# usage: upload_pod.sh PODNAME   uploads every TAG:STREAM of that pod (jobs.txt) as in/TAG_meta.npz + in/TAG_STREAM.npy,
# checks sha256 of the array bytes on both ends, and touches in/UPLOAD_OK only when every array matches.
R="<local data dir>/release_from_mac/scores/part25/C"
N=$1; set -- $(awk -v n=$N '$1==n{print $3, $4}' "$R/pods.txt"); IP=$1; PORT=$2
SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
P2="<local data dir>/paper1_local_runs/probe2"
O2="<local data dir>/release/overnight2"
src(){ case $1 in
  q2a_*) echo "$P2/${1#q2a_}_states.npz";;
  q3o_*) echo "$O2/q3o/$1_states.npz";;
  kimi_edaic|kimi_pcgita|kimi_pitt) echo "$O2/kimi/$1_states.npz";;
  kimi_*) echo "$O2/kimi_new/$1_states.npz";;
esac; }
ok=1
for j in $(awk -v n=$N '$1==n{$1="";print}' "$R/jobs.txt"); do
  T=${j%%:*}; S=${j#*:}; NPZ=$(src $T)
  /usr/local/bin/python3 - "$NPZ" <<PY | ssh $SSHO -p $PORT root@$IP "cat > /workspace/in/${T}_meta.npz"
import sys, io, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
b = io.BytesIO(); np.savez(b, label=z["label"], spk=z["spk"], name=z["name"]); sys.stdout.buffer.write(b.getvalue())
PY
  t0=$(date +%s)
  LOC=$(/usr/local/bin/python3 - "$NPZ" "$S" "$R/up/${T}_${S}.sha" <<PY | ssh $SSHO -p $PORT root@$IP "cat > /workspace/in/${T}_${S}.npy.part && mv /workspace/in/${T}_${S}.npy.part /workspace/in/${T}_${S}.npy" ; cat "$R/up/${T}_${S}.sha"
import sys, hashlib, numpy as np
z = np.load(sys.argv[1], allow_pickle=True); a = np.ascontiguousarray(z[sys.argv[2]])
open(sys.argv[3], "w").write(hashlib.sha256(a.tobytes()).hexdigest())
fp = sys.stdout.buffer
np.lib.format.write_array_header_1_0(fp, np.lib.format.header_data_from_array_1_0(a))
fp.write(memoryview(a).cast('B')); fp.flush()
PY
)
  t1=$(date +%s)
  REM=$(ssh -n $SSHO -p $PORT root@$IP "python3 -c \"import numpy as np,hashlib;a=np.load('/workspace/in/${T}_${S}.npy');print(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())\"")
  m=NO; [ "$LOC" = "$REM" ] && [ -n "$LOC" ] && m=yes; [ $m = yes ] || ok=0
  echo "$(date -u +%H:%M:%S) $N $T $S src=$NPZ upload $((t1-t0))s local_sha $LOC remote_sha $REM match=$m"
done
if [ $ok = 1 ]; then ssh -n $SSHO -p $PORT root@$IP "touch /workspace/in/UPLOAD_OK"; echo "$N UPLOAD_OK"; else echo "$N UPLOAD MISMATCH"; fi
