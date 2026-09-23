#!/bin/bash
# usage: upload.sh NPZ DS STREAM IP PORT
NPZ="$1"; DS="$2"; ST="$3"; IP="$4"; PORT="$5"
SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
SC=<local data dir>/scratch/p16
ssh -n $SSHO -p $PORT root@$IP 'mkdir -p /workspace/in /workspace/out /workspace/logs'
scp -q $SSHO -P $PORT $SC/pod/twin_perlayer.py root@$IP:/workspace/
/usr/local/bin/python3 - "$NPZ" <<PY | ssh $SSHO -p $PORT root@$IP "cat > /workspace/in/${DS}_meta.npz"
import sys, io, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
b = io.BytesIO(); np.savez(b, label=z["label"], spk=z["spk"], name=z["name"]); sys.stdout.buffer.write(b.getvalue())
PY
t0=$(date +%s)
/usr/local/bin/python3 - "$NPZ" "$ST" <<PY | ssh $SSHO -p $PORT root@$IP "cat > /workspace/in/${DS}_${ST}.npy.part && mv /workspace/in/${DS}_${ST}.npy.part /workspace/in/${DS}_${ST}.npy"
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=True); a = np.ascontiguousarray(z[sys.argv[2]])
fp = sys.stdout.buffer
np.lib.format.write_array_header_1_0(fp, np.lib.format.header_data_from_array_1_0(a))
fp.write(memoryview(a).cast('B')); fp.flush()
PY
t1=$(date +%s)
LOC=$(/usr/local/bin/python3 -c "
import numpy as np,hashlib,sys;z=np.load('$NPZ',allow_pickle=True);a=np.ascontiguousarray(z['$ST']);print(hashlib.sha256(a.tobytes()).hexdigest())")
REM=$(ssh -n $SSHO -p $PORT root@$IP "python3 -c \"import numpy as np,hashlib;a=np.load('/workspace/in/${DS}_${ST}.npy');print(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())\"")
echo "$DS $ST upload $((t1-t0))s local_sha $LOC remote_sha $REM match=$([ "$LOC" = "$REM" ] && echo yes || echo NO)"
