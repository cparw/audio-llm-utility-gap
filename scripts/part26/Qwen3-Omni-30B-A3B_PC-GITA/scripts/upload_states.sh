#!/bin/bash
# upload the stage npz to /workspace/in, check sha256 on the pod against expected.sha256, then touch in/UPLOAD_OK
W="scores/part26/Qwen3-Omni-30B-A3B_PC-GITA"
IP=$1; PORT=$2
SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
t0=$(date +%s)
ssh $SSHO -p $PORT root@$IP "cat > /workspace/in/q3o_pcgita_encstates.npz.part && mv /workspace/in/q3o_pcgita_encstates.npz.part /workspace/in/q3o_pcgita_encstates.npz" < "$W/stage/q3o_pcgita_encstates.npz"
t1=$(date +%s)
chk=$(ssh -n $SSHO -p $PORT root@$IP 'cd /workspace/in && sha256sum -c expected.sha256 2>&1')
echo "$(date -u +%H:%M:%S) upload $((t1-t0))s"; echo "$chk"
n=$(echo "$chk" | grep -c ': OK$')
if [ "$n" = "7" ]; then ssh -n $SSHO -p $PORT root@$IP 'touch /workspace/in/UPLOAD_OK'; echo "UPLOAD_OK 7/7 sha256 match"; else echo "UPLOAD MISMATCH ($n/7 OK)"; fi
