#!/bin/bash
# usage: up_chunks.sh LOCALFILE REMOTENAME IP PORT NCHUNKS TAG
F="$1"; RN="$2"; IP="$3"; PORT="$4"; N="$5"; TAG="$6"
SC=<local data dir>/scratch/vp16
SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
SZ=$(stat -f %z "$F"); MB=$(( (SZ + 1048575) / 1048576 )); PER=$(( (MB + N - 1) / N ))
t0=$(date +%s)
for i in $(seq 0 $((N-1))); do
  ( dd if="$F" bs=1048576 skip=$((i*PER)) count=$PER 2>/dev/null | ssh $SSHO -p $PORT root@$IP "cat > /workspace/in/$RN.part$(printf %02d $i)" ) &
done
wait
t1=$(date +%s)
ssh -n $SSHO -p $PORT root@$IP "cd /workspace/in && cat $RN.part* > $RN && rm -f $RN.part* && sha256sum $RN" > $SC/remote_$TAG.sha
LOC=$(shasum -a 256 "$F" | cut -d' ' -f1); REM=$(cut -d' ' -f1 $SC/remote_$TAG.sha)
echo "$TAG $RN bytes $SZ chunks $N upload $((t1-t0))s local $LOC remote $REM match=$([ "$LOC" = "$REM" ] && echo yes || echo NO)"
