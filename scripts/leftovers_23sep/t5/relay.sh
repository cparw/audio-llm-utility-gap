#!/bin/bash
# relay.sh SRC DST RELPATH... : copy /workspace/RELPATH from pod SRC to pod DST directly (pod to pod), with a throwaway key made on DST
T5="<local data dir>/leftovers_23sep/t5"; SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
S=$1; D=$2; shift 2
read -r _ _ sip sport _ < <(grep "^$S " "$T5/pods.txt"); read -r _ _ dip dport _ < <(grep "^$D " "$T5/pods.txt")
PUB=$(ssh -n $SSHO -p $dport root@$dip '[ -f /root/.ssh/t5relay ] || ssh-keygen -q -t ed25519 -N "" -f /root/.ssh/t5relay; cat /root/.ssh/t5relay.pub')
ssh -n $SSHO -p $sport root@$sip "grep -qF '$PUB' /root/.ssh/authorized_keys || echo '$PUB' >> /root/.ssh/authorized_keys"
for f in "$@"; do
  ssh -n $SSHO -p $dport root@$dip "mkdir -p /workspace/$(dirname $f); scp -q -i /root/.ssh/t5relay -o StrictHostKeyChecking=no -o BatchMode=yes -P $sport root@$sip:/workspace/$f /workspace/$f" || { echo "RELAY FAIL $f"; exit 1; }
  a=$(ssh -n $SSHO -p $sport root@$sip "sha256sum /workspace/$f" | cut -d' ' -f1); b=$(ssh -n $SSHO -p $dport root@$dip "sha256sum /workspace/$f" | cut -d' ' -f1)
  [ "$a" = "$b" ] && echo "relayed $f $S -> $D sha ok" || { echo "RELAY SHA MISMATCH $f"; exit 2; }
done
