#!/bin/bash
# PART 26 Kimi-Audio E-DAIC: upload to the CPU pod. usage: upload_cpu.sh IP PORT scripts|states
# scripts: p25a_nested5.py, p25a_podverify.py, driver26c.sh, guard26c.sh, the six release fold files; sha256 compared.
# states : pull/<gpu>/out/p26_kimi_edaic_encstates.npz (the GPU pod's enc-only file) plus files.sha256, then in/UPLOAD_OK.
C="scores/part26/Kimi-Audio_E-DAIC"
IP=$1; PORT=$2; WHAT=$3
SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
if [ "$WHAT" = "scripts" ]; then
  ssh -n $SSHO -p $PORT root@$IP 'mkdir -p /workspace/folds /workspace/in /workspace/logs /workspace/out'
  scp -q $SSHO -P $PORT "$C/podfiles/cpu/p25a_nested5.py" "$C/podfiles/cpu/p25a_podverify.py" "$C/podfiles/cpu/driver26c.sh" "$C/podfiles/cpu/guard26c.sh" root@$IP:/workspace/ || exit 1
  scp -q $SSHO -P $PORT "$C"/podfiles/cpu/folds/*.csv root@$IP:/workspace/folds/ || exit 1
  L=$(cd "$C/podfiles/cpu" && shasum -a 256 p25a_nested5.py p25a_podverify.py driver26c.sh guard26c.sh folds/*.csv | awk '{print $1}')
  R=$(ssh -n $SSHO -p $PORT root@$IP 'cd /workspace && sha256sum p25a_nested5.py p25a_podverify.py driver26c.sh guard26c.sh folds/*.csv' | awk '{print $1}')
  [ "$L" = "$R" ] && echo "SCRIPTS_SHA_MATCH $(echo "$L" | wc -l | tr -d ' ') files" || { echo "SCRIPTS_SHA_MISMATCH"; exit 1; }
elif [ "$WHAT" = "states" ]; then
  F=$(ls "$C"/pull/p26-kimi-edaic-gpu/out/p26_kimi_edaic_encstates.npz) || exit 1
  scp -q $SSHO -P $PORT "$F" root@$IP:/workspace/in/ || exit 1
  (cd "$(dirname "$F")" && shasum -a 256 p26_kimi_edaic_encstates.npz) > "$C/launch/cpu_in_files.sha256"
  scp -q $SSHO -P $PORT "$C/launch/cpu_in_files.sha256" root@$IP:/workspace/in/files.sha256 || exit 1
  R=$(ssh -n $SSHO -p $PORT root@$IP 'cd /workspace/in && sha256sum -c files.sha256 && touch UPLOAD_OK && echo UPLOAD_OK')
  echo "$R"
fi
