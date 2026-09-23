#!/bin/bash
# upload stage/ to /workspace on one pod and start setup, smoke and the fold driver under nohup.  usage: launch24.sh FOLD IP PORT
K=$1; IP=$2; PORT=$3; H=$(cd "$(dirname "$0")" && pwd)
SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
(cd "$H/stage" && COPYFILE_DISABLE=1 tar -cf - .) | ssh $SSHO -p $PORT root@$IP 'mkdir -p /workspace && tar -xf - -C /workspace --no-same-owner && cd /workspace && sha256sum $(find . -maxdepth 3 -type f \( -name "*.sh" -o -name "*.py" -o -name "*.csv" -o -name "*.txt" \) -not -path "./hf/*" -not -path "./edaicfull/cut/*" | sort)' > "$H/create/p24-ft$K.upload_sha.txt" || { echo "UPLOAD_FAIL $K"; exit 1; }
ssh -n $SSHO -p $PORT root@$IP "cd /workspace && mkdir -p logs out && echo 'LAUNCH fold $K \$(date -u +%H:%M:%S)' >> logs/DRIVER.log && \
  (nohup bash setup.sh > logs/setup.out 2>&1 < /dev/null &) && \
  (nohup bash smoke24.sh > logs/smoke24.out 2>&1 < /dev/null &) && \
  (nohup bash run_fold24.sh $K > logs/run_fold.out 2>&1 < /dev/null &) && sleep 2 && ps -eo pid,args | grep -E 'setup.sh|smoke24.sh|run_fold24.sh' | grep -v grep" || { echo "START_FAIL $K"; exit 1; }
