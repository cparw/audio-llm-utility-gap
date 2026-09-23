#!/bin/bash
# upload the s1 stage to /workspace on one pod and start setup, smoke, the seed 1 driver and the on-pod guard under nohup.
# usage: launch24_s1.sh FOLD IP PORT STAGE_DIR SHA_OUT
K=$1; IP=$2; PORT=$3; ST=$4; SHAOUT=$5
SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
(cd "$ST" && COPYFILE_DISABLE=1 tar -cf - .) | ssh $SSHO -p $PORT root@$IP 'mkdir -p /workspace && tar -xf - -C /workspace --no-same-owner && cd /workspace && sha256sum $(find . -maxdepth 3 -type f \( -name "*.sh" -o -name "*.py" -o -name "*.csv" -o -name "*.txt" \) -not -path "./hf/*" -not -path "./edaicfull/cut/*" | sort)' > "$SHAOUT" || { echo "UPLOAD_FAIL $K"; exit 1; }
ssh -n $SSHO -p $PORT root@$IP "cd /workspace && mkdir -p logs out && echo 'LAUNCH seed1-only fold $K \$(date -u +%H:%M:%S) gpu \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)' >> logs/DRIVER.log && \
  (nohup bash setup.sh > logs/setup.out 2>&1 < /dev/null &) && \
  (nohup bash smoke24.sh > logs/smoke24.out 2>&1 < /dev/null &) && \
  (nohup bash run_fold24_s1.sh $K > logs/run_fold.out 2>&1 < /dev/null &) && \
  (nohup bash guard24.sh > logs/guard24.out 2>&1 < /dev/null &) && sleep 2 && ps -eo pid,args | grep -E 'setup.sh|smoke24.sh|run_fold24_s1.sh|guard24.sh' | grep -v grep; tail -1 logs/DRIVER.log" || { echo "START_FAIL $K"; exit 1; }
