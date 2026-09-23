#!/bin/bash
# upload_one.sh name ip port : scripts, folds, JOBS, start driver, then npz inputs with sha256 check, then UPLOAD_DONE
V="<local data dir>/leftovers_23sep/verify/t5v"; cd "$V"
n=$1; ip=$2; port=$3; SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
echo "$n" > jobs/$n.name
ssh -n $SSHO -p $port root@$ip 'mkdir -p /workspace/scripts /workspace/folds /workspace/in'
scp -q $SSHO -P $port scripts/v5_refit.py scripts/v5_envfolds.py scripts/v5_driver.sh root@$ip:/workspace/scripts/
scp -q $SSHO -P $port folds/*.csv root@$ip:/workspace/folds/
scp -q $SSHO -P $port jobs/$n root@$ip:/workspace/JOBS
scp -q $SSHO -P $port jobs/$n.name root@$ip:/workspace/POD_NAME
ssh -n $SSHO -p $port root@$ip 'cd /workspace && nohup bash scripts/v5_driver.sh > logs/driver.stdout 2>&1 &'
echo "$(date -u +%T) $n driver started"
while read -r f s m; do
  [ -z "$f" ] && continue
  scp -q $SSHO -P $port inputs/$f.npz root@$ip:/workspace/in/ || { echo "$n scp $f FAILED"; exit 1; }
  l=$(shasum -a 256 inputs/$f.npz | cut -d' ' -f1); r=$(ssh -n $SSHO -p $port root@$ip "sha256sum /workspace/in/$f.npz" | cut -d' ' -f1)
  [ "$l" = "$r" ] && echo "$(date -u +%T) $n $f sha ok" || { echo "$n $f SHA MISMATCH"; exit 1; }
done < jobs/$n
ssh -n $SSHO -p $port root@$ip 'touch /workspace/UPLOAD_DONE'; echo "$(date -u +%T) $n UPLOAD_DONE"
