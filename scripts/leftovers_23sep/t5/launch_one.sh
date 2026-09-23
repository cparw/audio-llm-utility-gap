#!/bin/bash
# launch_one.sh NAME : upload scripts, fold files and states for NAME's jobs, then start the driver detached
SP=<local data dir>/scratch/t5
T5="<local data dir>/leftovers_23sep/t5"; F=<local data dir>/release/folds
NAME=$1; read -r _ id ip port cost < <(grep "^$NAME " "$T5/pods.txt"); read -r _ mode var jobs < <(grep "^$NAME " $SP/assign.txt)
SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
for i in $(seq 1 20); do ssh -n $SSHO -p $port root@$ip 'mkdir -p /workspace/inputs /workspace/folds /workspace/out /workspace/logs && echo up' 2>/dev/null | grep -q up && break; sleep 10; done
ssh -n $SSHO -p $port root@$ip "cd /workspace; echo $NAME > POD_NAME; echo $mode > MODE; echo $var > VARIANTS; echo '$jobs' > JOBS" || exit 1
scp -q $SSHO -P $port $SP/curves_from_states.py $SP/t5_env.py $SP/t5_foldcheck.py $SP/t5_variant.py $SP/t5_vrefit.py $SP/driver_t5.sh root@$ip:/workspace/ || exit 2
FILES=""; NPZ=()
for J in $jobs; do
  DS=${J%%_*}; case $DS in pitt) FD=pitt;; edaic30|edaicfull) FD=edaic;; *) FD=$DS;; esac
  FILES="$FILES $F/${FD}_groupkfold5_pod.csv $F/${FD}_groupkfold5_mac.csv"; NPZ+=("$T5/inputs/${J}_states.npz")
done
scp -q $SSHO -P $port $(echo $FILES | tr ' ' '\n' | sort -u) root@$ip:/workspace/folds/ || exit 3
t0=$(date +%s); scp -q $SSHO -P $port "${NPZ[@]}" root@$ip:/workspace/inputs/ || exit 4; t1=$(date +%s)
# check the uploaded npz hashes against the local files
for f in "${NPZ[@]}"; do l=$(shasum -a 256 "$f" | cut -d' ' -f1); r=$(ssh -n $SSHO -p $port root@$ip "sha256sum /workspace/inputs/$(basename "$f")" | cut -d' ' -f1); [ "$l" = "$r" ] || { echo "$NAME SHA MISMATCH $(basename "$f")"; exit 5; }; done
ssh -n $SSHO -p $port root@$ip 'cd /workspace && nohup bash driver_t5.sh > logs/driver.out 2>&1 < /dev/null & echo launched' 
echo "$NAME $id upload $((t1-t0))s for ${#NPZ[@]} npz, sha ok, driver launched ($mode $var: $jobs)"
