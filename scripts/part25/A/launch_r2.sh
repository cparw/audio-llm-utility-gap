#!/bin/bash
# PART25 A retry: create one pod (H200 first, the original omni-final GPU; H100 80GB HBM3 only as fallback), wait for ssh,
# register it in pods_r2.txt for the watchdog, upload pod files, start guard + setup + driver, stream the audio.
# usage: launch_r2.sh NAME DS COND
R2="<local data dir>/release_from_mac/scores/part25/A/retry_tf554"; A="$(dirname "$R2")"
n=$1; ds=$2; cond=$3
K=$(tr -d '\n\r ' < ~/.runpod_key); PUB=$(cat ~/.ssh/runpod_af3.pub)
SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
log(){ echo "$(date -u +%H:%M:%S) $n $*" >> "$R2/logs/launch_$n.log"; }
ID=""
for GPU in "NVIDIA H200" "NVIDIA H100 80GB HBM3"; do
  cat > "$R2/launch/create_$n.json" <<J
{"name": "$n", "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", "gpuTypeIds": ["$GPU"], "gpuCount": 1, "cloudType": "SECURE", "containerDiskInGb": 40, "volumeInGb": 80, "volumeMountPath": "/workspace", "ports": ["22/tcp"], "env": {"PUBLIC_KEY": "$PUB"}}
J
  c=$(curl -s -m 60 -X POST -H "Authorization: Bearer $K" -H "Content-Type: application/json" -d @"$R2/launch/create_$n.json" https://rest.runpod.io/v1/pods -o "$R2/launch/create_${n}_resp.json" -w "%{http_code}")
  log "create $GPU -> HTTP $c $(head -c 150 "$R2/launch/create_${n}_resp.json")"
  if [ "$c" = "201" ] || [ "$c" = "200" ]; then ID=$(/usr/local/bin/python3 -c "import json;print(json.load(open('$R2/launch/create_${n}_resp.json'))['id'])"); break; fi
done
[ -z "$ID" ] && { log "CREATE FAILED"; exit 1; }
COST=$(/usr/local/bin/python3 -c "import json;print(json.load(open('$R2/launch/create_${n}_resp.json')).get('costPerHr'))")
log "CREATED $ID cost $COST"
echo "$n $ID" >> "$R2/launch/created_ids.txt"
IP=""; PORT=""
for i in $(seq 1 60); do
  curl -s -m 30 -H "Authorization: Bearer $K" "https://rest.runpod.io/v1/pods/$ID" -o "$R2/launch/pod_$n.json"
  read IP PORT < <(/usr/local/bin/python3 -c "import json;d=json.load(open('$R2/launch/pod_$n.json'));print(d.get('publicIp') or '', (d.get('portMappings') or {}).get('22') or '')" 2>/dev/null)
  [ -n "$IP" ] && [ -n "$PORT" ] && break; sleep 10
done
[ -z "$PORT" ] && { log "NO SSH MAPPING after 10 min; deleting"; curl -s -m 30 -X DELETE -H "Authorization: Bearer $K" "https://rest.runpod.io/v1/pods/$ID"; exit 1; }
grep -q "^$n " "$R2/pods_r2.txt" 2>/dev/null || echo "$n $ID $IP $PORT $COST" >> "$R2/pods_r2.txt"
log "REGISTERED $IP $PORT"
for i in $(seq 1 30); do ssh -n $SSHO -p $PORT root@$IP 'echo ok' >/dev/null 2>&1 && break; sleep 10; done
cd "$R2/podfiles"
COPYFILE_DISABLE=1 tar -cf - setup_r2.sh driver_r2.sh guard_r2.sh zs_compare.py extract_probe_layers.py nested_repeats_all.py p25a_nested5.py p25a_podverify.py folds saved \
 | ssh $SSHO -p $PORT root@$IP "mkdir -p /workspace && cd /workspace && tar --no-same-owner -xf - && mkdir -p logs && (nohup bash guard_r2.sh > /dev/null 2>&1 &) && (nohup bash setup_r2.sh > logs/setup.out 2>&1 &) && (nohup bash driver_r2.sh $ds $cond > logs/driver.out 2>&1 &) && echo STARTED" >> "$R2/logs/launch_$n.log" 2>&1 || { log "START FAILED"; exit 1; }
log "STARTED setup/driver/guard"
for t in 1 2 3; do
  t0=$(date +%s)
  /usr/local/bin/python3 upload_data.py "$A/stage" $ds | ssh $SSHO -p $PORT root@$IP "mkdir -p /workspace/data && cd /workspace/data && tar --no-same-owner -xf - && ls $ds | wc -l && touch /workspace/DATA_OK" >> "$R2/logs/launch_$n.log" 2>&1
  if ssh -n $SSHO -p $PORT root@$IP 'test -f /workspace/DATA_OK' 2>/dev/null; then log "DATA UPLOADED try $t in $(( $(date +%s) - t0 ))s"; break; fi
  log "upload try $t failed"
done
log "LAUNCH DONE"
