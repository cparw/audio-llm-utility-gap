#!/bin/bash
# create the PART26 Qwen3-Omni Pitt CPU pod (REST), poll until ssh is reachable, write launch/ records.
# usage: create26.sh NAME
R="scores/part26/Qwen3-Omni-30B-A3B_Pitt"
N=$1; KEY=$(tr -d '\n\r ' < ~/.runpod_key); PUB=$(cat ~/.ssh/runpod_af3.pub)
body=$(/usr/local/bin/python3 -c "
import json,sys
print(json.dumps({'name':'$N','computeType':'CPU','cpuFlavorIds':['cpu5c','cpu3c','cpu3g'],'vcpuCount':32,
 'imageName':'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404','containerDiskInGb':40,'ports':['22/tcp'],
 'env':{'PUBLIC_KEY':sys.argv[1]}}))" "$PUB")
echo "$body" > "$R/launch/create_$N.json"
date -u > "$R/launch/START_UTC_$N.txt"
r=$(curl -s -m 60 -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods -d "$body")
echo "$r" > "$R/launch/create_${N}_resp.json"
id=$(echo "$r" | /usr/local/bin/python3 -c "import json,sys;print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
[ -n "$id" ] || { echo "CREATE FAILED: $r"; exit 1; }
cost=$(echo "$r" | /usr/local/bin/python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('costPerHr'))")
echo "created $N $id cost $cost flavor $(echo "$r" | /usr/local/bin/python3 -c "import json,sys;print(json.load(sys.stdin).get('cpuFlavorId'))")"
for i in $(seq 1 60); do
  curl -s -m 30 -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$id" > "$R/launch/get_$N.json"
  set -- $(/usr/local/bin/python3 -c "
import json;d=json.load(open('$R/launch/get_$N.json'))
print(d.get('publicIp') or '-', (d.get('portMappings') or {}).get('22','-'))")
  IP=$1; PORT=$2
  if [ "$IP" != "-" ] && [ "$PORT" != "-" ]; then
    if ssh -n -i ~/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -p $PORT root@$IP 'echo SSH_OK' 2>/dev/null | grep -q SSH_OK; then
      echo "$N $id $IP $PORT $cost" > "$R/launch/ports_$N.txt"; echo "READY $N $id $IP $PORT $cost"; exit 0; fi
  fi
  sleep 10
done
echo "NOT READY after 10 min: $N $id"; exit 2
