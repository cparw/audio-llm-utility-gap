#!/bin/bash
# create N CPU pods named p25C-<k> and append "name id ip port cost" to pods.txt once ssh is reachable
# usage: create_pods.sh K1 K2 ...
R="<local data dir>/release_from_mac/scores/part25/C"
KEY=$(tr -d '\n\r ' < ~/.runpod_key); PUB=$(cat ~/.ssh/runpod_af3.pub)
for k in "$@"; do
  body=$(/usr/local/bin/python3 -c "
import json,sys
print(json.dumps({'name':'p25C-$k','computeType':'CPU','cpuFlavorIds':['cpu5c','cpu3c','cpu3g'],'vcpuCount':32,
 'imageName':'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404','containerDiskInGb':40,'ports':['22/tcp'],
 'env':{'PUBLIC_KEY':sys.argv[1]}}))" "$PUB")
  r=$(curl -s -m 60 -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods -d "$body")
  echo "$k $r" >> "$R/create.log"
  id=$(echo "$r" | /usr/local/bin/python3 -c "import json,sys;print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
  echo "p25C-$k id=$id cost=$(echo "$r" | /usr/local/bin/python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('costPerHr'),d.get('cpuFlavorId'))" 2>/dev/null)"
  [ -n "$id" ] && echo "p25C-$k $id" >> "$R/pods_ids.txt"
done
