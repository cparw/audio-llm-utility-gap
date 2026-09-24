#!/bin/bash
# PART 26 Qwen3-Omni PC-GITA: create one CPU pod (cpu5c, cpu3c, cpu3g; 32 vCPU), save request and response.
W="scores/part26/Qwen3-Omni-30B-A3B_PC-GITA"
N=${1:-p26-q3o-pcgita}
KEY=$(tr -d '\n\r ' < ~/.runpod_key); PUB=$(cat ~/.ssh/runpod_af3.pub)
body=$(/usr/local/bin/python3 -c "
import json,sys
print(json.dumps({'name':sys.argv[2],'computeType':'CPU','cpuFlavorIds':['cpu5c','cpu3c','cpu3g'],'vcpuCount':32,
 'imageName':'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404','containerDiskInGb':40,'ports':['22/tcp'],
 'env':{'PUBLIC_KEY':sys.argv[1]}}))" "$PUB" "$N")
echo "$body" > "$W/launch/create_$N.json"
r=$(curl -s -m 60 -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods -d "$body")
echo "$r" > "$W/launch/create_${N}_resp.json"
echo "$(date -u +%H:%M:%S) $N $(echo "$r" | /usr/local/bin/python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('id'),d.get('costPerHr'),d.get('cpuFlavorId'),d.get('vcpuCount'),d.get('machine',{}).get('dataCenterId'))" 2>&1)"
