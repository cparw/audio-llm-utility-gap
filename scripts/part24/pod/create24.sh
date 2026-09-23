#!/bin/bash
# create N pods of one GPU type: bash create24.sh "NVIDIA H200" -> prints "name id cost gpu" lines, exit 1 if any failed
GPU="$1"; K=$(tr -d '\n\r ' < ~/.runpod_key); PUB=$(cat ~/.ssh/runpod_af3.pub); D=$(dirname "$0")/create; mkdir -p $D; ok=1
for k in 0 1 2 3 4; do
  n=p24-ft$k
  /usr/local/bin/python3 - "$n" "$GPU" "$PUB" > $D/$n.req.json <<'PY'
import json, sys
n, g, pub = sys.argv[1:4]
print(json.dumps({"name": n, "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", "gpuTypeIds": [g], "gpuCount": 1,
                  "cloudType": "SECURE", "containerDiskInGb": 40, "volumeInGb": 80, "volumeMountPath": "/workspace",
                  "ports": ["22/tcp"], "env": {"PUBLIC_KEY": pub}}))
PY
  done_k=0
  for a in 1 2 3; do
    c=$(curl -s -m 60 -X POST -H "Authorization: Bearer $K" -H "Content-Type: application/json" -d @$D/$n.req.json https://rest.runpod.io/v1/pods -o $D/$n.resp.json -w "%{http_code}")
    if [ "$c" = "201" ] || [ "$c" = "200" ]; then
      /usr/local/bin/python3 -c "import json;d=json.load(open('$D/$n.resp.json'));print('$n',d['id'],d.get('costPerHr'),(d.get('machine') or {}).get('gpuTypeId'))"; done_k=1; break
    else echo "CREATE_FAIL $n attempt $a HTTP $c $(head -c 200 $D/$n.resp.json)" >&2; sleep 8; fi
  done
  [ $done_k = 1 ] || { ok=0; break; }
done
[ $ok = 1 ]
