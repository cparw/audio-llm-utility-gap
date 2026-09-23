#!/bin/bash
# usage: pull_and_delete.sh PODNAME   (pull out/ logs/ scripts into pull/<name>, check the file count, then DELETE the pod)
OUT="<local data dir>/leftovers_23sep/p16twins"
SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"; KEY=$(tr -d '\n\r ' < ~/.runpod_key)
n=$1; set -- $(awk -v n="$n" '$1==n{print $2, $3, $4}' "$OUT/pods.txt"); id=$1; ip=$2; port=$3
mkdir -p "$OUT/pull/$n"
cnt=$(ssh -n $SSHO -p $port root@$ip 'cd /workspace && { find out logs -type f; ls -1 *.py *.sh 2>/dev/null; ls -1 *DONE 2>/dev/null; } > /root/pull.list; wc -l < /root/pull.list' | tr -d ' ')
ssh -n $SSHO -p $port root@$ip 'cd /workspace && tar -cf - -T /root/pull.list' > "$OUT/pull/$n.tar"
got=$(tar -tf "$OUT/pull/$n.tar" | grep -vc '/$'); tar -xf "$OUT/pull/$n.tar" -C "$OUT/pull/$n"
echo "$n pod files $cnt tar files $got"
if [ -n "$cnt" ] && [ "$cnt" = "$got" ] && [ "$cnt" -gt 0 ]; then
  c=$(curl -s -m 30 -X DELETE -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$id"); sleep 2
  s=$(curl -s -m 30 -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$id")
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $n $id DELETE http $c, GET after -> $s" | tee -a "$OUT/pod_deletions.log"
else echo "PULL CHECK FAILED for $n, pod kept"; exit 1; fi
