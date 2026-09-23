#!/bin/bash
# manual pull + sha check + delete for one pod: pull_delete.sh <name>
R="<local data dir>/leftovers_23sep/verify/pod3b_vf"; cd "$R"
n=$1; read -r _ id ip port c < <(grep "^$n " pods.txt)
SSHO="-i $HOME/.ssh/runpod_af3 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
KEY=$(tr -d '\n\r ' < ~/.runpod_key); mkdir -p pull wd
[ -f "wd/$n.gone" ] && { echo "$n already gone"; exit 0; }
ssh -n $SSHO -p $port root@$ip 'cd /workspace && [ -f ALL_DONE ]' || { echo "$n not done"; exit 1; }
ssh -n $SSHO -p $port root@$ip 'cd /workspace && tar -cf - out logs *.sh *.py ALL_DONE PIP_DONE $(ls FAILED 2>/dev/null)' > pull/$n.manual.tar 2> wd/$n.manual.err || { echo "$n tar failed"; exit 1; }
rm -rf pull/$n; mkdir -p pull/$n; tar -xf pull/$n.manual.tar -C pull/$n || { echo "$n extract failed"; exit 1; }
bad=0; while read -r h f; do l=$(shasum -a 256 "pull/$n/out/$f" | cut -d' ' -f1); [ "$l" = "$h" ] || { bad=$((bad+1)); echo "sha mismatch $f"; }; done < pull/$n/out/SHA256SUMS.txt
[ $bad -eq 0 ] || { echo "$n sha mismatch, not deleting"; exit 1; }
code=$(curl -s -m 30 -X DELETE -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$id")
sleep 3; st=$(curl -s -m 30 -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$id")
echo "MANUAL_PULL_DELETE $(date -u +%H:%M:%SZ) HTTP $code status_get $st" > wd/$n.gone
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $n pulled $(ls pull/$n/out | wc -l | tr -d ' ') out files, sha ok, DELETE HTTP $code, GET now $st" | tee -a manual_actions.log
