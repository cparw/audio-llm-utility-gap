#!/bin/bash
# PART26 pod-side guard (independent of the Mac): podStop via the pod's own key when ALL_DONE is older than 45 min
# (the Mac watchdog should have pulled and deleted by then) or after 3 h uptime.
eval $(cat /proc/1/environ | tr "\0" "\n" | grep -E "^RUNPOD_(POD_ID|API_KEY)=" | sed "s/^/export /")
L=/workspace/logs/guard26.log; t0=$(date +%s); mkdir -p /workspace/logs
echo "$(date -u +%H:%M:%S) guard start pod $RUNPOD_POD_ID" >> $L
stop(){ echo "$(date -u +%H:%M:%S) GUARD STOP: $1" >> $L; sync
  curl -s -m 30 -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
    -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}" >> $L 2>&1; echo >> $L; }
while true; do
  now=$(date +%s)
  if [ -f /workspace/ALL_DONE ]; then a=$(( now - $(stat -c %Y /workspace/ALL_DONE) )); [ $a -gt 2700 ] && stop "ALL_DONE for ${a}s, not deleted by the Mac watchdog"; fi
  [ $(( now - t0 )) -gt 10800 ] && stop "guard uptime over 3 h"
  sleep 60
done
