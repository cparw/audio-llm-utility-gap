#!/bin/bash
# PART26 pod-side guard (independent of the Mac), adapted from part25/A guard25a.sh for a CPU pod: stops the pod with
# its own key when ALL_DONE is older than 30 min (the Mac watchdog should have pulled and deleted it by then), or when
# no python has run for 30 min after PIP_DONE, or after 3 h uptime.
eval $(cat /proc/1/environ | tr "\0" "\n" | grep -E "^RUNPOD_(POD_ID|API_KEY)=" | sed "s/^/export /")
L=/workspace/logs/guard26.log; idle=0; t0=$(date +%s); mkdir -p /workspace/logs
echo "$(date -u +%H:%M:%S) guard start pod $RUNPOD_POD_ID key_present=$([ -n "$RUNPOD_API_KEY" ] && echo yes || echo no)" >> $L
stop(){ echo "$(date -u +%H:%M:%S) GUARD STOP: $1" >> $L; echo "GUARD_STOP $1 $(date -u +%H:%M:%S)" >> /workspace/logs/DRIVER.log; sync
  curl -s -m 30 -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
    -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}" >> $L 2>&1; echo >> $L; }
while true; do
  now=$(date +%s)
  if [ -f /workspace/ALL_DONE ]; then a=$(( now - $(stat -c %Y /workspace/ALL_DONE) )); [ $a -gt 1800 ] && stop "ALL_DONE for ${a}s, not deleted by the Mac watchdog"; fi
  py=$(ps -eo args | grep -E "^(/[^ ]*/)?python[0-9.]*( |$)" | grep -v -e jupyter | wc -l)
  if [ "$py" = "0" ] && [ -f /workspace/PIP_DONE ]; then idle=$((idle+60)); else idle=0; fi
  [ $idle -ge 1800 ] && stop "idle ${idle}s (no python)"
  [ $(( now - t0 )) -gt 10800 ] && stop "guard uptime over 3 h"
  sleep 60
done
