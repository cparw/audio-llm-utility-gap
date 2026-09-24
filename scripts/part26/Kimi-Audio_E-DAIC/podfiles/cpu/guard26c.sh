#!/bin/bash
# PART 26 pod-side guard (copy of part26 guard26.sh; marker PIP_DONE instead of HF_DONE). Independent of the Mac:
# REST-free podStop via the pod's own key keeps /workspace but stops billing when ALL_DONE is older than 45 min (the Mac
# watchdog should have pulled and deleted by then), or GPU 0% with no python for 40 min after setup, or after 5 h uptime.
eval $(cat /proc/1/environ | tr "\0" "\n" | grep -E "^RUNPOD_(POD_ID|API_KEY)=" | sed "s/^/export /")
L=/workspace/logs/guard26.log; idle=0; t0=$(date +%s)
echo "$(date -u +%H:%M:%S) guard start pod $RUNPOD_POD_ID" >> $L
stop(){ echo "$(date -u +%H:%M:%S) GUARD STOP: $1" >> $L; echo "GUARD_STOP $1 $(date -u +%H:%M:%S)" >> /workspace/logs/DRIVER.log; sync
  curl -s -m 30 -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
    -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}" >> $L 2>&1; echo >> $L; }
while true; do
  now=$(date +%s)
  if [ -f /workspace/ALL_DONE ]; then a=$(( now - $(stat -c %Y /workspace/ALL_DONE) )); [ $a -gt 2700 ] && stop "ALL_DONE for ${a}s, not deleted by the Mac watchdog"; fi
  g=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " ")
  py=$(ps -eo args | grep -E "^(/[^ ]*/)?python[0-9.]*( |$)" | grep -v -e jupyter | wc -l)
  if { [ "$g" = "0" ] || [ -z "$g" ]; } && [ "$py" = "0" ] && [ -f /workspace/PIP_DONE ]; then idle=$((idle+60)); else idle=0; fi
  [ $idle -ge 2400 ] && stop "idle ${idle}s (GPU 0%, no python)"
  [ $(( now - t0 )) -gt 18000 ] && stop "guard uptime over 5 h"
  sleep 60
done
