#!/bin/bash
# PART24 on-pod backstop. The Mac watchdog normally pulls and deletes this pod within minutes of ALL_DONE. If it does not
# (Mac asleep, network down), this STOPS (does not delete) the pod through the pod-scoped key (GraphQL podStop), which ends
# GPU billing and keeps /workspace: when ALL_DONE is older than 30 min, or GPU 0% with no python for 40 min, or after 5 h.
eval $(cat /proc/1/environ | tr "\0" "\n" | grep -E "^RUNPOD_(POD_ID|API_KEY)=" | sed "s/^/export /")
L=/workspace/logs/guard24.log; idle=0; t0=$(date +%s)
echo "$(date -u +%H:%M:%S) guard start pod $RUNPOD_POD_ID" >> $L
stop(){ echo "$(date -u +%H:%M:%S) GUARD STOP: $1" >> $L; echo "GUARD_STOP $1 $(date -u +%H:%M:%S)" >> /workspace/logs/DRIVER.log; sync
  curl -s -m 30 -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
    -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}" >> $L 2>&1; echo >> $L; }
while true; do
  now=$(date +%s)
  if [ -f /workspace/ALL_DONE ]; then a=$(( now - $(stat -c %Y /workspace/ALL_DONE) )); [ $a -gt 1800 ] && stop "ALL_DONE for ${a}s, not deleted by the Mac watchdog"; fi
  g=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " ")
  py=$(ps -eo args | grep -E "^(/[^ ]*/)?python[0-9.]*( |$)" | grep -v -e jupyter | wc -l)
  if [ "$g" = "0" ] && [ "$py" = "0" ]; then idle=$((idle+60)); else idle=0; fi
  [ $idle -ge 2400 ] && stop "idle ${idle}s (GPU 0%, no python)"
  [ $(( now - t0 )) -gt 18000 ] && stop "guard uptime over 5 h"
  sleep 60
done
