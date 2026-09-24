#!/bin/bash
# keep trying to get an H200 for ADReSS-2020 (the H100 80GB HBM3 pod matched 153/156 zero-shot scores bit for bit, not all)
R2="<local data dir>/release_from_mac/scores/part25/A/retry_tf554"
for i in $(seq 1 40); do
  bash "$R2/scripts/launch_r2_h200only.sh" p25Ar-adress2020-h200 adress2020 ad
  grep -q "LAUNCH DONE" "$R2/logs/launch_p25Ar-adress2020-h200.log" && exit 0
  grep -q "CREATED" "$R2/logs/launch_p25Ar-adress2020-h200.log" && exit 1
  sleep 30
done
