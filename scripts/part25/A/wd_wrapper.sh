#!/bin/bash
R2="<local data dir>/release_from_mac/scores/part25/A/retry_tf554"
for i in $(seq 1 120); do [ -s "$R2/pods_r2.txt" ] && break; sleep 10; done
sleep 240   # let the other launches register too
exec bash "$R2/watchdog_r2.sh" >> "$R2/watchdog_r2.log" 2>&1
