#!/bin/bash
# PART 25 track C pod driver v2 (replaces run_C.sh after pip was installed): runs each TAG:STREAM as soon as its own
# in/TAG_STREAM.npy is complete (the upload renames .part to .npy only after the stream ends), then touches ALL_DONE.
cd /workspace; mkdir -p logs out
log(){ echo "$(date -u +%H:%M:%S) $*" >> logs/DRIVER.log; }
log "driver v2 start jobs: $1"
for j in $1; do
  T=${j%%:*}; S=${j#*:}
  while [ ! -f in/${T}_${S}.npy ] || [ ! -f in/${T}_meta.npz ]; do sleep 5; done
  log "start $T $S"
  python3 -u rep5_refit.py $T $S > logs/${T}_${S}.log 2>&1 && log "done $T $S $(tail -1 logs/${T}_${S}.log)" || { log "FAILED $T $S"; echo "$T $S" >> FAILED; }
done
touch ALL_DONE; log "ALL_DONE"
