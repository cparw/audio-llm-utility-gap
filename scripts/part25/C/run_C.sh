#!/bin/bash
# PART 25 track C pod driver. usage: bash run_C.sh "TAG:STREAM TAG:STREAM ..."
# Waits for every in/TAG_meta.npz and in/TAG_STREAM.npy plus in/UPLOAD_OK, installs pinned versions, runs each refit,
# then touches ALL_DONE (the Mac watchdog pulls and deletes the pod).
cd /workspace; mkdir -p logs out
log(){ echo "$(date -u +%H:%M:%S) $*" >> logs/DRIVER.log; }
log "driver start jobs: $1"
python3 -m pip install -q --break-system-packages "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" pandas joblib > logs/pip.log 2>&1 || { log "PIP FAILED"; echo pip > FAILED; }
python3 -c "import sklearn,numpy,scipy;print(sklearn.__version__,numpy.__version__,scipy.__version__)" >> logs/DRIVER.log 2>&1
grep -m1 "model name" /proc/cpuinfo >> logs/DRIVER.log; grep -c processor /proc/cpuinfo >> logs/DRIVER.log
grep -q avx512f /proc/cpuinfo && log "avx512f present" || log "avx512f ABSENT"
touch PIP_DONE
while [ ! -f in/UPLOAD_OK ]; do sleep 5; done
log "upload ok"
for j in $1; do
  T=${j%%:*}; S=${j#*:}
  log "start $T $S"
  python3 -u rep5_refit.py $T $S > logs/${T}_${S}.log 2>&1 && log "done $T $S $(tail -1 logs/${T}_${S}.log)" || { log "FAILED $T $S"; echo "$T $S" >> FAILED; }
done
touch ALL_DONE; log "ALL_DONE"
