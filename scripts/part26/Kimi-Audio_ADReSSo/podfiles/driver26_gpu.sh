#!/bin/bash
# PART 26 GPU pod driver, Kimi-Audio ADReSSo encoder states.
# Waits for setup_kimi.sh (byte copy of release/edaic_conflict_new/kimi/setup_kimi.sh), checks the uploaded audio and
# manifest against UPLOAD_SHA256.txt, runs kimi_encstates26.py, hashes out/, then touches ALL_DONE. The Mac watchdog
# pulls out/, logs/ and the driver files, then deletes the pod.
cd /workspace; mkdir -p logs out
log(){ echo "$(date -u +%H:%M:%S) $*" >> logs/DRIVER.log; }
log "driver start"
for i in $(seq 1 180); do [ -f KIMI_SETUP_DONE ] && break; sleep 10; done
[ -f KIMI_SETUP_DONE ] || { log "SETUP NOT DONE after 30 min"; echo setup > FAILED; touch ALL_DONE; exit 1; }
grep -q KIMI_IMPORT_OK logs/KIMI_SETUP.log || { log "KIMI IMPORT FAILED"; echo import > FAILED; touch ALL_DONE; exit 1; }
log "setup ok"
sha256sum -c UPLOAD_SHA256.txt > logs/upload_check.txt 2>&1 || { log "UPLOAD SHA FAILED"; echo upload_sha > FAILED; touch ALL_DONE; exit 1; }
log "upload sha ok $(grep -c ': OK' logs/upload_check.txt) files"
source /workspace/kimienv/bin/activate
export HF_HOME=/workspace/hf_cache PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
pip freeze > logs/pip_freeze.txt 2>&1
(cd /workspace/kimi_src && git log -1 --format="kimi_src commit %H %cd" && git -C kimia_infer/models/tokenizer/glm4 log -1 --format="glm4 commit %H") >> logs/DRIVER.log 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >> logs/DRIVER.log 2>&1
python -u kimi_encstates26.py /workspace/mf_adresso.csv /workspace/out/kimi_adresso ad > logs/kimi_adresso_enc.log 2>&1 \
  && log "done $(grep RESULT logs/kimi_adresso_enc.log)" || { log "FAILED extraction"; tail -20 logs/kimi_adresso_enc.log >> logs/DRIVER.log; echo extract >> FAILED; }
cd out && sha256sum * > ../logs/out_sha256.txt; cd ..
touch ALL_DONE; log "ALL_DONE"
