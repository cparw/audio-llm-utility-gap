#!/bin/bash
# PART 26 (second try) CPU pod driver, Kimi-Audio ADReSSo encoder nested probe (five saved repeat folds + single split).
# Same steps as part26/Qwen2-Audio_ADReSSo/podfiles/driver26.sh: installs the pinned versions, checks the uploaded files
# against UPLOAD_SHA256.txt, runs p25a_nested5.py (byte copy of the PART 25 A script) with one BLAS thread per fit,
# then touches ALL_DONE. The Mac watchdog pulls out/, logs/ and the driver files, then deletes the pod.
cd /workspace; mkdir -p logs out
log(){ echo "$(date -u +%H:%M:%S) $*" >> logs/DRIVER.log; }
log "driver start"
python3 -m pip install -q --break-system-packages "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" joblib > logs/pip.log 2>&1 || { log "PIP FAILED"; echo pip > FAILED; }
python3 -c "import sklearn,numpy,scipy,joblib,platform;print('versions',sklearn.__version__,numpy.__version__,scipy.__version__,joblib.__version__,platform.python_version())" >> logs/DRIVER.log 2>&1
python3 -c "import numpy; numpy.show_config()" > logs/numpy_config.txt 2>&1
python3 -c "import threadpoolctl,json;print(json.dumps(threadpoolctl.threadpool_info()))" > logs/threadpool_before.txt 2>&1
grep -m1 "model name" /proc/cpuinfo >> logs/DRIVER.log; nproc >> logs/DRIVER.log
touch PIP_DONE
sha256sum -c UPLOAD_SHA256.txt >> logs/DRIVER.log 2>&1 || { log "UPLOAD SHA FAILED"; echo upload_sha >> FAILED; touch ALL_DONE; exit 1; }
log "upload sha ok"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 P25_NJ=24
python3 -u p25a_nested5.py in/kimi_adresso_enc_states_slim.npz adresso folds out/p26_kimi_adresso > logs/nested5.log 2>&1 \
  && log "done $(tail -1 logs/nested5.log)" || { log "FAILED nested5"; tail -20 logs/nested5.log >> logs/DRIVER.log; echo nested5 >> FAILED; }
cd out && sha256sum * > ../logs/out_sha256.txt; cd ..
touch ALL_DONE; log "ALL_DONE"
