#!/bin/bash
# PART26 CPU pod driver, Kimi-Audio Pitt encoder nested probe (states from this cell GPU pod p26-kimi-pitt-gpu).
# 1 pinned pip install (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1); 2 wait for the Mac upload (UPLOAD_OK);
# 3 sha256 check of every uploaded input; 4 p25a_nested5.py (PART25 A nested probe, unchanged) on the saved fold files;
# 5 p25a_podverify.py (PART25 A pod-side independent check, unchanged); 6 sha256 of out/; ALL_DONE.
cd /workspace; mkdir -p logs out
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore P25_NJ=${P25_NJ:-30} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; touch ALL_DONE; exit 1; }
log "DRIVER START kimi pitt P25_NJ=$P25_NJ nproc=$(nproc)"
python3 -m pip install -q --break-system-packages "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" joblib > logs/pip.log 2>&1 || fail "pip install"
python3 -c "import sklearn,numpy,scipy,joblib,platform;print('VERSIONS sk',sklearn.__version__,'np',numpy.__version__,'sp',scipy.__version__,'joblib',joblib.__version__,'py',platform.python_version(),platform.platform())" >> $L 2>&1
python3 -c "import numpy; numpy.show_config()" > logs/numpy_config.log 2>&1
grep -m1 "model name" /proc/cpuinfo >> $L; grep -q avx512f /proc/cpuinfo && log "avx512f present" || log "avx512f ABSENT"
touch PIP_DONE
for i in $(seq 1 180); do [ -f UPLOAD_OK ] && break; sleep 10; done
[ -f UPLOAD_OK ] || fail "upload not done after 30 min"
sha256sum -c in.sha256 > logs/in_check.log 2>&1 || fail "input sha256 check failed"
log "INPUT CHECK OK $(grep -c ': OK' logs/in_check.log) files"
python3 p25a_nested5.py in/kimi_pitt_encstates.npz pitt folds out/p26_kimi_pitt > logs/probe.log 2>&1 || fail "probe failed: $(tail -2 logs/probe.log | tr '\n' ' ' | cut -c1-200)"
log "PROBE DONE: $(grep 'NESTED5 DONE' logs/probe.log)"
python3 p25a_podverify.py in/kimi_pitt_encstates.npz pitt folds out/p26_kimi_pitt > logs/podverify.log 2>&1 || log "podverify crashed"
log "PODVERIFY: $(grep PODVERIFY logs/podverify.log)"
( cd out && sha256sum * > ../out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
