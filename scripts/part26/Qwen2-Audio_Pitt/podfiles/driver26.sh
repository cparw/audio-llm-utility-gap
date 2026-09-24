#!/bin/bash
# PART26 Qwen2-Audio Pitt CPU pod driver. usage: bash driver26.sh
# 1 pinned libraries (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1); 2 wait for UPLOAD_OK and check every uploaded file
# against the Mac sha256 list; 3 nested five-repeat encoder probe on the saved fold files (p25a_nested5.py, unchanged
# PART25 A script, one BLAS thread per fit); 4 pod-side independent check (p25a_podverify.py, unchanged); ALL_DONE.
cd /workspace; mkdir -p logs out
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore P25_NJ=${P25_NJ:-32} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; touch ALL_DONE; exit 1; }
log "DRIVER START q2a pitt"
python3 -m pip install -q --break-system-packages "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" joblib > logs/pip.log 2>&1 || fail "pip"
python3 -c "import sklearn,numpy,scipy,joblib,platform;print('VERSIONS sk',sklearn.__version__,'np',numpy.__version__,'sp',scipy.__version__,'joblib',joblib.__version__,'py',platform.python_version(),platform.platform())" >> $L 2>&1
grep -m1 "model name" /proc/cpuinfo >> $L; echo "nproc $(nproc)" >> $L
grep -q avx512f /proc/cpuinfo && log "avx512f present" || log "avx512f ABSENT"
touch PIP_DONE
for i in $(seq 1 360); do [ -f UPLOAD_OK ] && break; sleep 5; done
[ -f UPLOAD_OK ] || fail "upload not ready after 30 min"
sha256sum -c files.sha256 > logs/upload_check.log 2>&1 || fail "uploaded file sha256 check failed"
log "UPLOAD CHECK OK $(grep -c ': OK' logs/upload_check.log) files"
python3 p25a_nested5.py in/q2a_pitt_encstates.npz pitt folds out/p26_q2a_pitt > logs/probe_pitt.log 2>&1 || fail "probe failed"
log "PROBE DONE: $(grep 'NESTED5 DONE' logs/probe_pitt.log)"
python3 p25a_podverify.py in/q2a_pitt_encstates.npz pitt folds out/p26_q2a_pitt > logs/podverify_pitt.log 2>&1 || log "podverify crashed"
log "PODVERIFY: $(grep PODVERIFY logs/podverify_pitt.log)"
( cd out && sha256sum * > ../out_sha256.txt ); cp out_sha256.txt logs/
log "ALL DONE"
touch ALL_DONE
