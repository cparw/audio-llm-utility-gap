#!/bin/bash
# PART26 Qwen2-Audio PC-GITA CPU pod driver. 1 pinned fit libraries (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1);
# 2 wait for the Mac upload and check its sha256 list; 3 nested five-repeat encoder probe on the saved fold files
# (p25a_nested5.py, the PART25 A Qwen2.5-Omni script, unchanged); 4 pod-side independent check (p25a_podverify.py,
# unchanged); ALL_DONE. One BLAS thread per fit (OMP/OPENBLAS/MKL_NUM_THREADS=1).
cd /workspace; mkdir -p logs out in folds
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore P25_NJ=${P25_NJ:-32} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; touch ALL_DONE; exit 1; }
log "DRIVER START"
PKG='scikit-learn==1.9.1 numpy==2.1.2 scipy==1.18.1 joblib'
{ python3 -m pip install -q $PKG || python3 -m pip install --break-system-packages -q $PKG; } > logs/pip.log 2>&1 || fail "pip install failed"
python3 -c "import sklearn,numpy,scipy,joblib,platform;print('VERSIONS sk',sklearn.__version__,'np',numpy.__version__,'sp',scipy.__version__,'joblib',joblib.__version__,'py',platform.python_version(),platform.platform())" >> $L 2>&1
python3 -c "import numpy; numpy.show_config()" > logs/numpy_config.log 2>&1
log "CPU $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) nproc $(nproc)"
grep -q avx512f /proc/cpuinfo && log "avx512f present" || log "avx512f ABSENT"
python3 -c "import sklearn,numpy,scipy,sys;sys.exit(0 if (sklearn.__version__,numpy.__version__,scipy.__version__)==('1.9.1','2.1.2','1.18.1') else 1)" || fail "pinned versions not active"
touch PIP_DONE
for i in $(seq 1 360); do [ -f in/UPLOAD_OK ] && break; sleep 5; done
[ -f in/UPLOAD_OK ] || fail "upload not done after 30 min"
sha256sum -c in_sha256.txt > logs/upload_check.log 2>&1 || fail "upload sha256 check failed"
log "UPLOAD CHECK OK $(grep -c ': OK' logs/upload_check.log) files"
python3 p25a_nested5.py in/q2a_pcgita_encstates.npz pcgita folds out/p26_q2a_pcgita > logs/probe.log 2>&1 || fail "probe failed"
log "PROBE DONE: $(grep 'NESTED5 DONE' logs/probe.log)"
python3 p25a_podverify.py in/q2a_pcgita_encstates.npz pcgita folds out/p26_q2a_pcgita > logs/podverify.log 2>&1 || log "podverify crashed"
log "PODVERIFY: $(grep PODVERIFY logs/podverify.log)"
( cd out && sha256sum * > ../logs/out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
