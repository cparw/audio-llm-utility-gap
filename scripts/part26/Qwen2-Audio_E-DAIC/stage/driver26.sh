#!/bin/bash
# PART26 pod driver, Qwen2-Audio E-DAIC (30 s) encoder nested probe. usage: bash driver26.sh
# 1 pinned fit libraries (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1); 2 wait for in/UPLOAD_OK and check the input
# sha256 list; 3 p25a_nested5.py (the PART25 A Qwen2.5-Omni nested probe, unchanged) on the saved fold files;
# 4 p25a_podverify.py (unchanged, independent pod-side check); ALL_DONE.  One BLAS thread per fit.
cd /workspace; mkdir -p logs out
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 P25_NJ=${P25_NJ:-30}
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; touch ALL_DONE; exit 1; }
log "DRIVER START q2a edaic P25_NJ=$P25_NJ"
python3 -m pip install -q --break-system-packages "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" joblib > logs/pip.log 2>&1 || fail "pip install failed"
python3 -c "import sklearn,numpy,scipy,joblib,platform;print('VERSIONS sk',sklearn.__version__,'np',numpy.__version__,'sp',scipy.__version__,'joblib',joblib.__version__,'py',platform.python_version(),platform.platform())" >> $L 2>&1
python3 -c "import sklearn,numpy,scipy;assert (sklearn.__version__,numpy.__version__,scipy.__version__)==('1.9.1','2.1.2','1.18.1')" || fail "pinned versions not installed"
python3 -c "import numpy; numpy.show_config()" > logs/numpy_config.txt 2>&1
log "CPU $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) nproc $(nproc) avx512f $(grep -q avx512f /proc/cpuinfo && echo yes || echo no)"
touch PIP_DONE
for i in $(seq 1 360); do [ -f in/UPLOAD_OK ] && break; sleep 5; done
[ -f in/UPLOAD_OK ] || fail "upload not done after 30 min"
( sha256sum -c in/files.sha256 ) > logs/input_check.log 2>&1 || fail "input sha256 check failed"
log "INPUT CHECK OK $(grep -c ': OK' logs/input_check.log) files"
python3 p25a_nested5.py in/q2a_edaic_encstates.npz edaic folds out/p26_q2a_edaic > logs/probe_edaic.log 2>&1 || fail "probe failed: $(tail -2 logs/probe_edaic.log | tr '\n' ' ')"
log "PROBE DONE: $(grep 'NESTED5 DONE' logs/probe_edaic.log)"
python3 p25a_podverify.py in/q2a_edaic_encstates.npz edaic folds out/p26_q2a_edaic > logs/podverify_edaic.log 2>&1 || log "podverify crashed: $(tail -2 logs/podverify_edaic.log | tr '\n' ' ')"
log "PODVERIFY: $(grep PODVERIFY logs/podverify_edaic.log)"
( cd out && sha256sum * > ../out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
