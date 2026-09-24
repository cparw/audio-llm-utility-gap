#!/bin/bash
# PART26 CPU pod driver, Kimi-Audio ADReSS-2020 encoder nested probe.
# 1 pinned pip install (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1); 2 sha256 check of every uploaded file against the Mac list;
# 3 p25a_nested5.py (PART25 A script, unchanged) on the saved fold files, one BLAS thread per fit; 4 p25a_podverify.py
# (PART25 A pod-side independent check, unchanged); 5 sha256 of out/; ALL_DONE.
cd /workspace; mkdir -p logs out
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore P25_NJ=${P25_NJ:-30} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; ( cd out && sha256sum * > ../out_sha256.txt 2>/dev/null ); touch ALL_DONE; exit 1; }
log "DRIVER START kimi adress2020 cpu"
python3 -m pip install -q --break-system-packages "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" joblib > logs/pip.log 2>&1 || fail "pip install failed"
python3 -c "import sklearn,numpy,scipy,joblib,platform;print('VERSIONS sk',sklearn.__version__,'np',numpy.__version__,'sp',scipy.__version__,'joblib',joblib.__version__,'py',platform.python_version(),platform.platform())" >> $L 2>&1
python3 -c "import numpy;numpy.show_config()" > logs/numpy_config.txt 2>&1
grep -m1 "model name" /proc/cpuinfo >> $L; grep -q avx512f /proc/cpuinfo && log "avx512f present" || log "avx512f ABSENT"; log "nproc $(nproc)"
python3 -c "import sklearn,numpy,scipy;assert (sklearn.__version__,numpy.__version__,scipy.__version__)==('1.9.1','2.1.2','1.18.1'),'version mismatch'" || fail "pinned versions not installed"
touch PIP_DONE
sha256sum -c upload.sha256 > logs/upload_check.log 2>&1 || fail "upload sha256 check failed"
log "UPLOAD CHECK OK $(grep -c ': OK' logs/upload_check.log) files"
python3 p25a_nested5.py in/kimi_adress2020_encstates.npz adress2020 folds out/p26_kimi_adress2020 > logs/probe_adress2020.log 2>&1 || fail "probe failed: $(tail -2 logs/probe_adress2020.log | tr '\n' ' ')"
log "PROBE DONE: $(grep 'NESTED5 DONE' logs/probe_adress2020.log)"
python3 p25a_podverify.py in/kimi_adress2020_encstates.npz adress2020 folds out/p26_kimi_adress2020 > logs/podverify_adress2020.log 2>&1 || log "podverify crashed"
log "PODVERIFY: $(grep PODVERIFY logs/podverify_adress2020.log)"
( cd out && sha256sum * > ../out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
