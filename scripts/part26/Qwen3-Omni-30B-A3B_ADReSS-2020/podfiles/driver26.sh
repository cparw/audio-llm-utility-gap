#!/bin/bash
# PART26 CPU pod driver, Qwen3-Omni-30B-A3B ADReSS-2020 encoder nested probe.
# 1 pinned pip install (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1); 2 wait for UPLOAD_OK and check every uploaded file
# against the Mac sha256 list; 3 run p25a_nested5.py (PART25 A script, unchanged) on the saved fold files, one BLAS
# thread per fit; 4 run p25a_podverify.py (PART25 A pod-side independent check, unchanged); 5 sha256 of out/; ALL_DONE.
cd /workspace; mkdir -p logs out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore P25_NJ=${P25_NJ:-32}
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; ( cd out && sha256sum * > ../out_sha256.txt 2>/dev/null ); touch ALL_DONE; exit 1; }
log "DRIVER START"
python3 -m pip install -q --break-system-packages "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" joblib threadpoolctl > logs/pip.log 2>&1 || fail "pip install failed"
python3 - >> $L 2>&1 <<'PY' || fail "version check failed"
import sklearn, numpy, scipy, joblib, platform, os
from threadpoolctl import threadpool_info
print("VERSIONS sklearn", sklearn.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__, "joblib", joblib.__version__, "python", platform.python_version(), platform.platform())
assert (sklearn.__version__, numpy.__version__, scipy.__version__) == ("1.9.1", "2.1.2", "1.18.1"), "version pin mismatch"
print("THREADPOOLS", [(d.get("internal_api"), d.get("num_threads"), d.get("architecture")) for d in threadpool_info()])
print("ENV OMP", os.environ.get("OMP_NUM_THREADS"), "OPENBLAS", os.environ.get("OPENBLAS_NUM_THREADS"), "MKL", os.environ.get("MKL_NUM_THREADS"))
PY
grep -m1 "model name" /proc/cpuinfo >> $L; log "cpus $(nproc)"; grep -q avx512f /proc/cpuinfo && log "avx512f present" || log "avx512f ABSENT"
touch PIP_DONE
for i in $(seq 1 360); do [ -f UPLOAD_OK ] && break; sleep 5; done
[ -f UPLOAD_OK ] || fail "upload not done after 30 min"
sha256sum -c in_sha256.txt > logs/upload_check.log 2>&1 || fail "upload sha256 check failed"
log "UPLOAD CHECK OK $(grep -c ': OK' logs/upload_check.log) files"
python3 p25a_nested5.py in/q3o_adress2020_states.npz adress2020 folds out/p26_q3o_adress2020 > logs/probe.log 2>&1 || fail "probe failed: $(tail -2 logs/probe.log | tr '\n' ' ')"
log "PROBE DONE: $(grep 'NESTED5 DONE' logs/probe.log)"
python3 p25a_podverify.py in/q3o_adress2020_states.npz adress2020 folds out/p26_q3o_adress2020 > logs/podverify.log 2>&1 || log "podverify crashed"
log "PODVERIFY: $(grep PODVERIFY logs/podverify.log)"
( cd out && sha256sum * > ../out_sha256.txt ); cp out_sha256.txt logs/out_sha256.txt
log "ALL DONE"
touch ALL_DONE
