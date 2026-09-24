#!/bin/bash
# PART26 CPU pod driver, cell Qwen3-Omni-30B-A3B x NeuroVoz.
# 1 wait for setup and the upload; 2 check every input against the Mac sha256 list; 3 record versions and BLAS threads;
# 4 nested five-repeat encoder probe on the saved fold files (p25a_nested5.py, unchanged PART25 A script);
# 5 pod-side independent check (p25a_podverify.py, unchanged); ALL_DONE.
cd /workspace; mkdir -p logs out
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore P25_NJ=${P25_NJ:-30} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=logs/DRIVER.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > FAILED; touch ALL_DONE; exit 1; }
log "DRIVER START q3o neurovoz P25_NJ=$P25_NJ"
for i in $(seq 1 120); do [ -f PIP_DONE ] && [ -f DATA_OK ] && break; sleep 10; done
[ -f PIP_DONE ] && [ -f DATA_OK ] || fail "setup or upload not ready after 20 min"
cp inputs.sha256 logs/inputs.sha256
sha256sum -c inputs.sha256 > logs/input_check.log 2>&1 || fail "input sha256 check failed"
log "INPUT CHECK OK $(grep -c ': OK' logs/input_check.log) files"
python3 - > out/p26_env.json 2> logs/env_check.err <<'PY' || fail "env check failed"
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import json, platform, numpy, scipy, sklearn, joblib, threadpoolctl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
info = threadpoolctl.threadpool_info()
print(json.dumps({"python": platform.python_version(), "platform": platform.platform(), "numpy": numpy.__version__,
                  "scipy": scipy.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__,
                  "threadpoolctl": threadpoolctl.__version__, "cpu_count": os.cpu_count(), "threadpools": info}, indent=1))
assert (numpy.__version__, scipy.__version__, sklearn.__version__) == ("2.1.2", "1.18.1", "1.9.1")
assert info and all(p["num_threads"] == 1 for p in info)
PY
log "ENV OK $(python3 -c "import json;d=json.load(open('out/p26_env.json'));print('sk',d['sklearn'],'np',d['numpy'],'sp',d['scipy'],'threads',[p['num_threads'] for p in d['threadpools']],'cpus',d['cpu_count'])")"
python3 p25a_nested5.py data/q3o_neurovoz_encstates.npz neurovoz folds out/p26_q3o_neurovoz > logs/probe.log 2>&1 || fail "probe failed"
log "PROBE DONE: $(grep 'NESTED5 DONE' logs/probe.log)"
python3 p25a_podverify.py data/q3o_neurovoz_encstates.npz neurovoz folds out/p26_q3o_neurovoz > logs/podverify.log 2>&1 || log "podverify crashed"
log "PODVERIFY: $(grep PODVERIFY logs/podverify.log)"
( cd out && sha256sum * > ../logs/out_sha256.txt )
log "ALL DONE"
touch ALL_DONE
