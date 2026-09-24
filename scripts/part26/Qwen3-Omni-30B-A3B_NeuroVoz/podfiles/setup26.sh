#!/bin/bash
# PART26 CPU pod setup: the pinned fit libraries of PART25 A (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1). Marker PIP_DONE.
cd /workspace; mkdir -p logs out
PIP="python3 -m pip install --break-system-packages -q"
{ $PIP "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" joblib threadpoolctl \
  && python3 -c "import sklearn,numpy,scipy,joblib,platform,os;print('VERSIONS sk',sklearn.__version__,'np',numpy.__version__,'sp',scipy.__version__,'joblib',joblib.__version__,'py',platform.python_version(),platform.platform(),'cpus',os.cpu_count())" \
  && touch /workspace/PIP_DONE; } > logs/pip.log 2>&1
echo "$(date -u +%H:%M:%S) SETUP_END PIP=$([ -f PIP_DONE ] && echo ok)" >> logs/DRIVER.log
