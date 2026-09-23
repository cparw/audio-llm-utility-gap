#!/bin/bash
cd /workspace; mkdir -p out logs
python3 -m pip install --break-system-packages -q scikit-learn==1.9.1 numpy==2.1.2 scipy==1.18.1 pandas joblib > logs/pip.log 2>&1 && touch PIP_DONE
python3 -c "import sklearn,numpy,scipy,platform;print(sklearn.__version__,numpy.__version__,scipy.__version__,platform.python_version())" >> logs/pip.log 2>&1
python3 vf_spread.py all 0 200 > logs/DRIVER.log 2>&1 || echo "run failed" > FAILED
cp -f /workspace/lmh/meta.json out/lmh_meta.json 2>/dev/null
lscpu > out/lscpu.txt 2>/dev/null
cd out && sha256sum * > SHA256SUMS.txt; cd ..
touch ALL_DONE
