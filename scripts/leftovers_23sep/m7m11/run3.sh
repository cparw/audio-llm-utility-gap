cd /workspace; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
echo "start $(date -u +%H:%M:%S)" > logs/DRIVER.log
python3 -m pip install --break-system-packages -q scikit-learn==1.9.1 numpy==2.1.2 scipy==1.18.1 pandas > logs/pip.log 2>&1
python3 -c "import sklearn,numpy,scipy,pandas;print(sklearn.__version__,numpy.__version__,scipy.__version__,pandas.__version__)" > logs/versions.txt 2>&1; touch PIP_DONE
python3 m7m11_verify.py data out/m7m11_verify.csv > logs/verify.log 2>&1 && echo "verify done $(date -u +%H:%M:%S)" >> logs/DRIVER.log || echo "verify FAILED" > FAILED
sha256sum out/* > out/sha256.txt
touch ALL_DONE
