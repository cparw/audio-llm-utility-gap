cd /workspace; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
echo "start $(date -u +%H:%M:%S)" > logs/DRIVER.log
python3 m7m11_verify.py data out/m7m11_verify.csv > logs/verify.log 2>&1 && echo "verify done $(date -u +%H:%M:%S)" >> logs/DRIVER.log || echo "verify FAILED" > FAILED
sha256sum out/* > out/sha256.txt
touch ALL_DONE
