cd /workspace; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
echo "start $(date -u +%H:%M:%S)" > logs/DRIVER.log
python3 pod1_adress2020_rep5_refit.py data/adress2020_orig_enc.npz out/adress2020_pod4b > logs/refit.log 2>&1 && echo "refit done $(date -u +%H:%M:%S)" >> logs/DRIVER.log || echo "refit FAILED" > FAILED
sha256sum data/adress2020_orig_enc.npz out/* > out/sha256.txt
touch ALL_DONE
