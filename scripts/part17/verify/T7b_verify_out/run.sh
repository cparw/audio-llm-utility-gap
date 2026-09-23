set -x
cd /workspace/verify_t7b
date -u
python3 -c "import sklearn,numpy,scipy,joblib,threadpoolctl,sys;print(sys.version.split()[0],sklearn.__version__,numpy.__version__,scipy.__version__,joblib.__version__,threadpoolctl.__version__)"
env | grep -E "THREADS" || echo no_thread_env
python3 -W ignore T7b_verify.py stack
VT_NJOBS=14 VT_TAG=main python3 -W ignore T7b_verify.py probe llm enc ans proj
VT_NJOBS=14 VT_TAG=final1 VT_FINAL_THREADS=1 python3 -W ignore T7b_verify.py probe llm enc ans proj
date -u
echo VERIFY_PROBES_DONE
