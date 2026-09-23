#!/bin/bash
# usage: drive.sh DS  -> runs enc, llm, ans with default OpenBLAS kernel, forced SkylakeX (AVX512) and forced Haswell (AVX2)
DS=$1; cd /workspace
grep -m1 "model name" /proc/cpuinfo > out/cpu_model.txt
python3 -c "import numpy, threadpoolctl, json; print(json.dumps(threadpoolctl.threadpool_info(), indent=1))" > out/threadpool_info.txt 2>&1
for K in default SkylakeX Haswell; do
  for ST in enc llm ans; do
    if [ $K = default ]; then python3 -u v_stage.py in/q3o_${DS}_states.npz $ST ${DS}_${ST}_default > logs/${DS}_${ST}_default.log 2>&1
    else OPENBLAS_CORETYPE=$K python3 -u v_stage.py in/q3o_${DS}_states.npz $ST ${DS}_${ST}_$K > logs/${DS}_${ST}_$K.log 2>&1; fi
  done
done
touch ${DS}_DONE
