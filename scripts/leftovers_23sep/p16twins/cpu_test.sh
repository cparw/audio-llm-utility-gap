#!/bin/bash
# PC-GITA encoder per-layer refit on this CPU, default OpenBLAS kernel and forced kernels, to test CPU-kernel dependence
cd /workspace
grep -m1 "model name" /proc/cpuinfo > out/cpu_model.txt
python3 -c "import numpy; numpy.show_config()" > out/numpy_config.txt 2>&1
python3 -c "import numpy, threadpoolctl, json; print(json.dumps(threadpoolctl.threadpool_info(), indent=1))" > out/threadpool_info.txt 2>&1
python3 -u twin_perlayer_tag.py pcgita enc _default > logs/cpu_default.log 2>&1
for K in Haswell SkylakeX Zen Cooperlake SapphireRapids; do
  OPENBLAS_CORETYPE=$K python3 -u twin_perlayer_tag.py pcgita enc _$K > logs/cpu_$K.log 2>&1
done
touch CPU_TEST_DONE
