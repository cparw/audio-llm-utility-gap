"""Capture the pod environment: library versions, BLAS build and CPU, numpy SIMD dispatch."""
import json, os, sys, platform, subprocess, io, contextlib
import numpy, scipy, sklearn, joblib, threadpoolctl
out = dict(python=sys.version, executable=sys.executable, machine=platform.machine(), numpy=numpy.__version__, scipy=scipy.__version__,
           sklearn=sklearn.__version__, joblib=joblib.__version__, threadpoolctl=threadpoolctl.__version__,
           os_cpu_count=os.cpu_count(), env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_CORETYPE")})
try:
    import pandas; out["pandas"] = pandas.__version__
except Exception as e: out["pandas"] = repr(e)
out["threadpool_info"] = threadpoolctl.threadpool_info()
buf = io.StringIO()
with contextlib.redirect_stdout(buf): numpy.show_runtime()
out["numpy_show_runtime"] = buf.getvalue()
try:
    ls = subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines()
    out["lscpu"] = [l for l in ls if any(k in l for k in ("Model name", "CPU(s):", "Vendor", "Thread", "Core", "Socket", "Flags"))]
except Exception as e: out["lscpu"] = repr(e)
try:
    flags = open("/proc/cpuinfo").read().split("flags")[1].split("\n")[0]
    out["cpu_flags_simd"] = sorted({f for f in flags.split() if f.startswith(("avx", "sse4", "fma", "amx"))})
except Exception as e: out["cpu_flags_simd"] = repr(e)
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
print("ENV", out["numpy"], out["scipy"], out["sklearn"], [l for l in out.get("lscpu", []) if "Model name" in l], out["threadpool_info"][0].get("architecture") if out["threadpool_info"] else None)
