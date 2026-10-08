#!/bin/bash
# PART 27 B pod driver. Waits for setup.sh (HF model, E-DAIC fetch, pip), pins the pod library versions, rebuilds the
# PART 23 whole windows with build_windows_p23a.py unchanged, extracts the 400 Hz low-pass states (GPU), then the unfiltered
# same-GPU control (GPU) while the low-pass probes run (CPU), then the control probes, then the report. Touches ALL_DONE or FAILED.
cd /workspace
L=/workspace/logs/DRIVER.log
log(){ echo "$* $(date -u +%H:%M:%S)" >> $L; }
fail(){ log "FAILED: $*"; echo "$*" > /workspace/FAILED; exit 1; }
log "DRIVER_START"
while true; do
  [ -f PIP_DONE ] && [ -f FETCH_DONE ] && [ -f HF_DONE ] && break
  if [ -f PIP_DONE ] && [ -f FETCH_DONE ] && ! pgrep -f "bash setup.sh" >/dev/null && [ ! -f HF_DONE ]; then fail "HF download did not finish"; fi
  sleep 10
done
log "SETUP_MARKERS_OK"
grep -q "FETCH_DONE ok 275 fails 0 wav_on_disk 275" logs/fetch.log || fail "fetch not 275/275: $(tail -1 logs/fetch.log)"
V=$(python3 -c "import sklearn,numpy,scipy;print(sklearn.__version__,numpy.__version__,scipy.__version__)")
if [ "$V" != "1.9.1 2.1.2 1.18.1" ]; then
  log "PIN: found $V, installing scikit-learn==1.9.1 numpy==2.1.2 scipy==1.18.1"
  python3 -m pip install --break-system-packages -q "scikit-learn==1.9.1" "numpy==2.1.2" "scipy==1.18.1" > logs/pip_pin.log 2>&1
  V=$(python3 -c "import sklearn,numpy,scipy;print(sklearn.__version__,numpy.__version__,scipy.__version__)")
  [ "$V" = "1.9.1 2.1.2 1.18.1" ] || fail "pin failed: $V"
fi
log "LIBS sklearn numpy scipy $V"
ln -sfn /workspace/edaicfull /workspace/p23a/edaicfull
cd /workspace/p23a
python3 build_windows_p23a.py > /workspace/logs/build.log 2>&1
grep -q "built 275 / 275   failures 0" /workspace/logs/build.log || fail "build: $(head -2 /workspace/logs/build.log)"
( cd /workspace/edaicfull/cut && sha256sum *.wav > /workspace/edaicfull/cut_sha256.txt )
log "BUILD_OK"; touch /workspace/BUILD_OK
export HF_HOME=/workspace/hf
TAG=lp400 LP_HZ=400 python3 p27b_extract.py > /workspace/logs/extract_lp400.log 2>&1 || fail "extract lp400 exit $?"
[ -f A_extract_lp400.RESULT ] || fail "extract lp400 no RESULT"
log "EXTRACT_LP400_DONE $(cat A_extract_lp400.RESULT)"
# CPU: low-pass probes; GPU at the same time: unfiltered same-GPU control extraction
mkdir -p probe_lp400 probe_unf
( ln -sfn shards_lp400 shards && ln -sfn probe_lp400 probe
  for k in enc proj llm ans; do python3 p23a_probe.py probe $k > /workspace/logs/probe_lp400_$k.log 2>&1 & done; wait
  echo "PROBES_LP400_DONE $(date -u +%H:%M:%S)" >> $L ) &
PPID_LP=$!
TAG=unf LP_HZ=0 python3 p27b_extract.py > /workspace/logs/extract_unf.log 2>&1 || fail "extract unf exit $?"
[ -f A_extract_unf.RESULT ] || fail "extract unf no RESULT"
log "EXTRACT_UNF_DONE $(cat A_extract_unf.RESULT)"
wait $PPID_LP
for k in enc proj llm ans; do [ -f probe_lp400/${k}_nested.RESULT ] || fail "lp400 probe $k missing"; done
ln -sfn shards_unf shards && ln -sfn probe_unf probe
for k in enc proj llm ans; do python3 p23a_probe.py probe $k > /workspace/logs/probe_unf_$k.log 2>&1 & done; wait
for k in enc proj llm ans; do [ -f probe_unf/${k}_nested.RESULT ] || fail "unf probe $k missing"; done
log "PROBES_UNF_DONE"
rm -f shards probe
mkdir -p /workspace/out
P27_D=/workspace/p23a OUT=/workspace/out python3 p27b_report.py > /workspace/logs/report.log 2>&1 || fail "report exit $?"
cp A_extract_perclip_lp400.csv A_extract_perclip_unf.csv A_extract_meta_lp400.json A_extract_meta_unf.json /workspace/out/
mkdir -p /workspace/out/probe_lp400 /workspace/out/probe_unf
cp probe_lp400/* /workspace/out/probe_lp400/; cp probe_unf/* /workspace/out/probe_unf/
tar -cf /workspace/out/shards_lp400.tar shards_lp400; tar -cf /workspace/out/shards_unf.tar shards_unf
log "ALL_DONE"; touch /workspace/ALL_DONE
