#!/bin/bash
# waits for all 25 original-window chunks, verifies, extracts, verifies each wav, then scores and re-runs the analysis
cd /root/och
until [ "$(sha256sum -c /workspace/p20/ochunks.sha256 2>/dev/null | grep -c ': OK$')" = "25" ]; do sleep 20; done
echo "chunks OK $(date -u)"
cat o_* > /root/orig468.tar
echo "280f547333b9a9eefbff030e12ca4219aa9849f133ac9d737de8aa53b1f463ab  /root/orig468.tar" | sha256sum -c || exit 1
mkdir -p /workspace/p20/orig && cd /workspace/p20/orig && tar xf /root/orig468.tar --no-same-owner
cd /workspace/p20/orig/segments && sha256sum -c /workspace/p20/source_segments.sha256 | grep -c ': OK$'
[ "$(sha256sum -c /workspace/p20/source_segments.sha256 2>/dev/null | grep -c ': OK$')" = "468" ] || { echo "WAV SHA FAIL"; exit 1; }
echo "468 wavs verified $(date -u)"
cd /workspace/p20 && export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 && python3 score_orig468.py > score_orig.log 2>&1 || { echo "SCORE FAIL"; exit 1; }
python3 analyze_noinv.py > analyze2.log 2>&1 || { echo "ANALYSIS FAIL"; exit 1; }
echo "ALL ORIG DONE $(date -u)"
