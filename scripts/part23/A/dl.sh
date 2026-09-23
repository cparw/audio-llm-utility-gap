#!/bin/bash
set -o pipefail
pid=$1; D=/workspace/p23a/edaicfull
[ -f $D/tmp/$pid.ok ] && exit 0
for try in 1 2 3 4 5 6; do
  rm -rf $D/tmp/$pid; mkdir -p $D/tmp/$pid
  if curl -sfL --retry 3 --connect-timeout 30 https://dcapswoz.ict.usc.edu/wwwedaic/data/${pid}_P.tar.gz | tar --no-same-owner -xzf - -C $D/tmp/$pid --wildcards "*_AUDIO.wav" "*_Transcript.csv" 2>>$D/tmp/$pid.err; then
    w=$(find $D/tmp/$pid -name "${pid}_AUDIO.wav" | head -1); t=$(find $D/tmp/$pid -name "${pid}_Transcript.csv" | head -1)
    if [ -n "$w" ] && [ -n "$t" ]; then mv "$w" $D/wav/; mv "$t" $D/tr/; rm -rf $D/tmp/$pid; touch $D/tmp/$pid.ok; echo "OK $pid try $try"; exit 0; fi
  fi
  echo "RETRY $pid try $try"; sleep 5
done
echo "FAIL $pid"
