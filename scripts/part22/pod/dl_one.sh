#!/bin/bash
pid=$1; D=/workspace/edaicfull
[ -s $D/wav/${pid}_AUDIO.wav ] && [ -s $D/tr/${pid}_Transcript.csv ] && exit 0
for a in 1 2 3 4; do
  mkdir -p $D/tmp/$pid
  curl -sf --retry 3 https://dcapswoz.ict.usc.edu/wwwedaic/data/${pid}_P.tar.gz | tar -xz -C $D/tmp/$pid --wildcards "*_AUDIO.wav" "*_Transcript.csv" 2>>$D/tmp/err_$pid.log
  if [ -s $D/tmp/$pid/${pid}_P/${pid}_AUDIO.wav ] && [ -s $D/tmp/$pid/${pid}_P/${pid}_Transcript.csv ]; then
    mv $D/tmp/$pid/${pid}_P/${pid}_AUDIO.wav $D/wav/; mv $D/tmp/$pid/${pid}_P/${pid}_Transcript.csv $D/tr/; rm -rf $D/tmp/$pid; echo "OK $pid"; exit 0
  fi
  rm -rf $D/tmp/$pid; sleep 3
done
echo "FAIL $pid"
