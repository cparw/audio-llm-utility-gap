cd /workspace/p23a
while [ ! -f A_extract.RESULT ]; do sleep 15; done
cd gate && HF_HOME=/workspace/hf WINDOW_S=0 python3 extract_full.py /workspace/p23a/ref/mf_gate10_p23a.csv /workspace/p23a/gate/gate mdd > gate.log 2>&1
echo "gate exit $?" >> gate.log
