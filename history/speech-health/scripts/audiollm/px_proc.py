import pandas as pd, numpy as np
P="/project2/msoleyma_946/speech_health/results_chaitanya/committed/paradox"
A=pd.read_csv(f"{P}/audio_alignment_audit.csv",index_col=0)
A["ratio"]=A.dcaps_proc_s/A.tar_audio_s
print("dcaps_proc / tarball-audio duration ratio over 275 sessions:")
print("  median %.3f  p5 %.3f  p95 %.3f  n>1.0 (longer than source) %d"%(
  A.ratio.median(),A.ratio.quantile(.05),A.ratio.quantile(.95),(A.ratio>1.0).sum()))
print("  n>1.05: %d   n>1.5: %d   max %.2f"%((A.ratio>1.05).sum(),(A.ratio>1.5).sum(),A.ratio.max()))
S=pd.read_csv(f"{P}/daic_segments_scored.csv")
