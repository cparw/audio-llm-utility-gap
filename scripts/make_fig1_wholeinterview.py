# Figure 1, option C from fig_options.py, with the E-DAIC panel swapped to the whole-interview run (PART 23).
# PC-GITA and Pitt panels unchanged. E-DAIC answer line = whole-window zero-shot from the same run (0.8366).
import sys, matplotlib
sys.path.insert(0, "scripts")
import fig_options as fo
matplotlib.rcParams["font.serif"] = ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"]
W = "scores/part23/"
fo.PATHS["E-DAIC"] = (W + "o25_whole_encoder_perlayer.csv", W + "o25_whole_llm_perlayer.csv")
fo.ANSWER["E-DAIC"] = 0.8366
fo.option_C(fo.load(), sys.argv[1])
print("wrote", sys.argv[1])
