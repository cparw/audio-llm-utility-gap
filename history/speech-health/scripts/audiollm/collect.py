import json, glob, os

CELLS = ["dcaps_interview", "kcl_read", "italian_read", "neurovoz_vowel",
         "pcgita_vowel", "pcgita_read_v2check", "dcaps_interview300", "kcl_read300", "italian_read300"]
OUT = "/scratch1/parwatka/scores"

hdr = ("%-22s %-6s %-6s %-6s %-8s %-8s %-8s %-8s %-6s" %
       ("cell", "model", "n", "spk", "AUC_prob", "AUC_word", "frac_yes", "mass", "fail"))
print(hdr)
print("-" * len(hdr))
for c in CELLS:
    for mk in ("af3", "kimi"):
        f = os.path.join(OUT, "%s_%s_v2_summary.json" % (c, mk))
        if not os.path.exists(f):
            print("%-22s %-6s  (not present)" % (c, mk)); continue
        d = json.load(open(f))
        fmt = lambda x: ("%.3f" % x) if isinstance(x, (int, float)) else "None"
        print("%-22s %-6s %-6s %-6s %-8s %-8s %-8s %-8s %-6s" % (
            c, mk, d["n_clips"], d["n_speakers"],
            fmt(d["auc_prob"]), fmt(d["auc_word"]),
            fmt(d["frac_yes_word"]), fmt(d["answer_mass_mean"]), d["n_failed"]))
print()
print("full json dump")
for c in CELLS:
    for mk in ("af3", "kimi"):
        f = os.path.join(OUT, "%s_%s_v2_summary.json" % (c, mk))
        if os.path.exists(f):
            print("=====", os.path.basename(f))
            print(json.dumps(json.load(open(f)), indent=1))
