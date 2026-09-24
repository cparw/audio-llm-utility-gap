"""Aggregate the PCLM runs: method table, attention profiles, cross-dataset agreement."""
import json, glob, os
import numpy as np

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/committed"
ORDER = [("pcgita", "read"), ("neurovoz", "read"), ("dcaps", "interview")]
METHODS = ["single_nested", "single_oracle", "mean_layers", "concat_layers",
           "mixer_static", "mixer_inputcond"]

runs = {}
for ds, task in ORDER:
    p = f"{OUT}/pclm_{ds}_{task}.json"
    if os.path.exists(p):
        runs[f"{ds}:{task}"] = json.load(open(p))

lines = []
A = lines.append
A("# PCLM on clinical speech: learned layer weighting vs honest single-layer selection")
A("")
A("Qwen2-Audio encoder, 33 layers, mean-pooled per clip. GroupKFold(5) by speaker,")
A("zero speaker overlap asserted every fold. Pooled out-of-fold AUC. 95% CI from")
A("2000 speaker-clustered bootstrap resamples. Shuffle = mean AUC under speaker-level")
A("label permutation. perm p from that permutation null.")
A("")
A("## Main table")
A("")
A("| dataset | n clips | n spk | method | AUC | 95% CI | shuffle | perm p |")
A("|---|---|---|---|---|---|---|---|")
for k, r in runs.items():
    for m in METHODS:
        v = r["results"][m]
        A("| %s | %d | %d | %s | %.3f | %.3f-%.3f | %.3f | %.3f |"
          % (k, r["n_clips"], r["n_speakers"], m, v["auc"],
             v["ci95"][0], v["ci95"][1], v["null_mean"], v["perm_p"]))
A("")
A("## The question: does the mixer beat nested single-layer selection?")
A("")
A("| dataset | comparison | delta AUC | 95% CI | P(delta>0) |")
A("|---|---|---|---|---|")
for k, r in runs.items():
    for dk, dv in r["deltas"].items():
        A("| %s | %s | %+.4f | %+.4f..%+.4f | %.3f |"
          % (k, dk.replace("_", " "), dv["mean"], dv["lo"], dv["hi"], dv["frac_gt0"]))
A("")
A("## Where the attention mass sits")
A("")
A("| dataset | mixer | peak layer | peak weight | peak/uniform | entropy/max | mass L0-3 | mass L4-15 | mass L16-32 |")
A("|---|---|---|---|---|---|---|---|---|")
prof = {}
U = 1.0 / 33
for k, r in runs.items():
    for name in ["attn_static", "attn_inputcond"]:
        a = np.array(r[name])
        a = a / a.sum()
        prof[(k, name)] = a
        ent = -(a * np.log(a + 1e-12)).sum() / np.log(33)
        A("| %s | %s | %d | %.4f | %.2fx | %.3f | %.3f | %.3f | %.3f |"
          % (k, name.replace("attn_", ""), int(a.argmax()), a.max(), a.max() / U,
             ent, a[:4].sum(), a[4:16].sum(), a[16:].sum()))
A("")
A("Uniform weight would be %.4f per layer; entropy/max = 1.000 means perfectly flat." % U)
A("")
A("This is the central diagnostic. The learned static attention is close to uniform,")
A("so the mixer is doing little more than averaging the layers, which is why")
A("mixer_static and mean_layers land on nearly the same AUC.")
A("")
A("## Do the peaks agree across datasets?")
A("")
ks = list(runs)
for name in ["attn_static", "attn_inputcond"]:
    A("### %s" % name.replace("attn_", ""))
    A("")
    A("| | " + " | ".join(ks) + " |")
    A("|---|" + "---|" * len(ks))
    for i in ks:
        row = []
        for j in ks:
            c = np.corrcoef(prof[(i, name)], prof[(j, name)])[0, 1]
            row.append("%.3f" % c)
        A("| %s | %s |" % (i, " | ".join(row)))
    A("")
A("Pearson r between the per-dataset 33-length attention profiles. Values near 0")
A("mean the mixers are not agreeing on a shared readout depth.")
A("")
A("## Single-layer probe AUC per layer (for reference, honest OOF, no selection)")
A("")
A("| dataset | argmax layer | max AUC | AUC at L0 | AUC at L16 | AUC at L32 |")
A("|---|---|---|---|---|---|")
for k, r in runs.items():
    pl = np.array(r["per_layer_auc"])
    A("| %s | %d | %.3f | %.3f | %.3f | %.3f |"
      % (k, int(pl.argmax()), pl.max(), pl[0], pl[16], pl[32]))
A("")
A("Nested layer picks per outer fold:")
for k, r in runs.items():
    A("  %-20s %s   (oracle layer %d)" % (k, r["nested_layer_picks"], r["oracle_layer"]))
A("")

# ---- prompt-conditioned runs
A("## Prompt-conditioned mixer (attention weights conditioned on elicitation task)")
A("")
tc = {}
for ds in ["pcgita", "neurovoz"]:
    p = f"{OUT}/pclm_taskcond_{ds}.json"
    if os.path.exists(p):
        tc[ds] = json.load(open(p))
if not tc:
    A("(not available)")
for ds, r in tc.items():
    A("")
    A("### %s, all tasks pooled (%d clips, %d speakers, tasks %s)"
      % (ds, r["n_clips"], r["n_speakers"], ", ".join(r["tasks"])))
    A("")
    A("| method | AUC | 95% CI | shuffle | perm p |")
    A("|---|---|---|---|---|")
    for m in ["single_nested", "mean_layers", "mixer_static", "mixer_taskcond"]:
        v = r["results"][m]
        A("| %s | %.3f | %.3f-%.3f | %.3f | %.3f |"
          % (m, v["auc"], v["ci95"][0], v["ci95"][1], v["null_mean"], v["perm_p"]))
    A("")
    A("| comparison | delta AUC | 95% CI | P(delta>0) |")
    A("|---|---|---|---|")
    for dk, dv in r["deltas"].items():
        A("| %s | %+.4f | %+.4f..%+.4f | %.3f |"
          % (dk.replace("_", " "), dv["mean"], dv["lo"], dv["hi"], dv["frac_gt0"]))
    A("")
    A("Per-task attention peaks learned by the prompt-conditioned mixer:")
    A("")
    A("| task | peak layer | peak weight | mass L0-3 | mass L4-15 | mass L16-32 |")
    A("|---|---|---|---|---|---|")
    for t, v in r["attn_taskcond"].items():
        v = np.array(v)
        A("| %s | %d | %.3f | %.3f | %.3f | %.3f |"
          % (t, int(v.argmax()), v.max(), v[:4].sum(), v[4:16].sum(), v[16:].sum()))

txt = "\n".join(lines)
open(f"{OUT}/PCLM_SUMMARY.md", "w").write(txt + "\n")
print(txt)
print("\nwrote %s/PCLM_SUMMARY.md" % OUT)
