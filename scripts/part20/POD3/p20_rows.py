import json, os
D = "scores/part20/POD3"; F = f"{D}/p14_ft966_oof.csv"
T = "reports/part20/rows/POD3_provisional.tsv"
s = json.load(open(f"{D}/p14_ft966_oof.sidecar.json"))["results"]
what = {"conflict": "Qwen2.5-Omni projector FT on 966 Part14 clips (OOF by speaker), conflict AUC",
        "agreement483": "same FT, agreement AUC on 483 rows",
        "agreement311": "same FT, agreement AUC on 311 distinct segments (paper convention)",
        "diff_conflict_minus_agreement483": "same FT, paired conflict minus agreement (483 rows)",
        "diff_conflict_minus_agreement311": "same FT, paired conflict minus agreement (311 distinct)"}
ids = {"conflict": "P20_POD3_ft966_conflict", "agreement483": "P20_POD3_ft966_agree483", "agreement311": "P20_POD3_ft966_agree311",
       "diff_conflict_minus_agreement483": "P20_POD3_ft966_diff483", "diff_conflict_minus_agreement311": "P20_POD3_ft966_diff311"}
new = not os.path.exists(T)
with open(T, "a") as fh:
    if new: fh.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\ttwo_dp\tvs_half\n")
    for k in ids:
        v = s[k]
        if k.startswith("diff"): vs = "interval excludes zero" if (v["lo"] > 0 or v["hi"] < 0) else "interval includes zero"
        else: vs = "above 0.5" if v["auc"] > 0.5 else "below 0.5"
        fh.write(f"{ids[k]}\t{what[k]}\t{v['auc']:.4f}\t{v['lo']:.4f}\t{v['hi']:.4f}\t{v['n']}\t{v['n_spk']}\t{F}\t{v['auc']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]\t{vs}\n")
print(open(T).read())
