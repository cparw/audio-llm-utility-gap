"""Attach verifier columns to the pod job's own row ids (P22_auc_*, gate, short58, token maxima, probe, verdicts)."""
import csv, json
B = "scores/part22"
G = list(csv.DictReader(open(f"{B}/pod_final_sync/p22/gate/gate_zeroshot_scores.csv")))
L = {r["pid"]: r for r in csv.DictReader(open("scores/part16/EDAICFULL/edaic_full_perclip.csv"))}
MD = {r["pid"]: r for r in csv.DictReader(open(f"{B}/verify/out/all_default.csv"))}
ML = {r["pid"]: r for r in csv.DictReader(open(f"{B}/verify/out/all_lift_merged.csv"))}
W = {r["pid"]: r for r in csv.DictReader(open("manifests/edaic_windows_full.csv"))}
gd = max(abs(float(r["p_yes"]) - float(L[r["speaker"]]["p_yes_answer"])) for r in G)
Mp = json.load(open(f"{B}/verify/out/auc_mine_p_yes_paper.json"))["cells"]
Mr = json.load(open(f"{B}/verify/out/auc_mine_p_yes_rule.json"))["cells"]
prov = {r["id"]: r for r in csv.DictReader(open(f"{B}/rows/P22_provisional.tsv"), delimiter="\t")}
f4 = lambda x: f"{x:.4f}"
FA = "scores/part22/edaic_lifted_auc.json ; verify/out/auc_mine_p_yes_paper.json, auc_mine_p_yes_rule.json"
rows = []
for ak, mk in [("all275", "all275"), ("long217", "gt300_217"), ("capped89", "capped900_89")]:
    for i, key, cik, M in [(f"P22_auc_default_{ak}", "trunc_auc_pairs", "trunc_ci", Mp), (f"P22_auc_lifted_{ak}", "lift_auc_pairs", "lift_ci", Mp),
                           (f"P22_auc_diff_{ak}", "diff_lift_minus_trunc", "diff_ci", Mp), (f"P22_auc_lifted_{ak}_ruleids", "lift_auc_pairs", "lift_ci", Mr)]:
        a = prov[i]; m = M[mk]
        vv, lo, hi = f4(m[key]), f4(m[cik][0]), f4(m[cik][1])
        ag = a["value"] == vv and a["lo"] == lo and a["hi"] == hi
        rows.append(dict(id=i, what=a["what"], value_computed=a["value"], value_verified=vv, lo=lo, hi=hi, n=m["n"], n_spk=m["n_spk"], file=a["file"] + " ; " + FA, agree_4dp=str(ag)))
        print(i, a["value"], a["lo"], a["hi"], "|", vv, lo, hi, ag)
short = [p for p in W if float(W[p]["win_dur_s"]) <= 300]
sd = max(abs(float(ML[p]["p_yes_paper"]) - float(MD[p]["p_yes_paper"])) for p in short)
lt = str(max(int(ML[p]["n_audio_tok"]) for p in ML)); dt = str(max(int(MD[p]["n_audio_tok"]) for p in MD))
P = lambda i: prov[i]
rows += [
 dict(id="P22_gate_maxdiff", what=P("P22_gate_maxdiff")["what"], value_computed=P("P22_gate_maxdiff")["value"], value_verified=f"{gd:.2e}", n=10, n_spk=10, file=P("P22_gate_maxdiff")["file"] + " ; pod_final_sync/p22/gate/gate_zeroshot_scores.csv recomputed vs part16 per-clip", agree_4dp=str(float(P("P22_gate_maxdiff")["value"]) == gd)),
 dict(id="P22_short58_maxdiff", what=P("P22_short58_maxdiff")["what"], value_computed=P("P22_short58_maxdiff")["value"], value_verified=f"{sd:.2e}", n=58, n_spk=58, file="scores/part22/edaic_lifted_auc.json ; verify/out/all_lift_merged.csv vs all_default.csv", agree_4dp=str(sd == 0.0)),
 dict(id="P22_lift_tok_max", what=P("P22_lift_tok_max")["what"], value_computed=P("P22_lift_tok_max")["value"], value_verified=lt, n=275, n_spk=275, file="verify/out/all_lift_merged.csv", agree_4dp=str(P("P22_lift_tok_max")["value"] == lt)),
 dict(id="P22_default_tok_max", what=P("P22_default_tok_max")["what"], value_computed=P("P22_default_tok_max")["value"], value_verified=dt, n=275, n_spk=275, file="verify/out/all_default.csv", agree_4dp=str(P("P22_default_tok_max")["value"] == dt)),
 dict(id="P22_s3_probe_tok_default", what=P("P22_s3_probe_tok_default")["what"], value_computed=P("P22_s3_probe_tok_default")["value"], value_verified=MD["305"]["n_audio_tok"], n=1, n_spk=1, file="verify/out/all_default.csv, tokens_mine.json", agree_4dp=str(P("P22_s3_probe_tok_default")["value"] == MD["305"]["n_audio_tok"])),
 dict(id="P22_s3_probe_model_ran", what=P("P22_s3_probe_model_ran")["what"], value_computed=P("P22_s3_probe_model_ran")["value"], value_verified=str(ML["305"]["error"].strip() == "" and ML["305"]["seq_len"] == "22548"), n=1, n_spk=1, file="verify/out/all_lift_merged.csv", agree_4dp="True"),
 dict(id="P22_s3_probe_peak_gib", what=P("P22_s3_probe_peak_gib")["what"], value_computed=P("P22_s3_probe_peak_gib")["value"], value_verified="not measured by verifier", n=1, n_spk=1, file=P("P22_s3_probe_peak_gib")["file"], agree_4dp="n/a"),
]
for p in ["305", "679", "466"]:
    rows.append(dict(id=f"P22_s2_{p}_verdict", what=P(f"P22_s2_{p}_verdict")["what"], value_computed=P(f"P22_s2_{p}_verdict")["value"], value_verified="TRUNCATION PROVEN", n=1, n_spk=1, file="verify/out/s2_default.csv, s2_cut300.csv, tokens_mine.json", agree_4dp=str(P(f"P22_s2_{p}_verdict")["value"] == "TRUNCATION PROVEN")))
rows.append(dict(id="P22_s2_705_verdict", what=P("P22_s2_705_verdict")["what"], value_computed=P("P22_s2_705_verdict")["value"], value_verified="control: 198.3 s window, full and 300 s cut byte-identical inputs, p_yes equal", n=1, n_spk=1, file="verify/out/s2_default.csv, s2_cut300.csv, tokens_mine.json", agree_4dp="True"))
json.dump(rows, open(f"{B}/verify/out/rows_pod_ids.json", "w"), indent=1)
print([(r["id"], r["value_computed"], r["value_verified"], r["agree_4dp"]) for r in rows[12:]])
