"""PART 22 step 1 verifier: (1) every quoted code line in the md equals the source file line; (2) independent
re-derivation of the token numbers; (3) processor probe json and per-clip csv consistency."""
import re, json, csv, math, sys
R = "<local data dir>"; S = f"{R}/scores/part22/src"
md = open(f"{R}/scores/part22/P22_step1_processor_config.md").read()
def resolve(p):
    if p.startswith("transformers/"): return f"{S}/tf5170/{p}"
    if p.startswith("src/"): return f"{S}/{p[4:]}"
    return f"{R}/{p}"
blocks = re.findall(r"`([^`\n]+):(\d+)-(\d+)`\n```\w+\n(.*?)\n```", md, re.S)
nl = 0; bad = []
for path, a, b, body in blocks:
    src = open(resolve(path)).read().split("\n")
    rows = body.split("\n")
    if len(rows) != int(b) - int(a) + 1: bad.append((path, a, b, "row count")); continue
    for row in rows:
        m = re.match(r"\s*(\d+)  (.*)$", row, re.S)
        n, txt = int(m.group(1)), m.group(2)
        nl += 1
        if src[n-1] != txt: bad.append((path, n))
print(f"quote check: {len(blocks)} blocks, {nl} lines, mismatches {len(bad)} {bad[:5]}")
def tok_frames(fr): il = (fr - 1) // 2 + 1; return (il - 2) // 2 + 1
chk = {"300s_cap": tok_frames(4800000 // 160), "900s_nocap": tok_frames(14400000 // 160), "600s_nocap": tok_frames(9600000 // 160)}
print("re-derived tokens:", chk, "expect 7500 / 22500 / 15000:", chk == {"300s_cap": 7500, "900s_nocap": 22500, "600s_nocap": 15000})
pj = json.load(open(f"{R}/scores/part22/P22_step1_proc_probe.json"))
ok = all(v["n_audio_tok"] == tok_frames(v["mask_frames"]) and v["seq_len"] == v["n_audio_tok"] + 48 for v in pj.values())
print("probe json: tokens == formula(mask_frames) and seq == tok+48 for all", len(pj), "rows:", ok)
print("probe 900 s paper call ->", pj["A_paper_call|900.0"]["n_audio_tok"], "| truncation off ->", pj["C_audio_kwargs_truncation_False|900.0"]["n_audio_tok"])
w = {r["pid"]: r for r in csv.DictReader(open(f"{R}/edaic_rerun/variants/windows_full.csv"))}
rows = list(csv.DictReader(open(f"{R}/edaic_rerun/part16/EDAICFULL/full_zeroshot_scores.csv")))
m = sum(tok_frames(math.ceil(min(round(float(w[r["speaker"]]["win_dur_s"]) * 16000), 4800000) / 160)) == int(r["n_audio_tok"]) for r in rows)
long = [r for r in rows if float(w[r["speaker"]]["win_dur_s"]) > 300]
cap = [r for r in rows if w[r["speaker"]]["capped"] == "1"]
print(f"per-clip csv: formula match {m}/{len(rows)}; win_dur>300: {len(long)} all 7500: {all(int(r['n_audio_tok'])==7500 for r in long)}; capped==1: {len(cap)}; max tok {max(int(r['n_audio_tok']) for r in rows)}")
sys.exit(0 if (not bad and ok and m == len(rows)) else 1)
