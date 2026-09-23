"""PART 23 A report (pod, CPU). Writes one per-clip csv + one sidecar json per result, provisional rows, .RESULT markers.
usage: python3 p23a_report.py items12      (items 1 and 2: window rebuild check, audio tokens, zero-shot)
       python3 p23a_report.py item3        (item 3: nested probes mean-of-five, paired intervals, per-layer files, best stage)
Bootstrap: 2000 draws, fresh numpy default_rng(0) per cell, speakers resampled with replacement with the draw code of
probe_boot.py / T7b_boot.py (rng.choice over sorted unique speakers; one clip per speaker here), draws with one class
skipped (counted), percentiles 2.5 / 97.5. Paired cells recompute both quantities on the same draw; mean-of-five cells
recompute all five per-repeat AUCs inside each draw.
"""
import os, sys, csv, json, hashlib, platform, time, numpy as np

D = "/workspace/p23a"; OUT = f"{D}/out"; PR = f"{D}/probe"
os.makedirs(OUT, exist_ok=True)
GD = "scores/part23/A"   # file column: path relative to the repository root
MID = "Qwen/Qwen2.5-Omni-7B"
PROMPT = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
NB = 2000
MODE = sys.argv[1]
CMD = f"cd /workspace/p23a && python3 p23a_report.py {MODE}"

def auc_rank(y, s):  # identical to probe_boot.py
    y = np.asarray(y, float); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float); sv = s[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j+1] == sv[i]: j += 1
        r[o[i:j+1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[y == 1].sum() - n1*(n1+1)/2.0) / (n1*n0))

def sha1mb(p):
    with open(p, "rb") as fh: return hashlib.sha256(fh.read(1 << 20)).hexdigest()

def libs():
    import sklearn, scipy, joblib, pandas
    out = dict(python=sys.version.split()[0], numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
               joblib=joblib.__version__, pandas=pandas.__version__, platform=platform.platform())
    try:
        m = json.load(open(f"{D}/A_extract_meta.json"))["libs"]; out.update({f"extract_{k}": v for k, v in m.items()})
    except Exception: pass
    return out

WIN = list(csv.DictReader(open(f"{D}/edaicfull/windows_full.csv")))
PIDS = [r["pid"] for r in WIN]
P22 = {r["pid"]: r for r in csv.DictReader(open(f"{D}/ref/edaic_lifted_perclip.csv"))}
T7 = {r["pid"]: r for r in csv.DictReader(open(f"{D}/ref/T7b_edaic300_meanof5_perclip.csv"))}
P16 = {r["pid"]: r for r in csv.DictReader(open(f"{D}/ref/edaic_full_perclip.csv"))}
EX = {r["pid"]: r for r in csv.DictReader(open(f"{D}/A_extract_perclip.csv"))}
assert [r["pid"] for r in csv.DictReader(open(f"{D}/ref/T7b_edaic300_meanof5_perclip.csv"))] == PIDS, "clip order differs from T7b"
y = np.array([int(P22[p]["label"]) for p in PIDS])
assert all(int(T7[p]["label"]) == int(P22[p]["label"]) == int(P16[p]["label"]) for p in PIDS), "label sources disagree"
spk = np.array(PIDS)
uspk = np.array(sorted(set(spk))); idx_of = {s: i for i, s in enumerate(spk)}
N, NS = len(y), len(uspk)

def boot(stat):
    rng = np.random.default_rng(0); vals = []
    for _ in range(NB):
        pick = rng.choice(len(uspk), size=len(uspk), replace=True)
        ii = np.array([idx_of[uspk[p]] for p in pick]); yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy): continue
        vals.append(stat(ii, yy))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)

ROWS = f"{OUT}/A_provisional.tsv"
if not os.path.exists(ROWS):
    open(ROWS, "w").write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\ttwo_dp\tvs_half\n")
def row(rid, what, value, lo, hi, file, kind="auc", n=None, ns=None):
    n = N if n is None else n; ns = NS if ns is None else ns
    f4 = lambda v: "" if v is None else f"{v:.4f}"
    f2 = lambda v: "" if v is None else f"{v:.2f}"
    if lo is None: vs = "n/a"
    elif kind == "auc": vs = "above 0.5" if lo > 0.5 else ("below 0.5" if hi < 0.5 else "includes 0.5")
    else: vs = "difference: excludes 0" if (lo > 0 or hi < 0) else "difference: includes 0"
    two = f2(value) + (f" [{f2(lo)}, {f2(hi)}]" if lo is not None else "")
    # drop an earlier line with the same id so a rerun replaces, not duplicates
    lines = open(ROWS).read().splitlines(); lines = [l for l in lines if not l.startswith(rid + "\t")]
    lines.append("\t".join([rid, what, f4(value), f4(lo), f4(hi), str(n), str(ns), f"{GD}/{file}", two, vs]))
    open(ROWS, "w").write("\n".join(lines) + "\n")
    iv = f" [{lo:.4f}, {hi:.4f}]" if lo is not None else ""
    print(f"PASTE {rid}: {value:.4f}{iv} n={n} n_spk={ns} {GD}/{file} | two-dp {two} | {vs}", flush=True)

def sidecar(name, results, sources, extra=None):
    sc = dict(result_file=f"{GD}/{name}", n=N, n_speakers=NS, n_positive=int(y.sum()), bootstrap=dict(n_draws=NB, seed="numpy default_rng(0), fresh per cell",
              unit="speaker (one clip per speaker)", percentiles=[2.5, 97.5], draw_code="rng.choice(len(uspk), size=len(uspk), replace=True)"),
              model_id=MID, prompt_verbatim=PROMPT, dtype="bfloat16", auc="rank formula (probe_boot.py auc_rank)",
              sources={s: dict(sha256_first1MB=sha1mb(s), bytes=os.path.getsize(s)) for s in sources},
              command=CMD, library_versions=libs(), results=results, written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if extra: sc.update(extra)
    base = name.rsplit(".", 1)[0]
    json.dump(sc, open(f"{OUT}/{base}.sidecar.json", "w"), indent=1)

def mark(name):
    open(f"{D}/{name}.RESULT", "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    open(f"{OUT}/{name}.RESULT", "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")

pz = np.array([float(EX[p]["p_yes"]) for p in PIDS])
pz300 = np.array([float(T7[p]["p_yes_answer"]) for p in PIDS])

if MODE == "items12":
    assert len(EX) == 275
    # ---------- item 1 ----------
    BR = {r["pid"]: r for r in csv.DictReader(open(f"{D}/edaicfull/build_report.csv"))}
    ds = np.array([float(BR[p]["d_stitch"]) for p in PIDS]); dw = np.array([float(BR[p]["d_win"]) for p in PIDS])
    ss = np.array([float(BR[p]["stitched_s"]) for p in PIDS]); ws = np.array([float(BR[p]["win_s"]) for p in PIDS])
    src = np.array([float(r["src_dur_s"]) for r in WIN]); wd = np.array([float(r["win_dur_s"]) for r in WIN])
    to = np.array([int(EX[p]["audio_tok_observed"]) for p in PIDS])
    ff = np.array([int(EX[p]["feat_frames_formula"]) for p in PIDS]); fo = np.array([int(EX[p]["feat_frames_observed"]) for p in PIDS])
    ns_ = np.array([int(EX[p]["wave_samples"]) for p in PIDS])
    qf = lambda F: ((F - 1) // 2 + 1 - 2) // 2 + 1          # Qwen2.5-Omni _get_feat_extract_output_lengths
    tf = qf(fo)                                              # formula applied to the processor's mel-frame count
    tf_floor = np.array([int(EX[p]["audio_tok_expected_from_formula"]) for p in PIDS])   # formula applied to n_samples // 160
    whole = np.abs(fo - ns_ / 160.0) <= 1.0                  # mel frames cover the whole wave (to within one 10 ms hop)
    sq = np.array([int(EX[p]["seq_len"]) for p in PIDS]); fit = np.array([int(EX[p]["fit_in_32k"]) for p in PIDS])
    pp = np.array([int(EX[p]["proj_positions"]) for p in PIDS]); ep = np.array([int(EX[p]["enc_positions"]) for p in PIDS])
    capped = np.array([int(r["capped"]) for r in WIN]); t22 = np.array([int(P22[p]["audio_tok_lifted"]) for p in PIDS])
    s22 = np.array([int(P22[p]["seq_len_lifted"]) for p in PIDS])
    name1 = "A1_edaic_whole_audiotok.csv"
    with open(f"{OUT}/{name1}", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "win_dur_s", "capped", "audio_tok_expected_from_formula", "audio_tok_observed", "seq_len", "fit_in_32k",
                    "wave_samples", "audio_tok_formula_on_nsamples_floor160", "mel_frames_cover_whole_wave",
                    "src_dur_s", "stitched_s_rebuilt", "d_stitch_s", "win_s_rebuilt", "d_win_s", "feat_frames_formula", "feat_frames_observed",
                    "proj_positions", "enc_positions", "audio_tok_part22_lifted", "seq_len_part22_lifted", "agree_part22", "cut_wav_sha256_first1MB"])
        for i, p in enumerate(PIDS):
            w.writerow([p, WIN[i]["win_dur_s"], capped[i], tf[i], to[i], sq[i], fit[i], ns_[i], tf_floor[i], int(whole[i]), WIN[i]["src_dur_s"], ss[i], ds[i], ws[i], dw[i],
                        ff[i], fo[i], pp[i], ep[i], t22[i], s22[i], int(to[i] == t22[i] and sq[i] == s22[i]), EX[p]["cut_wav_sha256_first1MB"]])
    at_cap = int((to == 22500).sum()); cut_any = int(((to < tf) | (~whole)).sum())
    r1 = dict(n_built=len(BR), max_abs_d_stitch_s=float(np.abs(ds).max()), max_abs_d_win_s=float(np.abs(dw).max()),
              n_tok_equal_formula=int((to == tf).sum()), n_tok_equal_formula_on_nsamples_floor160=int((to == tf_floor).sum()),
              max_abs_tok_minus_formula_on_nsamples_floor160=int(np.abs(to - tf_floor).max()), n_mel_frames_cover_whole_wave=int(whole.sum()),
              n_tok_ge_floor_25_per_s=int((to >= np.floor(ns_ / 16000 * 25)).sum()), n_tok_below_formula_ie_cut=cut_any, n_at_22500=at_cap, n_capped_in_table=int(capped.sum()),
              n_capped_and_at_22500=int(((to == 22500) & (capped == 1)).sum()), n_fit_in_32k=int(fit.sum()), max_seq_len=int(sq.max()),
              min_audio_tok=int(to.min()), median_audio_tok=float(np.median(to)), max_audio_tok=int(to.max()),
              n_proj_positions_equal_audio_tok=int((pp == to).sum()), n_enc_positions_equal_2x_minus=int(((ep + 1) // 2 == to).sum()),
              n_feat_frames_equal_formula=int((fo == ff).sum()),
              n_agree_part22_audio_tok=int((to == t22).sum()), n_agree_part22_seq_len=int((sq == s22).sum()))
    print("ITEM1", json.dumps(r1), flush=True)
    sidecar(name1, r1, [f"{D}/edaicfull/windows_full.csv", f"{D}/edaicfull/build_report.csv", f"{D}/build_windows_p23a.py",
                        f"{D}/A_extract_perclip.csv", f"{D}/ref/edaic_lifted_perclip.csv", f"{D}/p23a_extract.py"],
            dict(extract_command="cd /workspace/p23a && nohup python3 p23a_extract.py > extract.log 2>&1 &",
                 build_command="python3 build_windows_p23a.py  (= edaic_rerun/part16/EDAICFULL/build_windows.py with D=/workspace/p23a/edaicfull)",
                 formula="audio_tok_expected_from_formula = ((F-1)//2+1 - 2)//2+1 with F = the processor's mel-frame count (feature_attention_mask sum); 900 s -> F 90000 -> 22500. Column audio_tok_formula_on_nsamples_floor160 uses F = n_samples//160 instead (can be one frame short of the processor's count for non-multiple lengths)."))
    row("A1_max_abs_d_stitch", "item 1: rebuilt stitched duration minus windows_full.csv src_dur_s, max abs (s)", r1["max_abs_d_stitch_s"], None, None, name1)
    row("A1_max_abs_d_win", "item 1: rebuilt window duration minus windows_full.csv win_dur_s, max abs (s)", r1["max_abs_d_win_s"], None, None, name1)
    row("A1_n_at_22500", "item 1: files with exactly 22,500 audio tokens (900 s cap), truncation lifted", at_cap, None, None, name1)
    row("A1_n_cut", "item 1: files with fewer audio tokens than the formula for their whole window (still cut)", cut_any, None, None, name1)
    row("A1_n_fit_32k", "item 1: files whose sequence fits in 32,768", int(fit.sum()), None, None, name1)
    row("A1_n_agree_part22_tok", "item 1: files whose audio token count equals PART 22 audio_tok_lifted", int((to == t22).sum()), None, None, name1)
    mark("A1_edaic_whole_audiotok")

    # ---------- item 2 ----------
    p22 = np.array([float(P22[p]["p_yes_lifted"]) for p in PIDS])
    mass = np.array([float(EX[p]["answer_mass"]) for p in PIDS]); pr_ = np.array([float(EX[p]["p_yes_rule"]) for p in PIDS])
    mr = np.array([float(EX[p]["answer_mass_rule"]) for p in PIDS]); m22 = np.array([float(P22[p]["mass_lifted"]) for p in PIDS])
    name2 = "A2_edaic_whole_zeroshot_perclip.csv"
    with open(f"{OUT}/{name2}", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "speaker", "label", "win_dur_s", "capped", "audio_tok", "p_yes_whole", "answer_mass_whole", "p_yes_rule_whole",
                    "answer_mass_rule_whole", "p_yes_part22_lifted", "abs_diff_vs_part22", "p_yes_300s_T7b"])
        for i, p in enumerate(PIDS):
            w.writerow([p, p, y[i], WIN[i]["win_dur_s"], WIN[i]["capped"], EX[p]["audio_tok_observed"], repr(pz[i]), repr(mass[i]), repr(pr_[i]),
                        repr(mr[i]), repr(p22[i]), repr(abs(pz[i] - p22[i])), repr(pz300[i])])
    a_w = auc_rank(y, pz); lo_w, hi_w, u_w = boot(lambda ii, yy: auc_rank(yy, pz[ii]))
    a_3 = auc_rank(y, pz300); lo_3, hi_3, u_3 = boot(lambda ii, yy: auc_rank(yy, pz300[ii]))
    lo_d, hi_d, u_d = boot(lambda ii, yy: auc_rank(yy, pz[ii]) - auc_rank(yy, pz300[ii]))
    a_r = auc_rank(y, pr_); lo_r, hi_r, u_r = boot(lambda ii, yy: auc_rank(yy, pr_[ii]))
    a_22 = auc_rank(y, p22)
    r2 = dict(auc_whole=a_w, ci_whole=[lo_w, hi_w], usable=u_w, auc_300s_recomputed=a_3, ci_300s_recomputed=[lo_3, hi_3],
              reference_300s="0.8278 [0.7703, 0.8822]", paired_whole_minus_300=a_w - a_3, ci_paired=[lo_d, hi_d],
              auc_whole_rule_set=a_r, ci_rule_set=[lo_r, hi_r], auc_part22_lifted_recomputed=a_22,
              max_abs_diff_vs_part22=float(np.abs(pz - p22).max()), mean_abs_diff_vs_part22=float(np.abs(pz - p22).mean()),
              pearson_r_vs_part22=float(np.corrcoef(pz, p22)[0, 1]), n_absdiff_gt_1e4=int((np.abs(pz - p22) > 1e-4).sum()),
              matches_part22_to_1e4_every_clip=bool(np.abs(pz - p22).max() <= 1e-4),
              answer_mass_min=float(mass.min()), answer_mass_median=float(np.median(mass)), max_abs_mass_vs_part22=float(np.abs(mass - m22).max()),
              answer_mass_rule_median=float(np.median(mr)))
    print("ITEM2", json.dumps(r2), flush=True)
    sidecar(name2, r2, [f"{D}/A_extract_perclip.csv", f"{D}/ref/edaic_lifted_perclip.csv", f"{D}/ref/T7b_edaic300_meanof5_perclip.csv", f"{D}/p23a_extract.py"],
            dict(extract_command="cd /workspace/p23a && nohup python3 p23a_extract.py > extract.log 2>&1 &",
                 yes_ids=[7414, 9454, 9693, 9834, 14004], no_ids=[902, 2152, 2308, 2753, 8996], rule_set_adds=[14080, 5664],
                 p_yes="P(Yes)/(P(Yes)+P(No)) at the first answer position (softmax over the full vocabulary, float32)",
                 processor_call='proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True, truncation=False)'))
    row("A2_zeroshot_whole", "item 2: Qwen2.5-Omni zero-shot answer AUC, E-DAIC whole window (900 s cap, truncation lifted)", a_w, lo_w, hi_w, name2)
    row("A2_zeroshot_300s_ref", "item 2: same, 300 s window (T7b p_yes, recomputed here; reference 0.8278 [0.7703, 0.8822])", a_3, lo_3, hi_3, name2)
    row("A2_zeroshot_whole_minus_300s", "item 2: whole minus 300 s zero-shot AUC, paired", a_w - a_3, lo_d, hi_d, name2, kind="diff")
    row("A2_zeroshot_whole_ruleset", "item 2 supplement: whole-window AUC with the rule id set (+14080, +5664)", a_r, lo_r, hi_r, name2)
    row("A2_maxabs_vs_part22", "item 2: max abs per-clip p_yes difference vs PART 22 p_yes_lifted", r2["max_abs_diff_vs_part22"], None, None, name2)
    row("A2_pearson_vs_part22", "item 2: Pearson r per-clip p_yes vs PART 22 p_yes_lifted", r2["pearson_r_vs_part22"], None, None, name2)
    mark("A2_edaic_whole_zeroshot")

elif MODE == "item3":
    STREAMS = [("enc", "encoder"), ("proj", "projector"), ("llm", "LM"), ("ans", "answer state")]
    REF300 = dict(enc=(0.5884, "[0.5268, 0.6523]", -0.2394), llm=(0.7001, "[0.6231, 0.7727]", -0.1276),
                  ans=(0.7435, "[0.6793, 0.8065]", -0.0843), proj=(0.5610, "[0.4871, 0.6339]", -0.2668))
    V, PER, PICKS = {}, {}, {}
    for k, _ in STREAMS:
        zz = [np.load(f"{PR}/{k}_rep{r}.npz") for r in range(5)]
        V[k] = np.array([z["oof"] for z in zz]); PER[k] = [float(z["auc"]) for z in zz]; PICKS[k] = [z["picks"].tolist() for z in zz]
        assert np.allclose(PER[k], [auc_rank(y, v) for v in V[k]])
    V300 = {k: np.array([[float(T7[p][f"oof_{k}_r{r}"]) for p in PIDS] for r in range(5)]) for k, _ in STREAMS}
    name3 = "A3_edaic_whole_meanof5_perclip.csv"
    with open(f"{OUT}/{name3}", "w", newline="") as fh:
        w = csv.writer(fh); oc = [f"oof_{k}_r{r}" for k in ("enc", "llm", "ans", "proj") for r in range(5)]
        w.writerow(["pid", "speaker", "label", "p_yes_whole"] + oc + [f"fold_rs{r}" for r in range(5)])
        for i, p in enumerate(PIDS):
            w.writerow([p, p, y[i], repr(pz[i])] + [repr(float(V[k][r][i])) for k in ("enc", "llm", "ans", "proj") for r in range(5)]
                       + [int(np.load(f"{PR}/enc_rep{r}.npz")["fold"][i]) for r in range(5)])
    a_w = auc_rank(y, pz)
    res = {}
    for k, lab in STREAMS:
        m = float(np.mean(PER[k]))
        lo, hi, u = boot(lambda ii, yy, k=k: float(np.mean([auc_rank(yy, v[ii]) for v in V[k]])))
        dlo, dhi, du = boot(lambda ii, yy, k=k: float(np.mean([auc_rank(yy, v[ii]) for v in V[k]])) - auc_rank(yy, pz[ii]))
        m300 = float(np.mean([auc_rank(y, v) for v in V300[k]]))
        wlo, whi, wu = boot(lambda ii, yy, k=k: float(np.mean([auc_rank(yy, v[ii]) for v in V[k]])) - float(np.mean([auc_rank(yy, v[ii]) for v in V300[k]])))
        res[k] = dict(per_repeat=PER[k], per_repeat_4dp=[round(a, 4) for a in PER[k]], meanof5=m, ci=[lo, hi], usable=u,
                      meanof5_of_4dp_rounded=round(float(np.mean([round(a, 4) for a in PER[k]])), 4),
                      auc_of_mean_oof_NOT_USED=auc_rank(y, V[k].mean(0)), layers_picked=PICKS[k],
                      minus_whole_answer=m - a_w, ci_minus_whole_answer=[dlo, dhi], excludes_zero="yes" if (dlo > 0 or dhi < 0) else "no",
                      ref_300s_meanof5=REF300[k][0], ref_300s_ci=REF300[k][1], ref_300s_minus_answer=REF300[k][2],
                      meanof5_300s_recomputed_from_T7b=m300, whole_minus_300s_probe=m - m300, ci_whole_minus_300s_probe=[wlo, whi])
        print(f"ITEM3 {lab:12s} whole mean-of-5 {m:.4f} [{lo:.4f}, {hi:.4f}]  (300 s {REF300[k][0]:.4f} {REF300[k][1]}) | "
              f"minus whole answer {m-a_w:.4f} [{dlo:.4f}, {dhi:.4f}] (300 s {REF300[k][2]:.4f}) | per-repeat {res[k]['per_repeat_4dp']} layers {PICKS[k]}", flush=True)
        row(f"A3_{k}_meanof5_whole", f"item 3: {lab} probe, mean of five per-repeat AUCs {res[k]['per_repeat_4dp']}, whole window (300 s: {REF300[k][0]:.4f} {REF300[k][1]})",
            m, lo, hi, name3)
        row(f"A3_{k}_meanof5_minus_answer_whole", f"item 3: {lab} probe mean-of-five minus whole-window zero-shot answer, paired (300 s: {REF300[k][2]:.4f})",
            m - a_w, dlo, dhi, name3, kind="diff")
        row(f"A3_{k}_meanof5_whole_minus_300s", f"item 3 extra: {lab} probe mean-of-five, whole minus 300 s window (T7b vectors), paired",
            m - m300, wlo, whi, name3, kind="diff")
    json.dump({k: dict(per_repeat=PER[k], layers_picked=PICKS[k], mean_of_repeat_aucs=float(np.mean(PER[k]))) for k, _ in STREAMS},
              open(f"{OUT}/A3_edaic_whole_nested_repeats.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/A3_edaic_whole_oof_repeats.npz", **{f"{k}_oof_repeats": V[k] for k, _ in STREAMS},
                        label=y, speaker=spk, p_yes=pz, clip=np.array([f"{p}.wav" for p in PIDS]))
    # sanity: saved fold files vs GroupKFold(5, shuffle=True, random_state=r) on this stack (not used; folds were READ)
    from sklearn.model_selection import GroupKFold
    fold_check = {}
    for r in range(5):
        f = np.load(f"{PR}/enc_rep{r}.npz")["fold"]; g = np.zeros(N, int)
        for kk, (tr, te) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(np.zeros(N), y, groups=spk)): g[te] = kk
        fold_check[f"rs{r}"] = bool((f == g).all())
    srcs = [f"{PR}/{k}_rep{r}.npz" for k, _ in STREAMS for r in range(5)] + [f"{D}/folds/edaic_full_groupkfold5_shuffle_rs{r}.csv" for r in range(5)] + \
           [f"{D}/A_extract_perclip.csv", f"{D}/ref/T7b_edaic300_meanof5_perclip.csv", f"{D}/p23a_probe.py", f"{D}/ref/probe_boot.py"]
    sidecar(name3, dict(streams=res, zero_shot_whole_auc=a_w, saved_folds_equal_groupkfold_on_this_stack=fold_check), srcs,
            dict(probe="nested: outer folds READ from folds/edaic_full_groupkfold5_shuffle_rs{0..4}.csv; inner GroupKFold(4, shuffle=True, random_state=rep) "
                       "layer selection by inner OOF AUC; StandardScaler + LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0); float32 features; "
                       "five repeats; reported value = mean of the five per-repeat AUCs (paper convention)",
                 probe_commands=[f"cd /workspace/p23a && nohup python3 p23a_probe.py probe {k} > probe_{k}.log 2>&1 &" for k, _ in STREAMS],
                 states="enc 32x1280 per-layer mean over time; proj 1x3584 mean over audio positions; llm 29x3584 mean over audio-token positions; ans 29x3584 at the first answer position"))
    mark("A3_edaic_whole_meanof5")

    # ---------- per-layer files ----------
    PL = {}
    for key, fn in [("enc", "o25_whole_encoder_perlayer.csv"), ("llm", "o25_whole_llm_perlayer.csv"), ("ans", "o25_whole_ans_perlayer.csv"), ("proj", "o25_whole_proj_perlayer.csv")]:
        R = list(csv.DictReader(open(f"{PR}/{fn}"))); import shutil; shutil.copy(f"{PR}/{fn}", f"{OUT}/{fn}")
        oo = np.array([float(r["auc_oof"]) for r in R]); mm = np.array([float(r["auc_mean"]) for r in R])
        ref = f"{D}/ref/{fn.replace('whole', 'full')}"
        Rr = list(csv.DictReader(open(ref))); ro = np.array([float(r["auc_oof"]) for r in Rr])
        PL[key] = dict(file=f"{GD}/{fn}", header=list(R[0].keys()), ref_header=list(Rr[0].keys()), n_rows=len(R), peak_auc_oof=float(oo.max()), peak_stage=int(oo.argmax()),
                       peak_auc_mean=float(mm.max()), peak_auc_mean_stage=int(mm.argmax()), ref_300s_peak_auc_oof=float(ro.max()), ref_300s_peak_stage=int(ro.argmax()))
        print(f"PERLAYER {key}: {fn} rows {len(R)} header {PL[key]['header']} peak auc_oof {oo.max():.4f} @ {int(oo.argmax())} (300 s file peak {ro.max():.4f} @ {int(ro.argmax())})", flush=True)
    base = "o25_whole_perlayer"
    json.dump(dict(files=PL, method="part16/EDAICFULL/perlayer.py layer_curve verbatim: GroupKFold(n_splits=5) by speaker (no shuffle), StandardScaler + "
                   "LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0) per stage; auc_mean/auc_std over the 5 test folds, auc_oof on pooled OOF",
                   command="cd /workspace/p23a && nohup python3 p23a_probe.py perlayer > perlayer.log 2>&1 &", n=N, n_speakers=NS,
                   sources={s: sha1mb(s) for s in [f"{D}/A_extract_perclip.csv", f"{D}/p23a_probe.py", f"{D}/ref/perlayer.py"]},
                   model_id=MID, prompt_verbatim=PROMPT, library_versions=libs()), open(f"{OUT}/{base}.sidecar.json", "w"), indent=1)
    mark("o25_whole_perlayer")

    # ---------- supplement: best single stage (NOT the paper convention; chosen on the test results, optimistic) ----------
    nameS = "A3s_edaic_whole_beststage.csv"; supp = {}
    with open(f"{OUT}/{nameS}", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["stream", "stage", "auc_r0", "auc_r1", "auc_r2", "auc_r3", "auc_r4", "meanof5", "is_best"])
        for key, lab in [("enc", "encoder"), ("llm", "LM"), ("ans", "answer state")]:
            z = np.load(f"{PR}/fixed_{key}.npz"); per = z["per_repeat"]; m = per.mean(1); b = int(m.argmax()); O = z["oof"][b]
            for L in range(len(m)): w.writerow([key, L] + [repr(float(a)) for a in per[L]] + [repr(float(m[L])), int(L == b)])
            lo, hi, u = boot(lambda ii, yy, O=O: float(np.mean([auc_rank(yy, v[ii]) for v in O])))
            dlo, dhi, du = boot(lambda ii, yy, O=O: float(np.mean([auc_rank(yy, v[ii]) for v in O])) - auc_rank(yy, pz[ii]))
            supp[key] = dict(best_stage=b, meanof5=float(m[b]), ci=[lo, hi], minus_whole_answer=float(m[b]) - a_w, ci_minus=[dlo, dhi])
            row(f"A3s_{key}_best_single_stage", f"SUPPLEMENT (not paper convention, stage chosen on test AUC): {lab} best single stage {b}, mean of five per-repeat AUCs, whole window",
                float(m[b]), lo, hi, nameS)
            row(f"A3s_{key}_best_single_stage_minus_answer", f"SUPPLEMENT: {lab} best single stage {b} minus whole-window zero-shot answer, paired",
                float(m[b]) - a_w, dlo, dhi, nameS, kind="diff")
    sidecar(nameS, supp, [f"{PR}/fixed_{k}.npz" for k in ("enc", "llm", "ans")] + [f"{D}/p23a_probe.py"],
            dict(note="SUPPLEMENT, not the paper convention: each single stage probed on the same five saved outer splits with no inner selection; "
                      "the best stage is picked on the reported AUC itself, so the value is optimistic.",
                 probe_commands=[f"cd /workspace/p23a && nohup python3 p23a_probe.py fixed {k} > fixed_{k}.log 2>&1 &" for k in ("enc", "llm", "ans")]))
    mark("A3s_edaic_whole_beststage")

elif MODE == "ctl":
    # EXTRA (labelled): same-GPU 300 s control. The unmodified part16 extract_full.py (default processor call, which encodes the
    # first 300 s) was run on THIS pod (H100 NVL) on the same 275 cut files, and probed with the same saved folds, so the
    # whole-vs-300 s comparison is free of the H100 NVL vs H200 numeric difference shown in A2d_gpu_gate_diagnostic.
    STREAMS = [("enc", "encoder"), ("proj", "projector"), ("llm", "LM"), ("ans", "answer state")]
    C = {r["clip"].replace(".wav", ""): r for r in csv.DictReader(open(f"{D}/ctl300/ctl300_zeroshot_scores.csv"))}
    assert len(C) == 275
    pc = np.array([float(C[p]["p_yes"]) for p in PIDS]); tokc = np.array([int(C[p]["n_audio_tok"]) for p in PIDS])
    V = {k: np.array([np.load(f"{PR}/{k}_rep{r}.npz")["oof"] for r in range(5)]) for k, _ in STREAMS}
    VC = {k: np.array([np.load(f"{D}/probe_ctl300/{k}_rep{r}.npz")["oof"] for r in range(5)]) for k, _ in STREAMS}
    PC = {k: [float(np.load(f"{D}/probe_ctl300/{k}_rep{r}.npz")["auc"]) for r in range(5)] for k, _ in STREAMS}
    PKC = {k: [np.load(f"{D}/probe_ctl300/{k}_rep{r}.npz")["picks"].tolist() for r in range(5)] for k, _ in STREAMS}
    name4 = "A4_ctl300_samegpu_perclip.csv"
    with open(f"{OUT}/{name4}", "w", newline="") as fh:
        w = csv.writer(fh); oc = [f"ctl300_oof_{k}_r{r}" for k in ("enc", "llm", "ans", "proj") for r in range(5)]
        w.writerow(["pid", "speaker", "label", "p_yes_whole", "p_yes_ctl300_samegpu", "audio_tok_ctl300", "p_yes_300s_T7b_h200"] + oc)
        for i, p in enumerate(PIDS):
            w.writerow([p, p, y[i], repr(pz[i]), repr(pc[i]), tokc[i], repr(pz300[i])] + [repr(float(VC[k][r][i])) for k in ("enc", "llm", "ans", "proj") for r in range(5)])
    a_w = auc_rank(y, pz); a_c = auc_rank(y, pc)
    lo, hi, u = boot(lambda ii, yy: auc_rank(yy, pc[ii]))
    dlo, dhi, du = boot(lambda ii, yy: auc_rank(yy, pz[ii]) - auc_rank(yy, pc[ii]))
    r4 = dict(zero_shot_ctl300_samegpu=a_c, ci=[lo, hi], whole_minus_ctl300=a_w - a_c, ci_whole_minus_ctl300=[dlo, dhi],
              ctl300_vs_T7b_max_abs_p_yes=float(np.abs(pc - pz300).max()), ctl300_vs_T7b_pearson=float(np.corrcoef(pc, pz300)[0, 1]),
              ctl300_n_at_7500_tok=int((tokc == 7500).sum()), streams={})
    row("A4_ctl300_zeroshot", "EXTRA same-GPU control: zero-shot answer AUC, 300 s window (unmodified extract_full.py, default call) on this H100 NVL pod", a_c, lo, hi, name4)
    row("A4_zeroshot_whole_minus_ctl300", "EXTRA same-GPU control: whole minus 300 s zero-shot AUC, paired, both on this pod", a_w - a_c, dlo, dhi, name4, kind="diff")
    row("A4_ctl300_vs_T7b_maxabs", "EXTRA: max abs per-clip p_yes, 300 s window on this H100 NVL pod vs published 300 s (T7b, H200)", r4["ctl300_vs_T7b_max_abs_p_yes"], None, None, name4)
    for k, lab in STREAMS:
        mc = float(np.mean(PC[k])); mw = float(np.mean([auc_rank(y, v) for v in V[k]]))
        lo, hi, u = boot(lambda ii, yy, k=k: float(np.mean([auc_rank(yy, v[ii]) for v in VC[k]])))
        alo, ahi, au = boot(lambda ii, yy, k=k: float(np.mean([auc_rank(yy, v[ii]) for v in VC[k]])) - auc_rank(yy, pc[ii]))
        wlo, whi, wu = boot(lambda ii, yy, k=k: float(np.mean([auc_rank(yy, v[ii]) for v in V[k]])) - float(np.mean([auc_rank(yy, v[ii]) for v in VC[k]])))
        r4["streams"][k] = dict(ctl300_per_repeat=PC[k], ctl300_layers=PKC[k], ctl300_meanof5=mc, ci=[lo, hi], ctl300_minus_ctl300_answer=mc - a_c,
                                ci_minus_answer=[alo, ahi], whole_meanof5=mw, whole_minus_ctl300=mw - mc, ci_whole_minus_ctl300=[wlo, whi])
        row(f"A4_ctl300_{k}_meanof5", f"EXTRA same-GPU control: {lab} probe mean of five {[round(a, 4) for a in PC[k]]}, 300 s window, this pod", mc, lo, hi, name4)
        row(f"A4_ctl300_{k}_minus_answer", f"EXTRA same-GPU control: {lab} probe mean-of-five minus 300 s zero-shot answer, paired, this pod", mc - a_c, alo, ahi, name4, kind="diff")
        row(f"A4_{k}_whole_minus_ctl300", f"EXTRA same-GPU control: {lab} probe mean-of-five, whole minus 300 s, paired, both on this pod", mw - mc, wlo, whi, name4, kind="diff")
        print(f"CTL {lab:12s} 300 s same-GPU {mc:.4f} [{lo:.4f}, {hi:.4f}] minus answer {mc-a_c:.4f} [{alo:.4f}, {ahi:.4f}] | whole {mw:.4f} whole-300 {mw-mc:.4f} [{wlo:.4f}, {whi:.4f}]", flush=True)
    sidecar(name4, r4, [f"{D}/ctl300/ctl300_zeroshot_scores.csv", f"{D}/ctl300/extract_full.py", f"{D}/A_extract_perclip.csv", f"{D}/p23a_probe.py"]
            + [f"{D}/probe_ctl300/{k}_rep{r}.npz" for k, _ in STREAMS for r in range(5)],
            dict(note="EXTRA, not requested: same-GPU 300 s control so that whole-vs-300 s differences are not mixed with the H100 NVL vs H200 numeric difference",
                 extract_command="cd /workspace/p23a/ctl300 && HF_HOME=/workspace/hf WINDOW_S=0 python3 extract_full.py /workspace/p23a/ctl300/mf_all275_p23a.csv /workspace/p23a/ctl300/ctl300 mdd",
                 probe_commands=[f"cd /workspace/p23a && CTL300=1 python3 p23a_probe.py probe {k}" for k, _ in STREAMS]))
    mark("A4_ctl300_samegpu")
