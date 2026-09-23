#!/usr/local/bin/python3
"""PART17 task 1. Table 2 E-DAIC agreement cells on the 311 DISTINCT agreement segments.

OLD = all 483 agreement rows (repeats included), NEW = 311 distinct agreement segments
(first occurrence in manifest row order). Conflict cell alongside (already distinct).
Supplementary: paired conflict-minus-agreement difference on distinct segments.

AUC: rank formula (scipy.stats.rankdata, ties averaged).
Bootstrap: 2000 draws, numpy default_rng(0) fresh per cell, speakers resampled with
replacement (sorted unique ids, rng.integers(0, K, K)), percentile 2.5 / 97.5.
Zero writes under release/omni_final.
"""
import os, sys, re, json, hashlib, datetime
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

RR = "<local data dir>/release/edaic_rerun/"
OUT = RR + "part17/"
ROWS = OUT + "rows/"
MAN = RR + "part14_manifest_new.csv"
PITT_MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
JSONL = "<authors-notes-not-released>"
TEX_DISK = "<local data dir>/paper1_final/AudioLLMHealthUtilityGap.tex"
DRAWS = 2000
SEED = 0
COMMAND = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/bin/python3 " + os.path.abspath(__file__)

# (key, model display, table-2 row label, stream, score file, sidecar with model id, OLD value verified tonight)
STREAMS = [
    ("o25",   "Qwen2.5-Omni",     "Qwen2.5-Omni",     "audio",      RR + "part14/p14_o25_zeroshot_scores.csv",  RR + "part14/p14_o25_zeroshot.json",  0.9482),
    ("q2a",   "Qwen2-Audio",      "Qwen2-Audio",      "audio",      RR + "part14/p14_q2a_zeroshot_scores.csv",  RR + "part14/p14_q2a_zeroshot.json",  0.9478),
    ("q3o",   "Qwen3-Omni-30B-A3B", "Qwen3-Omni",     "audio",      RR + "part14/p14_q3o_zeroshot_scores.csv",  RR + "part14/p14_q3o_zeroshot.json",  0.9638),
    ("af2",   "Audio Flamingo 2", "Audio Flamingo 2", "audio",      RR + "part14/p14_af2.csv",                  RR + "part14/p14_af2.json",           0.6083),
    ("af3",   "Audio Flamingo 3", "Audio Flamingo 3", "audio",      RR + "part14/p14_af3.csv",                  RR + "part14/p14_af3.json",           0.8095),
    ("kimi",  "Kimi-Audio",       "Kimi-Audio",       "audio",      RR + "part14/p14_kimi_zeroshot_scores.csv", RR + "part14/p14_kimi_zeroshot.json", 0.8744),
    ("o25t",  "Qwen2.5-Omni",     None,               "transcript", RR + "part16/pod_sync/p14_o25_text.csv",    RR + "part16/pod_sync/p14_o25_text.json", 0.9596),
    ("q2at",  "Qwen2-Audio",      None,               "transcript", RR + "part16/pod_sync/p14_q2a_text.csv",    RR + "part16/pod_sync/p14_q2a_text.json", 0.9081),
]
POD1_JOINED = {"o25t": RR + "part16/POD1/p14_o25_1a_joined.csv", "q2at": RR + "part16/POD1/p14_q2a_1a_joined.csv"}


def sha1mb(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read(1024 * 1024)).hexdigest()


def auc_rank(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spk_groups(spk):
    uniq = np.unique(spk)
    return uniq, [np.where(spk == u)[0] for u in uniq]


def boot_auc(y, s, spk):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); spk = np.asarray(spk)
    uniq, g = spk_groups(spk)
    rng = np.random.default_rng(SEED)
    K = len(uniq); vals = []
    for _ in range(DRAWS):
        d = rng.integers(0, K, K)
        rows = np.concatenate([g[i] for i in d])
        a = auc_rank(y[rows], s[rows])
        if not np.isnan(a):
            vals.append(a)
    vals = np.asarray(vals)
    return auc_rank(y, s), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), K, len(vals)


def boot_paired(dc, da):
    """ONE speaker draw per replicate; conflict and agreement AUC both recomputed inside it."""
    uniq = np.unique(np.concatenate([dc.speaker_id.values, da.speaker_id.values]))
    cg = [np.where(dc.speaker_id.values == u)[0] for u in uniq]
    ag = [np.where(da.speaker_id.values == u)[0] for u in uniq]
    cy, cs = dc.label.values.astype(int), dc.p.values.astype(float)
    ay, as_ = da.label.values.astype(int), da.p.values.astype(float)
    rng = np.random.default_rng(SEED)
    K = len(uniq); diffs = []
    for _ in range(DRAWS):
        d = rng.integers(0, K, K)
        ci = np.concatenate([cg[i] for i in d]); ai = np.concatenate([ag[i] for i in d])
        a1 = auc_rank(cy[ci], cs[ci]); a2 = auc_rank(ay[ai], as_[ai])
        if np.isnan(a1) or np.isnan(a2):
            continue
        diffs.append(a1 - a2)
    diffs = np.asarray(diffs)
    pt = auc_rank(cy, cs) - auc_rank(ay, as_)
    return pt, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), K, len(diffs)


def model_id(sidecar):
    try:
        j = json.load(open(sidecar))
    except Exception:
        return None
    for k in ("checkpoint", "model", "model_id"):
        if isinstance(j.get(k), str):
            return j[k]
    return None


def texts(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from texts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from texts(v)


def parse_table2(tex):
    """Return {row label: (dep_conflict, dep_agree, alz_conflict, alz_agree)} from the tab:conflict table."""
    i = tex.find("\\label{tab:conflict}")
    if i < 0:
        return None
    s = tex.rfind("\\begin{table", 0, i); e = tex.find("\\end{table}", i)
    block = tex[s:e]
    out = {}
    for line in block.splitlines():
        if "&" not in line or not line.rstrip().endswith("\\\\"):
            continue
        cells = [c.strip() for c in line.rstrip()[:-2].split("&")]
        label = re.sub(r"\\cite\{[^}]*\}|\$[^$]*\$", "", cells[0]).strip()
        nums = [re.sub(r"\\textbf\{([^}]*)\}", r"\1", c) for c in cells[1:]]
        if label and nums and all(re.fullmatch(r"[0-9]*\.[0-9]+", c) for c in nums):
            k, j = label, 2
            while k in out:          # same model can appear in two blocks (e.g. E-DAIC and Pitt)
                k = f"{label}#{j}"; j += 1
            out[k] = tuple(nums)
    return out, block


def live_table2():
    """Most recent authors' message in the working notes that carries the tab:conflict table (line-streamed)."""
    found = None
    with open(JSONL) as f:
        for ln, line in enumerate(f, 1):
            if "tab:conflict" not in line:
                continue
            d = json.loads(line)
            if d.get("type") != "user" or d.get("toolUseResult") is not None:
                continue
            c = d.get("message", {}).get("content")
            typed = [c] if isinstance(c, str) else [x.get("text", "") for x in (c or []) if isinstance(x, dict) and x.get("type") == "text"]
            for t in typed:
                if "\\label{tab:conflict}" in t and "\\begin{tabular" in t:
                    r = parse_table2(t)
                    if r and r[0]:
                        found = dict(line=ln, timestamp=d.get("timestamp"), rows=r[0], block=r[1])
    return found


def main():
    os.makedirs(ROWS, exist_ok=True)
    log = []
    P = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))

    # ---------------- step 1: manifest
    m = pd.read_csv(MAN)
    m["clip_id"] = m["set"].astype(str) + "_" + m.speaker_id.astype(str) + "_" + m.seg_uid.astype(str)
    P(f"manifest {MAN}: rows {len(m)}, distinct clip_id {m.clip_id.nunique()}, speakers {m.speaker_id.nunique()}")
    arm_stats = {}
    for arm in ("conflict", "agreement"):
        g = m[m["set"] == arm]
        st = dict(rows=len(g), distinct_seg_uid=int(g.seg_uid.nunique()),
                  distinct_speaker_start_end=int(g[["speaker_id", "start", "end"]].drop_duplicates().shape[0]),
                  distinct_text=int(g.text.nunique()), speakers=int(g.speaker_id.nunique()),
                  pos_rows=int(g.label.sum()),
                  max_attr_variants_within_seg=int(g.groupby("seg_uid")[["speaker_id", "start", "end", "label", "text"]].nunique().max().max()))
        arm_stats[arm] = st
        P(f"  {arm}: {st}")
    agr = m[m["set"] == "agreement"]
    rep_counts = agr.groupby("seg_uid").size()
    P(f"  agreement repeat distribution (times a segment appears -> n segments): {rep_counts.value_counts().sort_index().to_dict()}; extra repeat rows {len(agr) - agr.seg_uid.nunique()}")
    P(f"  conflict/agreement seg_uid overlap: {len(set(m[m['set']=='conflict'].seg_uid) & set(agr.seg_uid))}; "
      f"speaker sets equal: {set(m[m['set']=='conflict'].speaker_id) == set(agr.speaker_id)}")

    # Pitt agreement arm: are there repeats at all?
    pitt = {}
    try:
        pm = pd.read_csv(PITT_MAN)
        for arm, g in pm.groupby("set"):
            pitt[arm] = dict(rows=len(g), distinct_segment_path=int(g.segment_path.nunique()))
        P(f"Pitt manifest {PITT_MAN}: {pitt}")
    except Exception as ex:
        P(f"Pitt manifest not readable: {ex}")

    # ---------------- step 2 + 3: load, verify alignment, check repeats
    first_mask = ~m.duplicated("seg_uid", keep="first").values   # first occurrence in manifest row order
    agr_mask = (m["set"] == "agreement").values
    con_mask = (m["set"] == "conflict").values
    align, repeat_diff, per = {}, {}, {}
    wide = m[["clip_id", "seg_uid", "set", "pair_id", "speaker_id", "label"]].copy()
    for key, name, row, stream, path, sc, old_expected in STREAMS:
        s = pd.read_csv(path)
        if "clip" in s.columns:
            ids = s["clip"].astype(str); spk = s["speaker"]
        elif "clip_path" in s.columns:
            ids = s["clip_path"].astype(str).map(os.path.basename); spk = s["speaker_id"]
        else:
            ids = s["id"].astype(str); spk = s["speaker_id"]
        ids = ids.str.replace(".wav", "", regex=False).values
        ok_id = int((ids == m.clip_id.values).sum()); ok_lab = int((s.label.values == m.label.values).sum())
        ok_spk = int((spk.astype(int).values == m.speaker_id.values).sum())
        align[key] = dict(rows=len(s), id_match=ok_id, label_match=ok_lab, speaker_match=ok_spk, nan_p=int(s.p_yes.isna().sum()))
        assert len(s) == len(m) and ok_id == len(m) and ok_lab == len(m) and ok_spk == len(m), f"alignment failed {key}"
        d = pd.DataFrame({"seg_uid": m.seg_uid.values, "p": s.p_yes.values.astype(float)})[agr_mask]
        rd = d.groupby("seg_uid").p.agg(lambda x: float(x.max() - x.min()))
        repeat_diff[key] = float(rd.max())
        wide[f"p_{key}"] = s.p_yes.values.astype(float)
        P(f"{key:5s} {path}: alignment {align[key]} -> positional join OK; max |p_yes diff| among agreement repeats {repeat_diff[key]:.3g}")

    # POD1 joined files: cross-check the transcript streams
    pod1 = {}
    for key, jp in POD1_JOINED.items():
        j = pd.read_csv(jp)
        pod1[key] = dict(rows=len(j), key_match=int((j.key.values == m.clip_id.values).sum()),
                         max_abs_diff_text=float(np.abs(j.p_yes_text.values - wide[f"p_{key}"].values).max()),
                         max_abs_diff_audio=float(np.abs(j.p_yes_audio.values - wide[f"p_{key[:-1]}"].values).max()))
        P(f"POD1 {jp}: {pod1[key]}")

    # ---------------- step 6: printed paper values
    live = live_table2()
    disk = None
    try:
        r = parse_table2(open(TEX_DISK).read())
        disk = r[0] if r else None
    except Exception:
        pass
    if live:
        P(f"paper Table 2 source: authors' paste in the working notes, line {live['line']} at {live['timestamp']}; rows {live['rows']}")
    else:
        P("paper Table 2: live tex could NOT be read from the working notes")
    P(f"on-disk tex (stale) {TEX_DISK} mtime {datetime.datetime.fromtimestamp(os.path.getmtime(TEX_DISK))}: Table 2 rows {disk}")

    # ---------------- step 5: cells
    table, frag, lines, sc_results = [], [], [], []
    perclip_path = OUT + "T1_perclip_distinct_agreement.csv"
    all_path = OUT + "T1_perclip_all966.csv"
    for key, name, row, stream, path, sc, old_expected in STREAMS:
        base = wide[["clip_id", "seg_uid", "set", "speaker_id", "label"]].assign(p=wide[f"p_{key}"])
        con = base[con_mask]
        old = base[agr_mask]
        new = base[agr_mask & first_mask]
        res = {}
        for ver, sub, fcsv in (("conflict", con, all_path), ("agreement_OLD_483rows", old, all_path), ("agreement_NEW_311distinct", new, perclip_path)):
            a, lo, hi, K, usable = boot_auc(sub.label.values, sub.p.values, sub.speaker_id.values)
            chk = roc_auc_score(sub.label.values, sub.p.values)
            assert abs(chk - a) < 1e-12, (key, ver, a, chk)
            res[ver] = dict(value=a, lo=lo, hi=hi, n_rows=len(sub), n_segments=int(sub.seg_uid.nunique()), n_spk=K,
                            usable=usable, n_pos=int(sub.label.sum()), n_neg=int((1 - sub.label).sum()), file=fcsv)
            table.append(dict(model=name, stream=stream, arm=ver.split("_")[0],
                              version=("OLD_483rows" if "OLD" in ver else "NEW_311distinct" if "NEW" in ver else "distinct_483"),
                              value=round(a, 4), lo=round(lo, 4), hi=round(hi, 4), n_rows=len(sub),
                              n_segments=int(sub.seg_uid.nunique()), n_spk=K, n_pos=res[ver]["n_pos"],
                              n_neg=res[ver]["n_neg"], usable_draws=usable, source=path))
        # supplementary paired differences
        for ver, sub in (("paired_conflict_minus_agreement_NEW_311distinct", new), ("paired_conflict_minus_agreement_OLD_483rows", old)):
            pt, lo, hi, K, usable = boot_paired(con, sub)
            res[ver] = dict(value=pt, lo=lo, hi=hi, n_rows=len(con) + len(sub), n_segments=int(con.seg_uid.nunique() + sub.seg_uid.nunique()),
                            n_spk=K, usable=usable, file=all_path)
            table.append(dict(model=name, stream=stream, arm="conflict_minus_agreement(SUPPLEMENTARY)",
                              version=("NEW_311distinct" if "NEW" in ver else "OLD_483rows"),
                              value=round(pt, 4), lo=round(lo, 4), hi=round(hi, 4), n_rows=len(con) + len(sub),
                              n_segments=res[ver]["n_segments"], n_spk=K, n_pos=None, n_neg=None, usable_draws=usable, source=path))
        o, n_ = res["agreement_OLD_483rows"], res["agreement_NEW_311distinct"]
        repro = abs(round(o["value"], 4) - old_expected) < 1e-9
        if not repro:
            P(f"HIGH: {key} OLD agreement {o['value']:.4f} does not reproduce expected {old_expected:.4f}")
        # paper value
        paper, psrc = None, None
        if row and live and row in live["rows"]:
            paper = live["rows"][row][1]; psrc = f"live tex pasted {live['timestamp']} (notes line {live['line']}) Table 2 Depression Agree."
        new2, old2 = f"{n_['value']:.2f}", f"{o['value']:.2f}"
        near = abs(n_["value"] * 100 - np.floor(n_["value"] * 100) - 0.5) < 0.01
        if paper is not None:
            changes = "yes" if new2 != paper else "no"
            cmp = f"paper {paper} (live tex) -> new {new2}"
        else:
            changes = "yes" if new2 != old2 else "no"
            cmp = f"not printed in the live tex Table 2; round(OLD)={old2} vs round(NEW)={new2}"
        res.update(paper_value=paper, paper_source=psrc, old_2dp=old2, new_2dp=new2, changes_at_2dp=changes,
                   old_reproduces_expected=repro, old_expected=old_expected, near_rounding_boundary=bool(near),
                   model_id=model_id(sc), source=path, max_abs_pyes_diff_among_repeats=repeat_diff[key])
        sc_results.append(dict(key=key, model=name, stream=stream, **{k: v for k, v in res.items()}))
        c = res["conflict"]; pn = res["paired_conflict_minus_agreement_NEW_311distinct"]
        line = (f"{name} | {stream} | OLD {o['value']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] n=483 | "
                f"NEW {n_['value']:.4f} [{n_['lo']:.4f}, {n_['hi']:.4f}] n={n_['n_rows']} n_spk={n_['n_spk']} | "
                f"paper value {paper if paper else 'n/a'} | changes at 2 dp: {changes} ({cmp}) | "
                f"conflict {c['value']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}] n={c['n_rows']} n_spk={c['n_spk']} | "
                f"SUPP paired conf-minus-agr distinct {pn['value']:.4f} [{pn['lo']:.4f}, {pn['hi']:.4f}] usable {pn['usable']}/{DRAWS} | {perclip_path}")
        lines.append(line)
        tag = f"{key}"
        for ver, lab in (("conflict", "conflict"), ("agreement_OLD_483rows", "agreement OLD 483 rows"),
                         ("agreement_NEW_311distinct", "agreement NEW 311 distinct"),
                         ("paired_conflict_minus_agreement_NEW_311distinct", "SUPPLEMENTARY paired conflict minus agreement, distinct")):
            r_ = res[ver]
            frag.append(dict(id=f"T1_{tag}_{ver}", what=f"E-DAIC Table 2 {name} {stream} {lab}",
                             value=f"{r_['value']:.4f}", lo=f"{r_['lo']:.4f}", hi=f"{r_['hi']:.4f}",
                             n=r_["n_rows"], n_spk=r_["n_spk"], file=r_["file"]))

    # ---------------- files
    tab_path = OUT + "T1_table2_distinct.csv"
    tdf = pd.DataFrame(table)
    tdf["n_pos"] = tdf["n_pos"].astype("Int64"); tdf["n_neg"] = tdf["n_neg"].astype("Int64")
    tdf.to_csv(tab_path, index=False)
    wide.assign(in_distinct_set=(con_mask | (agr_mask & first_mask)).astype(int),
                n_repeats_in_arm=wide.seg_uid.map(wide.groupby("seg_uid").size()).values).to_csv(all_path, index=False)
    dist = wide[agr_mask & first_mask].copy()
    dist["n_repeats_in_483"] = dist.seg_uid.map(rep_counts).values
    dist["first_manifest_row"] = np.where(agr_mask & first_mask)[0]
    dist.to_csv(perclip_path, index=False)
    pd.DataFrame(frag).to_csv(ROWS + "T1.tsv", sep="\t", index=False,
                              columns=["id", "what", "value", "lo", "hi", "n", "n_spk", "file"])

    srcs = [MAN, PITT_MAN] + [s[4] for s in STREAMS] + list(POD1_JOINED.values()) + [JSONL, TEX_DISK, os.path.abspath(__file__)]
    side = dict(task="PART17 task 1: Table 2 E-DAIC agreement cells on 311 distinct segments",
                created=datetime.datetime.now().isoformat(timespec="seconds"),
                command=COMMAND, seed=SEED, draws=DRAWS,
                bootstrap="speakers resampled with replacement from sorted unique ids via default_rng(0).integers(0,K,K), fresh rng per cell; percentile 2.5/97.5; distinct cell: a drawn speaker contributes its distinct agreement segments once per draw",
                auc="rank formula scipy.stats.rankdata ties averaged; cross-checked against sklearn roc_auc_score to 1e-12",
                dedup_rule="first occurrence of seg_uid in manifest row order (all repeats carry identical p_yes in every stream, so the rule is immaterial)",
                join="positional after verifying clip id (set_speaker_seguid), label and speaker on all 966 rows",
                sources=[dict(path=p, sha256_first_1MB=sha1mb(p)) for p in srcs if os.path.exists(p)],
                n=dict(conflict_rows=483, agreement_rows_old=int(agr_mask.sum()), agreement_distinct_new=int((agr_mask & first_mask).sum())),
                n_speakers=int(m.speaker_id.nunique()),
                manifest_arm_stats=arm_stats, agreement_repeat_distribution={int(k): int(v) for k, v in rep_counts.value_counts().sort_index().items()},
                pitt_manifest_arms=pitt, alignment=align, pod1_crosscheck=pod1,
                paper_table2_source=(dict(transcript=JSONL, line=live["line"], timestamp=live["timestamp"], rows=live["rows"]) if live else "live tex not found"),
                ondisk_tex_table2_stale=disk,
                model_ids={s[0]: model_id(s[5]) for s in STREAMS},
                results=sc_results,
                outputs=[tab_path, perclip_path, all_path, ROWS + "T1.tsv", OUT + "T1_table2_distinct.json", OUT + "T1_run.log"])
    json.dump(side, open(OUT + "T1_table2_distinct.json", "w"), indent=1, default=str)

    P("\n==== PASTE LINES ====")
    for l in lines:
        P(l)
    open(OUT + "T1_run.log", "w").write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
