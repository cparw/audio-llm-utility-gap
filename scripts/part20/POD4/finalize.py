"""Build the analysis spec for one POD4 result, run analyze.py, touch <name>.RESULT.
usage: finalize.py NAME"""
import json, sys, os, shutil, subprocess, csv
NAME = sys.argv[1]
W = "/workspace/pod4"; O = "/workspace/scores/part20/POD4"; M = "scores/part20/POD4"
os.makedirs(O, exist_ok=True)
PAPER = {
    "o25": (f"{W}/ref/omni_final__omni_pitt_zeroshot_scores.csv",
            "<local data dir>/release/omni_final/omni_pitt_zeroshot_scores.csv"),
    "q2a": (f"{W}/ref/probe2__pitt_zeroshot_scores.csv",
            "<local data dir>/paper work/paper1_local_runs/probe2/pitt_zeroshot_scores.csv"),
}
MODEL = {"o25": "Qwen/Qwen2.5-Omni-7B", "q2a": "Qwen/Qwen2-Audio-7B-Instruct"}
CMDS = json.load(open(f"{W}/commands.json"))
src_csv = f"{W}/out/{NAME}.csv"
dst_csv = f"{O}/{NAME}.csv"; mac_csv = f"{M}/{NAME}.csv"
shutil.copyfile(src_csv, dst_csv)
rows = list(csv.DictReader(open(dst_csv)))
fam = NAME.split("_")[0]
pp, pm = PAPER[fam]
ARMS = [(None, "overall"), ("conflict", "conflict"), ("agreement", "agreement")]
isnoinv = "_noinv_" in NAME
man = f"{W}/man_cut.csv" if isnoinv else f"{W}/man_orig.csv"
sources = [
    dict(role="clip manifest used by the scorer", pod_path=man, mac_origin=("manifests/part20/pitt_noinv_cut/manifest.csv (orig_clip_id,spk,label) + <local data dir>/DementiaBank/pitt_conflict_manifest.csv (set)" if isnoinv else "<local data dir>/DementiaBank/segments/ (468 wavs, sha256 per clip in the per-clip csv) + <local data dir>/DementiaBank/pitt_conflict_manifest.csv (set)")),
    dict(role="paper per-clip file (comparison)", pod_path=pp, mac_path=pm),
    dict(role="scorer", pod_path=f"{W}/score_zs.py"),
    dict(role="analysis", pod_path=f"{W}/analyze.py"),
]
if isnoinv:
    sources.append(dict(role="interviewer-free cut manifest (as built on the Mac)", pod_path=f"{W}/cut/manifest.csv",
                        mac_path="manifests/part20/pitt_noinv_cut/manifest.csv"))
    sources.append(dict(role="interviewer-free cut tarball", pod_path=f"{W}/cut/pitt_noinv_clips.tgz",
                        mac_path="<local data dir>/release/scores/part20/pitt_noinv_cut/pitt_noinv_clips.tgz"))
else:
    sources.append(dict(role="original window wavs (directory; per-clip sha256 in the per-clip csv)", pod_path="/workspace/data/pitt468/", mac_path="<local data dir>/DementiaBank/segments/"))
    sources.append(dict(role="sha256 list of the 468 Mac segment wavs, all 468 matched on the pod (gate.py)", pod_path=f"{W}/ref/pitt468_mac_sha256.txt"))
cells, paired = [], []
tag = NAME
for arm, an in ARMS:
    cells.append(dict(id=f"{tag}_{an}", what=f"{NAME} zero-shot AUC, {an}", file=dst_csv, file_mac=mac_csv, arm=arm))
for arm, an in ARMS:
    cells.append(dict(id=f"paper_{fam}_orig_recording_{an}", what=f"paper {fam} zero-shot on the original 468 windows, recording wording, {an} (recomputed)", file=pp, file_mac=pm, arm=arm))
if isnoinv:
    for arm, an in ARMS:
        paired.append(dict(id=f"{tag}_PAIRED_paperorig_minus_noinv_{an}", what=f"PAIRED original (paper file) minus interviewer-free, {fam}, {an}",
                           a=pp, a_mac=pm, b=dst_csv, b_mac=mac_csv, arm=arm))
    rerun = f"{W}/out/{fam}_orig_recording_rerun{'_bf16' if NAME.endswith('_bf16') else ''}.csv"
    if os.path.exists(rerun):
        rn = os.path.basename(rerun)
        for arm, an in ARMS:
            paired.append(dict(id=f"{tag}_PAIRED_podorig_minus_noinv_{an}", what=f"PAIRED original (this pod's rerun, same code) minus interviewer-free, {fam}, {an}",
                               a=rerun, a_mac=f"{M}/{rn}", b=dst_csv, b_mac=mac_csv, arm=arm))
elif "_voice" in NAME:
    for arm, an in ARMS:
        paired.append(dict(id=f"{tag}_PAIRED_voice_minus_paperrecording_{an}", what=f"PAIRED voice wording minus recording wording (paper file), {fam}, original windows, {an}",
                           a=dst_csv, a_mac=mac_csv, b=pp, b_mac=pm, arm=arm))
    rerun = f"{W}/out/{fam}_orig_recording_rerun.csv"
    if os.path.exists(rerun):
        for arm, an in ARMS:
            paired.append(dict(id=f"{tag}_PAIRED_voice_minus_podrecording_{an}", what=f"PAIRED voice wording minus recording wording (this pod's rerun, same code), {fam}, {an}",
                               a=dst_csv, a_mac=mac_csv, b=rerun, b_mac=f"{M}/{os.path.basename(rerun)}", arm=arm))
else:  # rerun of the paper condition on this pod
    for arm, an in ARMS:
        paired.append(dict(id=f"{tag}_PAIRED_podrerun_minus_paper_{an}", what=f"PAIRED this pod's rerun minus paper file, {fam}, recording wording, original windows, {an}",
                           a=dst_csv, a_mac=mac_csv, b=pp, b_mac=pm, arm=arm))
    noinv = f"{W}/out/{fam}_noinv_recording{'_bf16' if NAME.endswith('_bf16') else ''}.csv"
    if os.path.exists(noinv):
        nn = os.path.basename(noinv)
        for arm, an in ARMS:
            paired.append(dict(id=f"{tag}_PAIRED_podorig_minus_noinv_{an}", what=f"PAIRED original (this pod's rerun, same code) minus interviewer-free, {fam}, {an}",
                               a=dst_csv, a_mac=mac_csv, b=noinv, b_mac=f"{M}/{nn}", arm=arm))
spec = dict(name=NAME, per_clip_csv=dst_csv, per_clip_csv_mac=mac_csv, model=rows[0]["model"], dtype=rows[0]["dtype"],
            prompt=rows[0]["prompt"], command=CMDS[fam if not NAME.endswith("_bf16") else fam + "_bf16"], sources=sources,
            cells=cells, paired=paired, sidecar=f"{O}/{NAME}.sidecar.json", mass_col=True)
assert len({r["prompt"] for r in rows}) == 1
json.dump(spec, open(f"{W}/spec_{NAME}.json", "w"), indent=1)
subprocess.run(["python3", f"{W}/analyze.py", f"{W}/spec_{NAME}.json"], check=True)
open(f"{O}/{NAME}.RESULT", "w").close()
print("RESULT_MARKER", f"{O}/{NAME}.RESULT", flush=True)
