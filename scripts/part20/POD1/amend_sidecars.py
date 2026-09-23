"""Adds exact pod commands, pod environment and launch-script hashes to every POD1 sidecar (idempotent)."""
import json, glob, hashlib
M="scores/part20/POD1"
S="/workspace/scores/part20/POD1"
def cmd(outdir, tag, mode, fl): return f"cd {S} && HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 python3 {S}/balanced_ft_pitt.py {S}/mf_pitt468_p20.csv {S}/pitt_groupkfold5_pod.csv {S}/{outdir} {tag} {mode} {fl}"
runs = {"balanced_folds": [cmd("balanced_folds","balanced","balanced",f) for f in ("0,3","1,4","2")],
        "control_folds": [cmd("control_folds","uniform","uniform",f) for f in ("0,1,2","4,3")],
        "balanced_rerun_fold3": [cmd("balanced_rerun_fold3","balanced","balanced","3")],
        "balanced_r2_folds": [cmd("balanced_r2_folds","balanced","balanced",f) for f in ("0,1,2","3,4")],
        "control_r2_folds": [cmd("control_r2_folds","uniform","uniform",f) for f in ("0,1,2","3,4")]}
def sha(p):
    with open(p,"rb") as fh: return hashlib.sha256(fh.read(1<<20)).hexdigest()
for sc in sorted(glob.glob(M+"/*.sidecar.json")):
    d=json.load(open(sc))
    dirs=sorted({v.split("/")[-2] for k,v in d["sources_absolute"].items() if k.endswith("_json")})
    ce={}
    for k in dirs:
        if k=="balanced_r1_fold3rerun_folds":
            ce[k]=["assembled on the Mac by copying: folds 0,1,2,4 from balanced_folds, fold 3 from balanced_rerun_fold3"]
            ce["balanced_folds"]=runs["balanced_folds"]; ce["balanced_rerun_fold3"]=runs["balanced_rerun_fold3"]
        else: ce[k]=runs[k]
    ref=d["sources_absolute"]["standard_per_clip"]
    if "POD1/" in ref:
        rdir={"standard_rerun_ft_pitt_oof.csv":"control_folds","standard_rerun_ft_pitt_r2_oof.csv":"control_r2_folds"}[ref.split("/")[-1]]
        ce["reference_run_"+rdir]=runs[rdir]
    d["command_pod_exact"]=ce
    d["command_stats_exact"]="cd "+M+" && RESULT_TITLE='<the result field>' /usr/local/bin/python3 "+" ".join(d["command_stats"].split()[1:])
    d["pod"]={"id":"x9oiqepu2mmpqk","gpu":"NVIDIA H100 NVL 94GB","image":"runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404","python":"3.12.3",
              "torch":"2.8.0+cu128","transformers":"5.17.0","sklearn":"1.9.1","numpy":"2.1.2","librosa":"1.0.0",
              "model_snapshot":"/workspace/hf/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00"}
    d["audio"]={"mac_source":"<local data dir>/DementiaBank/segments (468 wav, 516863704 bytes)",
                "pod_path":"/workspace/data/pitt468 -> /workspace/data/segments","md5_check":"md5 of all 468 wav identical Mac vs pod"}
    for extra in ("launch.sh","launch_r2.sh"):
        d["sources_absolute"][extra]=f"{M}/{extra}"; d["sha256_first_1MB"][extra]=sha(f"{M}/{extra}")
    d["training_order_seed"]="numpy RandomState(fold*10+epoch) as in sft_projector.py; torch.manual_seed(1000+fold) (thinker has no dropout)"
    json.dump(d,open(sc,"w"),indent=1)
    print(sc.split("/")[-1], sorted(ce))
