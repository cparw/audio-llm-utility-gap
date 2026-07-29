import os, csv, argparse, gc
import numpy as np, torch, librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

def find_encoder(model):
    for path in ("audio_tower", "model.audio_tower"):
        m = model; ok = True
        for a in path.split("."):
            if hasattr(m, a): m = getattr(m, a)
            else: ok = False; break
        if ok: return m
    for name, mod in model.named_modules():
        if "AudioEncoder" in type(mod).__name__ and hasattr(mod, "layers"):
            return mod
    raise RuntimeError("audio encoder submodule not found")

def wins(y, sr=16000, win=30.0):
    n = int(win*sr)
    return [y] if len(y) <= n else [y[i:i+n] for i in range(0, len(y), n)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = list(csv.DictReader(open(args.manifest)))
    if args.limit: rows = rows[:args.limit]
    proc = AutoProcessor.from_pretrained(args.model)
    fe = proc.feature_extractor; sr = fe.sampling_rate
    print("loading model on CPU...", flush=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    enc = find_encoder(model).to("cuda").eval()
    model = None; gc.collect(); torch.cuda.empty_cache()
    print("encoder:", type(enc).__name__, "on", next(enc.parameters()).device, "| clips:", len(rows), "| sr:", sr, flush=True)
    feats = None; meta = []
    for i, r in enumerate(rows):
        try:
            y, _ = librosa.load(r["filepath"], sr=sr, mono=True)
            acc = None; nw = 0
            for w in wins(y, sr):
                x = fe(w, sampling_rate=sr, return_tensors="pt")["input_features"].to("cuda", dtype=torch.float16)
                with torch.no_grad():
                    hs = enc(input_features=x, output_hidden_states=True).hidden_states
                pooled = [h.float().mean(dim=1).squeeze(0).cpu().numpy() for h in hs]
                if acc is None: acc = [p.copy() for p in pooled]
                else:
                    for li in range(len(pooled)): acc[li] += pooled[li]
                nw += 1
            vecs = [a/nw for a in acc]
            if feats is None: feats = [[] for _ in vecs]
            for li, v in enumerate(vecs): feats[li].append(v)
            meta.append((r["speaker_id"], int(r["label"]), r["task_type"], r.get("group",""), r["filepath"]))
            if (i+1) % 50 == 0: print(f"  {i+1}/{len(rows)}", flush=True)
        except Exception as e:
            print("FAIL", r["filepath"], repr(e), flush=True)
    L = len(feats)
    arr = {f"layer_{li:02d}": np.stack(feats[li]) for li in range(L)}
    np.savez_compressed(os.path.join(args.out, "encoder_features.npz"),
        speaker=np.array([m[0] for m in meta]), label=np.array([m[1] for m in meta]),
        task=np.array([m[2] for m in meta]), group=np.array([m[3] for m in meta]),
        path=np.array([m[4] for m in meta]), n_layers=L, **arr)
    print("SAVED", L, "layers x", len(meta), "clips ->", args.out, flush=True)

if __name__ == "__main__": main()
