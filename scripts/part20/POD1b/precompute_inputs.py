"""Compute the processor inputs for all 468 Pitt clips ONCE, exactly as sft_projector.py inputs_for() does
(librosa.load sr=16000, first 30 s, chat template with the verbatim prompt, Qwen2_5OmniProcessor), on CPU,
in parallel worker processes, and save them so the three seed runs do not each repeat this CPU work.
usage: precompute_inputs.py MANIFEST OUT.pt"""
import os, sys, csv, time, torch, librosa
torch.set_num_threads(2)
from multiprocessing import get_context
from transformers import Qwen2_5OmniProcessor as Proc
MAN, OUT = sys.argv[1], sys.argv[2]
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
proc = Proc.from_pretrained("Qwen/Qwen2.5-Omni-7B")
rows = list(csv.DictReader(open(MAN)))
def one(i):
    torch.set_num_threads(2)
    r = rows[i]
    x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp.pop("use_audio_in_video", None)
    return i, {k: v.numpy() for k, v in dict(inp).items()}   # numpy over the pipe (tensor fd passing fails)
if __name__ == "__main__":
    t = time.time()
    with get_context("fork").Pool(64) as pool:
        res = dict(pool.map(one, range(len(rows)), chunksize=2))
    inputs = [{k: torch.from_numpy(v) for k, v in res[i].items()} for i in range(len(rows))]
    torch.save({"paths": [r["path"] for r in rows], "prompt": P, "inputs": inputs}, OUT)
    print(f"saved {len(res)} inputs to {OUT} in {time.time()-t:.1f} s; keys {sorted(res[0].keys())}", flush=True)
