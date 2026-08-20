#!/usr/bin/env python3
"""BFS crawl of the Pitt media tree using the established TalkBank session.
Normalises the server's inconsistent slashes instead of guessing URL shapes."""
import re, subprocess, sys, os, time

JAR = "/Volumes/G-Drive Pro/DementiaBank/cookies.txt"
OUT = "/Volumes/G-Drive Pro/DementiaBank/Pitt"
ROOT = "https://media.talkbank.org/dementia/English/Pitt"

def fetch(url, timeout=30):
    r = subprocess.run(["curl","-s","-b",JAR,"--max-time",str(timeout),url],
                       capture_output=True)
    return r.stdout.decode("utf-8","ignore")

def norm(u):
    head, rest = u.split("://",1)
    rest = rest.replace(":443","")
    while "//" in rest: rest = rest.replace("//","/")
    return head+"://"+rest

def listing(url):
    html = fetch(url.rstrip("/")+"/")
    hrefs = re.findall(r'href="(https://media\.talkbank\.org[^"]*)"', html)
    out=set()
    for h in hrefs:
        if "?f=save" in h: continue
        out.add(norm(h))
    return out

root = norm(ROOT)
dirs=[root]; media=[]; seen=set()
while dirs:
    d = dirs.pop(0)
    if d in seen: continue
    seen.add(d)
    for h in listing(d):
        if not h.startswith(root): continue
        if h == d or len(h) <= len(d): continue
        if h.lower().endswith((".mp3",".wav",".mp4")):
            media.append(h)
        elif "." not in h.rsplit("/",1)[-1]:
            dirs.append(h)
print(f"directories: {len(seen)}  media files: {len(media)}", flush=True)
with open("/Volumes/G-Drive Pro/DementiaBank/all_media_urls.txt","w") as f:
    f.write("\n".join(sorted(set(media)))+"\n")

n=0; bad=0
for u in sorted(set(media)):
    rel = u[len(root)+1:]
    dest = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    n+=1
    if os.path.exists(dest) and os.path.getsize(dest)>1000: continue
    subprocess.run(["curl","-s","-b",JAR,"--max-time","180","-o",dest,u])
    with open(dest,"rb") as fh: head=fh.read(3)
    if head not in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        bad+=1
        print(f"SUSPECT {rel} starts {head!r}", flush=True)
    if n%100==0: print(f"progress {n}/{len(set(media))} {time.strftime('%H:%M:%S')}", flush=True)
print(f"done: {n} files, {bad} suspect", flush=True)
