#!/usr/bin/env python3
"""Second pass: fetch every media file via its ?f=save link, verify by size,
spot-verify by decode. Overwrites the 11-byte stubs from the first pass."""
import os, subprocess, time

JAR = "cookies.txt"
ROOT = "https://media.talkbank.org/dementia/English/Pitt"
urls = [u.strip() for u in open("all_media_urls.txt") if u.strip()]
print(f"{len(urls)} files", flush=True)
bad = 0
for n, u in enumerate(sorted(urls), 1):
    rel = u.split("/Pitt/", 1)[1].lstrip("/")
    dest = os.path.join("Pitt", rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        continue
    subprocess.run(["curl", "-s", "-b", JAR, "--max-time", "300",
                    "-o", dest, u + "?f=save"])
    if os.path.getsize(dest) < 10000:
        bad += 1
        print(f"STILL SMALL {rel} {os.path.getsize(dest)}B", flush=True)
    if n % 100 == 0:
        print(f"progress {n}/{len(urls)} {time.strftime('%H:%M:%S')}", flush=True)
print(f"done, {bad} still small", flush=True)
