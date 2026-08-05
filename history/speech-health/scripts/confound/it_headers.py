"""
ITALIAN CONFOUND, step 1: file-level forensics from raw bytes + filenames.
No audio decoding. Parses the RIFF chunk graph byte by byte, so we see every chunk,
including any LIST/INFO/fact/bext/id3 tag an encoder left behind.
"""
import os, re, csv, struct, json, hashlib

ROOT = "/project2/msoleyma_946/speech_health/italian"
R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/final"
os.makedirs(OUT, exist_ok=True)

def parse_riff(path):
    d = {"path": path, "err": ""}
    try:
        sz = os.path.getsize(path)
        d["file_bytes"] = sz
        with open(path, "rb") as f:
            head = f.read(12)
            d["riff_magic"] = head[0:4].decode("latin1")
            d["riff_size"] = struct.unpack("<I", head[4:8])[0]
            d["wave_magic"] = head[8:12].decode("latin1")
            d["riff_size_matches"] = (d["riff_size"] + 8 == sz)
            chunks = []
            pos = 12
            fmt = {}
            data_off = data_len = None
            extra = {}
            while pos + 8 <= sz:
                f.seek(pos)
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                cid = hdr[0:4].decode("latin1", "replace")
                clen = struct.unpack("<I", hdr[4:8])[0]
                chunks.append(f"{cid}:{clen}")
                body_off = pos + 8
                if cid == "fmt ":
                    b = f.read(min(clen, 40))
                    (af, nch, srate, brate, balign, bits) = struct.unpack("<HHIIHH", b[:16])
                    fmt = dict(audio_format=af, channels=nch, sample_rate=srate,
                               byte_rate=brate, block_align=balign, bits=bits,
                               fmt_chunk_len=clen)
                    if clen > 16:
                        fmt["fmt_ext_bytes"] = clen - 16
                        fmt["fmt_ext_hex"] = b[16:min(len(b), 40)].hex()
                elif cid == "data":
                    data_off, data_len = body_off, clen
                elif clen < 4096:
                    raw = f.read(clen)
                    extra[cid] = raw[:400].decode("latin1", "replace")
                pos = body_off + clen + (clen & 1)
            d.update(fmt)
            d["chunks"] = "|".join(chunks)
            d["n_chunks"] = len(chunks)
            d["extra_chunks"] = json.dumps(extra) if extra else ""
            d["data_offset"] = data_off
            d["data_len"] = data_len
            if data_off is not None and data_len is not None:
                d["data_len_declared_ok"] = (data_off + data_len <= sz)
                d["trailing_bytes"] = sz - (data_off + data_len)
                ba = fmt.get("block_align", 1) or 1
                d["n_frames"] = data_len // ba
                d["dur_s"] = round(d["n_frames"] / fmt.get("sample_rate", 1), 4)
    except Exception as e:
        d["err"] = f"{type(e).__name__}:{e}"
    return d

# filename code:  TASK + SPKCODE + YY + SEX + DDMMYYYY + HHMM
FN = re.compile(r"^(B1|B2|D1|D2|FB1|PR1|PR11|VA1|VA2|VE1|VE2|VI1|VI2|VO1|VO2|VU1|VU2|[A-Z]{1,3}\d{0,2})(.+?)(\d{2})([MF])(\d{6}|\d{8})(\d{4})$")

def parse_name(fn):
    base = os.path.splitext(os.path.basename(fn))[0].rstrip(".")
    m = FN.match(base)
    if not m:
        return {"fn_parsed": 0, "fn_base": base}
    task, code, yy, sex, dd, hhmm = m.groups()
    if len(dd) == 8:
        date = f"{dd[4:8]}-{dd[2:4]}-{dd[0:2]}"
    else:
        date = f"20{dd[4:6]}-{dd[2:4]}-{dd[0:2]}"
    return {"fn_parsed": 1, "fn_base": base, "fn_task": task, "fn_spkcode": code,
            "fn_birthyy": yy, "fn_sex": sex, "fn_datefmt": len(dd),
            "fn_date": date,
            "fn_time": f"{hhmm[0:2]}:{hhmm[2:4]}", "fn_hhmm": hhmm}

rows = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    for fn in sorted(filenames):
        if not fn.lower().endswith(".wav"):
            continue
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, ROOT)
        grp = rel.split(os.sep)[0]
        d = parse_riff(p)
        d.update(parse_name(fn))
        d["group_dir"] = grp
        d["rel"] = rel
        d["subdir"] = os.path.dirname(rel)
        st = os.stat(p)
        d["mtime"] = st.st_mtime
        rows.append(d)

keys = []
for r in rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(f"{OUT}/italian_headers.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"n_wav = {len(rows)}")
from collections import Counter, defaultdict

def tab(field):
    c = defaultdict(Counter)
    for r in rows:
        c[r.get("group_dir")][r.get(field)] += 1
    print(f"\n--- {field} by group ---")
    for g in sorted(c):
        print(f"  {g[:34]:36s} " + "  ".join(f"{k}={v}" for k, v in sorted(c[g].items(), key=lambda x: -x[1])))

for fld in ["sample_rate", "bits", "channels", "audio_format", "fmt_chunk_len",
            "n_chunks", "chunks", "riff_size_matches", "trailing_bytes", "fn_task"]:
    tab(fld)

print("\n--- any non-empty extra chunks ---")
ec = Counter(r.get("extra_chunks", "")[:200] for r in rows if r.get("extra_chunks"))
for k, v in ec.most_common(20):
    print(f"  n={v}  {k}")

print("\n--- recording DATE by group ---")
c = defaultdict(Counter)
for r in rows:
    c[r["group_dir"]][r.get("fn_date", "NA")] += 1
for g in sorted(c):
    print(f"  {g[:34]:36s}")
    for k, v in sorted(c[g].items()):
        print(f"      {k}  n={v}")

nf=[r for r in rows if not r.get("fn_parsed")]
print(f"\n--- filename parse failures: {len(nf)} ---")
for r in nf[:20]:
    print("   ", r["rel"])
for fld in ["fn_datefmt","fn_sex"]:
    tab(fld)
print(f"\nwrote {OUT}/italian_headers.csv")
