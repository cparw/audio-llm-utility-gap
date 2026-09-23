#!/usr/bin/env python3
"""PART 22 step-1 quote checker (independent). For every `path:A-B` reference in the step-1 markdown that is followed
by a fenced block of numbered lines ("  NN  code"), open the real file and compare each quoted line, byte for byte
after stripping trailing whitespace, to the file line with that number.

usage: python3 P22_verify_step1.py MD PREFIX=ROOT [PREFIX=ROOT ...]
  e.g. transformers/=/usr/local/lib/python3.12/dist-packages/transformers/  src/hf/=/path/to/snapshot/
References whose prefix has no mapping are reported as skipped.
"""
import re, sys, json

md, maps = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
lines = open(md, encoding="utf-8").read().split("\n")
ref_re = re.compile(r"^`([^`]+):(\d+)-(\d+)`\s*$")
num_re = re.compile(r"^ *(\d+)  (.*)$")
res = []
i = 0
while i < len(lines):
    m = ref_re.match(lines[i])
    if not m:
        i += 1; continue
    path, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    j = i + 1
    while j < len(lines) and not lines[j].startswith("```"):
        j += 1
    k = j + 1
    quoted = []
    while k < len(lines) and not lines[k].startswith("```"):
        mm = num_re.match(lines[k])
        if mm:
            quoted.append((int(mm.group(1)), mm.group(2)))
        elif lines[k].strip().isdigit():
            quoted.append((int(lines[k].strip()), ""))
        k += 1
    real = None
    for pre, root in maps.items():
        if path.startswith(pre):
            real = root + path[len(pre):]
    rec = dict(ref=f"{path}:{a}-{b}", n_quoted=len(quoted), expected_n=b - a + 1)
    if real is None:
        rec["status"] = "SKIPPED (no root mapping)"
    else:
        try:
            src = open(real, encoding="utf-8").read().split("\n")
            bad = []
            for n, code in quoted:
                actual = src[n - 1].rstrip() if 0 < n <= len(src) else None
                if actual != code.rstrip():
                    bad.append(dict(line=n, quoted=code, actual=actual))
            nums = [n for n, _ in quoted]
            rec["file"] = real
            rec["line_numbers_contiguous"] = nums == list(range(a, b + 1))
            rec["mismatches"] = bad
            rec["status"] = "OK" if not bad and rec["line_numbers_contiguous"] else "MISMATCH"
        except FileNotFoundError:
            rec["status"] = f"FILE NOT FOUND {real}"
    res.append(rec)
    i = k + 1
summary = dict(n_refs=len(res), ok=sum(r["status"] == "OK" for r in res),
               mismatch=sum(r["status"] == "MISMATCH" for r in res),
               skipped=sum(r["status"].startswith("SKIPPED") for r in res),
               missing=sum(r["status"].startswith("FILE NOT FOUND") for r in res))
print(json.dumps(dict(summary=summary, refs=res), indent=1))
