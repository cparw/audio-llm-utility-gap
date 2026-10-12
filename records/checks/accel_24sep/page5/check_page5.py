# page5 track: independent check of page 5 content and section placement
import csv, re, subprocess, sys, json
D = "checks/accel_24sep/page5"
P = "<local data dir>/final.pdf"
out = {}
info = subprocess.run(["pdfinfo", P], capture_output=True, text=True).stdout
out["pages"] = int(re.search(r"Pages:\s+(\d+)", info).group(1))

def lines(page):
    rows = list(csv.DictReader(open(f"{D}/page{page}_words.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE))
    words = [r for r in rows if r["level"] == "5"]
    L = {}
    for w in words:
        k = (int(w["block_num"]), int(w["par_num"]), int(w["line_num"]))
        L.setdefault(k, []).append(w)
    res = []
    for k, ws in L.items():
        left = min(float(w["left"]) for w in ws)
        top = min(float(w["top"]) for w in ws)
        bot = max(float(w["top"]) + float(w["height"]) for w in ws)
        res.append(dict(col="L" if left < 306 else "R", left=round(left, 1), top=round(top, 1), bottom=round(bot, 1),
                        text=" ".join(w["text"] for w in ws)))
    res.sort(key=lambda r: (r["col"], r["top"]))
    return res

p5 = lines(5)
out["p5_line_count"] = len(p5)
# every line on page 5 must be a reference start "[n]" or a continuation of one
entries = []
first_non_ref = []
for r in p5:
    m = re.match(r"^\[(\d+)\]", r["text"])
    if m:
        entries.append(int(m.group(1)))
    elif not entries:
        first_non_ref.append(r)
out["p5_ref_numbers"] = entries
out["p5_refs_contiguous"] = entries == list(range(entries[0], entries[-1] + 1))
out["p5_text_before_first_ref"] = [r["text"] for r in first_non_ref if r["col"] == "L"]
# right column's first line is a continuation of [22]; confirm it ends the [22] entry
rc = [r for r in p5 if r["col"] == "R"]
out["p5_right_col_first_line"] = rc[0]["text"] if rc else None
bad_kw = re.compile(r"CONCLUSION|Acknowledg|Compliance|REFERENCES|Table|Fig\.|Figure|Section|\bAUC\b")
out["p5_suspicious_lines"] = [r["text"] for r in p5 if bad_kw.search(r["text"])]
out["p5_text_extent_pt"] = dict(top=min(r["top"] for r in p5), bottom=max(r["bottom"] for r in p5))
out["p5_left_col_bottom_pt"] = max(r["bottom"] for r in p5 if r["col"] == "L")
out["p5_right_col_bottom_pt"] = max(r["bottom"] for r in p5 if r["col"] == "R")

p4 = lines(4)
def find(lines_, pat):
    return [(i, r) for i, r in enumerate(lines_) if re.search(pat, r["text"])]
concl = find(p4, r"^5\. CONCLUSIONS")
ack = find(p4, r"^Acknowledgments\.")
comp = find(p4, r"^Compliance with ethical standards\.")
refs = find(p4, r"^6\. REFERENCES")
ci, cr = concl[0]
ai, ar = ack[0]
out["p4_conclusions_heading"] = cr
out["p4_conclusions_last_line"] = p4[ai - 1]
out["p4_ack_first_line"] = ar
out["p4_ack_last_line"] = p4[comp[0][0] - 1]
out["p4_compliance_first_line"] = comp[0][1]
out["p4_compliance_last_line"] = p4[refs[0][0] - 1]
out["p4_references_heading"] = refs[0][1]
out["p4_ref1_lines"] = [r["text"] for r in p4[refs[0][0] + 1:] if r["col"] == "R"]
out["p4_right_col_bottom_pt"] = max(r["bottom"] for r in p4 if r["col"] == "R")
out["p4_left_col_bottom_pt"] = max(r["bottom"] for r in p4 if r["col"] == "L")
# where each heading appears across the whole PDF
hits = {}
for pg in range(1, out["pages"] + 1):
    for r in lines(pg):
        for key, pat in [("CONCLUSIONS", r"CONCLUSIONS"), ("Acknowledgments", r"^Acknowledgments"),
                         ("Compliance", r"^Compliance with ethical"), ("REFERENCES", r"REFERENCES"),
                         ("ref [1]", r"^\[1\] "), ("ref [30]", r"^\[30\] ")]:
            if re.search(pat, r["text"]):
                hits.setdefault(key, []).append(dict(page=pg, col=r["col"], top=r["top"], text=r["text"]))
out["heading_hits_all_pages"] = hits
json.dump(out, open(f"{D}/check_page5.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
