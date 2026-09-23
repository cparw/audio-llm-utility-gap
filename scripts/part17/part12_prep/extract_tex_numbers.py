#!/usr/local/bin/python3
"""B. Emit every numeric literal in a LaTeX file with its location and context.

For each item: line number, kind (point / range / change / interval / percent / integer / sci),
the printed text, value(s), printed decimals, the enclosing sentence (prose) or table row (tables),
the clause around the number, the section heading, whether it sits inside a red span
(\\textcolor{red}{...}, {\\color{red} ...} or a bare \\color{red} to the end of its group), and whether
it is commented out (% comment or a comment environment). Tables are parsed row by row and each cell
gets a key "row label | column label" built from the multicolumn header.

Ranges "A to B", "A--B", "between A and B", "A on X to B on Y" are one item with two values.
"from A to B" with a change verb (lifts, falls, rises, moves ...) is kind=change (before, after).
"(A to B)" or "[A, B]" right after a number is kind=interval, attached to that number.

Usage: extract_tex_numbers.py TEX [--out numbers.csv]
"""
import argparse
import csv
import os
import re
import sys

WORDNUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "eleven": 11, "twelve": 12}
CHANGE_VERBS = r"(lift|lifts|lifted|move|moves|moved|fall|falls|fell|rise|rises|rose|drop|drops|dropped|improv\w*|goes|go|chang\w*|shrink\w*|narrow\w*|increas\w*|decreas\w*|break\w*|breaks|jump\w*|climb\w*|collaps\w*|reduc\w*)"


# ----------------------------------------------------------------------------- masks
def comment_mask(text):
    """True for every character that is commented out (line % comments and comment environments)."""
    m = [False] * len(text)
    pos = 0
    for line in text.split("\n"):
        i = 0
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                for k in range(pos + i, pos + len(line)):
                    m[k] = True
                break
            i += 1
        pos += len(line) + 1
    for mo in re.finditer(r"\\begin\{comment\}.*?\\end\{comment\}", text, re.S):
        for k in range(mo.start(), mo.end()):
            m[k] = True
    return m


def match_brace(text, i):
    """text[i] == '{' ; return index of the matching '}' (ignores escaped braces)."""
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return len(text) - 1


def red_mask(text):
    m = [False] * len(text)
    spans = []
    for mo in re.finditer(r"\\textcolor\{red\}\s*\{", text):
        b = mo.end() - 1
        e = match_brace(text, b)
        spans.append((mo.start(), e + 1))
    for mo in re.finditer(r"\\color\{red\}", text):
        # find the enclosing group: scan back for the unmatched '{'
        depth = 0
        k = mo.start() - 1
        start = None
        while k >= 0:
            c = text[k]
            if c == "}" and (k == 0 or text[k - 1] != "\\"):
                depth += 1
            elif c == "{" and (k == 0 or text[k - 1] != "\\"):
                if depth == 0:
                    start = k
                    break
                depth -= 1
            k -= 1
        if start is None:
            end = text.find("\n\n", mo.end())
            end = len(text) if end < 0 else end
            spans.append((mo.start(), end))
        else:
            spans.append((mo.start(), match_brace(text, start) + 1))
    for a, b in spans:
        for k in range(a, min(b, len(text))):
            m[k] = True
    return m, spans


# LaTeX constructs whose numbers are not results: blank them out (offsets preserved)
SKIP_PATTERNS = [
    r"\\(?:cite|ref|label|url|eqref|includegraphics|graphicspath|bibliographystyle|bibliography|thanks|name|address|href)\*?(\[[^\]]*\])?\{[^{}]*\}",
    r"\\(?:setlength|addtolength)\s*\{?\\?[A-Za-z@]*\}?\s*\{[^{}]*\}",
    r"\\setlength\\[A-Za-z]+\{[^{}]*\}", r"\\tabcolsep", r"\\(?:vspace|hspace)\*?\{[^{}]*\}",
    r"\\cmidrule(\([^)]*\))?\{[^{}]*\}", r"\\addlinespace(\[[^\]]*\])?", r"\\multicolumn\{\d+\}\{[^{}]*\}",
    r"\\begin\{[^{}]*\}(\{[^{}]*\})*(\[[^\]]*\])?", r"\\end\{[^{}]*\}",
    r"@\{[^{}]*\}", r"\\extracolsep\{[^{}]*\}", r"\{@\{\\extracolsep\{\\fill\}\}[^{}]*@\{\}\}",
    r"\\(?:re)?newcommand\{[^{}]*\}\[\d+\]", r"#\d", r"\\renewcommand.*", r"\\let\\\w+\\\w+", r"\\footnote\{\\url\{[^{}]*\}\}", r"\\url\{[^{}]*\}", r"\\ell_\d", r"\\[A-Za-z]+(?=\d)",
    r"Qwen2\.5-Omni", r"Qwen3-Omni(-30B-A3B)?", r"Qwen2-Audio", r"Qwen\d(\.\d)?", r"Audio Flamingo 2 and 3",
    r"Audio Flamingo [23]", r"Flamingo [23]", r"ADReSS-2020", r"ADReSS-M", r"AVEC\s+2019", r"PHQ-8", r"whisper-large-v\d",
    r"W911NF-[\d-]+", r"R61MH\d+", r"AG0\d+", r"GroupKFold", r"MDVR-KCL", r"eGeMAPS(v\d+)?", r"Kimi-Audio",
    r"\b[A-Za-z]+\d+[A-Za-z\d]*\b", r"\b\d+[A-Za-z]+\d*\b(?<!\d\\s)",
    r"\bSection~?", r"\bTable~?", r"\bFig\.~?", r"\bEq\.~?",
]
SKIP_RE = re.compile("|".join("(?:%s)" % p for p in SKIP_PATTERNS))

NUM_RE = re.compile(r"(?<![\w.])(\$?)([+\-\u2212]?)(\d+(?:\.\d+)?)(?!\.\d)(?![A-Za-z])")
SCI_RE = re.compile(r"(\d+)\^\{?(-?\d+)\}?")


def blank(text, a, b):
    return text[:a] + " " * (b - a) + text[b:]


def masked(text):
    t = text
    for mo in SCI_RE.finditer(text):
        pass
    for mo in SKIP_RE.finditer(text):
        t = blank(t, mo.start(), mo.end())
    return t


# ----------------------------------------------------------------------------- structure
def headings(text, cmask=None):
    hs = []
    for mo in re.finditer(r"\\(sub)*section\*?\{([^{}]*)\}", text):
        if cmask is not None and cmask[mo.start()]:
            continue
        hs.append((mo.start(), mo.group(2)))
    for mo in re.finditer(r"\\begin\{abstract\}", text):
        hs.append((mo.start(), "Abstract"))
    for mo in re.finditer(r"\\begin\{keywords\}", text):
        hs.append((mo.start(), "Keywords"))
    for mo in re.finditer(r"\\maketitle", text):
        hs.append((mo.start(), "Title block"))
    hs.sort()
    return hs


def heading_at(hs, pos):
    h = "Preamble"
    for p, name in hs:
        if p <= pos:
            h = name
        else:
            break
    return h


def split_cells(row):
    cells, cur, i = [], "", 0
    while i < len(row):
        c = row[i]
        if c == "\\" and i + 1 < len(row):
            cur += row[i:i + 2]
            i += 2
            continue
        if c == "{":
            j = match_brace(row, i)
            cur += row[i:j + 1]
            i = j + 1
            continue
        if c == "&":
            cells.append(cur)
            cur = ""
        else:
            cur += c
        i += 1
    cells.append(cur)
    return cells


def unmulticol(s):
    """Replace every \\multicolumn{n}{spec}{text} by text (brace aware)."""
    while True:
        mo = re.search(r"\\multicolumn\{\d+\}\{", s)
        if not mo:
            return s
        e1 = match_brace(s, mo.end() - 1)
        if e1 + 1 >= len(s) or s[e1 + 1] != "{":
            return s
        e2 = match_brace(s, e1 + 1)
        s = s[:mo.start()] + s[e1 + 2:e2] + s[e2 + 1:]


def strip_tex(s):
    s = re.sub(r"\\cite\{[^{}]*\}", "", s)
    s = unmulticol(s)
    s = re.sub(r"\\(?:emph|textbf|textit|mathrm|text)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(midrule|toprule|bottomrule|hline|noindent|small|footnotesize|centering)", "", s)
    s = re.sub(r"\\cmidrule(\([^)]*\))?\{[^{}]*\}", "", s)
    s = re.sub(r"\\addlinespace(\[[^\]]*\])?", "", s)
    s = s.replace("~", " ").replace("\\,", " ").replace("\\%", "%").replace("$", "")
    s = re.sub(r"[{}]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def multicol_expand(cells):
    out = []
    for c in cells:
        mo = re.match(r"\s*\\multicolumn\{(\d+)\}", c)
        if mo:
            out.extend([strip_tex(c)] * int(mo.group(1)))
        else:
            out.append(strip_tex(c))
    return out


def parse_tables(text, cmask):
    """Yield (row_start, row_end, row_label, [(cell_start, cell_end, col_label)], table_label, caption, group)."""
    out = []
    for tm in re.finditer(r"\\begin\{(tabular\*?)\}", text):
        env = tm.group(1)
        end = text.find("\\end{%s}" % env, tm.end())
        if end < 0:
            continue
        # body starts after the column spec argument(s)
        k = tm.end()
        while k < len(text) and text[k] in " \n{":
            if text[k] == "{":
                k = match_brace(text, k) + 1
            else:
                k += 1
        body_start, body = k, text[k:end]
        # table float for caption/label
        fl_start = text.rfind("\\begin{table", 0, tm.start())
        fl_end = text.find("\\end{table", end)
        flo = text[fl_start:fl_end] if fl_start >= 0 else ""
        lab = re.search(r"\\label\{([^{}]*)\}", flo)
        cap = re.search(r"\\caption\{", flo)
        caption = ""
        if cap:
            cs = fl_start + cap.end() - 1
            caption = strip_tex(text[cs + 1:match_brace(text, cs)])
        mid = body.find("\\midrule")
        header_rows = []
        rows = []
        pos = 0
        for rm in re.finditer(r"\\\\(\[[^\]]*\])?", body):
            rows.append((pos, rm.start()))
            pos = rm.end()
        rows.append((pos, len(body)))
        ncol = None
        col_labels = []
        group = ""
        for a, b in rows:
            if mid >= 0 and a < mid < b:
                a = mid + len("\\midrule")
            raw = body[a:b]
            abs_a = body_start + a
            if all(cmask[abs_a + i] for i in range(len(raw)) if not raw[i].isspace()) and raw.strip():
                continue
            clean = re.sub(r"\\(toprule|midrule|bottomrule|hline)", "", raw)
            clean = re.sub(r"\\cmidrule(\([^)]*\))?\{[^{}]*\}", "", clean)
            clean = re.sub(r"\\addlinespace(\[[^\]]*\])?", "", clean)
            if not clean.strip():
                continue
            cells = split_cells(clean)
            if mid >= 0 and a < mid:
                header_rows.append(multicol_expand(cells))
                continue
            exp = multicol_expand(cells)
            if ncol is None:
                ncol = max([len(h) for h in header_rows] + [len(exp)])
                col_labels = []
                for c in range(ncol):
                    parts = [h[c] for h in header_rows if c < len(h) and h[c]]
                    col_labels.append(" ".join(parts))
            if len(cells) == 1 and "\\multicolumn" in cells[0]:
                group = strip_tex(cells[0])
                continue
            # absolute offsets of cells inside the original row
            offs = []
            cur = abs_a + raw.find(clean.lstrip()[:1]) if clean.strip() else abs_a
            search_from = abs_a
            for c in cells:
                cs = text.find(c.strip(), search_from, body_start + b + 1) if c.strip() else search_from
                if cs < 0:
                    cs = search_from
                offs.append((cs, cs + len(c.strip())))
                search_from = cs + len(c.strip())
            row_label = strip_tex(cells[0])
            cell_list = [(offs[i][0], offs[i][1], col_labels[i] if i < len(col_labels) else "col%d" % i) for i in range(len(cells))]
            out.append((abs_a, body_start + b, row_label, cell_list, lab.group(1) if lab else "", caption, group))
    return out


# ----------------------------------------------------------------------------- sentences
ABBR = ("e.g.", "i.e.", "et al.", "Fig.", "Eq.", "vs.", "Sec.", "Tab.", "cf.", "approx.")


def sentences(text, a, b):
    """Split text[a:b] into sentences; return list of (start, end)."""
    out = []
    s = a
    for mo in re.finditer(r"[.!?](?=\s+[A-Z\\(\[$]|\s*\n\s*\n|\s*$)", text[a:b]):
        e = a + mo.end()
        prev = text[max(a, e - 8):e]
        if any(prev.endswith(x) for x in ABBR):
            continue
        # a decimal like "0.5." ends a sentence, "0.75 to" does not match because no space-capital follows
        out.append((s, e))
        s = e
    if s < b:
        out.append((s, b))
    return out


def clause_of(text, a, b, pos):
    """Clause around pos inside sentence text[a:b], split at , ; : and 'while' / 'whereas' / 'but'."""
    seg = text[a:b]
    cuts = [0]
    for mo in re.finditer(r"[,;:]\s|\s(while|whereas|but|against)\s", seg):
        cuts.append(mo.end())
    cuts.append(len(seg))
    rel = pos - a
    for i in range(len(cuts) - 1):
        if cuts[i] <= rel < cuts[i + 1]:
            return seg[cuts[i]:cuts[i + 1]]
    return seg


# ----------------------------------------------------------------------------- numbers
def decimals(s):
    return len(s.split(".")[1]) if "." in s else 0


def find_numbers(text, mtext, a, b):
    """Return list of dicts for numeric literals in text[a:b] (using the masked copy for detection)."""
    nums = []
    for mo in NUM_RE.finditer(mtext, a, b):
        sign = mo.group(2).replace("\u2212", "-")
        raw = mo.group(3)
        st, en = mo.start(3), mo.end(3)
        if sign:
            st = mo.start(2)
        # sign only counts when it is math minus/plus ($-0.12$, $+0.01$) or directly before the digit after a space
        if sign and not (mtext[max(0, st - 1)] in " ($[" or mtext[max(0, st - 2):st] == "$"):
            sign = ""
            st = mo.start(3)
        after = mtext[en:en + 6]
        unit = ""
        if after.startswith("\\%") or after.startswith("%"):
            unit = "%"
        elif re.match(r"\s*percent", mtext[en:en + 9]):
            unit = "%"
        elif re.match(r"\\,\s*s\b", after) or re.match(r"\s*s\b", after):
            unit = "s"
        before = mtext[max(0, st - 4):st]
        sci = re.match(r"\^\{?(-?\d+)\}?", mtext[en:en + 8])
        if sci:
            val = float(raw) ** float(sci.group(1))
            nums.append(dict(start=st, end=en + sci.end(), raw=raw + "^" + sci.group(1), value=val, dp=None, unit="sci"))
            continue
        if re.match(r"\^\{?-?\d", mtext[max(0, st - 3):st + 1] if False else ""):
            continue
        if before.endswith("^{-") or before.endswith("^{") or before.endswith("^"):
            continue
        val = float(sign + raw) if sign != "+" else float(raw)
        nums.append(dict(start=st, end=en, raw=(sign + raw), value=val, dp=decimals(raw), unit=unit))
    # word numbers only in checkable patterns ("seven years", "four of the six", "six of nine")
    for mo in re.finditer(r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b(?=\s+(years|of\s+(the\s+)?(six|seven|nine|nine)\b|percent))",
                          text[a:b], re.I):
        st = a + mo.start()
        if mtext[st] == " " and text[st] != " ":
            pass
        nums.append(dict(start=st, end=a + mo.end(), raw=mo.group(1), value=float(WORDNUM[mo.group(1).lower()]), dp=0,
                         unit="word" + ("_years" if mo.group(2).lower() == "years" else "")))
    nums.sort(key=lambda d: d["start"])
    return nums


def group_numbers(text, nums):
    """Combine consecutive numbers into ranges / changes / intervals."""
    out = []
    i = 0
    while i < len(nums):
        n = nums[i]
        if i + 1 < len(nums):
            m = nums[i + 1]
            between = text[n["end"]:m["start"]]
            pre = text[max(0, n["start"] - 60):n["start"]]
            bw = between.replace("$", "")
            kind = None
            labels = ("", "")
            if re.fullmatch(r"\s*(--|\u2013|\u2014)\s*", bw):
                kind = "range"
            elif re.fullmatch(r"\s*,\s*", bw) and pre.rstrip().endswith("[") and text[m["end"]:m["end"] + 2].strip().startswith("]"):
                kind = "interval"
            elif re.fullmatch(r"\s+to\s+", bw):
                if re.search(r"\(\s*(95\\?%\s*)?(bootstrap\s+)?(interval\s+)?$", pre) and out and not pre.rstrip().endswith("between"):
                    kind = "interval"
                elif re.search(r"\bfrom\s+$", pre) and re.search(CHANGE_VERBS, text[max(0, n["start"] - 120):n["start"]], re.I):
                    kind = "change"
                else:
                    kind = "range"
            elif re.fullmatch(r"\s+and\s+", bw) and re.search(r"between\s+$", pre):
                kind = "range"
            else:
                lab = re.fullmatch(r"\s+on\s+([A-Za-z][\w\-']*)\s+to\s+", bw)
                if lab:
                    kind = "range"
                    after = re.match(r"\s+on\s+([A-Za-z][\w\-']*)", text[m["end"]:m["end"] + 40])
                    labels = (lab.group(1), after.group(1) if after else "")
            if kind and n["unit"] not in ("sci",) and m["unit"] not in ("sci",) and not n["unit"].startswith("word"):
                out.append(dict(start=n["start"], end=m["end"], raw=text[n["start"]:m["end"]], kind=kind,
                                value=n["value"], value2=m["value"], dp=n["dp"], dp2=m["dp"], unit=n["unit"] or m["unit"],
                                labels=labels))
                i += 2
                continue
        k = "point"
        if n["unit"] == "%":
            k = "percent"
        elif n["unit"] == "sci":
            k = "sci"
        elif n["unit"].startswith("word"):
            k = n["unit"]
        elif n["dp"] == 0:
            k = "integer"
        out.append(dict(start=n["start"], end=n["end"], raw=n["raw"], kind=k, value=n["value"], value2="",
                        dp=n["dp"], dp2="", unit=n["unit"], labels=("", "")))
        i += 1
    return out


def line_of(starts, pos):
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def extract(path):
    text = open(path, encoding="utf-8").read()
    starts = [0]
    for mo in re.finditer("\n", text):
        starts.append(mo.end())
    mo = re.search(r"^[ \t]*\\begin\{document\}", text, re.M)       # first uncommented \begin{document}
    doc_start = mo.start() if mo else 0
    doc_end = text.find("\\end{document}")
    doc_end = len(text) if doc_end < 0 else doc_end
    cmask = comment_mask(text)
    rmask, rspans = red_mask(text)
    hs = headings(text, cmask)
    mtext = masked(text)
    items = []
    tables = parse_tables(text, cmask)
    table_ranges = []
    for (ra, rb, rlabel, cells, tlabel, caption, group) in tables:
        table_ranges.append((ra, rb))
        for ci, (cs, ce, clab) in enumerate(cells):
            if ci == 0 or ce <= cs:
                continue
            nums = group_numbers(text, find_numbers(text, mtext, cs, ce))
            role = []
            if len(nums) == 2 and re.search(r"clips|speakers|\(", clab + text[cs:ce], re.I):
                role = ["clips", "speakers"]
            for j, n in enumerate(nums):
                n.update(dict(section=heading_at(hs, cs), context=strip_tex(text[ra:rb]), clause=strip_tex(text[cs:ce]),
                              prev_context="", para_before="", table=tlabel, table_caption=caption[:200], table_row=rlabel,
                              table_col=clab + ((" [" + role[j] + "]") if role else ""), table_group=group,
                              table_key="%s | %s" % (rlabel, clab + ((" [" + role[j] + "]") if role else ""))))
                items.append(n)
    # prose: everything in the document outside tabular bodies
    blocks = []
    cur = doc_start
    for ra, rb in sorted(table_ranges):
        pass
    tab_spans = []
    for tm in re.finditer(r"\\begin\{(tabular\*?)\}.*?\\end\{\1\}", text, re.S):
        tab_spans.append((tm.start(), tm.end()))
    cur = doc_start
    for a, b in sorted(tab_spans):
        if a > cur:
            blocks.append((cur, a))
        cur = max(cur, b)
    blocks.append((cur, doc_end))
    for a, b in blocks:
        # paragraphs
        for pm in re.finditer(r"(?:[^\n]|\n(?!\s*\n))+", text[a:b]):
            pa, pb = a + pm.start(), a + pm.end()
            sents = sentences(text, pa, pb)
            prev = ""
            for sa, sb in sents:
                nums = group_numbers(text, find_numbers(text, mtext, sa, sb))
                # live (uncommented) paragraph text before this sentence, for dataset / stream carry-over
                para = "".join(text[k] for k in range(pa, sa) if not cmask[k])
                for n in nums:
                    n.update(dict(section=heading_at(hs, sa), context=strip_tex(text[sa:sb]),
                                  clause=strip_tex(clause_of(text, sa, sb, n["start"])), prev_context=prev,
                                  para_before=strip_tex(para)[-800:],
                                  table="", table_caption="", table_row="", table_col="", table_group="", table_key=""))
                    items.append(n)
                live = "".join(text[k] for k in range(sa, sb) if not cmask[k]).strip()
                if live:
                    prev = strip_tex(live)
    out = []
    for n in sorted(items, key=lambda d: d["start"]):
        if n["start"] < doc_start:
            continue
        in_c = all(cmask[k] for k in range(n["start"], max(n["start"] + 1, n["end"])))
        in_r = any(rmask[k] for k in range(n["start"], max(n["start"] + 1, n["end"])))
        out.append(dict(
            item_id="L%d_%d" % (line_of(starts, n["start"]), n["start"] - starts[line_of(starts, n["start"]) - 1]),
            line=line_of(starts, n["start"]), col=n["start"] - starts[line_of(starts, n["start"]) - 1],
            kind=n["kind"], text=n["raw"].replace("\n", " "), value=n["value"], value2=n["value2"],
            decimals=n["dp"] if n["dp"] is not None else "", decimals2=n["dp2"], unit=n["unit"],
            range_labels="%s|%s" % n["labels"] if any(n["labels"]) else "",
            in_red="yes" if in_r else "no", in_comment="yes" if in_c else "no", section=n["section"],
            table=n["table"], table_key=n["table_key"], table_row=n["table_row"], table_col=n["table_col"],
            table_group=n["table_group"], clause=n["clause"], context=n["context"], prev_context=n["prev_context"],
            table_caption=n["table_caption"]))
    red = [(line_of(starts, a), line_of(starts, b), strip_tex(text[a:b])) for a, b in rspans]
    return out, red


FIELDS = ["item_id", "line", "col", "kind", "text", "value", "value2", "decimals", "decimals2", "unit", "range_labels",
          "in_red", "in_comment", "section", "table", "table_key", "table_row", "table_col", "table_group",
          "clause", "context", "prev_context", "table_caption", "para_before"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.tex)[0] + "_numbers.csv"
    items, red = extract(a.tex)
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(items)
    with open(os.path.splitext(out)[0] + "_redspans.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["line_start", "line_end", "text"])
        for r in red:
            w.writerow(r)
    live = [i for i in items if i["in_comment"] == "no"]
    print("%s: %d numeric items (%d live, %d commented), %d red spans -> %s" % (
        a.tex, len(items), len(live), len(items) - len(live), len(red), out))


if __name__ == "__main__":
    main()
