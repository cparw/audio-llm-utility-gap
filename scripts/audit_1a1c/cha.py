import re
BUL="\x15"
def clean_words(t):
    t=t.replace("\n"," ").replace("\t"," ")
    t=re.sub(BUL+r"\d+_\d+"+BUL," ",t)
    t=re.sub(r"\[[^\]]*\]"," ",t)
    t=re.sub(r"[<>]"," ",t)
    t=re.sub(r"&[-+=]?\S+"," ",t)
    t=re.sub(r"\+[\"/.,!?]+"," ",t)
    t=t.replace("(","").replace(")","")
    t=re.sub(r"@\S*"," ",t)
    t=re.sub(r"[‘’“”\"]"," ",t)
    toks=[]
    for w in t.split():
        w=w.strip(".,!?;:+…")
        if not w: continue
        if w.lower() in ("xxx","yyy","www"): continue
        if not re.search(r"[A-Za-zÀ-ɏ]",w): continue
        toks.append(w)
    return toks
HDR=re.compile(r"^\*([A-Za-z]{3}\d?):\t(.*)$")
def parse_cha(path):
    lines=open(path,encoding="utf-8",errors="ignore").read().split("\n")
    out=[]; n_no_bullet=0; i=0
    while i<len(lines):
        m=HDR.match(lines[i])
        if not m: i+=1; continue
        who=m.group(1); body=m.group(2); i+=1
        while i<len(lines) and lines[i].startswith("\t"):
            body+=" "+lines[i][1:]; i+=1
        b=re.search(BUL+r"(\d+)_(\d+)"+BUL, body)
        if not b: n_no_bullet+=1; continue
        out.append(dict(who=who,start_ms=int(b.group(1)),end_ms=int(b.group(2)),words=clean_words(body)))
    parse_cha.last_no_bullet=n_no_bullet
    return out

HDR2=HDR
def parse_cha_all(path):
    """Every *TIER: utterance in order, timed or not."""
    lines=open(path,encoding="utf-8",errors="ignore").read().split("\n")
    out=[]; i=0
    while i<len(lines):
        m=HDR2.match(lines[i])
        if not m: i+=1; continue
        who=m.group(1); body=m.group(2); i+=1
        while i<len(lines) and lines[i].startswith("\t"):
            body+=" "+lines[i][1:]; i+=1
        b=re.search(BUL+r"(\d+)_(\d+)"+BUL, body)
        out.append(dict(who=who,
                        start_ms=int(b.group(1)) if b else None,
                        end_ms=int(b.group(2)) if b else None,
                        timed=bool(b), words=clean_words(body)))
    return out

def bound(utts, file_end_ms):
    """Give every utterance an interval [lo,hi] in ms it must lie within."""
    n=len(utts)
    prev=[0]*n; nxt=[file_end_ms]*n
    last=0
    for i,u in enumerate(utts):
        prev[i]=last
        if u["timed"]: last=u["end_ms"]
    nx=file_end_ms
    for i in range(n-1,-1,-1):
        nxt[i]=nx
        if utts[i]["timed"]: nx=utts[i]["start_ms"]
    for i,u in enumerate(utts):
        if u["timed"]: u["lo"],u["hi"]=u["start_ms"],u["end_ms"]
        else: u["lo"],u["hi"]=prev[i],max(nxt[i],prev[i])
    return utts
