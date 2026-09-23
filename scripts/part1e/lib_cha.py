import os,re,glob
def par_text(path):
    lines=open(path,encoding="utf-8",errors="replace").read().split("\n")
    merged=[];cur=None
    for ln in lines:
        if ln[:1] in ("*","@","%"):
            if cur is not None: merged.append(cur)
            cur=ln
        elif cur is not None and (ln.startswith("\t") or ln.startswith(" ")):
            cur=cur+" "+ln.strip()
        else:
            if cur is not None: merged.append(cur); cur=None
    if cur is not None: merged.append(cur)
    out=[]
    for m in merged:
        if m.startswith("*PAR:"):
            t=m[5:]
            t=re.sub(r"\x15[^\x15]*\x15"," ",t)
            t=re.sub(r"\d+_\d+"," ",t)
            t=re.sub(r"\[[^\]]*\]"," ",t)
            t=re.sub(r"&[a-zA-Z=+:]*"," ",t)
            t=re.sub(r"\(([^)]*)\)",r"\1",t)
            t=re.sub(r"[^a-zA-Z' ]"," ",t)
            out.append(" ".join(t.lower().split()))
    return " ".join(out)
def meta(path):
    d={"age":"","sex":"","grp":"","mmse":"","pid":""}
    for ln in open(path,encoding="utf-8",errors="replace"):
        if ln.startswith("@ID:") and "|PAR|" in ln:
            p=ln.split("\t")[1].strip().split("|")
            d["age"]=p[3].rstrip(";");d["sex"]=p[4];d["grp"]=p[5];d["mmse"]=p[8]
        elif ln.startswith("@PID:"): d["pid"]=ln.split("\t")[1].strip()
    return d
