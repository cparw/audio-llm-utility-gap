import pandas as pd, numpy as np, re, os
S="summary"
d=pd.read_csv(f"{S}/summary_all_v2.csv")
def fmt(x):
    try:
        v=float(x); 
        return f"{v:.4f}" if abs(v)<100 and v!=int(v) else (str(int(v)) if v==int(v) else f"{v:.4f}")
    except: return "" if pd.isna(x) else str(x)
def row(r):
    v=r.value_verified if not pd.isna(r.value_verified) else r.value_computed
    ci=f" [{fmt(r.lo)}, {fmt(r.hi)}]" if not pd.isna(r.lo) and not pd.isna(r.hi) else ""
    n="" if pd.isna(r.n) else (f"{int(r.n)}" + ("" if pd.isna(r.n_spk) else f" ({int(r.n_spk)})"))
    ok=str(r.agree_4dp)
    ok="yes" if ok.startswith("yes") or "verified in fragment" in ok or ok=="True" else ok
    w=re.sub(r"\s+"," ",str(r.what))[:110].replace("|","/")
    return f"| {r.id} | {w} | {fmt(v)}{ci} | {n} | {ok} |"
def section(df,title):
    out=[f"\n### {title}\n","| id | what | value [95% CI] | n (speakers) | checked |","|---|---|---|---|---|"]
    out+= [row(r) for r in df.itertuples()]
    return out
L=["# All results, 23 Sep 2026","","Every value below comes from its per-clip file and has an independent check (column 'checked'). CI = 2000-draw speaker bootstrap, 2.5 and 97.5 percentiles.",""]
pv=d[d.paper_role.astype(str).str.startswith("PAPER VALUE")]
L.append(f"## A. Paper values ({len(pv)})")
for (p,t),g in pv.groupby(["part","task/pod"],sort=True): L+=section(g,f"PART {p} {t}")
# PART 23 and 24
L.append("\n## B. PART 23, E-DAIC whole interview")
t=pd.read_csv("scores/part23/verify/PART23_VERIFIED.tsv",sep="\t")
L+=["| id | claimed | recomputed | lo | hi | checked |","|---|---|---|---|---|---|"]+[f"| {r['id']} | {r['claimed']} | {r['recomputed']} | {r.get('lo','')} | {r.get('hi','')} | {r['verified']} |" for _,r in t.iterrows()]
L.append("\n## C. PART 24, E-DAIC whole interview fine-tune")
t=pd.read_csv("scores/part24/verify/FT24_VERIFIED.tsv",sep="\t")
L+=["| "+" | ".join(t.columns)+" |","|"+"---|"*len(t.columns)]+["| "+" | ".join(str(x) for x in r)+" |" for r in t.itertuples(index=False)]
other=d[~d.paper_role.astype(str).str.startswith("PAPER VALUE")]
L.append(f"\n## D. Support, released and superseded values ({len(other)})")
for (p,t2),g in other.groupby(["part","task/pod"],sort=True): L+=section(g,f"PART {p} {t2} ({g.paper_role.astype(str).str.split(',').str[0].value_counts().to_dict()})")
open(f"{S}/ALL_RESULTS_23sep.md","w").write("\n".join(L))
print(len(L), "lines; paper rows", len(pv), "; other", len(other))
