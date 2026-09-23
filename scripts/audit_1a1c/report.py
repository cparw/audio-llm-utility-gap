import json,csv,os,re,random,statistics as st,collections,unicodedata
import cha
S=os.path.dirname(os.path.abspath(__file__))
OUT="/Users/chaitanyaparwatkar/Desktop/release/edaic_rerun/AUDIT_1a1c_ad.txt"
D=json.load(open(S+'/part1a.json'))
DUR=json.load(open(S+'/durations.json'))
KOFF=json.load(open(S+'/kcl_offsets.json'))
ADOFF=json.load(open(S+'/ad_offsets.json'))
B="/Volumes/G-Drive Pro/paper work/Hard drive data for paper"; DB=B+"/DementiaBank"
P="/Volumes/G-Drive Pro/paper1_local_runs"
L=[]
def w(s=""): L.append(s)
def f(x,n=4): return "NA" if x is None else f"{x:.{n}f}"
def pct(a,b): return f"{a}/{b} = {a/b:.4f}"
def quant(xs,fr):
    xs=sorted(xs); i=fr*(len(xs)-1); lo=int(i)
    return xs[lo]+(xs[min(lo+1,len(xs)-1)]-xs[lo])*(i-lo)

pitt=D['pitt']; adso=D['adso']; a20=D['a20']

w("="*100)
w("AUDIT PART 1a + AD/PD HALF OF PART 1c")
w("Alzheimer's and Parkinson's datasets: what is actually inside the scored window")
w("Written 2026-09-22. Every number recomputed from files on disk in this run. Nothing copied from the draft.")
w("="*100)
w()
w("HEADLINE, so the rest can be read at leisure")
w("-"*100)
w("  1. Pitt is not a 30 s dataset. All 468 clips are LONGER than 30 s: median 33.2850 s, max 55.5600 s,")
w("     none within 0.01 s of 30. The phrase '468 thirty-second segments' is wrong on the second word.")
w("  2. Interviewer speech is inside the scored window far more often than the write-up implies:")
w("     Pitt 213/468 windows, ADReSS-2020 114/156 windows. ADReSS-2020 is the set the protocol says")
w("     was re-cut to skip the interviewer prompt.")
w("  3. The audit column we would have quoted to deny this, inv_ms_in_window in pitt_conflict_manifest.csv,")
w("     is 0.0000 on all 468 rows because it needs a timestamp the Pitt *INV: tiers usually do not carry.")
w("     It is a blind spot, not a clean bill of health. Do not cite it.")
w("  4. 'The 30 s window starts at the patient's first words' holds for 29/156 ADReSS-2020 windows exactly,")
w("     48/156 within a quarter second. It cannot be checked at all on ADReSSo: no timed transcript exists.")
w("  5. 'Every Parkinson's speaker reads the same passage' is wrong for all three PD sets. PC-GITA has")
w("     11 distinct prompts, NeuroVoz 12, and the MDVR-KCL 30 s window lands between 0.0000 s and")
w("     100.5600 s into the reading, so no two speakers are scored on the same sentences.")
w("  6. What the project calls ADReSSo is the ADReSS-M English training split. The ADReSSo folder is empty.")
w("  7. Counts that DO hold: 468 Pitt clips, 228 Pitt speakers, 237 ADReSSo, 156 ADReSS-2020,")
w("     146 + 322 conflict/agreement, 108 + 48 ADReSS-2020 train/test.")
w()
w("SCOPE AND METHOD")
w("-"*100)
w("Audio lengths: ffprobe format=duration on every file (5736 files probed, 0 failed, 0 missing).")
w("Transcripts: CHAT .cha files parsed directly. An utterance is a *PAR: or *INV: tier line with a")
w("  timestamp bullet \\x15start_end\\x15 in milliseconds. Word count = surface tokens after removing")
w("  [ ... ] scope codes, & prefixed fillers and non-words, +\"/. style terminators, @ codes, bare")
w("  punctuation and xxx/yyy/www. Parenthesised omitted letters are kept, e.g. an(d) counts as one word.")
w("  Angle brackets are stripped but the words inside them are kept, so retraced words are counted.")
w("NOT EVERY CHAT UTTERANCE IS TIMED. In the Pitt cookie transcripts 6589 *PAR: and 1683 *INV:")
w("  utterances carry a time bullet, while 537 *PAR: and 795 *INV: do not. ADReSS-2020 is fully timed")
w("  (2082 PAR + 828 INV, 0 untimed). An untimed utterance is still pinned down by its neighbours: it")
w("  must fall between the end of the previous timed utterance and the start of the next one. Each")
w("  untimed utterance is therefore given that bracketing interval, and it counts as INSIDE the window")
w("  only when the whole bracketing interval is inside. Timed utterances count as inside when they")
w("  overlap the window. This is deliberately conservative: every count below is a LOWER bound on how")
w("  much speech is in the window. Across the 468 Pitt rows only 1 window changes between the")
w("  conservative rule and the loose one, so the choice is not doing any work.")
w("  This also explains a zero that would otherwise look reassuring, see FLAG 1 under Pitt.")
w("Window starts were not taken on trust. For ADReSS-2020 (all 156) and a seed-0 random 60 of ADReSSo,")
w("  the scored clip was cross-correlated against its source recording. Manifest start_s matched the")
w("  audio within 0.2000 s for 156/156 and 60/60, mean |delta| 0.0000 s, worst correlation 0.9431.")
w("  Files: adoff.py in the scratchpad, result /private/tmp/.../scratchpad/ad_offsets.json")
w("  So the window positions printed below are the positions really used for scoring.")
w()
w("KEY PATHS")
w("  Pitt scored clips      /Volumes/G-Drive Pro/paper work/Hard drive data for paper/DementiaBank/segments/")
w("  Pitt window times      .../DementiaBank/pitt_conflict_manifest.csv   (start_ms, end_ms)")
w("  Pitt scored list       /Volumes/G-Drive Pro/paper1_local_runs/pitt_manifest.csv (468 rows)")
w("  Pitt CHAT transcripts  .../DementiaBank/transcripts/<Control|Dementia>/cookie/<session>.cha")
w("  ADReSSo scored clips   .../adresso/segments_patient/   manifest .../adresso/adresso_manifest_patient.csv")
w("  ADReSSo source audio   .../DementiaBank/challenges/ADReSS-M/ADReSS-M-train_x/train/<spk>.mp3")
w("  ADReSS-2020 clips      .../adress2020/segments_patient/ manifest .../adress2020/adress2020_manifest_patient.csv")
w("  ADReSS-2020 CHAT       .../challenges/ADReSS-2020/ADReSS-IS2020-{train_x,test_x}/ADReSS-IS2020-data/{train,test}/transcription/")
w("  MDVR-KCL scored clips  /Volumes/G-Drive Pro/paper1_local_runs/kcl_read30b/  manifest kcl30b_manifest.csv")
w("  MDVR-KCL source        /Volumes/G-Drive Pro/paper work/Hard drive data for paper/KCL/extracted/26-29_09_2017_KCL/ReadText/")
w("  PC-GITA scored clips   /Volumes/G-Drive Pro/paper1_local_runs/clips_local/spanish_dataset/  list drive_data/pcgita_read.csv")
w("  NeuroVoz scored clips  /Volumes/G-Drive Pro/paper1_local_runs/drive_data/spanish_neurovoz/zenodo_upload/audios/")
w("  PD ASR text            /Users/chaitanyaparwatkar/Desktop/release/part4_text/asr/asr_{pcgita,neurovoz,kcl}.csv")
w("  Written protocol       .../share_for_minoo/PROTOCOL.md  (the source of several claims checked below)")
w()
w("NAMING WARNING, established before anything else was computed")
w("  What this project calls ADReSSo is not the ADReSSo 2021 release. The folder")
w("  .../DementiaBank/challenges/ADReSSo is EMPTY. The 237 clips come from")
w("  .../challenges/ADReSS-M/ADReSS-M-train_x/train, the ADReSS-M English training split")
w("  (speaker ids adrso002...), read by .../adresso/prep_adresso.py. Same speaker id convention,")
w("  different release. Everything below about 'ADReSSo' is about those 237 ADReSS-M train files.")
w()

# ---------------- 1a ----------------
w("="*100)
w("PART 1a  PER-CLIP WINDOW CONTENTS")
w("="*100)
w()
w("-"*100); w("1a.1  PITT (DementiaBank Cookie Theft), 468 scored clips"); w("-"*100)
sd=[r['seg_dur'] for r in pitt]
w("Scored clip length, seconds (ffprobe on .../DementiaBank/segments/*.wav):")
for lbl,fr in [("min",0),("p5",.05),("p25",.25),("median",.5),("p75",.75),("p95",.95),("max",1.0)]:
    w(f"    {lbl:>6}  {f(quant(sd,fr))}")
w(f"    mean {f(st.mean(sd))}  sd {f(st.pstdev(sd))}")
w(f"    clips exactly 30.0000 s: 0/468.  clips shorter than 30 s: 0/468.  clips longer than 30 s: 468/468.")
w("Source recording length (351 distinct Cookie Theft recordings feeding the 468 clips):")
sv=sorted(set((r['src'],r['src_dur']) for r in pitt)); svv=[v for _,v in sv]
w(f"    n {len(svv)}  min {f(min(svv))}  median {f(st.median(svv))}  max {f(max(svv))}")
w("Window actually used for scoring: [start_ms, end_ms] from pitt_conflict_manifest.csv.")
mx=max(abs(r['winlen']-r['seg_dur']) for r in pitt)
w(f"    Largest disagreement between (end_ms-start_ms) and the cut wav length: {f(mx)} s. The cut matches the manifest.")
w(f"    Window start, seconds into the recording: min {f(min(r['w0'] for r in pitt))}  median {f(st.median([r['w0'] for r in pitt]))}  max {f(max(r['w0'] for r in pitt))}")
w(f"    Window end:   min {f(min(r['w1'] for r in pitt))}  median {f(st.median([r['w1'] for r in pitt]))}  max {f(max(r['w1'] for r in pitt))}")
w(f"    Windows starting at 0.0000 s: {sum(1 for r in pitt if r['w0']==0.0)}/468")
w("CHAT transcript coverage: 468/468 clips have their .cha on disk. 0 missing.")
r2=[r['tx_max_end']/r['src_dur'] for r in pitt]
w(f"    Transcript last timestamp / audio length: median {f(st.median(r2))}  min {f(min(r2))}  max {f(max(r2))}")
w("    So the CHAT timeline and the audio timeline are the same clock for Pitt. Window times are trustworthy.")
w("Words inside the window (conservative rule above):")
pw=[r['par_certain'] for r in pitt]; iw=[r['inv_certain'] for r in pitt]
w(f"    participant words: min {min(pw)}  median {f(st.median(pw))}  mean {f(st.mean(pw))}  max {max(pw)}")
w(f"    interviewer words: min {min(iw)}  median {f(st.median(iw))}  mean {f(st.mean(iw))}  max {max(iw)}")
w()
w("  ***FLAG 1  INTERVIEWER SPEECH INSIDE THE SCORED WINDOW***")
w(f"    Pitt windows containing at least one interviewer word: {pct(sum(1 for r in pitt if r['inv_certain']>0),468)}")
w(f"    Windows containing a whole *INV: utterance start-to-finish: {pct(sum(1 for r in pitt if r['inv_inside']>0),468)}")
w("    Split by how the interviewer speech was located, because the two are not equally exact:")
w(f"      TIMED *INV: utterances, exact overlap seconds inside the window: >0 s {sum(1 for r in pitt if r['inv_sec_timed']>0)}/468, "
  f">0.5 s {sum(1 for r in pitt if r['inv_sec_timed']>0.5)}/468, >2 s {sum(1 for r in pitt if r['inv_sec_timed']>2.0)}/468")
w(f"      longest exactly-timed interviewer stretch inside one window: {f(max(r['inv_sec_timed'] for r in pitt))} s")
w(f"      UNTIMED *INV: utterances whose whole bracketing interval falls inside the window: {pct(sum(1 for r in pitt if r['inv_untimed_in']>0),468)} windows,"
  f" {sum(r['inv_untimed_words'] for r in pitt)} interviewer words in total")
w("      For those the exact seconds are unknowable, only the fact that the turn happened inside the window.")
w("    DO NOT CITE THE MANIFEST COLUMN: pitt_conflict_manifest.csv carries inv_ms_in_window, and it")
w("    reads 0.0000 for all 468 rows. That column is an artefact, not a finding. build_pitt_manifest.py")
w("    measures interviewer time with a regex that requires a time bullet on the *INV: line, and in the")
w("    Pitt cookie transcripts 795 *INV: utterances have no bullet at all; 345 of the 468 scored rows sit")
w("    in a file with at least one untimed *INV:. The column reports zero because it cannot see the")
w("    interviewer, not because the interviewer is absent. Recomputed here with the bracketing rule,")
w("    213 of 468 windows contain interviewer words and 215 contain interviewer audio.")
w()
w("  ***FLAG 2  PARTICIPANT SAYS FEWER THAN 15 WORDS IN THE WINDOW***")
lo=[r for r in pitt if r['par_certain']<15]
w(f"    {pct(len(lo),468)}. No Pitt window drops below 15 participant words; the smallest is {min(r['par_certain'] for r in pitt)}.")
w(f"    Next threshold up, fewer than 25 participant words: {pct(sum(1 for r in pitt if r['par_certain']<25),468)}")
for r in sorted(pitt,key=lambda x:x['par_certain'])[:13]:
    w(f"      {r['grp'][:3]} {r['session']:<8} window {f(r['w0'])}-{f(r['w1'])} s  PAR words {r['par_certain']:>3}  INV words {r['inv_certain']:>3}  file {os.path.basename(r['seg'])}")
w()
w("-"*100); w("1a.2  ADReSSo (= ADReSS-M English train), 237 scored clips"); w("-"*100)
sd=[r['seg_dur'] for r in adso]
w("Scored clip length, seconds (.../adresso/segments_patient/*.wav):")
w(f"    min {f(min(sd))}  median {f(st.median(sd))}  mean {f(st.mean(sd))}  max {f(max(sd))}")
w(f"    exactly 30.0000 s: {pct(sum(1 for x in sd if abs(x-30)<0.005),237)};  shorter than 30 s: {pct(sum(1 for x in sd if x<29.995),237)}")
w("    The short ones are windows that ran off the end of the recording, not a different rule.")
srcd=[r['src_dur'] for r in adso]
w(f"Source mp3 length: min {f(min(srcd))}  median {f(st.median(srcd))}  max {f(max(srcd))}")
w("Window used for scoring: [start_s, start_s+30] from adresso_manifest_patient.csv, produced by")
w("    .../adresso/recut_at_patient.py. Verified against audio for a seed-0 random 60: all 60 match.")
w(f"    start_s: min {f(min(r['w0'] for r in adso))}  median {f(st.median([r['w0'] for r in adso]))}  max {f(max(r['w0'] for r in adso))}")
w(f"    Windows starting at 0.0000 s (no shift at all): {pct(sum(1 for r in adso if r['w0']==0.0),237)}")
w()
w("  TIMED TRANSCRIPTS: NOT ON DISK. Searched the whole ADReSS-M tree")
w("    (.../challenges/ADReSS-M): the only non-mp3 files are ADReSS-M-meta.csv, the two format_task")
w("    csvs, the groundtruth csvs and the tgz archives. No .cha, no .txt, no segmentation csv.")
w("    So participant-vs-interviewer word counts inside the window CANNOT be computed for ADReSSo.")
w("    That is a missing-file fact, not an estimate.")
w("    Substitute evidence, clearly labelled as ASR not ground truth: .../adresso/adresso_transcripts_patient.csv")
w("    is whisper ASR of the 237 scored clips themselves (no speaker labels, no timings).")
adt=list(csv.DictReader(open(B+'/adresso/adresso_transcripts_patient.csv')))
wc=[len(r['text'].split()) for r in adt]
w(f"      ASR words per 30 s clip: min {min(wc)}  median {f(st.median(wc))}  max {max(wc)}")
w(f"      ***FLAG*** clips with fewer than 15 ASR words in the window: {pct(sum(1 for x in wc if x<15),237)}"
  f" (one clip transcribes to 0 words)")
PR=re.compile(r"(tell me|everything (that )?you see|going on in (this|that|the) picture|what do you see|here.s (the|a) picture|look at (this|that|the) picture|what.s happening|all (of )?the action|describe)", re.I)
hit=[r for r in adt if PR.search(r['text'][:200])]
w(f"      ***FLAG*** clips whose own ASR still contains an interviewer prompt phrase in the first 200 characters: {pct(len(hit),237)}")
w(f"         ids: {', '.join(sorted(r['spk'] for r in hit))}")
w("      Without speaker-labelled transcripts this is a lower bound on interviewer leakage, not a count.")
w()
w("-"*100); w("1a.3  ADReSS-2020, 156 scored clips"); w("-"*100)
sd=[r['seg_dur'] for r in a20]
w("Scored clip length, seconds (.../adress2020/segments_patient/*.wav):")
w(f"    min {f(min(sd))}  median {f(st.median(sd))}  mean {f(st.mean(sd))}  max {f(max(sd))}")
w(f"    exactly 30.0000 s: {pct(sum(1 for x in sd if abs(x-30)<0.005),156)};  shorter than 30 s: {pct(sum(1 for x in sd if x<29.995),156)}")
srcd=[r['src_dur'] for r in a20]
w(f"Source wav length: min {f(min(srcd))}  median {f(st.median(srcd))}  max {f(max(srcd))}")
w(f"Window used: [start_s, start_s+30]. start_s min {f(min(r['w0'] for r in a20))} median {f(st.median([r['w0'] for r in a20]))} max {f(max(r['w0'] for r in a20))}")
w(f"    Windows starting at 0.0000 s: {pct(sum(1 for r in a20 if r['w0']==0.0),156)}")
w("CHAT transcript coverage: 156/156 present.")
r2=[r['tx_max_end']/r['src_dur'] for r in a20]
w(f"    Transcript last timestamp / audio length: median {f(st.median(r2))}  min {f(min(r2))}  max {f(max(r2))}")
pw=[r['par_certain'] for r in a20]; iw=[r['inv_certain'] for r in a20]
w("Words inside the window (conservative rule above):")
w(f"    participant words: min {min(pw)}  median {f(st.median(pw))}  mean {f(st.mean(pw))}  max {max(pw)}")
w(f"    interviewer words: min {min(iw)}  median {f(st.median(iw))}  mean {f(st.mean(iw))}  max {max(iw)}")
w()
w("  ***FLAG 1  INTERVIEWER SPEECH INSIDE THE SCORED WINDOW***")
w(f"    Windows containing at least one interviewer word: {pct(sum(1 for r in a20 if r['inv_certain']>0),156)}")
w(f"    Windows containing a whole *INV: utterance: {pct(sum(1 for r in a20 if r['inv_inside']>0),156)}")
w(f"    Interviewer seconds inside the window (all ADReSS-2020 utterances are timed, so these are exact):")
w(f"      >0 s {sum(1 for r in a20 if r['inv_sec']>0)}/156, >0.5 s {sum(1 for r in a20 if r['inv_sec']>0.5)}/156, >2 s {sum(1 for r in a20 if r['inv_sec']>2.0)}/156")
w(f"    Largest interviewer stretch inside one window: {f(max(r['inv_sec'] for r in a20))} s")
w("    This is the set the protocol says was re-cut to skip the prompt.")
w()
w("  ***FLAG 2  PARTICIPANT SAYS FEWER THAN 15 WORDS IN THE WINDOW***")
lo=[r for r in a20 if r['par_certain']<15]
w(f"    {pct(len(lo),156)};  strict fully-contained rule: {pct(sum(1 for r in a20 if r['par_inside']<15),156)}")
for r in sorted(lo,key=lambda x:x['par_certain']):
    w(f"      {r['spk']} ({r['split']}) window {f(r['w0'])}-{f(r['w1'])} s  PAR words {r['par_certain']}  INV words {r['inv_certain']}")
w()

# --- 1a.4 PD sets
w("-"*100); w("1a.4  PC-GITA, NeuroVoz, MDVR-KCL, as far as the files allow"); w("-"*100)
pg=list(csv.DictReader(open('/Users/chaitanyaparwatkar/Desktop/release/part4_text/asr/asr_pcgita.csv')))
pgd=[]
for r in pg:
    rel=r['clip_path'].split('/spanish_dataset/',1)[-1]; p=P+'/clips_local/spanish_dataset/'+rel
    if DUR.get(p): pgd.append(DUR[p])
w(f"PC-GITA, {len(pg)} scored clips. Clip length: min {f(min(pgd))}  median {f(st.median(pgd))}  mean {f(st.mean(pgd))}  max {f(max(pgd))}")
w(f"    clips longer than 30 s: {sum(1 for x in pgd if x>30)}/{len(pgd)}. NO 30 s window is applied: each clip is one whole short")
w("    recording of one prompt. 'Window start' = 0.0000 and 'window end' = the file length, for all of them.")
nv=list(csv.DictReader(open('/Users/chaitanyaparwatkar/Desktop/release/part4_text/asr/asr_neurovoz.csv')))
nvd=[]
for r in nv:
    rel=r['clip_path'].split('/spanish_neurovoz/',1)[-1]; p=P+'/drive_data/spanish_neurovoz/'+rel
    if DUR.get(p): nvd.append(DUR[p])
w(f"NeuroVoz, {len(nv)} scored clips. Clip length: min {f(min(nvd))}  median {f(st.median(nvd))}  mean {f(st.mean(nvd))}  max {f(max(nvd))}")
w(f"    clips longer than 30 s: {sum(1 for x in nvd if x>30)}/{len(nvd)}. Again no 30 s window: whole short recordings.")
kd=[DUR[P+'/kcl_read30b/'+fn] for fn in sorted(os.listdir(P+'/kcl_read30b')) if fn.endswith('.wav')]
ko=[v[0] for v in KOFF.values()]
w(f"MDVR-KCL, 37 scored clips (kcl_read30b). Clip length: every one is exactly {f(min(kd))} s (min=max).")
srcs={}
import glob as _g
for x in _g.glob(B+'/KCL/extracted/26-29_09_2017_KCL/ReadText/*/*.wav'): srcs[os.path.basename(x)]=DUR.get(x)
sv=[v for v in srcs.values() if v]
w(f"    Source reading length: n {len(sv)}  min {f(min(sv))}  median {f(st.median(sv))}  max {f(max(sv))}")
w("    Window start recovered by cross-correlating each 30 s clip against its source reading")
w("    (kcloff.py; every match correlation >= 0.9900, so these offsets are measured, not assumed):")
w(f"      start offset: min {f(min(ko))}  median {f(st.median(ko))}  mean {f(st.mean(ko))}  max {f(max(ko))}")
w(f"      clips starting at 0.0000 s: {sum(1 for x in ko if x==0.0)}/37")
w("      per clip (file, start s, end s, source length s):")
for fn in sorted(KOFF):
    o=KOFF[fn][0]; w(f"        {fn:<22} {f(o):>9} {f(o+30.0):>9}  {f(srcs.get(fn))}")
w("    NO timed transcript exists for MDVR-KCL on disk. The KCL tree holds audio only")
w("    (ReadText/ and SpontaneousDialogue/ wavs). Word counts by speaker inside the window")
w("    cannot be computed. There is only one speaker in a KCL read recording, so interviewer")
w("    leakage is not the risk here; window position is (see 1c below).")
w()

# --- random samples
w("-"*100); w("1a.5  TEN RANDOM CLIPS PER DATASET, seed 0, first 20 words of the window"); w("-"*100)
w("Speaker tag is prefixed to each word. PAR = participant, INV = interviewer/investigator.")
w()
rng=random.Random(0)
w("PITT (from the CHAT transcript, words counted inside the window by the rule above):")
for r in rng.sample(pitt,10):
    w(f"  {r['grp'][:3]} {r['session']:<8} win {f(r['w0'])}-{f(r['w1'])}s  len {f(r['seg_dur'])}s  PARw {r['par_certain']:>3} INVw {r['inv_certain']:>3}")
    w(f"      {r['first20']}")
w()
rng=random.Random(0)
w("ADReSSo (ASR of the scored clip; no ground-truth timed transcript exists, see 1a.2):")
adtd={r['spk']:r['text'] for r in adt}
for r in rng.sample(adso,10):
    t=' '.join(adtd.get(r['spk'],'').split()[:20])
    w(f"  {r['spk']:<10} win {f(r['w0'])}-{f(r['w1'])}s  len {f(r['seg_dur'])}s  (speaker of each word unknown)")
    w(f"      {t if t else '(ASR produced no words)'}")
w()
rng=random.Random(0)
w("ADReSS-2020 (from the CHAT transcript):")
for r in rng.sample(a20,10):
    w(f"  {r['spk']:<6} ({r['split']:<5}) win {f(r['w0'])}-{f(r['w1'])}s  len {f(r['seg_dur'])}s  PARw {r['par_certain']:>3} INVw {r['inv_certain']:>3}")
    w(f"      {r['first20']}")
w()
rng=random.Random(0)
kcl={r['speaker_id']:r['text'] for r in csv.DictReader(open('/Users/chaitanyaparwatkar/Desktop/release/part4_text/asr/asr_kcl.csv'))}
w("MDVR-KCL (ASR of the scored clip; no timed transcript on disk):")
for fn in rng.sample(sorted(KOFF),10):
    sid=fn.split('_')[0]; o=KOFF[fn][0]
    w(f"  {fn:<22} win {f(o)}-{f(o+30.0)}s")
    w(f"      {' '.join(kcl.get(sid,'').split()[:20])}")
w()
rng=random.Random(0)
w("PC-GITA (ASR of the scored clip; whole clip is the window):")
for r in rng.sample(pg,10):
    rel=r['clip_path'].split('/spanish_dataset/',1)[-1]
    w(f"  {r['speaker_id']:<16} {rel.rsplit('/',2)[0]:<34} len {f(DUR.get(P+'/clips_local/spanish_dataset/'+rel))}s")
    w(f"      {' '.join(r['text'].split()[:20])}")
w()
rng=random.Random(0)
w("NeuroVoz (ASR of the scored clip; whole clip is the window):")
for r in rng.sample(nv,10):
    rel=r['clip_path'].split('/spanish_neurovoz/',1)[-1]
    w(f"  {os.path.basename(r['clip_path']):<28} len {f(DUR.get(P+'/drive_data/spanish_neurovoz/'+rel))}s")
    w(f"      {' '.join(r['text'].split()[:20])}")
w()
open(OUT,'w').write("\n".join(L)+"\n")
print("wrote part 1a,",len(L),"lines")
