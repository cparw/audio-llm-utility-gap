import pandas as pd, numpy as np, datetime
B='/Users/chaitanyaparwatkar/Desktop/release/edaic_rerun/confounds/'
OUT='/Users/chaitanyaparwatkar/Desktop/release/edaic_rerun/AUDIT_1d_confounds.txt'
r=pd.read_csv(B+'confound_auc.csv')
d=pd.read_csv(B+'clip_features.csv'); f=d[d.err.fillna('')=='']
NAMES={'pitt':'Pitt (DementiaBank)','adresso':'ADReSSo','adress2020':'ADReSS-2020','pcgita':'PC-GITA','neurovoz':'NeuroVoz','kcl':'MDVR-KCL','edaic':'E-DAIC'}
ORDER=['pitt','adresso','adress2020','pcgita','neurovoz','kcl','edaic']
FEATS=['noise_floor_db','snr_db','quiet_centroid_hz','quiet_rolloff85_hz','quiet_tilt_db_per_khz','dur_s','loudness_db','sample_rate']
SRC={
 'pitt':'/Volumes/G-Drive Pro/paper1_local_runs/pitt_manifest.csv (468 rows), audio base remapped to /Volumes/G-Drive Pro/paper work/Hard drive data for paper/DementiaBank/segments/',
 'adresso':'/Volumes/G-Drive Pro/paper1_local_runs/adresso_manifest.csv (237 rows), audio /Volumes/G-Drive Pro/paper work/Hard drive data for paper/adresso/segments_patient/',
 'adress2020':'/Volumes/G-Drive Pro/paper1_local_runs/adress2020_manifest.csv (156 rows), audio /Volumes/G-Drive Pro/paper work/Hard drive data for paper/adress2020/segments_patient/',
 'pcgita':'/Volumes/G-Drive Pro/paper1_local_runs/drive_data/pcgita_read.csv (1117 rows, 17 "las que sobraron" rows dropped to match the Table 1 n of 1100), audio read straight out of the archive /Volumes/G-Drive Pro/paper1_local_runs/drive_data/PC-GITA/OneDrive_1_12-11-2023.zip (no extraction, the Mac volume has only 7.0 GiB free)',
 'neurovoz':'/Volumes/G-Drive Pro/paper1_local_runs/drive_data/neurovoz.csv filtered to task_type=read (1270 rows, matches the Table 1 n), audio /Volumes/G-Drive Pro/paper1_local_runs/drive_data/spanish_neurovoz/zenodo_upload/audios/',
 'kcl':'/Volumes/G-Drive Pro/paper1_local_runs/kcl_manifest.csv (37 rows), audio /Volumes/G-Drive Pro/paper work/Hard drive data for paper/share_for_minoo/kcl_read_clips/ (full read recordings, not the 30 s recuts)',
 'edaic':'/Volumes/G-Drive Pro/paper1_local_runs/drive_data/dcaps.csv (275 rows), audio /Volumes/G-Drive Pro/paper1_local_runs/drive_data/dcaps_proc/<pid>.wav, labels cross checked 275/275 against /Users/chaitanyaparwatkar/Desktop/af2_run/speech-health/results/health/paradox/daic_segments_scored.csv',
}
L=[]
def w(s=''): L.append(s)
RULE='='*100
w('PART 1d  RECORDING CONFOUNDS')
w('Written '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M')+' by the Part 1d audit run.')
w(RULE)
w('QUESTION')
w('  Can a one number recording property of a clip, measured on the quiet frames only, predict the')
w('  diagnosis label at speaker level? If yes, a probe could be reading the recording setup and not')
w('  the speech. This is the check Paper 2 runs.')
w()
w('WHAT WAS RUN')
w('  Script     : '+B+'extract_features.py  (feature extraction, 10 processes)')
w('               '+B+'analyze.py           (speaker level AUC + permutation test)')
w('               '+B+'build_manifest.py    (clip list)')
w('  Outputs    : '+B+'clip_features.csv    (per clip features, 2373 rows, 0 errors)')
w('               '+B+'clip_manifest.csv    (clip list actually used)')
w('               '+B+'confound_auc.csv     (this table as data)')
w('  Python     : /usr/local/bin/python3 3.10.2, librosa 0.11.0, soundfile 0.13.1, numpy 2.2.6, pandas 2.3.3.')
w('               The default python3 on PATH (/opt/homebrew/bin/python3, 3.14.6) has pandas but NOT librosa')
w('               or soundfile, so the run used /usr/local/bin/python3 instead.')
w('  Audio load : native sample rate, no resampling, channels averaged to mono.')
w()
w('FEATURE DEFINITIONS  (frame 2048 samples, hop 512, at the native rate of each file)')
w('  Frame RMS is computed from the STFT magnitude. "Quietest fifth" is the bottom 20 percent of frames')
w('  ranked by RMS, rounded to at least 1 frame. "Loudest fifth" is the top 20 percent.')
w('    noise_floor_db        20*log10(root mean square of the RMS over the quietest fifth)')
w('    snr_db                20*log10(loudest fifth RMS / quietest fifth RMS)')
w('    quiet_centroid_hz     mean spectral centroid over the quietest fifth frames only')
w('    quiet_rolloff85_hz    mean 85 percent spectral roll off over the quietest fifth frames only')
w('    quiet_tilt_db_per_khz slope of a straight line fit to 20*log10(mean magnitude) against frequency')
w('                          in kHz, over the quietest fifth frames, bins above 50 Hz only. More negative')
w('                          means the quiet spectrum falls off faster with frequency.')
w('    dur_s                 whole file duration in seconds')
w('    loudness_db           20*log10(overall RMS of the whole file)')
w('    sample_rate           native sample rate in Hz')
w()
w('AUC PROTOCOL')
w('  Every feature is averaged within speaker first, then one AUC is taken across speakers against the')
w('  diagnosis label. AUC is the rank statistic')
w('    r = pd.Series(s).rank().values ; n1=(y==1).sum() ; n0=(y==0).sum()')
w('    auc = (r[y==1].sum() - n1*(n1+1)/2) / (n1*n0)')
w('  sklearn is not installed in this environment, so no sklearn call was made anywhere.')
w('  Both directions are printed (auc and 1-auc). strength = max(auc, 1-auc), because the sign of a')
w('  recording feature is arbitrary. FLAG is set when strength > 0.65.')
w('  perm_p is a two sided permutation p value on |auc - 0.5|, 20000 label shuffles, numpy seed 0.')
w('  It is in '+B+'confound_auc.csv, and quoted below for the flagged rows.')
w()
w('SAMPLING')
w('  Datasets over 600 clips were sampled down to 600 with pandas .sample(600, random_state=0):')
w('    PC-GITA   1100 clips sampled to 600 (100 of 100 speakers still present)')
w('    NeuroVoz  1270 clips sampled to 600 (107 of 107 speakers still present)')
w('  Every other dataset was used whole. Speaker level averaging happens after the sampling.')
w()
w('SOURCES PER DATASET')
for ds in ORDER: w('  '+NAMES[ds]+': '+SRC[ds])
w()
w(RULE)
w('PER DATASET TABLES')
w(RULE)
for ds in ORDER:
    g=r[r.dataset==ds].set_index('feature').loc[FEATS].reset_index()
    fg=f[f.dataset==ds]
    srs=', '.join(str(int(s)) for s in sorted(fg.sample_rate.unique()))
    w()
    w(NAMES[ds])
    w('  clips used '+str(len(fg))+', speakers '+str(int(g.n_speakers.iloc[0]))+
      ', positive speakers '+str(int(g.n_pos.iloc[0]))+', negative speakers '+str(int(g.n_neg.iloc[0]))+
      ', sample rates present '+srs+' Hz')
    w('  '+f'{"feature":<23}{"auc":>9}{"1-auc":>9}{"strength":>10}{"perm_p":>9}{"mean_pos":>13}{"mean_neg":>13}  FLAG')
    for _,x in g.iterrows():
        if np.isnan(x.auc):
            w('  '+f'{x.feature:<23}{"constant":>9}{"":>9}{"":>10}{"":>9}{x.mean_pos:>13.4f}{x.mean_neg:>13.4f}  no variance, AUC undefined')
        else:
            w('  '+f'{x.feature:<23}{x.auc:>9.4f}{x.auc_rev:>9.4f}{x.strength:>10.4f}{x.perm_p_two_sided:>9.4f}{x.mean_pos:>13.4f}{x.mean_neg:>13.4f}  {"FLAG" if x.strength>0.65 else ""}')
w()
w(RULE)
w('FLAGGED, RANKED BY CONFOUND STRENGTH  (strength > 0.65)')
w(RULE)
fl=r[r.FLAG=='FLAG'].sort_values('strength',ascending=False)
w(f'{"rank":<6}{"dataset":<12}{"feature":<23}{"strength":>10}{"auc":>9}{"perm_p":>9}{"n_spk":>7}{"mean_pos":>13}{"mean_neg":>13}')
for i,(_,x) in enumerate(fl.iterrows(),1):
    w(f'{i:<6}{NAMES[x.dataset]:<12}{x.feature:<23}{x.strength:>10.4f}{x.auc:>9.4f}{x.perm_p_two_sided:>9.4f}{int(x.n_speakers):>7}{x.mean_pos:>13.4f}{x.mean_neg:>13.4f}')
w()
w('Datasets with no flagged feature: Pitt, ADReSSo, ADReSS-2020, E-DAIC.')
w('Highest unflagged value per clean dataset:')
for ds in ['pitt','adresso','adress2020','edaic']:
    g=r[(r.dataset==ds)&r.strength.notna()].sort_values('strength',ascending=False).iloc[0]
    w(f'  {NAMES[ds]:<20} {g.feature:<23} strength {g.strength:.4f}  perm_p {g.perm_p_two_sided:.4f}')
open(OUT,'w').write('\n'.join(L)+'\n')
print(OUT,'written', len(L),'lines')
