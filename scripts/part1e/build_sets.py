import glob,os,json
HD="/Volumes/G-Drive Pro/paper work/Hard drive data for paper"
PL="/Volumes/G-Drive Pro/paper1_local_runs"
sets={
 "pitt_segments":            sorted(glob.glob(f"{HD}/DementiaBank/segments/*.wav")),
 "pitt_cookie_source_mp3":   sorted(glob.glob(f"{HD}/DementiaBank/Pitt/Control/cookie/*.mp3"))+sorted(glob.glob(f"{HD}/DementiaBank/Pitt/Dementia/cookie/*.mp3")),
 "adress2020_patient30s":    sorted(glob.glob(f"{HD}/adress2020/segments_patient/*.wav")),
 "adress2020_seg30s":        sorted(glob.glob(f"{HD}/adress2020/segments/*.wav")),
 "adress2020_fullwave":      sorted(glob.glob(f"{HD}/DementiaBank/challenges/ADReSS-2020/*/ADReSS-IS2020-data/*/Full_wave_enhanced_audio/*.wav"))+sorted(glob.glob(f"{HD}/DementiaBank/challenges/ADReSS-2020/*/ADReSS-IS2020-data/*/Full_wave_enhanced_audio/*/*.wav")),
 "adresso_patient30s":       sorted(glob.glob(f"{HD}/adresso/segments_patient/*.wav")),
 "adresso_seg30s":           sorted(glob.glob(f"{HD}/adresso/segments/*.wav")),
 "adresso_source_mp3":       sorted(glob.glob(f"{HD}/DementiaBank/challenges/ADReSS-M/ADReSS-M-train_x/train/*.mp3")),
 "kcl_read_clips":           sorted(glob.glob(f"{HD}/share_for_minoo/kcl_read_clips/*.wav")),
 "kcl_read30b":              sorted(glob.glob(f"{PL}/kcl_read30b/*.wav")),
 "pcgita_control":           sorted(glob.glob(f"{PL}/drive_data/pcgita_control/*.wav")),
 "neurovoz":                 sorted(glob.glob(f"{PL}/drive_data/spanish_neurovoz/zenodo_upload/audios/*.wav")),
 "neurovoz_silencemode1":    sorted(glob.glob(f"{PL}/drive_data/spanish_neurovoz/zenodo_upload_silencemode_1/audios/*.wav")),
 "edaic_dcaps_proc":         sorted(glob.glob(f"{PL}/drive_data/dcaps_proc/*.wav")),
}
for k,v in sets.items(): print(f"{k:28s} {len(v)}")
json.dump(sets,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"sets.json"),"w"))
