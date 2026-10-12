# page5 check

PDF checked: the compiled final paper, final.pdf (not in this repository)
SHA256: a7d320b2801a2f4355ed89fe1e419a87979bf9cda089a90a65390b999a236132 (196592 bytes, pdfTeX, created Wed Sep 23 16:30:32 2026 PDT, letter 612 x 792 pt)
Tools: poppler 26.04.0 (pdfinfo, pdftotext -tsv, pdftoppm, pdfimages, pdffonts), /usr/local/bin/python3 with the independent script check_page5.py. Nothing imported from other checks.

## Verdicts

1. Page count is 5: YES. pdfinfo says Pages: 5.
2. Page 5 holds only references: YES.
   - 88 text lines on page 5, every one is part of a reference entry.
   - Entries on page 5 are [2] through [30], contiguous, no gaps (29 entries). [1] sits on page 4.
   - No text before [2] in the left column. Right column opens with "speech, 2020, pp. 2172-2176." which is the tail of [22], then [23] to [30].
   - No section heading, table, figure, caption, "Section", "AUC", CONCLUSIONS, Acknowledgments, Compliance or REFERENCES text on page 5 (keyword scan found 0 lines).
   - Fonts on page 5: only NimbusRomNo9L Regular and Italic. No bold, so no heading on the page.
   - pdfimages lists 0 raster images in the whole PDF. The page 5 render (render-5.png) shows two columns of references and nothing else.
   - Text extent on page 5: top 75.9 pt, left column bottom 719.0 pt, right column bottom 302.5 pt (ends at [30], "2022.").
   - No funding acknowledgement or ethics statement is on page 5; both are on page 4 (below).
3. Conclusions end on page 4: YES.
   - Heading "5. CONCLUSIONS": page 4, right column, top 352.7 pt.
   - Last line of Conclusions: page 4, right column, top 529.6 pt: "and future work should focus on further narrowing the utility gap."
4. Acknowledgments: page 4, right column, run-in paragraph, lines at top 542.4 pt to 594.5 pt. First line "Acknowledgments. Research was sponsored by the ARO under", last line "and AG05133." (ARO W911NF-25-2-0040, NIMH R61MH135407, Pitt corpus NIA AG03705 and AG05133.)
5. Compliance with ethical standards: page 4, right column, run-in paragraph, lines at top 605.3 pt to 647.1 pt. First line "Compliance with ethical standards. This study is a secondary anal-", last line "authors declare no conflict of interest."
6. "6. REFERENCES" heading: page 4, right column, top 674.0 pt. Reference [1] (Cummins et al., Speech Commun. 2015) is complete on page 4 (top 694.2 pt, 3 lines). Page 4 both columns end at 723.1 pt.
7. Order on page 4 right column: end of 4.4 (release URL line) > 5. CONCLUSIONS > Acknowledgments > Compliance with ethical standards > 6. REFERENCES > [1]. Each heading appears exactly once in the PDF.

Bottom line: YES. The paper body, Conclusions, Acknowledgments and Compliance all finish on page 4. Page 5 is references only ([2] to [30]).

## Full page 5 text (pdftotext reading order: left column, then right column)

Verbatim from the PDF. Page range dashes are written as hyphens here.

```
[2] N. P. Narendra, Björn Schuller, and Paavo Alku, “The detection
of Parkinson’s disease from speech using voice source information,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 29,
pp. 1925-1936, 2021.
[3] Erfan Loweimi, Sofia de la Fuente Garcia, and Saturnino Luz,
“Zero-shot speech-based depression and anxiety assessment with
LLMs,” in Proc. Interspeech, 2025, pp. 489-493.
[4] Sejal Bhalla et al., “SpeechDx: A multi-task benchmark for
clinical speech AI,” arXiv:2606.17339, 2026.
[5] Zhongren Dong et al., “Foundation model-based evaluation of
neuropsychiatric disorders: A lifespan-inclusive, multi-modal,
and multi-lingual study,” IEEE J. Sel. Topics Signal Process.,
vol. 19, no. 5, pp. 796-809, 2025.
[6] Jingyi Chen et al., “Do audio LLMs really LISTEN, or just transcribe? Measuring lexical vs. acoustic emotion cues reliance,”
in Proc. EACL, 2026, pp. 5848-5877.
[7] Jiacheng Pang, Ashutosh Chaubey, and Mohammad Soleymani,
“Do audio LLMs listen or read? Analyzing and mitigating
paralinguistic failures with VoxParadox,” in Proc. ICML, 2026.
[8] Yunfei Chu et al.,
“Qwen2-Audio technical report,”
arXiv:2407.10759, 2024.
[9] Jin Xu et al.,
“Qwen2.5-Omni technical report,”
arXiv:2503.20215, 2025.
[10] Jin Xu et al.,
“Qwen3-Omni technical report,”
arXiv:2509.17765, 2025.
[11] Sreyan Ghosh et al., “Audio Flamingo 2: An audio-language
model with long-audio understanding and expert reasoning abilities,” in Proc. ICML, 2025, pp. 19358-19405.
[12] Arushi Goel et al., “Audio Flamingo 3: Advancing audio
intelligence with fully open large audio language models,” in
Proc. NeurIPS, 2025.
[13] Kimi Team et al.,
“Kimi-Audio technical report,”
arXiv:2504.18425, 2025.
[14] Juan Rafael Orozco-Arroyave et al., “New Spanish speech corpus database for the analysis of people suffering from Parkinson’s disease,” in Proc. LREC, 2014, pp. 342-347.
[15] Janaı́na Mendes-Laureano et al., “NeuroVoz: A Castillian
Spanish corpus of parkinsonian speech,” Sci. Data, vol. 11, no.
1, pp. 1367, 2024.
[16] Hagen Jaeger, Dhaval Trivedi, and Michael Stadtschnitzer, “Mobile device voice recordings at King’s College London (MDVRKCL) from both early and advanced Parkinson’s disease patients
and healthy controls,” Zenodo, doi:10.5281/zenodo.2867216,
2019.
[17] Fabien Ringeval et al., “AVEC 2019 workshop and challenge:
State-of-mind, detecting depression with AI, and cross-cultural
affect recognition,” in Proc. Audio/Visual Emotion Challenge
and Workshop (AVEC), 2019, pp. 3-12.
[18] Kurt Kroenke et al., “The PHQ-8 as a measure of current
depression in the general population,” J. Affect. Disord., vol.
114, no. 1-3, pp. 163-173, 2009.
[19] James T. Becker et al., “The natural history of Alzheimer’s
disease: Description of study cohort and accuracy of diagnosis,”
Arch. Neurol., vol. 51, no. 6, pp. 585-594, 1994.
[20] Saturnino Luz et al., “Detecting cognitive decline using speech
only: The ADReSSo challenge,” in Proc. Interspeech, 2021, pp.
3780-3784.
[21] Saturnino Luz et al., “Multilingual Alzheimer’s dementia recognition through spontaneous speech: A signal processing grand
challenge,” in Proc. ICASSP, 2023, pp. 1-2.
[22] Saturnino Luz et al., “Alzheimer’s dementia recognition through
spontaneous speech: The ADReSS challenge,” in Proc. Inter-

speech, 2020, pp. 2172-2176.
[23] Ilya Loshchilov and Frank Hutter, “Decoupled weight decay
regularization,” in Proc. ICLR, 2019.
[24] Francesco Barbieri et al., “TweetEval: Unified benchmark and
comparative evaluation for tweet classification,” in Findings of
EMNLP, 2020, pp. 1644-1650.
[25] Brian MacWhinney, The CHILDES Project: Tools for Analyzing Talk, Lawrence Erlbaum, 3rd edition, 2000.
[26] Kathleen C. Fraser, Jed A. Meltzer, and Frank Rudzicz, “Linguistic features identify Alzheimer’s disease in narrative speech,”
J. Alzheimer’s Dis., vol. 49, no. 2, pp. 407-422, 2016.
[27] Alec Radford et al., “Robust speech recognition via large-scale
weak supervision,” in Proc. ICML, 2023, pp. 28492-28518.
[28] Florian Eyben et al., “The Geneva minimalistic acoustic parameter set (GeMAPS) for voice research and affective computing,”
IEEE Trans. Affective Comput., vol. 7, no. 2, pp. 190-202, 2016.
[29] Benjamin Elizalde et al., “CLAP learning audio concepts from
natural language supervision,” in Proc. ICASSP, 2023, pp. 1-5.
[30] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu,
Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, “LoRA:
Low-rank adaptation of large language models,” in Proc. ICLR,
2022.
```

## Files written (all in checks/accel_24sep/page5/)

- REPORT.md (this file)
- check_page5.py (the check script) and check_page5.json (its full output, with line positions)
- page1_words.tsv to page5_words.tsv (pdftotext -tsv word boxes)
- page4_raw.txt (text extract). page5_raw.txt and page5_layout.txt are byte-identical to checks/final_2325Z/part12/final_runs/AudioLLMHealthUtilityGap_20260923_163127/page5.txt and checks/final_2325Z/part12/page5_layout.txt, so they are kept there only
- render-4.png, render-5.png (110 dpi renders of pages 4 and 5, not in this repository)

## Notes

- pdftotext -bbox and -bbox-layout crash on this PDF (libc++ out_of_range abort, exit 134). This is a poppler bug, not a PDF defect; -tsv and plain extraction work, and all positions above come from -tsv.
- No file in the Overleaf copy, the repository, omni_final or part12 was changed by this check.
