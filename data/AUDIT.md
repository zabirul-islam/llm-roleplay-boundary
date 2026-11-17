# WP1 — public student-data audit (verified 2026-11-10)

Plan: `docs/ncomms-paper-plan.md` (rev 2026-11-10b), WP1. Licences read from each dataset's own documentation,
card or paper (no emails). Files downloaded to the Empire shared home `~/ncomms-data/` — **never into git**.
Hashes: `MANIFEST.md`. Splits: `split.py` (frozen, hashes in `MANIFEST.md`).

**Must-have:** readable item text · option each student chose · ≥ 1,000 students · research-use licence.
**Preferred:** real class IDs · answer sequences over time.

## Chosen
| # | Dataset | Licence (quoted, source) | Verified content |
|---|---|---|---|
| 1 | **EEDI text release** — He-Yueya, Ma, Gandhi, Domingue, Brunskill, Goodman, arXiv 2407.15645; repo `github.com/joyheyueya/psychometric-alignment`; data in the public Google Drive folder linked from its README | "The EEDI dataset is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License" and "With permission from Eedi, we have extracted the text from these question images and released this modified dataset" (`dataset_documentation.pdf`) | Read on Empire: **443,433 answers, 2,287 students, 573 questions, 176 GroupIds** (real classes; 168 home classes, 101 with ≥ 10 students). Columns: QuestionId, AnswerId, UserId, IsCorrect, CorrectAnswer, **AnswerValue (1–4)**, Gender, DateOfBirth, PremiumPupil, DateAnswered, Confidence, **GroupId**, QuizId, SchemeOfWorkId, age, **problem (question + options as text)**. Sept 2018–May 2020, mostly age 11–12, England. Questions with graphs were dropped by the releasers. |
| 2 | **DBE-KT22** — Abdelrahman, Abdelfattah, Wang, Lin, arXiv 2208.12651; Australian Data Archive doi:10.26193/6DZWOH; copy used: HF `Unggi/dbe-kt22_raw_data` @ be0a892 (public, not gated) | Paper §3.5: shared "under an unrestricted access policy"; abstract: public access "upon proper citation to the authors". ADA licence page not readable by script (bot wall) — owner to glance once. | Read on Empire: `Transaction.csv` 161,953 rows with **`answer_choice_id`**, `answer_state`, `start_time`, `end_time`, `difficulty_feedback`, `trust_feedback`, `selection_change`, `hint_used`; `Question_Choices.csv` has `choice_text`, `is_correct`. 1,264 students with answers, 212 questions. Cohorts by first-answer year: 2019 302 · 2020 528 · 2021 402 · 2022 32. University database course (ANU). |
| 3 | **FoundationalASSIST** — Worden, N. Heffernan, C. Heffernan, Sonkar, arXiv 2602.00070; HF `ASSISTments/FoundationalASSIST` | CC BY-NC 4.0 (HF card; ASSISTments learning brief). **Gated**: one click "agree to use this dataset for non-commercial use ONLY" on Hugging Face. | Not yet downloaded (needs the owner's click). Card: 5,000 students, 3,395 problems with full text + options, 1,722,169 interactions incl. the option selected, 2019–2024, grades 6–8 maths. No class ID found. |

## Not used
| Dataset | Why |
|---|---|
| Eedi NeurIPS 2020 original (arXiv 2007.12061) | Same data as #1 but question images only, "shared solely for the purpose of this competition and must not be used for any other purpose". #1 is the permitted text version. |
| EdNet KT1 (riiid, CC BY-NC 4.0) | No question text (content file = IDs, tags, correct answer). |
| XES3G5M (MIT) | Right/wrong only, no option chosen. |
| MAP/MIME (Kaggle 2025, MIT per HF mirror) | 15 items; contains synthetic explanations. |
| Eedi "Mining Misconceptions" (Kaggle 2024) | No student responses; licence not found. |
| PISA 2022 public files | Too few released items; scored credit, not option. |

## Licence rules for this project (binding)
1. All three are **non-commercial**: use them for this research paper only — never in ALIVE product features, the
   patent filings, or anything sold or licensed.
2. EEDI is **NoDerivatives**: never redistribute the data or any modified copy (including the split files with
   answers). Release code, split seeds/hashes and aggregate results only.
3. Cite each dataset's paper (and Eedi as the source of #1).
4. Public de-identified data may sit on Empire AI; ALIVE student data never goes there.

## Splits (frozen 2026-11-10, `split.py`, seed 391; ids and dates only were read)
- EEDI: 30 of 101 eligible home classes held out (621 students) + 172 of 573 questions held out.
- DBE-KT22: cohorts from 2021 on held out (434 students) + 64 of 212 questions held out.
- FoundationalASSIST: split after download (by student × problem).

## Sources
- https://arxiv.org/abs/2407.15645 · https://github.com/joyheyueya/psychometric-alignment (README, dataset_documentation.pdf)
- https://arxiv.org/abs/2208.12651 (PDF §3.5) · https://dataverse.ada.edu.au/dataset.xhtml?persistentId=doi:10.26193/6DZWOH · https://huggingface.co/datasets/Unggi/dbe-kt22_raw_data
- https://arxiv.org/abs/2602.00070 · https://huggingface.co/datasets/ASSISTments/FoundationalASSIST · https://www.assistments.org/learning-briefs/foundationalassist
- https://arxiv.org/abs/2007.12061 (guide §2.2) · https://github.com/riiid/ednet · https://github.com/ai4ed/XES3G5M

## FoundationalASSIST (added 2026-11-16; registered test under addendum A2, in preparation)
- Source: Worden, Heffernan, Heffernan & Sonkar (2026), FoundationalASSIST, AIED 2026 / arXiv 2602.00070; HF
  `ASSISTments/FoundationalASSIST` (gated; access granted to the first author's RPI-verified account).
- Licence: CC BY-NC 4.0 plus the ASSISTments Responsible Use Guidelines: non-commercial research only; stored on secured
  institutional systems (axis2, RPI; owner-only directory); not shared; no re-identification attempts; PII found must
  be reported to etrials@assistments.org; data deleted or returned when the research is complete. Data never in git;
  only hashes and aggregates are committed.
- Content: 3,395 Illustrative Mathematics grade 6–8 problems; 5,000 students; 1.72M first answers with
  `discrete_score` (1 = correct on the first try without hints or seeing the answer). No class identifiers.
- Metadata audit (Problems.csv only, 2026-11-16; no response read): 795 single-answer multiple choice, 380 select-all,
  2,188 fill-in, 32 ordering. Eligible for role-play (multiple choice, one correct option, 2–4 options, no image in
  stem or options): 524 items in 472 problem sets.
- Split and seal (2026-11-16T12:04:39, `sim/assist_seal.py`, commits 3985deb + column-name fix; README's `user_xid` is
  `user_id` in the file): 5,000 students, 1,000 held out; 522 eligible items in 472 problem sets (the committed rule;
  an earlier ad-hoc count said 524); 283 sets held out whole = 344 items, 303 of them eligible. Rows: cc 1,284,618;
  hc 322,967; ch 116,596; hh 29,203. ch, hh and the raw Interactions.csv are mode 0600 in `assist-blocks/sealed/`;
  hh reads require `prereg/addendum-assist.sha256`. `split-assist.json` (ids only) SHA-256 63a6d2ec…, kept on axis2.
  The split read only `user_id` (all rows) and Problems.csv metadata; the seal routed rows by `user_id` and
  `problem_id`; no other response field was used.
- Items (`assist_prep.py items`, dataset's own `clean_problem_body`, prep venv bs4 4.12.3 / pandas 2.2.3): 3,368
  unique problem ids (15 ids repeated in Problems.csv: 14 identical rows, 1 differing only in fill-in fields); 484
  eligible with a resolved key (40 excluded: answer string not among the options). Held-out eligible 284; 22 removed
  by the registered near-duplicate rule; primary 262 items in 241 templates (problem sets joined with TF-IDF ≥ 0.8).
- Freeze (cc + hc only; byte-identical on two runs): cc 1,217,158 pairs (62,269 duplicate rows, smallest id kept;
  5,191 rows without a 0/1 score), 4,000 students, 2,997 items; hc 306,217 pairs, 1,000 anchored students; ridge α:
  B2 1, B3 10, B1 b 1, B1 log a 1; ratings 3,366/3,368 parsed; ρ(raw rating, cc proportion) 0.289 (all items),
  0.172 (eligible items).
