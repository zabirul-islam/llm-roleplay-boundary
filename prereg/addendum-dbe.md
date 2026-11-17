# Addendum A1 to prereg v3.1 (osf.io/ygvdf): DBE-KT22 secondary replication (v3, after review pass 6)

## 0. Scope, timing and information boundary
This secondary, non-confirmatory cross-domain replication was specified after examining the EEDI results and before
analytical access to DBE held-out responses. It retains the selected arm definitions and hyperparameters, with the
dataset-specific adaptations documented below.

- **Timing.** The EEDI confirmatory and secondary results were seen first (commits fa9b528, 8ca1ea0; H1 not
  rejected, Δρ = +0.005). EEDI's failure to reject H1 is not evidence that a useful advantage is absent, and this
  addendum does not use it as such. The registration names DBE-KT22 as a secondary replication (§1, §9) without
  defining it; this addendum supplies the definition. Nothing here changes the EEDI analysis.
- **Availability versus analytical access.** The raw `Transaction.csv` (SHA-256 c5f554ed…, the manifest copy) sat on
  Empire with mode 0664 from 2026-11-10 to 2026-11-16, readable by other accounts. There are two known study-team
  reads in that period. (1) `data/split.py` (commit 8a8632e, 2026-11-10) traversed every row with `csv.DictReader` and used three
  fields (`student_id`, `question_id`, `start_time`) to form the frozen split (`split-dbe.json`, SHA-256 3585d840…);
  its only outputs are that file and the counts it holds. (2) On 2026-11-16, before sealing, an inline script for this
  addendum traversed every row and used `student_id` and `question_id` to count student–item pairs per block and
  attempts per pair (the counts in §1); it printed only those counts. No other DBE response fields were used
  analytically or included in retained outputs before sealing, and no manual inspection of responses took place. This account rests on the git history, the code and
  the authors' statement; sealing the file now cannot prove that no access occurred earlier.
- **Sealed 2026-11-16T00:00:48** (`sim/dbe.py seal`): rows split into cc 70,402, hc 42,753, ch 30,281 and hh 18,517
  (transactions, repeat attempts included); ch, hh and the raw file are mode 0600 in `sealed/`; every read is logged.
  Reading DBE hh requires this addendum's lock file (`prereg/addendum-dbe.sha256`); ch is never read. The open blocks
  cc and hc, the item metadata and the split were copied to axis2 by logged reads; hh stays on Empire until the lock.
- **What was read to write this addendum.** Item text and option metadata; ids and dates; cc and hc responses for the
  calibration fits (§2), the key-scoring diagnostic (§1) and the calibration-item diagnostic (§3). No hh response.

## 1. Data and responses
- **Split (frozen 2026-11-10).** Calibration students: cohorts 2019–2020 (830; 825 have cc responses). Held-out
  students: cohorts 2021–2022 (434). Held-out items: 64 of 212. Student–item pairs: cc 70,242; hc 40,848; ch 30,220;
  hh 17,706.
- **Response = first attempt** per student × item: the earliest timestamp-valid `start_time` (format
  `2019-08-07 17:12:08.722 -0700`), compared as timezone-aware instants, ties broken by the numerically smaller
  transaction `id`. A row with a missing or unparseable `start_time` is excluded; affected rows and affected pairs are
  both counted, and a pair with no timestamp-valid row is lost (0 rows and 0 pairs in cc and hc). A first attempt
  without a valid choice is kept and scored incorrect (1 in cc, 0 in hc). The first correct or first valid attempt is
  never selected. 98.6% of pairs have one attempt. First-attempt records used: cc 70,242 (2PL, B1–B3 targets and the
  B4 isotonic map); hc 40,848 (anchors).
- **Scoring by the current key.** Correct = the chosen choice is the item's single `is_correct` choice in
  `Question_Choices.csv`. Real and simulated answers use the same key. Current-key scoring retrospectively re-scores
  logged choice ids using the frozen released key. Disagreement with recorded scoring does not establish which
  historical field changed or that the current item version was the version originally presented.
  - The earlier 69,697/70,401 agreement on cc is an all-attempt diagnostic; fitting and evaluation use the
    first-attempt records defined above.
  - **Key disagreement** for item j is the fraction of retained first attempts whose current-key binary score
    differs from the recorded binary score, among attempts with an interpretable recorded score. The `answer_state`
    mapping is frozen in code (`dbe_prep.ANSWER_STATE`: "true" → 1, "false" → 0, case-insensitive; anything else is
    missing). Missing or unrecognised states are counted separately.
  - On cc first attempts: 704 of 70,242 comparable records disagree, all in items 168 (66%), 216 (62%) and 209 (21%);
    no missing states. On hc first attempts (the held-out cohorts on calibration items): 0 of 40,848. The three
    calibration items stay in the calibration fits.
  - The key-consistent sensitivity includes primary items with at least one comparable hh record and disagreement
    ≤ 0.05; items with no comparable records are listed as unassessable. This is a response-defined diagnostic
    subset. Per-item disagreement on hh is reported.
- **Item text (frozen conversion, `dbe_prep.convert`).** From `question_rich_text`, because the plain `question_text`
  drops the LaTeX and flattens tables. Each LaTeX image becomes `$<source>$` (URL-decoded text after `gif.latex?`);
  table rows become lines with cells joined by " | "; line breaks are kept; other tags are dropped; entities are
  unescaped. All 44 images are LaTeX; no unrecoverable content remains (a non-LaTeX image would become "[image]" and be
  counted; there are none). Seven items contain tables. Options follow in ascending choice id, lettered A, B, C, …;
  every item has exactly one correct choice. Output `items-dbe.json` (SHA-256 9ec9606f…).
- **Item sets.** The registered EEDI rule, applied unchanged: held-out items with TF-IDF cosine < 0.8 to every
  calibration item. It removes 33 of the 64 held-out items, leaving **31 primary items in 30 templates** (one pair).
  Because this is a large reduction, all 64 held-out items (50 templates at cosine ≥ 0.8) form a prespecified
  sensitivity analysis. Output `components-dbe.json` (SHA-256 54ae18f2…).
- **No classes.** DBE has no class identifiers. The registered class-level analyses (H2 description, within-class
  ρ, home-group sensitivity) are not applicable, and no classes are formed from years or student partitions. A
  cohort-level variance ratio over all held-out students is reported descriptively (§3); it does not recreate H2.

## 2. What changes and what is carried over
| Element | DBE | Kind |
|---|---|---|
| System prompt | "You are a university student taking an introductory database course, answering a practice quiz. {LEVEL} Answer as that student really would, not as an expert. Reply with only the letter of your answer." (no age) | Domain substitution |
| LEVEL sentences | novice "You often find database topics hard and you make the common mistakes students in your course make."; average "You are an average student in this course: you get many questions right but slip on harder ones."; strong "You are strong at databases and usually, but not always, get questions right."; A1/A2 "You usually get about {X}% of database questions like these right." | Domain substitution ("maths" → database; "students your age" → students in your course) |
| B4 rating | system "You are an experienced database lecturer at a university."; question "What percentage of students in a university introductory database course would answer this question correctly? Reply with only a number from 0 to 100." τ = 0, one request per item | Domain substitution |
| Answer parsing | `sim.parse_choice` (first standalone A–D letter), then a range check: a letter beyond the item's option count is invalid; invalid = incorrect. Held-out items have 2 (14), 3 (4) or 4 (46) options | Method adaptation |
| Response rule, item text, scoring | §1 | Method adaptation |
| 2PL, anchors | Refit on DBE cc first attempts (`calib.fit_2pl`, registered settings); anchor MAP from each held-out student's hc first attempts (`calib.theta_map`); θ = 0 without anchors (1 of 434). E-anchor is retrospective item-response holdout, as registered; hc and hh are split by item identity, not time | Refit on DBE |
| B1–B3 | Refit on DBE cc items (148): BGE-M3 and TF-IDF on the converted text, registered ridge grid and CV (chosen α = 1.0 for all four) | Refit on DBE |
| B4 isotonic map | Fitted on DBE cc items with Qwen's DBE ratings (all 212 items rated; ρ raw rating vs cc proportion = 0.436) | Refit on DBE |
| Arms, comparator | A2 τ 0.6 (primary contrast), A0 τ 0.8, A1 σθ 1.0 τ 0.8; B4. Qwen3.5-27B only; R = 5; seeding as registered (per-request seed from "391-dbe-r{r}-{arm}-{u}-{q}") | Carried over |
| Analysis code | `analyze.Prepared.stats` (locked) for the point estimates; no bootstrap (§3) | Carried over; inference removed |

Frozen artifacts (axis2 `runs/dbe/`, SHA-256): `frozen/item-predictions.json` d25cc1b6…, `frozen/irt-cc.json`
ced559c3…, `frozen/anchors-hc.json` 5468077b…, `frozen/settings.json` d122f8f4…, `rate-req.jsonl` 71613d1a…,
`rate-qwen3.5-27b.jsonl` cea54659…, `emb-bge-m3.npz` 0c14b710…. A second freeze run gave byte-identical files.

## 3. Analysis
- **Evaluation support.** Primary: the 31 primary items with at least one hh first attempt; equal item weight;
  item difficulty = proportion correct among the held-out students who attempted the item. An item without a B4
  rating would be excluded from every contrast and listed (none lack one). An item with no hh first attempt is
  non-estimable and listed. Undefined correlations are non-estimable, never filled in.
- **Primary DBE contrast (no confirmatory claim).** Δρ = mean over replicates of ρ(A2) − ρ(B4).
- **Inference and reporting.** DBE-KT22 will be reported as a descriptive secondary replication. This reporting
  decision is fixed before access to DBE held-out responses, but after the EEDI confirmatory result was known. For the
  primary analysis and each prespecified sensitivity analysis, we will report the point estimates and contrasts defined
  in this addendum, all five generation-replicate values, and their Monte Carlo standard errors. Monte Carlo standard
  errors summarize generation-replicate variability only; they do not quantify uncertainty from sampling students,
  items, or templates and must not be interpreted as total uncertainty.

  We will report no DBE confidence intervals or inferential p-values and will make no DBE claim of statistical
  superiority, exclusion of the 0.10 margin, or confirmatory replication. DBE results will be used to describe the
  observed predictor-ranking pattern across domains, not to establish whether the population contrast exceeds or
  falls below 0.10.

  The existing design run used 64 calibration items in 46 templates. It is retained solely as a calibration-item
  diagnostic, not as validation of the primary held-out analysis (31 items in 30 templates) or the all-64 held-out
  sensitivity analysis (64 items in 50 templates). It examined one-sided non-coverage and did not establish the
  operating characteristics of a two-sided 95% interval or both proposed directional tests. In the
  misspecified-generator diagnostic, mean reference truth was 0.067 rather than the 0.10 boundary; the absence of
  rejection in four eligible null cohorts is insufficient to establish error control.

  Sampling uncertainty for the primary DBE contrast is not quantified here. No specific interval width or power claim
  is made. No additional validation simulation is required for this descriptive reporting scope.
- **Calibration-item diagnostic, recorded** (`results/dbe/design-dbe.json`; criterion committed at 0d6ca3f before the
  run; reference-truth Monte Carlo SE 0.0063–0.0064):

  | Generator | Mean truth | One-sided non-coverage, all cohorts | Cohorts with \|truth − 0.10\| ≤ 0.02 | Rejection at 0.10, truth ≥ 0.10 |
  |---|---|---|---|---|
  | 2PL | 0.101 | 24/500 = 0.048 [0.033, 0.070] | 24/497 non-covered | 15/292 = 0.051 [0.031, 0.083] |
  | Misspecified | 0.067 | 23/500 = 0.046 [0.031, 0.068] | 1/31 non-covered | 0/4 |
- **Sensitivity support.** Each sensitivity filter is applied separately to the 31-item primary set, except the
  explicitly designated all-64-item analysis: items without LaTeX; items without tables; four-option items only;
  key-consistent items only (§1). Filters are not combined. Frozen predictions and fitted parameters are reused; no
  sensitivity analysis triggers refitting.
- **Cohort variance ratio** (`dbe_prep.cohort_variance`). Within each analysis set and replicate, each student's score
  is their proportion correct over observed retained first-attempt items. Students with no such items are excluded
  from both real and simulated score calculations and counted. Sample variances use n − 1. The cohort variance ratio
  (simulated / real) is calculated separately within each replicate, then the five ratios are averaged; replicate
  values are also reported. Positive real variance with zero simulated variance gives ratio zero. Zero real variance
  makes the ratio undefined. Reported for A2, A0 and A1 on the primary set and for A2 on each sensitivity set.
- **Also reported (descriptive, same rules):** ρ, MAE, bias and calibration slope for every arm and baseline; A2
  against B1–B3; E-pop (A1 against B4); invalid-output rate by option count; key consistency per item.

## 4. Before opening DBE hh
- [x] Sealing done and logged (§0); block counts match §1
- [x] Item text, item sets, fits, B4 ratings frozen and hashed (§1, §2); freeze is deterministic
- [x] Inference scope fixed as descriptive (§3); the calibration-item diagnostic is recorded, not used as validation
- [x] Review pass 6 answered (`review/rosalind-2026-11-16-pass6.md`; independent verification
  `review/pass6-verification.md`, `review/council-2026-11-16-pass6.md`)
- [x] Final closure review: "ready to freeze", all 13 rows closed, no new blocker (`review/rosalind-2026-11-16-closure.md`);
  selftests of `dbe_prep.py`, `dbe.py` and `analyze.py` pass on axis2, code identical to the repository
- [ ] `prereg/addendum-dbe.sha256` committed (addendum, `sim/dbe.py`, `sim/dbe_prep.py`, artifact hashes above,
  diagnostic output); addendum and lock deposited on the OSF registration. Held-out access is permitted only after both.

**Stopping rule.** The next review is limited to verifying closure of the enumerated pass-6 issues, the
diagnostic-shape clarification, and the generation-only uncertainty caveat. Each closure must identify the revised
passage and its supporting implementation or output location. A new blocking issue must identify a demonstrable
factual error, an ambiguity capable of changing a reported result, or a claim unsupported by the evidence, with a
precise location and explanation of consequence. Additional analyses, simulations, stylistic preferences, and expanded
scientific aims are advisory unless necessary to correct such a defect. Once the closure checklist is satisfied and no
qualifying defect remains, the owner will freeze the addendum and implementation, record their commit identifiers or
hashes, deposit the addendum on OSF, and only then access DBE held-out responses. No further discretionary review
round will be opened. Any subsequently discovered substantive error will be handled through a dated, transparent
correction or deviation record.
