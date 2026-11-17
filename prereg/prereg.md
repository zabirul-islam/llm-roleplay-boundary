# Pre-registration v3.1: what calibrated LLM students can and cannot reproduce about real classes

**Status:** DRAFT v3.1 (2026-11-14), not locked. No held-out answer (block hh) has been read; block ch will never be
read (`results/calib-v3/provenance.json`). v2 is kept verbatim in `prereg-v2-record.md`; v3 → v3.1 answers the third
outside review (`review/rosalind-2026-11-14-pass3.md`, response in `review/response-rosalind.md`). Lock = §11 checklist
complete → `prereg.sha256` pushed to git → OSF timestamped registration by the owner. The first hh read comes after
the OSF timestamp.
**Authors:** Md Zabirul Islam, Ge Wang. **Plan:** `docs/ncomms-paper-plan.md`. **Reviews:** `paper-ncomms/review/`.
**Data:** public, de-identified, non-commercial licences (`paper-ncomms/data/AUDIT.md`); ALIVE under IRB 2347.

**Question.** For the frozen models and procedures studied here, can anchor-conditioned LLM role-play rank item
difficulty at least 0.10 better (Spearman ρ) than a calibration-selected direct-rating predictor from the same model?
We test exclusion of that advantage (H1, the only confirmatory test) and describe score dispersion and split-half
consistency within home classes (H2, descriptive). Comparisons with text-conditioned psychometric and embedding predictors show whether any benefit extends beyond
conventional supervised prediction. Conclusions are restricted to the evaluated procedures, the evaluated cohort, its
administration pattern and this public benchmark; they are not impossibility claims about LLM student simulation.

## 1. Data, blocks and eligibility
| Dataset | Role | Split (frozen 2026-11-10, ids and dates only) | SHA-256 |
|---|---|---|---|
| EEDI text release (He-Yueya et al. 2024) | primary | 30 whole home classes (621 students) + 172 of 573 items held out | `split-eedi.json` f678a7c01565bd4d55359ba79223bbbd3e27762fa06d79fe1d7670d868044556 |
| DBE-KT22 | secondary replication | cohorts ≥ 2021 (434 students) + 64 of 212 items held out | `split-dbe.json` 3585d840fbb2a99cb16b44670192ea86b4ae9b356caa77e1fb876c7311bd3c0d |

- **Four blocks** (`sim/heldout.py`, written 2026-11-14 on Empire; every read appended to `access.log`):

  | Block | Rows | Use |
  |---|---:|---|
  | cc (calibration students × calibration items) | 225,999 | fitting and development |
  | hc (held-out students × calibration items) | 84,999 | anchors for held-out students' abilities, and the membership audit |
  | ch (calibration students × held-out items) | 96,154 | sealed forever (item cold start) |
  | hh (held-out students × held-out items) | 36,281 | sealed until lock; the confirmatory test |

- **Information boundary and provenance.** The protected evaluation outcomes are hh students' answer choices and
  correctness. Item text, answer keys, identifiers, assignment incidence and dates may be processed before
  registration only for splitting, key reconciliation, home-class construction, template detection, figure detection
  and administration matching. These are permitted metadata uses; they are not evidence that the evaluation items
  were unseen as text.
  - Mechanical partitioning of the raw release is distinct from analytical use of responses. The provenance ledger
    (`sim/provenance.py` → `results/calib-v3/provenance.json`) lists, for every derived artifact used in §§2–6, its
    producing code, source files, fields read, the blocks whose response values it uses, creation time and SHA-256:
    `items.json`, `homes.json`, the cc/hc blocks, the split, the template map and 134-item list
    (`components.json`), the embeddings, the B4 rating requests and replies, the development scores and the §6
    selections.
  - Metadata taken from all rows, including the sealed blocks: `items.json` (item text; the majority CorrectAnswer
    over all rows) and `homes.json` (each student's majority GroupId). No AnswerValue from ch or hh enters any
    artifact.
  - Final 2PL item parameters, baseline fitting and §6 choices use cc response values only. hc response values are
    used only for held-out students' anchor abilities. Target-item answer keys are used for scoring and are not part
    of any simulator or direct-rating input.
  - **Historical access.** Before the 2026-11-14 quarantine, the v2 code (`calib.read_side`, 2026-11-10 to 11-12)
    streamed the raw CSV and kept only rows with a calibration student and a calibration item (= cc); other rows were
    parsed and discarded, never stored or summarised. `calib.guard` refused the held-out side from commit 52ef64d
    (2026-11-10). No other EEDI use exists in the repository. The access logs on Empire and axis2 contain no hh or
    ch read (checked by `provenance.py`); the sealed files are mode 0600 and the raw file matches its manifest hash. File
    permissions and the logs are access controls, not proof of historical non-access.
- **Home class** = the class where the student gave most answers (ties → lowest id). 924 students answered in more
  than one class. One row per student × item, as released (0 repeats in the open blocks). Dates: Sep 2019 – Apr 2020.
- **Membership audit (2026-11-14, ids only).** 213 of the 621 held-out students also answered in a group other than
  their home class (6.3% of hc rows). Linking each held-out home class to every group its students answered in
  chains 28 of the 30 classes into one component (95% of students); requiring ≥ 5 shared students per link still
  leaves 55% in one component. Groups seen only on sealed hh rows are not visible to the audit. The 30 home classes
  are therefore not treated as independent sampling units (§8).
- **Answer key** = the majority CorrectAnswer over an item's rows (items 43 and 860 have conflicting rows); a tie
  would exclude the item.
- **Confirmatory item set** = the 134 held-out items with no calibration near-duplicate (TF-IDF cosine < 0.8;
  `audit.py`). They form 128 templates: 6 pairs with cosine ≥ 0.8 between them, 122 singletons (`components.json`).
  Sensitivity analyses: all 172 items; the 14 held-out items that refer to a figure the text release lacks removed
  (regex frozen in `audit.py`); home-group rows only (answers given inside the student's home group; a
  prespecified sensitivity analysis, not a device for asserting class independence).
- **What is predicted.** The observed subset of hh (not a complete matrix). Item difficulty = proportion correct
  among the held-out students assigned the item. Score variance is "under the observed administration"; it is not
  read as latent heterogeneity.
- **Pretraining exposure.** EEDI items have been public since 2020. Checkpoints are pinned (§3); a near-duplicate
  audit and an option-permutation arm are sensitivity checks, not decontamination certificates.

## 2. Estimands
- **E-anchor (primary): anchor-assisted, item-response holdout.**
  - Each held-out student's ability is the MAP estimate from all their hc responses under item parameters fitted on
    cc only (N(0,1) prior; hc never updates item parameters or population hyperparameters).
  - A student with no usable anchor receives the prior MAP, θ = 0, and is counted separately.
  - The simulated student answers exactly the student's observed hh items.
  - The hc/hh partition is by item identity, not chronology. The estimand is retrospective prediction of withheld
    item responses given complementary-item responses. It does not establish forecasting from information available
    before each target response.
- **E-pop (secondary).** Abilities are sampled from N(0, σθ²). These are population-draw predictions, not
  individual twins.
- **Inner validation reproduces the boundary.** 5 folds over home classes × items (seed 391). For fold k:
  - the 2PL is refit on (classes ≠ k) × (items ≠ k);
  - test cells = (classes k) × (items k);
  - anchors = fold-k students' answers on items ≠ k.
- **Anchor-budget curve (secondary).** Calibration-side item parameters stay fixed; only each student's ability MAP
  is re-estimated at each budget. One frozen seeded ordering of each student's usable hc items (seed "391-budget-u")
  defines nested anchor subsets for k ∈ {5, 10, 20, all}; students with fewer than k anchors use all of theirs.

## 3. Simulators (prompts verbatim in `sim/sim.py` and `sim/knob.py`; hashes at lock)
- **Models.**
  - Qwen3.5-27B (primary; HF sha fc05daec18b0…; thinking disabled through the chat template);
  - Gemma-4-31B-it (secondary; 842da3794eaa…);
  - both bf16, vLLM 0.31.0, `--max-num-seqs 256`, max 8 output tokens.
  - Answer = the first standalone A–D letter. Invalid output is kept and scored as incorrect (its own option bin).
- **System prompt:** "You are a {11|12}-year-old school student in England answering a maths quiz. {LEVEL} Answer as
  that student really would, not as an expert. Reply with only the letter of your answer." The age is seeded per
  student and replicate. The user turn is the item text.
- **Arms** (the LEVEL text varies):

  | Arm | LEVEL / extra | Ability source |
  |---|---|---|
  | A0 uncalibrated | one of three fixed level sentences (novice/average/strong; seeded thirds) | none |
  | A1 population | "You usually get about X% of maths questions like these right." | θ ~ N(0, σθ²), σθ ∈ {0.6, 1.0, 1.4} |
  | A2 anchor | same sentence | θ = anchor MAP |
  | A3 history | "Earlier in this quiz you answered the questions shown in this conversation; your answers are your own. Keep answering as the same student." + 10 of the student's own anchor items as user turns, with their chosen letter as assistant turns | the student's own answers |
  | δ variants (A1δ, A2δ) | X replaced by (X − α)/β, where α, β come from an OLS of achieved on declared accuracy in the other folds' A1 (σθ 1.0, τ 0.8) run | as A1 / A2 |
  | Permutation (A1, A2 at τ 0.8) | options A–D shuffled per student × item and remapped before scoring | as A1 / A2 |

  - X = the 2PL-implied mean proportion correct over the training items, rounded.
  - Temperature τ ∈ {0.6, 0.8, 1.0} is crossed with A1, A2 and A3 (A0 at 0.8; δ variants at 0.8 and 1.0).
- **No per-item misconception text.** A transferable misconception procedure has not been validated, so none is
  used. A3 is the only arm explicitly conditioned on observed response history. It is a history-prompting
  comparator, not a response-trained model. The other arms keep their student-level ability or persona across items
  but receive no observed error history.
- **Replicates.** R = 5 simulated classes per real class. Replicate r draws its own ages, A0 level thirds and E-pop
  abilities from replicate-specific seeds and sends a per-request vLLM sampling seed (`knob.py --rep r`). The confirmatory
  hh run uses r = 1…5; development replicate 0 (the 2026-11-14 round) had no per-request seed.
- **One writer per output file** (`sim.py` lock); a second runner on the same file exits.
- **Development stopping rule.** One calibration-side round of the arms above (2026-11-14). No prompt rewording
  after its scores were seen.

## 4. Baselines (same folds, same information)
- **B1 text-conditioned 2PL:**
  - ridge regressions from frozen BGE-M3 item-text embeddings to the cc-fit b and log a of the training items;
  - predicted item p = mean over the item's assigned students of logistic(a(θ − b));
  - θ from the same source as the competing arm (anchor MAP for E-anchor; the A1 replicate's draws for E-pop).
- **B2:** ridge from embeddings to logit item p. **B3:** ridge from TF-IDF to logit item p.
- **B4:** the same model's direct rating ("What percentage of 11- to 12-year-old students in England would answer
  this question correctly?", τ = 0, one request per item), with an isotonic map fitted on training items.
- **B5:** human split-half agreement of item difficulty, a reference, not a mathematical ceiling on model–human
  correlation.
- **Frozen artifacts** (`score3.py freeze` on all of cc, 2026-11-14; `runs/v3/frozen/`, hashes in the lock manifest):
  the cc 2PL (401 items), B2/B3/B4 item predictions and B1's b̂ and â for all 172 held-out items, and the anchor MAP
  of all 621 held-out students. Two runs gave byte-identical files. B1's item p is averaged over the hh assignment
  after lock from these frozen numbers. `settings.json` records:
  - the IRT fit (`calib.fit_2pl`, joint MAP, 400 iterations, lr 0.05, seed 391) and the anchor MAP (`theta_map`,
    30 Newton steps);
  - ridge: grid 10⁻³…10³, 5-fold CV with permutation seed 391; chosen α = 1.0 for B1 b, B1 log a, B2 and B3;
  - logit targets clipped to [0.01, 0.99]; TF-IDF over the text of all scored items;
  - isotonic map (pool-adjacent violators, unweighted over cc items with a parsed rating); outside the training range
    it returns the end value;
  - rating parse: the first number in the reply, capped at 100, divided by 100; an unparsed reply gives no B4
    prediction.
  - No baseline is retuned on hc or hh outcomes, and nothing is re-estimated inside the bootstrap.
- **Distractor baseline:** uniform guessing, 1/3 on each valid wrong option and 0 on invalid output.

## 5. Endpoints (implemented in `sim/analyze.py`; secondary distractor and within-class code in `sim/score3.py`)
- **Evaluation support and weighting.** Primary endpoints use the 134-item set and its observed hh cells. Items with
  no observed hh response are non-estimable and listed. For every item, real and simulated proportions use exactly
  the same assigned students; invalid simulated output counts as incorrect.
- **Difficulty.** Within each replicate, pooled Spearman ρ across item proportions (equal item weight, average ranks
  for ties). The primary estimate is the mean of the five replicate correlations, not the correlation of
  replicate-averaged predictions. MAE and mean bias (predicted − observed) weight items equally; the calibration
  slope is the unweighted OLS slope of observed on predicted item proportion, with an intercept.
- **Within-class ρ (secondary).** A class contributes when at least five primary items each have responses from at
  least three of its students. Correlations are computed within replicate and averaged over eligible classes, then
  replicates.
- **Student structure.**
  - A student's score is the proportion correct over their observed primary items; variances use n − 1.
  - Halves: for each class, the templates its students observed, in a seeded random order (seed "391-half-c"),
    alternately assigned to half 1 and half 2 (`analyze.halves`). Frozen before any hh read.
  - A class is eligible for the H2 description when at least five of its students have an observed response in each
    half (from the observed pattern, not from outcomes). The same students are used for real and simulated statistics.
  - Half-score consistency = Pearson correlation across the class's students between their two half-score
    proportions (students with both half-scores defined; at least 3).
  - Within each eligible class, full-score variance and half-score consistency use the same students who have
    observed responses in both halves.
  - Reported for every eligible class and replicate, for A0 and the selected arm: real and simulated score variances,
    the variance ratio s²_sim / s²_real and its logarithm, real and simulated half-score consistency, and the paired
    difference r_sim − r_real.
  - Positive real variance with zero simulated variance: ratio 0, log −∞, kept as the most under-dispersed value.
    Zero real variance: reported separately; the ratio is undefined. An undefined correlation stays undefined, and
    so does any paired difference that uses it. Each count is reported per arm and replicate.
  - Per replicate: the median log ratio over classes with positive real variance (−∞ entries included), and the
    median paired difference over classes where both correlations are defined. Each median states its denominator;
    the second is labelled conditional on estimability. Distributions across classes and replicates are reported.
  - An empty summary set produces an undefined median with denominator zero. For an even number of observations,
    the median is the arithmetic mean of the two central ordered values; a central −∞ produces a median of −∞.
    Software interpolation must not turn this defined extended-real median into NaN.
- **Undefined classes.** No class is removed or smoothed because of its outcomes. Non-estimable classes remain
  listed in the results and are not treated as matches.
- **Distractors (secondary).** Human eligibility: at least five real wrong responses to the item. Conditional
  wrong-answer distributions use the three valid wrong options plus an invalid-output category (zero for real
  answers). Jensen–Shannon divergence uses base-2 logarithms and unsmoothed proportions; modal ties follow option
  order, invalid last. Items with zero simulated wrong responses are non-estimable and stay in the reported coverage
  denominator. For each method: eligibility, estimable coverage and conditional divergence; any comparison across
  methods uses a stated common estimable item set.
- Invalid-output rate per arm.

## 6. Final development selections
- **Registered simulators and comparator.** Development is complete. The primary simulator is **A2 at τ = 0.6**; the
  E-pop simulator is **A1 at σθ = 1.0, τ = 0.8**; the primary comparator is **B4** (direct rating + isotonic map).
  These were fixed before any analytical access to hh responses and will not be reselected on evaluation results.
  "Selected baseline" means selected on calibration data, not the best baseline on hh.
- **How they were chosen** (development record; `sim/select6.py` on `results/calib-v3/calib-v3.json`, all 28 arms
  complete → `choices-6.json`, full hash in the lock manifest):
  - Rule as written before scoring: the highest mean fold ρ in the family; arms within 1 SE of the best → fewer
    added parts (A2 < A2δ < A3; A1 < A1δ), then τ = 0.8.
  - E-anchor: best = A2δ τ 0.8 (0.207, SE 0.029); A2 τ 0.6 (0.197) is within 1 SE with fewer parts; A2 τ 0.8
    (0.155) is outside.
  - E-pop: best = A1 σθ 0.6 τ 1.0 (0.164, SE 0.082); eight A1 cells are within 1 SE. The τ = 0.8 preference leaves
    three cells (σθ 0.6, 1.0, 1.4); `select6.py` breaks that remaining tie by higher ρ (σθ 1.0, 0.140), a step
    written after the scores were visible. The written criteria are therefore not a complete, uniquely resolving
    algorithm; `choices-6.json` is the authoritative record.
  - Baseline: highest mean fold ρ among B1–B4 under each estimand: B4 for both (0.434, SE 0.022; next B1 anchor
    0.384 for E-anchor, B3 0.271 for E-pop).
- **Duplicate responses.** A3 τ 1.0 and A0 contain duplicate request ids (6,478 and 44,945; two runners appended to
  one file). A0 is the H2 descriptive control arm. The retained-row manifest (`score3.py retained`; on axis2 with
  the EEDI-derived files, SHA-256 in `results/calib-v3/server-artifacts.json`) records, for every duplicated id, the
  retained file line and the file's SHA-256; that manifest, not file order at read time, defines the retained row.
  The confirmatory hh runs are new runs under the single-writer lock, so this concerns development data only.

## 7. Confirmatory hypotheses
The single confirmatory test is H1 for EEDI × Qwen, E-anchor, on the 134-item set, at one-sided α = 0.05 with no
multiplicity adjustment.

- **H1: exclusion of a prespecified advantage.** Δρ = (1/5) Σr ρS,r − ρB4. Null H01: Δρ ≥ 0.10; alternative
  Δρ < 0.10.
  - If the H1 test rejects at α = 0.05, an advantage of at least 0.10 is excluded for the selected role-play
    simulator relative to the selected direct-rating predictor, in this cohort, across the item-template population
    the test set represents. Otherwise the exclusion question is inconclusive.
  - 0.10 is a prespecified benchmark threshold for this study, not an empirically validated smallest useful effect.
    Its relation to calibration-side split-half agreement (0.71–0.86) is descriptive. Excluding 0.10 neither
    establishes equivalence nor excludes every positive advantage.
  - Point estimates and intervals are reported whatever the decision. Superiority to zero, comparisons with B1–B3,
    the "beats all" contrast, MAE, bias, calibration slope and within-class ρ are secondary and create no
    confirmatory claim.
- **H2 (secondary, descriptive; §5).** The class-level dispersion and consistency results for A0 and the selected
  arm are reported whatever they show. H2 makes no confirmatory equivalence, non-inferiority or composite-success
  declaration and carries no test. The earlier H2 test and its failed validation are recorded in §11 and §12.
- **Scope.** Neither H1 nor the H2 description establishes generalisation to new classes, schools or students.

## 8. Inference (`analyze.core`)
- **Conditioning.** The evaluated students, their home-class memberships and their anchors are fixed. Because the
  membership audit (§1) shows that the 30 home classes are not independent units, classes and students are not
  resampled. Inference is to the item-template population the test set represents, conditional on this cohort and
  on the frozen fitted procedures.
- **Paired template bootstrap, 10,000 draws.** Each draw resamples the 128 item-template clusters with replacement,
  carrying each sampled template's complete observed human response vector and all predictions together. The same
  template multiplicities apply to every student, class, arm, replicate and baseline.
  - Item proportions do not change within a draw (students are fixed); repeated items are repeated observations for
    ranking.
- **Assumption.** Conditional on the cohort, distinct item-template clusters are treated as independent,
  exchangeable sampling units. Dependence within a template is retained by cluster resampling. The design validation
  includes correlated members within templates; it does not validate robustness to dependence between templates.
  Residual cross-template dependence, such as shared topic effects, remains an unvalidated assumption and limitation
  of the confirmatory inference.
- **Test (H1).** For the estimate T = Δρ and draws T*b, eb = T*b − T; p = [1 + Σb I(eb ≤ T − 0.10)] / (B + 1).
  This is a centred-bootstrap approximation, validated in §11. A non-estimable draw counts toward the tail; draws
  are never discarded or regenerated. If the original statistic is non-estimable, p = 1. The number of non-estimable draws is reported.
- **Interval.** A descriptive 95% basic interval for Δρ, [T − q0.975(e), T − q0.025(e)], whose coverage was checked
  in §11. With any non-estimable draw, no finite interval is reported. The interval does not replace the H1 decision.
  No interval is reported for the H2 statistics: the template bootstrap did not attain adequate coverage for them in
  the design validation (§11).
- **Simulation replicates.** R = 5 is fixed. Statistics are computed within replicate before averaging. Monte Carlo
  SE for Δρ = SD of the paired replicate differences / √5; bootstrap inference conditions on the five
  generated response matrices. Monte Carlo SE is reported separately, not added in quadrature. The calibration-side
  Monte Carlo SE / bootstrap SE ratio is recorded in §11; a ratio below 1/4 is a precision diagnostic only.

## 9. Secondary and exploratory (reported whatever they show)
The H2 description (§5, §7); Gemma; DBE-KT22; E-pop; anchor-budget curve; distractor endpoints; option-permutation arm; all-172-item, no-figure
and home-group-only sensitivity; persona adherence (declared vs achieved accuracy); the τ ablation. The H1 conclusion
is also reported descriptively at thresholds 0.05 and 0.15; these add no confirmatory hypothesis and do not change
the registered decision.

## 10. ALIVE (bounded prospective case study, IRB 2347, consented students only)
- **Spring 2027:** before each lecture opens, Rehearse-tab predictions over **all slides** are committed with hash and
  date.
- **Outcome:** each slide's real-difficulty status from the platform's student-difficulty records, independent of
  the simulator. **Comparators:** a content-only baseline (slide-text length/density) and chance.
- **Analysis:** lecture-clustered; any instructor revision prompted by a prediction is logged and reported.
- **AI interviews:** feasibility only. No AI-examiner score is an outcome; any interview score used comes from
  blinded human raters.
- **Status.** ALIVE is exploratory under this registration. Before the first prospective lecture, a timestamped
  addendum will freeze the difficulty-event definition, slide-level aggregation and denominator, the baseline
  formula, the handling of revised slides and missing observations, and the analysis for repeated students and
  lectures. Outcomes will not be defined after prediction performance is seen.
- **Fall 2026 (R370d/R390/R391):** prior evidence only (§12).

## 11. Before lock (checklist)
- [x] Quarantine + access log (2026-11-14)
- [x] Data audit; key, near-duplicate and figure rules
- [x] Knob-run diagnostic
- [x] Calibration-side v3 round scored; §6 selections recorded (2026-11-14)
- [x] Provenance ledger checked against the code's data dependencies and both access logs; no departure from the
  information boundary (`results/calib-v3/provenance.json`)
- [x] Membership and template audit (`results/calib-v3/components.json`)
- [x] Baselines and anchors frozen (`runs/v3/frozen/`, hashes in `results/calib-v3/server-artifacts.json`)
- [x] Retained-row manifest for duplicate ids (hash in `results/calib-v3/server-artifacts.json`)
- [x] Design validation of the final H1 procedure (`results/calib-v3/design4-h1.json`, `converge4-h1.json`;
  `sim/design.py`, which calls `analyze.core`). Pattern: real cc missingness of the 30 largest home classes × 134
  calibration items, 127 templates, R = 5, κ = 0.15. Truths are conditional on the cohort: the mean of Δρ over 40
  fresh item draws for the same simulated cohort. 500 simulations per scenario, B = 1,000; Wilson 95% intervals.

  | Generator | Truth Δρ | P(reject H1) | P(reject \| truth in null) | Coverage of Δρ interval |
  |---|---|---|---|---|
  | 2PL | 0.105 | 0.050 [0.034, 0.073] | 0.043 [0.026, 0.071] | 0.954 [0.932, 0.969] |
  | 2PL | 0.003 | 0.260 [0.224, 0.300] | n/a | 0.946 [0.923, 0.963] |
  | 2PL | −0.236 | 0.952 [0.930, 0.968] | n/a | 0.948 [0.925, 0.964] |
  | Misspecified | 0.109 | 0.052 [0.036, 0.075] | 0.059 [0.039, 0.089] | 0.934 [0.909, 0.953] |
  | Misspecified | 0.011 | 0.270 [0.233, 0.311] | n/a | 0.950 [0.927, 0.966] |
  | Misspecified | −0.225 | 0.940 [0.916, 0.958] | n/a | 0.942 [0.918, 0.959] |

  The misspecified generator uses a 3PL with guessing 0.2, slip 0.05 and a class effect (SD 0.4) for the real
  answers. No draw was non-estimable. The near-boundary scenarios had mean cohort-conditional truths of 0.105 and
  0.109. Their overall rejection probabilities are not exact boundary-size estimates. Rejection among cohorts
  classified as belonging to the null was 0.043 and 0.059; classification depends on estimated conditional truths.
  Reference-truth Monte Carlo uncertainty and the targeted boundary validation are reported separately (next item).
  Power was 0.952 and 0.940 in scenarios with mean conditional truths −0.236 and −0.225, respectively; these are near
  the development difference, not both exactly −0.237. Power at Δρ near 0 was 0.26–0.27. B = 1,000 and B = 10,000 on the same 50 data sets agree on every decision (largest |Δp| = 0.026).
- [x] Boundary coverage validation of H1 (Rosalind pass 4; `design.py boundary` → `results/calib-v3/design5-boundary.json`),
  criterion fixed before the run: 500 new cohorts per generator at the near-boundary parameters; each cohort's
  conditional truth from 400 independent reference item draws, with its Monte Carlo SE and convergence reported; H1
  algorithm unchanged. H1 rejects at c exactly when the one-sided 95% upper bound is below c, so non-coverage of the
  cohort's own truth estimates size at the boundary. Behaviour is taken as approximately nominal under a generator
  if the Wilson 95% interval for non-coverage over its 500 cohorts contains 0.05 or lies below it. If either
  generator fails, H1 is not claimed as validated and the specification is not locked.
  - Reference truths: Monte Carlo SE median 0.0044 (maximum 0.0048) under both generators. Truths from the first 200
    draws differ from the 400-draw truths by at most 0.017 (2PL) and 0.013 (misspecified). The 40-draw truths used
    in the previous item differ from the 400-draw truths with SD 0.014 and 0.013 (maximum 0.049 and 0.043). Mean
    reference truth 0.107 (SD across cohorts 0.006) and 0.111 (0.009).
  - Counts (k/n, Wilson 95% interval):

  | Generator | Non-coverage, all cohorts (criterion) | Non-coverage, \|truth − 0.10\| ≤ 0.02 | Rejection at 0.10, truth ≥ 0.10 | Rejection at 0.10, truth − 2 SE ≥ 0.10 |
  |---|---|---|---|---|
  | 2PL | 27/500 = 0.054 [0.037, 0.077] | 25/490 = 0.051 [0.035, 0.074] | 16/436 = 0.037 [0.023, 0.059] | 7/188 = 0.037 [0.018, 0.075] |
  | Misspecified | 25/500 = 0.050 [0.034, 0.073] | 21/427 = 0.049 [0.032, 0.074] | 15/467 = 0.032 [0.020, 0.052] | 8/286 = 0.028 [0.014, 0.054] |

  - The criterion is met under both generators: one-sided behaviour near the boundary is approximately nominal for
    the cohort-conditional estimand. This is a coverage validation; it does not address dependence between templates
    (§8).
- [x] The H1 + H2 + Holm procedure failed validation and was withdrawn before lock (`results/calib-v3/design3.json`,
  code at a91f725). H2a rejected in 0.22 of simulations at its boundary (2PL; 0.29 misspecified) and Holm FWER there
  was 0.068 [0.049, 0.094]. Coverage of the V0 and VS intervals was 0.65–0.91. Within a fixed cohort the template
  bootstrap was biased toward 0 for the variance-ratio statistics (mean T* − T = +0.030 for V0 against a sampling SD
  of 0.065), which the T-centred pivot reflects the wrong way. The scenario truth was the mean of the registered
  statistic itself, not a generating parameter (mean −0.226 at the boundary; log 0.8 = −0.223). On real cc simulator
  output the registered H2 statistics were non-estimable in 2–4 of 5 folds (`results/calib-v3/mc-ratio.json`).
- [x] Monte Carlo / bootstrap SE ratio for Δρ from 5 calibration-side replicates (`results/calib-v3/mc-ratio-h1.json`):
  median 0.195 over the five folds (range 0.084–0.322), a precision diagnostic only. Replicate 0 reproduces
  `calib-v3.json` (fold means: A2 τ 0.6 0.196 vs 0.197; B4 0.434 vs 0.434).
- [x] hh request builder written (`knob.py build-hh`; dry run on hc 2026-11-14; an hh read was refused and logged);
  hashed at lock
- [x] Environment lock (`prereg/environment.lock`: vLLM 0.31.0, serving arguments, model revisions, Python and
  package versions) and full code and artifact hashes in `prereg/prereg.sha256`
- [ ] OSF registration by the owner

## 12. History and prior evidence (non-confirmatory)
- **v1 → v2 (2026-11-10)**, after the Astra review: H3 moved to exploratory; the confirmatory family narrowed;
  misconception prompting dropped; matched administration added.
- **v2 → v3 (2026-11-14), after development data and the outside review.**
  - The knob run (cc only) was scored before the v3 rules existed: score variance matched, the mean was
    overpredicted (0.65 vs 0.51), and difficulty ρ was 0.06–0.16 across cells.
  - Together with the Rosalind review (4/10) this motivated the boundary framing: H4 (tutoring and class breadth)
    removed; H3 (PPI++) moved to future work; strong supervised baselines; two estimands.
  - Knob-run code is hashed in `knob-v2-prompts.sha256`; the diagnostic is in `results/knob/DIAGNOSTIC.md`.
- **v3 → v3.1 (2026-11-14), after Rosalind pass 3 (6/10).**
  - H1 became a single exclusion test (the three-outcome rule had overlapping labels and Holm covered only the
    advantage claim); H2's component tests and non-estimability rule were defined; §8 was rewritten.
  - The membership audit replaced the class-cluster bootstrap with the cohort-conditional template bootstrap (owner
    decision, 2026-11-14). The v3 design simulation (`design2-*.json`, 100 sims, the earlier rule, class resampling)
    is a preliminary diagnostic only; it does not validate the registered tests.
  - The δ correction for validation fold k uses other folds' A1 runs, whose fitted 2PLs may use fold-k responses. The
    δ development scores are therefore not fully out-of-fold estimates of the whole calibration pipeline. This is
    confined to cc development data; neither registered simulator uses δ.
  - This registration supersedes prospective statements in `review/response-rosalind.md`, which is a development
    record, not a specification.
  - H2 became descriptive (owner decision, 2026-11-15, development stage, before any hh outcome was accessed). The
    validation of the H1 + H2 + Holm procedure showed anti-conservative H2 tests and inadequate interval coverage,
    and the registered H2 statistics were often non-estimable on real calibration output (§11). H1 is now the only
    confirmatory test, at α = 0.05 without multiplicity adjustment, and was re-validated with the final code. The H2
    eligibility rule was not changed. A mean-centred bootstrap pivot and a higher class-size threshold were
    considered and not adopted. Restricting consistency summaries to defined simulated correlations was not adopted
    as a repair for confirmatory H2 inference. Such summaries are reported descriptively, explicitly conditional on
    estimability. The failed
    validation results are kept.
- **ALIVE prior evidence:**
  - R370d (seeded personas, ChatGeV3): 27/41 real difficulty slides caught vs 21.7 chance, p = 0.050.
  - R390 (population calibrated on 6 students): 14/41 vs 12.4, p = 0.32.
  - R391: Qwen 23/41 vs 19.3, p = 0.13; Gemma 19/41 vs 17.6, p = 0.38.
  - The arms flag different numbers of slides, so caught counts are not comparable across arms.
  - Interview examinees: the level ordering held in every arm; verdict leaks 6 / 25 / 5.
