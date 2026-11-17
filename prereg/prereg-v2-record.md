# Pre-registration — calibrated synthetic classes (Nature Communications paper)

**Status:** DRAFT v2, not locked. v1 written 2026-11-10 after the splits were frozen and before any held-out real
answer was read; v2 the same day after one independent methods review (Astra/GPT-6, verdict NOT READY, review kept in
`paper-ncomms/.harness/rounds/1_reviewer_astra.md`). Every fatal point is fixed below; the remaining pin-before-lock
items are listed in §8. Lock = §8 complete → second review → `prereg.sha256` + OSF timestamped registration, all
before the held-out responses are opened.
**Authors:** Md Zabirul Islam, Ge Wang. **Plan:** `docs/ncomms-paper-plan.md` (rev 2026-11-10b).
**Data:** public, de-identified, non-commercial licences (`paper-ncomms/data/AUDIT.md`); ALIVE data under IRB 2347.

## 1. Data, units and frozen splits
| Dataset | Role | Calibration side | Held-out side | Split SHA-256 |
|---|---|---|---|---|
| EEDI text release | **primary (confirmatory)** | students whose home class is not held out × 401 items | 30 home classes (621 students) × 172 items | `split-eedi.json` f678a7c0… |
| DBE-KT22 | secondary, descriptive | cohorts 2019–2020 × 148 items | cohorts 2021–2022 (434 students) × 64 items | `split-dbe.json` 3585d840… |
| FoundationalASSIST | exploratory only | split on arrival | — | — |

- `split.py` (a1ad6eb7…) read ids and dates only. **Home class** = the GroupId where the student gave most answers
  (ties: lowest id); 947 students appear in more than one class; classes with < 10 home students stay calibration.
- Quadrants: calibration = calibration students × calibration items (all fitting). Held-out test cells = held-out
  students × held-out items only (H1, H2). The two mixed quadrants are never used for fitting or testing.
- Response record: one row per student × item as released (the releasers kept the latest attempt); no other
  attempt selection is possible or made. Item difficulty = proportion correct **among the students who answered it
  in the held-out cells** (a respondent-specific quantity, not an intrinsic population difficulty).
- Pretraining exposure: EEDI items have been public since 2020 and may be in the models' training data. Disclosed;
  not testable here.

## 2. Simulator (fully specified for unseen items)
- **No held-out response-derived statistic enters any prompt or simulator parameter.** Item parameters are never
  needed for held-out items: the LLM answers the real item text; only **student** attributes are sampled.
- **Persona** = ability only. A 2PL model (joint MAP, N(0,1) priors on θ and b, log-normal(0,0.5) on a, θ
  standardised, fixed iterations and seed; `calib.py fit_2pl`) is fitted on calibration cells; persona abilities are
  drawn from N(0, σ_θ) and rendered by a fixed mapping to the model-implied expected proportion correct on the
  calibration items ("you usually get about 60 % of questions like these right") plus the age band.
- **Misconception prompting: NONE (decided).** A per-item favoured wrong option needs that item's own response
  data, which does not exist for held-out items without a transfer model; per the review's rule, without a frozen
  transferable procedure the feature is omitted. A text-based transfer variant may be studied as exploratory only.
- **Models:** Qwen3.5-27B (primary, pinned sha fc05daec…, chat template with thinking disabled) and base
  Gemma-4-31B-it (842da379…), both bf16 on vLLM 0.31.0. ChatGeV3 only in the ALIVE tier.
- **Decoding:** temperature τ (knob), max 8 tokens; answer = first standalone A–D letter (EEDI items all have four
  options; checked before lock). Invalid output is **kept** and counted as incorrect; its rate is reported per arm;
  no arm is ever excluded.
- **Knobs (frozen grid, 9 cells):** σ_θ ∈ {0.6, 1.0, 1.4} × τ ∈ {0.6, 0.8, 1.0}. No slip/guess, no mixture weight.
- **Selection:** 5 grouped folds on the calibration side (home classes and items both split, seed 391); score each
  cell on the double-held-out fold intersections with the loss L = MSE(item proportion correct) + 0.5 × mean JS
  divergence of option distributions + 0.5 × (log var ratio of student proportion correct)². Choose the cell with the
  lowest mean fold loss; tie (within 1 SE of the best) → the smaller σ_θ, then the smaller τ. The chosen cell is
  committed before lock.

## 3. Administration design (matched)
Each held-out real home class is mirrored by one simulated class: one simulated student per real student, who
answers **exactly the items that real student answered** in the held-out cells. Scores are proportion correct over
the student's own items in both arms, so exposure and denominators match. Simulation randomness: R = 3 independent
replicate classes per real class (seeds fixed), with the same seeds across compared arms.

## 4. Confirmatory hypotheses (EEDI × Qwen only; Holm over H1 and H2; α = 0.05)
- **H1 fidelity.** Spearman ρ between simulated and real held-out item difficulty (§1) is higher for the calibrated
  simulator than for **each** of three baselines: (a) uncalibrated personas (three levels, σ_θ, τ fixed at 1.0/0.8),
  (b) one direct difficulty rating per item by the same model (prompt frozen in §8), (c) ridge regression on item-text
  TF-IDF fitted on calibration items (α by 5-fold CV on calibration items). H1 holds only if all three comparisons
  hold; its p-value = the maximum of the three one-sided p-values (intersection–union). Inference: crossed bootstrap
  over held-out items and held-out classes (10,000 draws, paired across arms, replicate seeds averaged within draw).
- **H2 dispersion.** Two endpoints per held-out class, as log ratios simulated/real: (i) variance of student
  proportion correct; (ii) mean wrong-answer entropy (Miller–Madow corrected; items with ≥ 10 wrong answers in both
  arms). Claims: the **uncalibrated** log ratio < log 0.8 for both endpoints, and the **calibrated** log ratio lies in
  [log 0.8, log 1.25] for both (TOST). H2's p-value = the maximum of the four component p-values; inference by
  bootstrap over held-out classes.
- Secondary/descriptive: Gemma, DBE (two cohorts, conditioned on, no cohort bootstrap), every other measure.

## 5. Exploratory (not confirmatory)
- **H3 statistical value** (PPI++). Before held-out access the observation unit, target, predictor, sampling design,
  correction and variance estimator will be written in §8; repeated subsampling of the held-out pool assesses
  finite-pool behaviour only.
- Calibration-size curve: H1's ρ when calibrating on 5, 10, 20, 50, 200 and all calibration students.
- FoundationalASSIST replication; text-based misconception transfer.
- **Persona adherence** (idea from MatrAIx, arXiv 2608.04205): per simulated student, declared accuracy (the "about
  X %" in its prompt) vs achieved proportion correct on its calibration items; OLS slope, intercept and mean
  absolute gap per model (`metrics.adherence`). Slope 1 = the persona is obeyed, 0 = ignored.
- **ALIVE AI interview, real vs simulated:** per-level strengthsShare of simulated examinees (R390/R391) vs
  consented students' completed interview attempts. Run only with ≥ 5 consented students; else report the count.

## 6. H4 — breadth (prediction, then partial human checks; not confirmatory)
- Simulated tutor settings (no AI / direct answer / hint-first with varied explanations; immediate vs delayed help),
  100 classes × 30 students per setting. Measures as in §4 plus the share of concepts mastered by < 20 % of a class.
  Wrong-answer entropy measures distractor concentration, not proven distinct misconceptions.
- Predictions are committed before each check is opened: (a) the ALIVE help-timing arm (IRB 2347, consented, after
  final grades); (b) published experiments with public materials or reported distributions — the eligible list is
  frozen now: Bassner et al. 2025 (doi:10.1016/j.caeai.2025.100537), Bastani et al. 2025 (doi:10.1073/pnas.2422633122),
  Kestin et al. 2025 (doi:10.1038/s41598-025-97652-6); extraction = reported group means, SDs and distributions only;
  already-published results are retrospective agreement, not prospective validation; (c) ALIVE Spring 2027 classes.
- **ALIVE Spring 2027 difficulty prediction (prospective):** before each lecture opens, the Rehearse-tab predictions
  (difficult slides, item difficulty) are committed with hash and date; after the lecture closes they are scored
  against the students' real learning difficulties with `scoreRehearsal` and the R370d permutation metric (caught
  vs chance expectation). Simulator knobs = the EEDI-selected cell, unchanged; consented students only (IRB 2347).

## 7. Integrity
- Held-out response files are quarantined: the held-out rows are written to a separate file readable only by the
  analysis step, every read is appended to an access log, and the scripts refuse `--side held` until lock.
- The OSF registration (timestamped) and the full code/environment hashes precede the first held-out read.
- Every arm, dataset and endpoint is reported whatever it shows; deviations go to §9 with a date.

## 8. To pin before lock (checklist)
- [ ] Exact prompts: persona, item, direct-rating baseline; ability-to-text mapping table.
- [ ] Calibration-only design simulation: detectable H1 difference and probability of H2 equivalence at the actual
      class/item counts (precision justification).
- [ ] Knob selection run (§2) and the chosen cell.
- [ ] H3 PPI++ specification (§5).
- [ ] Held-out quarantine file + access log in place; environment lock (vLLM, Python, numpy versions).
- [ ] Second independent review of v3.

## 9. Deviations and prior evidence (non-confirmatory, already known at writing)
- **2026-11-14 — pivot after development data (v2 → v3).** The knob run (calibration side only) was scored and
  inspected *before* the v3 scoring rules existed: score variance matched (log ratio −0.05), mean overpredicted
  (0.65 vs 0.51), item-difficulty ρ 0.07–0.11. Outside review (Rosalind, 4/10; `paper-ncomms/review/`) and these
  results motivated a boundary-study framing (H4 removed, H3 → future work, strong psychometric baselines, two
  estimands, three-outcome decision rule). The v2 hypotheses above stay verbatim as the record. Prompts and code used
  in that run are hashed in `knob-v2-prompts.sha256`. **Development stopping rule:** one further calibration-side
  round (arms A0–A3, τ ablation, permutation arm, δ offset; baselines B1–B5), no prompt rewording after scores are
  seen, then freeze. No held-out (H×H) answer has been read.
- ALIVE R370d (seeded personas, ChatGeV3): 27 / 41 real difficulty slides caught vs 21.7 chance, p = 0.050.
- ALIVE R390 (population calibrated on 6 consented students, ChatGeV3): 14 / 41 vs 12.4, p = 0.32; holdout 15 / 41
  vs 15.3, p = 0.61. Persona interviews: ordering held; 6 verdict leaks.
- ALIVE R391: Qwen3.5-27B 23 / 41 vs 19.3, p = 0.13; base Gemma-4 19 / 41 vs 17.6, p = 0.38. Interviews invalid
  (free-queue preemption); re-run 2026-11-12 with 0 failures: ordering held for both (average strengthsShare
  Qwen 0.40, Gemma 0.45 vs ChatGeV3 0.83); verdict leaks Qwen 25, Gemma 5 (`results/r391/README.md`).
- 2026-11-10 v1 → v2 after methods review: H3 moved to exploratory; confirmatory family narrowed to EEDI × Qwen;
  misconception prompting dropped; slip/guess/mixture knobs dropped; matched administration design added.
