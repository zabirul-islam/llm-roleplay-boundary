# Addendum A2 to prereg v3.1 (osf.io/ygvdf): FoundationalASSIST replication of H1 (v1, for review; descriptive after validation)

## 0. Scope, timing and information boundary
This addendum registers a descriptive secondary replication of the H1 contrast on a new public dataset. It was specified after the EEDI confirmatory and
secondary results and the DBE-KT22 descriptive results were seen (commits fa9b528, 8ca1ea0, 0592dcc), and before
analytical access to any FoundationalASSIST held-out response. EEDI's inconclusive H1 (Δρ +0.005, 95% interval
−0.21 to 0.21) motivated it: the EEDI test had 130 items; this dataset supplies 262. The test is reported on its own
and is not pooled with EEDI. It retains the selected arm definitions and hyperparameters, with the dataset-specific
adaptations documented below.

- **Dataset.** Worden, Heffernan, Heffernan & Sonkar (2026), FoundationalASSIST (AIED 2026; arXiv 2602.00070),
  Hugging Face `ASSISTments/FoundationalASSIST`, gated, CC BY-NC 4.0 plus the ASSISTments Responsible Use Guidelines.
  Access granted 2026-11-16 to the first author's RPI-verified account. Data are stored on axis2 (RPI) in owner-only
  directories, never in git; only hashes and aggregates are committed (`data/MANIFEST.md`, `data/AUDIT.md`).
- **Access history.** Downloaded 2026-11-16T12:01 (hashes in `data/MANIFEST.md`). Before sealing, the study team read:
  Problems.csv and Skills.csv metadata (item counts, types, options); the header line of Interactions.csv; and, in the
  split script, every Interactions row's `user_id` (to list students). The seal (2026-11-16T12:04:39) routed rows by
  `user_id` and `problem_id`. No other response field was used analytically or included in retained outputs before
  sealing, and no manual inspection of responses took place. After sealing, the open blocks cc and hc were read for the
  calibration fits and the validation (§2, §3). No hh response has been read.
- **Sealing.** `sim/assist_seal.py`: held-out students = seeded 20% (1,000 of 5,000); held-out items = all problems in a
  seeded 60% of the problem sets that contain an eligible item (283 sets, 344 items), so no shared stem crosses the
  split. Rows: cc 1,284,618; hc 322,967; ch 116,596; hh 29,203. ch, hh and the raw Interactions.csv are mode 0600 in
  `sealed/`; reading hh requires this addendum's lock (`prereg/addendum-assist.sha256`); ch is never read. Split file
  SHA-256 63a6d2ec… (on axis2).

## 1. Items and responses
- **Eligible items.** Single-answer multiple choice, 2–4 options, no image in stem or options (`assist_seal.eligible`),
  and a key that resolves: the single Multiple Choice Answer must equal one stored option exactly (40 items fail and
  are excluded). 484 eligible items; 284 held out.
- **Item text** (`assist_prep.items`): the dataset authors' own `clean_problem_body` (Code/clean_utils.py; MathML to
  text such as (4/3) and x^2, tables to rows) applied to the stem and to each option; options in stored order,
  lettered A, B, C, D. Run in a separate preparation environment (beautifulsoup4 4.12.3, pandas 2.2.3); the simulation
  environment is unchanged. Output `items-assist.json` (SHA-256 0463e794…).
- **Item sets.** The registered near-duplicate rule, unchanged: held-out eligible items with TF-IDF cosine < 0.8 to
  every calibration item. It removes 22 of 284, leaving **262 primary items**. Templates for resampling = connected
  components joining items of the same problem set or with cosine ≥ 0.8: 241 templates (primary), 255 (all 284).
  Output `components-assist.json` (SHA-256 6b2a60ad…).
- **Response.** Within each student–problem pair, retain the row with the numerically smallest `id`, then exclude
  that row if `discrete_score` is missing or not a valid binary score (0 or 1, including the accepted representations
  `0.0` and `1.0`); do not substitute another row (counted: 62,269 duplicate rows and 5,191 excluded rows in cc;
  15,397 and 1,353 in hc). **Correct = `discrete_score`** (1 = correct on the first try without hints or seeing the
  answer); the primary human target is therefore platform-defined unaided first-try success, not answer content alone.
  Simulated correctness refers to the item's key.
- **Option-match sensitivity.** It uses these same retained rows. Its human score is 1 only when nonmissing
  `answer_text` equals the correct option's raw text after CSV decoding and leading/trailing whitespace removal from
  both strings using `.strip()`. Comparison is case-sensitive; no other normalization is applied. Missing or empty
  `answer_text` scores 0. Simulated answers retain the frozen option-key scoring. No fits are recomputed
  (`assist_prep.option_match`).
- **No classes.** The dataset has no class identifiers; no class-level analysis is run and none is formed.

## 2. What changes and what is carried over
| Element | ASSIST | Kind |
|---|---|---|
| System prompt | "You are a {age}-year-old middle-school student in the United States answering a math problem. {LEVEL} Answer as that student really would, not as an expert. Reply with only the letter of your answer." Age seeded per student and replicate from {11, 12, 13} | Domain substitution |
| LEVEL sentences | EEDI sentences with "maths" → "math"; A1/A2 "You usually get about {X}% of math questions like these right." | Domain substitution |
| B4 rating | system "You are an experienced middle-school math teacher in the United States."; question "What percentage of US students in grades 6 to 8 would answer this question correctly on their first try, without hints? Reply with only a number from 0 to 100." τ = 0, one request per item (3,368; 3,366 parsed) | Domain substitution |
| Answer parsing | `sim.parse_choice`, then the item's option range (2–4); invalid = incorrect | Method adaptation (as DBE) |
| Response rule, item text, item sets | §1 | Method adaptation |
| 2PL, anchors | Refit on ASSIST cc (4,000 students × 2,997 items, all item types; registered settings); anchor MAP from each held-out student's hc responses (1,000 anchored); θ = 0 without anchors | Refit on ASSIST |
| B1–B3 | Refit on ASSIST cc items (all types), BGE-M3 and TF-IDF on the cleaned text; registered ridge grid and CV (α: B2 1, B3 10, B1 b 1, B1 log a 1) | Refit on ASSIST |
| B4 isotonic map | Fitted on cc items with parsed ratings | Refit on ASSIST |
| Arms, comparator | A2 τ 0.6 (H1), A0 τ 0.8, A1 σθ 1.0 τ 0.8; B4. Qwen3.5-27B only; R = 5; per-request seed from "391-assist-r{r}-{arm}-{u}-{q}" | Carried over |
| Analysis code | `analyze.Prepared.stats` (locked) for point estimates; no bootstrap inference (§3) | Carried over; inference removed |

Frozen artifacts (axis2 `runs/assist/`, SHA-256): `frozen/item-predictions.json` 8a47ca15…, `frozen/irt-cc.json`
2af27949…, `frozen/anchors-hc.json` 080610e9…, `frozen/settings.json` 9bf6c3c8…, `rate-req.jsonl` b75a3fd0…,
`rate-qwen3.5-27b.jsonl` ac8250e6…, `emb-bge-m3.npz` 870ecdb8…. A second freeze run gave byte-identical files.
ρ(raw rating, calibration proportion) is 0.289 over all items and 0.172 over eligible items.

## 3. Analysis
- **H1 contrast on ASSIST (descriptive after validation; see below).** Δρ = mean over five replicates of ρ(A2) − ρ(B4) over the 262 primary items
  (equal item weight; item proportion = mean correctness among held-out students who answered it). H0: Δρ ≥ 0.10,
  one-sided α = 0.05, the locked `analyze.core` procedure, conditional on the evaluated cohort; templates as §1.
  Specified as a test conditional on the validation below; because the validation criterion was not met, only the
  descriptive quantities are reported.
- **Calibration-pattern validation diagnostic.** The run used 1,000 calibration students, 262 calibration items in
  251 templates, and 1,000 bootstrap draws per simulated dataset, rather than the registered 10,000. The held-out
  primary set contains 262 items in 241 templates. The diagnostic failed the prespecified criterion under the 2PL
  generator; ASSIST therefore remains descriptive only. These results do not establish validation at the exact
  held-out template structure. No bootstrap-convergence run was performed for ASSIST. Design (`assist_prep.py
  design`; criterion committed before the run): real cc missingness of the sampled calibration students and items,
  templates by the §1 rule, one cohort, R = 5, ρB = 0.172, κ = 0.15; scenario calibrated to Δρ = 0.10;
  1,000 cohorts per generator (2PL; misspecified 3PL with guessing 0.2, slip 0.05); truths from 400 independent item
  draws. **Criterion, per generator:** Wilson 95% upper limit of one-sided non-coverage ≤ 0.075, Wilson 95% lower
  limit of two-sided 95% interval coverage ≥ 0.925, and mean reference truth within 0.10 ± 0.02. If any part fails
  under either generator, the ASSIST results are descriptive only, as for DBE: no p-value and no interval.
- **Validation result** (`results/assist/design-assist.json`; reference-truth Monte Carlo SE 0.0037):

  | Generator | Mean truth | One-sided non-coverage | Two-sided 95% coverage | Criterion |
  |---|---|---|---|---|
  | 2PL | 0.099 | 60/1,000 = 0.060 [0.047, 0.077] | 936/1,000 = 0.936 [0.919, 0.950] | not met (upper 0.0765 > 0.075; lower 0.919 < 0.925) |
  | Misspecified | 0.105 | 42/1,000 = 0.042 [0.031, 0.056] | 954/1,000 = 0.954 [0.939, 0.965] | met |

  **The criterion is not met under the 2PL generator, so the ASSIST results are descriptive only.** The registered
  consequence applies: no ASSIST p-value, no interval and no test of the 0.10 margin are reported. The procedure is not
  re-tuned after this result. The descriptive reporting rules of addendum A1 §3 apply: point estimates, the five
  generation-replicate values and generation-only Monte Carlo SE, which does not quantify uncertainty from sampling
  students, items or templates.
- **Evaluation support.** A1's evaluation-support rules apply to ASSIST. Within each analysis set, items without a
  parsed B4 rating are excluded from every contrast and listed. Items 53422 and 53714 lack parsed ratings; neither is
  eligible or belongs to any held-out set. Items without a retained human hh response are non-estimable and listed.
  Each contrast uses identical item support for its constituent correlations; undefined correlations remain undefined.
  **Zero-response guard** (`assist_prep.responded`): before computing human item means or any human-referenced metric,
  items with zero retained human hh responses are excluded and listed, and that exclusion applies to every constituent
  of each contrast.
- **Secondary (descriptive, same rules as DBE):** point estimates, five replicate values and generation-only Monte
  Carlo SE for E-pop (A1 vs B4) and A2 vs B1–B3; ρ, MAE, bias and calibration slope for every arm and baseline; cohort
  variance ratio (simulated / real, per replicate then averaged); sensitivity sets, each applied alone: all 284
  held-out eligible items (no near-duplicate filter), option-match correctness, three- and four-option items only. No
  sensitivity analysis triggers refitting.

## 4. Before opening ASSIST hh
- [x] Download, hashes, licence record; split and seal (§0)
- [x] Item text, item sets, fits, B4 ratings frozen and hashed; freeze deterministic (§1, §2)
- [x] Validation (§3): criterion not met under the 2PL generator; ASSIST is descriptive
- [x] Review of this addendum (`review/rosalind-2026-11-16-assist.md`), verified by council
  (`review/council-2026-11-16-assist.md`); fixes applied
- [x] One closure check: "ready to freeze", all rows closed (`review/rosalind-2026-11-16-assist-closure.md`); selftests of
  `assist_prep.py`, `assist_seal.py`, `dbe_prep.py` and `analyze.py` pass on axis2, code identical to the repository
- [ ] `prereg/addendum-assist.sha256` committed; OSF registration update (hashes in the justification; files in the
  associated project). Held-out access is permitted only after both.

**Stopping rule.** One review, then one closure check limited to its listed issues. A new blocking issue must identify a
demonstrable factual error, an ambiguity capable of changing a reported result, or a claim unsupported by the
evidence, with location and consequence. Everything else is advisory. Then freeze, hash, deposit on OSF, and only then
access held-out responses.
