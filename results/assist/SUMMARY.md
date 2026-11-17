# FoundationalASSIST secondary replication (addendum A2, descriptive) — 2026-11-16

Registration: osf.io/ygvdf, update approved 2026-11-16T18:25:13Z (addendum lock `prereg/addendum-assist.sha256`,
repository commit 974c488). ASSIST hh was first read after the update: five request builds and one analysis on axis2,
each logged as a locked read. Qwen3.5-27B, replicates r1–r5, 15 arm-replicate runs, each 24,284/24,284 answered
(121,420 answers per arm); one invalid answer in total (A1, a two-option item), scored incorrect.
Held-out responses: 27,999 student–problem pairs after the response rule (779 duplicate rows, 425 rows with an
invalid score); 999 of the 1,000 held-out students answered at least one held-out eligible item.

**Items.** Of the 262 primary items, 0 lack a parsed B4 rating and 13 have no retained human response; the
zero-response guard (A2 §3) excludes and lists them (`itemsNoRetainedResponse` in `results-assist.json`), leaving
**249 items**. All-held-out set: 284 − 16 = 268 items.

**Descriptive only.** The design-validation criterion was not met under the 2PL generator (A2 §3), so no p-value, no
interval and no test of the 0.10 margin is reported. Monte Carlo SE covers generation-replicate variability only; it
does not include uncertainty from sampling students, items or templates.

## Primary contrast and comparisons (249 items)
| Contrast | ρ arm (mean of 5) | ρ comparator | Δρ | Δρ per replicate | MC SE (generation only) |
|---|---|---|---|---|---|
| A2 τ 0.6 vs B4 (registered H1 contrast) | 0.222 | 0.228 | −0.006 | −0.034, +0.006, −0.019, −0.021, +0.036 | 0.012 |
| E-pop A1 τ 0.8 vs B4 | 0.211 | 0.228 | −0.017 | +0.047, −0.084, −0.015, −0.000, −0.034 | 0.021 |
| A2 vs B1 (anchor θ) | 0.222 | 0.383 | −0.161 | −0.189 to −0.119 | 0.012 |
| A2 vs B2 (embedding ridge) | 0.222 | 0.173 | +0.049 | +0.021 to +0.091 | 0.012 |
| A2 vs B3 (TF-IDF ridge) | 0.222 | 0.167 | +0.055 | +0.027 to +0.097 | 0.012 |

## Difficulty metrics (249 items; arms averaged over replicates)
| Predictor | ρ | MAE | Bias | Calibration slope |
|---|---|---|---|---|
| A2 τ 0.6 | 0.222 | 0.282 | +0.040 | 0.114 |
| A0 τ 0.8 | 0.240 | 0.273 | +0.047 | 0.133 |
| A1 τ 0.8 | 0.211 | 0.264 | +0.022 | 0.112 |
| B1 anchor | 0.383 | 0.148 | −0.016 | 0.835 |
| B2 | 0.173 | 0.166 | +0.027 | 0.210 |
| B3 | 0.167 | 0.169 | −0.045 | 0.558 |
| B4 | 0.228 | 0.196 | −0.111 | 0.728 |

## Sensitivity (A2 vs B4; each alone)
| Set | Items | ρ A2 | ρ B4 | Δρ | MC SE |
|---|---|---|---|---|---|
| All held-out eligible items | 268 | 0.218 | 0.229 | −0.011 | 0.011 |
| Option-match correctness | 249 | −0.011 | 0.091 | −0.102 | 0.012 |
| Three- and four-option items only | 96 | 0.251 | 0.040 | +0.211 | 0.015 |

**Option-match caveat (exploratory, `explore-optionmatch.json`).** On the 22,476 retained primary-item rows, the
registered exact-text rule agrees with `discrete_score` on 81.2%. Disagreement is one-sided: 4,199 rows scored correct
by the platform do not match the key text, against 31 the other way, and 5,538 answer texts do not occur literally in
the stored options field. The option-match row therefore reflects text-encoding differences between `answer_text` and
the stored options at least as much as answer content; it is reported as registered and not re-specified.

## Cohort variance ratio (simulated / real student-score variance; 999 held-out students, primary items)
| Arm | Ratio per replicate | Mean |
|---|---|---|
| A2 τ 0.6 | 0.60, 0.63, 0.61, 0.61, 0.58 | 0.61 |
| A0 τ 0.8 | 0.61, 0.68, 0.64, 0.64, 0.63 | 0.64 |
| A1 τ 0.8 | 0.61, 0.69, 0.70, 0.64, 0.69 | 0.67 |

## Reading (descriptive)
On ASSIST, role-play and direct rating rank the 249 items about equally (ρ 0.222 vs 0.228; Δρ −0.006, replicate range
−0.034 to +0.036). This matches EEDI (Δρ +0.005) and differs from DBE, where role-play did not rank items at all.
All predictors are weak on this item set; the anchored-IRT baseline is highest (0.383). Simulated cohorts carry about
60–67% of the real between-student score variance. Role-play is far less calibrated than direct rating (slope 0.11 vs
0.73) at similar bias. The two contrasts sensitive to item subset move in opposite directions (three-/four-option
items: +0.211; option-match scoring: −0.102, see caveat), so the near-zero primary contrast is not uniform across item
types. No inferential claim is attached.

Files: `results-assist.json` (SHA-256 fc7070d7…; axis2 `runs/assist/results-assist.json`), `explore-optionmatch.json`.
