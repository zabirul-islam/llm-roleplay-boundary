# DBE-KT22 secondary replication (addendum A1, descriptive) — 2026-11-16

Registration: osf.io/ygvdf, update approved 2026-11-16T05:59:46Z (addendum lock 902a718). DBE hh read after the
update: one logged read on Empire (01:10:13 EST) to copy the block; on axis2, five request builds and one analysis,
all logged after the lock. Qwen3.5-27B, replicates r1–r5, 15 arm-replicate runs, all 17,706/17,706 answered.
402 held-out students with first attempts (385 with a primary-item attempt); 31 primary items in 30 templates.
First attempts: 17,706 pairs, no bad timestamps, no invalid first choice. Key consistency (hh first attempts):
17,706 comparable records, 0 disagreements, no missing states, no unassessable items.

**Descriptive only.** No interval and no p-value is reported. Monte Carlo SE covers generation-replicate variability
only; it does not include uncertainty from sampling students, items or templates.

## Primary contrast and comparisons (31 items)
| Contrast | ρ arm (mean of 5) | ρ comparator | Δρ | Δρ per replicate | MC SE (generation only) |
|---|---|---|---|---|---|
| A2 τ 0.6 vs B4 (primary) | 0.014 | 0.357 | −0.343 | −0.280, −0.396, −0.324, −0.350, −0.367 | 0.020 |
| E-pop A1 τ 0.8 vs B4 | 0.025 | 0.357 | −0.332 | −0.291, −0.373, −0.304, −0.369, −0.325 | 0.017 |
| A2 vs B1 (anchor θ) | 0.014 | 0.621 | −0.607 | −0.544 to −0.660 | 0.020 |
| A2 vs B2 (embedding ridge) | 0.014 | 0.576 | −0.563 | −0.499 to −0.615 | 0.020 |
| A2 vs B3 (TF-IDF ridge) | 0.014 | 0.671 | −0.658 | −0.594 to −0.710 | 0.020 |

## Difficulty metrics (31 items)
| Predictor | ρ | MAE | Bias | Calibration slope |
|---|---|---|---|---|
| A2 τ 0.6 | 0.014 | 0.312 | +0.061 | −0.015 |
| A0 τ 0.8 | 0.076 | 0.258 | −0.047 | 0.032 |
| A1 τ 0.8 | 0.025 | 0.305 | +0.051 | −0.015 |
| B1 anchor | 0.621 | 0.147 | +0.060 | 1.74 |
| B2 | 0.576 | 0.147 | +0.081 | 1.68 |
| B3 | 0.671 | 0.146 | +0.075 | 2.14 |
| B4 | 0.357 | 0.163 | +0.015 | 1.00 |
Invalid output: 0 for every arm and option count.

## Sensitivity (A2 vs B4; each filter alone on the primary set; all-64 separate)
| Set | Items | ρ A2 | ρ B4 | Δρ | MC SE |
|---|---|---|---|---|---|
| All held-out items | 64 | 0.043 | 0.341 | −0.299 | 0.009 |
| Without LaTeX | 28 | −0.015 | 0.303 | −0.317 | 0.026 |
| Without tables | 31 | 0.014 | 0.357 | −0.343 | 0.020 |
| Four-option only | 27 | 0.064 | 0.394 | −0.329 | 0.024 |
| Key-consistent only | 31 | 0.014 | 0.357 | −0.343 | 0.020 |
"Without tables" and "key-consistent only" equal the primary set: no primary item contains a table, and no held-out
item showed key disagreement.

## Cohort variance ratio (simulated / real student-score variance; all held-out students, not classes)
| Set, arm | Students | Ratio per replicate | Mean |
|---|---|---|---|
| Primary, A2 | 385 | 0.52, 0.54, 0.51, 0.53, 0.59 | 0.54 |
| Primary, A0 | 385 | 1.42, 1.58, 1.53, 1.59, 1.49 | 1.53 |
| Primary, A1 | 385 | 0.56, 0.68, 0.48, 0.53, 0.56 | 0.56 |
| All held-out, A2 | 402 | 0.49, 0.55, 0.49, 0.57, 0.54 | 0.53 |

## Reading (descriptive)
On DBE the role-play arms do not rank the 31 items (ρ 0.01–0.08), while direct rating (0.36) and the text
baselines (0.58–0.67) do. Every replicate and every sensitivity set shows the same direction. This differs from EEDI,
where role-play matched direct rating (Δρ +0.005). No inferential claim is attached; with 31 items the sampling
uncertainty is not quantified.
