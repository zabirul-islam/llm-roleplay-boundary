# Confirmatory result, block hh (prereg v3.1, osf.io/ygvdf) — 2026-11-15

Run: `sim/hh.py` → locked `analyze.core` (sha256 524c6ac8… = lock), B = 10,000, seed 391. Qwen3.5-27B, replicates
r = 1–5, 621 students, 30 home classes, 27,674 cells; all 15 arm-replicate runs complete (27,674/27,674 answered).
Primary items: 134 registered; 4 had no observed hh cell (QuestionId 499, 608, 757, 847) and are non-estimable and
excluded (prereg §5), leaving 130 items. No non-estimable bootstrap draw.

## H1 (sole confirmatory test)
| | Value |
|---|---|
| ρ, A2 τ 0.6 per replicate | 0.341, 0.386, 0.417, 0.422, 0.421 (mean 0.397) |
| ρ, B4 (direct rating + isotonic) | 0.392 |
| Δρ | **+0.005** |
| p (H0: Δρ ≥ 0.10) | **0.191** |
| Decision at α = 0.05 | **not rejected** — the exclusion question is inconclusive |
| 95% basic interval for Δρ | [−0.208, 0.215] |
| Monte Carlo SE / bootstrap SE | 0.016 / 0.107 |

An advantage of 0.10 or more for role-play is not excluded. The point estimate is near zero; the interval is wide.

## H2 (descriptive; no test, no interval)
All 30 classes eligible; no zero variance and no undefined correlation in any class, arm or replicate (n = 30 each).
| Per replicate (r1–r5) | A2 τ 0.6 (selected) | A0 τ 0.8 (control) |
|---|---|---|
| Median log variance ratio (sim/real) | −0.615, −0.633, −0.571, −0.672, −0.788 | −0.877, −0.917, −0.882, −0.702, −0.826 |
| Median paired consistency difference (r_sim − r_real) | −0.240, −0.209, −0.085, −0.133, −0.161 | −0.175, −0.170, −0.201, −0.200, −0.195 |

Both arms under-disperse: simulated class score variance is about 0.46–0.57 (A2) and 0.40–0.50 (A0) of the real
variance, and simulated split-half consistency is lower than the real one in every replicate.

Full output: `results/hh/h1.json` (aggregate statistics only; no row-level data).

# Secondary analyses (prereg v3.1 §9) — 2026-11-15, `sim/secondary.py`, B = 10,000, seed 391
Post-lock code; every result is reported whatever it shows; none is confirmatory. All GPU runs complete (Qwen and
Gemma, r1–5; Gemma ratings 573/573). Full output: `results/hh/secondary.json`.

## Item difficulty, 130 items (mean over replicates for simulator arms)
| Predictor | ρ | MAE | Bias (pred − obs) | Calibration slope |
|---|---|---|---|---|
| A2 τ 0.6 (role-play, anchors) | 0.397 | 0.280 | +0.097 | 0.21 |
| A1 τ 0.8 (population draws) | 0.354 | 0.234 | +0.037 | 0.23 |
| A0 τ 0.8 (fixed level) | 0.333 | 0.302 | +0.118 | 0.17 |
| B4 direct rating + isotonic | 0.392 | 0.158 | −0.095 | 1.10 |
| B1 text 2PL, anchor θ | 0.307 | 0.148 | +0.001 | 0.61 |
| B1 text 2PL, population θ | 0.247 | 0.163 | −0.077 | 0.62 |
| B2 embedding ridge | 0.242 | 0.162 | −0.065 | 0.66 |
| B3 TF-IDF ridge | 0.204 | 0.156 | −0.040 | 0.66 |
Role-play ranks items about as well as B4 but its proportions are compressed: calibration slope 0.17–0.23 against 1.10
for B4, and MAE 0.23–0.30 against 0.15–0.16 for the baselines.

## Δρ, role-play A2 against each baseline (95% basic interval; p for Δ > 0)
| Baseline | Δρ | Interval | p (Δ > 0) |
|---|---|---|---|
| B1 anchor | +0.090 | [−0.135, 0.312] | 0.21 |
| B2 | +0.155 | [−0.091, 0.402] | 0.10 |
| B3 | +0.193 | [−0.045, 0.436] | 0.054 |
| B4 | +0.005 | [−0.208, 0.215] | 0.47 |
| "Beats all" (min Δ over B1–B4) | +0.005 | [−0.174, 0.221] | 0.41 |
H1 at other thresholds (descriptive, O1): p = 0.347 at 0.05, 0.089 at 0.15; the registered decision (0.10) is unchanged.

## Other contrasts against B4 (Δρ, interval, p for H0 Δ ≥ 0.10)
| Analysis | ρ arm | ρ B4 | Δρ | Interval | p(0.10) | Items |
|---|---|---|---|---|---|---|
| E-pop A1 τ 0.8 | 0.354 | 0.392 | −0.038 | [−0.252, 0.168] | 0.098 | 130 |
| τ ablation, A2 τ 0.8 | 0.375 | 0.392 | −0.017 | [−0.228, 0.192] | 0.138 | 130 |
| Anchor budget k = 5 | 0.395 | 0.392 | +0.003 | [−0.217, 0.219] | 0.195 | 130 |
| Anchor budget k = 10 | 0.389 | 0.392 | −0.003 | [−0.211, 0.203] | 0.168 | 130 |
| Anchor budget k = 20 | 0.379 | 0.392 | −0.013 | [−0.226, 0.196] | 0.148 | 130 |
| (all anchors = primary) | 0.397 | 0.392 | +0.005 | [−0.208, 0.215] | 0.191 | 130 |
| No-figure items (13 removed) | 0.407 | 0.440 | −0.032 | [−0.254, 0.186] | 0.122 | 117 |
| Home-group rows only (26,009 of 27,674) | 0.381 | 0.402 | −0.021 | [−0.228, 0.182] | 0.128 | 129 |
| All held-out items (168 observed of 172) | 0.336 | 0.387 | −0.051 | [−0.233, 0.132] | 0.055 | 168 |
| Gemma A2 vs Gemma's own B4 | 0.272 | 0.297 | −0.025 | [−0.258, 0.209] | 0.147 | 126 |
No sensitivity analysis changes the picture: Δρ stays between −0.05 and +0.01 and no interval excludes 0.10.

## Gemma-4-31B (secondary model; B4 = Gemma's own rating with an isotonic map fitted on cc by the registered rule)
ρ: A2 0.265, A0 0.195, A1 0.157, Gemma B4 0.297 (MAE 0.25, 0.31, 0.26, 0.17).

## Within-class ρ (classes with ≥ 5 items each answered by ≥ 3 of its students; 30 classes in every replicate)
A2 0.193, A0 0.199, A1 0.198.

## Persona adherence (declared accuracy in the prompt vs achieved simulated accuracy per student; 3,105 student-replicates)
A2: r = 0.71, slope 0.54 (unrounded 0.5446; corrected from 0.55 on 2026-11-16, Rosalind paper review), achieved − declared = +0.12. A1: r = 0.81, slope 0.77, +0.11.

## Distractors (111 items with ≥ 5 real wrong answers; JS base 2, lower is closer)
| | Estimable items (r1–r5) | Wrong-answer JS | Modal wrong answer agrees |
|---|---|---|---|
| A2 τ 0.6 | 108–110 | 0.269 | 0.40 |
| A0 τ 0.8 | 99–102 | 0.295 | 0.44 |
| A1 τ 0.8 | 110–111 | 0.198 | 0.40 |
| Uniform guessing over the three wrong options | 111 | 0.043 | n/a |
Every simulator arm is further from the real wrong-answer distribution than uniform guessing.

## Option permutation (contamination sensitivity check, not a certificate)
| Arm (τ 0.8) | ρ original order | ρ permuted | Accuracy original | Accuracy permuted |
|---|---|---|---|---|
| A2 | 0.375 | 0.343 | 0.631 | 0.620 |
| A1 | 0.354 | 0.284 | 0.603 | 0.595 |
Shuffling the options lowers ρ by 0.03–0.07 with accuracy nearly unchanged.

## Not yet run
DBE-KT22 replication (registered secondary dataset).
