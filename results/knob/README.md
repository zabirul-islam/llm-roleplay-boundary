# Knob selection run (prereg §2) — calibration side only, Qwen3.5-27B, 2026-11-12/14

9 cells × 45,044 requests (5 grouped folds, matched administration), Empire AI Alpha free burst; scored 2026-11-14
with `knob.py score`. 0 invalid outputs in every cell. Full per-fold numbers: `knob-result-qwen3.5-27b.json`.

| Cell (σθ, τ) | mean L | MSE p | JS | log var ratio | adherence slope / intercept |
|---|---:|---:|---:|---:|---|
| 0.6, 0.6 | 0.477 | 0.184 | 0.269 | −0.553 | 0.81 / 0.33 |
| 0.6, 0.8 | 0.428 | 0.163 | 0.247 | −0.521 | 0.86 / 0.27 |
| 0.6, 1.0 | 0.320 | 0.133 | 0.213 | −0.387 | 0.86 / 0.22 |
| 1.0, 0.6 | 0.291 | 0.161 | 0.246 | +0.010 | 0.85 / 0.27 |
| **1.0, 0.8 (chosen)** | 0.249 | 0.136 | 0.215 | −0.050 | 0.86 / 0.23 |
| 1.0, 1.0 (lowest) | 0.236 | 0.131 | 0.197 | −0.025 | 0.86 / 0.21 |
| 1.4, 0.6 | 0.313 | 0.148 | 0.233 | +0.294 | 0.83 / 0.25 |
| 1.4, 0.8 | 0.260 | 0.129 | 0.199 | +0.239 | 0.82 / 0.24 |
| 1.4, 1.0 | 0.253 | 0.127 | 0.197 | +0.214 | 0.81 / 0.21 |

Chosen by the frozen rule: lowest = (1.0, 1.0) with SE 0.026; (1.0, 0.8) lies within 1 SE and has the smaller τ.

## What the calibration side already shows (descriptive, pooled over the 5 folds; rows = the selected cell + three others, not per-metric best)
| Cell | mean proportion correct, simulated | real | Spearman ρ, item difficulty sim vs real |
|---|---:|---:|---:|
| 1.0, 0.8 | 0.647 | 0.513 | 0.11 |
| 1.0, 1.0 | 0.619 | 0.513 | 0.07 |
| 0.6, 0.8 | 0.684 | 0.513 | 0.07 |
| 1.4, 0.8 | 0.623 | 0.513 | 0.09 |

- **Score variance is matched at σθ = 1.0** (log variance ratio ≈ 0, under the observed administration). The mean
  is not: simulated students overpredict accuracy by ~14 points, and matched variance alone is not evidence of
  realistic student differences.
- **Item difficulty is barely reproduced** (ρ ≈ 0.07–0.11): whether a simulated student gets an item right follows its
  declared rate (adherence slope ≈ 0.86) far more than the item's real difficulty. With this simulator, H1 (beat the
  text-regression and direct-rating baselines) is unlikely to hold.
- Simulated students are too accurate (0.62–0.68 vs 0.51; a persona told X % scores ≈ 0.23 + 0.86·X).
- This is calibration data, so the design may still change before the lock; any change goes into the prereg first.
