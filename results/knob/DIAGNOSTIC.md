# Knob run — diagnostic audit (prereg v3 Step 2b) and data audit (Step 2), 2026-11-14

Calibration side only (open blocks cc/hc + id/text files). Raw numbers: `diagnostic.json`, `audit-data.json`.
Reproduce: `python3 audit.py knob --blocks <dir> --out-dir <knob dir> --model qwen3.5-27b` and `audit.py data`.

## Is the weak result an artefact? No pipeline fault found
| Check | Result |
|---|---|
| Parser | 405,387 of 405,396 replies are a bare letter; 9 are "letter + text" (first letter taken correctly); 0 invalid |
| Ids | request ↔ answer join 1:1 in every cell; 0 duplicates, 0 missing, 0 extra |
| Answer key | keys agree with hand-solved items 0, 130, 199, 285, 552, 773, 847; items 43 and 860 have rows with conflicting CorrectAnswer → majority rule (both calibration items) |
| Options in prompt | every item shows its four options (three formats: `A)` lines, `A.` lines, inline) |
| Ability → % | pct orders personas identically across σθ (monotone in θ) |
| Repeated attempts | 0 repeated student × item rows in the open blocks |

## All nine cells (means over 5 folds; per-fold numbers in `diagnostic.json`)
Selected cell (frozen v2 rule) = σθ 1.0, τ 0.8. Per-metric best: ρ → (0.6, 1.0); MAE → (1.4, 1.0); loss L → (1.0, 1.0).

| σθ, τ | ρ | MAE | bias | MAE after isotonic | log var ratio | L |
|---|---:|---:|---:|---:|---:|---:|
| 0.6, 0.6 | 0.10 | 0.354 | +0.21 | 0.189 | −0.55 | 0.477 |
| 0.6, 0.8 | 0.11 | 0.324 | +0.17 | 0.187 | −0.52 | 0.428 |
| 0.6, 1.0 | **0.16** | 0.291 | +0.13 | 0.182 | −0.39 | 0.320 |
| 1.0, 0.6 | 0.09 | 0.321 | +0.17 | 0.187 | +0.01 | 0.291 |
| **1.0, 0.8 (selected)** | 0.14 | 0.288 | +0.14 | 0.183 | −0.05 | 0.249 |
| 1.0, 1.0 | 0.11 | 0.279 | +0.12 | 0.188 | −0.02 | **0.236** |
| 1.4, 0.6 | 0.08 | 0.304 | +0.14 | 0.189 | +0.29 | 0.313 |
| 1.4, 0.8 | 0.12 | 0.278 | +0.11 | 0.184 | +0.24 | 0.260 |
| 1.4, 1.0 | 0.06 | **0.271** | +0.09 | **0.178** | +0.21 | 0.253 |

- **No cell does well on difficulty.** Best mean ρ = 0.16; per fold ρ ranges −0.26 to +0.38, and fold 4 is negative in
  every cell. The selected cell did not trade away a good-ρ option: the grid has none.
- **Human reference:** split-half ρ of real item difficulty (each half ≈ half the fold's students, 200 splits) has
  median 0.71–0.86 across folds. The attainable ceiling is far above the simulator.
- **Mean bias is repairable, rank is not:** an isotonic map fitted on the other folds cuts MAE from ≈ 0.29 to ≈ 0.18.
  It cannot change ρ. Bias (+0.09 to +0.21) and item-order failure are separate problems.
- Score variance is matched at σθ = 1.0 under the observed administration. That alone is not evidence of realistic
  student differences.

## Data audit (`audit-data.json`)
- 2,287 students, 573 items. Dates span 2019-09-11 → 2020-04-29 (one school year).
- 924 students answered in more than one class (open blocks). The home class = the class with most answers.
- **Near-duplicate items:** 54 held-out × calibration pairs with TF-IDF cosine ≥ 0.8, covering 38 of the 172 held-out
  items. Several are templates that differ only in a number or variable (e.g. 541/549: "area n, scale 10" vs "area
  10, scale n", with different keys).
- **Figure-dependent items:** 40 items refer to a diagram/table/graph the text release does not contain (e.g. item
  46); 14 of them are held out.

## Rules frozen from this audit (go into prereg v3 before any further run)
1. Answer key = majority CorrectAnswer over an item's rows; a tie excludes the item (none at present).
2. **Confirmatory held-out item set** = held-out items with no calibration near-duplicate (cos ≥ 0.8): 172 − 38 = 134
   items. All 172 = sensitivity analysis. Templates as clusters in the bootstrap.
3. **Figure-dependent items** are kept in the primary set (every method, simulator or baseline, sees only text) and
   excluded in a sensitivity analysis. The regex is frozen in `audit.py`.
