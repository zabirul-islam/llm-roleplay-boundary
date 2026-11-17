# Post-lock exploratory checks (not registered) — 2026-11-16
Requested by the paper council (`council/verdict-2026-11-16-draft.md`). Code: `sim/explore_postlock.py`; output:
`explore-postlock.json`. None of this is confirmatory; the registered results are unchanged.

1. **Role-play and direct rating rank items differently (EEDI).** Item-level Spearman between A2 (mean of five
   replicates) and B4: 0.21, while each correlates about 0.4 with real difficulty (A2 mean 0.41). The idea that
   role-play only "echoes" the model's direct opinion is not supported.
2. **Wrong answers (EEDI, 110 items with ≥ 5 real wrong answers).** Real wrong answers are close to uniform over the
   three wrong options (median JS to uniform 0.024; 75% of items below 0.05; mean share of the most common wrong
   option 0.50). Simulated wrong answers concentrate on one option (mean top share 0.79; median JS to real 0.20).
   Wording for the paper: role-play concentrates errors that real students spread.
3. **DBE audit (31 primary items).** No pipeline bug: replies parse, keys match real students' answers (hh key
   consistency 0 disagreements), and mean simulated accuracy is 0.78 against 0.71 real. Two mechanisms:
   - **Ceiling.** Simulated accuracy is ≥ 0.99 on 55% of items (68% once context-dependent items are removed)
     regardless of the persona's declared level, including items real students find hard (e.g. item 179: real
     0.22, simulated 1.00; item 189: 0.33 vs 1.00).
   - **Missing context.** Six items (71, 75, 81, 147, 160, 161) ask how many rows a query returns on course tables
     whose contents are not in the item text; real students had the course database. Simulated accuracy on them is
     unrelated to real accuracy (e.g. item 147: real 0.90, simulated 0.00).
   - Excluding the six: ρ(role-play, real) 0.11 vs ρ(B4, real) 0.36 (25 items). The ceiling, not missing context,
     carries most of the gap.
   - Answer-letter bias: across all A2 replies, A 33,525, B 31,299, C 15,973, D 7,733 (includes 2- and 3-option items).
