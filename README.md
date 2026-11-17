# llm-roleplay-boundary

Code, prompts, registration documents and aggregate results for a pre-registered study of whether large language
models role-playing students rank test-item difficulty better than the same model rating each item directly
(manuscript under review; Md Zabirul Islam and Ge Wang, Rensselaer Polytechnic Institute).

- Registration: Open Science Framework, [osf.io/ygvdf](https://osf.io/ygvdf) (`prereg/prereg.md`, addenda for DBE-KT22
  and FoundationalASSIST in `prereg/addendum-*.md`; SHA-256 locks in `prereg/*.sha256`).
- Confirmatory test: EEDI H1 only. DBE-KT22 and FoundationalASSIST are descriptive by rules written before their
  held-out responses were read.

## Layout
| Path | Contents |
|---|---|
| `sim/` | Splitting and sealing (`heldout.py`, `assist_seal.py`), request building and the simulation client (`sim.py`), calibration fits and baselines (`calib.py`, `audit.py`), the locked inference (`analyze.py`), registered analyses (`hh.py`, `secondary.py`, `dbe.py`, `dbe_prep.py`, `assist_prep.py`), design validation (`design.py`), post-lock checks (`explore_postlock.py`, `distractor_pairs.py`) |
| `prereg/` | Registration, addenda, hash locks, environment lock |
| `results/` | Aggregate result files (JSON) and summaries; no row-level student data |
| `supplementary/Supplementary_Data_1.csv` | Replicate-level values reported in the paper |
| `make_numbers.py` | Generates every number in the manuscript from `results/` (and Supplementary Data 1) |
| `figures/` | Figure code (`make_figs.py`) and the editable Fig. 1 source (`src/fig1_protocol.drawio`) |
| `data/` | Split script, file hashes (`MANIFEST.md`) and the data-access audit (`AUDIT.md`) |

## Data
No dataset is redistributed here. Obtain each from its provider and verify it against the hashes in `data/MANIFEST.md`.
- **EEDI** text release (He-Yueya et al., arXiv 2407.15645), CC BY-NC-ND 4.0.
- **DBE-KT22** (Abdelrahman et al., arXiv 2208.12651; Australian Data Archive doi:10.26193/6DZWOH).
- **FoundationalASSIST** (Worden et al., arXiv 2602.00070; Hugging Face `ASSISTments/FoundationalASSIST`), gated,
  CC BY-NC 4.0 and the ASSISTments Responsible Use Guidelines.

## Reproducing
1. Split and seal each dataset (`sim/heldout.py`, `sim/assist_seal.py`, `sim/dbe_prep.py`); hold-out reads require the
   lock files in `prereg/`.
2. Fit calibration-side models and freeze artifacts (`sim/calib.py`, `sim/audit.py`); check hashes against `prereg/`.
3. Generate simulated answers with vLLM 0.31.0 and Qwen3.5-27B (`sim/sim.py`; serving arguments in
   `prereg/environment.lock`).
4. Run the registered analyses (`sim/hh.py`, `sim/secondary.py`, `sim/dbe.py`, `sim/assist_prep.py analyze`).
5. `python3 make_numbers.py` and `python3 figures/make_figs.py` rebuild the reported numbers, Supplementary Data 1 and
   the data figures from `results/`.

## Licence
Code: MIT (`LICENSE`). Result files are derived aggregates released under CC BY 4.0; the source datasets keep their own
licences.
