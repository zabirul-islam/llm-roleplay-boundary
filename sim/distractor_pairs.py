#!/usr/bin/env python3
"""Distractor comparison on matched item sets (prereg v3.1 §5 "any comparison across methods uses a stated common
estimable item set"; Rosalind 2026-11-16 B2; council verdict review/council-2026-11-16-rosalind-paper.md).

v1.3 compared each arm's mean JS over its own estimable items with uniform guessing over all eligible items. This
script evaluates uniform guessing on the SAME items as each arm, per replicate: for arm a and replicate r the set is
score3.distractors_v31(...)["estimableItems"]; arm and uniform JS are both means over that set; values are averaged
over the five replicates. No new JS code: score3.js and score3.distractors_v31 are the registered implementations.

    python3 distractor_pairs.py --blocks B --frozen F --components C --runs RUNS --secondary S --out O
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score3  # noqa: E402
from secondary import Ctx  # noqa: E402

ARMS = (("A2 t0.6", "a2-t0.6"), ("A1 t0.8", "a1-t0.8"), ("A0 t0.8", "a0-t0.8"))


def uniform(key_q):
    return Counter({w: 1 for w in (1, 2, 3, 4) if w != key_q})


def main() -> int:
    ap = argparse.ArgumentParser()
    for x in ("--blocks", "--frozen", "--components", "--runs", "--secondary", "--out"):
        ap.add_argument(x, required=True)
    a = ap.parse_args()
    C = Ctx(a)
    R = C.real(C.prim)
    wr = defaultdict(Counter)
    for u, q, c in R:
        if c != C.key[q]:
            wr[q][c] += 1
    sec = json.load(open(a.secondary))["distractors"]
    out = {"rule": "per arm and replicate: arm JS and uniform-guessing JS over the arm's estimable items "
                   "(>= 1 simulated wrong answer); means over 5 replicates", "arms": {}}
    for name, arm in ARMS:
        per = []
        for rows in C.sim(arm):
            d = score3.distractors_v31(rows, R, C.key)
            est = d["estimableItems"]
            uni = float(np.mean([score3.js(uniform(C.key[q]), wr[q]) for q in est]))
            per.append({"eligible": d["eligible"], "estimable": len(est), "armJS": d["wrongJS"], "uniformJS": uni})
        armJS = float(np.mean([p["armJS"] for p in per]))
        # reproduces the registered per-arm value in secondary.json (same rows, same function)
        assert abs(armJS - sec[name]["wrongJS"]) < 1e-12, (name, armJS, sec[name]["wrongJS"])
        assert [p["estimable"] for p in per] == sec[name]["estimable"], name
        out["arms"][name] = {"eligible": per[0]["eligible"], "estimable": [p["estimable"] for p in per],
                             "armJS": armJS, "uniformJS": float(np.mean([p["uniformJS"] for p in per])),
                             "armJS_r": [p["armJS"] for p in per], "uniformJS_r": [p["uniformJS"] for p in per],
                             "armMinusUniform_r": [p["armJS"] - p["uniformJS"] for p in per]}
    out["uniformAllEligible (v1.3 reference)"] = sec["uniform guessing"]
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in out["arms"].items():
        print(k, "cov", v["estimable"], "arm %.4f uniform %.4f" % (v["armJS"], v["uniformJS"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
