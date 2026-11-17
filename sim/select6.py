#!/usr/bin/env python3
"""Apply prereg v3 §6 to the calibration-side scores (score3 output) and print the frozen choices.

    python3 select6.py ../results/calib-v3/calib-v3.json [--out choices.json]

Rule (§6): highest mean fold rho within the family; arms within 1 SE (fold SD / sqrt(folds)) of the best are
broken by fewer added parts (A2 < A2δ < A3; A1 < A1δ), then tau = 0.8, then higher rho.
Strongest baseline: highest mean fold rho among B1-B4 under the estimand (B1 uses that estimand's theta;
B4 = the isotonic-mapped rating, as defined in §4).
"""
import json
import math
import sys

PARTS = {"a2": 0, "a2d": 1, "a3": 2, "a1": 0, "a1d": 1}
BASE = {"anchor": ["B1_textIRT_anchor", "B2_embed", "B3_tfidf", "B4_rating_iso"],
        "pop": ["B1_textIRT_pop", "B2_embed", "B3_tfidf", "B4_rating_iso"]}


def stats(entry):
    r = [f["rho"] for f in entry["folds"] if f["rho"] == f["rho"]]
    m = sum(r) / len(r)
    sd = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1))
    return m, sd / math.sqrt(len(r))


def pick(arms, fam):
    c = {n: stats(e) for n, e in arms.items() if e.get("complete") and n.split("-")[0] in fam}
    best = max(c, key=lambda n: c[n][0])
    near = [n for n in c if c[n][0] >= c[best][0] - c[best][1]]
    tau = lambda n: n.rsplit("-t", 1)[1]
    choice = min(near, key=lambda n: (PARTS[n.split("-")[0]], tau(n) != "0.8", -c[n][0]))
    return {"choice": choice, "best": best, "bestRho": c[best][0], "bestSE": c[best][1], "withinSE": sorted(near),
            "all": {n: {"rho": round(v[0], 4), "se": round(v[1], 4)} for n, v in sorted(c.items(), key=lambda x: -x[1][0])}}


def main():
    d = json.load(open(sys.argv[1]))
    out = {"E-anchor": pick(d["arms"], ("a2", "a2d", "a3")), "E-pop": pick(d["arms"], ("a1", "a1d"))}
    for est, k in (("E-anchor", "anchor"), ("E-pop", "pop")):
        b = {n: stats(d["baselines"][n]) for n in BASE[k]}
        out[est]["baseline"] = max(b, key=lambda n: b[n][0])
        out[est]["baselines"] = {n: {"rho": round(v[0], 4), "se": round(v[1], 4)} for n, v in b.items()}
    s = json.dumps(out, indent=1)
    print(s)
    if "--out" in sys.argv:
        open(sys.argv[sys.argv.index("--out") + 1], "w").write(s + "\n")


if __name__ == "__main__":
    # ponytail: tie-break check only; the scoring itself is tested in score3 selftest
    f = lambda m: {"complete": True, "folds": [{"rho": m + e} for e in (-.06, -.03, 0, .03, .06)]}   # SE 0.021
    t = pick({"a3-t0.8": f(0.30), "a2-t0.6": f(0.295), "a2-t0.8": f(0.29), "a2d-t0.8": f(0.10)}, ("a2", "a2d", "a3"))
    assert t["choice"] == "a2-t0.8", t
    main()
