#!/usr/bin/env python3
"""Design validation for prereg v3.1 (Rosalind pass 3, B7): the FINAL confirmatory procedure (analyze.core: H1 alone,
alpha 0.05, no multiplicity adjustment) on semi-synthetic data. No held-out answer is read.

History (kept, not overwritten): design2-*.json = the v3 three-outcome rule (code at 69e80d8); design3.json = the
H1 + H2 + Holm procedure, which failed for H2 (H2a 0.22 at its boundary, Holm FWER 0.068; code at a91f725). After
that result the owner made H2 descriptive (2026-11-15, before any hh access); this file now validates H1 only.

Pattern: the real cc missingness of the 30 largest home classes x 134 random calibration items, their TF-IDF
template map (cos >= 0.8), the frozen half assignment, R = 5. Estimand (owner rule 2026-11-14): conditional on the
cohort. A "world" = one cohort (abilities fixed) + an item population; an evaluation draws 134 items (templates
kept, pair members share difficulty r = 0.9). The truth of each statistic for that cohort = its mean over 40 fresh
item draws. Generators: "2pl" and "misspec" (3PL guess 0.2, slip 0.05, class effect SD 0.4) for the REAL answers;
simulated students are 2PL with item difficulty from the simulator prediction:
    real p ranks by -b;  B4 = r_b(-z) + e_b;  simulator item logit = r_a(-z) + e_a;  corr(e_a, e_b) = KAPPA (cc: 0.145)
    simulated ability = g * theta + tau * eta_(student, half)   (g: spread; tau: lowers split-half consistency)
Scenario parameters are found by bisection on the truth (calibrate), then each scenario runs N sims through
analyze.core with B draws; reported: H1 rejection rate (size where the truth is in the null), coverage of the basic
interval for delta, non-estimable frequency, each with a Wilson 95% interval.

    python3 design.py run --blocks B --out design3.json --sims 500 --draws 1000 --procs 96
    python3 design.py converge --blocks B --out converge.json --sims 50       # B = 1,000 vs 10,000 on the same data
    python3 design.py boundary --blocks B --params design4-h1.json --out design5-boundary.json --sims 500 --procs 90
    python3 design.py selftest

boundary (Rosalind pass 4): one-sided upper-bound coverage validation near delta = 0.10 with accurately known
cohort-conditional truths (400 independent reference item draws per cohort, own rng stream). H1 rejects at c iff the
one-sided 95% upper bound is below c, so non-coverage P(U < truth) near 0.10 is size at the boundary. The H1
algorithm (analyze.core) is unchanged; keep_draws only returns its bootstrap draws.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402

KAPPA, R, RHO_B, TRUTH_DRAWS = 0.15, 5, 0.434, 40


def pattern(blocks: str, n_classes=30, n_items=134, seed=391):
    from knob import load
    from audit import tfidf, _components
    rows, key, home, text = load(blocks)
    size = Counter(home[u] for u in {u for u, _, _ in rows})
    classes = [g for g, _ in size.most_common(n_classes)]
    items = random.Random(seed).sample(sorted(key, key=lambda x: (len(x), x)), n_items)
    cs, its = set(classes), set(items)
    users = sorted({u for u, q, _ in rows if home[u] in cs and q in its})
    ui, ii = {u: i for i, u in enumerate(users)}, {q: i for i, q in enumerate(items)}
    M = np.zeros((len(users), len(items)), bool)
    for u, q, _ in rows:
        if u in ui and q in ii:
            M[ui[u], ii[q]] = True
    cls = np.array([classes.index(home[u]) for u in users])
    X = tfidf([text[q] for q in items]); Sm = X @ X.T
    comps = _components([(items[i], items[j]) for i in range(n_items) for j in range(i + 1, n_items) if Sm[i, j] >= 0.8], items)
    tid = {q: t for t, c in enumerate(comps) for q in c}
    tmpl = np.array([tid[q] for q in items])
    return {"M": M, "cls": cls, "tmpl": tmpl, "half": analyze.halves(M, cls, tmpl)}


class World:
    def __init__(self, pat, kind, rng):
        self.p, self.kind, self.rng = pat, kind, rng
        S = pat["M"].shape[0]
        self.th = rng.normal(0, 1, S)
        if kind == "misspec":
            self.th = self.th + rng.normal(0, 0.4, int(pat["cls"].max()) + 1)[pat["cls"]]
        self.eta = rng.normal(0, 1, (S, 2))

    def draw(self, par, rng):
        """One evaluation: 134 items (template structure kept) -> analyze.core input."""
        p, I = self.p, len(self.p["tmpl"])
        z = rng.normal(0, 1, I)
        for t in np.unique(p["tmpl"]):                    # template members share difficulty (r = 0.9)
            m = np.where(p["tmpl"] == t)[0]
            if len(m) > 1:
                z[m[1:]] = 0.9 * z[m[0]] + math.sqrt(1 - 0.81) * z[m[1:]]
        a = np.exp(rng.normal(0, 0.3, I))
        e = rng.multivariate_normal([0, 0], [[1, KAPPA], [KAPPA, 1]], I)
        ra, rb = par["ra"], par["rb"]
        sim_b = -(ra * -z + math.sqrt(1 - ra * ra) * e[:, 0])          # simulator item difficulty (sd 1)
        base = rb * -z + math.sqrt(1 - rb * rb) * e[:, 1]
        P = 1 / (1 + np.exp(-a * (self.th[:, None] - z)))
        if self.kind == "misspec":
            P = (0.2 + 0.8 * P) * 0.95
        real = (rng.random(P.shape) < P).astype(float)
        hh = p["half"][p["cls"]]                                        # S x I, 1 = half one
        def sim(g, tau):
            th = g * self.th[:, None] + tau * np.where(hh == 1, self.eta[:, :1], self.eta[:, 1:])
            Q = 1 / (1 + np.exp(-a * (th - sim_b)))
            return (rng.random((R,) + Q.shape) < Q).astype(float)
        return {"obs": p["M"], "real": real, "base": base, "cls": p["cls"], "tmpl": p["tmpl"], "half": p["half"],
                "sims": {"S": sim(par["gS"], par["tauS"]), "A0": sim(par["g0"], 0.0)}}


def truth(world, par, rng, n=TRUTH_DRAWS) -> dict:
    vals = {k: [] for k in ("delta", "rhoB")}
    for _ in range(n):
        P = analyze.Prepared(world.draw(par, rng))
        s = P.stats(np.ones(P.obs.shape[1], int))
        for k in vals:
            vals[k].append(s[k])
    return {k: float(np.nanmean(v)) for k, v in vals.items()}


def mean_truth(pat, kind, par, worlds=6, seed=7) -> dict:
    rng = np.random.default_rng(seed)
    ts = [truth(World(pat, kind, rng), par, rng, 12) for _ in range(worlds)]
    return {k: float(np.mean([t[k] for t in ts])) for k in ts[0]}


def bisect(f, lo, hi, target, it=10):
    """f increasing in its argument."""
    for _ in range(it):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def calibrate(pat, kind) -> dict:
    base = {"ra": 0.4, "rb": 0.5, "gS": 1.0, "tauS": 0.0, "g0": 0.5}
    mt = lambda **kw: mean_truth(pat, kind, {**base, **kw})  # noqa: E731
    base["rb"] = bisect(lambda x: mt(rb=x)["rhoB"], 0.1, 0.99, RHO_B)
    sc = {}
    for d in (0.10, 0.0, -0.237):
        sc[f"H1 delta={d}"] = {**base, "ra": bisect(lambda x: mt(ra=x)["delta"], 0.02, 0.99, d)}
    return {k: {"par": v, "meanTruth": mt(**v)} for k, v in sc.items()}


def one(args):
    pat, kind, par, B, seed = args
    rng = np.random.default_rng(seed)
    w = World(pat, kind, rng)
    t = truth(w, par, rng)
    r = analyze.core(w.draw(par, rng), B=B, seed=seed)
    ci = r["ci95"]["delta"]
    return {"truth": t, "p": r["p"]["H1"], "reject": r["reject"]["H1"], "null": t["delta"] >= analyze.MARGIN,
            "cover": math.isfinite(ci[0]) and ci[0] <= t["delta"] <= ci[1],
            "nonEst": r["nonEstimableDraws"]["delta"] > 0, "pointNonEst": not math.isfinite(r["estimates"]["delta"])}


def wilson(k, n):
    if n == 0:
        return [float("nan")] * 3
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(p, 4), round(c - h, 4), round(c + h, 4)]


def summarise(res: list) -> dict:
    n = len(res)
    rate = lambda f: wilson(sum(1 for r in res if f(r)), n)  # noqa: E731
    out = {"n": n, "meanTruth": {k: float(np.mean([r["truth"][k] for r in res])) for k in res[0]["truth"]}}
    out["P(reject H1)"] = rate(lambda r: r["reject"])
    out["P(reject H1 | truth in null)"] = wilson(sum(1 for r in res if r["reject"] and r["null"]), sum(1 for r in res if r["null"]))
    out["coverage delta"] = rate(lambda r: r["cover"])
    out["P(any non-estimable draw)"] = rate(lambda r: r["nonEst"])
    out["P(point estimate non-estimable)"] = rate(lambda r: r["pointNonEst"])
    return out


def run(a) -> int:
    from multiprocessing import Pool
    pat = pattern(a.blocks)
    out = {"pattern": {"students": int(pat["M"].shape[0]), "items": int(pat["M"].shape[1]), "cells": int(pat["M"].sum()),
                       "templates": int(pat["tmpl"].max() + 1), "classes": int(pat["cls"].max() + 1)},
           "settings": {"sims": a.sims, "B": a.draws, "R": R, "kappa": KAPPA, "truthDraws": TRUTH_DRAWS}, "generators": {}}
    for kind in ("2pl", "misspec"):
        cal = calibrate(pat, kind)
        g = out["generators"][kind] = {}
        for name, c in cal.items():
            jobs = [(pat, kind, c["par"], a.draws, 10_000 * hash_(kind + name) % 2**31 + i) for i in range(a.sims)]
            with Pool(a.procs) as pool:
                res = pool.map(one, jobs)
            g[name] = {"par": c["par"], "calibrationTruth": c["meanTruth"], **summarise(res)}
            print(kind, name, json.dumps({k: g[name][k] for k in ("P(reject H1)", "P(reject H1 | truth in null)", "coverage delta")}), flush=True)
            Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


N_REF = 400


def bone(args):
    pat, kind, par, B, seed = args
    rng, ref = np.random.default_rng(seed), np.random.default_rng(seed + 7_919_000)
    w = World(pat, kind, rng)
    v = []
    for _ in range(N_REF):
        P = analyze.Prepared(w.draw(par, ref))
        v.append(P.stats(np.ones(P.obs.shape[1], int))["delta"])
    v = np.array(v); f = v[np.isfinite(v)]
    t = float(f.mean())
    r = analyze.core(w.draw(par, rng), B=B, seed=seed, keep_draws=True)
    p_t = analyze._p(r["estimates"]["delta"], r["draws"]["delta"], t, "<")
    return {"truth": t, "se": float(f.std(ddof=1) / math.sqrt(len(f))), "nFinite": int(len(f)),
            "t200": float(np.nanmean(v[:200])), "t40": float(np.nanmean(v[:40])),
            "reject": bool(r["reject"]["H1"]), "nonCover": bool(p_t <= analyze.ALPHA)}


def bsum(res: list) -> dict:
    def cnt(sel, ev):
        sub = [r for r in res if sel(r)]; k = sum(1 for r in sub if ev(r))
        return {"k": k, "n": len(sub), "wilson": wilson(k, len(sub))}
    t = np.array([r["truth"] for r in res]); se = np.array([r["se"] for r in res])
    d200 = np.abs([r["t200"] - r["truth"] for r in res]); d40 = np.array([r["t40"] - r["truth"] for r in res])
    nc = cnt(lambda r: True, lambda r: r["nonCover"])
    return {"n": len(res),
            "referenceTruth": {"draws": N_REF, "mean": float(t.mean()), "sdAcrossCohorts": float(t.std(ddof=1)),
                               "mcSE_median": float(np.median(se)), "mcSE_max": float(se.max()),
                               "max|t200-t400|": float(d200.max()), "sd(t40-t400)": float(d40.std(ddof=1)),
                               "max|t40-t400|": float(np.abs(d40).max()),
                               "minFiniteDraws": int(min(r["nFinite"] for r in res))},
            "reject at 0.10 (all)": cnt(lambda r: True, lambda r: r["reject"]),
            "reject | truth >= 0.10": cnt(lambda r: r["truth"] >= analyze.MARGIN, lambda r: r["reject"]),
            "reject | truth - 2 SE >= 0.10": cnt(lambda r: r["truth"] - 2 * r["se"] >= analyze.MARGIN, lambda r: r["reject"]),
            "nonCoverage (all, primary)": nc,
            "nonCoverage | |truth - 0.10| <= 0.02": cnt(lambda r: abs(r["truth"] - analyze.MARGIN) <= 0.02, lambda r: r["nonCover"]),
            "criterionMet (Wilson lower <= 0.05)": bool(nc["wilson"][1] <= 0.05)}


def boundary(a) -> int:
    from multiprocessing import Pool
    pat = pattern(a.blocks)
    prm = json.loads(Path(a.params).read_text())["generators"]
    out = {"settings": {"sims": a.sims, "B": a.draws, "R": R, "kappa": KAPPA, "referenceDraws": N_REF,
                        "params": a.params, "criterion": "Wilson 95% lower limit of primary non-coverage <= 0.05"},
           "generators": {}}
    for gi, kind in enumerate(("2pl", "misspec")):
        par = prm[kind]["H1 delta=0.1"]["par"]
        jobs = [(pat, kind, par, a.draws, 900_000_000 + 100_000 * gi + i) for i in range(a.sims)]
        with Pool(a.procs) as pool:
            res = pool.map(bone, jobs)
        out["generators"][kind] = {"par": par, **bsum(res), "cohorts": res}
        g = out["generators"][kind]
        print(kind, json.dumps({k: g[k] for k in g if k not in ("cohorts", "par")}), flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def hash_(s: str) -> int:
    return int.from_bytes(s.encode()[:6].ljust(6, b"_"), "big") % 997


def converge(a) -> int:
    """Same data, B = 1,000 vs 10,000: max |p difference| and decision agreement (H1 boundary, 2pl)."""
    pat = pattern(a.blocks)
    cal = calibrate(pat, "2pl")
    par = cal["H1 delta=0.1"]["par"]
    rows = []
    for i in range(a.sims):
        rng = np.random.default_rng(5000 + i)
        w = World(pat, "2pl", rng); d = w.draw(par, rng)
        r1, r2 = analyze.core(d, B=1000, seed=i), analyze.core(d, B=10000, seed=i)
        rows.append({"d": abs(r1["p"]["H1"] - r2["p"]["H1"]), "agree": r1["reject"] == r2["reject"]})
    out = {"sims": a.sims, "maxAbsPdiffH1": max(r["d"] for r in rows),
           "decisionAgreement": sum(r["agree"] for r in rows) / len(rows)}
    Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out))
    return 0


def selftest() -> int:
    rng = np.random.default_rng(1)
    S, I = 120, 30
    cls = np.repeat(np.arange(6), 20); M = rng.random((S, I)) < 0.7; tmpl = np.arange(I); tmpl[1] = 0
    pat = {"M": M, "cls": cls, "tmpl": tmpl, "half": analyze.halves(M, cls, tmpl)}
    par = {"ra": 0.9, "rb": 0.3, "gS": 1.0, "tauS": 0.0, "g0": 0.3}
    t = truth(World(pat, "2pl", rng), par, rng, 6)
    assert t["delta"] > 0.2, t
    t2 = truth(World(pat, "2pl", rng), {**par, "ra": 0.1}, rng, 6)
    assert t2["delta"] < t["delta"], (t2, t)                              # weaker simulator -> lower delta
    assert bisect(lambda x: x * x, 0, 2, 1.0) - 1 < 0.01
    assert wilson(5, 100)[1] < 0.05 < wilson(5, 100)[2]
    global N_REF
    N_REF, keep = 8, N_REF
    b = bone((pat, "2pl", par, 50, 3))
    N_REF = keep
    assert math.isfinite(b["truth"]) and b["se"] > 0 and isinstance(b["nonCover"], bool), b
    s2 = bsum([b, dict(b, truth=0.1, nonCover=True)])
    assert s2["nonCoverage (all, primary)"]["k"] == 1 + b["nonCover"] and s2["n"] == 2
    print("design selftest ok", {k: round(v, 3) for k, v in t.items()})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--blocks", required=True); r.add_argument("--out", required=True)
    r.add_argument("--sims", type=int, default=500); r.add_argument("--draws", type=int, default=1000)
    r.add_argument("--procs", type=int, default=8)
    c = sub.add_parser("converge"); c.add_argument("--blocks", required=True); c.add_argument("--out", required=True)
    c.add_argument("--sims", type=int, default=50)
    b = sub.add_parser("boundary"); b.add_argument("--blocks", required=True); b.add_argument("--out", required=True)
    b.add_argument("--params", required=True); b.add_argument("--sims", type=int, default=500)
    b.add_argument("--draws", type=int, default=1000); b.add_argument("--procs", type=int, default=8)
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"run": run, "converge": converge, "boundary": boundary, "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
