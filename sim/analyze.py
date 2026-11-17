#!/usr/bin/env python3
"""Confirmatory analysis for prereg v3.1 §5, §7, §8: ONE implementation, used by design.py (validation before lock)
and on block hh (after the OSF timestamp).

Inference is conditional on the evaluated cohort (owner rule 2026-11-14, after the membership audit chained 28 of 30
held-out classes into one component): students, home classes and anchors are FIXED; only item-template clusters are
resampled. So item proportions never change inside the bootstrap; a draw changes item multiplicities w_i, which enter
(i) the expanded item vector for Spearman (repeats are repeated observations, average ranks for ties) and (ii)
student scores as weighted proportions over each student's observed items.

H1 is the sole confirmatory test (alpha 0.05, no multiplicity adjustment; owner decision 2026-11-15 after the design
validation, before any hh access). H2 is descriptive only: Prepared.h2() reports classwise variances, variance ratios,
split-half correlations, paired consistency differences and estimability counts, with no test and no interval.

    core(data, B, seed) -> H1 estimate, p, decision, basic interval, MC SE + the H2 descriptive report
    python3 analyze.py selftest

Data (numpy):  obs S x I bool (observed real cells; simulated cells use the same mask)
               real S x I 0/1 (correct), sims {"S": R x S x I, "A0": R x S x I} 0/1 (invalid output = 0)
               base I (B4 item prediction), cls S (home class index), tmpl I (template index), half C x I (0/1)
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.stats import rankdata

MARGIN, ALPHA = 0.10, 0.05


def spearman_idx(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    a, b = rankdata(x[idx]), rankdata(y[idx])
    if a.std() == 0 or b.std() == 0 or len(idx) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _class_var(s: np.ndarray, cls: np.ndarray, C: int) -> np.ndarray:
    ok = ~np.isnan(s)
    n = np.bincount(cls[ok], minlength=C).astype(float)
    m1 = np.bincount(cls[ok], s[ok], minlength=C)
    m2 = np.bincount(cls[ok], s[ok] ** 2, minlength=C)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (m2 - m1 ** 2 / n) / (n - 1)
    v[n < 2] = np.nan
    return v


def _class_r(x: np.ndarray, y: np.ndarray, cls: np.ndarray, C: int) -> np.ndarray:
    ok = ~(np.isnan(x) | np.isnan(y))
    c, x, y = cls[ok], x[ok], y[ok]
    n = np.bincount(c, minlength=C).astype(float)
    sx, sy = np.bincount(c, x, minlength=C), np.bincount(c, y, minlength=C)
    sxx, syy, sxy = (np.bincount(c, x * x, minlength=C), np.bincount(c, y * y, minlength=C),
                     np.bincount(c, x * y, minlength=C))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (sxy - sx * sy / n) / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))
    r[n < 3] = np.nan
    return r


class Prepared:
    def __init__(self, d: dict):
        self.obs = d["obs"]; self.cls = d["cls"]; self.tmpl = d["tmpl"]; self.base = d["base"]
        S, I = self.obs.shape
        self.C = int(self.cls.max()) + 1
        n = self.obs.sum(0)
        self.keep = np.where(n > 0)[0]                     # items with no observed response: non-estimable, listed
        self.dropped = np.where(n == 0)[0].tolist()
        O = self.obs.astype(float)
        H1 = d["half"][self.cls].astype(float) * O          # student's half-1 observed cells
        H2 = (1 - d["half"][self.cls]).astype(float) * O
        self.O, self.H1, self.H2 = O, H1, H2
        self.p_real = (d["real"] * O).sum(0) / np.maximum(n, 1)
        self.p_sim = {k: (v * O).sum(1) / np.maximum(n, 1) for k, v in d["sims"].items()}   # R x I
        self.mat = {"real": d["real"] * O, **{f"{k}{r}": v[r] * O for k, v in d["sims"].items() for r in range(len(v))}}
        self.R = {k: len(v) for k, v in d["sims"].items()}
        # structurally eligible classes for H2: >= 5 students with an observed response in each half (original data)
        both = (H1.sum(1) > 0) & (H2.sum(1) > 0)
        self.both = both
        self.elig = np.where(np.bincount(self.cls[both], minlength=self.C) >= 5)[0]

    def stats(self, w: np.ndarray) -> dict:
        idx = np.repeat(self.keep, w[self.keep])
        out = {}
        rs = [spearman_idx(self.p_sim["S"][r], self.p_real, idx) for r in range(self.R["S"])]
        rb = spearman_idx(self.base, self.p_real, idx)
        out["rhoS_r"], out["rhoB"] = rs, rb
        out["delta"] = float(np.mean(rs) - rb)
        return out

    def h2(self) -> dict:
        """Descriptive H2 report (prereg v3.1 §9), original data only. Per eligible class and simulated replicate:
        real/simulated score variance, log variance ratio (real > 0 and simulated 0 -> -inf; real 0 -> recorded
        separately, ratio undefined), split-half r (undefined stays undefined) and paired difference r_sim - r_real.
        Replicate summaries are medians over the classes where the quantity is defined, with that denominator."""
        w = np.ones(self.obs.shape[1], int)
        den, d1, d2 = self.O @ w, self.H1 @ w, self.H2 @ w
        with np.errstate(invalid="ignore", divide="ignore"):
            sc = {k: np.where(self.both, (m @ w) / den, np.nan) for k, m in self.mat.items()}   # same students as r
            h1 = {k: (m * (self.H1 > 0)) @ w / d1 for k, m in self.mat.items()}
            h2 = {k: (m * (self.H2 > 0)) @ w / d2 for k, m in self.mat.items()}
        e = self.elig
        var = {k: _class_var(sc[k], self.cls, self.C)[e] for k in self.mat}
        r = {k: _class_r(h1[k], h2[k], self.cls, self.C)[e] for k in self.mat}
        vr, rr = var["real"], r["real"]
        out = {"eligibleClasses": [int(c) for c in e], "real": {"var": vr.tolist(), "r": rr.tolist(),
               "zeroVarClasses": int((vr == 0).sum()), "undefinedR": int((~np.isfinite(rr)).sum())}, "arms": {}}
        for arm, R in self.R.items():
            reps = []
            for j in range(R):
                v, rs = var[f"{arm}{j}"], r[f"{arm}{j}"]
                lr = np.full(len(e), np.nan)
                pos = vr > 0
                with np.errstate(divide="ignore"):
                    lr[pos] = np.log(v[pos] / vr[pos])                     # v = 0 -> -inf, kept in the median
                dd = rs - rr                                               # nan where either r is undefined
                okv, okd = ~np.isnan(lr), np.isfinite(dd)
                reps.append({"var": v.tolist(), "logRatio": lr.tolist(), "r": rs.tolist(), "D": dd.tolist(),
                             "simZeroVar": int(((v == 0) & pos).sum()), "simUndefinedR": int((~np.isfinite(rs)).sum()),
                             "medianLogRatio": float(np.median(lr[okv])) if okv.any() else float("nan"), "nLogRatio": int(okv.sum()),
                             "medianD_conditional": float(np.median(dd[okd])) if okd.any() else float("nan"), "nD": int(okd.sum())})
            out["arms"][arm] = reps
        return out


def _p(T: float, Tb: np.ndarray, c: float, alt: str) -> float:
    """Centred bootstrap p (prereg §8), eb = T*b - T. A non-estimable draw counts toward the tail; non-estimable T -> 1."""
    if not math.isfinite(T):
        return 1.0
    e = Tb - T
    bad = ~np.isfinite(e)
    hit = (e <= T - c) if alt == "<" else (e >= T - c)
    return float((1 + np.sum(hit[~bad]) + bad.sum()) / (len(Tb) + 1))


def _basic(T: float, Tb: np.ndarray) -> list:
    e = (Tb - T)[np.isfinite(Tb)]
    if not math.isfinite(T) or len(e) < len(Tb):
        return [float("nan"), float("nan")]                # reported with the non-estimable count, never filled in
    q = np.quantile(e, [0.025, 0.975])
    return [float(T - q[1]), float(T - q[0])]


def mc_se(x: list) -> float:
    x = [v for v in x if math.isfinite(v)]
    return float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def core(d: dict, B: int = 10000, seed: int = 391, keep_draws: bool = False) -> dict:
    P = Prepared(d)
    T = int(P.tmpl.max()) + 1
    est = P.stats(np.ones(P.obs.shape[1], int))
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for b in range(B):
        m = np.bincount(rng.integers(0, T, T), minlength=T)       # template multiplicities
        boot[b] = P.stats(m[P.tmpl])["delta"]
    p = _p(est["delta"], boot, MARGIN, "<")
    return {"estimates": est, "p": {"H1": p}, "reject": {"H1": p <= ALPHA},
            "ci95": {"delta": _basic(est["delta"], boot)},
            "nonEstimableDraws": {"delta": int((~np.isfinite(boot)).sum())},
            "mcSE": {"delta": mc_se([r - est["rhoB"] for r in est["rhoS_r"]])},
            "bootSE": {"delta": float(np.nanstd(boot, ddof=1))},
            "h2": P.h2(), "itemsDropped": P.dropped, "eligibleClasses": len(P.elig), "B": B,
            **({"draws": {"delta": boot}} if keep_draws else {})}


def halves(obs: np.ndarray, cls: np.ndarray, tmpl: np.ndarray, seed: int = 391) -> np.ndarray:
    """Frozen half assignment: per class, the templates it observed in a seeded random order, alternately 1/0."""
    import random
    C, I = int(cls.max()) + 1, obs.shape[1]
    H = np.zeros((C, I), int)
    for c in range(C):
        ts = sorted({int(tmpl[i]) for i in np.where(obs[cls == c].any(0))[0]})
        random.Random(f"{seed}-half-{c}").shuffle(ts)
        one = {t for j, t in enumerate(ts) if j % 2 == 0}
        H[c] = [int(tmpl[i]) in one for i in range(I)]
    return H


def selftest() -> int:
    rng = np.random.default_rng(0)
    S, I, C, R = 300, 60, 10, 5
    cls = np.repeat(np.arange(C), S // C)
    tmpl = np.arange(I); tmpl[1] = 0                                  # one 2-item template
    obs = rng.random((S, I)) < 0.6
    th, b = rng.normal(0, 1, S), rng.normal(0, 1, I)
    P = 1 / (1 + np.exp(-(th[:, None] - b)))
    real = (rng.random((S, I)) < P).astype(float)
    same = lambda: (rng.random((R, S, I)) < P).astype(float)         # noqa: E731 — a perfect simulator
    flat = (rng.random((R, S, I)) < P.mean(0)).astype(float)          # no student differences -> under-dispersed
    d = {"obs": obs, "real": real, "sims": {"S": same(), "A0": flat}, "base": -b + rng.normal(0, 2, I),
         "cls": cls, "tmpl": tmpl, "half": halves(obs, cls, tmpl)}
    r = core(d, B=300)
    e, h = r["estimates"], r["h2"]
    assert e["delta"] > 0.2 and r["p"]["H1"] > 0.5 and not r["reject"]["H1"], r["p"]   # simulator far better
    med = lambda arm, k: float(np.mean([x[k] for x in h["arms"][arm]]))  # noqa: E731
    assert med("A0", "medianLogRatio") < math.log(0.8), h["arms"]["A0"][0]["medianLogRatio"]   # flat arm under-dispersed
    assert abs(med("S", "medianLogRatio")) < 0.25 and abs(med("S", "medianD_conditional")) < 0.15
    d2 = dict(d, base=-b)                                                  # baseline = truth -> big negative delta
    d2["sims"] = {"S": (rng.random((R, S, I)) < P[:, rng.permutation(I)]).astype(float), "A0": flat}  # wrong item order
    r2 = core(d2, B=300)
    assert r2["estimates"]["delta"] < -0.3 and r2["p"]["H1"] < 0.01 and r2["reject"]["H1"], r2["p"]
    obs2 = obs.copy(); obs2[:, 5] = False
    r3 = core(dict(d, obs=obs2), B=50)
    assert r3["itemsDropped"] == [5]
    z = np.zeros((R, S, I)); z[:, cls == 0] = 1.0                          # class 0: every simulated answer correct
    real4 = real.copy(); real4[cls == 1] = 1.0                             # class 1: zero real variance
    h4 = Prepared(dict(d, real=real4, sims={"S": z, "A0": flat})).h2()
    a = h4["arms"]["S"][0]
    assert a["logRatio"][0] == -np.inf and a["simZeroVar"] >= 1, a["logRatio"][:2]       # real > 0, sim 0 -> -inf
    assert math.isnan(a["logRatio"][1]) and h4["real"]["zeroVarClasses"] == 1             # real 0 -> separate, undefined
    assert math.isnan(a["D"][0]) and a["nD"] < len(h4["eligibleClasses"])                 # undefined r stays undefined
    assert np.median(np.array([-np.inf, -np.inf, 0.1, 0.2])) == -np.inf                     # central -inf -> -inf
    assert math.isclose(np.median(np.array([-np.inf, 0.1, 0.2, 0.3])), 0.15)                # finite centre unaffected
    obs5 = obs.copy(); obs5[cls == 2] = False; obs5[cls == 2, :1] = True                     # class 2: one item only
    h5 = Prepared(dict(d, obs=obs5)).h2()
    assert 2 not in h5["eligibleClasses"] and h5["arms"]["S"][0]["nLogRatio"] == len(h5["eligibleClasses"])
    assert _p(float("nan"), np.zeros(9), 0, "<") == 1.0
    assert _p(0.0, np.array([np.nan] * 9), 0.1, "<") == 1.0                  # all draws non-estimable -> tail = all
    assert spearman_idx(np.array([1., 2, 3]), np.array([3., 2, 1]), np.array([0, 1, 2])) == -1.0
    print("analyze selftest ok", {k: round(v, 3) for k, v in e.items() if isinstance(v, float)},
          "A0 median log ratio", round(med("A0", "medianLogRatio"), 3))
    return 0


def cc(a) -> int:
    """Calibration-side run of the registered procedure (prereg v3.1 §8, §11): per fold, the 5 replicates of the
    registered arm (A2 t0.6) and A0 t0.8 against the fold's B4 (isotonic fitted on that fold's training items, as in
    score3). Reports Monte Carlo SE / bootstrap SE and checks that replicate 0 reproduces calib-v3.json."""
    import json
    from collections import defaultdict
    from pathlib import Path
    from audit import isotonic, tfidf, _components
    from knob import load, assign
    import score3
    rows, key, home, text = load(a.blocks)
    cf, itf = assign(rows, key, home)
    rate = {}
    import re
    for l in open(a.rate):
        d = json.loads(l); m = re.search(r"\d+(\.\d+)?", d.get("raw") or "")
        if m and not d["failed"]:
            rate[d["question"]] = min(100.0, float(m.group(0))) / 100
    dirs = [Path(a.runs)] + [Path(a.reps) / f"r{r}" for r in (1, 2, 3, 4)]
    def arm(name):
        out = []
        for d in dirs:
            reqs = {}
            for l in open(d / f"{name}-req.jsonl"):
                x = json.loads(l); reqs[x["id"]] = {"perm": None, "fold": x["fold"], "u": x["persona"], "q": x["question"]}
            got = score3.load_arm(d / f"{name}-{a.model}.jsonl", reqs)
            assert len(got) == len(reqs), (d, name, len(got), len(reqs))
            out.append({(reqs[i]["u"], reqs[i]["q"]): c for i, c in got.items()})
        return out
    S, A0 = arm("a2-t0.6"), arm(a.a0)
    res = {}
    for k in range(5):
        test = [(u, q, c) for u, q, c in rows if cf[home[u]] == k and itf[q] == k]
        train = [(u, q, c) for u, q, c in rows if cf[home[u]] != k and itf[q] != k]
        p_tr = score3.item_p(train, key); tr_q = [q for q in p_tr if q in rate]
        iso = isotonic(np.array([rate[q] for q in tr_q]), np.array([p_tr[q] for q in tr_q]))
        users = sorted({u for u, _, _ in test}); items = sorted({q for _, q, _ in test if q in rate}, key=lambda x: (len(x), x))
        ui, ii = {u: i for i, u in enumerate(users)}, {q: i for i, q in enumerate(items)}
        obs = np.zeros((len(users), len(items)), bool); real = np.zeros(obs.shape)
        for u, q, c in test:
            if q in ii:
                obs[ui[u], ii[q]] = True; real[ui[u], ii[q]] = float(c == key[q])
        def mat(reps):
            M = np.zeros((len(reps),) + obs.shape)
            for r, g in enumerate(reps):
                for (u, q), c in g.items():
                    if u in ui and q in ii:
                        M[r, ui[u], ii[q]] = float(c == key[q])
            return M
        classes = sorted({home[u] for u in users}); cls = np.array([classes.index(home[u]) for u in users])
        X = tfidf([text[q] for q in items]); Sm = X @ X.T
        comps = _components([(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items)) if Sm[i, j] >= 0.8], items)
        tid = {q: t for t, cpt in enumerate(comps) for q in cpt}; tmpl = np.array([tid[q] for q in items])
        d = {"obs": obs, "real": real, "sims": {"S": mat(S), "A0": mat(A0)}, "base": np.array([float(iso(rate[q])) for q in items]),
             "cls": cls, "tmpl": tmpl, "half": halves(obs, cls, tmpl)}
        r = core(d, B=a.draws, seed=391 + k)
        r["ratio"] = {s: r["mcSE"][s] / r["bootSE"][s] for s in r["mcSE"]}
        res[k] = r
        print(k, "rhoS_r", [round(x, 3) for x in r["estimates"]["rhoS_r"]], "rhoB", round(r["estimates"]["rhoB"], 3),
              "ratio", {s: round(v, 3) for s, v in r["ratio"].items()}, flush=True)
    summ = {s: float(np.median([res[k]["ratio"][s] for k in res])) for s in res[0]["ratio"]}
    Path(a.out).write_text(json.dumps({"folds": {k: {kk: v for kk, v in r.items()} for k, r in res.items()},
                                       "medianRatioMCoverBoot": summ}, indent=1, default=float))
    print("median MC/boot SE ratio", summ)
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("cc")
    for x in ("--blocks", "--runs", "--reps", "--rate", "--out"):
        c.add_argument(x, required=True)
    c.add_argument("--model", default="qwen3.5-27b"); c.add_argument("--a0", default="a0-t0.8")
    c.add_argument("--draws", type=int, default=1000)
    sub.add_parser("selftest")
    a = ap.parse_args()
    sys.exit(cc(a) if a.cmd == "cc" else selftest())
