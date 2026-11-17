#!/usr/bin/env python3
"""Measures for the Nature Comms paper, defined once (pre-registration §3–4, paper-ncomms/prereg/prereg.md).

Answers are rows (student, item, choice) with choice in 1..4 (None = missing) and a key {item: correct choice}.
numpy only.

    python3 metrics.py selftest
"""
from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict

import numpy as np


def p_correct(rows, key) -> dict:
    """Item -> proportion correct among answered (missing excluded)."""
    n, k = Counter(), Counter()
    for s, i, c in rows:
        if c is None:
            continue
        n[i] += 1
        k[i] += c == key[i]
    return {i: k[i] / n[i] for i in n}


def totals(rows, key) -> dict:
    """Student -> proportion correct over the items they answered."""
    n, k = Counter(), Counter()
    for s, i, c in rows:
        if c is None:
            continue
        n[s] += 1
        k[s] += c == key[i]
    return {s: k[s] / n[s] for s in n}


def between_var(rows, key) -> float:
    t = np.array(list(totals(rows, key).values()))
    return float(t.var(ddof=1)) if len(t) > 1 else float("nan")


def wrong_entropy(rows, key) -> dict:
    """Item -> (Shannon entropy in bits, effective number 2**H) of the WRONG options chosen; None if < 2 wrong answers."""
    w = defaultdict(Counter)
    for s, i, c in rows:
        if c is not None and c != key[i]:
            w[i][c] += 1
    out = {}
    for i, cnt in w.items():
        tot = sum(cnt.values())
        if tot < 2:
            continue
        h = -sum(v / tot * math.log2(v / tot) for v in cnt.values())
        out[i] = (h, 2 ** h)
    return out


def blind_spots(rows, key, concept_of: dict, threshold: float = 0.20) -> float:
    """Share of concepts that fewer than `threshold` of the class has mastered (mastered = >= 50% correct on the
    concept's items the student answered)."""
    per = defaultdict(lambda: [0, 0])
    for s, i, c in rows:
        if c is None or i not in concept_of:
            continue
        cell = per[(s, concept_of[i])]
        cell[0] += 1
        cell[1] += c == key[i]
    by_concept = defaultdict(list)
    for (s, con), (n, k) in per.items():
        by_concept[con].append(k / n >= 0.5)
    if not by_concept:
        return float("nan")
    return sum(np.mean(v) < threshold for v in by_concept.values()) / len(by_concept)


def spearman(a: dict, b: dict) -> float:
    ks = sorted(set(a) & set(b))
    if len(ks) < 3:
        return float("nan")
    ra = rank([a[k] for k in ks])
    rb = rank([b[k] for k in ks])
    return float(np.corrcoef(ra, rb)[0, 1])


def rank(x) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    x = np.asarray(x, float)
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def paired_bootstrap(a: dict, b: dict, truth: dict, draws: int = 10_000, seed: int = 391) -> tuple:
    """One-sided paired bootstrap over items for spearman(a,truth) - spearman(b,truth). Returns (diff, p)."""
    ks = sorted(set(a) & set(b) & set(truth))
    rng = np.random.default_rng(seed)
    obs = spearman({k: a[k] for k in ks}, truth) - spearman({k: b[k] for k in ks}, truth)
    worse = 0
    for _ in range(draws):
        s = rng.choice(len(ks), len(ks))
        ka = {j: a[ks[t]] for j, t in enumerate(s)}
        kb = {j: b[ks[t]] for j, t in enumerate(s)}
        kt = {j: truth[ks[t]] for j, t in enumerate(s)}
        worse += (spearman(ka, kt) - spearman(kb, kt)) <= 0
    return obs, (worse + 1) / (draws + 1)


def adherence(declared: dict, observed: dict) -> dict:
    """Persona adherence (exploratory, prereg §5; idea from MatrAIx, arXiv 2608.04205): does a simulated student told
    "you get about X% right" score X%? declared/observed: student -> proportion in [0, 1] (observed = totals()).
    OLS of observed on declared: slope 1 / intercept 0 = perfect adherence, slope 0 = the persona is ignored."""
    ks = sorted(set(declared) & set(observed))
    x = np.array([declared[k] for k in ks], float)
    y = np.array([observed[k] for k in ks], float)
    if len(ks) < 3 or x.var() == 0:
        return {"n": len(ks), "slope": float("nan"), "intercept": float("nan"), "meanAbsGap": float("nan")}
    slope, intercept = np.polyfit(x, y, 1)
    return {"n": len(ks), "slope": float(slope), "intercept": float(intercept),
            "meanAbsGap": float(np.abs(y - x).mean())}


def selftest() -> int:
    key = {"q1": 1, "q2": 2, "q3": 3}
    rows = [("a", "q1", 1), ("b", "q1", 2), ("c", "q1", 3), ("d", "q1", 2),
            ("a", "q2", 2), ("b", "q2", 2), ("c", "q2", 1), ("d", "q2", None),
            ("a", "q3", 1), ("b", "q3", 1), ("c", "q3", 1), ("d", "q3", 3)]
    pc = p_correct(rows, key)
    assert pc == {"q1": 0.25, "q2": 2 / 3, "q3": 0.25}
    ent = wrong_entropy(rows, key)
    # q1 wrong answers: 2,3,2 -> p = (2/3, 1/3) -> H = 0.9183 bits; q3 wrong: 1,1,1 -> H = 0, effective 1
    assert abs(ent["q1"][0] - 0.918296) < 1e-5 and ent["q3"] == (0.0, 1.0) and "q2" not in ent
    t = totals(rows, key)
    assert abs(t["a"] - 2 / 3) < 1e-9 and t["d"] == 0.5
    assert blind_spots(rows, key, {"q1": "x", "q2": "y", "q3": "x"}) == 0.0
    assert blind_spots(rows, key, {"q1": "x"}, threshold=0.5) == 1.0
    assert abs(spearman({1: 1, 2: 2, 3: 3, 4: 4}, {1: 10, 2: 20, 3: 30, 4: 40}) - 1) < 1e-12
    assert abs(spearman({1: 1, 2: 2, 3: 3}, {1: 3, 2: 2, 3: 1}) + 1) < 1e-12
    truth = {i: i for i in range(40)}
    good = {i: i + np.sin(i) for i in range(40)}
    bad = {i: (i * 7919) % 40 for i in range(40)}
    d, p = paired_bootstrap(good, bad, truth, draws=500)
    assert d > 0.5 and p < 0.01, (d, p)
    dec = {i: 0.2 + 0.06 * i for i in range(10)}
    a = adherence(dec, dec)
    assert abs(a["slope"] - 1) < 1e-9 and abs(a["intercept"]) < 1e-9 and a["meanAbsGap"] == 0
    flat = adherence(dec, {i: 0.9 for i in range(10)})
    assert abs(flat["slope"]) < 1e-9 and flat["n"] == 10
    print("metrics selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if sys.argv[1:] == ["selftest"] else 2)
