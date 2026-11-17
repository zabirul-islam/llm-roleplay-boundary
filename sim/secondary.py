#!/usr/bin/env python3
"""Registered secondary analyses on block hh (prereg v3.1 §1, §2, §4, §5, §9; osf.io/ygvdf). Post-lock code; every
result is reported whatever it shows and none is confirmatory. Bootstrap intervals and p-values come from the locked
analyze.core (B draws, seed 391), so all comparisons share the same template multiplicities.

    python3 secondary.py --blocks B --frozen F --components C --runs RUNS --out O [--draws 10000]
RUNS holds hh/r{1..5}/ and hh172/r{1..5}/ and v3/rate-*.jsonl. A part whose runs are missing or incomplete is
recorded as "pending" and skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402
import heldout  # noqa: E402
import score3  # noqa: E402
from audit import FIG, _components, isotonic, tfidf  # noqa: E402

QWEN, GEMMA, REPS, SEED = "qwen3.5-27b", "gemma-4-31b-it", (1, 2, 3, 4, 5), 391
order = lambda qs: sorted(qs, key=lambda x: (len(x), x))  # noqa: E731


class Ctx:
    def __init__(self, a):
        self.a = a
        self.items, self.homes = heldout.meta(a.blocks, "items.json"), heldout.meta(a.blocks, "homes.json")
        comp = json.load(open(a.components))
        self.prim = order(comp["primaryItems"]); self.tid = {q: t for t, m in enumerate(comp["templates"]) for q in m}
        self.key = {q: v["key"] for q, v in self.items.items() if v["key"] is not None}
        self.hh = [r for r in heldout.read("hh", a.blocks) if r["QuestionId"] in self.key]
        self.pred = json.load(open(Path(a.frozen) / "item-predictions.json"))
        self.anchor = json.load(open(Path(a.frozen) / "anchors-hc.json"))

    def real(self, qs, home_only=False):
        s = set(qs)
        return [(r["UserId"], r["QuestionId"], int(r["AnswerValue"])) for r in self.hh if r["QuestionId"] in s
                and (not home_only or r["GroupId"] == self.homes[r["UserId"]])]

    def sim(self, arm, model=QWEN, dirs=("hh",)):
        """[(u, q, choice)] per replicate, or None if any run is missing or incomplete."""
        out = []
        for r in REPS:
            rows = []
            for d in dirs:
                p = Path(self.a.runs) / d / f"r{r}"
                req, res = p / f"{arm}-req.jsonl", p / f"{arm}-{model}.jsonl"
                if not (req.exists() and res.exists()):
                    return None
                reqs = {x["id"]: x for x in map(json.loads, open(req))}
                got = score3.load_arm(res, reqs)
                if len(got) != len(reqs):
                    return None
                rows += [(reqs[i]["persona"], reqs[i]["question"], c) for i, c in got.items()]
            out.append(rows)
        return out


def assemble(C, real_rows, sims, base, qs):
    users = order({u for u, _, _ in real_rows}); ui = {u: i for i, u in enumerate(users)}; ii = {q: i for i, q in enumerate(qs)}
    obs = np.zeros((len(users), len(qs)), bool); real = np.zeros(obs.shape)
    for u, q, c in real_rows:
        obs[ui[u], ii[q]] = True; real[ui[u], ii[q]] = float(c == C.key[q])
    def mat(reps):
        M = np.zeros((len(reps),) + obs.shape)
        for k, rows in enumerate(reps):
            for u, q, c in rows:
                if u in ui and q in ii:
                    M[k, ui[u], ii[q]] = float(c == C.key[q])
        return M
    if set(qs) <= set(C.tid):
        tmpl = np.array([C.tid[q] for q in qs])
    else:                                                    # registered template rule (TF-IDF cos >= 0.8) on this set
        X = tfidf([C.items[q]["problem"] for q in qs]); S = X @ X.T
        comps = _components([(qs[i], qs[j]) for i in range(len(qs)) for j in range(i + 1, len(qs)) if S[i, j] >= 0.8], qs)
        t = {q: k for k, m in enumerate(comps) for q in m}; tmpl = np.array([t[q] for q in qs])
    classes = sorted({C.homes[u] for u in users}); cls = np.array([classes.index(C.homes[u]) for u in users])
    return {"obs": obs, "real": real, "sims": {"S": mat(sims["S"]), "A0": mat(sims.get("A0") or sims["S"])},
            "base": np.array([base[q] for q in qs], float), "cls": cls, "tmpl": tmpl, "half": analyze.halves(obs, cls, tmpl)}


def contrast(d, B):
    """Delta-rho of the S arm against the base predictor: estimate, interval, p at margins 0.10/0.05/0.15, p(Δ > 0)."""
    r = analyze.core(d, B=B, seed=SEED, keep_draws=True)
    e, D = r["estimates"], r["draws"]["delta"]
    return {"rhoS_r": e["rhoS_r"], "rhoS": float(np.mean(e["rhoS_r"])), "rhoBase": e["rhoB"], "delta": e["delta"],
            "ci95": r["ci95"]["delta"], "p_margin": {str(c): analyze._p(e["delta"], D, c, "<") for c in (0.10, 0.05, 0.15)},
            "p_superiority": analyze._p(e["delta"], D, 0.0, ">"), "mcSE": r["mcSE"]["delta"],
            "itemsDropped": r["itemsDropped"], "items": int((d["obs"].sum(0) > 0).sum()), "_draws": D}


def real_p(rows, key):
    return score3.item_p(rows, key)


def b1(C, rows, theta):
    """B1: mean over the item's assigned students of logistic(a(θ − b)), a and b from the frozen text regressions."""
    acc = defaultdict(list)
    for u, q, _ in rows:
        a, b = C.pred[q]["B1_a"], C.pred[q]["B1_b"]
        acc[q].append(1 / (1 + math.exp(-a * (theta(u) - b))))
    return {q: float(np.mean(v)) for q, v in acc.items()}


def pop_theta(C, r):
    users = order({u for u, _, _ in C.real(C.prim)})        # = knob.build_hh users (σθ = 1.0 for A1)
    z = np.random.default_rng(SEED * 1000 + r).standard_normal(len(users))
    return dict(zip(users, z.tolist()))


def diff_metrics(C, sims, real):
    per = [score3.difficulty(real_p(rows, C.key), real) for rows in sims]
    return {k: float(np.mean([m[k] for m in per])) for k in ("rho", "mae", "bias", "calSlope")} | {"items": per[0]["items"]}


def within_class(C, sims, real_rows):
    home = {u: C.homes[u] for u, _, _ in real_rows}
    v = [score3.within_class_rho(rows, real_rows, C.key, home) for rows in sims]
    return {"rho": float(np.mean([x for x, _ in v])), "classes": [n for _, n in v]}


def adherence(C, arm):
    """Declared accuracy (the X in the prompt) vs achieved simulated accuracy per student, pooled over replicates."""
    pts = []
    for r in REPS:
        p = Path(C.a.runs) / "hh" / f"r{r}"
        reqs = {x["id"]: x for x in map(json.loads, open(p / f"{arm}-req.jsonl"))}
        got = score3.load_arm(p / f"{arm}-{QWEN}.jsonl", reqs)
        dec, ach = {}, defaultdict(list)
        for i, c in got.items():
            x = reqs[i]; u = x["persona"]
            dec[u] = int(re.search(r"about (\d+)%", x["messages"][0]["content"]).group(1)) / 100
            ach[u].append(float(c == C.key[x["question"]]))
        pts += [(dec[u], float(np.mean(v))) for u, v in ach.items()]
    x, y = np.array(pts).T
    return {"students x reps": len(pts), "pearson": float(np.corrcoef(x, y)[0, 1]), "olsSlope": float(np.polyfit(x, y, 1)[0]),
            "meanAchievedMinusDeclared": float((y - x).mean())}


def uniform_js(C, real_rows):
    wr = defaultdict(Counter)
    for u, q, c in real_rows:
        if c != C.key[q]:
            wr[q][c] += 1
    elig = [q for q in wr if sum(wr[q].values()) >= score3.ELIG]
    return {"eligible": len(elig), "wrongJS": float(np.mean([score3.js(Counter({w: 1 for w in (1, 2, 3, 4) if w != C.key[q]}), wr[q])
                                                             for q in elig]))}


def rating(C, model):
    path = Path(C.a.runs) / "v3" / f"rate-{model}.jsonl"
    if not path.exists():
        return None
    out = {}
    for l in open(path):
        d = json.loads(l); m = re.search(r"\d+(\.\d+)?", d.get("raw") or "")
        if m and not d["failed"]:
            out[d["question"]] = min(100.0, float(m.group(0))) / 100
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    for x in ("--blocks", "--frozen", "--components", "--runs", "--out"):
        ap.add_argument(x, required=True)
    ap.add_argument("--draws", type=int, default=10000)
    a = ap.parse_args()
    C, B = Ctx(a), a.draws
    res, pending = {}, []
    R = C.real(C.prim); pr = real_p(R, C.key); qs = order(pr)          # the 130 items with observed cells
    b4 = {q: C.pred[q]["B4"] for q in qs}
    A2, A0, A1 = C.sim("a2-t0.6"), C.sim("a0-t0.8"), C.sim("a1-t0.8")
    th_anc = lambda u: C.anchor.get(u, {"thetaAnchor": 0.0})["thetaAnchor"]  # noqa: E731

    # 1. difficulty metrics (ρ, MAE, bias, calibration slope), arms and baselines
    base = {"B1_anchor": b1(C, R, th_anc), "B2": {q: C.pred[q]["B2"] for q in qs}, "B3": {q: C.pred[q]["B3"] for q in qs}, "B4": b4}
    b1pop = [b1(C, R, (lambda th: (lambda u: th[u]))(pop_theta(C, r))) for r in REPS]
    met = {"A2 t0.6": diff_metrics(C, A2, pr), "A0 t0.8": diff_metrics(C, A0, pr), "A1 t0.8": diff_metrics(C, A1, pr)}
    met |= {k: score3.difficulty(v, pr) for k, v in base.items()}
    met["B1_pop (mean over reps)"] = {k: float(np.mean([score3.difficulty(b, pr)[k] for b in b1pop])) for k in ("rho", "mae", "bias", "calSlope")}
    res["difficulty"] = met

    # 2. E-anchor A2 against every baseline; "beats all" = min over baselines of Δ, same multiplicities
    draws, cmp_ = {}, {}
    for k, v in base.items():
        c = contrast(assemble(C, R, {"S": A2, "A0": A0}, v, qs), B); draws[k] = c.pop("_draws"); cmp_[k] = c
    T = min(cmp_[k]["delta"] for k in cmp_); Dmin = np.min(np.vstack([draws[k] for k in draws]), axis=0)
    cmp_["beats_all (min Δ over B1-B4)"] = {"delta": T, "p_superiority": analyze._p(T, Dmin, 0.0, ">"), "ci95": analyze._basic(T, Dmin)}
    res["E-anchor A2 vs baselines"] = cmp_

    # 3. E-pop: A1 against B4 (registered E-pop comparator)
    c = contrast(assemble(C, R, {"S": A1, "A0": A0}, b4, qs), B); c.pop("_draws"); res["E-pop A1 vs B4"] = c

    # 4. within-class ρ
    res["within-class rho"] = {"A2 t0.6": within_class(C, A2, R), "A0 t0.8": within_class(C, A0, R), "A1 t0.8": within_class(C, A1, R)}

    # 5. sensitivity: no-figure, home-group-only, all 172
    nofig = [q for q in qs if not FIG.search(C.items[q]["problem"])]
    c = contrast(assemble(C, C.real(nofig), {"S": A2, "A0": A0}, b4, nofig), B); c.pop("_draws")
    res["sensitivity no-figure"] = c | {"figureItemsRemoved": [q for q in qs if q not in nofig]}
    Rh = C.real(C.prim, home_only=True); qh = order(real_p(Rh, C.key))
    c = contrast(assemble(C, Rh, {"S": A2, "A0": A0}, b4, qh), B); c.pop("_draws")
    res["sensitivity home-group-only"] = c | {"rows": len(Rh), "rowsAll": len(R)}
    A2x, A0x = C.sim("a2-t0.6", dirs=("hh", "hh172")), C.sim("a0-t0.8", dirs=("hh", "hh172"))
    if A2x and A0x:
        extra = json.load(open(Path(a.runs) / "hh" / "components-extra.json"))["primaryItems"]
        Rx = C.real(C.prim + extra); qx = order(real_p(Rx, C.key))
        c = contrast(assemble(C, Rx, {"S": A2x, "A0": A0x}, {q: C.pred[q]["B4"] for q in qx}, qx), B); c.pop("_draws")
        res["sensitivity all-172"] = c
    else:
        pending.append("all-172")

    # 6. persona adherence; 7. distractors (and the uniform-guessing baseline)
    res["persona adherence"] = {"A2 t0.6": adherence(C, "a2-t0.6"), "A1 t0.8": adherence(C, "a1-t0.8")}
    dis = {}
    for k, s in (("A2 t0.6", A2), ("A0 t0.8", A0), ("A1 t0.8", A1)):
        per = [score3.distractors_v31(rows, R, C.key) for rows in s]
        dis[k] = {"eligible": per[0]["eligible"], "estimable": [p["estimable"] for p in per],
                  "wrongJS": float(np.mean([p["wrongJS"] for p in per])), "modalWrongAgree": float(np.mean([p["modalWrongAgree"] for p in per]))}
    dis["uniform guessing"] = uniform_js(C, R)
    res["distractors"] = dis

    # 8. τ ablation, anchor-budget curve, option permutation, Gemma (need the secondary GPU runs)
    for name, arm in (("tau ablation A2 t0.8 vs B4", "a2-t0.8"), ("budget k=5", "a2b5-t0.6"), ("budget k=10", "a2b10-t0.6"),
                      ("budget k=20", "a2b20-t0.6")):
        s = C.sim(arm)
        if s:
            c = contrast(assemble(C, R, {"S": s, "A0": A0}, b4, qs), B); c.pop("_draws"); res[name] = c
        else:
            pending.append(name)
    perm = {}
    for arm in ("a2-t0.8", "a1-t0.8"):
        s, sp = C.sim(arm), C.sim(f"perm-{arm}")
        if s and sp:
            m, mp = diff_metrics(C, s, pr), diff_metrics(C, sp, pr)
            acc = lambda S: float(np.mean([np.mean([c == C.key[q] for _, q, c in rows]) for rows in S]))  # noqa: E731
            perm[arm] = {"rho": m["rho"], "rhoPermuted": mp["rho"], "accuracy": acc(s), "accuracyPermuted": acc(sp)}
        else:
            pending.append(f"perm {arm}")
    res["option permutation"] = perm
    rg = rating(C, GEMMA); G2, G0, G1 = C.sim("a2-t0.6", GEMMA), C.sim("a0-t0.8", GEMMA), C.sim("a1-t0.8", GEMMA)
    if rg and G2 and G0 and G1:
        from knob import load
        rows, key, _, _ = load(a.blocks)
        pcc = score3.item_p(rows, key); tr = [q for q in pcc if q in rg]
        iso = isotonic(np.array([rg[q] for q in tr]), np.array([pcc[q] for q in tr]))
        gq = [q for q in qs if q in rg]
        gb4 = {q: float(iso(rg[q])) for q in gq}
        c = contrast(assemble(C, C.real(gq), {"S": G2, "A0": G0}, gb4, gq), B); c.pop("_draws")
        res["Gemma A2 vs Gemma B4"] = c | {"itemsWithRating": len(gq)}
        res["Gemma difficulty"] = {"A2 t0.6": diff_metrics(C, G2, pr), "A0 t0.8": diff_metrics(C, G0, pr),
                                   "A1 t0.8": diff_metrics(C, G1, pr), "B4 (Gemma)": score3.difficulty(gb4, pr)}
    else:
        pending.append("Gemma")
    res["pending"] = pending
    Path(a.out).write_text(json.dumps({"registration": "osf.io/ygvdf", "draws": B, **res}, indent=1, default=float))
    print(json.dumps({"done": [k for k in res if k != "pending"], "pending": pending}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
