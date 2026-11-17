#!/usr/bin/env python3
"""FoundationalASSIST registered H1 test (addendum A2, prereg/addendum-assist.md). Every ASSIST rule lives here and is
frozen by the addendum lock; locked EEDI files are reused, never edited.

  items     metadata only (prep venv, bs4)  Problems.csv -> items-assist.json (dataset's own clean_problem_body)
  itemset   metadata only                   primary set (no calibration near-duplicate) + templates
  rate-build                                 B4 rating requests (US grades 6-8 wording), all items
  freeze    cc + hc                          2PL on cc, anchors from hc, B1-B3 ridge, B4 isotonic -> frozen/
  design    cc                               validation at the primary shape (one-sided + two-sided products)
  requests  hh (needs lock)                  simulator requests r1-5
  analyze   hh (needs lock)                  H1 (locked analyze.core) if validated, secondary descriptive

Frozen rules:
- Item text = clean_problem_body(Problem Body) from the dataset's Code/clean_utils.py (CC BY-NC 4.0, FoundationalED
  authors); options cleaned the same way, listed in stored order as A), B), ...; key = the option whose raw text equals
  the single Multiple Choice Answer (stripped). No match -> item not eligible (counted).
- Response = one row per user_id x problem_id; duplicates keep the smallest `id` (counted). Correct = discrete_score
  (1 = correct on the first try without hints or seeing the answer); a missing or non-0/1 score drops the row (counted).
- Simulated answer = sim.parse_choice, then the item's option range (dbe_prep.sim_choice); invalid = incorrect.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)
sys.path.insert(0, str(Path(__file__).resolve().parent))
SEED = 391
order = lambda qs: sorted(qs, key=lambda x: (len(str(x)), str(x)))  # noqa: E731

LEVELS_ASSIST = {
    "novice": "You often find math hard and you make the common mistakes students your age make.",
    "average": "You are an average math student: you get many questions right but slip on harder ones.",
    "strong": "You are strong at math and usually, but not always, get questions right.",
}
CAL_LEVEL_ASSIST = "You usually get about {pct}% of math questions like these right."
SYSTEM_ASSIST = ("You are a {age}-year-old middle-school student in the United States answering a math problem. "
                 "{level} Answer as that student really would, not as an expert. Reply with only the letter of your "
                 "answer.")
AGES = [11, 12, 13]
RATE_SYSTEM_ASSIST = "You are an experienced middle-school math teacher in the United States."
RATE_ASK_ASSIST = ("\n\nWhat percentage of US students in grades 6 to 8 would answer this question correctly on their "
                   "first try, without hints? Reply with only a number from 0 to 100.")
ARMS = ["a2-t0.6", "a0-t0.8", "a1-t0.8"]


def eligible_raw(r: dict) -> bool:
    import assist_seal
    return assist_seal.eligible(r)


# ---------------------------------------------------------------- items (prep venv)
def items(a) -> int:
    sys.path.insert(0, str(Path(a.code)))
    from clean_utils import clean_problem_body
    P = list(csv.DictReader(open(Path(a.raw) / "Problems.csv", newline="", encoding="utf-8")))
    out, n = {}, Counter()
    for r in P:
        q = r["problem_id"]; body = clean_problem_body(r["Problem Body"])
        rec = {"set": r["Problem Set Id"], "part": r["Problem Part"], "type": r["Answer Types"], "options": 0,
               "key": None, "eligible": False, "problem": body}
        if eligible_raw(r):
            raw_opts = [o.strip() for o in r["Multiple Choice Options"].split("||")]
            ans = r["Multiple Choice Answers"].strip()
            if ans in raw_opts:
                opts = [clean_problem_body(o) for o in raw_opts]
                rec.update(options=len(opts), key=raw_opts.index(ans) + 1, eligible=True,
                           problem=body + "\n" + "\n".join(f"{'ABCD'[i]}) {o}" for i, o in enumerate(opts)))
            else:
                n["keyNotFound"] += 1
        n["eligible"] += rec["eligible"]
        out[q] = rec
    Path(a.out).write_text(json.dumps(out, indent=0, sort_keys=True))
    print(json.dumps({"items": len(out), **n, "optionCounts": dict(Counter(v["options"] for v in out.values() if v["eligible"]))}))
    return 0


# ---------------------------------------------------------------- item sets
def components_of(qs, it, S, idx, cut=0.8):
    from audit import _components
    edges = [(qs[i], qs[j]) for i in range(len(qs)) for j in range(i + 1, len(qs))
             if it[qs[i]]["set"] == it[qs[j]]["set"] or S[idx[qs[i]], idx[qs[j]]] >= cut]
    return _components(edges, qs)


def itemset(a) -> int:
    from audit import tfidf
    it = json.load(open(a.items)); sp = json.load(open(a.split))
    held_e = order(q for q in sp["held_eligible_items"] if it[q]["eligible"])
    cal = order(set(it) - set(sp["held_items"]))
    allq = held_e + cal; idx = {q: i for i, q in enumerate(allq)}
    X = tfidf([it[q]["problem"] for q in allq]); S = X @ X.T; nh = len(held_e)
    near = [q for i, q in enumerate(held_e) if (S[i, nh:] >= 0.8).any()]
    prim = [q for q in held_e if q not in near]
    out = {"heldEligible": len(held_e), "nearDuplicateOfCalibration": near, "primaryItems": prim,
           "templates": components_of(prim, it, S, idx), "allHeldItems": held_e,
           "templatesAllHeld": components_of(held_e, it, S, idx)}
    out["templateCounts"] = {"primary": len(out["templates"]), "allHeld": len(out["templatesAllHeld"])}
    Path(a.out).write_text(json.dumps(out, indent=0))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in out.items()}))
    return 0


# ---------------------------------------------------------------- B4 ratings
def rate_build(a) -> int:
    it = json.load(open(a.items))
    with open(a.out, "w") as fh:
        for q in order(it):
            fh.write(json.dumps({"id": f"rate-q{q}", "persona": "rater", "level": "rater", "question": q, "temperature": 0.0,
                                 "messages": [{"role": "system", "content": RATE_SYSTEM_ASSIST},
                                              {"role": "user", "content": it[q]["problem"] + RATE_ASK_ASSIST}]}) + "\n")
    print(f"{len(it)} ASSIST rating requests -> {a.out}")
    return 0


# ---------------------------------------------------------------- responses
def responses(rows: list[dict], items_: dict) -> tuple[dict, Counter]:
    """(user_id, problem_id) -> correct 0/1: one row per pair (smallest id wins), discrete_score as correctness."""
    best, n = {}, Counter()
    for r in rows:
        q = r["problem_id"]
        if q not in items_:
            n["unknownItem"] += 1; continue
        k = (r["user_id"], q); i = int(r["id"])
        if k in best:
            n["duplicateRows"] += 1
            if i >= best[k][0]:
                continue
        best[k] = (i, r)
    out = {}
    for k, (_, r) in best.items():
        s = (r.get("discrete_score") or "").strip()
        if s in ("0", "1", "0.0", "1.0"):
            out[k] = float(float(s) == 1.0)
        else:
            n["badScore"] += 1
    n["pairs"] = len(out)
    return out, n


def option_match(row: dict, correct_raw: str) -> float:
    """Sensitivity scoring: 1 only when non-missing answer_text, stripped of leading/trailing whitespace, equals the
    correct option's raw text, likewise stripped; case-sensitive, no other normalization; missing text -> 0."""
    t = row.get("answer_text")
    return float(t is not None and t.strip() != "" and t.strip() == correct_raw.strip())


def responded(cells: dict, qs) -> tuple[list, list]:
    """Zero-response guard: items with at least one retained human response, and the listed items with none."""
    seen = {q for _, q in cells}
    return [q for q in qs if q in seen], [q for q in qs if q not in seen]


def block(blocks, name, items_):
    import assist_seal
    return responses(assist_seal.read(name, blocks), items_)


# ---------------------------------------------------------------- freeze (cc + hc)
def freeze(a) -> int:
    import numpy as np
    from calib import fit_2pl, theta_map
    from score3 import ridge_cv, logit
    from audit import tfidf, isotonic
    from dbe_prep import ratings
    from scipy.stats import spearmanr
    it = json.load(open(a.items)); comp = json.load(open(a.components))
    cc, ncc = block(a.blocks, "cc", it); hc, nhc = block(a.blocks, "hc", it)
    users = order({u for u, _ in cc}); tr = order({q for _, q in cc})
    ui, ii = {u: i for i, u in enumerate(users)}, {q: i for i, q in enumerate(tr)}
    resp = np.full((len(users), len(tr)), np.nan)
    for (u, q), c in cc.items():
        resp[ui[u], ii[q]] = c
    _, disc, diff = fit_2pl(resp)
    p_tr = {q: float(np.nanmean(resp[:, ii[q]])) for q in tr}
    tg = order(comp["allHeldItems"]); allq = order(set(tr) | set(tg))
    T = tfidf([it[q]["problem"] for q in allq]); tid = {q: i for i, q in enumerate(allq)}
    emb = np.load(a.emb); eid = {q: i for i, q in enumerate(emb["ids"])}; E = emb["emb"]
    y = logit([p_tr[q] for q in tr])
    f2, a2 = ridge_cv(E[[eid[q] for q in tr]], y); f3, a3 = ridge_cv(T[[tid[q] for q in tr]], y)
    fb, ab = ridge_cv(E[[eid[q] for q in tr]], diff); fa, aa = ridge_cv(E[[eid[q] for q in tr]], np.log(disc))
    rate = ratings(a.rate); xr = [q for q in tr if q in rate]
    iso = isotonic(np.array([rate[q] for q in xr]), np.array([p_tr[q] for q in xr]))
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    Et, Tt = E[[eid[q] for q in tg]], T[[tid[q] for q in tg]]
    pred = {q: {"B2": float(p2), "B3": float(p3), "B1_b": float(b), "B1_a": float(math.exp(la)),
                "B4_raw": rate.get(q), "B4": float(iso(rate[q])) if q in rate else None}
            for q, p2, p3, b, la in zip(tg, sig(f2(Et)), sig(f3(Tt)), fb(Et), fa(Et))}
    anc = defaultdict(list)
    for (u, q), c in hc.items():
        if q in ii:
            anc[u].append((ii[q], c))
    theta = {u: {"thetaAnchor": float(theta_map(disc[[i for i, _ in v]], diff[[i for i, _ in v]], np.array([c for _, c in v]))),
                 "anchors": len(v)} for u, v in sorted(anc.items())}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "item-predictions.json").write_text(json.dumps(pred, indent=0, sort_keys=True))
    (out / "irt-cc.json").write_text(json.dumps({"items": tr, "a": disc.tolist(), "b": diff.tolist()}))
    (out / "anchors-hc.json").write_text(json.dumps(theta, sort_keys=True))
    settings = {"ridgeAlpha": {"B2": a2, "B3": a3, "B1_b": ab, "B1_log_a": aa}, "ccCounts": dict(ncc), "hcCounts": dict(nhc),
                "ccStudents": len(users), "ccItems": len(tr), "anchoredStudents": len(theta), "ratedItems": len(rate),
                "heldWithoutRating": [q for q in tg if q not in rate],
                "rhoRawRatingVsCcP": float(spearmanr([rate[q] for q in xr], [p_tr[q] for q in xr])[0]),
                "rhoRawRatingVsCcP_eligibleOnly": float(spearmanr([rate[q] for q in xr if it[q]["eligible"]],
                                                                  [p_tr[q] for q in xr if it[q]["eligible"]])[0])}
    (out / "settings.json").write_text(json.dumps(settings, indent=1))
    print(json.dumps(settings))
    return 0


# ---------------------------------------------------------------- design (cc)
def bone2(args):
    """design.bone plus two-sided coverage of the 95% basic interval at the cohort's own reference truth."""
    import numpy as np
    import analyze
    import design as D
    pat, kind, par, B, seed = args
    rng, ref = np.random.default_rng(seed), np.random.default_rng(seed + 7_919_000)
    w = D.World(pat, kind, rng)
    v = []
    for _ in range(D.N_REF):
        P = analyze.Prepared(w.draw(par, ref)); v.append(P.stats(np.ones(P.obs.shape[1], int))["delta"])
    v = np.array(v); f = v[np.isfinite(v)]; t = float(f.mean())
    r = analyze.core(w.draw(par, rng), B=B, seed=seed, keep_draws=True)
    p_t = analyze._p(r["estimates"]["delta"], r["draws"]["delta"], t, "<")
    lo, hi = r["ci95"]["delta"]
    return {"truth": t, "se": float(f.std(ddof=1) / math.sqrt(len(f))), "reject": bool(r["reject"]["H1"]),
            "nonCover": bool(p_t <= analyze.ALPHA), "cover2": bool(math.isfinite(lo) and lo <= t <= hi),
            "nonEst": bool(not math.isfinite(r["estimates"]["delta"]))}


def calibrate_h1(pat, kind, rho_b):
    import design as D
    base = {"ra": 0.4, "rb": 0.5, "gS": 1.0, "tauS": 0.0, "g0": 0.5}
    mt = lambda **kw: D.mean_truth(pat, kind, {**base, **kw}, worlds=10)  # noqa: E731
    base["rb"] = D.bisect(lambda x: mt(rb=x)["rhoB"], 0.05, 0.99, rho_b, it=12)
    base["ra"] = D.bisect(lambda x: mt(ra=x)["delta"], 0.02, 0.99, 0.10, it=12)
    return base, mt(**base)


def design(a) -> int:
    """Validation of every reported inferential product at the primary shape (addendum A2 §3). Criterion fixed before
    the run: under each generator, Wilson 95% upper limit of one-sided non-coverage <= 0.075 AND Wilson 95% lower limit
    of two-sided 95% interval coverage >= 0.925; mean reference truth within 0.10 +/- 0.02."""
    from multiprocessing import Pool
    import numpy as np
    import analyze
    import design as D
    from audit import tfidf
    it = json.load(open(a.items)); comp = json.load(open(a.components)); st = json.load(open(a.settings))
    cc, _ = block(a.blocks, "cc", it)
    n_items, n_students = len(comp["primaryItems"]), a.students
    rng = random.Random(SEED)
    users = rng.sample(order({u for u, _ in cc}), n_students)
    qs = rng.sample(order({q for _, q in cc}), n_items)
    ui, ii = {u: i for i, u in enumerate(users)}, {q: i for i, q in enumerate(qs)}
    M = np.zeros((len(users), len(qs)), bool)
    for (u, q) in cc:
        if u in ui and q in ii:
            M[ui[u], ii[q]] = True
    X = tfidf([it[q]["problem"] for q in qs]); S = X @ X.T
    comps = components_of(qs, it, S, ii)
    t = {q: k for k, m in enumerate(comps) for q in m}; tmpl = np.array([t[q] for q in qs])
    cls = np.zeros(len(users), int)
    pat = {"M": M, "cls": cls, "tmpl": tmpl, "half": analyze.halves(M, cls, tmpl)}
    rho_b = st["rhoRawRatingVsCcP_eligibleOnly"]
    out = {"pattern": {"students": len(users), "items": len(qs), "cells": int(M.sum()), "templates": int(tmpl.max()) + 1,
                       "primaryShape": {"items": n_items, "templates": len(comp["templates"])}},
           "settings": {"sims": a.sims, "B": a.draws, "R": D.R, "kappa": D.KAPPA, "rhoB": rho_b, "referenceDraws": D.N_REF},
           "criterion": "per generator: Wilson upper(one-sided non-coverage) <= 0.075 AND Wilson lower(two-sided coverage) >= 0.925 AND |mean truth - 0.10| <= 0.02",
           "generators": {}}
    ok = True
    for gi, kind in enumerate(("2pl", "misspec")):
        par, cal = calibrate_h1(pat, kind, rho_b)
        jobs = [(pat, kind, par, a.draws, 970_000_000 + 100_000 * gi + i) for i in range(a.sims)]
        with Pool(a.procs) as pool:
            res = pool.map(bone2, jobs)
        n = len(res); k1 = sum(r["nonCover"] for r in res); k2 = sum(r["cover2"] for r in res)
        mean_t = float(np.mean([r["truth"] for r in res]))
        g = {"par": par, "calibrationTruth": cal, "n": n, "meanTruth": mean_t,
             "truthMcSE_median": float(np.median([r["se"] for r in res])),
             "oneSidedNonCoverage": {"k": k1, "n": n, "wilson": D.wilson(k1, n)},
             "twoSidedCoverage": {"k": k2, "n": n, "wilson": D.wilson(k2, n)},
             "rejectAt0.10": D.wilson(sum(r["reject"] for r in res), n),
             "nonEstimable": sum(r["nonEst"] for r in res)}
        g["criterionMet"] = bool(g["oneSidedNonCoverage"]["wilson"][2] <= 0.075 and g["twoSidedCoverage"]["wilson"][1] >= 0.925
                                 and abs(mean_t - 0.10) <= 0.02)
        ok &= g["criterionMet"]
        out["generators"][kind] = g
        print(kind, json.dumps(g), flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    out["criterionMetAll"] = bool(ok)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("criterionMetAll", ok)
    return 0


# ---------------------------------------------------------------- requests and analysis (hh; need the lock)
def requests(a) -> int:
    import numpy as np
    import assist_seal
    it = json.load(open(a.items)); comp = json.load(open(a.components)); fz = Path(a.frozen)
    irt = json.load(open(fz / "irt-cc.json")); anchors = json.load(open(fz / "anchors-hc.json"))
    disc, diff = np.array(irt["a"]), np.array(irt["b"])
    held = set(comp["allHeldItems"])
    hh, n = responses(assist_seal.read("hh", a.blocks), it)
    cells = sorted(k for k in hh if k[1] in held)
    users = order({u for u, _ in cells}); r = a.rep
    z = dict(zip(users, np.random.default_rng(SEED * 1000 + r).standard_normal(len(users))))
    age = {u: random.Random(f"{SEED}-assist-r{r}-{u}").choice(AGES) for u in users}
    lvl0 = {u: random.Random(f"{SEED}-assist-r{r}-a0-{u}").choice(list(LEVELS_ASSIST)) for u in users}
    pct = lambda th: int(round(100 * float(np.mean(1 / (1 + np.exp(-disc * (th - diff)))))))  # noqa: E731
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        kind, t = arm.rsplit("-t", 1)
        with (out / f"{arm}-req.jsonl").open("w") as fh:
            for u, q in cells:
                desc = (LEVELS_ASSIST[lvl0[u]] if kind == "a0" else CAL_LEVEL_ASSIST.format(
                    pct=pct(float(z[u])) if kind == "a1" else pct(anchors.get(u, {"thetaAnchor": 0.0})["thetaAnchor"])))
                fh.write(json.dumps({"id": f"as-u{u}-q{q}", "persona": u, "question": q, "options": it[q]["options"],
                                     "temperature": float(t), "rep": r,
                                     "seed": int(hashlib.sha256(f"{SEED}-assist-r{r}-{arm}-{u}-{q}".encode()).hexdigest()[:8], 16),
                                     "messages": [{"role": "system", "content": SYSTEM_ASSIST.format(age=age[u], level=desc)},
                                                  {"role": "user", "content": it[q]["problem"]}]}) + "\n")
    print(json.dumps({"rep": r, "cells": len(cells), "students": len(users), "responseCounts": dict(n)}))
    return 0


def analyze_(a) -> int:
    import numpy as np
    import analyze
    import assist_seal
    from score3 import difficulty
    from dbe_prep import sim_choice, cohort_variance
    it = json.load(open(a.items)); comp = json.load(open(a.components)); fz = Path(a.frozen)
    pred = json.load(open(fz / "item-predictions.json")); anchors = json.load(open(fz / "anchors-hc.json"))
    rows = assist_seal.read("hh", a.blocks)
    hh, nhh = responses(rows, it)
    prim = order(comp["primaryItems"]); allh = order(comp["allHeldItems"])
    tid = {q: k for k, m in enumerate(comp["templates"]) for q in m}
    tidAll = {q: k for k, m in enumerate(comp["templatesAllHeld"]) for q in m}
    usable = [q for q in prim if pred[q]["B4"] is not None]; usableAll = [q for q in allh if pred[q]["B4"] is not None]
    cells = {k: v for k, v in hh.items() if k[1] in set(usableAll)}
    usable, noResp = responded(cells, usable); usableAll, noRespAll = responded(cells, usableAll)   # guard (A2 §3)
    users = order({u for u, _ in cells}); ui = {u: i for i, u in enumerate(users)}
    # option-match correctness (sensitivity): first answer text equals the correct option's raw text
    import csv as _csv
    raw = {r["problem_id"]: r for r in _csv.DictReader(open(Path(a.raw) / "Problems.csv", newline="", encoding="utf-8"))}
    best = {}
    for r in rows:
        k = (r["user_id"], r["problem_id"])
        if k in cells and (k not in best or int(r["id"]) < int(best[k]["id"])):
            best[k] = r
    optmatch = {k: option_match(best[k], raw[k[1]]["Multiple Choice Answers"]) for k in best}

    def sims(arm):
        reps = []
        for r in range(1, 6):
            d = Path(a.runs) / f"r{r}"
            reqs = {x["id"]: x for x in map(json.loads, open(d / f"{arm}-req.jsonl"))}
            got = {}
            for l in open(d / f"{arm}-{a.model}.jsonl"):
                x = json.loads(l)
                if not x["failed"]:
                    got.setdefault(x["id"], x["choice"])
            assert len(got) == len(reqs), (arm, r, len(got), len(reqs))
            reps.append({(reqs[i]["persona"], reqs[i]["question"]): sim_choice(c, reqs[i]["options"]) for i, c in got.items()})
        return reps
    S = {arm: sims(arm) for arm in ARMS}

    def build(qs, base, tmap, realmap=None):
        realmap = realmap or cells
        ii = {q: i for i, q in enumerate(qs)}
        obs = np.zeros((len(users), len(qs)), bool); real = np.zeros(obs.shape)
        for (u, q), c in realmap.items():
            if q in ii:
                obs[ui[u], ii[q]] = True; real[ui[u], ii[q]] = c
        def mat(reps):
            Mx = np.zeros((5,) + obs.shape)
            for k, rep in enumerate(reps):
                for (u, q), ch in rep.items():
                    if q in ii:
                        Mx[k, ui[u], ii[q]] = float(ch is not None and ch == it[q]["key"])
            return Mx
        tmpl = np.array([tmap[q] for q in qs]); cls = np.zeros(len(users), int)
        return lambda arm: {"obs": obs, "real": real, "sims": {"S": mat(S[arm]), "A0": mat(S["a0-t0.8"])},
                            "base": np.array([base[q] for q in qs], float), "cls": cls, "tmpl": tmpl,
                            "half": analyze.halves(obs, cls, tmpl)}

    def descriptive(d):
        P = analyze.Prepared(d); e = P.stats(np.ones(P.obs.shape[1], int)); dr = [x - e["rhoB"] for x in e["rhoS_r"]]
        return {"rhoS_r": e["rhoS_r"], "rhoS": float(np.mean(e["rhoS_r"])), "rhoBase": e["rhoB"], "delta": e["delta"],
                "delta_r": dr, "mcSE_generationOnly": analyze.mc_se(dr), "items": int((d["obs"].sum(0) > 0).sum())}

    valid = json.load(open(a.validation)); ok = bool(valid.get("criterionMetAll"))
    b4 = {q: pred[q]["B4"] for q in usable}
    mk = build(usable, b4, tid)
    res = {"registration": "osf.io/ygvdf addendum A2", "validationCriterionMet": ok, "responseCounts": dict(nhh),
           "items": len(usable), "itemsWithoutB4": [q for q in prim if pred[q]["B4"] is None],
           "itemsWithoutB4AllHeld": [q for q in allh if pred[q]["B4"] is None],
           "itemsNoRetainedResponse": noResp, "itemsNoRetainedResponseAllHeld": noRespAll, "students": len(users)}
    if ok:
        r = analyze.core(mk("a2-t0.6"), B=10000, seed=SEED)
        res["H1 (A2 vs B4, confirmatory for ASSIST)"] = {"estimates": r["estimates"], "p": r["p"]["H1"], "reject": r["reject"]["H1"],
                                                         "ci95": r["ci95"]["delta"], "mcSE": r["mcSE"]["delta"],
                                                         "bootSE": r["bootSE"]["delta"], "nonEstimableDraws": r["nonEstimableDraws"]}
    res["A2 vs B4 (descriptive)"] = descriptive(mk("a2-t0.6"))
    res["E-pop A1 vs B4"] = descriptive(mk("a1-t0.8"))
    preal = {q: float(np.mean([c for (u, qq), c in cells.items() if qq == q])) for q in usable}
    th = lambda u: anchors.get(u, {"thetaAnchor": 0.0})["thetaAnchor"]  # noqa: E731
    b1 = {q: float(np.mean([1 / (1 + math.exp(-pred[q]["B1_a"] * (th(u) - pred[q]["B1_b"]))) for (u, qq) in cells if qq == q]))
          for q in usable}
    base = {"B1_anchor": b1, "B2": {q: pred[q]["B2"] for q in usable}, "B3": {q: pred[q]["B3"] for q in usable}, "B4": b4}
    for k in ("B1_anchor", "B2", "B3"):
        res[f"A2 vs {k}"] = descriptive(build(usable, base[k], tid)("a2-t0.6"))
    met = {k: difficulty(v, preal) for k, v in base.items()}
    for arm in ARMS:
        per = []
        for rep in S[arm]:
            acc = defaultdict(list)
            for (u, q), ch in rep.items():
                if (u, q) in cells and q in preal:
                    acc[q].append(float(ch is not None and ch == it[q]["key"]))
            per.append(difficulty({q: float(np.mean(v)) for q, v in acc.items()}, preal))
        met[arm] = {k: float(np.mean([m[k] for m in per])) for k in ("rho", "mae", "bias", "calSlope")}
    res["difficulty"] = met
    res["sensitivity all held-out eligible items"] = descriptive(build(usableAll, {q: pred[q]["B4"] for q in usableAll}, tidAll)("a2-t0.6"))
    res["sensitivity option-match correctness"] = descriptive(build(usable, b4, tid, realmap={k: v for k, v in optmatch.items()})("a2-t0.6"))
    q34 = [q for q in usable if it[q]["options"] >= 3]
    res["sensitivity 3-4 option items only"] = descriptive(build(q34, b4, tid)("a2-t0.6"))
    simsc = {arm: [{k: float(ch is not None and ch == it[k[1]]["key"]) for k, ch in rep.items()} for rep in S[arm]] for arm in ARMS}
    res["cohort variance ratio (sim/real; all held-out students)"] = {arm: cohort_variance(cells, simsc[arm], set(usable)) for arm in ARMS}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("validationCriterionMet", "items", "students")} |
                     ({"H1": {k: v for k, v in res["H1 (A2 vs B4, confirmatory for ASSIST)"].items() if k in ("p", "reject", "ci95")}} if ok else {}) |
                     {"A2 vs B4": {k: res["A2 vs B4 (descriptive)"][k] for k in ("rhoS", "rhoBase", "delta")}}))
    return 0


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    rows = [{"id": "5", "user_id": "u", "problem_id": "1", "discrete_score": "1"},
            {"id": "3", "user_id": "u", "problem_id": "1", "discrete_score": "0"},
            {"id": "7", "user_id": "v", "problem_id": "1", "discrete_score": ""},
            {"id": "8", "user_id": "w", "problem_id": "9", "discrete_score": "1"}]
    out, n = responses(rows, {"1": {}})
    assert out == {("u", "1"): 0.0} and n["duplicateRows"] == 1 and n["badScore"] == 1 and n["unknownItem"] == 1, (out, n)
    assert option_match({"answer_text": "  3/4 "}, "3/4") == 1.0 and option_match({"answer_text": "3/4"}, " 3/4") == 1.0
    assert option_match({"answer_text": "A"}, "a") == 0.0 and option_match({"answer_text": None}, "x") == 0.0
    assert option_match({"answer_text": "   "}, "") == 0.0 and option_match({}, "x") == 0.0
    ok, none = responded({("u", "1"): 1.0, ("v", "1"): 0.0, ("u", "2"): 1.0}, ["1", "2", "3"])
    assert ok == ["1", "2"] and none == ["3"]
    out2, n2 = responses([{"id": "2", "user_id": "u", "problem_id": "1", "discrete_score": "1.0"},
                          {"id": "10", "user_id": "u", "problem_id": "1", "discrete_score": "0"}], {"1": {}})
    assert out2 == {("u", "1"): 1.0} and n2["duplicateRows"] == 1           # numeric id: 2 < 10 (not string order)
    assert "maths" not in " ".join(LEVELS_ASSIST.values()) + CAL_LEVEL_ASSIST + SYSTEM_ASSIST + RATE_ASK_ASSIST
    assert SYSTEM_ASSIST.format(age=12, level="L").count("12-year-old") == 1
    print("assist_prep selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    spec = {"items": ["--raw", "--code", "--out"], "itemset": ["--items", "--split", "--out"], "rate-build": ["--items", "--out"],
            "freeze": ["--blocks", "--items", "--components", "--emb", "--rate", "--out"],
            "design": ["--blocks", "--items", "--components", "--settings", "--out"],
            "requests": ["--blocks", "--items", "--components", "--frozen", "--out-dir"],
            "analyze": ["--blocks", "--raw", "--items", "--components", "--frozen", "--runs", "--validation", "--out"]}
    for name, req in spec.items():
        p = sub.add_parser(name)
        for k in req:
            p.add_argument(k, required=True)
        if name == "design":
            p.add_argument("--sims", type=int, default=1000); p.add_argument("--draws", type=int, default=1000)
            p.add_argument("--procs", type=int, default=90); p.add_argument("--students", type=int, default=1000)
        if name == "requests":
            p.add_argument("--rep", type=int, required=True)
        if name == "analyze":
            p.add_argument("--model", default="qwen3.5-27b")
    sub.add_parser("selftest")
    a = ap.parse_args()
    fn = {"items": items, "itemset": itemset, "rate-build": rate_build, "freeze": freeze, "design": design,
          "requests": requests, "analyze": analyze_, "selftest": lambda _a: selftest()}
    return fn[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
