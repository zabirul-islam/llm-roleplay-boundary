#!/usr/bin/env python3
"""Knob selection on the CALIBRATION side only (pre-registration §2, paper-ncomms/prereg/prereg.md).

Grid sigma_theta x tau = {0.6, 1.0, 1.4} x {0.6, 0.8, 1.0}. Calibration home classes and calibration items are each
split into 5 folds (seed 391). For fold k: the 2PL model is fitted on (classes not k) x (items not k); the simulated
students mirror the real students of the fold-k classes one to one and answer exactly the fold-k items each real
student answered (matched administration, §3). Loss per fold and cell:
    L = MSE(item proportion correct) + 0.5 * mean JS(option distributions) + 0.5 * (log var ratio of student totals)^2
Invalid output counts as incorrect and is its own option bin (never dropped). Choice: lowest mean fold loss; cells
within 1 SE of the best -> the smaller sigma_theta, then the smaller tau.

    python3 knob.py build --blocks ~/ncomms-data/eedi-blocks --out-dir runs/knob
    python3 knob.py score --blocks ~/ncomms-data/eedi-blocks --out-dir runs/knob --model qwen3.5-27b
    python3 knob.py selftest
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib import LEVEL_CUTS, fit_2pl, theta_map  # noqa: E402
from metrics import adherence  # noqa: E402
from sim import CAL_LEVEL, LEVELS, SYSTEM, permute_options  # noqa: E402

SIGMAS, TAUS, FOLDS, SEED = (0.6, 1.0, 1.4), (0.6, 0.8, 1.0), 5, 391
INVALID = 0  # option bin for unparseable output


def cell(s: float, t: float) -> str:
    return f"s{s}-t{t}"


def folds(ids, seed: int) -> dict:
    ids = sorted(ids, key=lambda x: (len(x), x))
    random.Random(seed).shuffle(ids)
    return {x: n % FOLDS for n, x in enumerate(ids)}


def load(blocks: str):
    """Calibration answers (block cc) + home class and item text from the open id files (heldout.py)."""
    import heldout
    from calib import read_block
    rows, key = read_block(blocks, "cc")
    homes, items = heldout.meta(blocks, "homes.json"), heldout.meta(blocks, "items.json")
    users = {u for u, _, _ in rows}
    return rows, key, {u: homes[u] for u in users}, {q: items[q]["problem"] for q in key}


def assign(rows, key, home):
    cf = folds(set(home.values()), SEED)
    itf = folds(set(key), SEED + 1)
    return cf, itf


def build(a) -> int:
    rows, key, home, text = load(a.blocks)
    cf, itf = assign(rows, key, home)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fhs = {(s, t): (out / f"{cell(s, t)}-req.jsonl").open("w") for s in SIGMAS for t in TAUS}
    rng = np.random.default_rng(SEED)
    z = {u: float(v) for u, v in zip(sorted(home), rng.standard_normal(len(home)))}  # common across cells
    age = {u: random.Random(f"{SEED}-{u}").choice([11, 12]) for u in home}
    n = 0
    for k in range(FOLDS):
        train = [(u, q, c) for u, q, c in rows if cf[home[u]] != k and itf[q] != k]
        test = [(u, q) for u, q, c in rows if cf[home[u]] == k and itf[q] == k]
        users, items = sorted({u for u, _, _ in train}), sorted({q for _, q, _ in train})
        ui, ii = {u: i for i, u in enumerate(users)}, {q: i for i, q in enumerate(items)}
        resp = np.full((len(users), len(items)), np.nan)
        for u, q, c in train:
            resp[ui[u], ii[q]] = float(c == key[q])
        _, disc, diff = fit_2pl(resp)
        for s in SIGMAS:
            pct = {}
            for u in {u for u, _ in test}:
                th = s * z[u]
                pct[u] = (int(round(100 * float(np.mean(1 / (1 + np.exp(-disc * (th - diff))))))), th)
            for t in TAUS:
                for u, q in test:
                    p, th = pct[u]
                    level = "novice" if th < LEVEL_CUTS[0] else "strong" if th > LEVEL_CUTS[1] else "average"
                    fhs[(s, t)].write(json.dumps({
                        "id": f"k{k}-u{u}-q{q}", "fold": k, "persona": u, "level": level, "pct": p, "question": q,
                        "temperature": t, "messages": [
                            {"role": "system", "content": SYSTEM.format(age=age[u], level=CAL_LEVEL.format(pct=p))},
                            {"role": "user", "content": text[q]}]}) + "\n")
                    n += 1
        print(f"fold {k}: train {len(train)} answers / {len(users)} students / {len(items)} items; test {len(test)} cells", flush=True)
    for fh in fhs.values():
        fh.close()
    print(f"{n} requests in {len(fhs)} cells -> {out}")
    return 0


HISTORY_K = 10
HISTORY_LEVEL = ("Earlier in this quiz you answered the questions shown in this conversation; your answers are your "
                 "own. Keep answering as the same student.")
V3_ARMS = ["a0-t0.8", "a2-t0.6", "a2-t0.8", "a2-t1.0", "a3-t0.6", "a3-t0.8", "a3-t1.0",
           "a1d-t0.8", "a1d-t1.0", "a2d-t0.8", "a2d-t1.0", "perm-a1-t0.8", "perm-a2-t0.8"]


def delta_map(v2_dir: Path, model: str, fold: int, key: dict) -> tuple:
    """Prereg v3 bias correction: fit achieved = alpha + beta * declared on the v2 run (sigma 1.0, tau 0.8) of the OTHER
    folds, then declare pct' = (pct - alpha) / beta so the achieved rate lands on the target. Calibration side only."""
    reqs = {}
    for l in (v2_dir / f"{cell(1.0, 0.8)}-req.jsonl").open():
        d = json.loads(l)
        if d["fold"] != fold:
            reqs[d["id"]] = d
    n, k = Counter(), Counter()
    for l in (v2_dir / f"{cell(1.0, 0.8)}-{model}.jsonl").open():
        d = json.loads(l)
        if d["id"] in reqs and not d["failed"]:
            u = (reqs[d["id"]]["fold"], reqs[d["id"]]["persona"]); n[u] += 1
            k[u] += (d["choice"] or INVALID) == key[reqs[d["id"]]["question"]]
    dec = {}
    for r in reqs.values():
        dec[(r["fold"], r["persona"])] = r["pct"] / 100
    a = adherence({u: dec[u] for u in n}, {u: k[u] / n[u] for u in n})
    return a["intercept"], a["slope"]


def build3(a) -> int:
    """Prereg v3 calibration-side arms (same folds and matched administration as v2; A1 = the v2 cells, reused)."""
    rows, key, home, text = load(a.blocks)
    cf, itf = assign(rows, key, home)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    arms = a.arms.split(",") if a.arms else V3_ARMS
    fhs = {arm: (out / f"{arm}-req.jsonl").open("w") for arm in arms}
    r = a.rep                       # replicate 0 = the 2026-11-14 round (v2 draws); r > 0 = fresh simulated class
    rng = np.random.default_rng(SEED if r == 0 else SEED * 1000 + r)
    z = {u: float(v) for u, v in zip(sorted(home), rng.standard_normal(len(home)))}   # rep 0: same draws as v2
    sfx = "" if r == 0 else f"-r{r}"
    age = {u: random.Random(f"{SEED}{sfx}-{u}").choice([11, 12]) for u in home}
    lvl0 = {u: random.Random(f"{SEED}{sfx}-a0-{u}").choice(list(LEVELS)) for u in home}
    n = Counter()
    for k in range(FOLDS):
        train = [(u, q, c) for u, q, c in rows if cf[home[u]] != k and itf[q] != k]
        test = [(u, q) for u, q, c in rows if cf[home[u]] == k and itf[q] == k]
        anchors = defaultdict(list)                                   # fold-k students x non-fold items (= H x C)
        for u, q, c in rows:
            if cf[home[u]] == k and itf[q] != k:
                anchors[u].append((q, c))
        users, items = sorted({u for u, _, _ in train}), sorted({q for _, q, _ in train})
        ui, ii = {u: i for i, u in enumerate(users)}, {q: i for i, q in enumerate(items)}
        resp = np.full((len(users), len(items)), np.nan)
        for u, q, c in train:
            resp[ui[u], ii[q]] = float(c == key[q])
        _, disc, diff = fit_2pl(resp)
        pct_of = lambda th: int(round(100 * float(np.mean(1 / (1 + np.exp(-disc * (th - diff)))))))  # noqa: E731
        th_pop = {u: 1.0 * z[u] for u in {u for u, _ in test}}
        th_anc = {}
        for u in th_pop:
            an = [(ii[q], float(c == key[q])) for q, c in anchors[u] if q in ii]
            th_anc[u] = theta_map(disc[[i for i, _ in an]], diff[[i for i, _ in an]], np.array([y for _, y in an])) if an else 0.0
        alpha, beta = delta_map(Path(a.v2_dir), a.model, k, key) if any("d-" in x for x in arms) else (0.0, 1.0)
        corr = lambda p: int(min(99, max(1, round((p / 100 - alpha) / beta * 100))))  # noqa: E731
        hist = {}
        for u in th_pop:
            pick = random.Random(f"{SEED}-hist-{u}").sample(anchors[u], min(HISTORY_K, len(anchors[u])))
            msgs = []
            for q, c in pick:
                msgs += [{"role": "user", "content": text[q]}, {"role": "assistant", "content": "ABCD"[c - 1]}]
            hist[u] = msgs
        for arm in arms:
            kind, t = arm.rsplit("-t", 1)
            t = float(t)
            for u, q in test:
                level, item, perm = None, text[q], None
                if kind == "a0":
                    desc, level = LEVELS[lvl0[u]], lvl0[u]
                elif kind in ("a2", "a2d", "perm-a2"):
                    p = pct_of(th_anc[u]); desc = CAL_LEVEL.format(pct=corr(p) if kind == "a2d" else p)
                elif kind == "a1":                                    # A1 at sigma 1.0 (th_pop = 1.0 * z)
                    desc = CAL_LEVEL.format(pct=pct_of(th_pop[u]))
                elif kind in ("a1d", "perm-a1"):
                    p = pct_of(th_pop[u]); desc = CAL_LEVEL.format(pct=corr(p) if kind == "a1d" else p)
                elif kind == "a3":
                    desc = HISTORY_LEVEL
                else:
                    raise SystemExit(f"unknown arm {arm}")
                if kind.startswith("perm"):
                    item, perm = permute_options(item, random.Random(f"{SEED}-perm-{u}-{q}"))
                msgs = [{"role": "system", "content": SYSTEM.format(age=age[u], level=desc)}]
                msgs += hist[u] if kind == "a3" else []
                msgs.append({"role": "user", "content": item})
                rec = {"id": f"k{k}-u{u}-q{q}", "fold": k, "persona": u, "question": q, "key": key[q],
                       "temperature": t, "perm": perm, "thetaPop": round(th_pop[u], 3),
                       "thetaAnchor": round(th_anc[u], 3), "anchors": len(anchors[u]), "messages": msgs}
                if r:
                    rec["seed"] = int(hashlib.sha256(f"{SEED}-r{r}-{arm}-{u}-{q}".encode()).hexdigest()[:8], 16)
                fhs[arm].write(json.dumps(rec) + "\n")
                n[arm] += 1
        print(f"fold {k}: alpha {alpha:.3f} beta {beta:.3f}; anchors/student median "
              f"{int(np.median([len(anchors[u]) for u in th_pop]))}", flush=True)
    for fh in fhs.values():
        fh.close()
    print(json.dumps(dict(n)))
    return 0


def js(p: Counter, q: Counter) -> float:
    """Jensen-Shannon divergence (bits) of two option-count distributions."""
    ks = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    P = np.array([p[k] / sp for k in ks])
    Q = np.array([q[k] / sq for k in ks])
    M = (P + Q) / 2
    kl = lambda x, y: float(np.sum(np.where(x > 0, x * np.log2(np.where(x > 0, x, 1) / y), 0)))  # noqa: E731
    return (kl(P, M) + kl(Q, M)) / 2


def loss(real: list, sim: list, key: dict) -> dict:
    """real/sim: (student, item, choice) with choice in 1..4 or INVALID. Same students x items in both."""
    def stats(rows):
        n, k, opt, tn, tk = Counter(), Counter(), defaultdict(Counter), Counter(), Counter()
        for u, q, c in rows:
            ok = c == key[q]
            n[q] += 1; k[q] += ok; opt[q][c] += 1; tn[u] += 1; tk[u] += ok
        return {q: k[q] / n[q] for q in n}, opt, np.array([tk[u] / tn[u] for u in sorted(tn)])
    pr, orl, tr = stats(real)
    ps, osm, ts = stats(sim)
    qs = sorted(set(pr) & set(ps))
    mse = float(np.mean([(pr[q] - ps[q]) ** 2 for q in qs]))
    jsd = float(np.mean([js(orl[q], osm[q]) for q in qs]))
    vr, vs = tr.var(ddof=1), ts.var(ddof=1)
    lvr = math.log(vs / vr) if vr > 0 and vs > 0 else float("inf")
    return {"mse": mse, "js": jsd, "logVarRatio": lvr, "L": mse + 0.5 * jsd + 0.5 * lvr ** 2, "items": len(qs)}


def choose(cells: dict) -> str:
    """cells: name -> list of fold losses. Lowest mean; within 1 SE of the best -> smaller sigma, then smaller tau."""
    mean = {c: float(np.mean(v)) for c, v in cells.items()}
    best = min(mean, key=mean.get)
    se = float(np.std(cells[best], ddof=1) / math.sqrt(len(cells[best])))
    near = [c for c in mean if mean[c] <= mean[best] + se]
    return min(near, key=lambda c: (float(c.split("-")[0][1:]), float(c.split("-")[1][1:])))


def score(a) -> int:
    rows, key, home, _ = load(a.blocks)
    cf, itf = assign(rows, key, home)
    real = defaultdict(list)
    for u, q, c in rows:
        if cf[home[u]] == itf[q]:
            real[itf[q]].append((u, q, c))
    out = Path(a.out_dir)
    result, fold_losses = {}, {}
    for s in SIGMAS:
        for t in TAUS:
            name = cell(s, t)
            f = out / f"{name}-{a.model}.jsonl"
            reqs = {json.loads(l)["id"]: json.loads(l) for l in (out / f"{name}-req.jsonl").open()}
            got = {}
            for l in f.open() if f.exists() else []:
                d = json.loads(l)
                if not d["failed"]:
                    got[d["id"]] = d["choice"] or INVALID
            if len(got) < len(reqs):
                result[name] = {"complete": False, "answered": len(got), "requests": len(reqs)}
                continue
            sim = defaultdict(list)
            for i, c in got.items():
                r = reqs[i]
                sim[r["fold"]].append((r["persona"], r["question"], c))
            per = [loss(real[k], sim[k], key) for k in range(FOLDS)]
            declared = {r["persona"]: r["pct"] / 100 for r in reqs.values()}
            n, k_ = Counter(), Counter()
            for i, c in got.items():
                u = reqs[i]["persona"]; n[u] += 1; k_[u] += c == key[reqs[i]["question"]]
            fold_losses[name] = [p["L"] for p in per]
            result[name] = {"complete": True, "meanL": float(np.mean(fold_losses[name])), "folds": per,
                            "invalidRate": sum(c == INVALID for c in got.values()) / len(got),
                            "adherence": adherence(declared, {u: k_[u] / n[u] for u in n})}
    done = len(fold_losses) == len(SIGMAS) * len(TAUS)
    summary = {"model": a.model, "chosen": choose(fold_losses) if done else None, "cells": result}
    (out / f"knob-result-{a.model}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"chosen": summary["chosen"], "meanL": {c: round(v["meanL"], 4) for c, v in result.items() if v.get("complete")},
                      "incomplete": [c for c, v in result.items() if not v.get("complete")]}, indent=2))
    return 0


def selftest() -> int:
    key = {"a": 1, "b": 2, "c": 3}
    real = [(u, q, (1 + (ord(u) + ord(q)) % 4) if u != "x" else key[q]) for u in "xyzw" for q in key]
    l0 = loss(real, list(real), key)
    assert l0["mse"] == 0 and l0["js"] == 0 and l0["logVarRatio"] == 0 and l0["L"] == 0
    worse = loss(real, [(u, q, INVALID) for u, q, _ in real], key)
    assert worse["L"] > 0 and worse["js"] > 0
    assert abs(js(Counter({1: 1}), Counter({2: 1})) - 1) < 1e-12
    f1, f2 = folds([str(i) for i in range(50)], SEED), folds([str(i) for i in reversed(range(50))], SEED)
    assert f1 == f2 and Counter(f1.values()) == Counter({k: 10 for k in range(FOLDS)})
    # tie rule: s1.0 is best, s0.6 sits within 1 SE -> pick s0.6; a far cell never wins
    cells = {"s1.0-t0.8": [1.0, 1.2, 0.8, 1.1, 0.9], "s0.6-t1.0": [1.05, 1.1, 1.0, 1.0, 1.0],
             "s0.6-t0.6": [3.0] * 5, "s1.4-t0.6": [1.0, 1.1, 0.9, 1.0, 1.0]}
    assert choose(cells) == "s0.6-t1.0", choose(cells)
    bo = budget_order("u7", ["3", "1", "2", "10"])
    assert bo == budget_order("u7", ["10", "2", "1", "3"]) and sorted(bo) == ["1", "10", "2", "3"]
    print("knob selftest ok")
    return 0


HH_ARMS = ["a2-t0.6", "a0-t0.8", "a1-t0.8"]   # registered primary, H2 control, E-pop (prereg v3.1 §6)


def budget_order(u: str, items: list) -> list:
    """prereg v3.1 §2 anchor-budget curve: one frozen seeded ordering of a student's usable hc items."""
    o = sorted(items, key=lambda x: (len(x), x))
    random.Random(f"{SEED}-budget-{u}").shuffle(o)
    return o


def build_hh(a) -> int:
    """Confirmatory requests (prereg v3.1): held-out students x primary held-out items, replicate a.rep (1..5).
    Uses the frozen cc 2PL and anchors (score3.py freeze); never refits. --target hh needs the lock (heldout.read
    refuses otherwise); --target hc is the pre-lock dry run on open data (anchors then overlap targets: test only).
    Arms a2b{k}-t0.6 (k = 5, 10, 20) = the anchor-budget curve."""
    import heldout
    fz = Path(a.frozen)
    irt = json.load(open(fz / "irt-cc.json")); anchors = json.load(open(fz / "anchors-hc.json"))
    disc, diff, ii = np.array(irt["a"]), np.array(irt["b"]), {q: i for i, q in enumerate(irt["items"])}
    items = heldout.meta(a.blocks, "items.json")
    primary = set(json.load(open(a.components))["primaryItems"]) if a.target == "hh" else set(ii)
    cells = [(r["UserId"], r["QuestionId"]) for r in heldout.read(a.target, a.blocks)
             if r["QuestionId"] in primary and items[r["QuestionId"]]["key"] is not None]
    users = sorted({u for u, _ in cells}, key=lambda x: (len(x), x))
    r = a.rep
    assert r >= 1, "confirmatory replicates are 1..5"
    z = dict(zip(users, np.random.default_rng(SEED * 1000 + r).standard_normal(len(users))))
    age = {u: random.Random(f"{SEED}-r{r}-{u}").choice([11, 12]) for u in users}
    lvl0 = {u: random.Random(f"{SEED}-r{r}-a0-{u}").choice(list(LEVELS)) for u in users}
    pct_of = lambda th: int(round(100 * float(np.mean(1 / (1 + np.exp(-disc * (th - diff)))))))  # noqa: E731
    hc = defaultdict(dict)
    if any("b" in arm.split("-")[0][2:] for arm in a.arms.split(",")):
        for row in heldout.read("hc", a.blocks):
            q = row["QuestionId"]
            if q in ii and items[q]["key"] is not None:
                hc[row["UserId"]][q] = float(int(row["AnswerValue"]) == items[q]["key"])
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    n = Counter()
    for arm in a.arms.split(","):
        kind, t = arm.rsplit("-t", 1)
        with (out / f"{arm}-req.jsonl").open("w") as fh:
            for u, q in cells:
                if kind == "a0":
                    desc = LEVELS[lvl0[u]]
                elif kind == "a1":
                    desc = CAL_LEVEL.format(pct=pct_of(float(z[u])))
                elif kind == "a2":
                    desc = CAL_LEVEL.format(pct=pct_of(anchors.get(u, {"thetaAnchor": 0.0})["thetaAnchor"]))
                elif kind.startswith("a2b"):
                    k = int(kind[3:]); use = budget_order(u, list(hc[u]))[:k]
                    th = theta_map(disc[[ii[x] for x in use]], diff[[ii[x] for x in use]], np.array([hc[u][x] for x in use])) if use else 0.0
                    desc = CAL_LEVEL.format(pct=pct_of(th))
                else:
                    raise SystemExit(f"unknown hh arm {arm}")
                msgs = [{"role": "system", "content": SYSTEM.format(age=age[u], level=desc)},
                        {"role": "user", "content": items[q]["problem"]}]
                fh.write(json.dumps({"id": f"hh-u{u}-q{q}", "persona": u, "question": q, "key": items[q]["key"],
                                     "temperature": float(t), "rep": r, "anchors": anchors.get(u, {}).get("anchors", 0),
                                     "seed": int(hashlib.sha256(f"{SEED}-r{r}-{arm}-{u}-{q}".encode()).hexdigest()[:8], 16),
                                     "messages": msgs}) + "\n")
                n[arm] += 1
    print(json.dumps({"target": a.target, "rep": r, "students": len(users), "cells": len(cells), **n}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "score"):
        p = sub.add_parser(name)
        p.add_argument("--blocks", required=True, help="heldout.py block dir")
        p.add_argument("--out-dir", required=True)
        if name == "score":
            p.add_argument("--model", required=True)
    b3 = sub.add_parser("build3")
    b3.add_argument("--blocks", required=True); b3.add_argument("--out-dir", required=True)
    b3.add_argument("--v2-dir", required=True, help="v2 knob dir (for the delta bias map)")
    b3.add_argument("--model", default="qwen3.5-27b"); b3.add_argument("--arms", default="")
    b3.add_argument("--rep", type=int, default=0, help="replicate (0 = the development round)")
    h = sub.add_parser("build-hh")
    for x in ("--blocks", "--frozen", "--components", "--out-dir"):
        h.add_argument(x, required=True)
    h.add_argument("--target", choices=["hh", "hc"], default="hh"); h.add_argument("--rep", type=int, required=True)
    h.add_argument("--arms", default=",".join(HH_ARMS))
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"build": build, "score": score, "build3": build3, "build-hh": build_hh, "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
