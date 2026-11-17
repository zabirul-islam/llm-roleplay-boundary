#!/usr/bin/env python3
"""Data audit + knob-run diagnostic (prereg v3 Steps 2 and 2b). Open blocks (cc, hc) and id/text files only.

    python3 audit.py data  --blocks ~/ncomms-data/eedi-blocks --split split-eedi.json        > audit-data.json
    python3 audit.py knob  --blocks ~/ncomms-data/eedi-blocks --out-dir ~/ncomms/runs/knob --model qwen3.5-27b > diag.json
    python3 audit.py components --blocks ~/ncomms-data/eedi-blocks --split split-eedi.json --out components.json
    python3 audit.py selftest
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout  # noqa: E402
from metrics import spearman  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+")
# Items that refer to a figure the text release does not contain (frozen 2026-11-14; prereg v3 sensitivity analysis).
FIG = re.compile(r"diagram|picture|image|graph|shown|shaded|figure|below|on the left|on the right|grid|chart|table", re.I)


def figure_items(items: dict) -> list:
    return sorted((q for q, v in items.items() if FIG.search(v["problem"])), key=lambda x: (len(x), x))


def tfidf(texts: list[str]) -> np.ndarray:
    docs = [Counter(TOKEN.findall(t.lower())) for t in texts]
    vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
    df = Counter(w for d in docs for w in d)
    X = np.zeros((len(docs), len(vocab)))
    for r, d in enumerate(docs):
        for w, c in d.items():
            X[r, vocab[w]] = (1 + math.log(c)) * math.log((1 + len(docs)) / (1 + df[w]))
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def near_duplicates(items: dict, held: set, thr: float = 0.8) -> list:
    """Pairs (held item, calibration item, cosine) with TF-IDF cosine >= thr. Text only."""
    q = sorted(items, key=lambda x: (len(x), x))
    X = tfidf([items[i]["problem"] for i in q])
    S = X @ X.T
    out = []
    for a in range(len(q)):
        for b in range(len(q)):
            if q[a] in held and q[b] not in held and S[a, b] >= thr:
                out.append((q[a], q[b], round(float(S[a, b]), 3)))
    return sorted(out, key=lambda x: -x[2])


def data(a) -> int:
    items, homes = heldout.meta(a.blocks, "items.json"), heldout.meta(a.blocks, "homes.json")
    held = set(json.load(open(a.split))["held_questions"])
    rep = {"items": len(items), "students": len(homes)}
    pair, groups, dates = Counter(), defaultdict(set), defaultdict(list)
    for b in ("cc", "hc"):
        for r in heldout.read(b, a.blocks):
            pair[(r["UserId"], r["QuestionId"])] += 1
            groups[r["UserId"]].add(r["GroupId"])
            dates[homes[r["UserId"]]].append(r["DateAnswered"][:10])
    rep["repeatedStudentItemRows_openBlocks"] = sum(v > 1 for v in pair.values())
    rep["studentsInSeveralClasses_openBlocks"] = sum(len(g) > 1 for g in groups.values())
    spans = sorted((max(d) > min(d), min(d), max(d)) for d in dates.values() if d)
    rep["classDateRange"] = {"first": min(s[1] for s in spans), "last": max(s[2] for s in spans)}
    rep["keyConflicts"] = {q: {k: v[k] for k in ("key", "keyRows", "keyConflictRows")}
                           for q, v in items.items() if v.get("keyConflictRows")}
    nd = near_duplicates(items, held)
    rep["nearDuplicatePairs_heldVsCalib_cos>=0.8"] = len(nd)
    rep["nearDuplicateHeldItems"] = sorted({h for h, _, _ in nd}, key=lambda x: (len(x), x))
    rep["nearDuplicateTop"] = nd[:30]
    rep["figureItems"] = figure_items(items)
    rep["figureItemsHeld"] = [q for q in rep["figureItems"] if q in held]
    print(json.dumps(rep, indent=1))
    return 0


def _components(edges: list, nodes: list) -> list:
    """Connected components (union-find) as sorted lists, largest first."""
    par = {n: n for n in nodes}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in edges:
        par[find(a)] = find(b)
    comp = defaultdict(list)
    for n in nodes:
        comp[find(n)].append(n)
    return sorted((sorted(c, key=lambda x: (len(x), x)) for c in comp.values()), key=lambda c: (-len(c), c))


def components(a) -> int:
    """prereg v3.1 §8 dependence units + item-template clusters. Ids, memberships and item text only (no answers).
    Membership graph: each held-out home class links to every GroupId in which any of its students has an hc row;
    GroupIds seen only on sealed hh rows are invisible here (disclosed). Templates: TF-IDF cosine >= 0.8 among the
    confirmatory items, connected components; singletons are their own template."""
    items, homes = heldout.meta(a.blocks, "items.json"), heldout.meta(a.blocks, "homes.json")
    sp = json.load(open(a.split))
    held_q = set(sp["held_questions"])
    held_u = set(sp["held_users"])
    classes = sorted({homes[u] for u in held_u if u in homes}, key=lambda x: (len(x), x))
    grp, size = defaultdict(set), Counter(homes[u] for u in held_u if u in homes)
    for r in heldout.read("hc", a.blocks):
        grp[homes[r["UserId"]]].add(r["GroupId"])
    edges = [(c, "G" + g) for c in classes for g in grp[c] | {c}]
    nodes = classes + sorted({e[1] for e in edges})
    units = [[n for n in c if not n.startswith("G")] for c in _components(edges, nodes)]
    units = [u for u in units if u]
    near = {h for h, _, _ in near_duplicates(items, held_q)}
    primary = sorted((q for q in held_q if q not in near and items[q]["key"] is not None), key=lambda x: (len(x), x))
    X = tfidf([items[q]["problem"] for q in primary])
    S = X @ X.T
    tedges = [(primary[i], primary[j]) for i in range(len(primary)) for j in range(i + 1, len(primary)) if S[i, j] >= 0.8]
    templates = _components(tedges, primary)
    nstud = sum(size.values())
    out = {"classes": len(classes), "students": nstud, "units": units,
           "unitStudents": [sum(size[c] for c in u) for u in units],
           "largestUnitShare": max(sum(size[c] for c in u) for u in units) / nstud,
           "sharedGroupsVisible": sum(len(grp[c] - {c}) for c in classes),
           "primaryItems": primary, "templates": templates,
           "templateSizes": dict(Counter(len(t) for t in templates))}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("primaryItems", "templates", "units")}, indent=1))
    return 0


def isotonic(x: np.ndarray, y: np.ndarray):
    """Pool-adjacent-violators fit of y on x; returns a step function (monotone non-decreasing)."""
    o = np.argsort(x, kind="mergesort")
    xs, ys = x[o], y[o].astype(float)
    blocks = [[ys[i], 1.0, xs[i], xs[i]] for i in range(len(xs))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2, lo2, hi2 = out.pop(); v1, w1, lo1, hi1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, lo1, hi2])
    his, vals = np.array([b[3] for b in out]), np.array([b[0] for b in out])
    return lambda z: vals[np.minimum(np.searchsorted(his, z, side="left"), len(vals) - 1)]


def knob(a) -> int:
    from knob import FOLDS, INVALID, SIGMAS, TAUS, assign, cell, load, loss
    rows, key, home, _ = load(a.blocks)
    cf, itf = assign(rows, key, home)
    real = defaultdict(list)
    for u, q, c in rows:
        if cf[home[u]] == itf[q]:
            real[itf[q]].append((u, q, c))
    out = Path(a.out_dir)
    rep = {"cells": {}, "parser": {}, "humanSplitHalf": {}, "pctMonotone": None}
    rng = np.random.default_rng(391)
    # human split-half reference per fold: item p from two random halves of the fold's students
    for k in range(FOLDS):
        by_u = defaultdict(list)
        for u, q, c in real[k]:
            by_u[u].append((q, c == key[q]))
        us = sorted(by_u)
        rs = []
        for _ in range(200):
            half = set(rng.choice(us, len(us) // 2, replace=False))
            p = [defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])]
            for u in us:
                for q, ok in by_u[u]:
                    cellp = p[u in half][q]; cellp[0] += 1; cellp[1] += ok
            pa = {q: v[1] / v[0] for q, v in p[0].items() if v[0] >= 5}
            pb = {q: v[1] / v[0] for q, v in p[1].items() if v[0] >= 5}
            rs.append(spearman(pa, pb))
        rep["humanSplitHalf"][k] = {"median": round(float(np.nanmedian(rs)), 3), "q05": round(float(np.nanquantile(rs, .05)), 3),
                                     "q95": round(float(np.nanquantile(rs, .95)), 3), "note": "each half has ~half the students"}
    real_p = {}
    for k in range(FOLDS):
        n, kk = Counter(), Counter()
        for u, q, c in real[k]:
            n[q] += 1; kk[q] += c == key[q]
        real_p[k] = {q: kk[q] / n[q] for q in n}
    pct_by_cell = {}
    for s in SIGMAS:
        for t in TAUS:
            name = cell(s, t)
            reqs = {}
            for l in (out / f"{name}-req.jsonl").open():
                d = json.loads(l); reqs[d["id"]] = d
            raw, got, dup = Counter(), {}, 0
            for l in (out / f"{name}-{a.model}.jsonl").open():
                d = json.loads(l)
                if d["failed"]:
                    continue
                dup += d["id"] in got
                got[d["id"]] = d["choice"] or INVALID
                r = (d["raw"] or "").strip()
                raw["single letter" if re.fullmatch(r"[ABCD]\.?", r) else "letter + text" if re.match(r"[ABCD]\b", r) else "other"] += 1
            rep["parser"][name] = {"replies": dict(raw), "duplicateIds": dup, "missing": len(set(reqs) - set(got)),
                                   "extra": len(set(got) - set(reqs))}
            pct_by_cell[name] = {(r["fold"], r["persona"]): r["pct"] for r in reqs.values()}
            sim = defaultdict(list)
            for i, c in got.items():
                r = reqs[i]; sim[r["fold"]].append((r["persona"], r["question"], c))
            folds = []
            for k in range(FOLDS):
                L = loss(real[k], sim[k], key)
                n, kk = Counter(), Counter()
                for u, q, c in sim[k]:
                    n[q] += 1; kk[q] += c == key[q]
                sp = {q: kk[q] / n[q] for q in n}
                qs = sorted(set(sp) & set(real_p[k]))
                d = np.array([sp[q] - real_p[k][q] for q in qs])
                # isotonic recalibration fitted on the OTHER folds' items, applied here: fixes level, cannot fix rank
                xs, ys = [], []
                for j in range(FOLDS):
                    if j == k:
                        continue
                    nj, kj = Counter(), Counter()
                    for u, q, c in sim[j]:
                        nj[q] += 1; kj[q] += c == key[q]
                    for q in nj:
                        if q in real_p[j]:
                            xs.append(kj[q] / nj[q]); ys.append(real_p[j][q])
                f = isotonic(np.array(xs), np.array(ys))
                iso = np.array([float(f(sp[q])) - real_p[k][q] for q in qs])
                folds.append({"rho": round(spearman(sp, real_p[k]), 4), "mae": round(float(np.abs(d).mean()), 4),
                              "bias": round(float(d.mean()), 4), "maeIsotonic": round(float(np.abs(iso).mean()), 4),
                              **{m: round(L[m], 4) for m in ("mse", "js", "logVarRatio", "L")}, "items": len(qs)})
            rep["cells"][name] = {"folds": folds, **{f"mean_{m}": round(float(np.mean([f[m] for f in folds])), 4)
                                                       for m in ("rho", "mae", "bias", "maeIsotonic", "mse", "js", "logVarRatio", "L")}}
    # pct must order personas the same way in every sigma (it is a monotone function of theta = sigma * z)
    base = pct_by_cell[cell(SIGMAS[0], TAUS[0])]
    rep["pctMonotone"] = all(spearman({i: base[k] for i, k in enumerate(base)},
                                      {i: pct_by_cell[cell(s, TAUS[0])][k] for i, k in enumerate(base)}) > 0.99 for s in SIGMAS[1:])
    best = {m: min(rep["cells"], key=lambda c: rep["cells"][c][f"mean_{m}"] * (-1 if m == "rho" else 1)) for m in ("rho", "mae", "L")}
    rep["perMetricBest"] = best
    print(json.dumps(rep, indent=1))
    return 0


def selftest() -> int:
    stem = "Tom has some apples and gives half of them to his sister, then buys three more apples at the shop."
    items = {"1": {"problem": stem + " How many now? A) 1 B) 5 C) 6 D) 7"},
             "2": {"problem": stem + " How many now? A) 1 B) 5 C) 6 D) 8"},
             **{str(i): {"problem": f"Unrelated question {w} about shapes and angles"} for i, w in
                zip(range(3, 9), ["one", "two", "three", "four", "five", "six"])}}
    nd = near_duplicates(items, {"1"})
    assert nd and nd[0][:2] == ("1", "2") and all(h == "1" for h, _, _ in nd)
    assert figure_items({"a": {"problem": "Look at the diagram"}, "b": {"problem": "Add 2 and 3"}}) == ["a"]
    f = isotonic(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.3, 0.1, 0.5, 0.6]))
    v = [float(f(z)) for z in (0.1, 0.2, 0.3, 0.4)]
    assert v == sorted(v) and abs(v[0] - 0.2) < 1e-9 and v[3] == 0.6, v
    u = _components([("c1", "Gx"), ("c2", "Gx")], ["c1", "c2", "c3", "Gx"])
    assert u == [["Gx", "c1", "c2"], ["c3"]], u
    print("audit selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data"); d.add_argument("--blocks", required=True); d.add_argument("--split", required=True)
    k = sub.add_parser("knob"); k.add_argument("--blocks", required=True); k.add_argument("--out-dir", required=True)
    k.add_argument("--model", required=True)
    c = sub.add_parser("components"); c.add_argument("--blocks", required=True); c.add_argument("--split", required=True)
    c.add_argument("--out", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"data": data, "knob": knob, "components": components, "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
