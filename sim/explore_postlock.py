#!/usr/bin/env python3
"""POST-LOCK EXPLORATORY checks requested by the paper council (council/verdict-2026-11-16-draft.md). Not registered;
reported as exploratory. Reads the frozen predictions, the hh blocks and the finished simulator outputs; changes nothing.

1. DBE pipeline audit: simulated vs real accuracy per item, answer-letter use, key/option spot checks, raw replies.
2. Item-level Spearman between role-play item proportions (A2, mean over replicates) and B4, on EEDI and DBE.
3. Wrong-answer shape on EEDI: per item, JS(real wrong answers, uniform over the 3 wrong options) and JS(simulated,
   real); share of items where real wrong answers are near uniform; mean concentration (top-option share).

    python3 explore_postlock.py --out explore-postlock.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout  # noqa: E402
import score3  # noqa: E402

N = Path.home() / "ncomms"


def eedi():
    items = heldout.meta(N / "data/eedi-blocks", "items.json")
    comp = json.load(open(N / "runs/v3/components.json")); prim = set(comp["primaryItems"])
    pred = json.load(open(N / "runs/v3/frozen/item-predictions.json"))
    key = {q: v["key"] for q, v in items.items() if v["key"] is not None}
    real = [(r["UserId"], r["QuestionId"], int(r["AnswerValue"])) for r in heldout.read("hh", N / "data/eedi-blocks")
            if r["QuestionId"] in prim and r["QuestionId"] in key]
    sims = []
    for r in range(1, 6):
        d = N / f"runs/hh/r{r}"
        reqs = {x["id"]: x for x in map(json.loads, open(d / "a2-t0.6-req.jsonl"))}
        got = score3.load_arm(d / "a2-t0.6-qwen3.5-27b.jsonl", reqs)
        sims.append([(reqs[i]["persona"], reqs[i]["question"], c) for i, c in got.items()])
    return key, pred, real, sims


def item_p(rows, key):
    n, k = Counter(), Counter()
    for u, q, c in rows:
        n[q] += 1; k[q] += c == key[q]
    return {q: k[q] / n[q] for q in n}


def js(p, q):
    return score3.js(p, q)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = {"label": "POST-LOCK EXPLORATORY (not registered)"}
    # --- EEDI
    key, pred, real, sims = eedi()
    pr = item_p(real, key); ps = [item_p(s, key) for s in sims]
    qs = sorted(set(pr) & set(ps[0]))
    a2 = {q: float(np.mean([p[q] for p in ps])) for q in qs}
    out["EEDI item-level rho(A2 mean, B4)"] = float(spearmanr([a2[q] for q in qs], [pred[q]["B4"] for q in qs])[0])
    out["EEDI rho(A2 mean, real)"] = float(spearmanr([a2[q] for q in qs], [pr[q] for q in qs])[0])
    wr, ws = defaultdict(Counter), defaultdict(Counter)
    for u, q, c in real:
        if c != key[q]:
            wr[q][c] += 1
    for s in sims:
        for u, q, c in s:
            if c != key[q] and c:
                ws[q][c] += 1
    elig = [q for q in wr if sum(wr[q].values()) >= score3.ELIG and sum(ws[q].values()) > 0]
    uni = {q: js(Counter({w: 1 for w in (1, 2, 3, 4) if w != key[q]}), wr[q]) for q in elig}
    sr = {q: js(ws[q], wr[q]) for q in elig}
    top = lambda c: max(c.values()) / sum(c.values())  # noqa: E731
    out["EEDI wrong answers"] = {"items": len(elig), "JS(uniform, real) median": float(np.median(list(uni.values()))),
                                 "share items JS(uniform, real) < 0.05": float(np.mean([v < 0.05 for v in uni.values()])),
                                 "JS(sim, real) median": float(np.median(list(sr.values()))),
                                 "top wrong option share, real (mean)": float(np.mean([top(wr[q]) for q in elig])),
                                 "top wrong option share, sim (mean)": float(np.mean([top(ws[q]) for q in elig]))}
    # --- DBE
    R = N / "runs/dbe"; it = json.load(open(R / "items-dbe.json")); comp = json.load(open(R / "components-dbe.json"))
    pdbe = json.load(open(R / "frozen/item-predictions.json")); prim = comp["primaryItems"]
    import dbe, dbe_prep
    hh, _ = dbe_prep.first_attempts(dbe.read("hh", N / "data/dbe-blocks"), it)
    real_p = {q: float(np.mean([c for (u, qq), c in hh.items() if qq == q])) for q in prim}
    sim_p, letters, raws = defaultdict(list), Counter(), []
    for r in range(1, 6):
        d = R / f"hh/r{r}"
        reqs = {x["id"]: x for x in map(json.loads, open(d / "a2-t0.6-req.jsonl"))}
        acc = defaultdict(list)
        for l in open(d / "a2-t0.6-qwen3.5-27b.jsonl"):
            x = json.loads(l); rq = reqs[x["id"]]; q = rq["question"]
            ch = dbe_prep.sim_choice(x["choice"], rq["options"]); letters[ch] += 1
            acc[q].append(float(ch is not None and ch == it[q]["key"]))
            if r == 1 and len(raws) < 5 and q in prim:
                raws.append({"q": q, "raw": x["raw"], "choice": x["choice"], "key": it[q]["key"], "options": rq["options"]})
        for q in prim:
            sim_p[q].append(float(np.mean(acc[q])))
    sp = {q: float(np.mean(v)) for q, v in sim_p.items()}
    out["DBE audit"] = {
        "real accuracy mean": float(np.mean(list(real_p.values()))), "sim A2 accuracy mean": float(np.mean(list(sp.values()))),
        "real p SD across items": float(np.std(list(real_p.values()))), "sim p SD across items": float(np.std(list(sp.values()))),
        "rho(sim A2, real)": float(spearmanr([sp[q] for q in prim], [real_p[q] for q in prim])[0]),
        "item-level rho(sim A2, B4)": float(spearmanr([sp[q] for q in prim], [pdbe[q]["B4"] for q in prim])[0]),
        "answer letters (all A2 runs)": {str(k): v for k, v in sorted(letters.items(), key=lambda kv: str(kv[0]))},
        "key letter distribution (primary items)": dict(Counter("ABCD"[it[q]["key"] - 1] for q in prim)),
        "raw reply examples": raws,
        "per item (real p, sim p, B4)": {q: [round(real_p[q], 3), round(sp[q], 3), round(pdbe[q]["B4"], 3)] for q in prim}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "DBE audit"} | {"DBE audit": {k: v for k, v in out["DBE audit"].items() if k not in ("per item (real p, sim p, B4)", "raw reply examples")}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
