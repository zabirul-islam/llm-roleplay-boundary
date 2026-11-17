#!/usr/bin/env python3
"""Confirmatory analysis on block hh (prereg v3.1 §5, §7, §8; registered at osf.io/ygvdf, 2026-11-15T16:19:38Z).

Post-lock wrapper, written after registration while the hh simulation ran, before any hh analysis: it only assembles the
registered inputs and calls the locked analyze.core unchanged (B = 10,000, seed 391). Inputs:
    hh answers (heldout.read, logged), the 134 primary items and 128 templates (components.json, frozen pre-lock),
    B4 item predictions (frozen/item-predictions.json), home classes (homes.json), halves (analyze.halves, seed 391),
    simulated answers of replicates 1..5 (score3.load_arm: first standalone letter; invalid output = incorrect).
Class index c (for the "391-half-c" seed) = position in the sorted home-class ids, as in analyze.cc.

    python3 hh.py --blocks B --frozen F --components C --runs R --out O [--model qwen3.5-27b] [--s a2-t0.6] [--a0 a0-t0.8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402
import heldout  # noqa: E402
import score3  # noqa: E402


def assemble(a) -> tuple[dict, dict]:
    items, homes = heldout.meta(a.blocks, "items.json"), heldout.meta(a.blocks, "homes.json")
    comp = json.load(open(a.components))
    prim = sorted(comp["primaryItems"], key=lambda x: (len(x), x))
    tid = {q: t for t, members in enumerate(comp["templates"]) for q in members}
    assert set(tid) == set(prim) and len(comp["templates"]) == 128, "template map must cover exactly the 134 primary items"
    key = {q: items[q]["key"] for q in prim}
    rows = [(r["UserId"], r["QuestionId"], int(r["AnswerValue"])) for r in heldout.read("hh", a.blocks)
            if r["QuestionId"] in key and key[r["QuestionId"]] is not None]
    users = sorted({u for u, _, _ in rows}, key=lambda x: (len(x), x))
    ui, ii = {u: i for i, u in enumerate(users)}, {q: i for i, q in enumerate(prim)}
    obs = np.zeros((len(users), len(prim)), bool); real = np.zeros(obs.shape)
    for u, q, c in rows:
        assert not obs[ui[u], ii[q]], f"duplicate cell {u} {q}"
        obs[ui[u], ii[q]] = True; real[ui[u], ii[q]] = float(c == key[q])
    pred = json.load(open(Path(a.frozen) / "item-predictions.json"))
    base = np.array([pred[q]["B4"] for q in prim], float)

    def arm(name):
        M = np.zeros((5,) + obs.shape)
        for k, r in enumerate(range(1, 6)):
            d = Path(a.runs) / f"r{r}"
            reqs = {x["id"]: x for x in map(json.loads, open(d / f"{name}-req.jsonl"))}
            got = score3.load_arm(d / f"{name}-{a.model}.jsonl", reqs)
            assert len(got) == len(reqs) == int(obs.sum()), (name, r, len(got), len(reqs), int(obs.sum()))
            for i, c in got.items():
                x = reqs[i]; M[k, ui[x["persona"]], ii[x["question"]]] = float(c == key[x["question"]])
        return M

    classes = sorted({homes[u] for u in users}); cls = np.array([classes.index(homes[u]) for u in users])
    tmpl = np.array([tid[q] for q in prim])
    d = {"obs": obs, "real": real, "sims": {"S": arm(a.s), "A0": arm(a.a0)}, "base": base, "cls": cls, "tmpl": tmpl,
         "half": analyze.halves(obs, cls, tmpl)}
    info = {"students": len(users), "items": len(prim), "cells": int(obs.sum()), "classes": len(classes),
            "templates": int(tmpl.max()) + 1, "arms": {"S": a.s, "A0": a.a0}, "model": a.model}
    return d, info


def main() -> int:
    ap = argparse.ArgumentParser()
    for x in ("--blocks", "--frozen", "--components", "--runs", "--out"):
        ap.add_argument(x, required=True)
    ap.add_argument("--model", default="qwen3.5-27b"); ap.add_argument("--s", default="a2-t0.6")
    ap.add_argument("--a0", default="a0-t0.8"); ap.add_argument("--draws", type=int, default=10000)
    a = ap.parse_args()
    d, info = assemble(a)
    # independent check of the assembly: replicate-1 pooled Spearman recomputed directly with scipy
    # (items with no observed hh cell are non-estimable and excluded, prereg §5; analyze.core lists them)
    n = d["obs"].sum(0); k = n > 0
    p_real = (d["real"] * d["obs"]).sum(0)[k] / n[k]
    p_s1 = (d["sims"]["S"][0] * d["obs"]).sum(0)[k] / n[k]
    r = analyze.core(d, B=a.draws, seed=391)
    assert r["itemsDropped"] == np.where(~k)[0].tolist()
    assert abs(spearmanr(p_s1, p_real)[0] - r["estimates"]["rhoS_r"][0]) < 1e-9
    assert abs(spearmanr(d["base"][k], p_real)[0] - r["estimates"]["rhoB"]) < 1e-9
    info["itemsNoObservedCell"] = [q for q, keep in zip(sorted(json.load(open(a.components))["primaryItems"], key=lambda x: (len(x), x)), k) if not keep]
    out = {"registration": "osf.io/ygvdf", "data": info, **r}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    e = r["estimates"]
    print(json.dumps({"rhoS_r": [round(x, 4) for x in e["rhoS_r"]], "rhoB4": round(e["rhoB"], 4), "delta": round(e["delta"], 4),
                      "p_H1": r["p"]["H1"], "reject_H1": r["reject"]["H1"], "ci95_delta": r["ci95"]["delta"],
                      "mcSE": r["mcSE"]["delta"], "bootSE": r["bootSE"]["delta"], **info}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
