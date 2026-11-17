#!/usr/bin/env python3
"""Request builders for the registered secondary analyses on block hh (prereg v3.1 §1, §3, §9; osf.io/ygvdf).
Post-lock helper: the locked files (knob.py, sim.py, analyze.py) are reused unchanged, never edited.

    extra-items  --blocks B --components C --out X   components-style file whose primaryItems = the held-out items
                                                     outside the 134 (all-172 sensitivity); feed X to knob.py build-hh
    perm --src DIR/a2-t0.8-req.jsonl --arm perm-a2-t0.8 --out DIR/perm-a2-t0.8-req.jsonl
        option-permutation arm (§3): the unpermuted request with options shuffled exactly as in the calibration round
        (sim.permute_options, rng "391-perm-{u}-{q}"), the permutation recorded for remapping, a fresh per-request seed
    python3 hh_secondary.py selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import permute_options  # noqa: E402

SEED = 391


def extra_items(a) -> int:
    import heldout
    items = heldout.meta(a.blocks, "items.json")
    prim = set(json.load(open(a.components))["primaryItems"])
    held = {r["QuestionId"] for r in heldout.read("hh", a.blocks) if items[r["QuestionId"]]["key"] is not None}
    extra = sorted(held - prim, key=lambda x: (len(x), x))
    assert prim <= held | prim and len(held | prim) == len(prim) + len(extra)
    Path(a.out).write_text(json.dumps({"primaryItems": extra, "note": "held-out items outside the 134 (all-172 sensitivity)",
                                       "heldOutWithKey": len(held), "primary": len(prim)}))
    print(json.dumps({"heldOutWithKeyObserved": len(held), "primary": len(prim), "extra": len(extra)}))
    return 0


def perm_request(x: dict, arm: str) -> dict:
    u, q, r = x["persona"], x["question"], x["rep"]
    item, perm = permute_options(x["messages"][-1]["content"], random.Random(f"{SEED}-perm-{u}-{q}"))
    y = dict(x, perm=perm, messages=x["messages"][:-1] + [{"role": "user", "content": item}])
    y["seed"] = int(hashlib.sha256(f"{SEED}-r{r}-{arm}-{u}-{q}".encode()).hexdigest()[:8], 16)
    return y


def perm(a) -> int:
    n = bad = 0
    with open(a.out, "w") as fh:
        for l in open(a.src):
            y = perm_request(json.loads(l), a.arm); n += 1; bad += y["perm"] is None
            fh.write(json.dumps(y) + "\n")
    print(json.dumps({"arm": a.arm, "requests": n, "unparsedOptions": bad}))
    return 0


def selftest() -> int:
    x = {"id": "hh-u1-q2", "persona": "1", "question": "2", "rep": 1, "seed": 5,
         "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "What is 2+2?\nA) 1\nB) 2\nC) 3\nD) 4"}]}
    y = perm_request(x, "perm-a2-t0.8")
    assert y["perm"] and sorted(y["perm"]) == [1, 2, 3, 4] and y["perm"] != [1, 2, 3, 4], y["perm"]
    assert y["seed"] != x["seed"] and y["id"] == x["id"] and y["messages"][0] == x["messages"][0]
    assert perm_request(x, "perm-a2-t0.8") == y                      # deterministic
    shown = y["messages"][-1]["content"]
    assert "4" in shown.split("\n")[y["perm"].index(4) + 1]           # letter showing original D holds its text
    print("hh_secondary selftest ok", y["perm"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extra-items")
    for k in ("--blocks", "--components", "--out"):
        e.add_argument(k, required=True)
    p = sub.add_parser("perm")
    for k in ("--src", "--arm", "--out"):
        p.add_argument(k, required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"extra-items": extra_items, "perm": perm, "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
