#!/usr/bin/env python3
"""Calibrated personas for the Nature Comms paper (pre-registration §2, paper-ncomms/prereg/prereg.md).

Fits a 2PL IRT model (+ wrong-option shares, descriptive only) on the CALIBRATION side only, then samples personas
by ability alone (prereg v2 §2) and writes them in the JSON form that `sim.py build --personas-file` reads.
numpy only.

    python3 calib.py fit --data all_data.csv --split split-eedi.json --out fit-eedi.json
    python3 calib.py personas --fit fit-eedi.json --n 300 --ability-sd 1.0 --seed 391 --out personas.json
    python3 calib.py selftest

`--side held` is refused until paper-ncomms/prereg/prereg.sha256 exists (pre-registration lock).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LOCK = Path(__file__).resolve().parents[1] / "prereg" / "prereg.sha256"
LEVEL_CUTS = (-0.5, 0.5)  # ability (logit) -> novice / average / strong wording only


def guard(side: str) -> None:
    if side == "held" and not LOCK.exists():
        raise SystemExit("held-out data is locked until the pre-registration is (paper-ncomms/prereg/prereg.sha256)")


def fit_2pl(resp: np.ndarray, iters: int = 400, lr: float = 0.05, seed: int = 391) -> tuple:
    """Joint MAP estimate of a 2PL model. resp: students x items with 1/0/nan. Returns (theta, a, b).
    N(0,1) prior on theta and b, log-normal(0, 0.5) on a; identified by standardising theta each step."""
    rng = np.random.default_rng(seed)
    S, I = resp.shape
    obs = ~np.isnan(resp)
    y = np.nan_to_num(resp)
    th, b, la = rng.normal(0, 0.1, S), np.zeros(I), np.zeros(I)
    for _ in range(iters):
        a = np.exp(la)
        z = a[None, :] * (th[:, None] - b[None, :])
        p = 1 / (1 + np.exp(-z))
        r = (y - p) * obs
        g_th = (r * a[None, :]).sum(1) - th
        g_b = -(r * a[None, :]).sum(0) - b
        g_la = (r * (th[:, None] - b[None, :])).sum(0) * a - la / 0.25
        th += lr * g_th / np.maximum(obs.sum(1), 1)
        b += lr * g_b / np.maximum(obs.sum(0), 1) * 10
        la += lr * g_la / np.maximum(obs.sum(0), 1) * 5
        th = (th - th.mean()) / (th.std() or 1)
    return th, np.exp(la), b


def theta_map(a: np.ndarray, b: np.ndarray, y: np.ndarray, iters: int = 30) -> float:
    """MAP ability for one student from answers y (1/0) to items with fixed 2PL params (a, b); N(0, 1) prior.
    Prereg v3 E-anchor: abilities come only from a student's own anchor answers, never from predicted responses."""
    th = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-a * (th - b)))
        g = float(np.sum(a * (y - p))) - th
        h = -float(np.sum(a * a * p * (1 - p))) - 1.0
        th -= g / h
    return th


def read_side(data: str, split: str, side: str) -> tuple:
    guard(side)
    sp = json.load(open(split))
    hu, hq = set(sp["held_users"]), set(sp["held_questions"])
    want = (lambda u, q: u in hu and q in hq) if side == "held" else (lambda u, q: u not in hu and q not in hq)
    rows, key = [], {}
    with open(data, newline="") as fh:
        for r in csv.DictReader(fh):
            if want(r["UserId"], r["QuestionId"]):
                rows.append((r["UserId"], r["QuestionId"], int(r["AnswerValue"])))
                key[r["QuestionId"]] = int(r["CorrectAnswer"])
    return rows, key


def read_block(blocks: str, block: str = "cc") -> tuple:
    """(student, item, choice) rows + key from a heldout.py block (v3; replaces read_side once the raw CSV is sealed)."""
    import heldout
    items = heldout.meta(blocks, "items.json")
    rows = [(r["UserId"], r["QuestionId"], int(r["AnswerValue"])) for r in heldout.read(block, blocks)
            if items[r["QuestionId"]]["key"] is not None]          # tied-key items are excluded (counted in AUDIT)
    return rows, {q: items[q]["key"] for _, q, _ in rows}


def fit(a) -> int:
    rows, key = read_block(a.blocks, "cc") if a.blocks else read_side(a.data, a.split, a.side)
    users = sorted({s for s, _, _ in rows})
    items = sorted(key, key=lambda x: (len(x), x))
    ui, ii = {u: n for n, u in enumerate(users)}, {q: n for n, q in enumerate(items)}
    resp = np.full((len(users), len(items)), np.nan)
    wrong = defaultdict(Counter)
    for s, q, c in rows:
        resp[ui[s], ii[q]] = float(c == key[q])
        if c != key[q]:
            wrong[q][c] += 1
    th, disc, diff = fit_2pl(resp)
    out = {"side": a.side, "students": len(users), "items": {
        q: {"a": round(float(disc[n]), 4), "b": round(float(diff[n]), 4), "key": key[q],
            "wrongShares": {str(o): round(v / sum(wrong[q].values()), 4) for o, v in sorted(wrong[q].items())}}
        for q, n in ii.items()},
        "abilitySd": round(float(th.std()), 4), "seed": 391}
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps({"students": len(users), "items": len(items), "answers": len(rows)}))
    return 0


def personas(a) -> int:
    # Prereg v2 §2: persona = ability only (no per-item misconception, no slip/guess). Ability is rendered as the
    # 2PL-implied expected proportion correct over the calibration items.
    f = json.load(open(a.fit))
    rng = np.random.default_rng(a.seed)
    disc = np.array([v["a"] for v in f["items"].values()])
    diff = np.array([v["b"] for v in f["items"].values()])
    out = []
    for p in range(a.n):
        th = float(rng.normal(0, a.ability_sd))
        level = "novice" if th < LEVEL_CUTS[0] else "strong" if th > LEVEL_CUTS[1] else "average"
        pct = int(round(100 * float(np.mean(1 / (1 + np.exp(-disc * (th - diff)))))))
        out.append({"id": f"c{p}", "ability": round(th, 3), "level": level, "pct": pct})
    Path(a.out).write_text(json.dumps({"fit": a.fit, "knobs": {"abilitySd": a.ability_sd, "seed": a.seed}, "personas": out}))
    print(f"{a.n} personas -> {a.out}")
    return 0


def selftest() -> int:
    rng = np.random.default_rng(0)
    S, I = 1500, 30
    th = rng.normal(0, 1, S)
    b = np.linspace(-2, 2, I)
    a = rng.uniform(0.8, 1.6, I)
    p = 1 / (1 + np.exp(-a * (th[:, None] - b)))
    resp = (rng.random((S, I)) < p).astype(float)
    resp[rng.random((S, I)) < 0.4] = np.nan                     # sparse, like real logs
    _, _, b_hat = fit_2pl(resp, iters=600)
    r = np.corrcoef(b, b_hat)[0, 1]
    assert r > 0.97, f"difficulty recovery r={r:.3f}"
    ta = rng.uniform(0.8, 1.6, 40); tb = rng.normal(0, 1, 40); tt = rng.normal(0, 1, 300)
    est = [theta_map(ta[:20], tb[:20], (rng.random(20) < 1 / (1 + np.exp(-ta[:20] * (t - tb[:20])))).astype(float)) for t in tt]
    r_th = np.corrcoef(tt, est)[0, 1]
    assert r_th > 0.85, f"anchor MAP recovery r={r_th:.3f} at k=20"
    try:
        guard("held")
        locked = LOCK.exists()
    except SystemExit:
        locked = False
    assert locked == LOCK.exists(), "held-out guard must follow the lock file"
    print(f"calib selftest ok (difficulty recovery r={r:.3f}, anchor theta k=20 r={r_th:.3f})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--blocks", default="", help="heldout.py block dir (v3); fits on block cc")
    f.add_argument("--data", default=""); f.add_argument("--split", default="")
    f.add_argument("--side", choices=["calib", "held"], default="calib"); f.add_argument("--out", required=True)
    p = sub.add_parser("personas")
    p.add_argument("--fit", required=True); p.add_argument("--n", type=int, default=300)
    p.add_argument("--ability-sd", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=391); p.add_argument("--out", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"fit": fit, "personas": personas, "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
