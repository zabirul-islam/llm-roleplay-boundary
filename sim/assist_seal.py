#!/usr/bin/env python3
"""FoundationalASSIST split, quarantine and guarded reads (addendum A2 in preparation, prereg/addendum-assist.md).

Split (frozen before any response is read; uses only user_id and problem_id of Interactions.csv plus problem
metadata from Problems.csv):
  - held-out students: seeded 20% of all user_id (random.Random(391), ids sorted first);
  - eligible items: single-answer multiple choice, 2-4 options, no <img> in stem or options;
  - held-out items: every problem in a seeded 60% of the problem sets that contain an eligible item (whole sets, so a
    shared stem never crosses the split); the analysis uses the eligible items among them.
Blocks as for EEDI/DBE: cc, hc open; ch sealed forever; hh sealed until prereg/addendum-assist.sha256 exists. The raw
Interactions.csv is moved into sealed/. Every read and write is logged in access.log.

    python3 assist_seal.py split --raw DIR/Data --out split-assist.json
    python3 assist_seal.py seal  --raw DIR/Data --split split-assist.json --out-dir BLOCKS
    python3 assist_seal.py selftest
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import shutil
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)
SEED, HOLD_STUDENTS, HOLD_SETS = 391, 0.20, 0.60
LOCK = Path(__file__).resolve().parents[1] / "prereg" / "addendum-assist.sha256"
OPEN, SEALED = ("cc", "hc"), ("ch", "hh")
order = lambda xs: sorted(xs, key=lambda x: (len(str(x)), str(x)))  # noqa: E731


def eligible(r: dict) -> bool:
    at = r["Answer Types"].lower()
    opts, ans = r["Multiple Choice Options"].split("||"), r["Multiple Choice Answers"].split("||")
    img = "<img" in r["Problem Body"].lower() or "<img" in r["Multiple Choice Options"].lower()
    return "multiple choice" in at and "check" not in at and len(ans) == 1 and 2 <= len(opts) <= 4 and not img


def _log(d: Path, msg: str) -> None:
    with open(d / "access.log", "a") as fh:
        fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg} argv={' '.join(sys.argv)}\n")


def split(a) -> int:
    P = list(csv.DictReader(open(Path(a.raw) / "Problems.csv", newline="", encoding="utf-8")))
    elig = {r["problem_id"] for r in P if eligible(r)}
    sets = order({r["Problem Set Id"] for r in P if r["problem_id"] in elig})
    held_sets = set(random.Random(f"{SEED}-sets").sample(sets, round(HOLD_SETS * len(sets))))
    held_items = order(r["problem_id"] for r in P if r["Problem Set Id"] in held_sets)
    users = set()
    with open(Path(a.raw) / "Interactions.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):                      # only user_id is used here
            users.add(r["user_id"])
    users = order(users)
    held_users = order(random.Random(f"{SEED}-students").sample(users, round(HOLD_STUDENTS * len(users))))
    out = {"dataset": "FoundationalASSIST", "seed": SEED, "held_users": held_users, "held_sets": order(held_sets),
           "held_items": held_items, "held_eligible_items": order(set(held_items) & elig),
           "counts": {"users": len(users), "held_users": len(held_users), "eligible_items": len(elig),
                      "eligible_sets": len(sets), "held_sets": len(held_sets), "held_items": len(held_items),
                      "held_eligible_items": len(set(held_items) & elig)}}
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps(out["counts"]))
    return 0


def _path(d: Path, block: str) -> Path:
    return d / "sealed" / f"{block}.csv" if block in SEALED else d / f"{block}.csv"


def read(block: str, out_dir, lock: Path = LOCK) -> list[dict]:
    d = Path(out_dir).expanduser()
    if block == "ch":
        _log(d, "REFUSED read ch (sealed forever)"); raise SystemExit("ASSIST block ch is sealed forever")
    if block == "hh" and not lock.exists():
        _log(d, "REFUSED read hh (no addendum lock)")
        raise SystemExit("ASSIST block hh is sealed until prereg/addendum-assist.sha256 exists")
    if block not in OPEN + SEALED:
        raise SystemExit(f"unknown block {block}")
    _log(d, f"READ {block}" + (" (LOCKED ANALYSIS)" if block == "hh" else ""))
    with open(_path(d, block), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def seal(a) -> int:
    d = Path(a.out_dir).expanduser(); (d / "sealed").mkdir(parents=True, exist_ok=True); os.chmod(d, 0o700)
    os.chmod(d / "sealed", 0o700)
    sp = json.load(open(a.split)); hu, hi = set(sp["held_users"]), set(sp["held_items"])
    raw = Path(a.raw) / "Interactions.csv"
    with open(raw, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh); cols = rd.fieldnames
        outs = {b: open(_path(d, b), "w", newline="", encoding="utf-8") for b in OPEN + SEALED}
        ws = {b: csv.DictWriter(f, fieldnames=cols) for b, f in outs.items()}
        for w in ws.values():
            w.writeheader()
        n = {b: 0 for b in ws}
        for r in rd:                                      # routed by user_id and problem_id only
            b = ("h" if r["user_id"] in hu else "c") + ("h" if r["problem_id"] in hi else "c")
            ws[b].writerow(r); n[b] += 1
        for f in outs.values():
            f.close()
    for b in OPEN + SEALED:
        os.chmod(_path(d, b), 0o600)
    shutil.move(str(raw), d / "sealed" / "Interactions.csv"); os.chmod(d / "sealed" / "Interactions.csv", 0o600)
    _log(d, f"SEAL rows {json.dumps(n)}; raw moved to sealed/; split {a.split}")
    print(json.dumps({"rows": n}))
    return 0


def selftest() -> int:
    import tempfile
    t = Path(tempfile.mkdtemp()); raw = t / "Data"; raw.mkdir()
    with open(raw / "Problems.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["Problem Set Id", "Answer Types", "Problem Body", "Multiple Choice Options",
                                        "Multiple Choice Answers", "problem_id"])
        w.writerow(["S1", "Multiple Choice", "2+2?", "3||4", "4", "10"])
        w.writerow(["S1", "Numeric", "and 3+3?", "", "", "11"])
        w.writerow(["S2", "Multiple Choice", "<img src=x>", "a||b", "a", "12"])
        w.writerow(["S3", "Check All That Apply", "pick", "a||b||c", "a||b", "13"])
        w.writerow(["S4", "Multiple Choice", "5+5?", "9||10||11", "10", "14"])
    with open(raw / "Interactions.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["user_id", "problem_id", "discrete_score"])
        for u in range(10):
            for q in ("10", "11", "14"):
                w.writerow([f"u{u}", q, u % 2])
    assert eligible({"Answer Types": "Multiple Choice", "Problem Body": "x", "Multiple Choice Options": "a||b",
                     "Multiple Choice Answers": "a"}) and not eligible({"Answer Types": "Check All That Apply",
                     "Problem Body": "x", "Multiple Choice Options": "a||b", "Multiple Choice Answers": "a||b"})
    split(argparse.Namespace(raw=str(raw), out=str(t / "split.json")))
    sp = json.load(open(t / "split.json"))
    assert sp["counts"]["eligible_items"] == 2 and sp["counts"]["eligible_sets"] == 2 and sp["counts"]["held_users"] == 2
    assert set(sp["held_items"]) in ({"10", "11"}, {"14"})            # a held set goes out whole (10 with 11)
    seal(argparse.Namespace(raw=str(raw), split=str(t / "split.json"), out_dir=str(t / "blocks")))
    assert not (raw / "Interactions.csv").exists() and oct(os.stat(t / "blocks/sealed/hh.csv").st_mode)[-3:] == "600"
    for b, lock in (("ch", t / "x"), ("hh", t / "nolock")):
        try:
            read(b, t / "blocks", lock=lock); raise AssertionError(b)
        except SystemExit:
            pass
    (t / "lock").write_text("x")
    hh = read("hh", t / "blocks", lock=t / "lock")
    assert hh and all(r["user_id"] in sp["held_users"] and r["problem_id"] in sp["held_items"] for r in hh)
    log = (t / "blocks/access.log").read_text(); assert "REFUSED read hh" in log and "LOCKED ANALYSIS" in log
    shutil.rmtree(t); print("assist_seal selftest ok"); return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split"); s.add_argument("--raw", required=True); s.add_argument("--out", required=True)
    q = sub.add_parser("seal")
    for k in ("--raw", "--split", "--out-dir"):
        q.add_argument(k, required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"split": split, "seal": seal, "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
