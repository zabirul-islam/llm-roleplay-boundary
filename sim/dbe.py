#!/usr/bin/env python3
"""DBE-KT22 quarantine and guarded reads (prereg v3.1 addendum A1, `prereg/addendum-dbe.md`).

Same four blocks as EEDI (heldout.py), formed from the frozen split-dbe.json by (student side) × (question side):
cc, hc open; ch sealed forever; hh sealed until prereg/addendum-dbe.sha256 (the addendum lock) exists. The raw
Transaction.csv is moved into sealed/. Every write and read is appended to access.log.

    python3 dbe.py seal --raw Transaction.csv --split split-dbe.json --out-dir ~/ncomms-data/dbe-blocks
    python3 dbe.py selftest
In code:  rows = dbe.read("cc", out_dir)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "prereg" / "addendum-dbe.sha256"
OPEN, SEALED = ("cc", "hc"), ("ch", "hh")


def _log(d: Path, msg: str) -> None:
    with open(d / "access.log", "a") as fh:
        fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg} argv={' '.join(sys.argv)}\n")


def _path(d: Path, block: str) -> Path:
    return d / "sealed" / f"{block}.csv" if block in SEALED else d / f"{block}.csv"


def read(block: str, out_dir, lock: Path = LOCK) -> list[dict]:
    d = Path(out_dir).expanduser()
    if block == "ch":
        _log(d, "REFUSED read ch (sealed forever)")
        raise SystemExit("DBE block ch is sealed forever")
    if block == "hh" and not lock.exists():
        _log(d, "REFUSED read hh (no addendum lock)")
        raise SystemExit("DBE block hh is sealed until the addendum is locked (prereg/addendum-dbe.sha256)")
    if block not in OPEN + SEALED:
        raise SystemExit(f"unknown block {block}")
    _log(d, f"READ {block}" + (" (LOCKED ANALYSIS)" if block == "hh" else ""))
    with open(_path(d, block), newline="") as fh:
        return list(csv.DictReader(fh))


def seal(a) -> int:
    d = Path(a.out_dir).expanduser(); (d / "sealed").mkdir(parents=True, exist_ok=True)
    os.chmod(d / "sealed", 0o700)
    sp = json.load(open(a.split))
    hu, hq = set(map(str, sp["held_users"])), set(map(str, sp["held_questions"]))
    with open(a.raw, newline="") as fh:
        rd = csv.DictReader(fh); cols = rd.fieldnames
        outs = {b: open(_path(d, b), "w", newline="") for b in OPEN + SEALED}
        ws = {b: csv.DictWriter(f, fieldnames=cols) for b, f in outs.items()}
        for w in ws.values():
            w.writeheader()
        n = {b: 0 for b in ws}
        for r in rd:                                   # routes rows by ids only; no answer field is inspected
            b = ("h" if r["student_id"] in hu else "c") + ("h" if r["question_id"] in hq else "c")
            ws[b].writerow(r); n[b] += 1
        for f in outs.values():
            f.close()
    for b in SEALED:
        os.chmod(_path(d, b), 0o600)
    raw = d / "sealed" / "Transaction.csv"
    shutil.move(a.raw, raw); os.chmod(raw, 0o600)
    _log(d, f"SEAL rows {json.dumps(n)}; raw moved to sealed/; split {a.split}")
    print(json.dumps({"rows": n}))
    return 0


def selftest() -> int:
    import tempfile
    t = Path(tempfile.mkdtemp())
    raw = t / "Transaction.csv"
    with open(raw, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "student_id", "question_id", "answer_state"])
        for i, (u, q) in enumerate([("1", "10"), ("1", "20"), ("2", "10"), ("2", "20")]):
            w.writerow([i, u, q, "true"])
    (t / "split.json").write_text(json.dumps({"held_users": [2], "held_questions": [20]}))
    out = t / "blocks"
    seal(argparse.Namespace(raw=str(raw), split=str(t / "split.json"), out_dir=str(out)))
    assert [r["id"] for r in read("cc", out)] == ["0"] and [r["id"] for r in read("hc", out)] == ["2"]
    assert not raw.exists() and (out / "sealed" / "Transaction.csv").exists()
    assert oct(os.stat(out / "sealed" / "hh.csv").st_mode)[-3:] == "600"
    for b, lock in (("ch", out / "lock"), ("hh", out / "no-lock")):
        try:
            read(b, out, lock=lock)
            raise AssertionError(f"{b} read allowed")
        except SystemExit:
            pass
    (out / "lock").write_text("x")
    assert [r["id"] for r in read("hh", out, lock=out / "lock")] == ["3"]
    log = (out / "access.log").read_text()
    assert "REFUSED read ch" in log and "REFUSED read hh" in log and "READ hh (LOCKED ANALYSIS)" in log
    shutil.rmtree(t)
    print("dbe selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seal")
    for k in ("--raw", "--split", "--out-dir"):
        s.add_argument(k, required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    return seal(a) if a.cmd == "seal" else selftest()


if __name__ == "__main__":
    sys.exit(main())
