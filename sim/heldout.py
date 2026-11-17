#!/usr/bin/env python3
"""Held-out quarantine for the Nature Comms paper (prereg v3 §1, §7; plan "four blocks").

EEDI answers are split once into four blocks by (student side) × (item side):
    cc  calibration students × calibration items   open           fitting
    hc  held-out students    × calibration items   open (budget)  anchors: ability inference only
    ch  calibration students × held-out items      SEALED forever never read
    hh  held-out students    × held-out items      SEALED to lock the confirmatory analysis only
Plus two open id/text files that hold no answers: items.json (QuestionId -> problem text, key = majority CorrectAnswer
over its rows, None on a tie -> the item is excluded everywhere and counted) and homes.json
(UserId -> home GroupId, computed from ids only as in data/split.py). The raw CSV is moved into sealed/ as well.
Every write and every read is appended to access.log. Reading hh needs paper-ncomms/prereg/prereg.sha256 (the lock);
reading ch is always refused.

    python3 heldout.py quarantine --data all_data.csv --split split-eedi.json --out-dir ~/ncomms-data/eedi-blocks [--seal-raw]
    python3 heldout.py selftest
In code:  rows = heldout.read("cc", out_dir)   # list of dict rows (csv.DictReader)
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "prereg" / "prereg.sha256"
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
        raise SystemExit("block ch (calibration students x held-out items) is sealed forever")
    if block == "hh" and not lock.exists():
        _log(d, "REFUSED read hh (no lock)")
        raise SystemExit("block hh is sealed until the pre-registration is locked (paper-ncomms/prereg/prereg.sha256)")
    if block not in OPEN + SEALED:
        raise SystemExit(f"unknown block {block}")
    _log(d, f"READ {block}" + (" (LOCKED ANALYSIS)" if block == "hh" else ""))
    with open(_path(d, block), newline="") as fh:
        return list(csv.DictReader(fh))


def meta(out_dir, name: str) -> dict:
    """items.json / homes.json: ids and item text only, no answers."""
    return json.load(open(Path(out_dir).expanduser() / name))


def quarantine(a) -> int:
    d = Path(a.out_dir).expanduser()
    (d / "sealed").mkdir(parents=True, exist_ok=True)
    os.chmod(d / "sealed", 0o700)
    sp = json.load(open(a.split))
    hu, hq = set(sp["held_users"]), set(sp["held_questions"])
    by_user, items = collections.defaultdict(collections.Counter), {}
    with open(a.data, newline="") as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames
        outs = {b: open(_path(d, b), "w", newline="") for b in OPEN + SEALED}
        ws = {b: csv.DictWriter(f, fieldnames=cols) for b, f in outs.items()}
        for w in ws.values():
            w.writeheader()
        n = collections.Counter()
        for r in rd:
            by_user[r["UserId"]][r["GroupId"]] += 1
            items.setdefault(r["QuestionId"], {"problem": r["problem"], "keys": collections.Counter()})["keys"][int(r["CorrectAnswer"])] += 1
            b = ("h" if r["UserId"] in hu else "c") + ("h" if r["QuestionId"] in hq else "c")
            ws[b].writerow(r)
            n[b] += 1
        for f in outs.values():
            f.close()
    for b in SEALED:
        os.chmod(_path(d, b), 0o600)
    home = {u: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for u, c in by_user.items()}
    for it in items.values():  # the release has a few items whose CorrectAnswer differs between rows (audit 2026-11-14)
        (k, c), *rest = it.pop("keys").most_common()
        tie = bool(rest) and rest[0][1] == c
        it.update({"key": None if tie else k, "keyRows": c, "keyConflictRows": sum(v for _, v in rest)})
    (d / "items.json").write_text(json.dumps(items))
    (d / "homes.json").write_text(json.dumps(home))
    if a.seal_raw:
        dst = d / "sealed" / Path(a.data).name
        shutil.move(a.data, dst)
        os.chmod(dst, 0o600)
    _log(d, f"QUARANTINE wrote {dict(n)} from {a.data} split={a.split} seal_raw={a.seal_raw}")
    print(json.dumps({"rows": dict(n), "items": len(items), "students": len(home), "dir": str(d)}))
    return 0


def selftest() -> int:
    import tempfile
    t = Path(tempfile.mkdtemp())
    rows = [{"UserId": u, "QuestionId": q, "GroupId": f"g{int(u) % 2}", "AnswerValue": "1", "CorrectAnswer": "1",
             "problem": f"item {q}"} for u in "0123" for q in "abc"]
    rows[0]["CorrectAnswer"] = "2"                               # item a: 3 rows say 1, 1 says 2 -> majority 1
    with open(t / "all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (t / "split.json").write_text(json.dumps({"held_users": ["3"], "held_questions": ["c"]}))
    out = t / "blocks"
    quarantine(argparse.Namespace(data=str(t / "all.csv"), split=str(t / "split.json"), out_dir=str(out), seal_raw=True))
    assert len(read("cc", out)) == 6 and len(read("hc", out)) == 2, "3 calib students x 2 items, 1 held x 2 items"
    for b, msg in (("ch", "forever"), ("hh", "locked")):
        try:
            read(b, out, lock=t / "no-lock")
            raise AssertionError(f"{b} must be refused")
        except SystemExit as e:
            assert msg in str(e)
    (t / "lock").write_text("x")
    assert len(read("hh", out, lock=t / "lock")) == 1
    assert oct(os.stat(out / "sealed" / "hh.csv").st_mode)[-3:] == "600" and not (t / "all.csv").exists()
    log = (out / "access.log").read_text()
    assert "REFUSED read ch" in log and "REFUSED read hh" in log and "LOCKED ANALYSIS" in log
    assert meta(out, "homes.json")["3"] == "g1" and meta(out, "items.json")["c"]["key"] == 1
    it = meta(out, "items.json")["a"]
    assert it["key"] == 1 and it["keyConflictRows"] == 1 and it["keyRows"] == 3
    print("heldout selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quarantine")
    q.add_argument("--data", required=True); q.add_argument("--split", required=True)
    q.add_argument("--out-dir", required=True); q.add_argument("--seal-raw", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args()
    return quarantine(a) if a.cmd == "quarantine" else selftest()


if __name__ == "__main__":
    sys.exit(main())
