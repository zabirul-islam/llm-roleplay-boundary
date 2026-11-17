#!/usr/bin/env python3
"""Simulated students answer real multiple-choice items (Nature Comms plan WP2, docs/ncomms-paper-plan.md).

    build    python3 sim.py build --data all_data.csv --split split-eedi.json --side calib \
                 --questions 50 --personas 10 --out requests.jsonl
    run      python3 sim.py run --requests requests.jsonl --url http://127.0.0.1:8096 --model qwen3.5-27b \
                 --out answers.jsonl [--workers 16]
    selftest python3 sim.py selftest

`build` reads item text and the split only (never another student's answer), so it can run before
calibration. `run` is resume-safe: one JSONL line per finished request, a requeued job skips what is done.
Stdlib only, so it runs with any python in the job.
"""
from __future__ import annotations  # the Empire login node's system python is older than 3.10

import argparse
import csv
import json
import random
import re
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TEMPERATURE, MAX_TOKENS, SEED = 0.8, 8, 391
# Uncalibrated persona baseline (plan H1 baseline 1). Calibrated personas replace this in WP4.
LEVELS = {
    "novice": "You often find maths hard and you make the common mistakes students your age make.",
    "average": "You are an average maths student: you get many questions right but slip on harder ones.",
    "strong": "You are strong at maths and usually, but not always, get questions right.",
}
CAL_LEVEL = "You usually get about {pct}% of maths questions like these right."
SYSTEM = ("You are a {age}-year-old school student in England answering a maths quiz. {level} "
          "Answer as that student really would, not as an expert. Reply with only the letter of your answer.")
LETTER = re.compile(r"\b([ABCD])\b")


OPT = re.compile(r"(?:(?<=\s)|^)([ABCD])[).]\s*")


def permute_options(text: str, rng: random.Random):
    """Shuffle the four options of an item (contamination sensitivity arm, prereg v3). Returns (new_text, perm) where
    perm[i] = original option index (1..4) shown at letter i, or (text, None) if the four options cannot be parsed."""
    ms = [m for m in OPT.finditer(text)]
    letters = [m.group(1) for m in ms]
    if letters[-4:] != list("ABCD"):
        return text, None
    ms = ms[-4:]
    head = text[:ms[0].start()]
    bodies = [text[ms[i].end(): ms[i + 1].start() if i < 3 else len(text)] for i in range(4)]
    seps = [re.search(r"\s*$", b).group(0) for b in bodies]       # keep each slot's trailing layout
    core = [b[: len(b) - len(sp)] if sp else b for b, sp in zip(bodies, seps)]
    order = list(range(4))
    while order == list(range(4)):
        rng.shuffle(order)
    out = head + "".join(f"{'ABCD'[i]}{ms[i].group(0)[1:]}{core[order[i]]}{seps[i]}" for i in range(4))
    return out, [o + 1 for o in order]


def parse_choice(text: str) -> int | None:
    """First standalone A-D letter in the reply -> 1..4 (the dataset's AnswerValue coding)."""
    m = LETTER.search((text or "").strip().upper())
    return "ABCD".index(m.group(1)) + 1 if m else None


def build(a) -> int:
    if a.side == "held" and not (Path(__file__).resolve().parents[1] / "prereg" / "prereg.sha256").exists():
        raise SystemExit("held-out items are locked until the pre-registration is (paper-ncomms/prereg/prereg.sha256)")
    split = json.load(open(a.split))
    held = set(split["held_questions"])
    items = {}
    with open(a.data, newline="") as fh:
        for r in csv.DictReader(fh):                       # item text only
            q = r["QuestionId"]
            if q not in items and ((q in held) == (a.side == "held")):
                items[q] = r["problem"]
    rng = random.Random(SEED)
    qids = sorted(items, key=lambda x: (len(x), x))
    qids = rng.sample(qids, min(a.questions, len(qids)))
    levels = list(LEVELS)
    # Calibrated personas (calib.py personas) replace the uncalibrated three-level baseline when given: ability only,
    # rendered as an expected percentage (prereg v2 §2; no per-item misconception text).
    cal = json.load(open(a.personas_file))["personas"] if a.personas_file else None
    n_personas = len(cal) if cal else a.personas
    with open(a.out, "w") as out:
        for p in range(n_personas):
            level, age = (cal[p]["level"] if cal else levels[p % 3]), rng.choice([11, 12])
            desc = (CAL_LEVEL.format(pct=cal[p]["pct"]) if cal else LEVELS[level])
            for q in qids:
                out.write(json.dumps({"id": f"p{p}-q{q}", "persona": p, "level": level, "age": age, "question": q,
                                      "messages": [{"role": "system", "content": SYSTEM.format(age=age, level=desc)},
                                                   {"role": "user", "content": items[q]}]}) + "\n")
    print(f"{n_personas * len(qids)} requests ({n_personas} personas x {len(qids)} {a.side} questions) -> {a.out}")
    return 0


def ask(url: str, model: str, messages: list, temperature: float = TEMPERATURE, seed: int | None = None) -> str:
    q = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": MAX_TOKENS}
    if seed is not None:                                  # replicate runs (prereg v3.1 §8): per-request sampling seed
        q["seed"] = seed
    body = json.dumps(q).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001 — recorded as a failure row, retried on resume
            last = f"__ERROR__ {e}"
    return last


def writer_lock(out: Path):
    """One writer per output file: two runners on one arm duplicated 51k rows on 2026-11-14. Exits 3 if held."""
    import fcntl
    fh = open(str(out) + ".lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(3)
    return fh


def run(a) -> int:
    reqs = [json.loads(l) for l in open(a.requests) if l.strip()]
    out, done = Path(a.out), set()
    _lk = writer_lock(out)  # noqa: F841 — held until the process exits
    if out.exists():
        for l in out.open():
            if l.strip():
                d = json.loads(l)
                if not d["failed"]:
                    done.add(d["id"])
    todo = [r for r in reqs if r["id"] not in done]
    print(f"{a.model}: {len(reqs)} requests, {len(todo)} to do", flush=True)
    lock, fh = threading.Lock(), out.open("a")

    def one(r):
        raw = ask(a.url, a.model, r["messages"], r.get("temperature", TEMPERATURE), r.get("seed"))  # knob runs set it per request
        failed = raw if raw.startswith("__ERROR__") else None
        rec = {"id": r["id"], "model": a.model, "persona": r["persona"], "level": r.get("level"), "question": r["question"],
               "raw": raw[:200], "choice": None if failed else parse_choice(raw), "think": "<think>" in raw, "failed": failed}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, todo))
    fh.close()
    return 0


def selftest() -> int:
    import tempfile
    o = Path(tempfile.mkdtemp()) / "x.jsonl"
    first = writer_lock(o)
    try:
        writer_lock(o)
        raise AssertionError("second writer was not refused")
    except SystemExit as e:
        assert e.code == 3
    first.close()
    assert parse_choice("B") == 2 and parse_choice("I think C.") == 3 and parse_choice("d") == 4
    assert parse_choice("") is None and parse_choice("none of these") is None
    assert parse_choice("A because") == 1
    for t in ("Q?\n\nA) 1\nB) 2\nC) 3\nD) 4", "Q? A. one B. two C. three D. four", "Q\nA) x     B) y     C) z     D) w\n"):
        new, perm = permute_options(t, random.Random(3))
        assert perm and sorted(perm) == [1, 2, 3, 4] and perm != [1, 2, 3, 4], (t, perm)
        opts = lambda s: [m.group(1) for m in OPT.finditer(s)]  # noqa: E731
        assert opts(new) == list("ABCD")
        # round trip: the text shown at letter i is the original option perm[i]
        orig = [t[m.end():] for m in OPT.finditer(t)]
        shown = [new[m.end():] for m in OPT.finditer(new)]
        for i in range(4):
            assert shown[i].split()[0] == orig[perm[i] - 1].split()[0], (t, new, perm)
    assert permute_options("no options here", random.Random(1)) == ("no options here", None)
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "req.jsonl").write_text("".join(json.dumps({"id": f"x{i}", "persona": 0, "level": "novice", "question": "1",
                                                     "messages": []}) + "\n" for i in range(3)))
    (d / "out.jsonl").write_text(json.dumps({"id": "x0", "failed": None}) + "\n" + json.dumps({"id": "x1", "failed": "__ERROR__"}) + "\n")
    global ask
    calls = []
    ask = lambda url, model, m, t=TEMPERATURE, s=None: calls.append(1) or "B"      # noqa: E731
    run(argparse.Namespace(requests=d / "req.jsonl", out=d / "out.jsonl", url="", model="t", workers=2))
    assert len(calls) == 2, "resume must skip the finished row and retry the failed one"
    print("sim selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True); b.add_argument("--split", required=True)
    b.add_argument("--side", choices=["calib", "held"], default="calib")
    b.add_argument("--questions", type=int, default=50); b.add_argument("--personas", type=int, default=10)
    b.add_argument("--out", required=True)
    b.add_argument("--personas-file", default="", help="calib.py personas output (replaces the 3-level baseline)")
    r = sub.add_parser("run")
    r.add_argument("--requests", required=True); r.add_argument("--url", required=True)
    r.add_argument("--model", required=True); r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=16)
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"build": build, "run": run, "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
