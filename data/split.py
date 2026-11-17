#!/usr/bin/env python3
"""Freeze the calibration / held-out splits for the Nature Comms paper (plan WP1, docs/ncomms-paper-plan.md).

Reads ONLY id and date columns, never answers, so writing the split does not "open" the held-out data.
Deterministic (seed 391). Output: one JSON per dataset with the held-out ids; its SHA-256 goes into MANIFEST.md
before any held-out answer is read.

    python3 split.py eedi <all_data.csv> <out.json>      # whole real classes (GroupId) + questions held out
    python3 split.py dbe  <Transaction.csv> <out.json>   # later course cohorts (first year >= 2021) + questions
    python3 split.py selftest

Stdlib only, so it runs on the Empire login node next to the data (the data never comes back to git:
EEDI is CC BY-NC-ND 4.0).
"""
import collections
import csv
import json
import random
import sys

SEED, HOLD_FRAC, MIN_CLASS = 391, 0.30, 10


def _pick(ids, frac, rng):
    ids = sorted(ids, key=lambda x: (len(x), x))
    return sorted(rng.sample(ids, round(len(ids) * frac)), key=lambda x: (len(x), x))


def split_eedi(rows):
    """Each student belongs to the class (GroupId) where they gave most answers — 947 students appear in more
    than one class, so holding out whole classes needs one home class per student. Classes with fewer than
    MIN_CLASS home students stay in calibration (too small to measure class-level spread)."""
    by_user = collections.defaultdict(collections.Counter)
    questions = set()
    for r in rows:
        by_user[r["UserId"]][r["GroupId"]] += 1
        questions.add(r["QuestionId"])
    home = {u: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for u, c in by_user.items()}
    size = collections.Counter(home.values())
    rng = random.Random(SEED)
    eligible = [g for g, n in size.items() if n >= MIN_CLASS]
    held_classes = _pick(eligible, HOLD_FRAC, rng)
    held_q = _pick(questions, HOLD_FRAC, rng)
    held_set = set(held_classes)
    held_users = sorted((u for u, g in home.items() if g in held_set), key=lambda x: (len(x), x))
    return {"dataset": "eedi", "seed": SEED, "unit": "home GroupId (class) + QuestionId",
            "held_classes": held_classes, "held_users": held_users, "held_questions": held_q,
            "counts": {"users": len(home), "classes": len(size), "eligible_classes": len(eligible),
                       "held_classes": len(held_classes), "held_users": len(held_users),
                       "questions": len(questions), "held_questions": len(held_q)}}


def split_dbe(rows):
    """A student's cohort is the year of their first answer; cohorts from 2021 on are held out (the later
    offerings of the course, so calibration only ever sees earlier classes)."""
    first, questions = {}, set()
    for r in rows:
        y = int(r["start_time"][:4])
        s = r["student_id"]
        first[s] = min(first.get(s, y), y)
        questions.add(r["question_id"])
    rng = random.Random(SEED)
    held_users = sorted((s for s, y in first.items() if y >= 2021), key=lambda x: (len(x), x))
    held_q = _pick(questions, HOLD_FRAC, rng)
    years = collections.Counter(first.values())
    return {"dataset": "dbe-kt22", "seed": SEED, "unit": "first-answer year (cohort >= 2021 held out) + question_id",
            "held_users": held_users, "held_questions": held_q,
            "counts": {"users": len(first), "cohorts": dict(sorted(years.items())), "held_users": len(held_users),
                       "questions": len(questions), "held_questions": len(held_q)}}


def selftest():
    rows = []
    for u in range(60):
        g = f"g{u // 12}"                                   # 5 classes x 12 students
        for q in range(10):
            rows.append({"UserId": str(u), "GroupId": g, "QuestionId": str(q)})
    rows.append({"UserId": "0", "GroupId": "g4", "QuestionId": "0"})   # one stray answer in another class
    a, b = split_eedi(rows), split_eedi(list(reversed(rows)))
    assert a == b, "split must not depend on row order"
    assert a["counts"]["held_classes"] == 2 and a["counts"]["held_questions"] == 3
    home0 = "g0"                                           # user 0 answered 10 in g0, 1 in g4
    assert ("0" in a["held_users"]) == (home0 in a["held_classes"])
    held = set(a["held_classes"])
    assert all((f"g{int(u) // 12}" in held) for u in a["held_users"])
    d = [{"student_id": str(s), "question_id": str(q), "start_time": f"{2019 + s % 4}-08-01"} for s in range(40) for q in range(5)]
    d.append({"student_id": "3", "question_id": "0", "start_time": "2019-01-01"})   # student 3 started 2019
    x = split_dbe(d)
    assert "3" not in x["held_users"] and "2" in x["held_users"]
    assert x == split_dbe(list(reversed(d)))
    print("split selftest ok")


def main():
    if sys.argv[1:] == ["selftest"]:
        return selftest()
    kind, src, out = sys.argv[1:4]
    cols = {"eedi": ("UserId", "GroupId", "QuestionId"), "dbe": ("student_id", "question_id", "start_time")}[kind]
    with open(src, newline="") as fh:
        rows = [{k: r[k] for k in cols} for r in csv.DictReader(fh)]   # ids and dates only
    res = split_eedi(rows) if kind == "eedi" else split_dbe(rows)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps(res["counts"]))


if __name__ == "__main__":
    main()
