#!/usr/bin/env python3
"""DBE-KT22 secondary replication (prereg v3.1 addendum A1, prereg/addendum-dbe.md). Every DBE rule lives here, frozen
by the addendum lock. The locked EEDI files are reused, never edited. Stages and the data they touch:

  items     metadata only      Questions/Question_Choices -> items-dbe.json (frozen text conversion, options, key)
  itemset   metadata only      held-out items, cc near-duplicate filter, templates -> components-dbe.json
  rate-build / embed           B4 rating requests (DBE wording); BGE-M3 embeddings of the item text
  freeze    cc + hc            2PL on cc, anchors from hc, B1-B3 ridge, B4 isotonic -> frozen/
  design    cc                 calibration-item diagnostic (64 calibration items, 46 templates); validates no DBE inference
  requests  hh (needs lock)    simulator requests r1-5 for the held-out cells
  analyze   hh (needs lock)    contrasts, difficulty metrics, descriptive statistics

Frozen rules (addendum §1-§3):
- Item text = question_rich_text converted to text: an <img> whose src holds "gif.latex?" becomes "$<latex>$" (URL
  decoded); any other image becomes "[image]" and is counted (none exist); <br>, <p>, <div>, <center> and table rows
  end a line; table cells are joined by " | "; other tags are dropped; entities are unescaped; runs of spaces collapse.
  Options follow in ascending choice id, lettered A, B, ...; the key is the one choice with is_correct = true.
- Response = first attempt per student x item: earliest start_time, ties broken by the smaller transaction id. A row
  with a missing or unparseable start_time is excluded and counted. The first attempt counts even when it has no valid
  choice (then incorrect). The first correct or first valid attempt is never selected.
- Simulated answer = sim.parse_choice (first standalone A-D letter), then a range check: a letter beyond the item's
  option count is invalid. Invalid output is incorrect.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
SEED = 391
order = lambda qs: sorted(qs, key=lambda x: (len(x), x))  # noqa: E731

# domain substitutions of the registered EEDI wording (sim.LEVELS, sim.CAL_LEVEL, sim.SYSTEM, score3.RATE_*)
LEVELS_DBE = {
    "novice": "You often find database topics hard and you make the common mistakes students in your course make.",
    "average": "You are an average student in this course: you get many questions right but slip on harder ones.",
    "strong": "You are strong at databases and usually, but not always, get questions right.",
}
CAL_LEVEL_DBE = "You usually get about {pct}% of database questions like these right."
SYSTEM_DBE = ("You are a university student taking an introductory database course, answering a practice quiz. {level} "
              "Answer as that student really would, not as an expert. Reply with only the letter of your answer.")
RATE_SYSTEM_DBE = "You are an experienced database lecturer at a university."
RATE_ASK_DBE = ("\n\nWhat percentage of students in a university introductory database course would answer this question "
                "correctly? Reply with only a number from 0 to 100.")


# ---------------------------------------------------------------- items
class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.out, self.cells, self.other_img = [], None, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "img":
            src = a.get("src") or ""
            if "gif.latex?" in src:
                self._put("$" + unquote(src.split("gif.latex?", 1)[1]) + "$")
            else:
                self.other_img += 1; self._put("[image]")
        elif tag == "br":
            self.out.append("\n")
        elif tag == "tr":
            self.cells = []
        elif tag in ("td", "th") and self.cells is not None:
            self.cells.append([])

    def handle_endtag(self, tag):
        if tag == "tr" and self.cells is not None:
            self.out.append("\n" + " | ".join(" ".join("".join(c).split()) for c in self.cells) + "\n"); self.cells = None
        elif tag in ("p", "div", "center", "table"):
            self.out.append("\n")

    def handle_data(self, data):
        self._put(data)

    def _put(self, s):
        if self.cells is not None and self.cells:
            self.cells[-1].append(s)
        elif self.cells is None:
            self.out.append(s)


def convert(rich: str) -> tuple[str, int]:
    p = _Text(); p.feed(rich); p.close()
    lines = [" ".join(l.split()) for l in "".join(p.out).split("\n")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return text, p.other_img


def items(a) -> int:
    Q = {r["id"]: r for r in csv.DictReader(open(Path(a.meta) / "Questions.csv", newline=""))}
    C = defaultdict(list)
    for r in csv.DictReader(open(Path(a.meta) / "Question_Choices.csv", newline="")):
        C[r["question_id"]].append(r)
    out, other = {}, 0
    for q in order(Q):
        text, n_other = convert(Q[q]["question_rich_text"]); other += n_other
        ch = sorted(C[q], key=lambda r: int(r["id"]))
        keys = [i for i, r in enumerate(ch) if r["is_correct"].strip().lower() == "true"]
        assert len(keys) == 1, (q, keys)
        opts = [" ".join(html.unescape(r["choice_text"]).split()) for r in ch]
        problem = text + "\n" + "\n".join(f"{'ABCDE'[i]}) {o}" for i, o in enumerate(opts))
        out[q] = {"problem": problem, "options": len(ch), "key": keys[0] + 1, "choiceIds": [int(r["id"]) for r in ch],
                  "hasLatex": "gif.latex?" in Q[q]["question_rich_text"], "hasTable": "<table" in Q[q]["question_rich_text"]}
    Path(a.out).write_text(json.dumps(out, indent=0, sort_keys=True))
    print(json.dumps({"items": len(out), "otherImages": other, "options": dict(Counter(v["options"] for v in out.values())),
                      "latex": sum(v["hasLatex"] for v in out.values()), "table": sum(v["hasTable"] for v in out.values())}))
    return 0


# ---------------------------------------------------------------- responses
TIME_FORMATS = ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z")   # DBE: "2019-08-07 17:12:08.722 -0700"


def parse_time(s):
    for f in TIME_FORMATS:
        try:
            return dt.datetime.strptime((s or "").strip(), f)
        except ValueError:
            pass
    return None


def _first_rows(rows: list[dict], items: dict) -> tuple[dict, Counter]:
    """(student, question) -> the retained first-attempt row: earliest timestamp-valid start_time (timezone-aware
    instants), ties broken by the numerically smaller transaction id. Rows with a missing or unparseable start_time
    are excluded; affected rows and pairs are counted (a pair with no valid row is lost)."""
    best, n, badpairs, allpairs = {}, Counter(), set(), set()
    for r in rows:
        q = r["question_id"]
        if q not in items:
            n["unknownItem"] += 1; continue
        k = (r["student_id"], q); allpairs.add(k)
        t = parse_time(r.get("start_time"))
        if t is None:
            n["badTimeRows"] += 1; badpairs.add(k); continue
        o = (t, int(r["id"]))
        if k not in best or o < best[k][0]:
            best[k] = (o, r)
    n["badTimePairs"] = len(badpairs); n["pairsLostToBadTime"] = len(allpairs - set(best))
    return {k: r for k, (_, r) in best.items()}, n


def current_key_score(r: dict, it: dict) -> float:
    cid = r.get("answer_choice_id", "").strip()
    return float(cid.isdigit() and int(cid) in it["choiceIds"] and it["choiceIds"].index(int(cid)) + 1 == it["key"])


def first_attempts(rows: list[dict], items: dict) -> tuple[dict, dict]:
    """(student, question) -> correct 0/1 under the frozen first-attempt rule (current-key scoring); plus counts."""
    first, n = _first_rows(rows, items)
    out = {}
    for k, r in first.items():
        cid = r.get("answer_choice_id", "").strip()
        n["invalidFirst"] += not (cid.isdigit() and int(cid) in items[k[1]]["choiceIds"])
        out[k] = current_key_score(r, items[k[1]])
    n["pairs"] = len(out)
    return out, dict(n)


ANSWER_STATE = {"true": 1.0, "false": 0.0}   # frozen mapping of the recorded score; anything else is missing


def key_inconsistent(rows: list[dict], items: dict, limit: float = 0.05) -> dict:
    """Addendum §1 diagnostic. For item j: among retained first attempts with an interpretable recorded answer_state,
    the fraction whose current-key score differs from the recorded score. Missing/unrecognised states are counted.
    Key-consistent = at least one comparable record and disagreement <= limit; items with none are unassessable."""
    first, _ = _first_rows(rows, items)
    n, bad, missing = Counter(), Counter(), Counter()
    for (u, q), r in first.items():
        rec = ANSWER_STATE.get((r.get("answer_state") or "").strip().lower())
        if rec is None:
            missing[q] += 1; continue
        n[q] += 1; bad[q] += current_key_score(r, items[q]) != rec
    rate = {q: bad[q] / n[q] for q in n}
    seen = {q for _, q in first}
    return {"rates": {q: v for q, v in rate.items() if v > 0}, "comparable": dict(n), "missingState": dict(missing),
            "inconsistent": order(q for q, v in rate.items() if v > limit),
            "consistent": order(q for q, v in rate.items() if v <= limit),
            "unassessable": order(q for q in seen if n[q] == 0)}


def cohort_variance(real: dict, sims: list, qs: set) -> dict:
    """Addendum §3. Per replicate: each student's score = proportion correct over their observed retained first-attempt
    items in the set (students with none excluded and counted); sample variances (n - 1) of real and simulated scores;
    ratio sim/real (positive real and zero simulated -> 0; zero real -> undefined). Mean over the five ratios is
    undefined if any replicate ratio is undefined."""
    per = defaultdict(list)
    for (u, q), c in real.items():
        if q in qs:
            per[u].append(q)
    users = sorted(per); out = []
    rv = np.var([np.mean([real[(u, q)] for q in per[u]]) for u in users], ddof=1) if len(users) > 1 else float("nan")
    for rep in sims:
        sv = np.var([np.mean([float(rep[(u, q)] ) for q in per[u]]) for u in users], ddof=1) if len(users) > 1 else float("nan")
        out.append(float("nan") if not rv > 0 else float(sv / rv))
    return {"studentsIncluded": len(users), "realVariance": float(rv), "ratioPerReplicate": out,
            "meanRatio": float(np.mean(out)) if all(np.isfinite(out)) else float("nan")}


def sim_choice(raw_choice, n_options: int):
    """sim.parse_choice output (1..4 or None) -> chosen option, or None (invalid) when outside the item's range."""
    return raw_choice if raw_choice and raw_choice <= n_options else None


# ---------------------------------------------------------------- item set
def itemset(a) -> int:
    from audit import tfidf, _components
    it = json.load(open(a.items)); sp = json.load(open(a.split))
    held = order(set(map(str, sp["held_questions"])) & set(it)); cal = order(set(it) - set(held))
    X = tfidf([it[q]["problem"] for q in held + cal]); S = X @ X.T
    nh = len(held)
    near = [held[i] for i in range(nh) if (S[i, nh:] >= 0.8).any()]
    prim = [q for q in held if q not in near]
    pi = [held.index(q) for q in prim]
    comps = _components([(prim[i], prim[j]) for i in range(len(prim)) for j in range(i + 1, len(prim))
                         if S[pi[i], pi[j]] >= 0.8], prim)
    compsAll = _components([(held[i], held[j]) for i in range(nh) for j in range(i + 1, nh) if S[i, j] >= 0.8], held)
    out = {"heldOut": len(held), "nearDuplicateOfCalibration": near, "primaryItems": prim, "templates": comps,
           "templateSizes": dict(Counter(map(len, comps))), "allHeldItems": held, "templatesAllHeld": compsAll}
    Path(a.out).write_text(json.dumps(out, indent=0))
    print(json.dumps({k: v if not isinstance(v, list) else len(v) for k, v in out.items()}))
    return 0


# ---------------------------------------------------------------- B4 ratings, embeddings
def rate_build(a) -> int:
    it = json.load(open(a.items))
    with open(a.out, "w") as fh:
        for q in order(it):
            fh.write(json.dumps({"id": f"rate-q{q}", "persona": "rater", "level": "rater", "question": q, "temperature": 0.0,
                                 "messages": [{"role": "system", "content": RATE_SYSTEM_DBE},
                                              {"role": "user", "content": it[q]["problem"] + RATE_ASK_DBE}]}) + "\n")
    print(f"{len(it)} DBE rating requests -> {a.out}")
    return 0


def embed(a) -> int:
    from sentence_transformers import SentenceTransformer
    it = json.load(open(a.items)); ids = order(it)
    m = SentenceTransformer("BAAI/bge-m3", device=a.device)
    E = m.encode([it[q]["problem"] for q in ids], batch_size=16, normalize_embeddings=True, show_progress_bar=False)
    np.savez(a.out, ids=np.array(ids), emb=E.astype(np.float32))
    print(f"{len(ids)} items -> {a.out} dim {E.shape[1]}")
    return 0


def ratings(path) -> dict:
    out = {}
    for l in open(path):
        d = json.loads(l); m = re.search(r"\d+(\.\d+)?", d.get("raw") or "")
        if m and not d["failed"]:
            out[d["question"]] = min(100.0, float(m.group(0))) / 100
    return out


# ---------------------------------------------------------------- freeze (cc + hc)
def block(blocks, name, it):
    import dbe
    return first_attempts(dbe.read(name, blocks), it)


def freeze(a) -> int:
    from calib import fit_2pl, theta_map
    from score3 import ridge_cv, logit
    from audit import tfidf, isotonic
    it = json.load(open(a.items)); comp = json.load(open(a.components)); sp = json.load(open(a.split))
    held = set(map(str, sp["held_questions"]))
    cc, ncc = block(a.blocks, "cc", it); hc, nhc = block(a.blocks, "hc", it)
    users = order({u for u, _ in cc}); tr = order({q for _, q in cc})
    ui, ii = {u: i for i, u in enumerate(users)}, {q: i for i, q in enumerate(tr)}
    resp = np.full((len(users), len(tr)), np.nan)
    for (u, q), c in cc.items():
        resp[ui[u], ii[q]] = c
    _, disc, diff = fit_2pl(resp)
    p_tr = {q: float(np.nanmean(resp[:, ii[q]])) for q in tr}
    allq = order(set(tr) | held)
    T = tfidf([it[q]["problem"] for q in allq]); tid = {q: i for i, q in enumerate(allq)}
    emb = np.load(a.emb); eid = {q: i for i, q in enumerate(emb["ids"])}; E = emb["emb"]
    y = logit([p_tr[q] for q in tr])
    f2, a2 = ridge_cv(E[[eid[q] for q in tr]], y); f3, a3 = ridge_cv(T[[tid[q] for q in tr]], y)
    fb, ab = ridge_cv(E[[eid[q] for q in tr]], diff); fa, aa = ridge_cv(E[[eid[q] for q in tr]], np.log(disc))
    rate = ratings(a.rate); xr = [q for q in tr if q in rate]
    iso = isotonic(np.array([rate[q] for q in xr]), np.array([p_tr[q] for q in xr]))
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    tg = order(comp["allHeldItems"]); Et, Tt = E[[eid[q] for q in tg]], T[[tid[q] for q in tg]]
    pred = {q: {"B2": float(p2), "B3": float(p3), "B1_b": float(b), "B1_a": float(math.exp(la)),
                "B4_raw": rate.get(q), "B4": float(iso(rate[q])) if q in rate else None}
            for q, p2, p3, b, la in zip(tg, sig(f2(Et)), sig(f3(Tt)), fb(Et), fa(Et))}
    anc = defaultdict(list)
    for (u, q), c in hc.items():
        if q in ii:
            anc[u].append((ii[q], c))
    theta = {u: {"thetaAnchor": float(theta_map(disc[[i for i, _ in v]], diff[[i for i, _ in v]], np.array([c for _, c in v]))),
                 "anchors": len(v)} for u, v in sorted(anc.items())}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "item-predictions.json").write_text(json.dumps(pred, indent=0, sort_keys=True))
    (out / "irt-cc.json").write_text(json.dumps({"items": tr, "a": disc.tolist(), "b": diff.tolist()}))
    (out / "anchors-hc.json").write_text(json.dumps(theta, sort_keys=True))
    settings = {"ridgeAlpha": {"B2": a2, "B3": a3, "B1_b": ab, "B1_log_a": aa}, "ccCounts": ncc, "hcCounts": nhc,
                "ccStudents": len(users), "ccItems": len(tr), "anchoredStudents": len(theta),
                "ratedItems": len(rate), "primaryWithoutRating": [q for q in tg if q not in rate],
                "rhoRawRatingVsCcP": float(spearmanr([rate[q] for q in xr], [p_tr[q] for q in xr])[0])}
    (out / "settings.json").write_text(json.dumps(settings, indent=1))
    print(json.dumps(settings))
    return 0


# ---------------------------------------------------------------- design (cc)
def design(a) -> int:
    """Calibration-item diagnostic (addendum §3; it validates no DBE interval or test). Criterion fixed before the run: under each
    generator the Wilson 95% lower limit of one-sided non-coverage over 500 cohorts is <= 0.05. Pattern: 434 seeded
    calibration students x 64 seeded calibration items (the hh shape), their real cc first-attempt missingness, DBE
    templates, one cohort (no classes), R = 5. Truths from 400 independent item draws (design.bone)."""
    from multiprocessing import Pool
    import analyze
    import design as D
    from audit import tfidf, _components
    it = json.load(open(a.items))
    cc, _ = block(a.blocks, "cc", it)
    rng = random.Random(SEED)
    users = rng.sample(order({u for u, _ in cc}), 434); qs = rng.sample(order({q for _, q in cc}), 64)
    ui, ii = {u: i for i, u in enumerate(users)}, {q: i for i, q in enumerate(qs)}
    M = np.zeros((len(users), len(qs)), bool)
    for (u, q) in cc:
        if u in ui and q in ii:
            M[ui[u], ii[q]] = True
    X = tfidf([it[q]["problem"] for q in qs]); S = X @ X.T
    comps = _components([(qs[i], qs[j]) for i in range(len(qs)) for j in range(i + 1, len(qs)) if S[i, j] >= 0.8], qs)
    t = {q: k for k, m in enumerate(comps) for q in m}; tmpl = np.array([t[q] for q in qs])
    cls = np.zeros(len(users), int)
    pat = {"M": M, "cls": cls, "tmpl": tmpl, "half": analyze.halves(M, cls, tmpl)}
    D.RHO_B = json.load(open(a.settings))["rhoRawRatingVsCcP"]
    out = {"pattern": {"students": len(users), "items": len(qs), "cells": int(M.sum()), "templates": int(tmpl.max()) + 1},
           "settings": {"sims": a.sims, "B": a.draws, "R": D.R, "kappa": D.KAPPA, "rhoB": D.RHO_B, "referenceDraws": D.N_REF,
                        "criterion": "Wilson 95% lower limit of primary non-coverage <= 0.05 under each generator"},
           "generators": {}}
    for gi, kind in enumerate(("2pl", "misspec")):
        par = D.calibrate(pat, kind)["H1 delta=0.1"]["par"]
        jobs = [(pat, kind, par, a.draws, 950_000_000 + 100_000 * gi + i) for i in range(a.sims)]
        with Pool(a.procs) as pool:
            res = pool.map(D.bone, jobs)
        g = out["generators"][kind] = {"par": par, **D.bsum(res), "cohorts": res}
        print(kind, json.dumps({k: g[k] for k in g if k not in ("cohorts", "par")}), flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


# ---------------------------------------------------------------- requests and analysis (hh; need the addendum lock)
ARMS = ["a2-t0.6", "a0-t0.8", "a1-t0.8"]


def requests(a) -> int:
    import dbe
    it = json.load(open(a.items)); comp = json.load(open(a.components)); fz = Path(a.frozen)
    irt = json.load(open(fz / "irt-cc.json")); anchors = json.load(open(fz / "anchors-hc.json"))
    disc, diff = np.array(irt["a"]), np.array(irt["b"])
    prim = set(comp["allHeldItems"])                     # all held-out items: primary set + all-64 sensitivity
    hh, n = first_attempts(dbe.read("hh", a.blocks), it)
    cells = sorted(k for k in hh if k[1] in prim)
    users = order({u for u, _ in cells}); r = a.rep
    z = dict(zip(users, np.random.default_rng(SEED * 1000 + r).standard_normal(len(users))))
    lvl0 = {u: random.Random(f"{SEED}-r{r}-a0-{u}").choice(list(LEVELS_DBE)) for u in users}
    pct = lambda th: int(round(100 * float(np.mean(1 / (1 + np.exp(-disc * (th - diff)))))))  # noqa: E731
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        kind, t = arm.rsplit("-t", 1)
        with (out / f"{arm}-req.jsonl").open("w") as fh:
            for u, q in cells:
                desc = (LEVELS_DBE[lvl0[u]] if kind == "a0" else CAL_LEVEL_DBE.format(
                    pct=pct(float(z[u])) if kind == "a1" else pct(anchors.get(u, {"thetaAnchor": 0.0})["thetaAnchor"])))
                fh.write(json.dumps({"id": f"dbe-u{u}-q{q}", "persona": u, "question": q, "options": it[q]["options"],
                                     "temperature": float(t), "rep": r,
                                     "seed": int(hashlib.sha256(f"{SEED}-dbe-r{r}-{arm}-{u}-{q}".encode()).hexdigest()[:8], 16),
                                     "messages": [{"role": "system", "content": SYSTEM_DBE.format(level=desc)},
                                                  {"role": "user", "content": it[q]["problem"]}]}) + "\n")
    print(json.dumps({"rep": r, "cells": len(cells), "students": len(users), "firstAttemptCounts": n}))
    return 0


def analyze_(a) -> int:
    import analyze
    import dbe
    from score3 import difficulty
    it = json.load(open(a.items)); comp = json.load(open(a.components)); fz = Path(a.frozen)
    pred = json.load(open(fz / "item-predictions.json")); anchors = json.load(open(fz / "anchors-hc.json"))
    hhrows = dbe.read("hh", a.blocks)
    hh, nhh = first_attempts(hhrows, it); kc = key_inconsistent(hhrows, it)
    prim = order(comp["primaryItems"]); tid = {q: k for k, m in enumerate(comp["templates"]) for q in m}
    tidAll = {q: k for k, m in enumerate(comp["templatesAllHeld"]) for q in m}
    usable = [q for q in prim if pred[q]["B4"] is not None]
    usableAll = [q for q in order(comp["allHeldItems"]) if pred[q]["B4"] is not None]
    cells = {k: v for k, v in hh.items() if k[1] in set(usableAll)}
    users = order({u for u, _ in cells}); ui = {u: i for i, u in enumerate(users)}

    def sims(arm):
        reps = []
        for r in range(1, 6):
            d = Path(a.runs) / f"r{r}"
            reqs = {x["id"]: x for x in map(json.loads, open(d / f"{arm}-req.jsonl"))}
            got = {}
            for l in open(d / f"{arm}-{a.model}.jsonl"):
                x = json.loads(l)
                if not x["failed"]:
                    got.setdefault(x["id"], x["choice"])
            assert len(got) == len(reqs), (arm, r, len(got), len(reqs))
            reps.append({(reqs[i]["persona"], reqs[i]["question"]): sim_choice(c, reqs[i]["options"]) for i, c in got.items()})
        return reps

    S = {arm: sims(arm) for arm in ARMS}

    def build(qs, base, tmap=None):
        tmap = tmap or tid
        ii = {q: i for i, q in enumerate(qs)}
        obs = np.zeros((len(users), len(qs)), bool); real = np.zeros(obs.shape)
        for (u, q), c in cells.items():
            if q in ii:
                obs[ui[u], ii[q]] = True; real[ui[u], ii[q]] = c
        def mat(reps):
            Mx = np.zeros((5,) + obs.shape)
            for k, rep in enumerate(reps):
                for (u, q), ch in rep.items():
                    if q in ii:
                        Mx[k, ui[u], ii[q]] = float(ch is not None and ch == it[q]["key"])
            return Mx
        tmpl = np.array([tmap[q] for q in qs]); cls = np.zeros(len(users), int)
        return lambda arm: {"obs": obs, "real": real, "sims": {"S": mat(S[arm]), "A0": mat(S["a0-t0.8"])},
                            "base": np.array([base[q] for q in qs], float), "cls": cls, "tmpl": tmpl,
                            "half": analyze.halves(obs, cls, tmpl)}

    def contrast(d):
        """Descriptive (addendum §3): point estimates, the five replicate values, Monte Carlo SE (generation
        variability only). No interval, no p-value, no bootstrap."""
        P = analyze.Prepared(d); e = P.stats(np.ones(P.obs.shape[1], int))
        dr = [x - e["rhoB"] for x in e["rhoS_r"]]
        return {"rhoS_r": e["rhoS_r"], "rhoS": float(np.mean(e["rhoS_r"])), "rhoBase": e["rhoB"], "delta": e["delta"],
                "delta_r": dr, "mcSE_generationOnly": analyze.mc_se(dr), "itemsDropped": P.dropped,
                "items": int((d["obs"].sum(0) > 0).sum())}

    b4 = {q: pred[q]["B4"] for q in usable}
    mk = build(usable, b4)
    res = {"registration": "osf.io/ygvdf addendum A1 (descriptive)", "firstAttemptCounts": nhh,
           "items": len(usable), "itemsWithoutB4": [q for q in prim if q not in usable], "students": len(users),
           "keyConsistency (answer_state vs current key, hh first attempts)": kc}
    main = contrast(mk("a2-t0.6")); res["A2 vs B4 (primary DBE contrast)"] = main
    res["E-pop A1 vs B4"] = contrast(mk("a1-t0.8"))
    preal = {q: float(np.mean([c for (u, qq), c in cells.items() if qq == q])) for q in usable}
    th = lambda u: anchors.get(u, {"thetaAnchor": 0.0})["thetaAnchor"]  # noqa: E731
    b1 = {q: float(np.mean([1 / (1 + math.exp(-pred[q]["B1_a"] * (th(u) - pred[q]["B1_b"]))) for (u, qq) in cells if qq == q]))
          for q in usable}
    base = {"B1_anchor": b1, "B2": {q: pred[q]["B2"] for q in usable}, "B3": {q: pred[q]["B3"] for q in usable}, "B4": b4}
    for k in ("B1_anchor", "B2", "B3"):
        res[f"A2 vs {k}"] = contrast(build(usable, base[k])("a2-t0.6"))
    met = {k: difficulty(v, preal) for k, v in base.items()}
    for arm in ARMS:
        per = []
        for rep in S[arm]:
            acc = defaultdict(list)
            for (u, q), ch in rep.items():
                if (u, q) in cells:
                    acc[q].append(float(ch is not None and ch == it[q]["key"]))
            per.append(difficulty({q: float(np.mean(v)) for q, v in acc.items()}, preal))
        met[arm] = {k: float(np.mean([m[k] for m in per])) for k in ("rho", "mae", "bias", "calSlope")}
        inv = Counter(); tot = Counter()
        for rep in S[arm]:
            for (u, q), ch in rep.items():
                if (u, q) in cells:
                    tot[it[q]["options"]] += 1; inv[it[q]["options"]] += ch is None
        met[arm]["invalidRateByOptions"] = {str(k): inv[k] / tot[k] for k in sorted(tot)}
    res["difficulty"] = met
    res["sensitivity all held-out items"] = contrast(build(usableAll, {q: pred[q]["B4"] for q in usableAll}, tidAll)("a2-t0.6"))
    sets = {"primary": set(usable), "all held-out items": set(usableAll)}
    for name, keep in (("no-LaTeX", lambda q: not it[q]["hasLatex"]), ("no-table", lambda q: not it[q]["hasTable"]),
                       ("four-option only", lambda q: it[q]["options"] == 4),
                       ("key-consistent items only", lambda q: q in kc["consistent"])):
        qs = [q for q in usable if keep(q)]                    # each filter alone, on the primary set; no refit
        res[f"sensitivity {name}"] = contrast(build(qs, b4)("a2-t0.6")); sets[name] = set(qs)
    simsc = {arm: [{k: float(ch is not None and ch == it[k[1]]["key"]) for k, ch in rep.items()} for rep in S[arm]] for arm in ARMS}
    res["cohort variance ratio (sim/real; all held-out students, not classes)"] = {
        name: {arm: cohort_variance(cells, simsc[arm], qs) for arm in (ARMS if name == "primary" else ["a2-t0.6"])}
        for name, qs in sets.items()}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("items", "students")} |
                     {"A2 vs B4": {k: main[k] for k in main if k != "rhoS_r"}}))
    return 0


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    rich = ('Consider <img src="http://latex.codecogs.com/gif.latex?T_{1}" border="0"/> and '
            '<img src="http://latex.codecogs.com/gif.latex?R%20%5Cbowtie%20S"/>:<br><center><table border="1">'
            '<tr><th>T1</th><th>T2</th></tr><tr><td>read(A)</td><td></td></tr></table></center>What &amp; why?')
    t, other = convert(rich)
    assert "$T_{1}$" in t and "$R \\bowtie S$" in t and other == 0, t
    assert "T1 | T2" in t and "read(A) |" in t and "What & why?" in t, t
    assert convert('x <img src="a.png"/> y')[1] == 1
    it = {"7": {"choiceIds": [11, 12, 13], "key": 2, "options": 3}}
    rows = [{"id": "5", "student_id": "u", "question_id": "7", "start_time": "2021-03-01 10:00:00.000 -0700", "answer_choice_id": "11"},
            {"id": "4", "student_id": "u", "question_id": "7", "start_time": "2021-03-01 10:00:00.000 -0700", "answer_choice_id": "12"},
            {"id": "1", "student_id": "u", "question_id": "7", "start_time": "2021-03-02 10:00:00 -0700", "answer_choice_id": "12"},
            {"id": "9", "student_id": "v", "question_id": "7", "start_time": "", "answer_choice_id": "12"},
            {"id": "8", "student_id": "w", "question_id": "7", "start_time": "2021-03-01 09:00:00.500 -0700", "answer_choice_id": ""},
            {"id": "10", "student_id": "w", "question_id": "7", "start_time": "2021-03-01 09:05:00.100 -0700", "answer_choice_id": "12"}]
    fa, n = first_attempts(rows, it)
    assert fa[("u", "7")] == 1.0                     # tie on start_time -> smaller id (4, correct choice 12)
    assert ("v", "7") not in fa and n["badTimeRows"] == 1  # missing time -> excluded, counted
    assert fa[("w", "7")] == 0.0 and n["invalidFirst"] == 1   # invalid first attempt counts, scored incorrect
    assert parse_time("2019-08-07 17:12:08.722 -0700").utcoffset().total_seconds() == -7 * 3600 and parse_time("x") is None
    for r, st in zip(rows, ("false", "true", "true", "true", "false", "true")):
        r["answer_state"] = st
    kc = key_inconsistent(rows, it)
    assert kc["inconsistent"] == [] and kc["rates"] == {} and kc["consistent"] == ["7"], kc   # first attempts agree
    rows[1]["answer_state"] = "false"
    assert key_inconsistent(rows, it)["inconsistent"] == ["7"]
    for r in rows:
        r["answer_state"] = ""
    k2 = key_inconsistent(rows, it)
    assert k2["unassessable"] == ["7"] and k2["missingState"]["7"] == 2 and k2["consistent"] == [], k2
    assert n["badTimePairs"] == 1 and n["pairsLostToBadTime"] == 1, n
    real = {("a", "1"): 1.0, ("a", "2"): 0.0, ("b", "1"): 1.0, ("b", "2"): 1.0, ("c", "1"): 0.0}
    cv = cohort_variance(real, [dict(real), {k: 1.0 for k in real}], {"1", "2"})
    assert cv["studentsIncluded"] == 3 and abs(cv["ratioPerReplicate"][0] - 1) < 1e-12 and cv["ratioPerReplicate"][1] == 0
    assert math.isnan(cohort_variance({("a", "1"): 1.0, ("b", "1"): 1.0}, [{("a", "1"): 0.0, ("b", "1"): 1.0}], {"1"})["meanRatio"])
    assert sim_choice(4, 3) is None and sim_choice(3, 3) == 3 and sim_choice(None, 4) is None
    assert "maths" not in " ".join(LEVELS_DBE.values()) + CAL_LEVEL_DBE + SYSTEM_DBE + RATE_ASK_DBE
    print("dbe_prep selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    spec = {"items": ["--meta", "--out"], "itemset": ["--items", "--split", "--out"], "rate-build": ["--items", "--out"],
            "embed": ["--items", "--out"], "freeze": ["--blocks", "--items", "--components", "--split", "--emb", "--rate", "--out"],
            "design": ["--blocks", "--items", "--settings", "--out"],
            "requests": ["--blocks", "--items", "--components", "--frozen", "--out-dir"],
            "analyze": ["--blocks", "--items", "--components", "--frozen", "--runs", "--out"]}
    for name, req in spec.items():
        p = sub.add_parser(name)
        for k in req:
            p.add_argument(k, required=True)
        if name == "embed":
            p.add_argument("--device", default="cpu")
        if name == "design":
            p.add_argument("--sims", type=int, default=500); p.add_argument("--draws", type=int, default=1000)
            p.add_argument("--procs", type=int, default=90)
        if name == "requests":
            p.add_argument("--rep", type=int, required=True)
        if name == "analyze":
            p.add_argument("--model", default="qwen3.5-27b")
    sub.add_parser("selftest")
    a = ap.parse_args()
    fn = {"items": items, "itemset": itemset, "rate-build": rate_build, "embed": embed, "freeze": freeze, "design": design,
          "requests": requests, "analyze": analyze_, "selftest": lambda _a: selftest()}
    return fn[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
