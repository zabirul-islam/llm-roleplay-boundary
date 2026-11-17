#!/usr/bin/env python3
"""Prereg v3 calibration-side scoring: simulator arms + supervised baselines on the same 5 grouped folds.

    python3 score3.py embed      --blocks B --out emb.npz                       # BGE-M3 item-text embeddings (text only)
    python3 score3.py rate-build --blocks B --out rate-req.jsonl                # B4 direct-rating requests (one per item)
    python3 score3.py score      --blocks B --runs D --model qwen3.5-27b --emb emb.npz --rate rate-<model>.jsonl
                                 --v2-dir K --out calib-v3.json
    python3 score3.py selftest

Endpoints (frozen 2026-11-14, prereg v3 §5): item difficulty Spearman rho (pooled over the fold; and within class,
classes with >= 5 shared items), MAE, mean bias and calibration slope of item proportion correct; distractors within
item conditional on an incorrect answer (wrong-option JS, modal-wrong agreement; eligibility >= 5 wrong answers in
both arms, else excluded and counted); student structure per class (log ratio of between-student score variance; zero
variance -> class counted as failing) and split-half cross-item consistency; invalid rate.
Baselines: B1 text-conditioned 2PL (ridge from embeddings to b and log a, same ability source as the arm), B2 ridge
embeddings -> logit p, B3 ridge TF-IDF -> logit p, B4 direct model rating with an isotonic map fitted on training
items. Ridge alpha by 5-fold CV over training items, grid 10^-3..10^3.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import isotonic, tfidf  # noqa: E402
from metrics import spearman  # noqa: E402

FOLDS, INVALID, ELIG = 5, 0, 5
RATE_SYSTEM = "You are an experienced maths teacher in England."
RATE_ASK = ("\n\nWhat percentage of 11- to 12-year-old students in England would answer this question correctly? "
            "Reply with only a number from 0 to 100.")
ALPHAS = [10.0 ** e for e in range(-3, 4)]


def logit(p):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return np.log(p / (1 - p))


def ridge_fit(X, y, alpha):
    mx, my = X.mean(0), y.mean()
    Xc = X - mx
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - my))
    return lambda Z: (Z - mx) @ w + my


def ridge_cv(X, y, seed=391):
    idx = np.random.default_rng(seed).permutation(len(y)); parts = np.array_split(idx, 5)
    best = min(ALPHAS, key=lambda a: sum(float(np.mean((ridge_fit(X[np.setdiff1d(idx, p)], y[np.setdiff1d(idx, p)], a)(X[p]) - y[p]) ** 2))
                                         for p in parts))
    return ridge_fit(X, y, best), best


def item_p(rows, key):
    n, k = Counter(), Counter()
    for u, q, c in rows:
        n[q] += 1; k[q] += c == key[q]
    return {q: k[q] / n[q] for q in n}


def difficulty(pred: dict, real: dict) -> dict:
    qs = sorted(set(pred) & set(real))
    x, y = np.array([pred[q] for q in qs]), np.array([real[q] for q in qs])
    slope = float(np.polyfit(x, y, 1)[0]) if x.var() > 0 else float("nan")
    return {"rho": spearman({q: pred[q] for q in qs}, {q: real[q] for q in qs}), "mae": float(np.abs(x - y).mean()),
            "bias": float((x - y).mean()), "calSlope": slope, "items": len(qs)}


def within_class_rho(sim_rows, real_rows, key, home):
    """Mean over classes (>= 5 items with >= 3 students in both arms) of item-difficulty rho inside the class."""
    def per(rows):
        d = defaultdict(lambda: [Counter(), Counter()])
        for u, q, c in rows:
            g = d[home[u]]; g[0][q] += 1; g[1][q] += c == key[q]
        return {g: {q: v[1][q] / v[0][q] for q in v[0] if v[0][q] >= 3} for g, v in d.items()}
    ps, pr = per(sim_rows), per(real_rows)
    rs = [spearman(ps[g], pr[g]) for g in pr if g in ps and len(set(ps[g]) & set(pr[g])) >= 5]
    rs = [r for r in rs if not math.isnan(r)]
    return float(np.mean(rs)) if rs else float("nan"), len(rs)


def js(p: Counter, q: Counter) -> float:
    ks = sorted(set(p) | set(q)); sp, sq = sum(p.values()), sum(q.values())
    P = np.array([p[k] / sp for k in ks]); Q = np.array([q[k] / sq for k in ks]); M = (P + Q) / 2
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / b), 0)))  # noqa: E731
    return (kl(P, M) + kl(Q, M)) / 2


def distractors(sim_rows, real_rows, key) -> dict:
    """Within item, conditional on a wrong answer (invalid output is its own bin). Eligibility >= ELIG wrong in both."""
    def wrong(rows):
        w = defaultdict(Counter)
        for u, q, c in rows:
            if c != key[q]:
                w[q][c] += 1
        return w
    ws, wr = wrong(sim_rows), wrong(real_rows)
    items = set(ws) | set(wr)
    ok = [q for q in items if sum(ws[q].values()) >= ELIG and sum(wr[q].values()) >= ELIG]
    if not ok:
        return {"eligible": 0, "excluded": len(items)}
    modal = [ws[q].most_common(1)[0][0] == wr[q].most_common(1)[0][0] for q in ok]
    return {"eligible": len(ok), "excluded": len(items) - len(ok), "wrongJS": float(np.mean([js(ws[q], wr[q]) for q in ok])),
            "modalWrongAgree": float(np.mean(modal))}


def distractors_v31(sim_rows, real_rows, key) -> dict:
    """prereg v3.1 §5 (Rosalind R2; hh analysis). distractors() above is the development rule kept for calib-v3.json.
    Eligible = >= ELIG real wrong answers. Simulated wrong bins = 3 valid wrong options + invalid (INVALID); real
    invalid bin is 0. Zero simulated wrong answers -> non-estimable, kept in the denominator. JS base 2, unsmoothed;
    modal ties -> option order, invalid last."""
    def wrong(rows):
        w = defaultdict(Counter)
        for u, q, c in rows:
            if c != key[q]:
                w[q][c] += 1
        return w
    ws, wr = wrong(sim_rows), wrong(real_rows)
    elig = sorted(q for q in wr if sum(wr[q].values()) >= ELIG)
    est = [q for q in elig if sum(ws[q].values()) > 0]
    mode = lambda c: min(c, key=lambda k: (-c[k], k == INVALID, k))  # noqa: E731
    return {"eligible": len(elig), "estimable": len(est), "nonEstimable": len(elig) - len(est),
            "wrongJS": float(np.mean([js(ws[q], wr[q]) for q in est])) if est else float("nan"),
            "modalWrongAgree": float(np.mean([mode(ws[q]) == mode(wr[q]) for q in est])) if est else float("nan"),
            "estimableItems": est}


def retained(a) -> int:
    """prereg v3.1 §6: for every duplicated request id, the retained (first) line number and the file's SHA-256."""
    import hashlib
    out = {}
    for f in a.files:
        first, dup = {}, Counter()
        for n, l in enumerate(open(f), 1):
            i = json.loads(l)["id"]
            if i in first:
                dup[i] += 1
            else:
                first[i] = n
        out[Path(f).name] = {"sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest(), "lines": n,
                             "duplicatedIds": len(dup), "retainedLine": {i: first[i] for i in sorted(dup)}}
        print(Path(f).name, n, "lines,", len(dup), "duplicated ids")
    Path(a.out).write_text(json.dumps(out, indent=0))
    return 0


def structure(sim_rows, real_rows, key, home, seed=391) -> dict:
    """Per class: log ratio of between-student score variance (sim/real); split-half consistency = correlation across
    students of the scores on two fixed random halves of the class's items, compared sim vs real."""
    rng = random.Random(seed)
    def by_class(rows):
        d = defaultdict(lambda: defaultdict(list))
        for u, q, c in rows:
            d[home[u]][u].append((q, c == key[q]))
        return d
    S, R = by_class(sim_rows), by_class(real_rows)
    lvr, fail, sh_s, sh_r = [], 0, [], []
    for g in R:
        if g not in S:
            continue
        tr = np.array([np.mean([o for _, o in v]) for v in R[g].values()])
        ts = np.array([np.mean([o for _, o in v]) for v in S[g].values()])
        if len(tr) < 3:
            continue
        vr, vs = tr.var(ddof=1), ts.var(ddof=1)
        if vr == 0 or vs == 0:
            fail += 1
        else:
            lvr.append(math.log(vs / vr))
        qs = sorted({q for v in R[g].values() for q, _ in v}); rng.shuffle(qs); half = set(qs[: len(qs) // 2])
        for D, out in ((S[g], sh_s), (R[g], sh_r)):
            a = [(np.mean([o for q, o in v if q in half] or [np.nan]), np.mean([o for q, o in v if q not in half] or [np.nan])) for v in D.values()]
            a = np.array([x for x in a if not np.isnan(x).any()])
            if len(a) >= 5 and a[:, 0].std() > 0 and a[:, 1].std() > 0:
                out.append((g, float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])))
    common = {g for g, _ in sh_s} & {g for g, _ in sh_r}
    ds, dr = dict(sh_s), dict(sh_r)
    return {"classes": len(lvr) + fail, "zeroVarianceClasses": fail,
            "logVarRatioMedian": float(np.median(lvr)) if lvr else float("nan"),
            "inBand_0.8_1.25": float(np.mean([math.log(0.8) <= x <= math.log(1.25) for x in lvr])) if lvr else float("nan"),
            "splitHalfSim": float(np.mean([ds[g] for g in common])) if common else float("nan"),
            "splitHalfReal": float(np.mean([dr[g] for g in common])) if common else float("nan")}


def load_arm(path: Path, reqs: dict) -> tuple:
    got = {}
    for l in path.open():
        d = json.loads(l)
        if d["failed"]:
            continue
        r = reqs[d["id"]]; c = d["choice"] or INVALID
        if r.get("perm") and c:
            c = r["perm"][c - 1]                    # displayed letter -> original option
        got.setdefault(d["id"], c)                  # duplicate id (two runners on one arm): first row wins, fixed rule
    return got


def score(a) -> int:
    from knob import assign, load, FOLDS as KF, cell
    from calib import fit_2pl
    rows, key, home, text = load(a.blocks)
    cf, itf = assign(rows, key, home)
    real = defaultdict(list)
    for u, q, c in rows:
        if cf[home[u]] == itf[q]:
            real[itf[q]].append((u, q, c))
    runs = Path(a.runs)
    arms = {}
    for f in sorted(runs.glob(f"*-{a.model}.jsonl")):
        name = f.name[: -len(f"-{a.model}.jsonl")]
        if name != "rate":                                            # B4 ratings are a baseline, not an arm
            arms[name] = (f, runs / f"{name}-req.jsonl")
    for s in (0.6, 1.0, 1.4):
        for t in (0.6, 0.8, 1.0):
            f = Path(a.v2_dir) / f"{cell(s, t)}-{a.model}.jsonl"
            if f.exists():
                arms[f"a1-{cell(s, t)}"] = (f, Path(a.v2_dir) / f"{cell(s, t)}-req.jsonl")
    out = {"model": a.model, "arms": {}, "baselines": {}}
    theta = {}
    for name, (f, rq) in arms.items():
        reqs = {}
        for l in rq.open():
            d = json.loads(l); d.pop("messages", None); reqs[d["id"]] = d
        got = load_arm(f, reqs)
        if len(got) < len(reqs):
            out["arms"][name] = {"complete": False, "answered": len(got), "requests": len(reqs)}
            continue
        sim = defaultdict(list)
        for i, c in got.items():
            r = reqs[i]; sim[r["fold"]].append((r["persona"], r["question"], c))
            if "thetaAnchor" in r:
                theta[(r["fold"], r["persona"])] = (r["thetaPop"], r["thetaAnchor"])
        folds = []
        for k in range(FOLDS):
            d = difficulty(item_p(sim[k], key), item_p(real[k], key))
            d["rhoWithinClass"], d["classesWithin"] = within_class_rho(sim[k], real[k], key, home)
            folds.append({**d, **distractors(sim[k], real[k], key), **structure(sim[k], real[k], key, home)})
        out["arms"][name] = {"complete": True, "invalidRate": sum(c == INVALID for c in got.values()) / len(got),
                             "folds": folds, "mean": {m: float(np.nanmean([f.get(m, np.nan) for f in folds]))
                                                      for m in folds[0] if isinstance(folds[0][m], float)}}
    # ---- baselines (item level; same folds, training = classes != k x items != k) ----
    emb = np.load(a.emb) if a.emb else None
    eid = {q: i for i, q in enumerate(emb["ids"])} if emb is not None else {}
    qs_all = sorted(key, key=lambda x: (len(x), x))
    T = tfidf([text[q] for q in qs_all]); tid = {q: i for i, q in enumerate(qs_all)}
    rate = {}
    if a.rate and Path(a.rate).exists():
        for l in open(a.rate):
            d = json.loads(l)
            m = re.search(r"\d+(\.\d+)?", d.get("raw") or "")
            if m and not d["failed"]:
                rate[d["question"]] = min(100.0, float(m.group(0))) / 100
    base = defaultdict(list)
    for k in range(FOLDS):
        train = [(u, q, c) for u, q, c in rows if cf[home[u]] != k and itf[q] != k]
        p_tr = item_p(train, key)
        tr_q = sorted(p_tr, key=lambda x: (len(x), x))
        te_p = item_p(real[k], key); te_q = sorted(te_p, key=lambda x: (len(x), x))
        y = logit([p_tr[q] for q in tr_q])
        preds = {}
        f3, _ = ridge_cv(T[[tid[q] for q in tr_q]], y)
        preds["B3_tfidf"] = dict(zip(te_q, 1 / (1 + np.exp(-f3(T[[tid[q] for q in te_q]])))))
        if emb is not None:
            E = emb["emb"]
            f2, _ = ridge_cv(E[[eid[q] for q in tr_q]], y)
            preds["B2_embed"] = dict(zip(te_q, 1 / (1 + np.exp(-f2(E[[eid[q] for q in te_q]])))))
            users = sorted({u for u, _, _ in train}); ui = {u: i for i, u in enumerate(users)}; ii = {q: i for i, q in enumerate(tr_q)}
            resp = np.full((len(users), len(tr_q)), np.nan)
            for u, q, c in train:
                resp[ui[u], ii[q]] = float(c == key[q])
            _, disc, diff = fit_2pl(resp)
            fb, _ = ridge_cv(E[[eid[q] for q in tr_q]], diff); fa, _ = ridge_cv(E[[eid[q] for q in tr_q]], np.log(disc))
            bq = dict(zip(te_q, fb(E[[eid[q] for q in te_q]]))); aq = dict(zip(te_q, np.exp(fa(E[[eid[q] for q in te_q]]))))
            for src, j in (("pop", 0), ("anchor", 1)):
                n, s = Counter(), defaultdict(float)
                for u, q, c in real[k]:
                    th = theta.get((k, u), (0.0, 0.0))[j]
                    n[q] += 1; s[q] += 1 / (1 + math.exp(-aq[q] * (th - bq[q])))
                preds[f"B1_textIRT_{src}"] = {q: s[q] / n[q] for q in n}
        if rate:
            xr = np.array([rate[q] for q in tr_q if q in rate]); yr = np.array([p_tr[q] for q in tr_q if q in rate])
            iso = isotonic(xr, yr)
            preds["B4_rating_iso"] = {q: float(iso(rate[q])) for q in te_q if q in rate}
            preds["B4_rating_raw"] = {q: rate[q] for q in te_q if q in rate}
        for b, pr in preds.items():
            base[b].append(difficulty(pr, te_p))
    for b, fs in base.items():
        out["baselines"][b] = {"folds": fs, "mean": {m: float(np.nanmean([f[m] for f in fs])) for m in fs[0]}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    rows_out = [(n, v["mean"]) for n, v in out["arms"].items() if v.get("complete")] + [(n, v["mean"]) for n, v in out["baselines"].items()]
    for n, m in sorted(rows_out, key=lambda x: -x[1].get("rho", -9)):
        print(f"{n:26s} rho {m.get('rho', float('nan')):+.3f}  mae {m.get('mae', float('nan')):.3f}  bias {m.get('bias', float('nan')):+.3f}"
              f"  rhoWithin {m.get('rhoWithinClass', float('nan')):+.3f}  modalWrong {m.get('modalWrongAgree', float('nan')):.2f}"
              f"  lvr {m.get('logVarRatioMedian', float('nan')):+.2f}  splitHalf {m.get('splitHalfSim', float('nan')):.2f}/{m.get('splitHalfReal', float('nan')):.2f}")
    print("incomplete:", [n for n, v in out["arms"].items() if not v.get("complete")])
    return 0


def freeze(a) -> int:
    """prereg v3.1 §4 (B4 of pass 3): final fits on ALL of cc, frozen before lock. Reads cc and hc (anchors) only.
    Writes item-level predictions for every non-cc item (B2, B3, B4; B1's b-hat and a-hat), the cc 2PL, and the
    anchor MAP of every held-out student from hc. B1's item p needs the hh assignment, so it is averaged after lock
    from these frozen numbers. Settings are recorded with the output."""
    from knob import load
    from calib import fit_2pl, read_block, theta_map
    import heldout
    rows, key, home, text = load(a.blocks)                       # cc rows (tied-key items excluded)
    items = heldout.meta(a.blocks, "items.json")
    qs_all = sorted(key, key=lambda x: (len(x), x))
    target = sorted((q for q, v in items.items() if q not in key and v["key"] is not None), key=lambda x: (len(x), x))
    allq = sorted(set(qs_all) | set(target), key=lambda x: (len(x), x))
    p_tr = item_p(rows, key)
    tr_q = sorted(p_tr, key=lambda x: (len(x), x))
    y = logit([p_tr[q] for q in tr_q])
    T = tfidf([items[q]["problem"] for q in allq]); tid = {q: i for i, q in enumerate(allq)}
    emb = np.load(a.emb); eid = {q: i for i, q in enumerate(emb["ids"])}; E = emb["emb"]
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    f3, a3 = ridge_cv(T[[tid[q] for q in tr_q]], y)
    f2, a2 = ridge_cv(E[[eid[q] for q in tr_q]], y)
    users = sorted({u for u, _, _ in rows}); ui = {u: i for i, u in enumerate(users)}; ii = {q: i for i, q in enumerate(tr_q)}
    resp = np.full((len(users), len(tr_q)), np.nan)
    for u, q, c in rows:
        resp[ui[u], ii[q]] = float(c == key[q])
    _, disc, diff = fit_2pl(resp)
    fb, ab = ridge_cv(E[[eid[q] for q in tr_q]], diff); fa, aa = ridge_cv(E[[eid[q] for q in tr_q]], np.log(disc))
    rate = {}
    for l in open(a.rate):
        d = json.loads(l)
        m = re.search(r"\d+(\.\d+)?", d.get("raw") or "")
        if m and not d["failed"]:
            rate[d["question"]] = min(100.0, float(m.group(0))) / 100
    xr = np.array([rate[q] for q in tr_q if q in rate]); yr = np.array([p_tr[q] for q in tr_q if q in rate])
    iso = isotonic(xr, yr)
    Et, Tt = E[[eid[q] for q in target]], T[[tid[q] for q in target]]
    pred = {q: {"B2": float(p2), "B3": float(p3), "B1_b": float(b), "B1_a": float(math.exp(la)),
                "B4_raw": rate.get(q), "B4": float(iso(rate[q])) if q in rate else None}
            for q, p2, p3, b, la in zip(target, sig(f2(Et)), sig(f3(Tt)), fb(Et), fa(Et))}
    hc, _ = read_block(a.blocks, "hc")
    anc = defaultdict(list)
    for u, q, c in hc:
        if q in ii:
            anc[u].append((ii[q], float(c == items[q]["key"])))
    theta = {u: {"thetaAnchor": float(theta_map(disc[[i for i, _ in an]], diff[[i for i, _ in an]], np.array([v for _, v in an]))),
                 "anchors": len(an)} for u, an in sorted(anc.items())}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "item-predictions.json").write_text(json.dumps(pred, indent=0, sort_keys=True))
    (out / "irt-cc.json").write_text(json.dumps({"items": tr_q, "a": disc.tolist(), "b": diff.tolist()}))
    (out / "anchors-hc.json").write_text(json.dumps(theta, sort_keys=True))
    (out / "settings.json").write_text(json.dumps({
        "irt": "calib.fit_2pl joint MAP (iters 400, lr 0.05, seed 391) on all cc; identification as in fit_2pl",
        "anchorMAP": "calib.theta_map, N(0,1) prior, 30 Newton steps, cc item parameters fixed; no anchors -> not listed (= prior 0)",
        "ridge": {"grid": ALPHAS, "cv": "5-fold, permutation seed 391", "alpha": {"B3": a3, "B2": a2, "B1_b": ab, "B1_loga": aa}},
        "logitClip": [0.01, 0.99], "tfidf": "audit.tfidf over all scored items' text (cc + target)",
        "isotonic": "audit.isotonic (PAV), unweighted over cc items with a parsed rating; outside the training range = the end value",
        "ratingParse": "first number in the reply, capped at 100, /100; failed or unparsed -> no B4 prediction",
        "ccItems": len(tr_q), "ccStudents": len(users), "targetItems": len(target), "heldOutStudentsWithAnchors": len(theta)},
        indent=1))
    print(json.dumps({"ccItems": len(tr_q), "target": len(target), "anchorStudents": len(theta),
                      "alphas": {"B3": a3, "B2": a2, "B1_b": ab, "B1_loga": aa}}))
    return 0


def embed(a) -> int:
    import heldout
    from sentence_transformers import SentenceTransformer
    items = heldout.meta(a.blocks, "items.json")
    ids = sorted(items, key=lambda x: (len(x), x))
    m = SentenceTransformer("BAAI/bge-m3", device=a.device)
    E = m.encode([items[q]["problem"] for q in ids], batch_size=16, normalize_embeddings=True, show_progress_bar=False)
    np.savez(a.out, ids=np.array(ids), emb=E.astype(np.float32))
    print(f"{len(ids)} items -> {a.out} dim {E.shape[1]}")
    return 0


def rate_build(a) -> int:
    import heldout
    items = heldout.meta(a.blocks, "items.json")
    with open(a.out, "w") as fh:
        for q in sorted(items, key=lambda x: (len(x), x)):
            fh.write(json.dumps({"id": f"rate-q{q}", "persona": "rater", "level": "rater", "question": q, "temperature": 0.0,
                                 "messages": [{"role": "system", "content": RATE_SYSTEM},
                                              {"role": "user", "content": items[q]["problem"] + RATE_ASK}]}) + "\n")
    print(f"{len(items)} rating requests -> {a.out}")
    return 0


def selftest() -> int:
    key = {f"q{i}": 1 for i in range(8)}
    home = {f"u{i}": "g1" if i < 6 else "g2" for i in range(12)}
    real = [(u, q, 1 if (int(u[1:]) + int(q[1:])) % 3 else 2) for u in home for q in key]
    same = structure(real, list(real), key, home)
    assert abs(same["logVarRatioMedian"]) < 1e-12 and same["zeroVarianceClasses"] == 0
    flat = structure([(u, q, 1) for u, q, _ in real], real, key, home)
    assert flat["zeroVarianceClasses"] == flat["classes"], "all-correct sim = zero variance in every class -> failing"
    d = distractors(real, list(real), key)
    assert d["eligible"] == 0 or d["modalWrongAgree"] == 1.0
    many = [(f"v{j}", q, 2 + (j % 2)) for j in range(12) for q in key]
    d2 = distractors(many, [(f"v{j}", q, 2 if j < 9 else 3) for j in range(12) for q in key], key)
    assert d2["eligible"] == 8 and d2["modalWrongAgree"] in (0.0, 1.0)
    dd = difficulty({"a": .2, "b": .5, "c": .9}, {"a": .3, "b": .6, "c": 1.0})
    assert abs(dd["rho"] - 1) < 1e-12 and abs(dd["bias"] + 0.1) < 1e-12 and abs(dd["calSlope"] - 1) < 1e-9
    X = np.random.default_rng(0).normal(size=(60, 5)); yv = X @ np.array([1, -1, .5, 0, 0]) + 0.01
    f, alpha = ridge_cv(X, yv)
    assert float(np.corrcoef(f(X), yv)[0, 1]) > 0.99
    dv = distractors_v31([(f"v{j}", q, 2) for j in range(12) for q in key] + [("w", "q0", 1)],
                         [(f"v{j}", q, 2 if j < 6 else 3) for j in range(12) for q in key], key)
    assert dv["eligible"] == 8 and dv["estimable"] == 8 and dv["modalWrongAgree"] == 1.0, dv
    dz = distractors_v31([(f"v{j}", q, 1) for j in range(12) for q in key], [(f"v{j}", q, 3) for j in range(12) for q in key], key)
    assert dz["eligible"] == 8 and dz["nonEstimable"] == 8 and math.isnan(dz["wrongJS"])
    print("score3 selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed"); e.add_argument("--blocks", required=True); e.add_argument("--out", required=True)
    e.add_argument("--device", default="cpu")
    r = sub.add_parser("rate-build"); r.add_argument("--blocks", required=True); r.add_argument("--out", required=True)
    s = sub.add_parser("score")
    for x in ("--blocks", "--runs", "--model", "--out", "--v2-dir"):
        s.add_argument(x, required=True)
    s.add_argument("--emb", default=""); s.add_argument("--rate", default="")
    d = sub.add_parser("retained"); d.add_argument("--out", required=True); d.add_argument("files", nargs="+")
    z = sub.add_parser("freeze")
    for x in ("--blocks", "--emb", "--rate", "--out"):
        z.add_argument(x, required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    return {"embed": embed, "rate-build": rate_build, "score": score, "freeze": freeze, "retained": retained, "selftest": lambda _a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
