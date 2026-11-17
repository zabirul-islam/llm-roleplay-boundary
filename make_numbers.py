"""Every number the manuscript reports, generated from the committed result files -> numbers.tex.

    python3 make_numbers.py          # writes numbers.tex
    python3 make_numbers.py --check  # also: every macro used in main.tex, no undefined macro, no bare decimal in
                                     # the abstract/results/discussion outside macros, figure/table refs excepted
main.tex must take every result number from a macro, so no number is typed by hand.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
J = lambda p: json.load(open(HERE / p))  # noqa: E731
H1, SEC, DBE, EXP, AS, ASX = (J("results/hh/h1.json"), J("results/hh/secondary.json"), J("results/dbe/results-dbe.json"),
                              J("results/explore/explore-postlock.json"), J("results/assist/results-assist.json"),
                              J("results/assist/explore-optionmatch.json"))
M: dict[str, str] = {}


def f(x: float, d: int = 2, sign: bool = False) -> str:
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return r"\ensuremath{" + s + "}" if s.startswith("-") else s   # math-mode minus sign


def n(x: int) -> str:
    return f"{x:,}".replace(",", "{,}")


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}\\%"


def put(name: str, val: str) -> None:
    assert re.fullmatch(r"[A-Za-z]+", name) and name not in M, name
    M[name] = val


# ---------------------------------------------------------------- EEDI confirmatory (h1.json)
e, d = H1["estimates"], SEC["difficulty"]
put("eStudents", n(H1["data"]["students"])); put("eClasses", n(H1["data"]["classes"])); put("eCells", n(H1["data"]["cells"]))
put("eItemsReg", n(H1["data"]["items"])); put("eItemsNoCell", n(len(H1["data"]["itemsNoObservedCell"])))
put("eItems", n(H1["data"]["items"] - len(H1["data"]["itemsNoObservedCell"])))
put("eRhoA", f(float(np.mean(e["rhoS_r"])), 3)); put("eRhoB", f(e["rhoB"], 3)); put("eDelta", f(e["delta"], 3, True))
put("eRhoAmin", f(min(e["rhoS_r"]), 3)); put("eRhoAmax", f(max(e["rhoS_r"]), 3))
put("eP", f(H1["p"]["H1"], 3)); put("eLo", f(H1["ci95"]["delta"][0], 2)); put("eHi", f(H1["ci95"]["delta"][1], 2))
put("eMcSE", f(H1["mcSE"]["delta"], 3)); put("eBootSE", f(H1["bootSE"]["delta"], 3)); put("eB", n(H1["B"]))
eb = SEC["E-anchor A2 vs baselines"]
for k, nm in (("B1_anchor", "One"), ("B2", "Two"), ("B3", "Three")):
    put(f"eDelta{nm}", f(eb[k]["delta"], 2, True)); put(f"ePsup{nm}", f(eb[k]["p_superiority"], 2))
put("ePsupFour", f(eb["B4"]["p_superiority"], 2))
put("ePfive", f(eb["B4"]["p_margin"]["0.05"], 3)); put("ePfifteen", f(eb["B4"]["p_margin"]["0.15"], 3))
for k, nm in (("A2 t0.6", "A"), ("A1 t0.8", "AOne"), ("A0 t0.8", "AZero"), ("B4", "B"), ("B1_anchor", "BOne"),
              ("B2", "BTwo"), ("B3", "BThree")):
    for m, mn in (("rho", "Rho"), ("mae", "Mae"), ("bias", "Bias"), ("calSlope", "Slope")):
        put(f"eD{nm}{mn}", f(d[k][m], 2, sign=(m == "bias")))
ep = SEC["E-pop A1 vs B4"]; put("eEpopDelta", f(ep["delta"], 3, True)); put("eEpopP", f(ep["p_margin"]["0.1"], 3))
sens = {"tau": "tau ablation A2 t0.8 vs B4", "kfive": "budget k=5", "kten": "budget k=10", "ktwenty": "budget k=20",
        "nofig": "sensitivity no-figure", "home": "sensitivity home-group-only", "allheld": "sensitivity all-172",
        "gemma": "Gemma A2 vs Gemma B4"}
dl = []
for nm, k in sens.items():
    s = SEC[k]; dl.append(s["delta"])
    put(f"eS{nm}Delta", f(s["delta"], 3, True)); put(f"eS{nm}Items", n(s["items"]))
put("eSensMin", f(min(dl + [e["delta"]]), 2, True)); put("eSensMax", f(max(dl + [e["delta"]]), 2, True))
put("eNoFigRemoved", n(len(SEC["sensitivity no-figure"]["figureItemsRemoved"])))
gd = SEC["Gemma difficulty"]; put("eGemmaRhoA", f(gd["A2 t0.6"]["rho"], 2)); put("eGemmaRhoB", f(gd["B4 (Gemma)"]["rho"], 2))
wc = SEC["within-class rho"]; put("eWithinMin", f(min(v["rho"] for v in wc.values()), 2))
put("eWithinMax", f(max(v["rho"] for v in wc.values()), 2))
pa = SEC["persona adherence"]
put("eAdhR", f(pa["A2 t0.6"]["pearson"], 2)); put("eAdhSlope", f(pa["A2 t0.6"]["olsSlope"], 2))
put("eAdhGap", f(pa["A2 t0.6"]["meanAchievedMinusDeclared"], 2, True)); put("eAdhN", n(pa["A2 t0.6"]["students x reps"]))
dz = SEC["distractors"]; put("eDisItems", n(dz["A2 t0.6"]["eligible"]))
put("eJsA", f(dz["A2 t0.6"]["wrongJS"], 2)); put("eJsAOne", f(dz["A1 t0.8"]["wrongJS"], 2))
put("eJsAZero", f(dz["A0 t0.8"]["wrongJS"], 2)); put("eJsUni", f(dz["uniform guessing"]["wrongJS"], 3))
op = SEC["option permutation"]
dr = [op[a]["rho"] - op[a]["rhoPermuted"] for a in op]
put("ePermMin", f(min(dr), 2)); put("ePermMax", f(max(dr), 2))
# H2 (descriptive): simulated / real class variance = exp(median log ratio)
for arm, nm in (("S", "A"), ("A0", "AZero")):
    v = [np.exp(r["medianLogRatio"]) for r in H1["h2"]["arms"][arm]]
    put(f"eVar{nm}Min", f(min(v), 2)); put(f"eVar{nm}Max", f(max(v), 2))
# exploratory (post-lock)
x = EXP["EEDI wrong answers"]
put("xRhoAB", f(EXP["EEDI item-level rho(A2 mean, B4)"], 2)); put("xRhoAReal", f(EXP["EEDI rho(A2 mean, real)"], 2))
put("xWrongItems", n(x["items"])); put("xJsUniReal", f(x["JS(uniform, real) median"], 3))
put("xShareNearUni", pct(x["share items JS(uniform, real) < 0.05"])); put("xJsSimReal", f(x["JS(sim, real) median"], 2))
put("xTopReal", f(x["top wrong option share, real (mean)"], 2)); put("xTopSim", f(x["top wrong option share, sim (mean)"], 2))

# ---------------------------------------------------------------- DBE (descriptive)
put("dStudents", n(DBE["students"])); put("dItems", n(DBE["items"])); put("dPairs", n(DBE["firstAttemptCounts"]["pairs"]))
dp = DBE["A2 vs B4 (primary DBE contrast)"]
put("dRhoA", f(dp["rhoS"], 2)); put("dRhoB", f(dp["rhoBase"], 2)); put("dDelta", f(dp["delta"], 2, True))
put("dMcSE", f(dp["mcSE_generationOnly"], 3))
dd = DBE["difficulty"]
for k, nm in (("a2-t0.6", "A"), ("a1-t0.8", "AOne"), ("a0-t0.8", "AZero"), ("B4", "B"), ("B1_anchor", "BOne"),
              ("B2", "BTwo"), ("B3", "BThree")):
    put(f"dD{nm}Rho", f(dd[k]["rho"], 2)); put(f"dD{nm}Slope", f(dd[k]["calSlope"], 2))
cv = DBE["cohort variance ratio (sim/real; all held-out students, not classes)"]["primary"]
put("dVarA", f(cv["a2-t0.6"]["meanRatio"], 2)); put("dVarAZero", f(cv["a0-t0.8"]["meanRatio"], 2))
au = EXP["DBE audit"]; per = au["per item (real p, sim p, B4)"]
rp = np.array([v[0] for v in per.values()]); sp = np.array([v[1] for v in per.values()]); b4 = np.array([v[2] for v in per.values()])
put("dSimAcc", f(au["sim A2 accuracy mean"], 2)); put("dRealAcc", f(au["real accuracy mean"], 2))
put("dCeiling", pct(float(np.mean(sp >= 0.99))))
CONTEXT = {"71", "75", "81", "147", "160", "161"}  # items needing the course tables (results/explore/SUMMARY.md)
keep = np.array([q not in CONTEXT for q in per])
assert keep.sum() == len(per) - len(CONTEXT)
put("dCeilingNoCtx", pct(float(np.mean(sp[keep] >= 0.99)))); put("dCtxItems", n(len(CONTEXT)))
put("dNoCtxItems", n(int(keep.sum())))
put("dNoCtxRhoA", f(spearmanr(sp[keep], rp[keep])[0], 2)); put("dNoCtxRhoB", f(spearmanr(b4[keep], rp[keep])[0], 2))
put("dRhoAB", f(au["item-level rho(sim A2, B4)"], 2))

# ---------------------------------------------------------------- ASSIST (descriptive)
ap = AS["A2 vs B4 (descriptive)"]
put("aStudents", n(AS["students"])); put("aItems", n(AS["items"])); put("aNoResp", n(len(AS["itemsNoRetainedResponse"])))
put("aItemsPrimary", n(AS["items"] + len(AS["itemsNoRetainedResponse"])))
put("aPairs", n(AS["responseCounts"]["pairs"]))
put("aRhoA", f(ap["rhoS"], 3)); put("aRhoB", f(ap["rhoBase"], 3)); put("aDelta", f(ap["delta"], 3, True))
put("aDeltaMin", f(min(ap["delta_r"]), 3, True)); put("aDeltaMax", f(max(ap["delta_r"]), 3, True))
put("aMcSE", f(ap["mcSE_generationOnly"], 3))
put("aEpopDelta", f(AS["E-pop A1 vs B4"]["delta"], 3, True))
for k, nm in (("B1_anchor", "One"), ("B2", "Two"), ("B3", "Three")):
    put(f"aDelta{nm}", f(AS[f"A2 vs {k}"]["delta"], 2, True))
ad = AS["difficulty"]
for k, nm in (("a2-t0.6", "A"), ("a1-t0.8", "AOne"), ("a0-t0.8", "AZero"), ("B4", "B"), ("B1_anchor", "BOne"),
              ("B2", "BTwo"), ("B3", "BThree")):
    put(f"aD{nm}Rho", f(ad[k]["rho"], 2)); put(f"aD{nm}Slope", f(ad[k]["calSlope"], 2))
    put(f"aD{nm}Bias", f(ad[k]["bias"], 2, True))
for k, nm in (("sensitivity all held-out eligible items", "All"), ("sensitivity option-match correctness", "Opt"),
              ("sensitivity 3-4 option items only", "Many")):
    s = AS[k]; put(f"aS{nm}Items", n(s["items"])); put(f"aS{nm}RhoA", f(s["rhoS"], 2))
    put(f"aS{nm}RhoB", f(s["rhoBase"], 2)); put(f"aS{nm}Delta", f(s["delta"], 2, True))
acv = AS["cohort variance ratio (sim/real; all held-out students)"]
put("aVarMin", f(min(v["meanRatio"] for v in acv.values()), 2)); put("aVarMax", f(max(v["meanRatio"] for v in acv.values()), 2))
c = ASX["counts_discreteScore_optionMatch"]
put("aOptAgree", pct(ASX["agreement"], 1)); put("aOptMiss", n(c["1,0"])); put("aOptRev", n(c["0,1"]))
put("aOptPairs", n(ASX["pairs"])); put("aOptNotFound", n(ASX["answerTextNotFoundLiterallyInOptionsField"]))

# ---------------------------------------------------------------- design validation (pre-lock)
D4, D5, DA = J("results/calib-v3/design4-h1.json"), J("results/calib-v3/design5-boundary.json"), J("results/assist/design-assist.json")
g4 = D4["generators"]
put("vPowZeroLo", f(min(g4[g]["H1 delta=0.0"]["P(reject H1)"][0] for g in g4), 2))
put("vPowZeroHi", f(max(g4[g]["H1 delta=0.0"]["P(reject H1)"][0] for g in g4), 2))
put("vPowDevLo", f(min(g4[g]["H1 delta=-0.237"]["P(reject H1)"][0] for g in g4), 2))
put("vPowDevHi", f(max(g4[g]["H1 delta=-0.237"]["P(reject H1)"][0] for g in g4), 2))
put("vSims", n(D4["settings"]["sims"]))
put("vDesItems", n(D4["pattern"]["items"])); put("vDesTemplates", n(D4["pattern"]["templates"]))
put("vDesClasses", n(D4["pattern"]["classes"]))
for g, nm in (("2pl", "Two"), ("misspec", "Mis")):
    w = D5["generators"][g]["nonCoverage (all, primary)"]["wilson"]
    put(f"vNonCov{nm}", f(w[0], 3)); put(f"vNonCov{nm}Lo", f(w[1], 3)); put(f"vNonCov{nm}Hi", f(w[2], 3))
ga = DA["generators"]["2pl"]
put("vaNonCov", f(ga["oneSidedNonCoverage"]["wilson"][0], 3)); put("vaNonCovHi", f(ga["oneSidedNonCoverage"]["wilson"][2], 4))
put("vaCov", f(ga["twoSidedCoverage"]["wilson"][0], 3)); put("vaCovLo", f(ga["twoSidedCoverage"]["wilson"][1], 3))
put("vaSims", n(ga["n"]))

# ---------------------------------------------------------------- added after the CI-coding pass (draft v1)
SP = J("results/explore/item-spread-eedi.json")
put("xSdReal", f(SP["SD real"], 2)); put("xSdA", f(SP["a2-t0.6"]["SD pred (mean over reps)"], 2))
put("xSdB", f(SP["B4"]["SD pred"], 2))
put("eGemmaPairA", f(SEC["Gemma A2 vs Gemma B4"]["rhoS"], 2)); put("eGemmaPairB", f(SEC["Gemma A2 vs Gemma B4"]["rhoBase"], 2))
put("ePsupThreeX", f(eb["B3"]["p_superiority"], 3))
put("dBaseMin", f(min(dd[k]["rho"] for k in ("B1_anchor", "B2", "B3")), 2))
put("dBaseMax", f(max(dd[k]["rho"] for k in ("B1_anchor", "B2", "B3")), 2))
put("xDbeRhoMean", f(au["rho(sim A2, real)"], 2))
put("dSdSim", f(au["sim p SD across items"], 2)); put("dSdReal", f(au["real p SD across items"], 2))
put("aVarA", f(acv["a2-t0.6"]["meanRatio"], 2))
put("eVarAllAMin", f(min(min(np.exp(r["medianLogRatio"]) for r in H1["h2"]["arms"]["S"]), cv["a2-t0.6"]["meanRatio"], acv["a2-t0.6"]["meanRatio"]), 2))
put("eVarAllAMax", f(max(max(np.exp(r["medianLogRatio"]) for r in H1["h2"]["arms"]["S"]), cv["a2-t0.6"]["meanRatio"], acv["a2-t0.6"]["meanRatio"]), 2))

# ---------------------------------------------------------------- added after the simulated review (draft v1.1)
RC, CH = J("results/explore/review-checks-eedi.json"), J("results/calib-v3/choices-6.json")
put("eUpper", f(RC["H1"]["upper95_onesided"], 2))
assert abs(RC["H1"]["p"] - H1["p"]["H1"]) < 1e-12 and RC["H1"]["ci95"] == H1["ci95"]["delta"]   # re-run reproduces h1.json
put("eInvalid", n(sum(v["invalid"] for v in RC["invalidOutput"].values())))
put("eAnswersPerArm", n(RC["invalidOutput"]["a2-t0.6"]["answers"]))
put("eRhoBraw", f(RC["B4ties"]["rhoB4raw"], 2)); put("eIsoDistinct", n(RC["B4ties"]["distinctIso"]))
put("xWrongMedian", n(int(RC["wrongAnswersPerEligibleItem"]["median"]))); put("xUniTopExp", f(RC["uniformChoiceTopShareExpected_sameCounts"], 2))
ea = CH["E-anchor"]
put("devRhoA", f(ea["all"]["a2-t0.6"]["rho"], 2)); put("devRhoB", f(ea["baselines"]["B4_rating_iso"]["rho"], 2))
cons = [r["medianD_conditional"] for r in H1["h2"]["arms"]["S"]]
put("eConsMin", f(min(cons), 2)); put("eConsMax", f(max(cons), 2))
cons0 = [r["medianD_conditional"] for r in H1["h2"]["arms"]["A0"]]
put("eConsAZeroMin", f(min(cons0), 2)); put("eConsAZeroMax", f(max(cons0), 2))
put("eHtwoClasses", n(H1["h2"]["arms"]["S"][0]["nLogRatio"]))
# B2 (Rosalind 2026-11-16): uniform guessing on each arm's own estimable items, per replicate (distractor_pairs.py)
DP = J("results/hh/distractor-pairs.json")["arms"]
for k, nm in (("A2 t0.6", "A"), ("A1 t0.8", "AOne"), ("A0 t0.8", "AZero")):
    assert abs(DP[k]["armJS"] - dz[k]["wrongJS"]) < 1e-12 and min(DP[k]["armMinusUniform_r"]) > 0
    put(f"eJsUniPair{nm}", f(DP[k]["uniformJS"], 3))
cov = {k: v["estimable"] for k, v in dz.items() if v.get("estimable")}
put("eCovAMin", n(min(cov["A2 t0.6"]))); put("eCovAMax", n(max(cov["A2 t0.6"])))
put("eCovAZeroMin", n(min(cov["A0 t0.8"]))); put("eCovAZeroMax", n(max(cov["A0 t0.8"])))
put("eCovAOneMin", n(min(cov["A1 t0.8"]))); put("eCovAOneMax", n(max(cov["A1 t0.8"])))
for k, nm in (("a2-t0.6", "A"), ("a1-t0.8", "AOne"), ("a0-t0.8", "AZero"), ("B4", "B"), ("B1_anchor", "BOne"), ("B2", "BTwo"), ("B3", "BThree")):
    put(f"dD{nm}Mae", f(dd[k]["mae"], 2)); put(f"dD{nm}Bias", f(dd[k]["bias"], 2, True)); put(f"aD{nm}Mae", f(ad[k]["mae"], 2))

for k, nm in (("sensitivity all held-out items", "All"), ("sensitivity no-LaTeX", "NoTex"), ("sensitivity four-option only", "Four"),
              ("sensitivity key-consistent items only", "Key")):
    put(f"dS{nm}Items", n(DBE[k]["items"])); put(f"dS{nm}Delta", f(DBE[k]["delta"], 2, True))
cvd = DBE["cohort variance ratio (sim/real; all held-out students, not classes)"]
for k, nm in (("all held-out items", "All"), ("no-LaTeX", "NoTex"), ("four-option only", "Four"), ("key-consistent items only", "Key")):
    put(f"dSVar{nm}", f(cvd[k]["a2-t0.6"]["meanRatio"], 2))

put("dEpopDelta", f(DBE["E-pop A1 vs B4"]["delta"], 2, True))
ba = eb["beats_all (min Δ over B1-B4)"]; put("eBeatsP", f(ba["p_superiority"], 2))


put("eBpopRho", f(d["B1_pop (mean over reps)"]["rho"], 3)); put("eBpopMae", f(d["B1_pop (mean over reps)"]["mae"], 3))
put("eBpopBias", f(d["B1_pop (mean over reps)"]["bias"], 3)); put("eBpopSlope", f(d["B1_pop (mean over reps)"]["calSlope"], 2))

# ---------------------------------------------------------------- Supplementary Data 1 (Rosalind B3)
SD1 = HERE / "supplementary" / "Supplementary_Data_1.csv"


def sd1_rows() -> list[tuple]:
    """Every five-replicate value array in the result files, plus the EEDI H2 per-replicate summaries and the matched
    distractor comparison. Long format: dataset, section, quantity, replicate, value, source."""
    rows = []

    def reps(ds, sec, src, obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                reps(ds, sec, src, v, f"{path}/{k}" if path else k)
        elif isinstance(obj, list) and len(obj) == 5 and all(isinstance(x, (int, float)) for x in obj):
            rows.extend((ds, sec, path, r + 1, x, src) for r, x in enumerate(obj))

    for ds, src, js_ in (("EEDI", "results/hh/secondary.json", SEC), ("DBE-KT22", "results/dbe/results-dbe.json", DBE),
                         ("FoundationalASSIST", "results/assist/results-assist.json", AS)):
        for k, v in js_.items():
            sec = "cohort-variance" if k.startswith("cohort variance") else "distractor-coverage" if k == "distractors" \
                else "within-class-counts" if k.startswith("within-class") else "contrasts"
            reps(ds, sec, src, {k: v})

    # EEDI stores rhoS_r per replicate and one deterministic baseline rho, so the replicate contrast is exact:
    # delta_r = rhoS_r - rhoBase, whose mean must equal the stored delta (Rosalind closure, B3).
    def eedi_deltas(obj, path):
        if isinstance(obj, dict) and isinstance(obj.get("rhoS_r"), list) and "rhoBase" in obj and "delta" in obj:
            dr = [x - obj["rhoBase"] for x in obj["rhoS_r"]]
            assert abs(float(np.mean(dr)) - obj["delta"]) < 1e-12, path
            rows.append(("EEDI", "contrasts", f"{path}/rhoBase", "all", obj["rhoBase"], "results/hh/secondary.json"))
            rows.extend(("EEDI", "contrasts", f"{path}/delta_r (rhoS_r - rhoBase)", r + 1, x, "results/hh/secondary.json")
                        for r, x in enumerate(dr))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                eedi_deltas(v, f"{path}/{k}" if path else k)

    eedi_deltas(SEC, "")
    q = {r[2] for r in rows if r[0] == "EEDI" and r[1] == "contrasts"}
    missing = [p for p in q if p.endswith("/rhoS_r") and p[:-7] + "/delta_r (rhoS_r - rhoBase)" not in q]
    assert not missing, missing   # every EEDI simulator-rho series has its contrast series
    for arm, nm in (("S", "A2 t0.6"), ("A0", "A0 t0.8")):
        for r, x in enumerate(H1["h2"]["arms"][arm]):
            for q in ("medianLogRatio", "nLogRatio", "medianD_conditional", "nD", "simZeroVar", "simUndefinedR"):
                rows.append(("EEDI", "H2", f"{nm}/{q}", r + 1, x[q], "results/hh/h1.json"))
            rows.append(("EEDI", "H2", f"{nm}/exp(medianLogRatio)", r + 1, math.exp(x["medianLogRatio"]), "results/hh/h1.json"))
    for k, v in J("results/hh/distractor-pairs.json")["arms"].items():
        for q in ("estimable", "armJS_r", "uniformJS_r", "armMinusUniform_r"):
            rows.extend(("EEDI", "distractors-matched", f"{k}/{q}", r + 1, x, "results/hh/distractor-pairs.json")
                        for r, x in enumerate(v[q]))
    return rows


def write_sd1() -> None:
    import csv
    SD1.parent.mkdir(exist_ok=True)
    with open(SD1, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "section", "quantity", "replicate", "value", "source"])
        w.writerows(sd1_rows())


def write() -> None:
    lines = ["% generated by make_numbers.py from results/*.json — do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")


def check() -> int:
    tex = (HERE / "main.tex").read_text()
    used = set(re.findall(r"\\([a-z][A-Za-z]+)", tex))
    unused = [k for k in M if k not in used]
    undefined = sorted(k for k in used if re.fullmatch(r"[eadx][A-Z][A-Za-z]*", k) and k not in M)
    body = tex.split(r"\begin{document}")[-1].split(r"\section*{Methods}")[0]
    body = re.sub(r"(?s)\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", "", body)
    body = re.sub(r"%.*", "", body)
    bare = sorted(set(re.findall(r"(?<![\\\w{.])\d+\.\d+", body)))
    secs = {(r[0], r[1]) for r in sd1_rows()}
    refs = set(re.findall(r"Supplementary Data~1, (EEDI|DBE-KT22|FoundationalASSIST) section \\texttt\{([a-z-]+|H2)\}", tex))
    broken = sorted(refs - secs)   # every Supplementary Data 1 pointer in the manuscript must resolve
    print(f"macros {len(M)}; unused {unused}; undefined {undefined}; bare decimals before Methods {bare}; "
          f"SD1 pointers {len(refs)}, broken {broken}")
    return 1 if (undefined or bare or broken) else 0   # ponytail: unused macros are harmless, listed only


if __name__ == "__main__":
    # selftest of the formatter
    assert f(-0.0063, 3, True) == r"\ensuremath{-0.006}" and f(0.0054, 3, True) == "+0.005" and f(0.2219, 3) == "0.222"
    assert n(24284) == "24{,}284" and pct(0.812, 1) == "81.2\\%"
    write(); write_sd1(); print("numbers.tex:", len(M), "macros;", SD1.name, len(sd1_rows()), "rows")
    sys.exit(check() if "--check" in sys.argv else 0)
