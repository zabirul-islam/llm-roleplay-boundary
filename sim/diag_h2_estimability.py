# Degeneracy vs eligibility threshold on cc folds (calibration data only): per threshold, eligible classes and the
# share of (fold, statistic) that are finite under the registered all-classes-finite rule.
import sys, types, io, contextlib, numpy as np
sys.path.insert(0, "/home/islamm11/ncomms/code")
import analyze
orig, saved = analyze.core, []
analyze.core = lambda d, B=10000, seed=391: (saved.append(d), orig(d, B=5, seed=seed))[1]
a = types.SimpleNamespace(model="qwen3.5-27b", a0="a0-t0.8", draws=5, blocks="/home/islamm11/ncomms/data/eedi-blocks",
    runs="/home/islamm11/ncomms/runs/v3", reps="/home/islamm11/ncomms/runs/v3rep",
    rate="/home/islamm11/ncomms/runs/v3/rate-qwen3.5-27b.jsonl", out="/home/islamm11/ncomms/scratch-ratio.json")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    analyze.cc(a)
import warnings; warnings.simplefilter("ignore")
for thr in (5, 8, 10, 15):
    fin = {"V0": 0, "VS": 0, "D": 0}; ncls = []
    for d in saved:
        P = analyze.Prepared(d)
        both = (P.H1.sum(1) > 0) & (P.H2.sum(1) > 0)
        P.elig = np.where(np.bincount(P.cls[both], minlength=P.C) >= thr)[0]; ncls.append(len(P.elig))
        s = P.stats(np.ones(P.obs.shape[1], int))
        for k in fin: fin[k] += np.isfinite(s[k])
    print(f"elig>={thr}: classes/fold {ncls}  finite folds {fin}")
# rNaN class sizes at threshold 5
for k, d in enumerate(saved):
    P = analyze.Prepared(d); w = np.ones(P.obs.shape[1], int); e = P.elig
    both = (P.H1.sum(1) > 0) & (P.H2.sum(1) > 0); nb = np.bincount(P.cls[both], minlength=P.C)[e]
    h1 = {n: (m * (P.H1 > 0)) @ w / (P.H1 @ w) for n, m in P.mat.items()}
    h2 = {n: (m * (P.H2 > 0)) @ w / (P.H2 @ w) for n, m in P.mat.items()}
    bad = sorted({int(nb[i]) for n in P.mat for i in np.where(~np.isfinite(analyze._class_r(h1[n], h2[n], P.cls, P.C)[e]))[0]})
    print(f"fold {k}: sizes (students with both halves) of classes with undefined r: {bad}")
