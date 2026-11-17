# Diagnostic: within one cohort, true sampling SD/mean of each statistic over fresh item draws vs the template
# bootstrap's SD and bias (mean T* - T). Calibration-side synthetic data only.
import sys, json, numpy as np
sys.path.insert(0, "/home/islamm11/ncomms/code")
import design, analyze
d = json.load(open("/home/islamm11/ncomms/runs/v3/design3.json"))
pat = design.pattern("/home/islamm11/ncomms/data/eedi-blocks")
keys = ("delta", "V0", "VS", "D")
for kind, sc in (("2pl", "H2a boundary: A0 ratio 0.8"), ("2pl", "H2b boundary: S ratio 0.8")):
    par = d["generators"][kind][sc]["par"]
    rng = np.random.default_rng(11)
    w = design.World(pat, kind, rng)
    Ts, bsd, bbias = {k: [] for k in keys}, {k: [] for k in keys}, {k: [] for k in keys}
    for j in range(30):
        P = analyze.Prepared(w.draw(par, rng)); T = int(P.tmpl.max()) + 1
        e = P.stats(np.ones(P.obs.shape[1], int))
        bs = {k: [] for k in keys}
        for b in range(300):
            m = np.bincount(rng.integers(0, T, T), minlength=T); s = P.stats(m[P.tmpl])
            for k in keys: bs[k].append(s[k])
        for k in keys:
            Ts[k].append(e[k]); bsd[k].append(np.nanstd(bs[k], ddof=1)); bbias[k].append(np.nanmean(bs[k]) - e[k])
    print(kind, sc)
    for k in keys:
        print(f"  {k:5s} trueSD {np.std(Ts[k], ddof=1):.4f}  bootSD {np.mean(bsd[k]):.4f}  ratio {np.mean(bsd[k])/np.std(Ts[k], ddof=1):.2f}  bootBias {np.mean(bbias[k]):+.4f}  mean T {np.mean(Ts[k]):+.4f}")
    sys.stdout.flush()
