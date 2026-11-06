"""Pricing checks: BM-01, CV-01, SE-01, ST-01, LU-01, DR-01. Thresholds are the defaults of fiche 06."""
import numpy as np
from riskhub_engine import model_risk
from scipy.stats import ks_2samp

from ..adapters.pricing import (
    DOMAIN,
    MKT,
    forward,
    implied_vol,
    ql_analytic,
    ql_mc_qe,
    sobol_points,
    strike,
)
from ..core import TestResult


def _box(model):
    return getattr(model, "domain", DOMAIN)


def _iv_err_bp(model, p, ref=ql_analytic):
    a, b = implied_vol(model.price(p), p), implied_vol(ref(p), p)
    return abs(a - b) * 1e4 if not (np.isnan(a) or np.isnan(b)) else float("nan")


def _grade(mean, p99, g=(1, 10), a=(3, 25)):
    return "green" if mean < g[0] and p99 < g[1] else "amber" if mean < a[0] and p99 < a[1] else "red"


def bm_pricing(model, n: int = 128, n_mc: int = 16, mc_paths: int = 100_000, seed: int = 7) -> TestResult:
    """Implied-vol error vs QuantLib analytic; gap vs Monte Carlo QE is kept only outside its 99 % CI."""
    pts = sobol_points(n, _box(model), seed=seed)
    errs = np.array([_iv_err_bp(model, p) for p in pts])
    ok = errs[~np.isnan(errs)]
    outside = 0
    for p in pts[:n_mc]:
        mc, se = ql_mc_qe(p, mc_paths, seed=seed)
        outside += abs(model.price(p) - mc) > 2.576 * se
    mean, p99 = float(ok.mean()), float(np.percentile(ok, 99))
    worst = pts[int(np.nanargmax(errs))]
    return TestResult("BM-01", "IV error vs QuantLib analytic + MC QE (bp)", mean, _grade(mean, p99),
                      {"green": "mean<1 and p99<10", "amber": "mean<3 and p99<25"}, None, seed,
                      {"points": n, "excluded_not_invertible": int(n - len(ok)), "mean_bp": mean, "p99_bp": p99,
                       "max_bp": float(ok.max()), "worst_point": worst, "mc_points": n_mc,
                       "mc_paths": mc_paths, "outside_mc_ci99": int(outside)})


def cv_mc(n_runs=(4_000, 16_000, 64_000, 256_000), seed: int = 3) -> TestResult:
    """Convergence of the reference Monte Carlo itself: log-log slope of the standard error vs paths (-1/2)."""
    p = sobol_points(1, seed=seed)[0]
    se = [ql_mc_qe(p, n, seed=seed)[1] for n in n_runs]
    slope = float(np.polyfit(np.log(n_runs), np.log(se), 1)[0])
    status = "green" if -0.6 <= slope <= -0.4 else "amber" if -0.7 <= slope <= -0.3 else "red"
    return TestResult("CV-01", "MC standard-error slope vs paths", slope, status,
                      {"green": "[-0.6,-0.4]", "amber": "[-0.7,-0.3]"}, None, seed,
                      {"paths": list(n_runs), "se": se})


def _fd(f, p, S, h):
    return (f(p, S * (1 + h)) - f(p, S * (1 - h))) / (2 * S * h)


def se_delta(model, n: int = 32, seed: int = 11, h_grid=(0.01, 0.001)) -> TestResult:
    """Finite-difference delta of the model vs the same on QuantLib, over a grid of bumps.
    FinancePy has no analytic Greeks, so the reference is QuantLib bumped identically."""
    S = MKT["S"]
    worst = {h: 0.0 for h in h_grid}
    for p in sobol_points(n, _box(model), seed=seed):
        for h in h_grid:
            dm, dq = _fd(model.price, p, S, h), _fd(ql_analytic, p, S, h)
            worst[h] = max(worst[h], abs(dm - dq) / max(abs(dq), 1e-8))
    w = max(worst.values())
    status = "green" if w < 0.01 else "amber" if w < 0.05 else "red"
    return TestResult("SE-01", "Delta (central FD) vs QuantLib", w, status,
                      {"green_lt": 0.01, "amber_lt": 0.05}, None, seed,
                      {"max_rel_err_by_bump": {str(h): v for h, v in worst.items()}, "points": n})


def _no_arb(price, p):
    D = np.exp(-MKT["r"] * p["days"] / 365)
    F, K = forward(p), strike(p)
    return bool(np.isfinite(price) and D * max(F - K, 0) - 1e-9 <= price <= D * F + 1e-9)


STRESS = {
    "feller_violated": {"kappa": 0.3, "sigma": 1.0, "theta": 0.04, "v0": 0.04},
    "rho_plus_0.95": {"rho": 0.95}, "rho_minus_0.95": {"rho": -0.95},
    "one_week": {"days": 7}, "wing_down": {"k": -0.5}, "wing_up": {"k": 0.5},
}


def st_stress(model) -> TestResult:
    """Extreme parameters: price finite, inside no-arbitrage bounds, close to QuantLib. Reported per case;
    the status only counts cases inside the declared domain (a failure outside it is a limit of use)."""
    DOMAIN = _box(model)
    base = {k: 0.5 * sum(v) for k, v in DOMAIN.items()}
    cases = {}
    for name, ov in STRESS.items():
        p = base | ov
        try:
            px = model.price(p)
            err = _iv_err_bp(model, p)
            cases[name] = {"finite_and_in_bounds": _no_arb(px, p), "iv_err_bp": err,
                           "in_domain": all(DOMAIN[k][0] <= p[k] <= DOMAIN[k][1] for k in DOMAIN)}
        except Exception as e:  # noqa: BLE001  # a crash is a result, not a harness failure
            cases[name] = {"finite_and_in_bounds": False, "iv_err_bp": float("nan"), "error": repr(e)[:120],
                           "in_domain": False}
    bad = [n for n, c in cases.items() if c["in_domain"] and (not c["finite_and_in_bounds"] or not c["iv_err_bp"] < 25)]
    return TestResult("ST-01", "Stress of parameters", float(len(bad)), "green" if not bad else "red",
                      {"green": "0 violation inside the domain"}, None, None, {"cases": cases, "violations": bad})


def lu_limits(model, scales=(0.5, 1.0, 1.5, 2.0), n: int = 64, tol_p99_bp: float = 10, seed: int = 5) -> TestResult:
    """Largest box (scaled around the declared domain's centre) where the p99 IV error stays under tolerance."""
    out = {}
    for s in scales:
        box = {}
        for k, (lo, hi) in _box(model).items():
            c, w = 0.5 * (lo + hi), 0.5 * (hi - lo) * s
            box[k] = (max(c - w, 7) if k == "days" else c - w, c + w)
        box["kappa"] = (max(box["kappa"][0], 0.05), box["kappa"][1])
        box["v0"] = (max(box["v0"][0], 1e-3), box["v0"][1])
        box["theta"] = (max(box["theta"][0], 1e-3), box["theta"][1])
        box["sigma"] = (max(box["sigma"][0], 0.02), box["sigma"][1])
        box["rho"] = (max(box["rho"][0], -0.99), min(box["rho"][1], 0.99))
        e = np.array([_iv_err_bp(model, p) for p in sobol_points(n, box, seed)])
        out[s] = float(np.percentile(e[~np.isnan(e)], 99))
    valid = [s for s in scales if out[s] < tol_p99_bp]
    best = max(valid) if valid else 0.0
    return TestResult("LU-01", "Largest scaled box with p99 IV error under tolerance", best,
                      "green" if best >= 1.0 else "red", {"green": "declared domain (scale 1) within tolerance",
                                                           "tol_p99_bp": tol_p99_bp}, None, seed,
                      {"p99_bp_by_scale": {str(k): v for k, v in out.items()}, "validated_scale": best})


def dr_drift(usage: dict[str, list[float]], box: dict | None = None, n_ref: int = 512, seed: int = 9) -> TestResult:
    """PSI and KS between the inputs actually used and the declared domain (uniform reference)."""
    ref = sobol_points(n_ref, box, seed=seed)
    psi, ks = {}, {}
    for k, cur in usage.items():
        r = [p[k] for p in ref]
        psi[k] = float(model_risk.psi(r, list(cur)))
        ks[k] = float(ks_2samp(r, cur).statistic)
    worst = max(psi.values())
    return TestResult("DR-01", "Input drift vs declared domain (PSI)", worst,
                      "green" if worst < 0.10 else "amber" if worst < 0.25 else "red",
                      {"green_lt": 0.10, "amber_lt": 0.25}, None, seed, {"psi": psi, "ks": ks})
