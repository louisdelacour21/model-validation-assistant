"""PricingModel adapters (call prices) and the independent QuantLib references.

A point is a dict: v0, kappa, theta, sigma, rho, k = ln(K/F), days (calendar days, ACT/365).
Market constants (spot, rate, dividend) come from MKT."""
from math import exp

import numpy as np
import QuantLib as ql
from scipy.optimize import brentq
from scipy.stats import norm, qmc

MKT = {"S": 100.0, "r": 0.02, "q": 0.01}
PARAMS = ("v0", "kappa", "theta", "sigma", "rho", "k", "days")
# Declared domain of use: the box inside which the model is claimed to be accurate.
DOMAIN = {"v0": (0.01, 0.10), "kappa": (0.5, 4.0), "theta": (0.01, 0.10), "sigma": (0.1, 0.8),
          "rho": (-0.9, -0.1), "k": (-0.3, 0.3), "days": (30, 730)}


def forward(p: dict) -> float:
    return MKT["S"] * exp((MKT["r"] - MKT["q"]) * p["days"] / 365)


def strike(p: dict) -> float:
    return forward(p) * exp(p["k"])


class FinancePyHeston:
    """FinancePy semi-analytic Heston (Lewis). Third-party code: this is the independent campaign."""
    name = "financepy-heston"
    domain = DOMAIN

    def price(self, p: dict, spot: float | None = None) -> float:
        from financepy.models.heston import Heston
        from financepy.products.equity import EquityVanillaOption
        from financepy.utils.date import Date
        from financepy.utils.global_types import OptionTypes
        d0 = Date(1, 1, 2026)
        s = MKT["S"] if spot is None else spot
        opt = EquityVanillaOption(d0.add_days(int(p["days"])), strike(p), OptionTypes.EUROPEAN_CALL)
        m = Heston(p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])
        return float(m.value_lewis(d0, opt, s, MKT["r"], MKT["q"]))


def _ql_process(p, spot, discretization=None):
    today = ql.Date(1, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    r = ql.YieldTermStructureHandle(ql.FlatForward(today, MKT["r"], dc))
    q = ql.YieldTermStructureHandle(ql.FlatForward(today, MKT["q"], dc))
    args = (r, q, ql.QuoteHandle(ql.SimpleQuote(spot)), p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])
    proc = ql.HestonProcess(*args) if discretization is None else ql.HestonProcess(*args, discretization)
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, strike(p)),
                           ql.EuropeanExercise(today + int(p["days"])))
    return proc, opt


def ql_analytic(p: dict, spot: float | None = None) -> float:
    proc, opt = _ql_process(p, MKT["S"] if spot is None else spot)
    opt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(proc)))
    return float(opt.NPV())


def ql_mc_qe(p: dict, paths: int = 50_000, steps: int = 50, seed: int = 1) -> tuple[float, float]:
    """Monte Carlo with Andersen's quadratic-exponential scheme: a genuinely different method."""
    proc, opt = _ql_process(p, MKT["S"], ql.HestonProcess.QuadraticExponentialMartingale)
    opt.setPricingEngine(ql.MCEuropeanHestonEngine(proc, "pseudorandom", timeSteps=steps,
                                                   antitheticVariate=True, requiredSamples=paths, seed=seed))
    return float(opt.NPV()), float(opt.errorEstimate())


def implied_vol(price: float, p: dict) -> float:
    """Black implied vol of a call; nan when the price is outside the invertible range."""
    F, K, T, D = forward(p), strike(p), p["days"] / 365, exp(-MKT["r"] * p["days"] / 365)
    lo, hi = D * max(F - K, 0), D * F
    if not (lo + 1e-10 < price < hi - 1e-10):
        return float("nan")
    def black(s):
        d1 = (np.log(F / K) + 0.5 * s * s * T) / (s * np.sqrt(T))
        return D * (F * norm.cdf(d1) - K * norm.cdf(d1 - s * np.sqrt(T))) - price
    try:
        return brentq(black, 1e-4, 5.0, xtol=1e-12)
    except ValueError:
        return float("nan")


def sobol_points(n: int, box: dict | None = None, seed: int = 7) -> list[dict]:
    box = box or DOMAIN
    m = max(1, (n - 1).bit_length())  # balanced Sobol needs 2**m points; keep the first n
    u = qmc.Sobol(d=len(PARAMS), scramble=True, seed=seed).random_base2(m)[:n]
    pts = []
    for row in u:
        p = {k: lo + x * (hi - lo) for x, (k, (lo, hi)) in zip(row, ((k, box[k]) for k in PARAMS))}
        p["days"] = max(7, int(round(p["days"])))
        pts.append(p)
    return pts
