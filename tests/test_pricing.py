import numpy as np
import pytest

from valkit.adapters.pricing import (
    DOMAIN,
    FinancePyHeston,
    implied_vol,
    ql_analytic,
    ql_mc_qe,
    sobol_points,
)
from valkit.checks import pricing as pc


def test_implied_vol_round_trip():
    p = sobol_points(1)[0]
    # price a call with Black at 25 % vol, invert it
    from scipy.stats import norm

    from valkit.adapters.pricing import MKT, forward, strike
    F, K, T, s = forward(p), strike(p), p["days"] / 365, 0.25
    d1 = (np.log(F / K) + 0.5 * s * s * T) / (s * np.sqrt(T))
    px = np.exp(-MKT["r"] * T) * (F * norm.cdf(d1) - K * norm.cdf(d1 - s * np.sqrt(T)))
    assert implied_vol(px, p) == pytest.approx(s, abs=1e-8)


def test_sobol_points_stay_in_domain():
    for p in sobol_points(20):
        assert all(DOMAIN[k][0] - 1e-9 <= p[k] <= DOMAIN[k][1] + 1e-9 for k in DOMAIN)


def test_reference_mc_agrees_with_analytic():
    p = sobol_points(1)[0]
    mc, se = ql_mc_qe(p, 100_000)
    assert abs(mc - ql_analytic(p)) < 4 * se


def test_financepy_passes_and_planted_defects_fail():
    class FlippedRho(FinancePyHeston):          # planted defect: sign of rho
        def price(self, p, spot=None):
            return super().price(p | {"rho": -p["rho"]}, spot)

    class BumpedVol(FinancePyHeston):           # planted defect: v0 +30 %
        def price(self, p, spot=None):
            return super().price(p | {"v0": p["v0"] * 1.3}, spot)

    assert pc.bm_pricing(FinancePyHeston(), n=16, n_mc=2).status == "green"
    assert pc.bm_pricing(FlippedRho(), n=16, n_mc=0).status == "red"
    assert pc.bm_pricing(BumpedVol(), n=16, n_mc=0).status == "red"


def test_stress_flags_crash_as_a_result_not_an_exception():
    class Crashes(FinancePyHeston):
        def price(self, p, spot=None):
            raise ZeroDivisionError("boom")
    r = pc.st_stress(Crashes())
    assert all("error" in c for c in r.evidence["cases"].values())


def test_drift_detects_shifted_inputs():
    ref = sobol_points(300, seed=2)
    ok = pc.dr_drift({"sigma": [p["sigma"] for p in ref]})
    shifted = pc.dr_drift({"sigma": [0.75 + 0.05 * i / 300 for i in range(300)]})
    assert ok.status == "green" and shifted.status == "red"
