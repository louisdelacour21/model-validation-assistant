import math

import numpy as np
from riskhub_engine import market
from scipy.stats import binom, chi2

from valkit.adapters.riskhub_var import RiskhubHistoricalVaR
from valkit.campaign import var_checks
from valkit.checks import backtesting as bt
from valkit.checks.benchmark import rp_reproducibility
from valkit.core import facts_json


def test_kupiec_zero_only_when_Np_integer():
    assert bt.kupiec_lr(500, 5, 0.01)[0] == 0          # Np = 5
    N, p = 250, 0.01                                    # Np = 2.5
    lo, hi = math.floor(N * p), math.ceil(N * p)
    lrs = {x: bt.kupiec_lr(N, x, p)[0] for x in (lo - 1, lo, hi, hi + 1)}
    assert lrs[lo] > 0 and lrs[hi] > 0
    assert min(lrs, key=lrs.get) in (lo, hi)


def test_kupiec_pvalue_is_chi2_sf_and_matches_riskhub():
    lr, p = bt.kupiec_lr(250, 7, 0.01)
    assert p == chi2.sf(lr, 1)
    assert np.isclose(lr, market.kupiec(250, 7, 0.01)["lr"])


def test_christoffersen_matches_riskhub():
    h = (np.random.default_rng(3).random(250) < 0.03).astype(int)
    assert np.isclose(bt.christoffersen_lr(h)[0], market.christoffersen(h.tolist(), 0.03)["ind"])


def test_traffic_light_bounds_at_250_99():
    assert [bt.traffic_light(250, x, 0.01) for x in (4, 5, 9, 10)] == ["green", "amber", "amber", "red"]
    assert binom.cdf(4, 250, 0.01) < 0.95 <= binom.cdf(5, 250, 0.01)
    assert binom.cdf(9, 250, 0.01) < 0.9999 <= binom.cdf(10, 250, 0.01)
    assert bt.bt_traffic_light([1] * 5 + [0] * 245, 0.01).evidence["multiplier_addon"] == 0.40


def test_kupiec_size_and_power():
    # measured, not assumed: the chi2 approximation over-rejects at N = 250, p = 1 % (about 9 %, not 5 %)
    N, p = 250, 0.01
    r = bt.bt_kupiec((np.random.default_rng(0).random(N) < p).astype(int), p)
    assert 0.07 < r.evidence["size_at_5pct"] < 0.12
    assert 0 < r.power <= 1                              # published next to the p-value


def test_es_z2_flags_underestimated_es():
    rng = np.random.default_rng(5)
    r = rng.standard_t(3, 250) * 0.01
    ok = bt.bt_es(r, np.full(250, 0.05), np.full(250, 0.10), 0.01)    # generous ES
    bad = bt.bt_es(r, np.full(250, 0.02), np.full(250, 0.021), 0.01)   # ES barely above VaR
    assert bad.evidence["Z2"] > ok.evidence["Z2"] and bad.status != "green"


def test_campaign_is_reproducible_and_flags_planted_bias():
    r = np.random.default_rng(7).standard_t(4, 1200) * 0.01
    model = RiskhubHistoricalVaR(0.99, 500)
    assert rp_reproducibility(lambda: facts_json(var_checks(model, r))).status == "green"

    class Biased(RiskhubHistoricalVaR):       # planted defect: VaR halved
        def forecast(self, h):
            v, e = super().forecast(h)
            return v / 2, e / 2
    res = {x.id: x for x in var_checks(Biased(0.99, 500), r)}
    assert res["BT-01"].status == "red" and res["BT-03"].status == "red" and res["BM-02"].status == "red"
