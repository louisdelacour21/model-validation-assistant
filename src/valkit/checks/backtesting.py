"""BT-01..04: VaR/ES backtests, each with the power it has at the real sample size."""
import numpy as np
from scipy.special import xlogy
from scipy.stats import binom, chi2, norm

from ..core import TestResult, p_status

# Basel add-on to the multiplier for 5..9 exceptions out of 250 (99 % VaR); 10+ is red, add-on 1
BASEL_ADDON = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def kupiec_lr(N: int, x: int, p: float) -> tuple[float, float]:
    ph = x / N
    lr = -2 * (xlogy(N - x, 1 - p) + xlogy(x, p) - xlogy(N - x, 1 - ph) - xlogy(x, ph))
    lr = max(float(lr), 0.0)
    return lr, float(chi2.sf(lr, 1))


def christoffersen_lr(hits) -> tuple[float, float]:
    """Independence test on the first-order transition counts."""
    h = np.asarray(hits, int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    lr = max(float(-2 * (ll0 - ll1)), 0.0)
    return lr, float(chi2.sf(lr, 1))


def traffic_light(N: int, x: int, p: float) -> str:
    c = binom.cdf(x, N, p)
    return "green" if c < 0.95 else "amber" if c < 0.9999 else "red"


def rejection_rate(pvalue, gen, n_sims: int, alpha: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    return float(np.mean([pvalue(gen(rng)) < alpha for _ in range(n_sims)]))


def _iid(N, q):
    return lambda rng: (rng.random(N) < q).astype(int)


def _clustered(N, q, stay=0.3):
    """Markov exceptions with marginal rate q and P(hit | hit before) = stay."""
    enter = q * (1 - stay) / (1 - q)
    def gen(rng):
        h, prev = np.zeros(N, int), 0
        u = rng.random(N)
        for i in range(N):
            prev = int(u[i] < (stay if prev else enter))
            h[i] = prev
        return h
    return gen


def bt_kupiec(hits, p: float, seed: int = 20261105, n_sims: int = 2000) -> TestResult:
    h = np.asarray(hits, int)
    N, x = len(h), int(h.sum())
    lr, pv = kupiec_lr(N, x, p)
    power = rejection_rate(lambda s: kupiec_lr(N, int(s.sum()), p)[1], _iid(N, 2 * p), n_sims, 0.05, seed)
    size = rejection_rate(lambda s: kupiec_lr(N, int(s.sum()), p)[1], _iid(N, p), n_sims, 0.05, seed + 1)
    return TestResult("BT-01", "Kupiec POF", pv, p_status(pv), {"amber_p": 0.05, "red_p": 0.01},
                      power, seed, {"N": N, "exceptions": x, "expected": N * p, "LR": lr,
                                    "power_vs_2p": power, "size_at_5pct": size})


def bt_christoffersen(hits, p: float, seed: int = 20261105, n_sims: int = 2000) -> TestResult:
    h = np.asarray(hits, int)
    N = len(h)
    lr_ind, p_ind = christoffersen_lr(h)
    _, p_pof = kupiec_lr(N, int(h.sum()), p)
    lr_cc = kupiec_lr(N, int(h.sum()), p)[0] + lr_ind
    p_cc = float(chi2.sf(lr_cc, 2))
    power = rejection_rate(lambda s: christoffersen_lr(s)[1], _clustered(N, p), n_sims, 0.05, seed)
    return TestResult("BT-02", "Christoffersen independence / conditional coverage", p_ind, p_status(p_ind),
                      {"amber_p": 0.05, "red_p": 0.01}, power, seed,
                      {"N": N, "LR_ind": lr_ind, "LR_cc": lr_cc, "p_cc": p_cc, "power_vs_clusters": power})


def bt_traffic_light(hits, p: float) -> TestResult:
    h = np.asarray(hits, int)
    N, x = len(h), int(h.sum())
    zone = traffic_light(N, x, p)
    addon = (1.0 if zone == "red" else BASEL_ADDON.get(x, 0.0)) if (N, p) == (250, 0.01) else None
    return TestResult("BT-03", "Basel traffic light", float(binom.cdf(x, N, p)), zone,
                      {"amber_cdf": 0.95, "red_cdf": 0.9999}, None, None,
                      {"N": N, "exceptions": x, "multiplier_addon": addon})


def _z2_null(N: int, p: float, n_sims: int, seed: int) -> np.ndarray:
    # shortcut: Gaussian tail assumed under H0 with 5000 draws, upgrade to the model's own tail law
    rng = np.random.default_rng(seed)
    z = norm.ppf(1 - p)
    es_over_sigma = norm.pdf(z) / p
    k = rng.binomial(N, p, n_sims)
    return np.array([(norm.ppf(1 - rng.random(ki) * p) / es_over_sigma).sum() / (N * p) - 1 for ki in k])


def bt_es(returns, var, es, p: float, seed: int = 20261105, n_sims: int = 5000) -> TestResult:
    """Acerbi-Szekely Z2: sum(X_t I_t / ES_t) / (N p) - 1 ; > 0 means ES is too low."""
    r, v, e = (np.asarray(a, float) for a in (returns, var, es))
    N = len(r)
    hit = r < -v
    z2 = float((-r[hit] / e[hit]).sum() / (N * p) - 1)
    pv = float(np.mean(_z2_null(N, p, n_sims, seed) >= z2))
    return TestResult("BT-04", "Expected shortfall Z2", pv, p_status(pv), {"amber_p": 0.05, "red_p": 0.01},
                      None, seed, {"N": N, "Z2": z2, "exceptions": int(hit.sum()),
                                   "null": "Gaussian tail, %d draws" % n_sims})
