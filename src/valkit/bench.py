"""Planted-defect bench: detection rate per defect and false-rejection rate of healthy models, with Wilson CI.

Flagged = opinion is not 'fit' (see findings.verdict); rejected = opinion 'not fit for use'. Defects are public
(taxonomy below); the third-party blind defects of fiche 06 are NOT part of this file."""
import json
from datetime import date
from pathlib import Path

import numpy as np

from .adapters.pricing import FinancePyHeston
from .adapters.riskhub_var import RiskhubHistoricalVaR
from .campaign import var_checks
from .checks import pricing as pc
from .findings import verdict

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


# ---- VaR: wrappers around the historical model; healthy data are iid Student-t(4), 1 % daily vol
class Scaled(RiskhubHistoricalVaR):
    def __init__(self, k): super().__init__(); self.k = k
    def forecast(self, h): v, e = super().forecast(h); return v * self.k, e * self.k


class WrongConf(RiskhubHistoricalVaR):         # claims 99 %, computes 95 %
    def forecast(self, h): return RiskhubHistoricalVaR(0.95, self.window).forecast(h)


class Static(RiskhubHistoricalVaR):            # VaR frozen at the first window
    _v = None
    def forecast(self, h):
        self._v = self._v or super().forecast(h)
        return self._v


class NoTail(RiskhubHistoricalVaR):            # ES reported equal to VaR
    def forecast(self, h): v, _ = super().forecast(h); return v, v


class Short(RiskhubHistoricalVaR):             # 50-day window, still declares 99 %
    def __init__(self): super().__init__(0.99, 50)


VAR_DEFECTS = {
    "VD-01 understated x0.70": lambda: Scaled(0.70), "VD-02 understated x0.85": lambda: Scaled(0.85),
    "VD-03 confidence 95 % declared 99 %": WrongConf, "VD-04 frozen VaR": Static,
    "VD-05 ES equals VaR": NoTail, "VD-06 50-day window": Short,
}


def _returns(seed: int, n: int = 1250) -> np.ndarray:
    r = np.random.default_rng(seed).standard_t(4, n)
    return r / np.sqrt(2) * 0.01  # t4 has variance 2


def _var_status(make, seed):
    m = make()
    return verdict(var_checks(m, _returns(seed)))


# ---- Pricing: wrappers around FinancePy
def _perturbed(**f):
    class P(FinancePyHeston):
        def price(self, p, spot=None):
            return super().price({**p, **{k: g(p[k]) for k, g in f.items()}}, spot)
    return P


PRICING_DEFECTS = {
    "PD-01 rho sign flipped": _perturbed(rho=lambda x: -x), "PD-02 v0 +30 %": _perturbed(v0=lambda x: x * 1.3),
    "PD-03 v0 +5 %": _perturbed(v0=lambda x: x * 1.05), "PD-04 kappa x1.5": _perturbed(kappa=lambda x: x * 1.5),
    "PD-05 vol of vol +10 %": _perturbed(sigma=lambda x: x * 1.1), "PD-06 ACT/360 maturity": _perturbed(days=lambda x: round(x * 365 / 360)),
    "PD-07 strike shifted by 1 %": _perturbed(k=lambda x: x + 0.01),
}


def _pricing_status(make, seed):
    m = make()
    return verdict([pc.bm_pricing(m, n=32, n_mc=0, seed=seed), pc.se_delta(m, n=8, seed=seed)])


def _row(statuses):
    n = len(statuses)
    fl, rj = sum(s >= 1 for s in statuses), sum(s >= 2 for s in statuses)
    return {"n": n, "flagged": fl, "flagged_ci95": wilson(fl, n), "rejected": rj, "rejected_ci95": wilson(rj, n)}


def run_bench(out_root: Path, n_var: int = 40, n_pricing: int = 12) -> Path:
    out = {"variant": {"var_healthy": "riskhub-historical on iid t4 data", "pricing_healthy": "FinancePy on fresh Sobol seeds"},
           "var": {"healthy": _row([_var_status(RiskhubHistoricalVaR, s) for s in range(n_var)])},
           "pricing": {"healthy": _row([_pricing_status(FinancePyHeston, 100 + s) for s in range(n_pricing)])}}
    for k, mk in VAR_DEFECTS.items():
        out["var"][k] = _row([_var_status(mk, s) for s in range(n_var)])
    for k, mk in PRICING_DEFECTS.items():
        out["pricing"][k] = _row([_pricing_status(mk, 100 + s) for s in range(n_pricing)])
    d = out_root / "bench" / date.today().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    (d / "bench.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    return d / "bench.json"
