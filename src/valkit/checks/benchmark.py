"""BM-02 (VaR against an independent quantile) and RP-01 (reproducibility)."""
import numpy as np

from ..core import TestResult
from ..adapters.base import RiskModel


def bm_var(model: RiskModel, returns: np.ndarray, n_windows: int = 250) -> TestResult:
    """Model VaR vs numpy.quantile on the same windows. inverted_cdf is the historical-order-statistic
    convention; the linear-interpolation gap is reported as an observation, not a finding."""
    W, q = model.window, round(1 - model.conf, 12)  # 1 - 0.99 = 0.010000000000000009 would pick the 6th of 500, not the 5th
    ts = range(len(returns) - n_windows, len(returns))
    rel, rel_lin = [], []
    for t in ts:
        h = returns[t - W:t]
        v = model.forecast(h)[0]
        ref = -np.quantile(h, q, method="inverted_cdf")
        rel.append(abs(v - ref) / ref)
        rel_lin.append(abs(v + np.quantile(h, q)) / (-np.quantile(h, q)))
    worst = float(max(rel))
    status = "green" if worst < 1e-3 else "amber" if worst < 2e-2 else "red"
    return TestResult("BM-02", "VaR vs numpy.quantile", worst, status,
                      {"green_lt": 1e-3, "amber_lt": 2e-2},
                      evidence={"windows": n_windows, "convention": "inverted_cdf",
                                "observation_gap_vs_linear_interp": float(max(rel_lin))})


def bm_es(model: RiskModel, returns: np.ndarray, n_windows: int = 250) -> TestResult:
    """Model ES vs the mean of the k = round(W(1-conf)) worst returns of the same window (historical-ES convention).
    Deterministic and sharp where the Z2 backtest has little power (ES silently equal to VaR)."""
    W, q = model.window, round(1 - model.conf, 12)
    k = max(1, round(W * q))
    rel = []
    for t in range(len(returns) - n_windows, len(returns)):
        h = returns[t - W:t]
        ref = -np.sort(h)[:k].mean()
        rel.append(abs(model.forecast(h)[1] - ref) / ref)
    worst = float(max(rel))
    return TestResult("BM-03", "ES vs mean of k worst returns", worst,
                      "green" if worst < 1e-3 else "amber" if worst < 2e-2 else "red",
                      {"green_lt": 1e-3, "amber_lt": 2e-2}, evidence={"windows": n_windows, "k": k})


def rp_reproducibility(run) -> TestResult:
    """`run` returns the facts text; two executions must be byte-identical."""
    same = run() == run()
    return TestResult("RP-01", "Two runs give identical facts", float(same), "green" if same else "red",
                      {"identical": True})
