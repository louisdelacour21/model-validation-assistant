from typing import Protocol

import numpy as np


class RiskModel(Protocol):
    name: str
    conf: float
    window: int

    def forecast(self, history: np.ndarray) -> tuple[float, float]:
        """(VaR, ES) as positive losses, from the returns observed before the forecast day."""


def rolling_forecasts(model: RiskModel, returns: np.ndarray):
    """One-day-ahead (VaR, ES) for every day after the first window, using the past only."""
    W = model.window
    f = np.array([model.forecast(returns[t - W:t]) for t in range(W, len(returns))])
    return returns[W:], f[:, 0], f[:, 1]
