import numpy as np
from riskhub_engine import market


class RiskhubHistoricalVaR:
    """riskhub-engine historical VaR/ES. Self-review: the author is also the developer."""

    def __init__(self, conf: float = 0.99, window: int = 500):
        self.name, self.conf, self.window = "riskhub-historical", conf, window

    def forecast(self, history: np.ndarray) -> tuple[float, float]:
        r = market.historical(history.tolist(), self.conf)
        return r["var"], r["es"]
