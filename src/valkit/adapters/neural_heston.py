"""Adapter for the Neural Heston pricer (github.com/louisdelacour21/Neural-heston): an MLP that outputs forward-unit
call prices C(k, T), with the price floored at intrinsic value as the model's own documentation prescribes.
Needs torch and a local checkout of the pricer: set `path` (checkout) and `checkpoint` in the campaign config."""
import sys
from math import exp, log
from pathlib import Path

from .pricing import MKT, strike

# Rectangle inscribed in the network's claimed training box (x = k / sqrt(T) in [-0.8, 0.5], T in [0.05, 2] years):
# at the shortest maturity (30 days) that is k in [-0.229, 0.143]. Rounded inward.
DOMAIN = {"v0": (0.01, 0.10), "kappa": (0.5, 4.0), "theta": (0.01, 0.10), "sigma": (0.1, 0.8),
          "rho": (-0.9, -0.1), "k": (-0.22, 0.14), "days": (30, 730)}


class NeuralHeston:
    name = "neural-heston"
    domain = DOMAIN

    def __init__(self, path: str = "../Pricer neuronal", checkpoint: str = "models/dml.pt"):
        root = Path(path).resolve()
        sys.path.insert(0, str(root / "src"))
        import torch
        from neural_heston.model import load
        self._torch = torch
        self.net = load(root / checkpoint)

    def price(self, p: dict, spot: float | None = None) -> float:
        S = MKT["S"] if spot is None else spot
        T = p["days"] / 365
        F = S * exp((MKT["r"] - MKT["q"]) * T)
        k = log(strike(p) / F)  # strike stays fixed when the spot is bumped
        x = self._torch.tensor([[p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"], k, T]],
                               dtype=self._torch.float32)
        with self._torch.no_grad():
            c = float(self.net(x)[0])
        c = max(c, max(1 - exp(k), 0.0))  # documented floor at intrinsic value (forward units)
        return exp(-MKT["r"] * T) * F * c
