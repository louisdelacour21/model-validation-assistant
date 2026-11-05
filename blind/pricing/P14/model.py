import numpy as np
from scipy.integrate import quad

R, Q, S_REF = 0.02, 0.01, 100.0


class Model:
    name = "blind-P14"
    domain = {"v0": (0.01, 0.16), "kappa": (0.5, 5.0), "theta": (0.01, 0.16),
              "sigma": (0.1, 1.0), "rho": (-0.9, 0.3), "k": (-0.3, 0.3),
              "days": (30, 730)}

    def _cf(self, u, p, T, x):
        v0, kappa, theta, sigma, rho = p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"]
        iu = 1j * u
        b = kappa - rho * sigma * iu
        d = np.sqrt(b * b + sigma**2 * (u * u + iu))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        C = (kappa * theta / sigma**2) * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (b - d) / sigma**2 * (1 - e) / (1 - g * e)
        return np.exp(iu * (x + (R - Q) * T) + C + D * v0)

    def price(self, p, spot=None) -> float:
        S = S_REF if spot is None else float(spot)
        T = p["days"] / 365.0
        K = S_REF * np.exp((R - Q) * T) * np.exp(p["k"])
        x, lk = np.log(S), np.log(K)
        fwd = S * np.exp((R - Q) * T)

        def f1(u):
            return (np.exp(-1j * u * lk) * self._cf(u - 1j, p, T, x) / (1j * u * fwd)).real

        def f2(u):
            return (np.exp(-1j * u * lk) * self._cf(u, p, T, x) / (1j * u)).real

        P1 = 0.5 + quad(f1, 0, 200, limit=500)[0] / np.pi
        P2 = 0.5 + quad(f2, 0, 200, limit=500)[0] / np.pi
        return float(S * np.exp(-Q * T) * P1 - K * np.exp(-(R - Q) * T) * P2)
